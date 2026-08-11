#!/usr/bin/env python3
"""Cross-platform inference benchmark: StandardTransformer vs FFFTransformer (hard).

Execution
---------
    # Auto GPU (if available) + CPU side-by-side
    python benchmark.py

    # Force devices
    python benchmark.py --device auto          # GPU (or CPU) then always also CPU
    python benchmark.py --device cuda --skip-cpu
    python benchmark.py --device cpu

    # Faster smoke / custom size
    python benchmark.py --n-tokens 50 --warmup 5 --n-embd 256 --n-layer 4

    # Match a trained checkpoint's config
    python benchmark.py --checkpoint fff_checkpoint.pt

Notes
-----
* Compares Dense MLP vs FFF hard routing.
* By default benchmarks the auto-detected accelerator **and** CPU for a
  side-by-side report (skip CPU with ``--skip-cpu``).
* Peak RAM via ``psutil`` when available, else ``tracemalloc`` fallback.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from device_utils import (
    apply_hardware_optimizations,
    device_label,
    print_device_info,
    resolve_device,
)
from models.transformer import FFFConfig, FFFTransformer, StandardTransformer

# ---------------------------------------------------------------------------
# Optional psutil (preferred for RSS); graceful fallback to tracemalloc
# ---------------------------------------------------------------------------

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


def _rss_mb() -> float:
    """Current process resident set size in MB."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    # tracemalloc tracks Python allocations only (underestimates native torch).
    current, _peak = tracemalloc.get_traced_memory()
    return current / (1024.0 * 1024.0)


def _peak_traced_mb() -> float:
    if not tracemalloc.is_tracing():
        return 0.0
    _current, peak = tracemalloc.get_traced_memory()
    return peak / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> int:
    """Total trainable parameters (unique tensors; respects weight tying)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def dense_mlp_params(d_model: int, n_layer: int, bias: bool) -> int:
    """Parameters in all ``Linear(D,4D)+Linear(4D,D)`` MLPs."""
    # Weights: D*(4D) + (4D)*D = 8 D²; biases optional: 4D + D = 5D
    per = 8 * d_model * d_model
    if bias:
        per += 5 * d_model
    return per * n_layer


def fff_stored_and_active(model: FFFTransformer) -> tuple[int, int]:
    """Return ``(stored_fff_params, hard_active_fff_params_per_token)``."""
    stored = 0
    active = 0
    for layer in model.fff_layers():
        stats = layer.active_params_per_token()
        stored += stats["stored_total"]
        active += stats["hard_active_per_token"]
    return stored, active


@dataclass
class ModelParamReport:
    name: str
    total_params: int
    active_params_per_token: int
    active_pct: float
    ffn_stored: int
    ffn_active: int


def param_report_standard(model: StandardTransformer) -> ModelParamReport:
    total = count_params(model)
    ffn = dense_mlp_params(
        model.config.n_embd, model.config.n_layer, model.config.bias
    )
    # Dense forward evaluates every parameter (embeddings for present ids, attn, MLP, head).
    return ModelParamReport(
        name="Standard Dense",
        total_params=total,
        active_params_per_token=total,
        active_pct=100.0,
        ffn_stored=ffn,
        ffn_active=ffn,
    )


def param_report_fff(model: FFFTransformer) -> ModelParamReport:
    total = count_params(model)
    ffn_stored, ffn_active = fff_stored_and_active(model)
    # Inactive FFF mass = unused routers + unused leaves (skipped by hard path).
    inactive = ffn_stored - ffn_active
    active = total - inactive
    return ModelParamReport(
        name="FFF Hard Routing",
        total_params=total,
        active_params_per_token=active,
        active_pct=100.0 * active / max(total, 1),
        ffn_stored=ffn_stored,
        ffn_active=ffn_active,
    )


# ---------------------------------------------------------------------------
# Generation timing
# ---------------------------------------------------------------------------


def make_fff_step(model: FFFTransformer) -> Callable[[Tensor], Tensor]:
    def step(idx: Tensor) -> Tensor:
        logits, _ = model(idx, mode="hard")
        return logits

    return step


def make_dense_step(model: StandardTransformer) -> Callable[[Tensor], Tensor]:
    def step(idx: Tensor) -> Tensor:
        return model(idx)

    return step


@dataclass
class SpeedResult:
    threads: int
    n_tokens: int
    elapsed_s: float
    ms_per_token: float
    tokens_per_s: float


@dataclass
class MemoryResult:
    baseline_mb: float
    after_load_mb: float
    peak_generate_mb: float
    backend: str


@dataclass
class ModelBenchResult:
    param: ModelParamReport
    memory: MemoryResult
    single: SpeedResult
    multi: SpeedResult


def _sync_device(device: torch.device) -> None:
    """Flush pending device work before timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_timed(
    forward_step: Callable[[Tensor], Tensor],
    input_ids: Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
) -> tuple[Tensor, float]:
    """Autoregressive loop with a custom logits step; returns ``(ids, elapsed_s)``.

    ``forward_step(idx_cond) -> logits (B, T, V)``.
    """
    device = input_ids.device
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        idx_cond = (
            input_ids
            if input_ids.size(1) <= block_size
            else input_ids[:, -block_size:]
        )
        logits = forward_step(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        # Greedy argmax — deterministic, avoids multinomial RNG overhead variance.
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_id), dim=1)
    _sync_device(device)
    elapsed = time.perf_counter() - t0
    return input_ids, elapsed


def benchmark_speed(
    forward_step: Callable[[Tensor], Tensor],
    prompt: Tensor,
    block_size: int,
    n_tokens: int,
    warmup: int,
    num_threads: int,
) -> SpeedResult:
    """Warmup then time ``n_tokens`` greedy generations under ``num_threads``."""
    torch.set_num_threads(num_threads)
    # Some builds also expose interop threads.
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(num_threads, 4)))
        except RuntimeError:
            pass  # may already be set

    model_device_prompt = prompt.clone()
    if warmup > 0:
        generate_timed(
            forward_step, model_device_prompt.clone(), warmup, block_size
        )
        _sync_device(prompt.device)

    _ids, elapsed = generate_timed(
        forward_step, model_device_prompt.clone(), n_tokens, block_size
    )
    _sync_device(prompt.device)
    ms = (elapsed * 1000.0) / max(n_tokens, 1)
    tps = n_tokens / max(elapsed, 1e-12)
    return SpeedResult(
        threads=num_threads,
        n_tokens=n_tokens,
        elapsed_s=elapsed,
        ms_per_token=ms,
        tokens_per_s=tps,
    )


def measure_memory_and_load(
    factory: Callable[[], nn.Module],
    forward_step_builder: Callable[[nn.Module], Callable[[Tensor], Tensor]],
    prompt: Tensor,
    block_size: int,
    gen_tokens: int = 32,
) -> tuple[nn.Module, Callable[[Tensor], Tensor], MemoryResult]:
    """Track RSS around load + a short generation; return model + step + memory."""
    gc.collect()
    if not _HAS_PSUTIL and not tracemalloc.is_tracing():
        tracemalloc.start()

    baseline = _rss_mb()
    peak = baseline

    model = factory()
    model.eval()
    after_load = _rss_mb()
    peak = max(peak, after_load, _peak_traced_mb() + baseline if not _HAS_PSUTIL else after_load)

    step = forward_step_builder(model)
    # Generation peak
    generate_timed(step, prompt.clone(), gen_tokens, block_size)
    during_gen = _rss_mb()
    if _HAS_PSUTIL:
        peak_gen = max(peak, during_gen)
        backend = "psutil.rss"
    else:
        peak_gen = baseline + _peak_traced_mb()
        backend = "tracemalloc (Python allocs only; install psutil for RSS)"

    return (
        model,
        step,
        MemoryResult(
            baseline_mb=baseline,
            after_load_mb=after_load,
            peak_generate_mb=peak_gen,
            backend=backend,
        ),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def _fmt_mb(x: float) -> str:
    return f"{x:.1f} MB"


def _fmt_tps(x: float) -> str:
    return f"{x:.2f} tok/s"


def _fmt_ms(x: float) -> str:
    return f"{x:.3f} ms"


def _cell(text: str, width: int) -> str:
    return f" {text:<{width - 1}}"


def print_comparison_table(
    dense: ModelBenchResult,
    fff: ModelBenchResult,
) -> None:
    """ASCII summary table matching the requested layout."""
    speedup_single = fff.single.tokens_per_s / max(dense.single.tokens_per_s, 1e-12)
    speedup_multi = fff.multi.tokens_per_s / max(dense.multi.tokens_per_s, 1e-12)
    # Primary latency / speedup rows use multi-thread (typical laptop default),
    # with single-thread called out explicitly above.
    speedup = speedup_multi

    col0, col1, col2 = 32, 20, 20
    total_w = col0 + col1 + col2 + 4
    sep = "+" + "-" * col0 + "+" + "-" * col1 + "+" + "-" * col2 + "+"
    header = (
        "|"
        + _cell("Metric", col0)
        + "|"
        + _cell("Standard Dense", col1)
        + "|"
        + _cell("FFF Hard Routing", col2)
        + "|"
    )

    rows = [
        (
            "Total Parameters",
            _fmt_int(dense.param.total_params),
            _fmt_int(fff.param.total_params),
        ),
        (
            "Active Params / Token",
            "100%",
            _fmt_pct(fff.param.active_pct),
        ),
        (
            "Active Params (abs)",
            _fmt_int(dense.param.active_params_per_token),
            _fmt_int(fff.param.active_params_per_token),
        ),
        (
            "FFN Active / Stored",
            f"{_fmt_int(dense.param.ffn_active)} / {_fmt_int(dense.param.ffn_stored)}",
            f"{_fmt_int(fff.param.ffn_active)} / {_fmt_int(fff.param.ffn_stored)}",
        ),
        (
            "Peak RAM Usage (MB)",
            _fmt_mb(dense.memory.peak_generate_mb),
            _fmt_mb(fff.memory.peak_generate_mb),
        ),
        (
            "Single-Thread Speed (tok/s)",
            _fmt_tps(dense.single.tokens_per_s),
            _fmt_tps(fff.single.tokens_per_s),
        ),
        (
            "Multi-Thread Speed (tok/s)",
            _fmt_tps(dense.multi.tokens_per_s),
            _fmt_tps(fff.multi.tokens_per_s),
        ),
        (
            "Latency per Token (ms)",
            _fmt_ms(dense.multi.ms_per_token),
            _fmt_ms(fff.multi.ms_per_token),
        ),
        (
            "Latency (1-thread, ms)",
            _fmt_ms(dense.single.ms_per_token),
            _fmt_ms(fff.single.ms_per_token),
        ),
        (
            "Throughput Speedup Ratio",
            "1.0x (Baseline)",
            f"{speedup:.2f}x",
        ),
        (
            "Speedup (1-thread)",
            "1.0x",
            f"{speedup_single:.2f}x",
        ),
    ]

    print()
    print(sep)
    print(header)
    print(sep)
    for metric, a, b in rows:
        print(
            "|"
            + _cell(metric, col0)
            + "|"
            + _cell(a, col1)
            + "|"
            + _cell(b, col2)
            + "|"
        )
    print(sep)
    print(
        f"Memory backend: dense={dense.memory.backend} | fff={fff.memory.backend}"
    )
    print(
        f"RAM baseline→load→gen | Dense "
        f"{dense.memory.baseline_mb:.1f}→{dense.memory.after_load_mb:.1f}→"
        f"{dense.memory.peak_generate_mb:.1f} MB | FFF "
        f"{fff.memory.baseline_mb:.1f}→{fff.memory.after_load_mb:.1f}→"
        f"{fff.memory.peak_generate_mb:.1f} MB"
    )
    print(
        f"FFF FFN sparsity: hard-active "
        f"{100.0 * fff.param.ffn_active / max(fff.param.ffn_stored, 1):.2f}% "
        f"of stored FFF params "
        f"({_fmt_int(fff.param.ffn_active)} / {_fmt_int(fff.param.ffn_stored)})"
    )
    _ = total_w  # layout width documented for maintainers


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


def config_from_checkpoint(path: Path) -> FFFConfig:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return FFFConfig(**ckpt["model_config"])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark StandardTransformer vs FFFTransformer (CPU, hard routing)"
    )
    p.add_argument("--n-tokens", type=int, default=200, help="Timed generation length")
    p.add_argument("--warmup", type=int, default=10, help="Warmup tokens before timing")
    p.add_argument("--prompt-len", type=int, default=32, help="Prompt length (batch=1)")
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--fff-depth", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=65)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional fff_checkpoint.pt to copy architecture (+ load FFF weights)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Primary device: auto|cuda|mps|cpu (default auto)",
    )
    p.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Do not also run a CPU baseline when primary device is a GPU",
    )
    return p


def run_pair_on_device(
    device: torch.device,
    config: FFFConfig,
    ckpt_path: Path | None,
    prompt_cpu: Tensor,
    n_tokens: int,
    warmup: int,
    cpu_count: int,
) -> tuple[ModelBenchResult, ModelBenchResult]:
    """Benchmark Dense vs FFF hard routing on a single device."""
    apply_hardware_optimizations(device)
    print("\n" + "-" * 72)
    print(f"Device run: {device} — {device_label(device)}")
    print("-" * 72)

    prompt = prompt_cpu.to(device)

    print("\n[1/2] Benchmarking StandardTransformer (dense MLP)...")

    def dense_factory() -> StandardTransformer:
        m = StandardTransformer(config).to(device)
        m.eval()
        return m

    dense_model, dense_step, dense_mem = measure_memory_and_load(
        dense_factory,
        make_dense_step,
        prompt,
        config.block_size,
        gen_tokens=min(32, n_tokens),
    )
    dense_param = param_report_standard(dense_model)

    dense_single = benchmark_speed(
        dense_step, prompt, config.block_size, n_tokens, warmup, num_threads=1
    )
    dense_multi = benchmark_speed(
        dense_step,
        prompt,
        config.block_size,
        n_tokens,
        warmup,
        num_threads=cpu_count,
    )

    dense_result = ModelBenchResult(
        param=dense_param, memory=dense_mem, single=dense_single, multi=dense_multi
    )
    print(
        f"  params={dense_param.total_params:,} | "
        f"speed={dense_multi.tokens_per_s:.2f} tok/s | "
        f"latency={dense_multi.ms_per_token:.3f} ms | "
        f"peak_RAM={dense_mem.peak_generate_mb:.1f} MB"
    )

    del dense_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[2/2] Benchmarking FFFTransformer (mode=hard)...")

    def fff_factory() -> FFFTransformer:
        m = FFFTransformer(config).to(device)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    fff_model, fff_step, fff_mem = measure_memory_and_load(
        fff_factory,
        make_fff_step,
        prompt,
        config.block_size,
        gen_tokens=min(32, n_tokens),
    )
    fff_param = param_report_fff(fff_model)
    fff_single = benchmark_speed(
        fff_step, prompt, config.block_size, n_tokens, warmup, num_threads=1
    )
    fff_multi = benchmark_speed(
        fff_step,
        prompt,
        config.block_size,
        n_tokens,
        warmup,
        num_threads=cpu_count,
    )
    fff_result = ModelBenchResult(
        param=fff_param, memory=fff_mem, single=fff_single, multi=fff_multi
    )
    print(
        f"  params={fff_param.total_params:,} | "
        f"active/token={fff_param.active_params_per_token:,} "
        f"({fff_param.active_pct:.1f}%) | "
        f"speed={fff_multi.tokens_per_s:.2f} tok/s | "
        f"latency={fff_multi.ms_per_token:.3f} ms | "
        f"peak_RAM={fff_mem.peak_generate_mb:.1f} MB"
    )

    print(f"\n=== Dense vs FFF on {device.type.upper()} ===")
    print_comparison_table(dense_result, fff_result)

    del fff_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dense_result, fff_result


def print_gpu_vs_cpu_summary(
    gpu_device: torch.device,
    gpu_fff: ModelBenchResult,
    cpu_fff: ModelBenchResult,
) -> None:
    """Side-by-side FFF hard-routing throughput: accelerator vs CPU."""
    gpu_tps = gpu_fff.multi.tokens_per_s
    cpu_tps = cpu_fff.multi.tokens_per_s
    speedup = gpu_tps / max(cpu_tps, 1e-12)
    print("\n" + "=" * 72)
    print("FFF Hard Routing — Accelerator vs CPU")
    print("=" * 72)
    print(f"  {device_label(gpu_device):<40} {gpu_tps:8.2f} tok/s")
    print(f"  {'CPU Fallback':<40} {cpu_tps:8.2f} tok/s")
    print(f"  GPU/Accelerator speedup vs CPU:        {speedup:.2f}x")
    print("=" * 72)


def main() -> None:
    args = build_argparser().parse_args()
    torch.set_grad_enabled(False)

    primary = resolve_device(args.device)
    cpu_count = os.cpu_count() or 1

    print("=" * 72)
    print("FFF vs Dense MLP — Cross-Platform Inference Benchmark")
    print("=" * 72)
    print_device_info(primary)
    print(f"logical_cpus={cpu_count} | psutil={'yes' if _HAS_PSUTIL else 'NO (tracemalloc fallback)'}")
    if not _HAS_PSUTIL:
        print(
            "WARNING: install psutil for accurate RSS peak memory "
            "(`pip install psutil`)."
        )
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        config = config_from_checkpoint(ckpt_path)
        print(f"Architecture from checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
        config = FFFConfig(
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            dropout=args.dropout,
            fff_depth=args.fff_depth,
            init_temp=1.0,
            tie_weights=True,
            bias=False,
        )

    print(
        f"config: V={config.vocab_size} D={config.n_embd} L={config.n_layer} "
        f"H={config.n_head} T={config.block_size} fff_depth={config.fff_depth}"
    )
    print(
        f"tokens: warmup={args.warmup} timed={args.n_tokens} "
        f"prompt_len={args.prompt_len} batch=1"
    )

    torch.manual_seed(args.seed)
    prompt_cpu = torch.randint(
        0, config.vocab_size, (1, args.prompt_len), dtype=torch.long
    )

    # Primary device (auto GPU or forced)
    _dense_p, fff_primary = run_pair_on_device(
        primary,
        config,
        ckpt_path,
        prompt_cpu,
        args.n_tokens,
        args.warmup,
        cpu_count,
    )

    # Side-by-side CPU when primary is an accelerator
    if primary.type != "cpu" and not args.skip_cpu:
        _dense_c, fff_cpu = run_pair_on_device(
            torch.device("cpu"),
            config,
            ckpt_path,
            prompt_cpu,
            args.n_tokens,
            args.warmup,
            cpu_count,
        )
        print_gpu_vs_cpu_summary(primary, fff_primary, fff_cpu)

    torch.set_num_threads(cpu_count)


if __name__ == "__main__":
    # Avoid OpenMP oversubscription surprises before we set threads explicitly.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
