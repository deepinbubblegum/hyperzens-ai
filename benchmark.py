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

    # CUDA batch-size scaling (Dense vs FFF Triton crossover)
    python benchmark.py --device cuda --skip-cpu --batch-sweep

    # FP16 Tensor Core sweep (Ampere)
    python benchmark.py --device cuda --skip-cpu --batch-sweep --precision fp16

Notes
-----
* Compares Dense MLP vs FFF PyTorch hard vs FFF C++ hard (CPU) / Triton (CUDA).
* On CUDA, optionally sweeps batch sizes to find where FFF Triton beats Dense.
* ``--precision fp16|bf16|fp32|both`` casts models for Tensor Core paths (default: both fp32+fp16 on CUDA sweeps).
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
from models.fff_layer import is_fff_cpp_available
from models.fff_hard_triton import is_triton_available, warmup_fff_model_triton
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


def make_fff_cpp_step(model: FFFTransformer) -> Callable[[Tensor], Tensor]:
    """Hard routing through the C++ CPU extension (``mode='hard_cpp'``)."""

    def step(idx: Tensor) -> Tensor:
        logits, _ = model(idx, mode="hard_cpp")
        return logits

    return step


def make_fff_triton_step(model: FFFTransformer) -> Callable[[Tensor], Tensor]:
    """Hard routing through the fused Triton CUDA kernel (``mode='triton'``)."""

    def step(idx: Tensor) -> Tensor:
        logits, _ = model(idx, mode="triton")
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
    fff_alt: ModelBenchResult | None = None,
    *,
    alt_name: str = "FFF Alt Hard",
) -> None:
    """ASCII summary: Dense vs FFF PyTorch Hard vs optional third backend."""
    names = ["Standard Dense", "FFF PyTorch Hard"]
    results: list[ModelBenchResult] = [dense, fff]
    if fff_alt is not None:
        names.append(alt_name)
        results.append(fff_alt)

    col0 = 34
    col_w = 20
    n_cols = len(names)
    sep = "+" + "-" * col0 + ("+" + "-" * col_w) * n_cols + "+"
    header = "|" + _cell("Metric", col0)
    for name in names:
        header += "|" + _cell(name, col_w)
    header += "|"

    def row(metric: str, values: list[str]) -> str:
        line = "|" + _cell(metric, col0)
        for v in values:
            line += "|" + _cell(v, col_w)
        return line + "|"

    # Speeds vs dense (multi-thread primary)
    speedups = [
        r.multi.tokens_per_s / max(dense.multi.tokens_per_s, 1e-12) for r in results
    ]

    rows = [
        (
            "Total Parameters",
            [_fmt_int(r.param.total_params) for r in results],
        ),
        (
            "Active Params / Token",
            [
                "100%" if i == 0 else _fmt_pct(r.param.active_pct)
                for i, r in enumerate(results)
            ],
        ),
        (
            "Active Params (abs)",
            [_fmt_int(r.param.active_params_per_token) for r in results],
        ),
        (
            "FFN Active / Stored",
            [
                f"{_fmt_int(r.param.ffn_active)} / {_fmt_int(r.param.ffn_stored)}"
                for r in results
            ],
        ),
        (
            "Latency 1-thread",
            [_fmt_ms(r.single.ms_per_token) for r in results],
        ),
        (
            "Throughput 1-thread",
            [_fmt_tps(r.single.tokens_per_s) for r in results],
        ),
        (
            f"Latency {results[0].multi.threads}-thread",
            [_fmt_ms(r.multi.ms_per_token) for r in results],
        ),
        (
            f"Throughput {results[0].multi.threads}-thread",
            [_fmt_tps(r.multi.tokens_per_s) for r in results],
        ),
        (
            "Speedup vs Dense (multi)",
            [f"{s:.2f}x" for s in speedups],
        ),
        (
            "Peak RAM (gen)",
            [_fmt_mb(r.memory.peak_generate_mb) for r in results],
        ),
    ]

    print(sep)
    print(header)
    print(sep)
    for metric, values in rows:
        print(row(metric, values))
    print(sep)

    print(
        "Memory backend: "
        + " | ".join(f"{n}={r.memory.backend}" for n, r in zip(names, results))
    )
    if fff_alt is not None:
        alt_vs_pt = fff_alt.multi.tokens_per_s / max(fff.multi.tokens_per_s, 1e-12)
        print(
            f"{alt_name} vs PyTorch Hard (multi-thread): {alt_vs_pt:.2f}x "
            f"({fff_alt.multi.tokens_per_s:.2f} / {fff.multi.tokens_per_s:.2f} tok/s)"
        )
    print(
        f"FFF FFN sparsity: hard-active "
        f"{100.0 * fff.param.ffn_active / max(fff.param.ffn_stored, 1):.2f}% "
        f"of stored FFF params "
        f"({_fmt_int(fff.param.ffn_active)} / {_fmt_int(fff.param.ffn_stored)})"
    )


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
    p.add_argument("--n-embd", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--fff-depth", type=int, default=6)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=50257)
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
    p.add_argument(
        "--batch-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On CUDA: run Dense vs FFF Triton batch-size scaling (default: on)",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128,256",
        help=(
            "Comma-separated batch sizes for CUDA batch sweep "
            "(FP16 also probes 512 when VRAM allows)"
        ),
    )
    p.add_argument(
        "--batch-sweep-tokens",
        type=int,
        default=100,
        help="Timed new tokens per sequence in batch sweep (default 100)",
    )
    p.add_argument(
        "--batch-sweep-prompt-len",
        type=int,
        default=32,
        help="Prompt length for batch sweep (default 32)",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="both",
        choices=("fp32", "fp16", "bf16", "both"),
        help=(
            "Compute dtype for CUDA batch sweep: fp32, fp16, bf16, or both "
            "(fp32 then fp16; default both)"
        ),
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
) -> tuple[ModelBenchResult, ModelBenchResult, ModelBenchResult | None]:
    """Benchmark Dense vs FFF PyTorch Hard (and FFF C++ Hard on CPU)."""
    apply_hardware_optimizations(device)
    print("\n" + "-" * 72)
    print(f"Device run: {device} — {device_label(device)}")
    print("-" * 72)

    prompt = prompt_cpu.to(device)

    print("\n[1/3] Benchmarking StandardTransformer (dense MLP)...")

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

    print("\n[2/3] Benchmarking FFFTransformer (PyTorch mode=hard)...")

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
    # Rename for table clarity
    fff_param = ModelParamReport(
        name="FFF PyTorch Hard",
        total_params=fff_param.total_params,
        active_params_per_token=fff_param.active_params_per_token,
        active_pct=fff_param.active_pct,
        ffn_stored=fff_param.ffn_stored,
        ffn_active=fff_param.ffn_active,
    )
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

    fff_alt_result: ModelBenchResult | None = None
    alt_name = "FFF Alt Hard"
    if device.type == "cpu":
        alt_name = "FFF C++ Hard"
        print("\n[3/3] Benchmarking FFFTransformer (C++ mode=hard_cpp)...")
        if not is_fff_cpp_available():
            print("  SKIP — C++ extension unavailable (fallback-only)")
        else:
            with torch.no_grad():
                layer = next(iter(fff_model.fff_layers()))
                probe = torch.randn(8, layer.in_features)
                y_pt = layer.forward_hard(probe)
                y_cpp = layer.forward_hard_cpp(probe)
                max_abs = float((y_pt - y_cpp).abs().max().item())
                if not torch.allclose(y_pt, y_cpp, atol=1e-5, rtol=1e-5):
                    raise AssertionError(
                        f"C++ hard ≠ PyTorch hard (max |Δ|={max_abs:.3e})"
                    )
                print(f"  numerical match OK (max |Δ|={max_abs:.3e})")

            fff_alt_step = make_fff_cpp_step(fff_model)
            generate_timed(
                fff_alt_step,
                prompt.clone(),
                min(32, n_tokens),
                config.block_size,
            )
            alt_mem = MemoryResult(
                baseline_mb=fff_mem.baseline_mb,
                after_load_mb=fff_mem.after_load_mb,
                peak_generate_mb=_rss_mb(),
                backend=fff_mem.backend,
            )
            alt_param = ModelParamReport(
                name="FFF C++ Hard Routing",
                total_params=fff_param.total_params,
                active_params_per_token=fff_param.active_params_per_token,
                active_pct=fff_param.active_pct,
                ffn_stored=fff_param.ffn_stored,
                ffn_active=fff_param.ffn_active,
            )
            alt_single = benchmark_speed(
                fff_alt_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup,
                num_threads=1,
            )
            alt_multi = benchmark_speed(
                fff_alt_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup,
                num_threads=cpu_count,
            )
            fff_alt_result = ModelBenchResult(
                param=alt_param,
                memory=alt_mem,
                single=alt_single,
                multi=alt_multi,
            )
            print(
                f"  params={alt_param.total_params:,} | "
                f"speed={alt_multi.tokens_per_s:.2f} tok/s | "
                f"latency={alt_multi.ms_per_token:.3f} ms | "
                f"peak_RAM={alt_mem.peak_generate_mb:.1f} MB"
            )
    elif device.type == "cuda":
        alt_name = "FFF Triton CUDA Hard"
        print("\n[3/3] Benchmarking FFFTransformer (Triton CUDA mode=triton)...")
        if not is_triton_available():
            print("  SKIP — Triton not installed (`pip install triton`)")
        else:
            with torch.no_grad():
                layer = next(iter(fff_model.fff_layers()))
                probe = torch.randn(
                    8, layer.in_features, device=device, dtype=torch.float32
                )
                y_pt = layer.forward_hard(probe)
                y_tr = layer.forward_hard_triton(probe)
                max_abs = float((y_pt - y_tr).abs().max().item())
                if not torch.allclose(y_pt, y_tr, atol=1e-3, rtol=1e-3):
                    raise AssertionError(
                        f"Triton hard ≠ PyTorch hard (max |Δ|={max_abs:.3e})"
                    )
                print(f"  numerical match OK (max |Δ|={max_abs:.3e})")

            # Autotune + CUDA compile BEFORE timed runs (cost excluded from latency).
            print(
                "  warming up Triton autotuner / CUDA compile "
                "(excluded from benchmark timing)..."
            )
            warmup_fff_model_triton(
                fff_model,
                sample_tokens=max(prompt.size(1), 32),
                n_iters=max(5, warmup),
            )
            _sync_device(device)
            print("  Triton warmup done — starting timed runs")

            fff_alt_step = make_fff_triton_step(fff_model)
            generate_timed(
                fff_alt_step,
                prompt.clone(),
                min(32, n_tokens),
                config.block_size,
            )
            _sync_device(device)
            alt_mem = MemoryResult(
                baseline_mb=fff_mem.baseline_mb,
                after_load_mb=fff_mem.after_load_mb,
                peak_generate_mb=_rss_mb(),
                backend=fff_mem.backend,
            )
            alt_param = ModelParamReport(
                name="FFF Triton CUDA Hard",
                total_params=fff_param.total_params,
                active_params_per_token=fff_param.active_params_per_token,
                active_pct=fff_param.active_pct,
                ffn_stored=fff_param.ffn_stored,
                ffn_active=fff_param.ffn_active,
            )
            # GPU: thread count is less meaningful; still report 1 vs multi for table.
            alt_single = benchmark_speed(
                fff_alt_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup,
                num_threads=1,
            )
            alt_multi = benchmark_speed(
                fff_alt_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup,
                num_threads=cpu_count,
            )
            fff_alt_result = ModelBenchResult(
                param=alt_param,
                memory=alt_mem,
                single=alt_single,
                multi=alt_multi,
            )
            print(
                f"  params={alt_param.total_params:,} | "
                f"speed={alt_multi.tokens_per_s:.2f} tok/s | "
                f"latency={alt_multi.ms_per_token:.3f} ms | "
                f"peak_RAM={alt_mem.peak_generate_mb:.1f} MB"
            )
    else:
        print(f"\n[3/3] Alt FFF backend skipped on device={device.type}")

    print(f"\n=== Dense vs FFF on {device.type.upper()} ===")
    print_comparison_table(
        dense_result, fff_result, fff_alt_result, alt_name=alt_name
    )

    del fff_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dense_result, fff_result, fff_alt_result


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


# ---------------------------------------------------------------------------
# CUDA batch-size scaling (Dense vs FFF Triton crossover)
# ---------------------------------------------------------------------------


@dataclass
class BatchSweepRow:
    """One row of the CUDA batch-size scaling table."""

    batch_size: int
    dense_tok_s: float
    fff_tok_s: float
    speedup: float
    dense_peak_ram_mb: float
    fff_peak_ram_mb: float
    dense_ms_per_tok: float = float("nan")
    fff_ms_per_tok: float = float("nan")
    note: str = ""


def _precision_to_dtype(name: str) -> torch.dtype:
    """Map CLI precision name to a torch.dtype."""
    key = name.lower().strip()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"unknown precision {name!r}")


def _dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "FP16"
    if dtype == torch.bfloat16:
        return "BF16"
    return "FP32"


def _cast_model_(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """In-place cast of floating parameters / buffers to ``dtype``."""
    if dtype == torch.float32:
        return model.float()
    if dtype == torch.float16:
        return model.half()
    if dtype == torch.bfloat16:
        return model.to(dtype=torch.bfloat16)
    raise TypeError(f"unsupported model dtype {dtype}")


def _cuda_peak_ram_mb(device: torch.device, host_rss_mb: float) -> float:
    """Prefer CUDA allocator peak when on GPU; else host RSS."""
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    return host_rss_mb


@torch.no_grad()
def _time_batched_generate(
    forward_step: Callable[[Tensor], Tensor],
    prompt: Tensor,
    block_size: int,
    n_tokens: int,
    warmup_tokens: int,
) -> tuple[float, float, float]:
    """Warmup (untimed) then time greedy generation.

    Returns ``(tokens_per_s, ms_per_token, peak_ram_mb)`` where tokens =
    ``B * n_tokens``.
    """
    device = prompt.device
    batch_size = int(prompt.size(0))

    if warmup_tokens > 0:
        generate_timed(forward_step, prompt.clone(), warmup_tokens, block_size)
        _sync_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    rss_before = _rss_mb()
    _ids, elapsed = generate_timed(
        forward_step, prompt.clone(), n_tokens, block_size
    )
    _sync_device(device)
    rss_after = _rss_mb()
    peak_host = max(rss_before, rss_after, _peak_traced_mb() + rss_before)
    peak_ram = _cuda_peak_ram_mb(device, peak_host)

    total_new_tokens = batch_size * n_tokens
    tok_s = total_new_tokens / max(elapsed, 1e-12)
    ms_per_tok = (elapsed * 1000.0) / max(total_new_tokens, 1)
    return tok_s, ms_per_tok, peak_ram


def print_batch_scaling_table(
    rows: list[BatchSweepRow],
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Print Dense vs FFF Triton throughput / latency / VRAM across batch sizes."""
    label = _dtype_label(dtype)
    headers = [
        "Batch",
        "Dense tok/s",
        "FFF tok/s",
        "Speedup",
        "Dense ms/tok",
        "FFF ms/tok",
        "Peak VRAM Dense",
        "Peak VRAM FFF",
    ]
    widths = [8, 13, 12, 10, 14, 12, 16, 14]

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for cell, w in zip(cells, widths):
            parts.append(f" {cell:<{w - 1}}")
        return "|" + "|".join(parts) + "|"

    sep = "+" + "+".join("-" * w for w in widths) + "+"
    print("\n" + "=" * 100)
    print(f"CUDA Batch Scaling Summary — Dense MLP vs FFF Triton Hard ({label})")
    print(
        "Columns: Throughput (tok/s) | Latency (ms/tok) | "
        "Speedup (FFF/Dense) | Peak VRAM (MB)"
    )
    print("=" * 100)
    print(sep)
    print(fmt_row(headers))
    print(sep)

    crossover: int | None = None
    max_ok: int | None = None
    for row in rows:
        if row.note:
            cells = [
                str(row.batch_size),
                "OOM/SKIP",
                "OOM/SKIP",
                "—",
                "—",
                "—",
                "—",
                "—",
            ]
        else:
            cells = [
                str(row.batch_size),
                f"{row.dense_tok_s:.2f}",
                f"{row.fff_tok_s:.2f}",
                f"{row.speedup:.2f}x",
                f"{row.dense_ms_per_tok:.3f}",
                f"{row.fff_ms_per_tok:.3f}",
                f"{row.dense_peak_ram_mb:.1f}",
                f"{row.fff_peak_ram_mb:.1f}",
            ]
            max_ok = row.batch_size
            if crossover is None and row.speedup >= 1.0:
                crossover = row.batch_size
        print(fmt_row(cells))
    print(sep)

    if crossover is not None:
        print(
            f"Crossover ({label}): FFF Triton surpasses Dense at "
            f"batch_size >= {crossover}"
        )
    else:
        ok = [r for r in rows if not r.note]
        if ok:
            best = max(ok, key=lambda r: r.speedup)
            print(
                f"No crossover in this {label} sweep (FFF best speedup "
                f"{best.speedup:.2f}x at batch_size={best.batch_size})."
            )
        else:
            print(f"No successful {label} batch-size measurements (all OOM/SKIP).")

    if max_ok is not None:
        print(f"Max achieved batch_size ({label}): {max_ok}")
    oom_rows = [r for r in rows if r.note == "OOM"]
    if oom_rows:
        first_oom = min(r.batch_size for r in oom_rows)
        print(
            f"VRAM limit hit ({label}): OOM at batch_size={first_oom} "
            f"(larger sizes not attempted)."
        )
    print("=" * 100)


def run_cuda_batch_scaling_benchmark(
    device: torch.device,
    config: FFFConfig,
    ckpt_path: Path | None,
    *,
    batch_sizes: list[int] | None = None,
    prompt_len: int = 32,
    n_tokens: int = 100,
    warmup: int = 10,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
) -> list[BatchSweepRow]:
    """Sweep batch sizes on CUDA: Standard Dense vs FFF Triton hard routing.

    Models and activations run in ``dtype`` (``float32`` / ``float16`` /
    ``bfloat16``). Token ids remain ``int64``; embeddings emit ``dtype``.
    """
    if device.type != "cuda":
        print("Batch-size scaling skipped (CUDA only).")
        return []
    if not is_triton_available():
        print(
            "Batch-size scaling skipped — Triton unavailable "
            "(`pip install triton`)."
        )
        return []
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Batch-size scaling skipped — BF16 not supported on this GPU.")
        return []

    batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # FP16/BF16: also probe 512 when VRAM allows (OOM stop below).
    if dtype in (torch.float16, torch.bfloat16) and 512 not in batch_sizes:
        batch_sizes = list(batch_sizes) + [512]
    label = _dtype_label(dtype)
    apply_hardware_optimizations(device)

    print("\n" + "-" * 72)
    print(f"CUDA Batch Size Scaling ({label}) — Dense vs FFF Triton")
    print("-" * 72)
    print(
        f"dtype={label} | batch_sizes={batch_sizes} | prompt_len={prompt_len} | "
        f"timed={n_tokens} | warmup={warmup}"
    )

    torch.manual_seed(seed)

    def _build_dense() -> StandardTransformer:
        m = StandardTransformer(config).to(device)
        _cast_model_(m, dtype)
        m.eval()
        return m

    def _build_fff() -> FFFTransformer:
        m = FFFTransformer(config).to(device)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
        _cast_model_(m, dtype)
        m.eval()
        return m

    print(f"Loading StandardTransformer (Dense, {label})...")
    dense_model = _build_dense()
    dense_step = make_dense_step(dense_model)

    print(f"Loading FFFTransformer (Triton mode, {label})...")
    fff_model = _build_fff()
    fff_step = make_fff_triton_step(fff_model)

    print(
        f"Warming up Triton autotuner / CUDA compile [{label}] "
        "(excluded from batch-sweep timing)..."
    )
    warmup_fff_model_triton(
        fff_model,
        sample_tokens=max(prompt_len, 32),
        n_iters=max(5, warmup),
    )
    _sync_device(device)
    warm_prompt = torch.randint(
        0, config.vocab_size, (1, prompt_len), device=device, dtype=torch.long
    )
    generate_timed(fff_step, warm_prompt, min(8, n_tokens), config.block_size)
    generate_timed(dense_step, warm_prompt.clone(), min(8, n_tokens), config.block_size)
    _sync_device(device)
    print(f"Warmup done — starting {label} batch-size sweep")

    rows: list[BatchSweepRow] = []
    max_achieved: int | None = None
    for bsz in batch_sizes:
        print(f"\n  [{label}] batch_size={bsz} ...")
        try:
            prompt = torch.randint(
                0,
                config.vocab_size,
                (bsz, prompt_len),
                device=device,
                dtype=torch.long,
            )
            dense_tok_s, dense_ms, dense_ram = _time_batched_generate(
                dense_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup_tokens=warmup,
            )
            fff_tok_s, fff_ms, fff_ram = _time_batched_generate(
                fff_step,
                prompt,
                config.block_size,
                n_tokens,
                warmup_tokens=warmup,
            )
            speedup = fff_tok_s / max(dense_tok_s, 1e-12)
            row = BatchSweepRow(
                batch_size=bsz,
                dense_tok_s=dense_tok_s,
                fff_tok_s=fff_tok_s,
                speedup=speedup,
                dense_peak_ram_mb=dense_ram,
                fff_peak_ram_mb=fff_ram,
                dense_ms_per_tok=dense_ms,
                fff_ms_per_tok=fff_ms,
            )
            max_achieved = bsz
            print(
                f"    Dense={dense_tok_s:.2f} tok/s ({dense_ms:.3f} ms/tok) | "
                f"FFF Triton={fff_tok_s:.2f} tok/s ({fff_ms:.3f} ms/tok) | "
                f"speedup={speedup:.2f}x | "
                f"Peak VRAM D/F={dense_ram:.1f}/{fff_ram:.1f} MB"
            )
            rows.append(row)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            row = BatchSweepRow(
                batch_size=bsz,
                dense_tok_s=float("nan"),
                fff_tok_s=float("nan"),
                speedup=float("nan"),
                dense_peak_ram_mb=float("nan"),
                fff_peak_ram_mb=float("nan"),
                note="OOM",
            )
            rows.append(row)
            if max_achieved is not None:
                print(
                    f"    OOM at batch_size={bsz} — stopping sweep "
                    f"(max achieved batch_size={max_achieved})."
                )
            else:
                print(
                    f"    OOM at batch_size={bsz} — stopping sweep "
                    "(no successful batch sizes)."
                )
            break
        gc.collect()
        torch.cuda.empty_cache()

    print_batch_scaling_table(rows, dtype=dtype)

    del dense_model, fff_model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


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
    print(f"FFF C++ hard extension: {'available' if is_fff_cpp_available() else 'UNAVAILABLE (PyTorch fallback)'}")
    print(f"FFF Triton CUDA kernel: {'available' if is_triton_available() else 'UNAVAILABLE (pip install triton)'}")
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
    _dense_p, fff_primary, _cpp_p = run_pair_on_device(
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
        _dense_c, fff_cpu, _cpp_c = run_pair_on_device(
            torch.device("cpu"),
            config,
            ckpt_path,
            prompt_cpu,
            args.n_tokens,
            args.warmup,
            cpu_count,
        )
        print_gpu_vs_cpu_summary(primary, fff_primary, fff_cpu)

    # CUDA batch-size scaling: find Dense → FFF Triton crossover
    if primary.type == "cuda" and args.batch_sweep:
        try:
            batch_sizes = [
                int(x.strip())
                for x in str(args.batch_sizes).split(",")
                if x.strip()
            ]
        except ValueError:
            print(
                f"ERROR: invalid --batch-sizes {args.batch_sizes!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.precision == "both":
            sweep_dtypes = [torch.float32, torch.float16]
        else:
            sweep_dtypes = [_precision_to_dtype(args.precision)]
        for sweep_dtype in sweep_dtypes:
            run_cuda_batch_scaling_benchmark(
                primary,
                config,
                ckpt_path,
                batch_sizes=batch_sizes,
                prompt_len=args.batch_sweep_prompt_len,
                n_tokens=args.batch_sweep_tokens,
                warmup=args.warmup,
                seed=args.seed,
                dtype=sweep_dtype,
            )

    torch.set_num_threads(cpu_count)


if __name__ == "__main__":
    # Avoid OpenMP oversubscription surprises before we set threads explicitly.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
