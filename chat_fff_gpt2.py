#!/usr/bin/env python3
"""Interactive CLI text generation — distilled Multi-Tree CMM student.

Loads ``fff_distill_checkpoint.pt``, rebuilds GPT-2 MLPs as Multi-Tree CMM
(:class:`~models.fff_layer.FastFeedforwardLinear` with ``K`` trees of depth
``d_sub``), then runs hard routing:

- CUDA: Triton hard CMM (one kernel invocation per tree, then sum)
- CPU: ``cmm_hard_forward_cpu`` when the C++ extension is available
- Fallback: batched PyTorch hard walk

``num_trees`` (``K``) and ``depth_per_tree`` (``d_sub``) are parsed from the
checkpoint (top-level keys, nested ``config``, or ``router_weights`` shapes).
Legacy single-tree checkpoints (``fff_depth`` only) load as ``K = 1``.

Example
-------
    python chat_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
    python chat_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from device_utils import (
    amp_autocast,
    apply_hardware_optimizations,
    print_device_info,
    resolve_device,
)
from eval_fff_gpt2 import (
    _apply_no_repeat_ngram,
    _apply_repetition_penalty,
    _topk_topp_filter,
    load_student_from_checkpoint,
    warmup_student_triton,
)
from train_fff_gpt2 import (
    iter_fff_blocks,
    set_student_routing_mode,
    set_student_temperature,
    _require_transformers,
)
from models.fff_hard_triton import is_triton_available
from models.fff_layer import (
    cmm_hparams_from_checkpoint,
    is_cmm_cpp_available,
    is_fff_cpp_available,
)

DEFAULT_CHECKPOINT = Path("fff_distill_checkpoint.pt")

# High-quality sampling defaults (nucleus + anti-repetition).
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.3
DEFAULT_NO_REPEAT_NGRAM = 3
DEFAULT_MAX_NEW_TOKENS = 80


@dataclass(frozen=True)
class GenMetrics:
    """CUDA-timed generation metrics for one completion."""

    text: str
    n_tokens: int
    elapsed_ms: float
    tok_per_s: float
    ms_per_token: float


def _device_display_name(device: torch.device) -> str:
    """Short GPU / host label for the welcome banner."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def _fff_active_ratio_pct(student: Any) -> tuple[int, int, int, float]:
    """Return ``(num_trees, depth, leaves_per_tree, hard_active/stored %)``.

    Hard CMM evaluates ``K · (depth`` routers ``+ 1`` leaf). The percentage
    uses stored vs hard-active param counts from the first FFF/CMM block.
    """
    block = next(iter_fff_blocks(student), None)
    if block is None:
        return 0, 0, 0, 0.0
    layer = block.fff
    stats = layer.active_params_per_token()
    stored = max(int(stats["stored_total"]), 1)
    active = int(stats["hard_active_per_token"])
    pct = 100.0 * active / stored
    n_trees = int(getattr(layer, "num_trees", 1))
    return n_trees, int(layer.depth), int(layer.num_leaves), pct


def print_welcome(
    *,
    model_name: str,
    num_trees: int,
    fff_depth: int,
    n_leaves: int,
    active_pct: float,
    device: torch.device,
    routing: str,
    ckpt_path: Path,
    step: Any,
) -> None:
    """Stylish terminal header with model / CMM / device info."""
    gpu = _device_display_name(device)
    n_leaves_total = max(num_trees, 1) * max(n_leaves, 1)
    width = 72
    bar = "═" * width
    print()
    print(f"╔{bar}╗")
    title = " Hyperzens CMM — Multi-Tree Hard Inference "
    pad = width - len(title)
    print(f"║{title}{' ' * pad}║")
    print(f"╠{bar}╣")

    def row(label: str, value: str) -> None:
        body = f"  {label:<22} {value}"
        print(f"║{body.ljust(width)}║")

    row("Model", model_name)
    row("Checkpoint", str(ckpt_path))
    row("Train step", str(step) if step is not None else "n/a")
    row("CMM trees K", str(num_trees))
    row("Depth / tree", f"{fff_depth}  (leaves/tree={n_leaves}, total={n_leaves_total})")
    row(
        "Active param ratio",
        f"{active_pct:.3f}% hard/stored  ·  {num_trees}/{n_leaves_total} leaves",
    )
    row("Routing", routing)
    row("Device", f"{device} — {gpu}")
    row(
        "Sampling",
        f"T={DEFAULT_TEMPERATURE} top_p={DEFAULT_TOP_P} "
        f"rep={DEFAULT_REPETITION_PENALTY} ngram={DEFAULT_NO_REPEAT_NGRAM} "
        f"max_new={DEFAULT_MAX_NEW_TOKENS}",
    )
    print(f"╚{bar}╝")
    print("Type a prompt and press Enter. Commands: exit | quit | Ctrl+C")
    print()


@torch.no_grad()
def warmup_model_forward(model: Any, device: torch.device, *, seq_len: int = 16) -> None:
    """Dummy forward pass to JIT / autotune before the interactive loop."""
    vocab = int(getattr(model.config, "vocab_size", 50257))
    dummy = torch.randint(0, vocab, (1, seq_len), device=device)
    with amp_autocast(device):
        _ = model(input_ids=dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_with_cuda_timing(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = 0,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM,
    do_sample: bool = True,
) -> GenMetrics:
    """Autoregressive generation timed with ``torch.cuda.Event`` when on CUDA.

    Falls back to ``perf_counter`` on non-CUDA devices. Returns continuation
    text (prompt stripped) plus speed / latency metrics.
    """
    model.eval()
    prompt = prompt.strip()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    n_ctx = int(getattr(model.config, "n_positions", 1024))
    generated: Tensor = input_ids
    sample = bool(do_sample) and temperature > 0

    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_evt.record()
    else:
        t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        cond = generated if generated.size(1) <= n_ctx else generated[:, -n_ctx:]
        with amp_autocast(device):
            logits = model(input_ids=cond).logits[:, -1, :].float()

        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
        logits = _apply_no_repeat_ngram(logits, generated, no_repeat_ngram_size)

        if not sample:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = _topk_topp_filter(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    if use_cuda_events:
        end_evt.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start_evt.elapsed_time(end_evt))
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    prompt_len = int(input_ids.size(1))
    n_new = int(generated.size(1) - prompt_len)
    elapsed_ms = max(elapsed_ms, 1e-6)
    tok_per_s = (n_new * 1000.0) / elapsed_ms
    ms_per_token = elapsed_ms / max(n_new, 1)
    gen_ids = generated[0, prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return GenMetrics(
        text=text,
        n_tokens=n_new,
        elapsed_ms=elapsed_ms,
        tok_per_s=tok_per_s,
        ms_per_token=ms_per_token,
    )


def print_completion(metrics: GenMetrics) -> None:
    """Print generated text and timing metrics (no blank-line padding)."""
    print(f"Generated Text: {metrics.text}")
    print(f"Generation Speed: {metrics.tok_per_s:.2f} tok/s")
    print(f"Token Latency: {metrics.ms_per_token:.2f} ms/token")
    print(f"Total tokens generated: {metrics.n_tokens}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive Multi-Tree CMM inference (distilled GPT-2 student)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to fff_distill_checkpoint.pt",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (cuda>mps>cpu) | cuda | mps | cpu",
    )
    p.add_argument("--tau", type=float, default=0.10, help="Soft-router τ buffer (unused in hard)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--no-repeat-ngram-size", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    p.add_argument(
        "--force-hard",
        action="store_true",
        help="Use PyTorch hard routing even if Triton is available",
    )
    p.add_argument(
        "--num-trees",
        type=int,
        default=None,
        help="Override checkpoint CMM forest size K",
    )
    p.add_argument(
        "--depth-per-tree",
        type=int,
        default=None,
        help="Override checkpoint per-tree depth d_sub",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    apply_hardware_optimizations(device)
    print_device_info(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    print("\nLoading distilled FFF/CMM student ...")
    student, ckpt = load_student_from_checkpoint(ckpt_path, device)
    model_name = str(ckpt.get("model_name", "gpt2"))
    num_trees, fff_depth = cmm_hparams_from_checkpoint(ckpt)
    print(
        f"  CMM architecture: K={num_trees} trees, "
        f"d_sub={fff_depth} (leaves/tree={1 << fff_depth})"
    )
    if args.num_trees is not None or args.depth_per_tree is not None:
        # Rebuild if CLI overrides disagree with the loaded architecture.
        want_k = int(args.num_trees) if args.num_trees is not None else num_trees
        want_d = int(args.depth_per_tree) if args.depth_per_tree is not None else fff_depth
        layer0 = next(iter_fff_blocks(student)).fff
        if want_k != int(layer0.num_trees) or want_d != int(layer0.depth):
            print(
                f"  CLI override trees={want_k} depth={want_d} "
                f"(checkpoint had K={num_trees} d={fff_depth}) — architecture "
                "must match the checkpoint; ignoring override."
            )
        else:
            num_trees, fff_depth = want_k, want_d
    set_student_temperature(student, float(args.tau))

    routing = "hard (PyTorch CMM)"
    cpu_cmm = is_cmm_cpp_available() if num_trees > 1 else is_fff_cpp_available()
    if device.type == "cuda" and is_triton_available() and not args.force_hard:
        set_student_routing_mode(student, "triton")
        routing = "triton (CUDA Hard CMM)"
        print("Routing mode: Triton CUDA Hard CMM")
        warmup_student_triton(student, device)
    elif device.type == "cpu" and cpu_cmm and not args.force_hard:
        set_student_routing_mode(student, "hard_cpp")
        routing = "hard_cpp (CMM CPU)"
        print("Routing mode: C++ Multi-Tree CMM")
    else:
        set_student_routing_mode(student, "hard")
        if device.type == "cuda" and not is_triton_available():
            print("Triton unavailable — using PyTorch hard CMM")
        else:
            print(f"Routing mode: {routing}")

    print("Dummy forward warmup ...")
    warmup_model_forward(student, device)
    print("  warmup done\n")

    _, _, TokCls = _require_transformers()
    tokenizer = TokCls.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_trees, depth, n_leaves, active_pct = _fff_active_ratio_pct(student)
    if depth == 0:
        n_trees, depth, n_leaves = num_trees, fff_depth, 1 << fff_depth
        active_pct = 100.0 * n_trees / max(n_trees * n_leaves, 1)

    print_welcome(
        model_name=model_name,
        num_trees=n_trees,
        fff_depth=depth,
        n_leaves=n_leaves,
        active_pct=active_pct,
        device=device,
        routing=routing,
        ckpt_path=ckpt_path,
        step=ckpt.get("step"),
    )

    while True:
        try:
            raw = input("Prompt > ")
        except EOFError:
            print("\nBye.")
            break
        except KeyboardInterrupt:
            print("\nInterrupted — bye.")
            break

        prompt = raw.strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        try:
            metrics = generate_with_cuda_timing(
                student,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                do_sample=True,
            )
        except KeyboardInterrupt:
            print("\nGeneration cancelled.")
            continue
        except Exception as exc:  # noqa: BLE001 — keep CLI alive
            print(f"generation error: {exc}")
            continue

        print_completion(metrics)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
