#!/usr/bin/env python3
"""Interactive CLI text generation — distilled FFF Student (Triton Hard CUDA).

Loads ``fff_distill_checkpoint.pt``, swaps GPT-2 MLPs for :class:`FFFBlock`
in Triton hard-routing mode, warms CUDA kernels, then opens a prompt loop.

Example
-------
    python chat_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
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


def _fff_active_ratio_pct(student: Any) -> tuple[int, int, float]:
    """Return ``(depth, n_leaves, hard_active/stored %)`` from the first FFF block.

    Hard routing evaluates ``depth`` routers + 1 leaf out of ``2^depth`` leaves,
    so the leaf activation share is ``1/L`` (e.g. 6.25% at depth 4, 3.125% at
    depth 5). The reported percentage uses stored vs hard-active param counts.
    """
    block = next(iter_fff_blocks(student), None)
    if block is None:
        return 0, 0, 0.0
    layer = block.fff
    stats = layer.active_params_per_token()
    stored = max(int(stats["stored_total"]), 1)
    active = int(stats["hard_active_per_token"])
    pct = 100.0 * active / stored
    return int(layer.depth), int(layer.num_leaves), pct


def print_welcome(
    *,
    model_name: str,
    fff_depth: int,
    n_leaves: int,
    active_pct: float,
    device: torch.device,
    routing: str,
    ckpt_path: Path,
    step: Any,
) -> None:
    """Stylish terminal header with model / FFF / device info."""
    gpu = _device_display_name(device)
    leaf_share = 100.0 / max(n_leaves, 1)
    width = 72
    bar = "═" * width
    print()
    print(f"╔{bar}╗")
    title = " Hyperzens FFF — Standalone Triton Inference "
    pad = width - len(title)
    print(f"║{title}{' ' * pad}║")
    print(f"╠{bar}╣")

    def row(label: str, value: str) -> None:
        body = f"  {label:<22} {value}"
        print(f"║{body.ljust(width)}║")

    row("Model", model_name)
    row("Checkpoint", str(ckpt_path))
    row("Train step", str(step) if step is not None else "n/a")
    row("FFF depth", f"{fff_depth}  (leaves={n_leaves})")
    row(
        "Active param ratio",
        f"{active_pct:.3f}% hard/stored  ·  leaf share {leaf_share:.3f}% (1/{n_leaves})",
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
        description="Interactive FFF Triton inference (distilled student)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to fff_distill_checkpoint.pt",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda (preferred) | auto | cpu")
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
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        # Prefer CUDA for Triton; fall back gracefully.
        if args.device.lower() == "cuda":
            print("Falling back to auto device detection ...", file=sys.stderr)
            device = resolve_device("auto")
        else:
            raise SystemExit(1) from exc

    apply_hardware_optimizations(device)
    print_device_info(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    print("\nLoading distilled FFF student ...")
    student, ckpt = load_student_from_checkpoint(ckpt_path, device)
    model_name = str(ckpt.get("model_name", "gpt2"))
    fff_depth = int(ckpt.get("fff_depth", 4))
    set_student_temperature(student, float(args.tau))

    routing = "hard (PyTorch)"
    if device.type == "cuda" and is_triton_available() and not args.force_hard:
        set_student_routing_mode(student, "triton")
        routing = "triton (CUDA Hard)"
        print("Routing mode: Triton CUDA Hard")
        warmup_student_triton(student, device)
    else:
        set_student_routing_mode(student, "hard")
        if device.type == "cuda" and not is_triton_available():
            print("Triton unavailable — using PyTorch hard routing")
        else:
            print(f"Routing mode: {routing}")

    print("Dummy forward warmup ...")
    warmup_model_forward(student, device)
    print("  warmup done\n")

    _, _, TokCls = _require_transformers()
    tokenizer = TokCls.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    depth, n_leaves, active_pct = _fff_active_ratio_pct(student)
    if depth == 0:
        depth, n_leaves = fff_depth, 1 << fff_depth
        active_pct = 100.0 / n_leaves

    print_welcome(
        model_name=model_name,
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
