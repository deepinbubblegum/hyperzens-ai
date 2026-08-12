#!/usr/bin/env python3
"""Evaluate distilled Gemma FFF-GeGLU student (``gemma_fff_distill.pt``).

1. Load Gemma scaffold, inject ``FFFGemmaBlock``, load checkpoint.
2. Report WikiText-2 validation CE / PPL for Teacher (Dense) vs Student (Hard).
3. Interactive generation CLI with nucleus sampling + tok/s / ms/token.

Hard / Triton mode uses discrete tree routing (one GeGLU leaf per token).

Example
-------
    python eval_gemma.py --checkpoint gemma_fff_distill.pt --device cuda
    python eval_gemma.py --skip-ppl --device cuda   # generation only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from device_utils import (
    apply_hardware_optimizations,
    print_device_info,
    resolve_device,
)
from distill_modern_llm import resolve_compute_dtype
from eval_distilled import print_ppl_table
from eval_qwen import (
    evaluate_perplexity,
    generate_completion,
    load_teacher,
)
from fff_distill import TokenChunkDataset, load_wikitext_tokens
from fff_modern_llm import (
    FFFGemmaBlock,
    iter_fff_swiglu_blocks,
    patch_model_with_fff_gemma,
    set_fff_routing_mode,
    set_fff_temperature,
    warmup_fff_swiglu,
)
from models.fff_hard_triton import is_triton_available

DEFAULT_CHECKPOINT = Path("gemma_fff_distill.pt")
DEFAULT_MODEL = "google/gemma-2-2b"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_MAX_NEW_TOKENS = 100


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_student_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_name: str | None = None,
    routing_mode: str = "triton",
) -> tuple[Any, dict[str, Any]]:
    """Scaffold Gemma → inject FFF GeGLU → load ``student_state_dict``."""
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = model_name or str(ckpt.get("model_name", DEFAULT_MODEL))
    fff_depth = int(ckpt.get("fff_depth", 4))
    cfg = ckpt.get("config") or {}
    init_tau = float(cfg.get("init_tau", 1.0))

    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading student scaffold `{name}` (float32 init) ...")
    print("  note: gated Gemma weights need `huggingface-cli login`")
    student = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    print(f"Injecting FFF GeGLU / FFFGemmaBlock (depth={fff_depth}) ...")
    n = patch_model_with_fff_gemma(
        student, fff_depth=fff_depth, init_temp=init_tau
    )
    missing, unexpected = student.load_state_dict(
        ckpt["student_state_dict"], strict=False
    )
    if missing:
        print(f"  warning: missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  warning: unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    if dtype != torch.float32:
        print(f"Casting student → {dtype} ...")
        student = student.to(dtype=dtype)
    student = student.to(device)
    student.eval()

    if routing_mode == "triton" and (
        device.type != "cuda" or not is_triton_available()
    ):
        print("  Triton unavailable — using PyTorch hard routing")
        routing_mode = "hard"
    set_fff_routing_mode(student, routing_mode)
    set_fff_temperature(student, float(cfg.get("min_tau", 0.10)))

    # Confirm GeGLU activation on first block.
    first = next(iter_fff_swiglu_blocks(student), None)
    if first is None or not isinstance(first, FFFGemmaBlock):
        print("  warning: expected FFFGemmaBlock instances after patch")
    elif first.fff.gate_activation != "gelu_tanh":
        print(
            f"  warning: gate_activation={first.fff.gate_activation!r} "
            "(expected gelu_tanh)"
        )

    print(f"Patched {n} MLPs | routing_mode={routing_mode} | dtype={dtype}")
    return student, ckpt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate distilled Gemma FFF-GeGLU student"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to gemma_fff_distill.pt",
    )
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-eval-batches", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument(
        "--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY
    )
    p.add_argument("--fp32", action="store_true", help="Eval in float32")
    p.add_argument("--skip-ppl", action="store_true", help="Skip WikiText PPL")
    p.add_argument("--skip-gen", action="store_true", help="Skip interactive CLI")
    p.add_argument(
        "--force-hard",
        action="store_true",
        help="Force PyTorch hard routing even if Triton is available",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.device.lower() == "cuda":
            print("Falling back to auto device ...", file=sys.stderr)
            device = resolve_device("auto")
        else:
            raise SystemExit(1) from exc

    apply_hardware_optimizations(device)
    dtype = resolve_compute_dtype(device, use_bf16=not args.fp32)
    if dtype == torch.float32 and device.type == "cuda" and not args.fp32:
        dtype = torch.float16

    print("=" * 72)
    print("Evaluate Distilled Gemma FFF-GeGLU")
    print("=" * 72)
    print_device_info(device)
    print(f"Triton available: {is_triton_available()}")
    print(f"dtype={dtype}")

    ckpt_path = Path(args.checkpoint)
    routing = "hard" if args.force_hard else "triton"
    student, ckpt = load_student_from_checkpoint(
        ckpt_path,
        device,
        dtype,
        model_name=args.model_name,
        routing_mode=routing,
    )
    model_name = args.model_name or str(ckpt.get("model_name", DEFAULT_MODEL))
    fff_depth = int(ckpt.get("fff_depth", 4))
    print(
        f"checkpoint: {ckpt_path} | step={ckpt.get('step')} | "
        f"fff_depth={fff_depth} | arch={ckpt.get('architecture')}"
    )
    n_fff = sum(1 for _ in iter_fff_swiglu_blocks(student))
    print(f"FFF GeGLU blocks: {n_fff}")

    _, AutoTokenizer = _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Dummy forward warmup ...")
    warmup_fff_swiglu(student, device)
    print("  warmup done")

    if not args.skip_ppl:
        print("\nLoading WikiText-2 validation ...")
        val_tokens = load_wikitext_tokens(
            tokenizer, split="validation", max_chars=None
        )
        val_ds = TokenChunkDataset(val_tokens, args.block_size)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        print(f"val chunks={len(val_ds)} block_size={args.block_size}")

        rows: list[tuple[str, float, float]] = []

        print(f"\n[1/2] Teacher Gemma (Dense GeGLU) `{model_name}` ...")
        teacher = load_teacher(model_name, device, dtype)
        ce_t, ppl_t = evaluate_perplexity(
            teacher,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            desc="teacher",
        )
        rows.append(("Teacher Gemma (Dense)", ce_t, ppl_t))
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

        routing_mode = next(iter_fff_swiglu_blocks(student)).routing_mode
        hard_label = (
            "Student FFF Hard (Triton)"
            if routing_mode == "triton"
            else "Student FFF Hard (PyTorch)"
        )
        print(f"\n[2/2] {hard_label} ...")
        ce_s, ppl_s = evaluate_perplexity(
            student,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            desc="student-hard",
        )
        rows.append((hard_label, ce_s, ppl_s))
        print_ppl_table(rows)

    if args.skip_gen:
        return

    # Reuse Qwen interactive loop printing; override banner via monkeypatch-style
    # call by wrapping — print Gemma banner then use generate_completion directly.
    print("\n" + "=" * 72)
    print("Interactive Generation — Gemma FFF-GeGLU (Hard / Triton)")
    print(
        f"do_sample=True temperature={args.temperature} top_p={args.top_p} "
        f"repetition_penalty={args.repetition_penalty} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print("Type a prompt and press Enter. Commands: exit | quit | Ctrl+C")
    print("=" * 72)

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
            text, tok_s, ms_tok, n_new = generate_completion(
                student,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
            )
        except KeyboardInterrupt:
            print("\nGeneration cancelled.")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"generation error: {exc}")
            continue

        print(f"Generated Text: {text}")
        print(f"Generation Speed: {tok_s:.2f} tok/s")
        print(f"Latency: {ms_tok:.2f} ms/token")
        print(f"Total tokens generated: {n_new}")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
