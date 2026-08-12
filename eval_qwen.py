#!/usr/bin/env python3
"""Evaluate distilled Qwen2.5 FFF-SwiGLU student (``qwen_fff_distill.pt``).

1. Load Qwen2.5-0.5B scaffold, inject ``FFFSwiGLUBlock``, load checkpoint.
2. Report WikiText-2 validation CE / PPL for Teacher (Dense) vs Student (Hard).
3. Interactive generation CLI with nucleus sampling + tok/s / ms/token.

Example
-------
    python eval_qwen.py --checkpoint qwen_fff_distill.pt --device cuda
    python eval_qwen.py --skip-ppl --device cuda   # generation only
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from device_utils import (
    amp_autocast,
    apply_hardware_optimizations,
    print_device_info,
    resolve_device,
)
from distill_modern_llm import resolve_compute_dtype
from eval_distilled import (
    _apply_no_repeat_ngram,
    _apply_repetition_penalty,
    _topk_topp_filter,
    print_ppl_table,
)
from fff_distill import TokenChunkDataset, load_wikitext_tokens
from fff_modern_llm import (
    iter_fff_swiglu_blocks,
    patch_model_with_fff_swiglu,
    set_fff_routing_mode,
    set_fff_temperature,
    warmup_fff_swiglu,
)
from models.fff_hard_triton import is_triton_available

DEFAULT_CHECKPOINT = Path("qwen_fff_distill.pt")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

# Interactive sampling defaults
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


def load_teacher(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Load frozen dense Qwen teacher for PPL baseline."""
    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading teacher `{model_name}` ({dtype}) ...")
    # Load FP32 then cast (stable on all platforms).
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if dtype != torch.float32:
        teacher = teacher.to(dtype=dtype)
    return teacher.to(device)


def load_student_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_name: str | None = None,
    routing_mode: str = "triton",
) -> tuple[Any, dict[str, Any]]:
    """Scaffold Qwen → inject FFF SwiGLU → load ``student_state_dict``."""
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = model_name or str(ckpt.get("model_name", DEFAULT_MODEL))
    fff_depth = int(ckpt.get("fff_depth", 4))
    cfg = ckpt.get("config") or {}
    init_tau = float(cfg.get("init_tau", 1.0))

    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading student scaffold `{name}` (float32 init) ...")
    student = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    print(f"Injecting FFF SwiGLU (depth={fff_depth}) ...")
    n = patch_model_with_fff_swiglu(
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
    print(f"Patched {n} MLPs | routing_mode={routing_mode} | dtype={dtype}")
    return student, ckpt


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    loader: DataLoader[Tensor],
    device: torch.device,
    *,
    max_batches: int | None = None,
    desc: str = "eval",
) -> tuple[float, float]:
    """Causal LM mean CE / PPL (HF shift via ``labels=input_ids``)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    vocab = int(model.config.vocab_size)

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        input_ids = batch.to(device, non_blocking=True).clamp(0, vocab - 1)
        with amp_autocast(device):
            loss = model(input_ids=input_ids, labels=input_ids).loss
        bsz, seqlen = input_ids.shape
        ntok = bsz * max(seqlen - 1, 1)
        total_nll += float(loss.detach().item()) * ntok
        total_tokens += ntok
        n_batches += 1

    if total_tokens == 0:
        return float("nan"), float("nan")
    mean_ce = total_nll / float(total_tokens)
    ppl = math.exp(min(mean_ce, 100.0))
    print(
        f"  [{desc}] batches={n_batches} tokens={total_tokens:,} "
        f"CE={mean_ce:.4f} PPL={ppl:.2f}"
    )
    return mean_ce, ppl


@torch.no_grad()
def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = True,
) -> tuple[str, float, float, int]:
    """Nucleus sample; return ``(text, tok/s, ms/token, n_new)``."""
    model.eval()
    prompt = prompt.strip()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    vocab = int(model.config.vocab_size)
    input_ids = input_ids.clamp(0, vocab - 1)
    n_ctx = int(getattr(model.config, "max_position_embeddings", 2048))
    generated: Tensor = input_ids
    sample = bool(do_sample) and temperature > 0

    use_cuda = device.type == "cuda"
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
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
            logits = _topk_topp_filter(logits, top_k=0, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        eos = tokenizer.eos_token_id
        if eos is not None and int(next_id.item()) == int(eos):
            break

    if use_cuda:
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    prompt_len = int(input_ids.size(1))
    n_new = int(generated.size(1) - prompt_len)
    elapsed_ms = max(elapsed_ms, 1e-6)
    tok_s = (n_new * 1000.0) / elapsed_ms
    ms_tok = elapsed_ms / max(n_new, 1)
    text = tokenizer.decode(
        generated[0, prompt_len:].tolist(),
        skip_special_tokens=True,
    ).strip()
    return text, tok_s, ms_tok, n_new


def interactive_loop(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> None:
    """``Prompt > `` generation loop; exit on quit / Ctrl+C."""
    print("\n" + "=" * 72)
    print("Interactive Generation — Qwen FFF-SwiGLU (Hard / Triton)")
    print(
        f"do_sample=True temperature={temperature} top_p={top_p} "
        f"repetition_penalty={repetition_penalty} max_new_tokens={max_new_tokens}"
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
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate distilled Qwen2.5 FFF-SwiGLU student"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to qwen_fff_distill.pt",
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
    # Prefer BF16 on Ampere; fall back to FP16 autocast path if BF16 unsupported
    # by keeping float32 params + amp when not CUDA-bf16.
    if dtype == torch.float32 and device.type == "cuda" and not args.fp32:
        dtype = torch.float16

    print("=" * 72)
    print("Evaluate Distilled Qwen2.5 FFF-SwiGLU")
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
        f"fff_depth={fff_depth}"
    )
    n_fff = sum(1 for _ in iter_fff_swiglu_blocks(student))
    print(f"FFF SwiGLU blocks: {n_fff}")

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

        print("\n[1/2] Teacher Qwen2.5-0.5B (Dense SwiGLU) ...")
        teacher = load_teacher(model_name, device, dtype)
        ce_t, ppl_t = evaluate_perplexity(
            teacher,
            val_loader,
            device,
            max_batches=args.max_eval_batches,
            desc="teacher",
        )
        rows.append(("Teacher Qwen2.5 (Dense)", ce_t, ppl_t))
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

    interactive_loop(
        student,
        tokenizer,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
