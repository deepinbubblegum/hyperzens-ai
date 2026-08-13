#!/usr/bin/env python3
"""Evaluate a distilled FFF Student checkpoint vs GPT-2 Teacher.

Loads ``fff_distill_checkpoint.pt``, reports validation CE / perplexity for:

* Teacher GPT-2 (Dense MLP)
* Student FFF Soft Routing (``τ = 0.10``)
* Student FFF Hard Triton CUDA (falls back to PyTorch hard if Triton missing)

Then generates sample completions with Hard Triton (nucleus sampling,
repetition penalty, no-repeat n-grams) and reports tok/s.

Example
-------
    python eval_fff_gpt2.py --checkpoint fff_distill_checkpoint.pt --device cuda
"""

from __future__ import annotations

import argparse
import math
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
from train_fff_gpt2 import (
    TokenChunkDataset,
    build_fff_student_from_teacher,
    iter_fff_blocks,
    load_gpt2_teacher,
    load_wikitext_tokens,
    set_student_routing_mode,
    set_student_temperature,
    _require_transformers,
)
from models.fff_hard_triton import is_triton_available, warmup_fff_hard_triton
from models.fff_layer import cmm_hparams_from_checkpoint


DEFAULT_CHECKPOINT = Path("fff_distill_checkpoint.pt")
DEFAULT_PROMPTS = (
    "The natural world is",
    "Computer science is",
)


def load_student_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild FFF student architecture and load ``student_state_dict``."""
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = str(ckpt.get("model_name", "gpt2"))
    num_trees, fff_depth = cmm_hparams_from_checkpoint(ckpt)
    cfg = ckpt.get("config") or {}
    init_tau = float(cfg.get("init_tau", 1.0))

    print(f"Loading teacher scaffold `{model_name}` for student architecture ...")
    teacher = load_gpt2_teacher(model_name, device=torch.device("cpu"))
    student = build_fff_student_from_teacher(
        teacher,
        fff_depth=fff_depth,
        num_trees=num_trees,
        init_temp=init_tau,
        fff_init_std=0.02,
    )
    missing, unexpected = student.load_state_dict(
        ckpt["student_state_dict"], strict=False
    )
    if missing:
        print(f"  warning: missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  warning: unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    student.to(device)
    student.eval()
    # Free scaffold teacher on CPU (eval script loads a real teacher separately).
    del teacher
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
    """Causal LM CE / PPL on token chunks (labels = input_ids, HF shift).

    Returns ``(mean_ce, perplexity)``.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        input_ids = batch.to(device, non_blocking=True)
        # GPT-2 HF: labels → internal shift; ignore_index handled by model.
        with amp_autocast(device):
            out = model(input_ids=input_ids, labels=input_ids)
            # out.loss is mean CE over non-masked tokens in the batch.
            loss = out.loss
        # Approximate token count: (T - 1) * B for causal LM.
        bsz, seqlen = input_ids.shape
        ntok = bsz * max(seqlen - 1, 1)
        total_nll += float(loss.detach().item()) * ntok
        total_tokens += ntok
        n_batches += 1

    if total_tokens == 0:
        return float("nan"), float("nan")
    mean_ce = total_nll / float(total_tokens)
    ppl = math.exp(min(mean_ce, 100.0))
    print(f"  [{desc}] batches={n_batches} tokens={total_tokens:,} CE={mean_ce:.4f} PPL={ppl:.2f}")
    return mean_ce, ppl


def _topk_topp_filter(
    logits: Tensor,
    *,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    """Filter last-dim logits with optional top-k / nucleus sampling."""
    filtered = logits
    if top_k > 0:
        k = min(top_k, filtered.size(-1))
        thresh = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < thresh, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        # Keep at least the first token.
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    return filtered


def _apply_repetition_penalty(
    logits: Tensor,
    prev_ids: Tensor,
    penalty: float,
) -> Tensor:
    """HF-style repetition penalty on tokens already present in ``prev_ids``.

    For each seen token id ``t``: if ``logits[t] < 0`` multiply by ``penalty``,
    else divide by ``penalty`` (``penalty > 1`` down-weights repeats).
    """
    if penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    # Unique ids — scatter is last-write-wins; uniqueness avoids redundant ops.
    uniq = torch.unique(prev_ids)
    score = logits[:, uniq]
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, uniq.unsqueeze(0).expand(logits.size(0), -1), score)


def _apply_no_repeat_ngram(
    logits: Tensor,
    prev_ids: Tensor,
    ngram_size: int,
) -> Tensor:
    """Ban tokens that would recreate an already-seen ``ngram_size``-gram."""
    if ngram_size <= 0 or prev_ids.size(1) < ngram_size:
        return logits
    seq = prev_ids[0].tolist()
    prefix = tuple(seq[-(ngram_size - 1) :])
    banned: set[int] = set()
    for i in range(len(seq) - ngram_size + 1):
        if tuple(seq[i : i + ngram_size - 1]) == prefix:
            banned.add(int(seq[i + ngram_size - 1]))
    for tid in banned:
        logits[0, tid] = float("-inf")
    return logits


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
    do_sample: bool = True,
    greedy: bool = False,
) -> tuple[str, float]:
    """Autoregressive generation with nucleus sampling + anti-repetition.

    Parameters match common HF ``generate`` knobs: ``do_sample``, ``temperature``,
    ``top_p``, ``repetition_penalty``, ``no_repeat_ngram_size``, ``max_new_tokens``.

    Returns
    -------
    generated_text:
        Continuation only (prompt stripped), whitespace-trimmed.
    tokens_per_second:
        New tokens / wall time (CUDA synchronized when applicable).
    """
    model.eval()
    prompt = prompt.strip()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Cap context to model block size.
    n_ctx = int(getattr(model.config, "n_positions", 1024))
    generated = input_ids
    sample = bool(do_sample) and not greedy

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        cond = generated if generated.size(1) <= n_ctx else generated[:, -n_ctx:]
        with amp_autocast(device):
            logits = model(input_ids=cond).logits[:, -1, :].float()

        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
        logits = _apply_no_repeat_ngram(logits, generated, no_repeat_ngram_size)

        if not sample or temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = _topk_topp_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - t0, 1e-12)
    prompt_len = int(input_ids.size(1))
    n_new = int(generated.size(1) - prompt_len)
    tok_s = n_new / elapsed
    gen_ids = generated[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return generated_text, tok_s


def print_generation_result(
    prompt: str,
    generated_text: str,
    tok_s: float,
) -> None:
    """Print speed / prompt / continuation without blank-line padding."""
    print(f"Speed (tok/s): {tok_s:.2f}")
    print(f"Prompt: {prompt.strip()}")
    print(f"Generated Text: {generated_text}")


def print_ppl_table(rows: list[tuple[str, float, float]]) -> None:
    """ASCII table: model | CE | PPL."""
    headers = ("Model", "CE Loss", "Perplexity")
    widths = (36, 12, 12)

    def fmt(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\n" + "=" * 72)
    print("Validation Perplexity — WikiText-2 (raw)")
    print("=" * 72)
    print(sep)
    print(fmt(headers))
    print(sep)
    for name, ce, ppl in rows:
        print(fmt((name, f"{ce:.4f}", f"{ppl:.2f}")))
    print(sep)
    print("=" * 72)


def warmup_student_triton(student: Any, device: torch.device) -> None:
    """Warm Triton autotune on each FFF layer (excluded from timed gen)."""
    if device.type != "cuda" or not is_triton_available():
        return
    print("Warming FFF Triton kernels ...")
    for block in iter_fff_blocks(student):
        layer = block.fff
        x = torch.randn(
            8,
            layer.in_features,
            device=device,
            dtype=next(layer.parameters()).dtype,
        )
        warmup_fff_hard_triton(
            x,
            layer.router_weights.detach(),
            layer.router_biases.detach(),
            layer.leaf_weights.detach(),
            layer.leaf_biases.detach(),
            int(layer.depth),
            n_iters=3,
        )
    torch.cuda.synchronize(device)
    print("  Triton warmup done")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate distilled FFF student")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to fff_distill_checkpoint.pt",
    )
    p.add_argument("--model-name", type=str, default=None, help="Override HF teacher id")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (cuda>mps>cpu) | cuda | mps | cpu",
    )
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-eval-batches", type=int, default=100)
    p.add_argument("--tau", type=float, default=0.10, help="Soft-routing temperature")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0, help="0 disables top-k (nucleus-only)")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decoding (disables do_sample)",
    )
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Generation prompt (repeatable). Defaults to built-in samples.",
    )
    p.add_argument("--skip-gen", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    apply_hardware_optimizations(device)

    print("=" * 72)
    print("Evaluate Distilled FFF Student")
    print("=" * 72)
    print_device_info(device)
    print(f"Triton available: {is_triton_available()}")

    ckpt_path = Path(args.checkpoint)
    student, ckpt = load_student_from_checkpoint(ckpt_path, device)
    model_name = args.model_name or str(ckpt.get("model_name", "gpt2"))
    fff_depth = int(ckpt.get("fff_depth", 4))
    print(f"checkpoint: {ckpt_path} | step={ckpt.get('step')} | fff_depth={fff_depth}")

    print(f"\nLoading teacher `{model_name}` ...")
    teacher = load_gpt2_teacher(model_name, device)

    _, _, TokCls = _require_transformers()
    tokenizer = TokCls.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading WikiText-2 validation split ...")
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

    # --- Teacher ---
    print("\n[1/3] Teacher GPT-2 (Dense MLP) ...")
    ce_t, ppl_t = evaluate_perplexity(
        teacher,
        val_loader,
        device,
        max_batches=args.max_eval_batches,
        desc="teacher",
    )
    rows.append(("Teacher GPT-2 (Dense)", ce_t, ppl_t))

    # --- Student soft ---
    print(f"\n[2/3] Student FFF Soft Routing (τ={args.tau:.2f}) ...")
    set_student_routing_mode(student, "soft")
    set_student_temperature(student, args.tau)
    ce_s, ppl_s = evaluate_perplexity(
        student,
        val_loader,
        device,
        max_batches=args.max_eval_batches,
        desc="student-soft",
    )
    rows.append((f"Student FFF Soft (τ={args.tau:.2f})", ce_s, ppl_s))

    # --- Student hard / triton ---
    hard_label = "Student FFF Hard (PyTorch)"
    if device.type == "cuda" and is_triton_available():
        set_student_routing_mode(student, "triton")
        hard_label = "Student FFF Hard (Triton)"
        warmup_student_triton(student, device)
    else:
        set_student_routing_mode(student, "hard")
        print("  Triton unavailable — using PyTorch hard routing")

    print(f"\n[3/3] {hard_label} ...")
    ce_h, ppl_h = evaluate_perplexity(
        student,
        val_loader,
        device,
        max_batches=args.max_eval_batches,
        desc="student-hard",
    )
    rows.append((hard_label, ce_h, ppl_h))

    print_ppl_table(rows)

    if args.skip_gen:
        return

    # Keep hard/triton mode for generation.
    prompts = args.prompt if args.prompt else list(DEFAULT_PROMPTS)
    print("\n" + "=" * 72)
    print(f"Text Generation — {hard_label}")
    print(
        f"do_sample=True temperature={args.temperature} top_p={args.top_p} "
        f"repetition_penalty={args.repetition_penalty} "
        f"no_repeat_ngram_size={args.no_repeat_ngram_size} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print("=" * 72)

    for i, prompt in enumerate(prompts):
        if i > 0:
            print("---")
        generated_text, tok_s = generate_text(
            student,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            do_sample=not args.greedy,
            greedy=args.greedy,
        )
        print_generation_result(prompt, generated_text, tok_s)
    print("=" * 72)


if __name__ == "__main__":
    main()
