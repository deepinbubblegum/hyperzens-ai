#!/usr/bin/env python3
"""Interactive Thai CoT chatbot for distilled FFF-SwiGLU (``fff_cot_agent.pt``).

Loads the Ultimate CoT student in Triton Hard Mode and renders
``<think>...</think>`` trajectories with dimmed terminal styling, then the
final answer in bright text.

Generation
----------
``do_sample=True``, ``temperature=0.6``, ``top_p=0.95``,
``repetition_penalty=1.2``, ``max_new_tokens=300``.

Commands
--------
    /clear   — reset conversation history
    /raw     — toggle showing raw ``<think>`` tags
    /exit    — quit

Example
-------
    python chat_fff_agent.py --checkpoint fff_cot_agent.pt --device cuda
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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
)
from fff_hf import (
    CONTEXT_LENGTH_256K,
    apply_context_length,
    encode_truncate_left,
    load_student_from_checkpoint,
    model_context_length,
    model_vocab_size,
    resolve_compute_dtype,
)
from fff_swiglu import iter_fff_swiglu_blocks, warmup_fff_swiglu
from models.fff_hard_triton import is_triton_available

DEFAULT_CHECKPOINT = Path("fff_cot_agent.pt")
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_MAX_NEW_TOKENS = 300

SYSTEM_PROMPT_COT_TH = (
    "คุณเป็นผู้ช่วย AI ที่คิดทีละขั้นอย่างมีเหตุผล "
    "สำหรับคำถามที่ต้องใช้การวิเคราะห์ ให้เขียนกระบวนการคิดไว้ใน "
    "<think>...</think> ก่อน แล้วตามด้วยคำตอบสุดท้ายที่ชัดเจน "
    "พูดภาษาไทยได้อย่างเป็นธรรมชาติ และสุภาพ"
)

_THINK_RE = re.compile(
    r"<think>(.*?)</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _require_transformers() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoTokenizer


def print_cot_header(
    *,
    model_name: str,
    routing_mode: str,
    dtype: torch.dtype,
) -> None:
    """Thai CoT terminal banner."""
    bar = "═" * 58
    print()
    print(f"\033[95m╔{bar}╗\033[0m")
    print(
        f"\033[95m║\033[0m  "
        f"\033[1m\033[97mไฮเปอร์เซนส์ CoT ไทยแชท\033[0m"
        f"  ·  FFF + <think>"
        f"{' ' * 10}\033[95m║\033[0m"
    )
    print(
        f"\033[95m║\033[0m  "
        f"Chain-of-Thought · Triton Hard · STE Distill"
        f"{' ' * 12}\033[95m║\033[0m"
    )
    print(f"\033[95m╚{bar}╝\033[0m")
    print(f"  โมเดล     : {model_name}")
    print(f"  โหมด      : {routing_mode}  |  dtype={dtype}")
    print(
        f"  สุ่มข้อความ: T={DEFAULT_TEMPERATURE}  top_p={DEFAULT_TOP_P}  "
        f"rep={DEFAULT_REPETITION_PENALTY}  max_new={DEFAULT_MAX_NEW_TOKENS}"
    )
    print("  คำสั่ง    : /clear  |  /raw  |  /exit")
    print()


def build_chat_prompt(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    system: str = SYSTEM_PROMPT_COT_TH,
) -> str:
    """Render multi-turn history with ChatML + assistant generation prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            pass

    parts: list[str] = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def split_think_answer(text: str) -> tuple[str | None, str]:
    """Extract first ``<think>`` block and the remaining final answer."""
    match = _THINK_RE.search(text)
    if not match:
        return None, text.strip()
    thought = match.group(1).strip()
    answer = (text[: match.start()] + text[match.end() :]).strip()
    # Drop leftover tags if the model emitted extras.
    answer = re.sub(r"</?think>", "", answer, flags=re.IGNORECASE).strip()
    return thought, answer


def print_styled_reply(text: str, *, show_raw: bool) -> None:
    """Dim thought process; bright final answer."""
    if show_raw:
        print(f"\033[96mผู้ช่วย\033[0m > {text}")
        return

    thought, answer = split_think_answer(text)
    print("\033[96mผู้ช่วย\033[0m")
    if thought:
        print("  \033[2m\033[90m┌─ ความคิด (think)\033[0m")
        for line in thought.splitlines() or [thought]:
            print(f"  \033[2m\033[90m│ {line}\033[0m")
        print("  \033[2m\033[90m└─\033[0m")
    if answer:
        print(f"  \033[1m\033[97m{answer}\033[0m")
    elif not thought:
        print(f"  \033[1m\033[97m{text.strip()}\033[0m")


@torch.no_grad()
def generate_reply(
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
) -> tuple[str, float, float, int]:
    """Nucleus sample; return ``(text, tok/s, ms/token, n_new)``."""
    model.eval()
    n_ctx = model_context_length(model)
    max_prompt = max(n_ctx - max_new_tokens, 64)
    # Truncate prompt tokens at encode time (avoids HF length warnings).
    prompt_ids = encode_truncate_left(
        tokenizer, prompt, max_length=max_prompt, add_special_tokens=False
    )
    if not prompt_ids:
        prompt_ids = [int(tokenizer.eos_token_id or 0)]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    vocab = model_vocab_size(model)
    input_ids = input_ids.clamp(0, vocab - 1)

    generated: Tensor = input_ids
    sample = temperature > 0
    eos_id = tokenizer.eos_token_id
    im_end_id: int | None = None
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is not None and int(tid) >= 0:
            im_end_id = int(tid)
    except Exception:  # noqa: BLE001
        im_end_id = None

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
        tok = int(next_id.item())
        if eos_id is not None and tok == int(eos_id):
            break
        if im_end_id is not None and tok == im_end_id:
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
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text, tok_s, ms_tok, n_new


def chat_loop(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_history_turns: int = 6,
) -> None:
    """Multi-turn CoT chat with styled ``<think>`` rendering."""
    history: list[dict[str, str]] = []
    show_raw = False

    while True:
        try:
            raw = input("\033[92mคุณ\033[0m > ")
        except EOFError:
            print("\nลาก่อนครับ")
            break
        except KeyboardInterrupt:
            print("\nยกเลิก — ลาก่อนครับ")
            break

        text = raw.strip()
        if not text:
            continue
        cmd = text.lower()
        if cmd in {"/exit", "exit", "quit", "q", "/q"}:
            print("ลาก่อนครับ")
            break
        if cmd in {"/clear", "clear"}:
            history.clear()
            print("  (ล้างประวัติบทสนทนาแล้ว)")
            continue
        if cmd in {"/raw", "raw"}:
            show_raw = not show_raw
            print(f"  (แสดง raw tags = {show_raw})")
            continue

        history.append({"role": "user", "content": text})
        max_msgs = max_history_turns * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]

        prompt = build_chat_prompt(tokenizer, history)
        try:
            reply, tok_s, ms_tok, n_new = generate_reply(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        except KeyboardInterrupt:
            print("\n  (ยกเลิกการสร้างข้อความ)")
            history.pop()
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ข้อผิดพลาด: {exc}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        print_styled_reply(reply, show_raw=show_raw)
        print(
            f"  \033[90m[{n_new} tokens · {tok_s:.1f} tok/s · "
            f"{ms_tok:.1f} ms/token]\033[0m"
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thai FFF CoT interactive chat CLI")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to fff_cot_agent.pt",
    )
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument(
        "--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY
    )
    p.add_argument("--max-history-turns", type=int, default=6)
    p.add_argument(
        "--max-context-length",
        type=int,
        default=CONTEXT_LENGTH_256K,
        help="Override context window (default 262144 = 256K)",
    )
    p.add_argument("--fp32", action="store_true")
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

    print_device_info(device)
    print(f"Triton available: {is_triton_available()}")

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
    routing_mode = next(iter_fff_swiglu_blocks(student)).routing_mode

    AutoTokenizer = _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_ctx = int(ckpt.get("max_context_length", args.max_context_length))
    apply_context_length(student, tokenizer, n_ctx=n_ctx)
    print(f"context_length={model_context_length(student):,}")

    print("Dummy forward warmup ...")
    warmup_fff_swiglu(student, device)
    print("  warmup done")

    print_cot_header(
        model_name=model_name,
        routing_mode=routing_mode,
        dtype=dtype,
    )

    chat_loop(
        student,
        tokenizer,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_history_turns=args.max_history_turns,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nลาก่อนครับ")
        sys.exit(0)
