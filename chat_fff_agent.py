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
from fff_swiglu import iter_fff_swiglu_blocks, set_fff_routing_mode, warmup_fff_swiglu
from models.fff_hard_triton import is_triton_available

DEFAULT_CHECKPOINT = Path("fff_cot_agent.pt")
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_MAX_NEW_TOKENS = 128

SYSTEM_PROMPT_CHAT_TH = (
    "คุณเป็นผู้ช่วยที่เป็นมิตร พูดภาษาไทยสุภาพ ชัดเจน "
    "ถ้าผู้ใช้ทักทาย ให้ทักทายกลับสั้นๆ แล้วถามว่าให้ช่วยอะไรได้ "
    "อย่าเขียนเรียงความ อย่าพูดถึงข้อจำกัดของ AI เอง ถ้าไม่ได้ถูกถาม"
)

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


def build_chat_input_ids(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    system: str,
    max_length: int,
) -> list[int]:
    """Tokenize ChatML via the tokenizer template (no string round-trip)."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    max_length = max(int(max_length), 8)
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    ids: Any = None
    try:
        ids = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        try:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:  # noqa: BLE001
            ids = None
    except Exception:  # noqa: BLE001
        ids = None
    if ids is None:
        text = build_chat_prompt(tokenizer, history, system=system)
        ids = encode_truncate_left(
            tokenizer, text, max_length=max_length, add_special_tokens=False
        )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    if len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def clean_reply_text(text: str) -> str:
    """Drop leaked ChatML role lines the 2B model sometimes emits."""
    text = text.strip()
    for stop in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0].strip()
    for sep in ("\nassistant\n", "\nassistant:", "\nAssistant\n"):
        if sep in text.lower() or sep in text:
            # Keep the body after the last assistant marker.
            parts = re.split(r"\nassistant\s*:?\s*\n", text, flags=re.I)
            if len(parts) > 1:
                text = parts[-1].strip()
    text = re.sub(
        r"^(?:user|assistant|system)\s*\n", "", text, count=1, flags=re.I
    ).strip()
    # A leading fake user turn: "....\nassistant\nreal reply"
    text = re.sub(
        r"^user\s*\n.*?\nassistant\s*\n", "", text, count=1, flags=re.I | re.S
    ).strip()
    return text


def build_chat_prompt(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    system: str = SYSTEM_PROMPT_CHAT_TH,
) -> str:
    """Render multi-turn history with ChatML + assistant generation prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:  # noqa: BLE001
                pass
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
    top_k: int = 20,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = 0,
    prompt_token_ids: list[int] | None = None,
) -> tuple[str, float, float, int]:
    """Sample a reply with KV cache; n-gram ban applies to **new tokens only**."""
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    n_ctx = model_context_length(model)
    max_prompt = max(n_ctx - max_new_tokens, 64)
    prompt_ids = prompt_token_ids
    if prompt_ids is None:
        prompt_ids = encode_truncate_left(
            tokenizer, prompt, max_length=max_prompt, add_special_tokens=False
        )
    if not prompt_ids:
        prompt_ids = [int(tokenizer.eos_token_id or 0)]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    vocab = model_vocab_size(model)
    tok_vocab = int(getattr(tokenizer, "vocab_size", vocab) or vocab)
    vocab_cap = min(vocab, tok_vocab)
    input_ids = input_ids.clamp(0, vocab_cap - 1)
    attention_mask = torch.ones_like(input_ids)

    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if tid is not None and int(tid) >= 0:
            eos_ids.append(int(tid))
        tid = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if tid is not None and int(tid) >= 0:
            eos_ids.append(int(tid))
    except Exception:  # noqa: BLE001
        pass
    eos_ids = list(dict.fromkeys(eos_ids)) or None

    sample = temperature > 0
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
    else:
        t0 = time.perf_counter()

    generated = _generate_cached(
        model,
        input_ids,
        attention_mask,
        device,
        max_new_tokens=max_new_tokens,
        sample=sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        vocab_cap=vocab_cap,
        eos_ids=eos_ids,
    )

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed_ms = max((time.perf_counter() - t0) * 1000.0, 1e-6)

    prompt_len = int(input_ids.size(1))
    n_new = int(generated.size(1) - prompt_len)
    tok_s = (n_new * 1000.0) / elapsed_ms
    ms_tok = elapsed_ms / max(n_new, 1)
    text = tokenizer.decode(
        generated[0, prompt_len:].tolist(),
        skip_special_tokens=True,
    ).strip()
    text = clean_reply_text(text)
    return text, tok_s, ms_tok, n_new


@torch.no_grad()
def _generate_cached(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: torch.device,
    *,
    max_new_tokens: int,
    sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    vocab_cap: int,
    eos_ids: list[int] | None,
) -> Tensor:
    """Prefill once, then decode with ``past_key_values`` (KV cache)."""
    prompt_len = int(input_ids.size(1))
    generated = input_ids
    attn = attention_mask
    past = None
    eos_set = set(eos_ids or [])
    use_cache = True

    for _ in range(max_new_tokens):
        if use_cache and past is not None:
            model_in = generated[:, -1:]
        else:
            model_in = generated

        with amp_autocast(device):
            try:
                out = model(
                    input_ids=model_in,
                    attention_mask=attn,
                    past_key_values=past if use_cache else None,
                    use_cache=use_cache,
                )
                past = getattr(out, "past_key_values", None) if use_cache else None
            except Exception:
                use_cache = False
                past = None
                out = model(
                    input_ids=generated,
                    attention_mask=attn,
                    use_cache=False,
                )

        logits = out.logits[:, -1, :].float()
        if logits.size(-1) > vocab_cap:
            logits[:, vocab_cap:] = float("-inf")

        continuation = generated[:, prompt_len:]
        if continuation.size(1) > 0:
            logits = _apply_repetition_penalty(
                logits, continuation, repetition_penalty
            )
            logits = _apply_no_repeat_ngram(
                logits, continuation, no_repeat_ngram_size
            )

        if not sample:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-8)
            logits = _topk_topp_filter(logits, top_k=top_k, top_p=top_p)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        generated = torch.cat([generated, next_id], dim=1)
        attn = torch.cat(
            [attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], dim=1
        )
        if int(next_id.item()) in eos_set:
            break

    return generated


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
    top_k: int = 20,
    system: str = SYSTEM_PROMPT_CHAT_TH,
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

        n_ctx = model_context_length(model)
        max_prompt = max(n_ctx - max_new_tokens, 64)
        prompt_ids = build_chat_input_ids(
            tokenizer, history, system=system, max_length=max_prompt
        )
        prompt = ""
        try:
            reply, tok_s, ms_tok, n_new = generate_reply(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                prompt_token_ids=prompt_ids,
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
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decode (temperature=0) — cleaner if FFF is undertrained",
    )
    p.add_argument(
        "--routing",
        type=str,
        default="triton",
        choices=("triton", "hard", "soft", "sum"),
        help=(
            "FFF path: triton/hard = 1 leaf; soft = mixture; "
            "sum = reconstruct original dense SwiGLU (sanity check)"
        ),
    )
    p.add_argument(
        "--smart-init",
        action="store_true",
        help=(
            "Ignore trained FFF weights; keep Qwen MLP slices. "
            "Use when distill checkpoint babbles (typical after max_length=64)."
        ),
    )
    p.add_argument(
        "--cot",
        action="store_true",
        help="Use the long CoT <think> system prompt (can make 2B ramble)",
    )
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
    routing = "hard" if args.force_hard else str(args.routing)
    smart_init = bool(args.smart_init)
    if smart_init and not args.force_hard and args.routing == "triton":
        routing = "sum"
        print("  --smart-init: default routing=sum (original SwiGLU reconstruction)")
    student, ckpt = load_student_from_checkpoint(
        ckpt_path,
        device,
        dtype,
        model_name=args.model_name,
        routing_mode="hard" if routing == "sum" else routing,
        smart_init_only=smart_init,
        noise_std=0.0 if smart_init else 1e-3,
    )
    if routing == "sum":
        set_fff_routing_mode(student, "sum")
    if hasattr(student, "config"):
        student.config.use_cache = True
    model_name = args.model_name or str(ckpt.get("model_name", DEFAULT_MODEL))
    routing_mode = next(iter_fff_swiglu_blocks(student)).routing_mode
    step = ckpt.get("step")
    train_cfg = ckpt.get("config") or {}
    train_len = train_cfg.get("max_length")
    print(f"checkpoint: {ckpt_path}  step={step}  model={model_name}")
    if train_len is not None:
        print(f"  trained max_length={train_len}")
    if not smart_init:
        print(
            "  note: this distill used short ChatML windows — Triton often babbles.\n"
            "  To chat with original Qwen quality:  python chat_fff_agent.py --smart-init"
        )

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
        temperature=0.0 if args.greedy else args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_history_turns=args.max_history_turns,
        top_k=args.top_k,
        system=SYSTEM_PROMPT_COT_TH if args.cot else SYSTEM_PROMPT_CHAT_TH,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nลาก่อนครับ")
        sys.exit(0)
