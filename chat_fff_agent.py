#!/usr/bin/env python3
"""Interactive Thai CoT + tool-using agent for distilled FFF-SwiGLU.

Loads ``fff_cot_agent.pt`` (Triton Hard when available) and runs a
**Hermes-style agent loop**:

1. Generate with KV cache.
2. If the reply contains ``<tool_call>`` blocks, execute sandboxed local
   tools (``agent_tools.py``) and inject a clipped ``<tool_response>``.
3. Generate again until a final answer (or ``--max-tool-rounds``).
4. Compact history: keep the user turn + final answer only — tool traces
   never accumulate across turns.

Renders ``<think>`` and tool rounds with dimmed terminal styling.

Commands
--------
    /clear   — reset conversation history
    /raw     — toggle showing raw ``<think>`` / tool tags
    /tools   — list sandboxed tools
    /exit    — quit

Example
-------
    python chat_fff_agent.py --checkpoint fff_cot_agent.pt --device cuda
    python chat_fff_agent.py --no-tools
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

from agent_tools import (
    ToolRegistry,
    ToolResult,
    build_system_prompt,
    format_tool_response_message,
    isolate_tool_phase,
    parse_tool_calls,
)
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
DEFAULT_MAX_TOOL_ROUNDS = 4
DEFAULT_MAX_PROMPT_TOKENS = 3072
DEFAULT_MAX_STORED_THINK_CHARS = 400
_TOOL_STOP = ("</tool_call>",)

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

SYSTEM_PROMPT_AGENT_TH = (
    "คุณเป็นผู้ช่วย AI บนเครื่องผู้ใช้ ใช้เครื่องมือท้องถิ่นเมื่อต้องอ่านไฟล์ "
    "ดูโฟลเดอร์ ดูสถานะระบบ หรือรันคำสั่ง inspect พื้นฐาน "
    "อย่าแต่งผลลัพธ์ของเครื่องมือเอง — เรียก <tool_call> แล้วรอ <tool_response> "
    "ก่อนสรุปคำตอบสุดท้าย สุภาพ ชัดเจน พูดไทยได้"
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
    print("  คำสั่ง    : /clear  |  /raw  |  /tools  |  /exit")
    print()


def _as_token_id_list(ids: Any) -> list[int] | None:
    """Normalize ``apply_chat_template`` output to a 1-D list of token ids."""
    if ids is None:
        return None
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        try:
            ids = ids["input_ids"]
        except Exception:  # noqa: BLE001
            return None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids:
        return None
    if not isinstance(ids[0], int):
        try:
            ids = [int(x) for x in ids]
        except (TypeError, ValueError):
            return None
        return ids
    return [int(x) for x in ids]


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
    raw: Any = None
    try:
        raw = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        try:
            raw = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:  # noqa: BLE001
            raw = None
    except Exception:  # noqa: BLE001
        raw = None
    ids = _as_token_id_list(raw)
    if ids is None:
        text = build_chat_prompt(tokenizer, history, system=system)
        ids = encode_truncate_left(
            tokenizer, text, max_length=max_length, add_special_tokens=False
        )
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
    """Dim thought process and tool traces; bright final answer."""
    if show_raw:
        print(f"\033[96mผู้ช่วย\033[0m > {text}")
        return

    thought, answer = split_think_answer(text)
    tool_calls = parse_tool_calls(text)
    if tool_calls and not answer:
        answer = ""
    print("\033[96mผู้ช่วย\033[0m")
    if thought:
        print("  \033[2m\033[90m┌─ ความคิด (think)\033[0m")
        for line in thought.splitlines() or [thought]:
            print(f"  \033[2m\033[90m│ {line}\033[0m")
        print("  \033[2m\033[90m└─\033[0m")
    if tool_calls:
        print("  \033[2m\033[90m┌─ tool_call\033[0m")
        for call in tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            print(f"  \033[2m\033[90m│ {call.name}({args})\033[0m")
        print("  \033[2m\033[90m└─\033[0m")
    visible = answer
    if tool_calls:
        visible = _THINK_RE.sub("", visible)
        visible = re.sub(
            r"<tool_call>.*?</tool_call>", "", visible, flags=re.DOTALL | re.I
        ).strip()
    if visible:
        print(f"  \033[1m\033[97m{visible}\033[0m")
    elif not thought and not tool_calls:
        print(f"  \033[1m\033[97m{text.strip()}\033[0m")


def compact_assistant_for_history(
    text: str,
    *,
    max_think_chars: int = DEFAULT_MAX_STORED_THINK_CHARS,
) -> str:
    """Store a short think + final answer; drop tool traces from memory.

    Tool payloads are re-injected only during the in-turn working buffer.
    Persisting them would grow the ChatML window linearly with every call.
    """
    thought, answer = split_think_answer(text)
    answer = re.sub(
        r"<tool_call>.*?</tool_call>", "", answer, flags=re.DOTALL | re.I
    ).strip()
    answer = re.sub(
        r"<tool_response>.*?</tool_response>", "", answer, flags=re.DOTALL | re.I
    ).strip()
    if thought:
        if len(thought) > max_think_chars:
            thought = thought[:max_think_chars] + "…"
        return f"<think>\n{thought}\n</think>\n{answer}".strip()
    return answer or text.strip()


def _chatml_token_len(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    system: str,
) -> int:
    """Token count of the ChatML prompt (no left-truncation)."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    try:
        raw = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=False,
        )
    except TypeError:
        try:
            raw = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
        except Exception:  # noqa: BLE001
            raw = None
    except Exception:  # noqa: BLE001
        raw = None
    ids = _as_token_id_list(raw)
    if ids is not None:
        return len(ids)
    text = build_chat_prompt(tokenizer, history, system=system)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001
        return max(len(text) // 2, 1)


def fit_history_to_budget(
    tokenizer: Any,
    history: list[dict[str, str]],
    *,
    system: str,
    max_tokens: int,
) -> list[dict[str, str]]:
    """Drop oldest turns until the ChatML prompt fits ``max_tokens``.

    Always keeps the most recent user turn (and anything after it — the
    in-flight tool traces) so the current question cannot be dropped.
    """
    if not history:
        return history
    hist = list(history)
    max_tokens = max(int(max_tokens), 64)
    while len(hist) > 2 and _chatml_token_len(tokenizer, hist, system=system) > max_tokens:
        # Drop the oldest user+assistant pair when possible.
        if hist[0]["role"] == "user" and len(hist) >= 4:
            hist = hist[2:]
        else:
            hist = hist[1:]
    return hist


def _print_tool_round(results: list[ToolResult]) -> None:
    """Live, dimmed tool I/O so the user sees progress without raw dumps."""
    for r in results:
        flag = "ok" if r.ok else "err"
        trunc = " truncated" if r.truncated else ""
        print(
            f"  \033[2m\033[33m⚙ {r.name} [{flag}] "
            f"{r.elapsed_ms:.0f} ms{trunc}\033[0m"
        )
        preview = r.payload.strip().splitlines()
        for line in preview[:4]:
            print(f"  \033[2m\033[90m  {line[:120]}\033[0m")
        if len(preview) > 4:
            print(f"  \033[2m\033[90m  … {len(preview) - 4} more lines\033[0m")


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
    stop_substrings: tuple[str, ...] = (),
) -> tuple[str, float, float, int]:
    """Sample a reply with KV cache; n-gram ban applies to **new tokens only**.

    Parameters
    ----------
    stop_substrings:
        If a decoded continuation contains any of these strings (e.g.
        ``</tool_call>``), generation stops so the model cannot hallucinate
        a ``<tool_response>``. Empty = no extra stops.
    """
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
        tokenizer,
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
        stop_substrings=stop_substrings,
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
    tokenizer: Any,
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
    stop_substrings: tuple[str, ...] = (),
) -> Tensor:
    """Prefill once, then decode with ``past_key_values`` (KV cache).

    Stops on EOS ids **or** when the decoded continuation contains a stop
    substring (used to cut generation at ``</tool_call>``).
    """
    prompt_len = int(input_ids.size(1))
    generated = input_ids
    attn = attention_mask
    past = None
    eos_set = set(eos_ids or [])
    use_cache = True
    stops = tuple(s.lower() for s in stop_substrings if s)

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
        if stops:
            piece = tokenizer.decode(
                generated[0, prompt_len:].tolist(),
                skip_special_tokens=False,
            ).lower()
            if any(s in piece for s in stops):
                break

    return generated


def _prompt_budget(model: Any, max_new_tokens: int, max_prompt_tokens: int) -> int:
    """Effective prompt cap: model window minus decode budget, then CLI cap.

    The checkpoint advertises 256K but a low-resource PC cannot materialize
    that many KV slots. ``max_prompt_tokens`` is the practical ceiling.
    """
    n_ctx = model_context_length(model)
    room = max(int(n_ctx) - int(max_new_tokens), 64)
    return max(min(room, int(max_prompt_tokens)), 64)


def _generate_once(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    working: list[dict[str, str]],
    *,
    system: str,
    max_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stop_substrings: tuple[str, ...] = (),
) -> tuple[str, float, float, int]:
    """Tokenize ``working`` ChatML and sample one continuation."""
    working = fit_history_to_budget(
        tokenizer, working, system=system, max_tokens=max_prompt
    )
    prompt_ids = build_chat_input_ids(
        tokenizer, working, system=system, max_length=max_prompt
    )
    return generate_reply(
        model,
        tokenizer,
        "",
        device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        prompt_token_ids=prompt_ids,
        stop_substrings=stop_substrings,
    )


def run_agent_turn(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    history: list[dict[str, str]],
    user_text: str,
    *,
    system: str,
    tools: ToolRegistry | None,
    max_tool_rounds: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> tuple[str, float, float, int]:
    """One user turn: generate → optional tool loop → compact into ``history``.

    In-flight tool traces live only in a working copy. After the final
    answer, ``history`` stores ``[user, compact assistant]`` so the next
    turn does not replay kilobytes of ``<tool_response>``.

    Returns
    -------
    reply, tok_s, ms_tok, n_new
        Aggregated decode stats across all generate rounds this turn.
    """
    max_prompt = _prompt_budget(model, max_new_tokens, max_prompt_tokens)
    working: list[dict[str, str]] = [
        *history,
        {"role": "user", "content": user_text},
    ]
    stop = _TOOL_STOP if tools is not None else ()
    total_new = 0
    total_ms = 0.0
    final_reply = ""
    reply = ""

    def _accumulate(n_new: int, ms_tok: float) -> None:
        nonlocal total_new, total_ms
        total_new += int(n_new)
        total_ms += float(n_new) * float(ms_tok)

    if tools is None:
        reply, _tok_s, ms_tok, n_new = _generate_once(
            model,
            tokenizer,
            device,
            working,
            system=system,
            max_prompt=max_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        _accumulate(n_new, ms_tok)
        final_reply = reply
    else:
        for round_i in range(max(int(max_tool_rounds), 0) + 1):
            allow_tools = round_i < int(max_tool_rounds)
            reply, _tok_s, ms_tok, n_new = _generate_once(
                model,
                tokenizer,
                device,
                working,
                system=system,
                max_prompt=max_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_substrings=stop if allow_tools else (),
            )
            _accumulate(n_new, ms_tok)
            reply = isolate_tool_phase(reply) if allow_tools else reply
            calls = parse_tool_calls(reply) if allow_tools else []
            if not calls:
                final_reply = reply
                break
            results = tools.execute_all(calls)
            _print_tool_round(results)
            working.append({"role": "assistant", "content": reply})
            working.append(
                {
                    "role": "user",
                    "content": format_tool_response_message(results),
                }
            )
        else:
            final_reply = compact_assistant_for_history(reply)
            if parse_tool_calls(reply):
                final_reply = (
                    final_reply
                    or "ครบจำนวนรอบเครื่องมือแล้ว — สรุปจากข้อมูลที่มีในบทสนทนา"
                )

    history.append({"role": "user", "content": user_text})
    history.append(
        {
            "role": "assistant",
            "content": compact_assistant_for_history(final_reply),
        }
    )
    avg_ms = total_ms / max(total_new, 1)
    avg_tok_s = (total_new * 1000.0) / max(total_ms, 1e-6)
    return final_reply, avg_tok_s, avg_ms, total_new


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
    tools: ToolRegistry | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> None:
    """Multi-turn CoT chat with optional Hermes tool loop + history compaction."""
    history: list[dict[str, str]] = []
    show_raw = False
    system = build_system_prompt(system, tools)
    if tools is not None:
        print(
            f"  เครื่องมือ : {', '.join(tools.names())}  "
            f"(workspace={tools.workspace})"
        )
        print(
            f"  งบบริบท  : max_prompt={max_prompt_tokens} tok  "
            f"max_tool_rounds={max_tool_rounds}  "
            f"tool_clip={tools.max_output_chars} chars"
        )
        print()

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
        if cmd in {"/tools", "tools"}:
            if tools is None:
                print("  (เครื่องมือปิดอยู่ — รันโดยไม่ใส่ --no-tools)")
            else:
                print(f"  workspace: {tools.workspace}")
                for spec in tools.specs():
                    params = ", ".join(
                        f"{k}: {v}" for k, v in spec.parameters.items()
                    )
                    print(f"  - {spec.name}({params})")
                    print(f"      {spec.description}")
            continue

        max_msgs = max_history_turns * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]

        try:
            reply, tok_s, ms_tok, n_new = run_agent_turn(
                model,
                tokenizer,
                device,
                history,
                text,
                system=system,
                tools=tools,
                max_tool_rounds=max_tool_rounds,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        except KeyboardInterrupt:
            print("\n  (ยกเลิกการสร้างข้อความ)")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ข้อผิดพลาด: {exc}")
            continue

        if len(history) > max_msgs:
            history = history[-max_msgs:]

        print_styled_reply(reply, show_raw=show_raw)
        print(
            f"  \033[90m[{n_new} tokens · {tok_s:.1f} tok/s · "
            f"{ms_tok:.1f} ms/token · hist={len(history)//2} turns]\033[0m"
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
        "--no-tools",
        action="store_true",
        help="Disable the local tool loop (plain chat)",
    )
    p.add_argument(
        "--max-tool-rounds",
        type=int,
        default=DEFAULT_MAX_TOOL_ROUNDS,
        help="Max tool-call rounds per user turn (default 4)",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=(
            "Practical ChatML prompt cap (default 3072). "
            "Does not allocate the advertised 256K window."
        ),
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Sandbox root for read_file / list_dir / run_shell",
    )
    p.add_argument(
        "--tool-output-chars",
        type=int,
        default=2400,
        help="Clip each tool payload to this many characters",
    )
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
    p.add_argument(
        "--tiny-smoke",
        action="store_true",
        help=(
            "Skip checkpoint / HF weights; run one generate via a tiny CMM "
            "stand-in (verifies the agent loop after FFF layer changes)"
        ),
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="One-shot user message then exit (skip the interactive loop)",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--force-hard",
        action="store_true",
        help="Force PyTorch hard routing even if Triton is available",
    )
    return p


def _run_tiny_agent_smoke(args: argparse.Namespace) -> None:
    """One-shot ``run_agent_turn`` on a tiny CMM stand-in (no HF checkpoint).

    Verifies the agent generate / history path after Multi-Tree CMM layer
    changes. Does not alter :func:`run_agent_turn` itself.
    """
    from sanity_cmm import TinyCmmCausalLM, TinyTokenizer

    device = torch.device("cpu")
    model = TinyCmmCausalLM()
    model.eval()
    tokenizer = TinyTokenizer()
    history: list[dict[str, str]] = []
    prompt = args.prompt or "Say hello in one short sentence."
    print("Tiny CMM agent smoke (no checkpoint) ...")
    reply, tok_s, ms_tok, n_new = run_agent_turn(
        model,
        tokenizer,
        device,
        history,
        prompt,
        system="You are a test agent.",
        tools=None,
        max_tool_rounds=0,
        max_new_tokens=min(int(args.max_new_tokens), 16),
        max_prompt_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
    )
    print(f"prompt: {prompt}")
    print(f"reply:  {reply!r}")
    print(
        f"  [{n_new} tokens · {tok_s:.1f} tok/s · {ms_tok:.1f} ms/token · "
        f"hist={len(history)}]"
    )
    if n_new < 1:
        raise SystemExit("tiny smoke produced no tokens")
    print("agent loop OK — no dimension mismatch")


def main() -> None:
    args = build_argparser().parse_args()
    if args.tiny_smoke:
        _run_tiny_agent_smoke(args)
        return
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

    tools: ToolRegistry | None = None
    if not args.no_tools:
        tools = ToolRegistry(
            Path(args.workspace),
            max_output_chars=int(args.tool_output_chars),
        )
        if args.cot:
            system = SYSTEM_PROMPT_COT_TH
        else:
            system = SYSTEM_PROMPT_AGENT_TH
    else:
        system = SYSTEM_PROMPT_COT_TH if args.cot else SYSTEM_PROMPT_CHAT_TH

    if args.prompt is not None:
        history: list[dict[str, str]] = []
        reply, tok_s, ms_tok, n_new = run_agent_turn(
            student,
            tokenizer,
            device,
            history,
            args.prompt,
            system=system,
            tools=tools,
            max_tool_rounds=args.max_tool_rounds,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            temperature=0.0 if args.greedy else args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print_styled_reply(reply, show_raw=False)
        print(
            f"  [{n_new} tokens · {tok_s:.1f} tok/s · {ms_tok:.1f} ms/token]"
        )
        return

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
        system=system,
        tools=tools,
        max_tool_rounds=args.max_tool_rounds,
        max_prompt_tokens=args.max_prompt_tokens,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nลาก่อนครับ")
        sys.exit(0)
