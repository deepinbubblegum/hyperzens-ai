#!/usr/bin/env python3
"""Ultimate multi-domain FFF distillation with Chain-of-Thought ``<think>`` tags.

Student
-------
``Qwen/Qwen3.5-2B`` patched with :class:`FFFSwiGLUBlock`
(depth=4 → 16 leaves). FP32 smart-init → BF16. STE hard-aware soft routing
matches Triton Hard inference. Fits RTX 3060 12GB comfortably with a 4-bit
4B teacher.

Teacher
-------
Default ``Qwen/Qwen3.5-4B`` in bitsandbytes 4-bit
(``bnb_4bit_compute_dtype=bfloat16``). Same-family Top-K KL (vocab aligned).

VRAM note
---------
Default: teacher 4B 4-bit + student 2B BF16 targets ~12GB cards with headroom.
Larger pair: ``--teacher-name Qwen/Qwen3.5-9B --student-name Qwen/Qwen3.5-4B``.

Data mixture (20% each)
-----------------------
1. **CoT reasoning** — Thai/English trajectories with ``<think>...</think>``
2. **Agent / tool use** — Hermes function-calling (``<tools>`` / ``<tool_call>``)
3. **Coding** — ``flytech/python-codes-25k``
4. **Thai conversation** — WangchanThaiInstruct
5. **English reasoning** — OpenHermes-2.5

ChatML assistant turns for CoT::

    <|im_start|>assistant
    <think>
    {reasoning_steps}
    </think>
    {final_response}<|im_end|>

Memory engine
-------------
* Top-K KL (``K=200``) + feature matching on last hidden states
* Freeze non-FFF backbone + AdamW8bit (fits RTX 3060 12GB)
* Gradient checkpointing on the student
* Model context ``262144`` (256K); train micro-batch ``max_length=64``
* ``batch_size=1``, ``grad_accum_steps=16``
* ``max_steps=4000``, Cosine ``lr_leaf=2e-4``, ``lr_router=5e-5``

Checkpoint
----------
``fff_cot_agent.pt``

Example
-------
    python train_fff_agent.py --device cuda --max-steps 4000
    # Stronger pair (needs more VRAM):
    python train_fff_agent.py --teacher-name Qwen/Qwen3.5-9B --student-name Qwen/Qwen3.5-4B --device cuda
    python train_fff_agent.py --smoke-synthetic-data --max-steps 20
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from device_utils import (
    apply_hardware_optimizations,
    pin_memory_for,
    print_device_info,
    resolve_device,
)
from fff_hf import (
    CONTEXT_LENGTH_256K,
    annealed_tau,
    apply_context_length,
    bf16_autocast,
    build_fff_student,
    build_student_optimizer,
    build_student_param_groups,
    encode_truncate_left,
    extract_teacher_topk,
    gather_student_leaf_probs,
    load_4bit_teacher,
    load_dense_teacher,
    model_context_length,
    model_vocab_size,
    resolve_compute_dtype,
    topk_kl_distill_loss,
)
from fff_swiglu import (
    iter_fff_swiglu_blocks,
    set_fff_temperature,
)

CHECKPOINT_NAME = "fff_cot_agent.pt"
DEFAULT_STUDENT = "Qwen/Qwen3.5-2B"
DEFAULT_TEACHER = "Qwen/Qwen3.5-4B"
# Stronger pairs if VRAM allows:
#   teacher 9B + student 4B, or teacher 9B + student 9B (≥24GB)

DOMAIN_COT = "cot"
DOMAIN_AGENT = "agent"
DOMAIN_CODE = "code"
DOMAIN_THAI = "thai"
DOMAIN_ENGLISH = "english"
DOMAINS: tuple[str, ...] = (
    DOMAIN_COT,
    DOMAIN_AGENT,
    DOMAIN_CODE,
    DOMAIN_THAI,
    DOMAIN_ENGLISH,
)

DEFAULT_SYSTEM = (
    "You are a helpful AI assistant with coding, Thai language, tool-use, "
    "and step-by-step reasoning skills. "
    "For non-trivial questions, think inside <think>...</think> before the "
    "final answer. When calling tools, emit <tool_call>...</tool_call> blocks."
)

# Primary + fallback HF dataset ids per domain (20% mixture each).
DOMAIN_DATASETS: dict[str, tuple[str, ...]] = {
    DOMAIN_COT: (
        "FreedomIntelligence/phoenix-sft-thai-cot",
        "RJTPP/medical-o1-reasoning-SFT-TH",
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "Jnx03/kanitakorn-th-sft-v3",
    ),
    DOMAIN_AGENT: (
        "NousResearch/hermes-function-calling-v1",
        "NousResearch/Hermes-Function-Calling-V1",
    ),
    DOMAIN_CODE: (
        "flytech/python-codes-25k",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
    ),
    DOMAIN_THAI: (
        "airesearch/WangchanThaiInstruct",
        "pythainlp/wangchanthaiinstruct",
        "Thaweewat/alpaca-cleaned-52k-th",
    ),
    DOMAIN_ENGLISH: (
        "teknium/OpenHermes-2.5",
        "teknium/openhermes",
        "HuggingFaceH4/ultrachat_200k",
    ),
}


def _require_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoTokenizer


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------


_ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "user",  # tool responses folded as user-visible context in ChatML
    "function": "user",
    "tool_call": "assistant",
}


def format_assistant_with_think(reasoning: str, final_response: str) -> str:
    """Wrap CoT trajectory in ``<think>...</think>`` before the final answer.

    Produces the assistant body used inside ChatML::

        <think>
        {reasoning_steps}
        </think>
        {final_response}
    """
    reasoning = reasoning.strip()
    final_response = final_response.strip()
    if not reasoning:
        return final_response
    # Avoid double-wrapping if the source already used think tags.
    if "<think>" in reasoning or "<think>" in final_response:
        if final_response.startswith("<think>"):
            return final_response
        if reasoning.startswith("<think>"):
            return f"{reasoning}\n{final_response}".strip()
    return f"<think>\n{reasoning}\n</think>\n{final_response}"


def messages_to_chatml(
    messages: list[dict[str, str]],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Render OpenAI-style messages → Qwen/Hermes ChatML string.

    Preserves ``<think>``, ``<tool_call>``, and ``<tool_response>`` text.
    """
    has_system = any(m.get("role") == "system" for m in messages)
    parts: list[str] = []
    if not has_system and default_system:
        parts.append(f"<|im_start|>system\n{default_system}<|im_end|>\n")
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def sharegpt_to_messages(conversations: Any) -> list[dict[str, str]]:
    """Convert ShareGPT ``conversations`` list → ``[{role, content}, ...]``."""
    if not isinstance(conversations, list):
        return []
    out: list[dict[str, str]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        raw_role = str(
            turn.get("from") or turn.get("role") or turn.get("speaker") or ""
        ).lower()
        content = str(
            turn.get("value") or turn.get("content") or turn.get("text") or ""
        ).strip()
        if not content:
            continue
        role = _ROLE_MAP.get(raw_role, "user")
        # Hermes tool turns: keep tag body; map tool → user for ChatML roles.
        if raw_role == "tool" and "<tool_response>" not in content:
            content = f"<tool_response>\n{content}\n</tool_response>"
        out.append({"role": role, "content": content})
    return out


def row_to_messages(row: dict[str, Any], domain: str) -> list[dict[str, str]] | None:
    """Map a heterogeneous HF row into ChatML messages for ``domain``."""
    # CoT / o1-style: Question + Complex_CoT + Response (Thai medical-o1, etc.)
    if domain == DOMAIN_COT or any(
        k in row for k in ("Complex_CoT", "complex_cot", "reasoning", "cot")
    ):
        q = str(
            row.get("Question")
            or row.get("question")
            or row.get("prompt")
            or row.get("instruction")
            or ""
        ).strip()
        think = str(
            row.get("Complex_CoT")
            or row.get("complex_cot")
            or row.get("reasoning")
            or row.get("cot")
            or row.get("thought")
            or ""
        ).strip()
        ans = str(
            row.get("Response")
            or row.get("response")
            or row.get("answer")
            or row.get("output")
            or ""
        ).strip()
        if q and (think or ans):
            assistant = format_assistant_with_think(think, ans or think)
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant},
            ]

    # Already message list — ensure assistant CoT domain wraps bare answers.
    for key in ("messages", "conversations"):
        if key in row and isinstance(row[key], list):
            msgs = sharegpt_to_messages(row[key])
            if len(msgs) >= 2:
                if domain == DOMAIN_COT:
                    for m in msgs:
                        if m["role"] == "assistant" and "<think>" not in m["content"]:
                            # Soft-wrap short answers with a minimal think stub.
                            m["content"] = format_assistant_with_think(
                                "พิจารณาคำถามและเหตุผลก่อนตอบ",
                                m["content"],
                            )
                return msgs

    # Wangchan / Alpaca Thai
    if "Instruction" in row and "Output" in row:
        instr = str(row.get("Instruction") or "").strip()
        inp = str(row.get("Input") or "").strip()
        out = str(row.get("Output") or "").strip()
        if instr and out:
            user = f"{instr}\n{inp}".strip() if inp else instr
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": out},
            ]

    if "instruction" in row and ("output" in row or "response" in row):
        instr = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        out = str(row.get("output") or row.get("response") or "").strip()
        if instr and out:
            user = f"{instr}\n{inp}".strip() if inp else instr
            asst = out
            if domain == DOMAIN_COT and "<think>" not in out:
                asst = format_assistant_with_think(
                    "วิเคราะห์โจทย์และเลือกคำตอบที่สมเหตุสมผล",
                    out,
                )
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ]

    # Coding: flytech python-codes / Magicoder-style
    if domain == DOMAIN_CODE:
        for uk, ak in (
            ("prompt", "completion"),
            ("problem", "solution"),
            ("question", "answer"),
            ("text", "code"),
            ("instruction", "code"),
        ):
            if uk in row and ak in row:
                u = str(row[uk] or "").strip()
                a = str(row[ak] or "").strip()
                if u and a:
                    return [
                        {"role": "system", "content": DEFAULT_SYSTEM},
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ]
        for key in ("text", "content", "code"):
            if key in row and str(row[key] or "").strip():
                body = str(row[key]).strip()
                return [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {
                        "role": "user",
                        "content": "Complete or explain the following code task.",
                    },
                    {"role": "assistant", "content": body},
                ]

    # Generic prompt/response
    for uk, ak in (
        ("prompt", "response"),
        ("question", "answer"),
        ("query", "answer"),
        ("input", "output"),
    ):
        if uk in row and ak in row:
            u = str(row[uk] or "").strip()
            a = str(row[ak] or "").strip()
            if u and a:
                return [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
    return None


def _synthetic_domain_messages(domain: str, n: int) -> list[list[dict[str, str]]]:
    """Offline multi-domain ChatML samples for smoke tests."""
    templates: dict[str, list[list[dict[str, str]]]] = {
        DOMAIN_COT: [
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": "ถ้ามีแอปเปิล 3 ผล แล้วได้เพิ่มอีก 2 ผล จะมีกี่ผล?",
                },
                {
                    "role": "assistant",
                    "content": format_assistant_with_think(
                        "เริ่มจาก 3 ผล แล้วบวกเพิ่ม 2 ผล รวมเป็น 3+2=5",
                        "มีแอปเปิลทั้งหมด 5 ผลครับ",
                    ),
                },
            ]
        ],
        DOMAIN_AGENT: [
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": "What's the weather in Bangkok tomorrow?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "<tool_call>\n"
                        '{"name": "get_weather", "arguments": '
                        '{"city": "Bangkok", "when": "tomorrow"}}\n'
                        "</tool_call>"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "<tool_response>\n"
                        '{"temp_c": 33, "condition": "partly cloudy"}\n'
                        "</tool_response>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Tomorrow in Bangkok looks partly cloudy around 33°C.",
                },
            ]
        ],
        DOMAIN_CODE: [
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": "Write a Python function that reverses a string.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "```python\n"
                        "def reverse_string(s: str) -> str:\n"
                        "    return s[::-1]\n"
                        "```"
                    ),
                },
            ]
        ],
        DOMAIN_THAI: [
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": "อธิบายว่า Fast Feedforward คืออะไร"},
                {
                    "role": "assistant",
                    "content": (
                        "Fast Feedforward (FFF) คือโครงข่ายที่แบ่งพื้นที่ "
                        "อินพุตด้วยต้นไม้ตัดสินใจ แล้วคำนวณเฉพาะใบเดียว "
                        "ตอน inference ทำให้เร็วขึ้นมากครับ"
                    ),
                },
            ]
        ],
        DOMAIN_ENGLISH: [
            [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": "Explain step by step: why is 2+2=4?",
                },
                {
                    "role": "assistant",
                    "content": format_assistant_with_think(
                        "2 is S(S(0)); addition is recursive. "
                        "2+2 = S(S(2)) which is 4.",
                        "Therefore 2+2 equals 4.",
                    ),
                },
            ]
        ],
    }
    seeds = templates[domain]
    out: list[list[dict[str, str]]] = []
    while len(out) < n:
        out.extend(seeds)
    return out[:n]


# ---------------------------------------------------------------------------
# Dataset loading + 20% mixture sampler
# ---------------------------------------------------------------------------


def _load_hf_split(name: str, *, max_rows: int) -> list[dict[str, Any]]:
    """Load up to ``max_rows`` from a HF dataset (best-effort configs)."""
    from datasets import load_dataset

    # Hermes function-calling configs
    config_tries: list[str | None]
    if "hermes-function-calling" in name.lower() or "Hermes-Function" in name:
        config_tries = [
            "func_calling",
            "func_calling_singleturn",
            "glaive_func_calling",
            None,
        ]
    else:
        config_tries = [None]

    last_err: Exception | None = None
    for cfg in config_tries:
        try:
            kwargs: dict[str, Any] = {"split": "train"}
            if cfg is not None:
                ds = load_dataset(name, cfg, **kwargs)
            else:
                try:
                    ds = load_dataset(name, **kwargs)
                except Exception:
                    raw = load_dataset(name)
                    if hasattr(raw, "keys"):
                        split = "train" if "train" in raw else next(iter(raw.keys()))
                        ds = raw[split]
                    else:
                        ds = raw
            rows: list[dict[str, Any]] = []
            for i, row in enumerate(ds):
                rows.append(dict(row))
                if i + 1 >= max_rows:
                    break
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"failed to load `{name}`: {last_err}")


def load_domain_message_lists(
    domain: str,
    *,
    max_samples: int,
    smoke_synthetic: bool,
) -> list[list[dict[str, str]]]:
    """Load ChatML message lists for one domain (with HF fallbacks)."""
    if smoke_synthetic:
        return _synthetic_domain_messages(domain, max(1, int(max_samples)))

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "datasets is required: pip install datasets\n"
            "Or pass --smoke-synthetic-data."
        ) from exc

    last_err: Exception | None = None
    for name in DOMAIN_DATASETS[domain]:
        try:
            print(f"  [{domain}] loading `{name}` ...")
            rows = _load_hf_split(name, max_rows=max(max_samples * 3, max_samples))
            msgs_list: list[list[dict[str, str]]] = []
            for row in rows:
                msgs = row_to_messages(row, domain)
                if msgs is not None and len(msgs) >= 2:
                    msgs_list.append(msgs)
                if len(msgs_list) >= max_samples:
                    break
            if msgs_list:
                print(f"  [{domain}] {len(msgs_list):,} samples from `{name}`")
                return msgs_list
            print(f"  [{domain}] `{name}` had no usable rows")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  [{domain}] `{name}` failed: {exc.__class__.__name__}: {exc}")

    print(f"  [{domain}] falling back to synthetic samples (last={last_err})")
    return _synthetic_domain_messages(domain, max(1, int(max_samples)))


class MultiDomainChatMixture(IterableDataset[Tensor]):
    """Streaming equal-weight mixture over ChatML domains (20% each when 5).

    Each yield is a right-padded ``input_ids`` tensor of shape ``(max_length,)``.
    Domain choice is uniform; within a domain samples cycle forever.
    """

    def __init__(
        self,
        domain_messages: dict[str, list[list[dict[str, str]]]],
        tokenizer: Any,
        *,
        max_length: int = 64,
        seed: int = 42,
        default_system: str = DEFAULT_SYSTEM,
    ) -> None:
        super().__init__()
        self.domains = [d for d in DOMAINS if domain_messages.get(d)]
        if not self.domains:
            raise ValueError("no domains with samples")
        self.domain_messages = domain_messages
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.default_system = default_system
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.pad_id = int(pad_id)
        # Pre-tokenize for throughput (still streamed via infinite cycle).
        self._tokenized: dict[str, list[Tensor]] = {}
        for domain in self.domains:
            encoded: list[Tensor] = []
            for msgs in domain_messages[domain]:
                text = messages_to_chatml(msgs, default_system=default_system)
                # Left-truncate to max_length inside the tokenizer (keeps assistant
                # tail; avoids HF warnings when raw ChatML >> model_max_length).
                ids = encode_truncate_left(
                    tokenizer, text, max_length=self.max_length
                )
                if not ids:
                    continue
                t = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
                t[: len(ids)] = torch.tensor(ids, dtype=torch.long)
                encoded.append(t)
            if not encoded:
                raise ValueError(f"domain `{domain}` produced zero tokenized samples")
            self._tokenized[domain] = encoded
            print(
                f"  mixture[{domain}]: {len(encoded):,} tokenized "
                f"(max_length={self.max_length})"
            )

    def __iter__(self) -> Iterator[Tensor]:
        rng = random.Random(self.seed + int(torch.initial_seed() % 10_000))
        cursors = {d: 0 for d in self.domains}
        while True:
            domain = rng.choice(self.domains)  # equal weight among available
            bucket = self._tokenized[domain]
            idx = cursors[domain] % len(bucket)
            cursors[domain] = idx + 1
            yield bucket[idx]


# ---------------------------------------------------------------------------
# Loss helpers (Top-K KL + feature matching)
# ---------------------------------------------------------------------------


def feature_matching_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
) -> Tensor:
    """MSE between L2-normalized mean-pooled last hidden states.

    Handles different widths (1.5B vs 7B) by adaptive-pooling both sides to
    ``min(D_s, D_t)`` before the MSE — cheap and VRAM-friendly vs full-seq
    feature maps.
    """
    # (B, T, D) → (B, D)
    s = student_hidden.float().mean(dim=1)
    t = teacher_hidden.float().mean(dim=1).detach()
    target = min(int(s.size(-1)), int(t.size(-1)))
    s = F.adaptive_avg_pool1d(s.unsqueeze(1), target).squeeze(1)
    t = F.adaptive_avg_pool1d(t.unsqueeze(1), target).squeeze(1)
    return F.mse_loss(F.normalize(s, dim=-1), F.normalize(t, dim=-1))


def ultimate_distill_loss(
    student_logits: Tensor,
    routing_probs: Tensor,
    *,
    teacher_topk_val: Tensor,
    teacher_topk_idx: Tensor,
    student_hidden: Tensor | None = None,
    teacher_hidden: Tensor | None = None,
    kl_temperature: float = 2.0,
    entropy_coef: float = 0.01,
    ce_coef: float = 0.1,
    feature_coef: float = 0.5,
    kl_topk: int = 200,
    labels: Tensor | None = None,
    pad_id: int = 0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Top-K KL + optional feature MSE + CE − leaf entropy."""
    loss, parts = topk_kl_distill_loss(
        student_logits,
        None,
        routing_probs,
        kl_temperature=kl_temperature,
        entropy_coef=entropy_coef,
        ce_coef=ce_coef,
        kl_topk=kl_topk,
        labels=labels,
        pad_id=pad_id,
        teacher_topk_val=teacher_topk_val,
        teacher_topk_idx=teacher_topk_idx,
    )
    loss_feat = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    if (
        feature_coef > 0.0
        and student_hidden is not None
        and teacher_hidden is not None
    ):
        loss_feat = feature_matching_loss(student_hidden, teacher_hidden)
        loss = loss + float(feature_coef) * loss_feat
    parts = dict(parts)
    parts["feature"] = loss_feat
    parts["total"] = loss
    return loss, parts


# ---------------------------------------------------------------------------
# Config / train
# ---------------------------------------------------------------------------


@dataclass
class UltimateConfig:
    student_name: str = DEFAULT_STUDENT
    teacher_name: str = DEFAULT_TEACHER
    fff_depth: int = 4
    init_tau: float = 1.0
    min_tau: float = 0.10
    kl_temperature: float = 2.0
    entropy_coef: float = 0.01
    ce_coef: float = 0.1
    feature_coef: float = 0.5
    kl_topk: int = 200
    max_context_length: int = CONTEXT_LENGTH_256K
    max_length: int = 64
    batch_size: int = 1
    grad_accum_steps: int = 16
    lr_leaf: float = 2e-4
    lr_router: float = 5e-5
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 4000
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    max_samples_per_domain: int = 8_000
    device: str = "cuda"
    checkpoint: str = CHECKPOINT_NAME
    use_bf16: bool = True
    teacher_4bit: bool = True
    freeze_backbone: bool = True
    adam_8bit: bool = True
    smoke_synthetic_data: bool = False


def build_argparser() -> argparse.ArgumentParser:
    d = UltimateConfig()
    p = argparse.ArgumentParser(
        description="Ultimate CoT multi-domain FFF-SwiGLU distillation"
    )
    p.add_argument("--student-name", type=str, default=d.student_name)
    p.add_argument(
        "--teacher-name",
        type=str,
        default=d.teacher_name,
        help="4-bit teacher id (default: Qwen/Qwen3.5-4B)",
    )
    p.add_argument("--fff-depth", type=int, default=d.fff_depth)
    p.add_argument("--init-tau", type=float, default=d.init_tau)
    p.add_argument("--min-tau", type=float, default=d.min_tau)
    p.add_argument("--kl-temperature", type=float, default=d.kl_temperature)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--ce-coef", type=float, default=d.ce_coef)
    p.add_argument("--feature-coef", type=float, default=d.feature_coef)
    p.add_argument("--kl-topk", type=int, default=d.kl_topk)
    p.add_argument(
        "--max-context-length",
        type=int,
        default=d.max_context_length,
        help="Model/tokenizer context window (default 262144 = 256K)",
    )
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=d.grad_accum_steps)
    p.add_argument("--lr-leaf", type=float, default=d.lr_leaf)
    p.add_argument("--lr-router", type=float, default=d.lr_router)
    p.add_argument("--min-lr", type=float, default=d.min_lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument(
        "--max-samples-per-domain", type=int, default=d.max_samples_per_domain
    )
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Train embeddings/attention/LM head too (needs ≫12GB VRAM)",
    )
    p.add_argument(
        "--adam-fp32",
        action="store_true",
        help="Use full-precision AdamW instead of 8-bit (more VRAM)",
    )
    p.add_argument("--smoke-synthetic-data", action="store_true")
    return p


def _warn_vocab_mismatch(teacher_name: str, student_name: str) -> None:
    t = teacher_name.lower()
    s = student_name.lower()
    teacher_is_llama = any(x in t for x in ("llama", "hermes-3", "meta-llama"))
    student_is_qwen = "qwen" in s
    if teacher_is_llama and student_is_qwen:
        print(
            "WARNING: Teacher looks like Llama/Hermes while student is Qwen — "
            "tokenizers/vocabs differ, so Top-K KL indices will NOT align. "
            f"Prefer teacher `{DEFAULT_TEACHER}`."
        )


def main() -> None:
    args = build_argparser().parse_args()
    max_length = (
        int(args.block_size) if args.block_size is not None else int(args.max_length)
    )
    cfg = UltimateConfig(
        student_name=args.student_name,
        teacher_name=args.teacher_name,
        fff_depth=args.fff_depth,
        init_tau=args.init_tau,
        min_tau=args.min_tau,
        kl_temperature=args.kl_temperature,
        entropy_coef=args.entropy_coef,
        ce_coef=args.ce_coef,
        feature_coef=args.feature_coef,
        kl_topk=args.kl_topk,
        max_context_length=args.max_context_length,
        max_length=max_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr_leaf=args.lr_leaf,
        lr_router=args.lr_router,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        max_samples_per_domain=args.max_samples_per_domain,
        device=args.device,
        checkpoint=args.checkpoint,
        use_bf16=not args.fp32,
        teacher_4bit=not args.no_4bit,
        freeze_backbone=not args.unfreeze_backbone,
        adam_8bit=not args.adam_fp32,
        smoke_synthetic_data=args.smoke_synthetic_data,
    )

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    compute_dtype = resolve_compute_dtype(device, use_bf16=cfg.use_bf16)

    n_leaves = 1 << cfg.fff_depth
    h_uniform = math.log(float(n_leaves))
    print("=" * 72)
    print(
        "Ultimate FFF Distill — CoT · Agent · Code · Thai · English "
        "(STE + TopK-KL + Feature)"
    )
    print("=" * 72)
    print_device_info(device)
    print(f"config: {asdict(cfg)}")
    print(f"student_dtype={compute_dtype} | kl_topk={cfg.kl_topk}")
    print(f"target leaf entropy H_uniform=log({n_leaves})={h_uniform:.4f}")
    if "Qwen3.5-9B" in cfg.student_name or (
        "Qwen3.5-9B" in cfg.teacher_name and "Qwen3.5-4B" in cfg.student_name
    ):
        print(
            "VRAM tip: large teacher/student pairs may need ≥16–24GB. "
            "Default is teacher Qwen/Qwen3.5-4B + student Qwen/Qwen3.5-2B for ~12GB."
        )
    _warn_vocab_mismatch(cfg.teacher_name, cfg.student_name)

    AutoTokenizer = _require_tokenizer()
    print("\nLoading student tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_context_length(tokenizer=tokenizer, n_ctx=cfg.max_context_length)

    print("\nBuilding 20% multi-domain mixture (incl. CoT <think>) ...")
    domain_messages: dict[str, list[list[dict[str, str]]]] = {}
    per = max(1, int(cfg.max_samples_per_domain))
    if cfg.smoke_synthetic_data:
        per = min(per, 64)
    for domain in DOMAINS:
        domain_messages[domain] = load_domain_message_lists(
            domain,
            max_samples=per,
            smoke_synthetic=cfg.smoke_synthetic_data,
        )

    mixture = MultiDomainChatMixture(
        domain_messages,
        tokenizer,
        max_length=cfg.max_length,
        seed=cfg.seed,
    )
    loader = DataLoader(
        mixture,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=pin_memory_for(device),
    )
    print(
        f"mixture ready | domains={list(mixture.domains)} | "
        f"max_length={cfg.max_length} | equal 20% weights"
    )

    if cfg.teacher_4bit:
        teacher = load_4bit_teacher(cfg.teacher_name, device)
    else:
        teacher = load_dense_teacher(cfg.teacher_name, device)
        if compute_dtype != torch.float32:
            teacher = teacher.to(dtype=compute_dtype)

    student = build_fff_student(
        cfg.student_name,
        device,
        fff_depth=cfg.fff_depth,
        init_temp=cfg.init_tau,
        compute_dtype=compute_dtype,
        freeze_backbone=cfg.freeze_backbone,
    )
    apply_context_length(teacher, n_ctx=cfg.max_context_length)
    apply_context_length(student, tokenizer, n_ctx=cfg.max_context_length)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"context_length={model_context_length(student):,} "
        f"(train microbatch max_length={cfg.max_length})"
    )

    n_fff = sum(1 for _ in iter_fff_swiglu_blocks(student))
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in student.parameters() if not p.requires_grad)
    print(
        f"FFF SwiGLU blocks={n_fff} | student trainable={n_train:,} | "
        f"frozen={n_frozen:,}"
    )

    optimizer = build_student_optimizer(
        build_student_param_groups(
            student, lr_leaf=cfg.lr_leaf, lr_router=cfg.lr_router
        ),
        weight_decay=cfg.weight_decay,
        adam_8bit=cfg.adam_8bit,
    )
    n_opt_steps = max(
        (cfg.max_steps + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps, 1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_opt_steps,
        eta_min=cfg.min_lr,
    )

    ckpt_path = Path(cfg.checkpoint)
    pad_id = int(tokenizer.pad_token_id)
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running: dict[str, float] = {
        "kl": 0.0,
        "entropy": 0.0,
        "ce": 0.0,
        "feature": 0.0,
        "total": 0.0,
    }
    log_count = 0

    print(
        f"lr_leaf={cfg.lr_leaf:.2e} lr_router={cfg.lr_router:.2e} | "
        f"CosineAnnealingLR T_max={n_opt_steps} eta_min={cfg.min_lr:.2e}"
    )

    pbar = tqdm(range(cfg.max_steps), desc="distill-cot", dynamic_ncols=True)
    for step in pbar:
        tau = annealed_tau(step, cfg.max_steps, cfg.init_tau, cfg.min_tau)
        set_fff_temperature(student, tau)
        lr_leaf = float(optimizer.param_groups[0]["lr"])
        lr_router = float(optimizer.param_groups[-1]["lr"])

        batch = next(data_iter)
        input_ids = batch.to(device, non_blocking=True)

        vocab_s = model_vocab_size(student)
        vocab_t = model_vocab_size(teacher)
        input_ids = input_ids.clamp(0, min(vocab_s, vocab_t) - 1)
        want_feat = cfg.feature_coef > 0.0

        with bf16_autocast(device):
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    output_hidden_states=want_feat,
                    use_cache=False,
                )
                teacher_logits = teacher_out.logits
                teacher_hidden = (
                    teacher_out.hidden_states[-1] if want_feat else None
                )
                del teacher_out
                vocab_limit = min(vocab_s, int(teacher_logits.size(-1)))
                topk_val, topk_idx = extract_teacher_topk(
                    teacher_logits,
                    k=cfg.kl_topk,
                    vocab_limit=vocab_limit,
                )
                del teacher_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            student_out = student(
                input_ids=input_ids,
                output_hidden_states=want_feat,
                use_cache=False,
            )
            student_logits = student_out.logits
            student_hidden = student_out.hidden_states[-1] if want_feat else None
            del student_out
            routing = gather_student_leaf_probs(student)
            loss, parts = ultimate_distill_loss(
                student_logits,
                routing,
                teacher_topk_val=topk_val,
                teacher_topk_idx=topk_idx,
                student_hidden=student_hidden,
                teacher_hidden=teacher_hidden,
                kl_temperature=cfg.kl_temperature,
                entropy_coef=cfg.entropy_coef,
                ce_coef=cfg.ce_coef,
                feature_coef=cfg.feature_coef,
                kl_topk=cfg.kl_topk,
                labels=input_ids,
                pad_id=pad_id,
            )
            loss = loss / float(cfg.grad_accum_steps)

        loss.backward()
        parts = {k: v.detach() for k, v in parts.items()}
        del student_logits, loss, topk_val, topk_idx, routing
        if student_hidden is not None:
            del student_hidden
        if teacher_hidden is not None:
            del teacher_hidden

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in student.parameters() if p.requires_grad),
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elif device.type == "cuda" and (step + 1) % 4 == 0:
            torch.cuda.empty_cache()

        for k in running:
            running[k] += float(parts[k].item())
        log_count += 1

        if (step + 1) % cfg.log_every == 0 or step == 0:
            inv = 1.0 / max(log_count, 1)
            msg = (
                f"step={step + 1}/{cfg.max_steps} "
                f"loss={running['total'] * inv:.4f} "
                f"kl={running['kl'] * inv:.4f} "
                f"feat={running['feature'] * inv:.4f} "
                f"ce={running['ce'] * inv:.4f} "
                f"H={running['entropy'] * inv:.4f} "
                f"(H*={h_uniform:.2f}) "
                f"τ={tau:.3f} lr_leaf={lr_leaf:.2e} lr_router={lr_router:.2e}"
            )
            pbar.set_postfix_str(
                f"L={running['total'] * inv:.3f} H={running['entropy'] * inv:.2f}",
                refresh=False,
            )
            tqdm.write(msg)
            running = {k: 0.0 for k in running}
            log_count = 0

        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.max_steps:
            payload = {
                "step": step + 1,
                "config": asdict(cfg),
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "fff_depth": cfg.fff_depth,
                "model_name": cfg.student_name,
                "teacher_name": cfg.teacher_name,
                "architecture": "fff_swiglu_cot_agent",
                "domains": list(DOMAINS),
                "compute_dtype": str(compute_dtype),
                "max_context_length": int(cfg.max_context_length),
            }
            torch.save(payload, ckpt_path)
            tqdm.write(f"saved checkpoint → {ckpt_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60.0:.1f} min. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
