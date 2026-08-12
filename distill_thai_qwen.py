#!/usr/bin/env python3
"""Thai instruction distillation: Qwen2.5-7B-Instruct (4-bit) → FFF Student.

Teacher
-------
Frozen ``Qwen/Qwen2.5-7B-Instruct`` loaded with ``bitsandbytes`` 4-bit
quantization (``bnb_4bit_compute_dtype=bfloat16``) so it fits ~12GB VRAM.

Student
-------
``Qwen/Qwen2.5-0.5B-Instruct`` (or ``1.5B``) with every ``mlp`` replaced by
:class:`FFFSwiGLUBlock` (depth=4 → 16 leaves). Smart-init from the student's
own SwiGLU MLP in **float32**, then cast to BF16. Training uses **STE**
hard-aware soft routing.

Data
----
Thai instruction/chat pairs (default ``airesearch/WangchanThaiInstruct``)
formatted with Qwen's official chat template::

    <|im_start|>user\\n{prompt}<|im_end|>\\n
    <|im_start|>assistant\\n{response}<|im_end|>

Loss / schedule
---------------
``L = TopK-KL(student ‖ teacher)·T² − λ_H · H(leaf)``
(+ optional small CE on labels). KL is computed over the teacher's
**Top-200** vocab candidates only (avoids OOM on Qwen's 151k vocab).
Feature MSE is **off by default** (7B vs 0.5B size mismatch).

    AdamW  lr_leaf=2e-4  lr_router=5e-5
    CosineAnnealingLR  max_steps=3000
    max_length=64  batch_size=1  grad_accum_steps=16
    + student gradient checkpointing

Checkpoint
----------
``qwen_thai_fff.pt``

Example
-------
    python distill_thai_qwen.py --device cuda --max-steps 3000
    python distill_thai_qwen.py --student-name Qwen/Qwen2.5-1.5B-Instruct
    python distill_thai_qwen.py --max-steps 20 --smoke-synthetic-data
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from device_utils import (
    apply_hardware_optimizations,
    pin_memory_for,
    print_device_info,
    resolve_device,
)
from distill_modern_llm import (
    bf16_autocast,
    build_student_param_groups,
    resolve_compute_dtype,
)
from fff_distill import annealed_tau, compute_leaf_entropy_loss
from fff_modern_llm import (
    iter_fff_swiglu_blocks,
    patch_model_with_fff_swiglu,
    set_fff_routing_mode,
    set_fff_temperature,
)

CHECKPOINT_NAME = "qwen_thai_fff.pt"
DEFAULT_STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TEACHER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = "airesearch/WangchanThaiInstruct"

# Fallback HF datasets tried in order when the primary id fails.
DATASET_FALLBACKS: tuple[str, ...] = (
    "airesearch/WangchanThaiInstruct",
    "Thaweewat/alpaca-cleaned-52k-th",
    "pythainlp/han-instruct-dataset-v4.0",
)


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Qwen chat template + Thai dataset
# ---------------------------------------------------------------------------


def format_qwen_chat(user: str, assistant: str) -> str:
    """Official Qwen ChatML turn (user → assistant)."""
    user = user.strip()
    assistant = assistant.strip()
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def _row_to_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    """Map heterogeneous Thai instruct schemas → ``(user, assistant)``."""
    # WangchanThaiInstruct
    if "Instruction" in row and "Output" in row:
        instr = str(row.get("Instruction") or "").strip()
        inp = str(row.get("Input") or "").strip()
        out = str(row.get("Output") or "").strip()
        if not out:
            return None
        user = f"{instr}\n{inp}".strip() if inp else instr
        return (user, out) if user else None

    # Alpaca-style (Thai)
    if "instruction" in row and "output" in row:
        instr = str(row.get("instruction") or "").strip()
        inp = str(row.get("input") or "").strip()
        out = str(row.get("output") or "").strip()
        if not out:
            return None
        user = f"{instr}\n{inp}".strip() if inp else instr
        return (user, out) if user else None

    # han-instruct / conversations list
    conv = row.get("conversations") or row.get("messages")
    if isinstance(conv, list) and len(conv) >= 2:
        user_txt = ""
        asst_txt = ""
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("from") or "").lower()
            content = str(turn.get("content") or turn.get("value") or "").strip()
            if role in ("user", "human") and content:
                user_txt = content
            elif role in ("assistant", "gpt", "bot") and content:
                asst_txt = content
        if user_txt and asst_txt:
            return user_txt, asst_txt

    # Generic prompt/response
    for uk, ak in (
        ("prompt", "response"),
        ("question", "answer"),
        ("input", "output"),
        ("query", "answer"),
    ):
        if uk in row and ak in row:
            u = str(row[uk] or "").strip()
            a = str(row[ak] or "").strip()
            if u and a:
                return u, a
    return None


def _synthetic_thai_pairs(n: int = 64) -> list[tuple[str, str]]:
    """Tiny offline Thai pairs for smoke tests (no HF download)."""
    seeds = [
        ("สวัสดีครับ แนะนำตัวหน่อยได้ไหม", "สวัสดีครับ ผมเป็นผู้ช่วย AI ที่พูดภาษาไทยได้ครับ มีอะไรให้ช่วยไหมครับ"),
        ("ประเทศไทยมีกี่จังหวัด", "ประเทศไทยมีทั้งหมด 77 จังหวัดครับ"),
        ("ช่วยแปลคำว่า hello เป็นภาษาไทย", "คำว่า hello แปลว่า สวัสดี ครับ"),
        ("เขียนบทกวีสั้นๆ เกี่ยวกับฝน", "ฝนโปรยปรายลงมา\nชุ่มฉ่ำผืนแผ่นดินไทย\nต้นไม้ใบเขียวชอุ่มใจ\nธรรมชาติงดงามยิ่งนัก"),
        ("1+1 เท่ากับเท่าไหร่", "1+1 เท่ากับ 2 ครับ"),
        ("อธิบายว่า AI คืออะไรแบบสั้นๆ", "AI หรือปัญญาประดิษฐ์ คือระบบคอมพิวเตอร์ที่เรียนรู้และทำงานคล้ายมนุษย์ได้ครับ"),
    ]
    out: list[tuple[str, str]] = []
    while len(out) < n:
        out.extend(seeds)
    return out[:n]


def load_thai_instruct_pairs(
    dataset_name: str,
    *,
    max_samples: int | None = 20_000,
    smoke_synthetic: bool = False,
) -> list[tuple[str, str]]:
    """Load ``(user, assistant)`` pairs from HF Thai instruct datasets."""
    if smoke_synthetic:
        n = int(max_samples) if max_samples is not None else 64
        pairs = _synthetic_thai_pairs(max(n, 1))
        print(f"Using synthetic Thai pairs: {len(pairs)}")
        return pairs

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required: pip install datasets\n"
            "Or pass --smoke-synthetic-data for offline smoke."
        ) from exc

    candidates = [dataset_name] + [
        d for d in DATASET_FALLBACKS if d != dataset_name
    ]
    last_err: Exception | None = None
    for name in candidates:
        try:
            print(f"Loading Thai instruct dataset `{name}` ...")
            # Prefer train split; Wangchan has train/test.
            try:
                ds = load_dataset(name, split="train")
            except Exception:
                ds = load_dataset(name)
                if hasattr(ds, "keys"):
                    split = "train" if "train" in ds else next(iter(ds.keys()))
                    ds = ds[split]

            pairs: list[tuple[str, str]] = []
            for row in ds:
                pair = _row_to_pair(dict(row))
                if pair is not None:
                    pairs.append(pair)
                if max_samples is not None and len(pairs) >= max_samples:
                    break
            if not pairs:
                raise RuntimeError(f"no usable rows in `{name}`")
            print(f"  loaded {len(pairs):,} instruction pairs from `{name}`")
            return pairs
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  failed `{name}`: {exc.__class__.__name__}: {exc}")

    print("All HF datasets failed — falling back to synthetic Thai pairs.")
    if last_err is not None:
        print(f"  last error: {last_err}")
    n = int(max_samples) if max_samples is not None else 64
    return _synthetic_thai_pairs(max(n, 1))


class ThaiChatDataset(Dataset[Tensor]):
    """Tokenized Qwen-chat Thai instruction sequences (fixed ``max_length``).

    Each item is ``input_ids`` of shape ``(max_length,)`` (right-padded with
    ``pad_id``). Shorter examples are padded; longer ones truncated from the
    left so the assistant completion is retained when possible.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Any,
        *,
        max_length: int = 512,
    ) -> None:
        self.max_length = int(max_length)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.pad_id = int(pad_id)

        self.examples: list[Tensor] = []
        for user, assistant in pairs:
            text = format_qwen_chat(user, assistant)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            if len(ids) > self.max_length:
                ids = ids[-self.max_length :]
            t = torch.full((self.max_length,), self.pad_id, dtype=torch.long)
            t[: len(ids)] = torch.tensor(ids, dtype=torch.long)
            self.examples.append(t)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Teacher (4-bit) / Student (FFF) construction
# ---------------------------------------------------------------------------


def load_4bit_teacher(model_name: str, device: torch.device) -> Any:
    """Load Qwen Instruct teacher in bitsandbytes 4-bit (BF16 compute)."""
    AutoModelForCausalLM, _ = _require_transformers()
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "BitsAndBytesConfig missing — upgrade transformers"
        ) from exc

    if device.type != "cuda":
        raise SystemExit(
            "4-bit teacher requires CUDA. Use --device cuda "
            "(or pass --teacher-name matching student size without 4-bit)."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "bitsandbytes is required for 4-bit teacher:\n"
            "  pip install bitsandbytes"
        ) from exc

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(
        f"Loading 4-bit teacher `{model_name}` "
        f"(bnb_4bit_compute_dtype=bfloat16) ..."
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map={"": device.index if device.index is not None else 0},
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_dense_teacher(model_name: str, device: torch.device) -> Any:
    """FP32→BF16 dense teacher (smoke / same-size fallbacks)."""
    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading dense teacher `{model_name}` (float32) ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.to(device)


def build_fff_student(
    student_name: str,
    device: torch.device,
    *,
    fff_depth: int = 4,
    init_temp: float = 1.0,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load Instruct student, inject FFF SwiGLU (FP32 init), cast dtype."""
    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading student `{student_name}` (float32 for FFF init) ...")
    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    for p in student.parameters():
        p.requires_grad_(True)

    print("Injecting FFF SwiGLU (smart-init FP32) ...")
    n = patch_model_with_fff_swiglu(
        student,
        fff_depth=fff_depth,
        init_temp=init_temp,
    )
    scale0 = math.sqrt(float(1 << fff_depth))
    print(
        f"  patched {n} MLPs | STE soft train | "
        f"output_scale init=√L={scale0:.4f}"
    )

    if compute_dtype != torch.float32:
        print(f"Casting student → {compute_dtype} ...")
        student = student.to(dtype=compute_dtype)
    student = student.to(device)
    student.train()
    set_fff_routing_mode(student, "soft")

    # Recompute activations on backward to cut VRAM (critical with large vocab).
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "config"):
            student.config.use_cache = False
        print("  gradient_checkpointing=ON (use_cache=False)")
    return student


def gather_student_leaf_probs(student: Any) -> Tensor:
    """Concatenate cached leaf probs from all FFF blocks → ``(N, L)``."""
    chunks: list[Tensor] = []
    for block in iter_fff_swiglu_blocks(student):
        p = block.leaf_probs()
        if p is not None:
            chunks.append(p)
    if not chunks:
        raise RuntimeError("no leaf probs cached — run a soft student forward first")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def extract_teacher_topk(
    teacher_logits: Tensor,
    *,
    k: int = 200,
    vocab_limit: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Select teacher Top-K logits / indices; drop the full ``V`` tensor ASAP.

    Parameters
    ----------
    teacher_logits:
        ``(B, T, V_t)``
    k:
        Number of teacher candidates (default 200).
    vocab_limit:
        Clip vocab to ``min(V_s, V_t)`` so indices are valid for the student.

    Returns
    -------
    topk_val, topk_idx:
        Both ``(B, T, K)`` in float32 / int64.
    """
    v = int(teacher_logits.size(-1) if vocab_limit is None else vocab_limit)
    v = min(v, int(teacher_logits.size(-1)))
    k_eff = min(int(k), v)
    topk_val, topk_idx = torch.topk(
        teacher_logits[..., :v].float(), k=k_eff, dim=-1
    )
    return topk_val, topk_idx


def thai_distill_loss(
    student_logits: Tensor,
    teacher_logits: Tensor | None,
    routing_probs: Tensor,
    *,
    kl_temperature: float = 2.0,
    entropy_coef: float = 0.01,
    ce_coef: float = 0.1,
    kl_topk: int = 200,
    labels: Tensor | None = None,
    pad_id: int = 0,
    teacher_topk_val: Tensor | None = None,
    teacher_topk_idx: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """``TopK-KL·T² − λ_H · H(leaf) + λ_ce · CE`` (VRAM-safe for large vocab).

    Instead of softmax / KL over the full Qwen vocabulary (~151,646), select
    the teacher's Top-``K`` logits and compute KL only on those candidates::

        topk_val, topk_idx = topk(teacher, K)
        student_topk = gather(student, topk_idx)
        KL(softmax(topk_val/τ) ‖ log_softmax(student_topk/τ)) · τ²

    Pass precomputed ``teacher_topk_val`` / ``teacher_topk_idx`` (recommended)
    so the full teacher logit tensor can be freed before the student forward.
    """
    t = max(float(kl_temperature), 1e-6)
    v_s = int(student_logits.size(-1))

    if teacher_topk_val is None or teacher_topk_idx is None:
        if teacher_logits is None:
            raise ValueError(
                "thai_distill_loss requires teacher_logits or teacher_topk_*"
            )
        v = min(v_s, int(teacher_logits.size(-1)))
        topk_val, topk_idx = extract_teacher_topk(
            teacher_logits, k=kl_topk, vocab_limit=v
        )
    else:
        topk_val = teacher_topk_val
        topk_idx = teacher_topk_idx

    # Gather student logits at teacher Top-K indices → (B, T, K)
    student_topk_logits = torch.gather(
        student_logits.float(), dim=-1, index=topk_idx
    )
    p_teacher = F.softmax(topk_val / t, dim=-1)
    log_q_student = F.log_softmax(student_topk_logits / t, dim=-1)
    # Flatten to (N, K) for batchmean KL (matches nn.KLDivLoss batchmean).
    loss_kl = F.kl_div(
        log_q_student.reshape(-1, log_q_student.size(-1)),
        p_teacher.reshape(-1, p_teacher.size(-1)),
        reduction="batchmean",
    ) * (t**2)

    loss_entropy = compute_leaf_entropy_loss(routing_probs.float())
    total = loss_kl - float(entropy_coef) * loss_entropy

    loss_ce = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    if labels is not None and ce_coef > 0.0:
        # Standard causal LM shift.
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.float().reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=int(pad_id),
        )
        total = total + float(ce_coef) * loss_ce

    return total, {
        "kl": loss_kl,
        "entropy": loss_entropy,
        "ce": loss_ce,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    student_name: str = DEFAULT_STUDENT
    teacher_name: str = DEFAULT_TEACHER
    dataset_name: str = DEFAULT_DATASET
    fff_depth: int = 4
    init_tau: float = 1.0
    min_tau: float = 0.10
    kl_temperature: float = 2.0
    entropy_coef: float = 0.01
    ce_coef: float = 0.1
    kl_topk: int = 200
    max_length: int = 64
    batch_size: int = 1
    grad_accum_steps: int = 16
    lr_leaf: float = 2e-4
    lr_router: float = 5e-5
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 3000
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    max_samples: int = 20_000
    device: str = "cuda"
    checkpoint: str = CHECKPOINT_NAME
    use_bf16: bool = True
    teacher_4bit: bool = True
    smoke_synthetic_data: bool = False


def build_argparser() -> argparse.ArgumentParser:
    d = DistillConfig()
    p = argparse.ArgumentParser(
        description="Thai Qwen Instruct → FFF-SwiGLU distillation (4-bit teacher)"
    )
    p.add_argument("--student-name", type=str, default=d.student_name)
    p.add_argument("--teacher-name", type=str, default=d.teacher_name)
    p.add_argument("--dataset-name", type=str, default=d.dataset_name)
    p.add_argument("--fff-depth", type=int, default=d.fff_depth)
    p.add_argument("--init-tau", type=float, default=d.init_tau)
    p.add_argument("--min-tau", type=float, default=d.min_tau)
    p.add_argument("--kl-temperature", type=float, default=d.kl_temperature)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--ce-coef", type=float, default=d.ce_coef)
    p.add_argument(
        "--kl-topk",
        type=int,
        default=d.kl_topk,
        help="Teacher Top-K vocab size for KL (default 200; avoids full-V OOM)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=d.max_length,
        help="Sequence block size (default 64 for 12GB VRAM)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Alias for --max-length",
    )
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
    p.add_argument("--max-samples", type=int, default=d.max_samples)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument("--fp32", action="store_true", help="Disable BF16 student")
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load dense teacher (needs large VRAM)",
    )
    p.add_argument(
        "--smoke-synthetic-data",
        action="store_true",
        help="Use built-in Thai pairs (skip HF dataset download)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    max_length = int(args.block_size) if args.block_size is not None else int(args.max_length)
    cfg = DistillConfig(
        student_name=args.student_name,
        teacher_name=args.teacher_name,
        dataset_name=args.dataset_name,
        fff_depth=args.fff_depth,
        init_tau=args.init_tau,
        min_tau=args.min_tau,
        kl_temperature=args.kl_temperature,
        entropy_coef=args.entropy_coef,
        ce_coef=args.ce_coef,
        kl_topk=args.kl_topk,
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
        max_samples=args.max_samples,
        device=args.device,
        checkpoint=args.checkpoint,
        use_bf16=not args.fp32,
        teacher_4bit=not args.no_4bit,
        smoke_synthetic_data=args.smoke_synthetic_data,
    )

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    compute_dtype = resolve_compute_dtype(device, use_bf16=cfg.use_bf16)

    n_leaves = 1 << cfg.fff_depth
    h_uniform = math.log(float(n_leaves))
    print("=" * 72)
    print("Thai Distill — 4-bit Qwen Teacher → FFF-SwiGLU Student (STE)")
    print("=" * 72)
    print_device_info(device)
    print(f"config: {asdict(cfg)}")
    print(f"student_dtype={compute_dtype} (no GradScaler)")
    print(f"target leaf entropy H_uniform=log({n_leaves})={h_uniform:.4f}")

    _, AutoTokenizer = _require_transformers()
    print("\nLoading student tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_thai_instruct_pairs(
        cfg.dataset_name,
        max_samples=cfg.max_samples,
        smoke_synthetic=cfg.smoke_synthetic_data,
    )
    dataset = ThaiChatDataset(pairs, tokenizer, max_length=cfg.max_length)
    if len(dataset) == 0:
        raise SystemExit("empty Thai dataset after tokenization")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory_for(device),
    )
    print(f"train examples={len(dataset)} max_length={cfg.max_length}")

    # Teacher first (4-bit occupies GPU); then student BF16.
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
    )

    n_fff = sum(1 for _ in iter_fff_swiglu_blocks(student))
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"FFF SwiGLU blocks={n_fff} | student trainable={n_train:,}")

    optimizer = torch.optim.AdamW(
        build_student_param_groups(
            student, lr_leaf=cfg.lr_leaf, lr_router=cfg.lr_router
        ),
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
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
        "total": 0.0,
    }
    log_count = 0

    print(
        f"optimizer: AdamW leaf/other lr={cfg.lr_leaf:.2e} "
        f"router lr={cfg.lr_router:.2e} | "
        f"CosineAnnealingLR T_max={n_opt_steps} eta_min={cfg.min_lr:.2e}"
    )

    pbar = tqdm(range(cfg.max_steps), desc="distill-thai", dynamic_ncols=True)
    for step in pbar:
        tau = annealed_tau(step, cfg.max_steps, cfg.init_tau, cfg.min_tau)
        set_fff_temperature(student, tau)
        lr_leaf = float(optimizer.param_groups[0]["lr"])
        lr_router = float(optimizer.param_groups[-1]["lr"])

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        input_ids = batch.to(device, non_blocking=True)

        # Clamp into both vocabs safely.
        vocab_s = int(student.config.vocab_size)
        vocab_t = int(getattr(teacher.config, "vocab_size", vocab_s))
        input_ids = input_ids.clamp(0, min(vocab_s, vocab_t) - 1)

        with bf16_autocast(device):
            # Teacher → Top-K only; free full (B,T,151k) logits before student.
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids)
                teacher_logits = teacher_out.logits
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

            student_logits = student(input_ids=input_ids).logits
            routing = gather_student_leaf_probs(student)
            loss, parts = thai_distill_loss(
                student_logits,
                None,
                routing,
                kl_temperature=cfg.kl_temperature,
                entropy_coef=cfg.entropy_coef,
                ce_coef=cfg.ce_coef,
                kl_topk=cfg.kl_topk,
                labels=input_ids,
                pad_id=pad_id,
                teacher_topk_val=topk_val,
                teacher_topk_idx=topk_idx,
            )
            loss = loss / float(cfg.grad_accum_steps)

        loss.backward()
        # Drop graph-held tensors before the next micro-batch.
        parts = {k: v.detach() for k, v in parts.items()}
        del student_logits, loss, topk_val, topk_idx, routing

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        elif device.type == "cuda" and (step + 1) % 4 == 0:
            # Periodic hygiene during accumulation.
            torch.cuda.empty_cache()

        for k in running:
            running[k] += float(parts[k].detach().item())
        log_count += 1

        if (step + 1) % cfg.log_every == 0 or step == 0:
            inv = 1.0 / max(log_count, 1)
            msg = (
                f"step={step + 1}/{cfg.max_steps} "
                f"loss={running['total'] * inv:.4f} "
                f"kl={running['kl'] * inv:.4f} "
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
                "architecture": "fff_swiglu_qwen_thai",
                "compute_dtype": str(compute_dtype),
            }
            torch.save(payload, ckpt_path)
            tqdm.write(f"saved checkpoint → {ckpt_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60.0:.1f} min. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
