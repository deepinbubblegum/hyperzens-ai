#!/usr/bin/env python3
"""Shared HuggingFace helpers for FFF-SwiGLU agent training and chat.

Used by:
* ``train_fff_agent.py`` — multi-domain CoT / tool / code / Thai distill
* ``chat_fff_agent.py`` — interactive Triton Hard chat

Includes BF16 autocast, 4-bit teacher load, FFF student patch, Top-K KL,
temperature annealing, and checkpoint restore.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fff_swiglu import (
    get_text_config,
    iter_fff_swiglu_blocks,
    patch_model_with_fff_swiglu,
    set_fff_routing_mode,
    set_fff_temperature,
)
from models.fff_hard_triton import is_triton_available

# Qwen3.5 native context (256K tokens). Train micro-batches stay short;
# this only configures model / tokenizer capacity for long-context inference.
CONTEXT_LENGTH_256K: int = 262_144


def annealed_tau(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
) -> float:
    """Exponential FFF temperature ``τ: tau_start → tau_end`` over steps."""
    if total_steps <= 1:
        return tau_end
    t = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(tau_start * (tau_end / tau_start) ** t)


def encode_truncate_left(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    add_special_tokens: bool = False,
) -> list[int]:
    """Tokenize ``text`` keeping the **tail** (left-truncation), no length warnings.

    Long ChatML / tool traces often exceed a tokenizer's advertised
    ``model_max_length``. Calling bare ``tokenizer.encode`` then slicing still
    triggers HF's warning. Truncating inside the tokenizer call avoids that
    and keeps the assistant end.
    """
    max_length = max(int(max_length), 1)
    # Prefer the HF call API (supports truncation_side); fall back safely.
    try:
        enc = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
            truncation_side="left",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    except TypeError:
        prev = getattr(tokenizer, "truncation_side", "right")
        try:
            tokenizer.truncation_side = "left"
            return list(
                tokenizer.encode(
                    text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_length,
                )
            )
        finally:
            tokenizer.truncation_side = prev


def compute_leaf_entropy_loss(
    routing_probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Mean-leaf occupancy entropy ``H(p) = -Σ_ℓ p̄_ℓ log p̄_ℓ`` (nats)."""
    if routing_probs.ndim != 2:
        raise ValueError(
            f"routing_probs must be (N, L), got {tuple(routing_probs.shape)}"
        )
    p_bar = routing_probs.mean(dim=0).clamp(min=eps)
    p_bar = p_bar / p_bar.sum().clamp(min=eps)
    return -(p_bar * p_bar.log()).sum()


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install 'transformers>=4.57'"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_hf_causal_lm(
    model_name: str,
    *,
    dtype: torch.dtype = torch.float32,
    quantization_config: Any | None = None,
    device_map: Any | None = None,
) -> Any:
    """Load a HF CausalLM (Qwen2 / Qwen3.5 text or multimodal text trunk)."""
    AutoModelForCausalLM, _ = _require_transformers()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs.pop("dtype", None)
    if device_map is not None:
        kwargs["device_map"] = device_map

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as first_exc:  # noqa: BLE001
        # Qwen3.5 multimodal repos sometimes register under image-text loaders.
        try:
            from transformers import AutoModelForImageTextToText

            print(
                f"  AutoModelForCausalLM failed ({first_exc.__class__.__name__}); "
                "retrying AutoModelForImageTextToText ..."
            )
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except Exception as second_exc:  # noqa: BLE001
            raise SystemExit(
                f"Failed to load `{model_name}`.\n"
                f"  CausalLM error: {first_exc}\n"
                f"  ImageText error: {second_exc}\n"
                "Tip: pip install -U 'transformers>=4.57' (required for Qwen3.5)."
            ) from second_exc


@contextmanager
def bf16_autocast(device: torch.device) -> Iterator[None]:
    """CUDA autocast in bfloat16 (Ampere+); no-op elsewhere."""
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    with nullcontext():
        yield


def resolve_compute_dtype(device: torch.device, *, use_bf16: bool) -> torch.dtype:
    """Pick parameter dtype: BF16 on CUDA when supported, else FP32."""
    if use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def build_student_param_groups(
    student: Any,
    *,
    lr_leaf: float,
    lr_router: float,
) -> list[dict[str, Any]]:
    """AdamW groups: SwiGLU leaf (+scale) / router / remaining trunk."""
    leaf_params: list[nn.Parameter] = []
    router_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for block in iter_fff_swiglu_blocks(student):
        fff = block.fff
        for p in (fff.w_gate_leaf, fff.w_up_leaf, fff.w_down_leaf):
            if p.requires_grad and id(p) not in seen:
                leaf_params.append(p)
                seen.add(id(p))
        if isinstance(block.output_scale, nn.Parameter) and block.output_scale.requires_grad:
            if id(block.output_scale) not in seen:
                leaf_params.append(block.output_scale)
                seen.add(id(block.output_scale))
        for p in (fff.router_weights, fff.router_biases):
            if p.requires_grad and id(p) not in seen:
                router_params.append(p)
                seen.add(id(p))

    other_params: list[nn.Parameter] = []
    for p in student.parameters():
        if p.requires_grad and id(p) not in seen:
            other_params.append(p)
            seen.add(id(p))

    groups: list[dict[str, Any]] = []
    if leaf_params:
        groups.append({"params": leaf_params, "lr": float(lr_leaf)})
    if other_params:
        groups.append({"params": other_params, "lr": float(lr_leaf)})
    if router_params:
        groups.append({"params": router_params, "lr": float(lr_router)})
    if not groups:
        raise RuntimeError("no trainable student parameters")
    return groups


def load_4bit_teacher(model_name: str, device: torch.device) -> Any:
    """Load CausalLM teacher in bitsandbytes 4-bit (BF16 compute)."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "BitsAndBytesConfig missing — upgrade transformers"
        ) from exc

    if device.type != "cuda":
        raise SystemExit(
            "4-bit teacher requires CUDA. Use --device cuda "
            "(or pass --no-4bit with a smaller dense teacher)."
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
    teacher = load_hf_causal_lm(
        model_name,
        quantization_config=bnb_cfg,
        device_map={"": device.index if device.index is not None else 0},
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_dense_teacher(model_name: str, device: torch.device) -> Any:
    """FP32 dense teacher (caller may cast dtype)."""
    print(f"Loading dense teacher `{model_name}` (float32) ...")
    teacher = load_hf_causal_lm(model_name, dtype=torch.float32)
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
    print(f"Loading student `{student_name}` (float32 for FFF init) ...")
    student = load_hf_causal_lm(student_name, dtype=torch.float32)
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
    """Select teacher Top-K logits / indices; drop the full ``V`` tensor ASAP."""
    v = int(teacher_logits.size(-1) if vocab_limit is None else vocab_limit)
    v = min(v, int(teacher_logits.size(-1)))
    k_eff = min(int(k), v)
    topk_val, topk_idx = torch.topk(
        teacher_logits[..., :v].float(), k=k_eff, dim=-1
    )
    return topk_val, topk_idx


def topk_kl_distill_loss(
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
    """``TopK-KL·T² − λ_H · H(leaf) + λ_ce · CE`` (VRAM-safe for large vocab)."""
    t = max(float(kl_temperature), 1e-6)
    v_s = int(student_logits.size(-1))

    if teacher_topk_val is None or teacher_topk_idx is None:
        if teacher_logits is None:
            raise ValueError(
                "topk_kl_distill_loss requires teacher_logits or teacher_topk_*"
            )
        v = min(v_s, int(teacher_logits.size(-1)))
        topk_val, topk_idx = extract_teacher_topk(
            teacher_logits, k=kl_topk, vocab_limit=v
        )
    else:
        topk_val = teacher_topk_val
        topk_idx = teacher_topk_idx

    student_topk_logits = torch.gather(
        student_logits.float(), dim=-1, index=topk_idx
    )
    p_teacher = F.softmax(topk_val / t, dim=-1)
    log_q_student = F.log_softmax(student_topk_logits / t, dim=-1)
    loss_kl = F.kl_div(
        log_q_student.reshape(-1, log_q_student.size(-1)),
        p_teacher.reshape(-1, p_teacher.size(-1)),
        reduction="batchmean",
    ) * (t**2)

    loss_entropy = compute_leaf_entropy_loss(routing_probs.float())
    total = loss_kl - float(entropy_coef) * loss_entropy

    loss_ce = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    if labels is not None and ce_coef > 0.0:
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


# Back-compat alias used by older call sites.
thai_distill_loss = topk_kl_distill_loss


def load_student_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_name: str | None = None,
    routing_mode: str = "triton",
    default_model: str = "Qwen/Qwen3.5-4B",
    max_context_length: int = CONTEXT_LENGTH_256K,
) -> tuple[Any, dict[str, Any]]:
    """Scaffold HF CausalLM → inject FFF SwiGLU → load ``student_state_dict``."""
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = model_name or str(ckpt.get("model_name", default_model))
    fff_depth = int(ckpt.get("fff_depth", 4))
    cfg = ckpt.get("config") or {}
    init_tau = float(cfg.get("init_tau", 1.0))
    n_ctx = int(ckpt.get("max_context_length", max_context_length))

    print(f"Loading student scaffold `{name}` (float32 init) ...")
    student = load_hf_causal_lm(name, dtype=torch.float32)
    apply_context_length(student, n_ctx=n_ctx)
    print(f"  context_length={model_context_length(student):,}")
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


def model_vocab_size(model: Any) -> int:
    """Resolve vocab size from top-level or nested ``text_config``."""
    cfg = model.config
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is None:
            continue
        v = getattr(obj, "vocab_size", None)
        if v is not None:
            return int(v)
    raise AttributeError("model config has no vocab_size")


def model_context_length(model: Any) -> int:
    """Resolve ``max_position_embeddings`` from top-level or nested ``text_config``.

    Returns
    -------
    int
        Context window in tokens (Qwen3.5 default is ``262144`` / 256K).
    """
    text_cfg = get_text_config(model)
    for obj in (text_cfg, model.config):
        v = getattr(obj, "max_position_embeddings", None)
        if v is not None:
            return int(v)
    return CONTEXT_LENGTH_256K


def apply_context_length(
    model: Any | None = None,
    tokenizer: Any | None = None,
    *,
    n_ctx: int = CONTEXT_LENGTH_256K,
) -> int:
    """Force model (+ optional tokenizer) context window to ``n_ctx`` tokens.

    Writes ``max_position_embeddings`` on both the outer config and nested
    ``text_config`` (Qwen3.5 multimodal wrappers). Sets
    ``tokenizer.model_max_length`` when a tokenizer is provided.

    Parameters
    ----------
    model:
        Optional HF CausalLM / image-text model with ``.config``.
    tokenizer:
        Optional HF tokenizer to align ``model_max_length``.
    n_ctx:
        Target context length (default ``262144`` = 256K).

    Returns
    -------
    int
        The applied context length.
    """
    n_ctx = max(int(n_ctx), 1)
    if model is not None:
        cfg = model.config
        cfg.max_position_embeddings = n_ctx
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            text_cfg.max_position_embeddings = n_ctx
        # Some Qwen builds also expose generation / sliding-window caps.
        for attr in ("max_sequence_length", "seq_length"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, n_ctx)
            if text_cfg is not None and hasattr(text_cfg, attr):
                setattr(text_cfg, attr, n_ctx)

    if tokenizer is not None:
        tokenizer.model_max_length = n_ctx
        # Prefer left truncation for long chat prompts (keep recent turns).
        if hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "left"
    return n_ctx
