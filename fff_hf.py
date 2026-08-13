#!/usr/bin/env python3
"""Shared HuggingFace helpers for FFF-SwiGLU agent training and chat.

Used by:
* ``train_fff_agent.py`` — multi-domain CoT / tool / code / Thai distill
* ``chat_fff_agent.py`` — interactive Triton Hard chat

Includes BF16 autocast, 4-bit teacher load, FFF student patch, Top-K KL,
temperature annealing, and checkpoint restore.
"""

from __future__ import annotations

import gc
import math
import sys
from contextlib import AbstractContextManager, contextmanager, nullcontext
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
    adapt_fff_swiglu_dims,
)
from models.fff_hard_triton import is_triton_available

# Qwen3.5 native context (256K tokens). Train micro-batches stay short;
# this only configures model / tokenizer capacity for long-context inference.
CONTEXT_LENGTH_256K: int = 262_144
# Sequence-axis chunk for teacher Top-K so ``(B, T, V)`` never becomes FP32.
TEACHER_TOPK_SEQ_CHUNK: int = 256


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
    max_memory: dict[Any, str] | None = None,
    extra_from_pretrained: dict[str, Any] | None = None,
) -> Any:
    """Load a HF CausalLM or Qwen3.5 / Qwen3.6 image-text Instruct checkpoint.

    Requests ``attn_implementation='sdpa'`` so both teacher and student use
    ``F.scaled_dot_product_attention`` (Flash / mem-efficient on CUDA, fast
    kernels on MPS). Falls back to the model's default attention if SDPA is
    rejected.

    ``max_memory`` is forwarded to Accelerate (e.g. ``{0: "10GiB", "cpu": "48GiB"}``)
    so a 4-bit 35B teacher can cap GPU VRAM and offload remaining layers to CPU RAM.
    """
    AutoModelForCausalLM, _ = _require_transformers()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs.pop("dtype", None)
    if device_map is not None:
        kwargs["device_map"] = device_map
        kwargs.setdefault("low_cpu_mem_usage", True)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if extra_from_pretrained:
        kwargs.update(extra_from_pretrained)

    # Qwen3.5 / Qwen3.6 Instruct checkpoints are image-text wrappers
    # (``pipeline_tag: image-text-to-text``), not plain CausalLM.
    name_l = model_name.lower()
    prefer_image_text = any(tag in name_l for tag in ("qwen3.5", "qwen3.6", "qwen3_5", "qwen3_6"))
    loaders: list[tuple[str, Any]] = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    if prefer_image_text:
        try:
            from transformers import AutoModelForImageTextToText

            loaders = [
                ("AutoModelForImageTextToText", AutoModelForImageTextToText),
                ("AutoModelForCausalLM", AutoModelForCausalLM),
            ]
        except ImportError:
            pass

    errors: list[str] = []
    model: Any | None = None
    for label, loader in loaders:
        try:
            model = _from_pretrained_prefer_sdpa(loader, model_name, kwargs)
            if label != "AutoModelForCausalLM":
                print(f"  loaded `{model_name}` via {label}")
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            print(f"  {label} failed ({exc.__class__.__name__}); trying next loader ...")
    if model is None:
        detail = "\n  ".join(errors) if errors else "no loader attempted"
        raise SystemExit(
            f"Failed to load `{model_name}`.\n  {detail}\n"
            "Tip: pip install -U 'transformers>=4.57' (required for Qwen3.5 / Qwen3.6)."
        )
    enable_model_sdpa(model)
    return model


def _from_pretrained_prefer_sdpa(loader: Any, model_name: str, kwargs: dict[str, Any]) -> Any:
    """``from_pretrained`` with SDPA, then retry without if the loader rejects it."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return loader.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        if kwargs.get("attn_implementation") != "sdpa":
            raise
        print("  SDPA attn_implementation rejected; retrying default attention ...")
        fallback = {k: v for k, v in kwargs.items() if k != "attn_implementation"}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loader.from_pretrained(model_name, **fallback)


def enable_fast_sdpa() -> None:
    """Turn on PyTorch SDPA backends (Flash / mem-efficient / math).

    CUDA uses FlashAttention-capable kernels when hardware allows. MPS uses
    the SDPA fast path. Safe no-ops when a backend flag is missing.
    """
    cuda_sdp = getattr(torch.backends, "cuda", None)
    if cuda_sdp is None:
        return
    for name, enabled in (
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", True),
    ):
        fn = getattr(cuda_sdp, name, None)
        if callable(fn):
            try:
                fn(enabled)
            except Exception:
                pass


def enable_model_sdpa(model: Any) -> None:
    """Force HuggingFace attention to SDPA on ``model`` (and nested text config)."""
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter("sdpa")
        except Exception:
            pass
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is None:
            continue
        if hasattr(obj, "_attn_implementation"):
            try:
                obj._attn_implementation = "sdpa"
            except Exception:
                pass


def attn_implementation_name(model: Any) -> str:
    """Best-effort attention backend label for logs."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is None:
            continue
        v = getattr(obj, "_attn_implementation", None) or getattr(
            obj, "attn_implementation", None
        )
        if v:
            return str(v)
    return "unknown"


@contextmanager
def bf16_autocast(device: torch.device) -> Iterator[None]:
    """Mixed-precision autocast: CUDA BF16, MPS BF16/FP16, otherwise no-op.

    Uses a single ``yield`` plus ``try/finally`` so an exception inside the
    training step cannot trigger ``generator didn't stop after throw()``.
    """
    cm: AbstractContextManager[Any]
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        cm = torch.amp.autocast(device_type="cuda", dtype=dtype)
    elif device.type == "mps":
        try:
            cm = torch.amp.autocast(
                device_type="mps", dtype=_mps_half_dtype()
            )
        except Exception:
            cm = nullcontext()
    else:
        cm = nullcontext()

    entered = False
    exc: tuple[Any, Any, Any] = (None, None, None)
    try:
        cm.__enter__()
        entered = True
        yield
    except Exception:
        exc = sys.exc_info()
        raise
    finally:
        if entered:
            cm.__exit__(*exc)


def _mps_half_dtype() -> torch.dtype:
    """Prefer bfloat16 on Apple Silicon; fall back to float16 if unsupported."""
    try:
        x = torch.zeros(2, device="mps", dtype=torch.bfloat16)
        _ = x * x
        return torch.bfloat16
    except Exception:  # noqa: BLE001 — older MPS builds reject bf16
        return torch.float16


def resolve_compute_dtype(device: torch.device, *, use_bf16: bool) -> torch.dtype:
    """Pick parameter dtype: BF16/FP16 on GPU accelerators, else FP32.

    * CUDA: ``bfloat16`` when the GPU supports it, else ``float16``.
    * MPS (Mac M-series): ``bfloat16`` when the runtime accepts it, else ``float16``.
    * CPU / ``use_bf16=False``: ``float32``.
    """
    if not use_bf16:
        return torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return _mps_half_dtype()
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


def freeze_non_fff_parameters(student: Any) -> tuple[int, int]:
    """Freeze embeddings, attention, norms, and LM head; keep FFF trainable.

    Full-model AdamW on a 2B student needs ~15GB of optimizer state plus a
    ~970MiB ``lm_head`` gradient — both blow a 12GB card. Distilling FFF
    only updates routers / leaf SwiGLU slices / ``output_scale``.

    Returns
    -------
    tuple[int, int]
        ``(n_trainable, n_frozen)`` element counts.
    """
    fff_ids: set[int] = set()
    for block in iter_fff_swiglu_blocks(student):
        for p in block.parameters():
            fff_ids.add(id(p))
    n_train = 0
    n_frozen = 0
    for p in student.parameters():
        if id(p) in fff_ids:
            p.requires_grad_(True)
            n_train += int(p.numel())
        else:
            p.requires_grad_(False)
            n_frozen += int(p.numel())
    return n_train, n_frozen


def build_student_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    weight_decay: float,
    adam_8bit: bool,
) -> torch.optim.Optimizer:
    """AdamW, preferring bitsandbytes 8-bit states on CUDA (≈4× less VRAM)."""
    kwargs: dict[str, Any] = {
        "weight_decay": float(weight_decay),
        "betas": (0.9, 0.95),
    }
    if adam_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise SystemExit(
                "bitsandbytes is required for 8-bit AdamW (default on 12GB).\n"
                "  pip install bitsandbytes\n"
                "Or pass --adam-fp32 if you have enough VRAM."
            ) from exc
        print("optimizer: AdamW8bit (FFF params only)")
        return bnb.optim.AdamW8bit(param_groups, **kwargs)
    print("optimizer: AdamW FP32")
    return torch.optim.AdamW(param_groups, **kwargs)


def load_4bit_teacher(model_name: str, device: torch.device) -> Any:
    """Load CausalLM teacher in bitsandbytes 4-bit (BF16 compute).

    4-bit NF4 requires CUDA. On MPS/CPU this falls back to a dense
    ``bfloat16`` / ``float16`` teacher instead of raising.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        dtype = resolve_compute_dtype(device, use_bf16=True)
        print(
            "CUDA not available for 4-bit NF4; loading dense teacher in "
            f"{dtype}."
        )
        return load_dense_teacher(model_name, device, dtype=dtype)

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "BitsAndBytesConfig missing — upgrade transformers"
        ) from exc

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


def load_dense_teacher(
    model_name: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Dense teacher in ``dtype`` (BF16/FP16 on Mac MPS, FP32 otherwise)."""
    print(f"Loading dense teacher `{model_name}` ({dtype}) ...")
    teacher = load_hf_causal_lm(model_name, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.to(device=device, dtype=dtype)


def build_fff_student(
    student_name: str,
    device: torch.device,
    *,
    fff_depth: int = 4,
    num_trees: int = 1,
    init_temp: float = 1.0,
    compute_dtype: torch.dtype = torch.bfloat16,
    freeze_backbone: bool = True,
    gradient_checkpointing: bool = True,
    gpu_max_memory: str = "10GiB",
    cpu_max_memory: str = "32GiB",
) -> Any:
    """Load Instruct student, inject FFF / CMM SwiGLU, freeze backbone.

    Hidden and MLP widths come from the student ``text_config`` (e.g. Qwen3.5-7B)
    and are snapped to ``num_trees * 2^{fff_depth}`` leaf slices.

    Weights load directly in ``compute_dtype`` (not FP32) so a 7B student fits
    CPU RAM / unified memory. FFF blocks are still smart-inited in FP32 then
    cast. On CUDA, Accelerate ``device_map='auto'`` with a 10GiB VRAM cap
    offloads frozen backbone layers to CPU when the 7B trunk exceeds 12GB.

    By default only FFF blocks stay trainable. Pass ``freeze_backbone=False``
    to finetune embeddings / attention / LM head.
    """
    print(f"Loading student `{student_name}` ({compute_dtype}) ...")
    student = load_hf_causal_lm(student_name, dtype=compute_dtype)
    for p in student.parameters():
        p.requires_grad_(True)

    text_cfg = get_text_config(student)
    d_model, used_i, n_leaves, slice_size = adapt_fff_swiglu_dims(
        int(text_cfg.hidden_size),
        int(text_cfg.intermediate_size),
        num_trees=num_trees,
        depth_per_tree=fff_depth,
    )
    print(
        f"  detected D={d_model} I={int(text_cfg.intermediate_size)} → "
        f"FFF I={used_i} | CMM K={num_trees} d={fff_depth} "
        f"leaves/tree={n_leaves} slice={slice_size}"
    )

    print("Injecting FFF SwiGLU CMM (smart-init FP32) ...")
    n = patch_model_with_fff_swiglu(
        student,
        fff_depth=fff_depth,
        num_trees=num_trees,
        init_temp=init_temp,
    )
    scale0 = math.sqrt(float(1 << fff_depth))
    print(
        f"  patched {n} MLPs | STE soft train | "
        f"output_scale init=√L={scale0:.4f} | num_trees={num_trees}"
    )

    if compute_dtype != torch.float32:
        print(f"Casting student → {compute_dtype} ...")
        student = student.to(dtype=compute_dtype)
    student.train()
    set_fff_routing_mode(student, "soft")

    if freeze_backbone:
        n_train, n_frozen = freeze_non_fff_parameters(student)
        print(
            f"  freeze_backbone=ON | FFF trainable={n_train:,} | "
            f"frozen={n_frozen:,}"
        )
    else:
        print("  freeze_backbone=OFF (full student trainable — needs ≫12GB VRAM)")

    if gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
        if hasattr(student, "config"):
            student.config.use_cache = False
        print("  gradient_checkpointing=ON (use_cache=False)")

    student = _place_student_on_device(
        student,
        device,
        gpu_max_memory=gpu_max_memory,
        cpu_max_memory=cpu_max_memory,
    )
    return student


def _place_student_on_device(
    student: Any,
    device: torch.device,
    *,
    gpu_max_memory: str = "10GiB",
    cpu_max_memory: str = "32GiB",
) -> Any:
    """Move student to ``device``; on CUDA cap VRAM and CPU-offload the rest.

    RTX 3060 12GB cannot hold a 7B bf16 trunk (~14GB) plus activations.
    ``max_memory={0: 10GiB, cpu: 32GiB}`` keeps ~2GB headroom.
    MPS / CPU use a plain ``.to(device)`` (unified / host memory).
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return student.to(device)
    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except ImportError:
        print(
            "  accelerate not installed — placing the full student on CUDA "
            "(7B may OOM on 12GB; pip install accelerate)"
        )
        return student.to(device)

    gpu_index = int(device.index) if device.index is not None else 0
    max_memory: dict[Any, str] = {
        gpu_index: str(gpu_max_memory),
        "cpu": str(cpu_max_memory),
    }
    no_split = [
        "FFFSwiGLUBlock",
        "FFFGemmaBlock",
        "Qwen3_5DecoderLayer",
        "Qwen3DecoderLayer",
        "Qwen2DecoderLayer",
        "LlamaDecoderLayer",
    ]
    print(f"  placing student with device_map=auto max_memory={max_memory} ...")
    device_map = infer_auto_device_map(
        student,
        max_memory=max_memory,
        no_split_module_classes=no_split,
    )
    student = dispatch_model(student, device_map=device_map)
    dmap = getattr(student, "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        from collections import Counter

        counts = Counter(str(v) for v in dmap.values())
        hist = " ".join(f"{dev}={n}" for dev, n in sorted(counts.items()))
        print(f"  student hf_device_map: {hist}")
    return student


def infer_model_input_device(model: Any, fallback: torch.device) -> torch.device:
    """Device that should receive ``input_ids`` for a possibly sharded HF model."""
    dmap = getattr(model, "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        for key in (
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "model.model.language_model.embed_tokens",
        ):
            if key in dmap:
                spec = dmap[key]
                break
        else:
            spec = next(iter(dmap.values()))
        if isinstance(spec, torch.device):
            return spec
        if isinstance(spec, int):
            return torch.device(f"cuda:{spec}")
        text = str(spec)
        if text.isdigit():
            return torch.device(f"cuda:{int(text)}")
        if text.startswith("cuda") or text == "cpu":
            return torch.device(text)
    try:
        param = next(model.parameters())
        if param.device.type != "meta":
            return param.device
    except StopIteration:
        pass
    return fallback


def gather_student_leaf_probs(student: Any) -> Tensor:
    """Concatenate cached leaf probs from all FFF blocks → ``(N, L)``.

    CMM blocks cache ``(N, K, L)``; those are flattened to ``(N·K, L)`` so
    load-balancing entropy is per-tree occupancy.
    """
    chunks: list[Tensor] = []
    for block in iter_fff_swiglu_blocks(student):
        p = block.leaf_probs()
        if p is None:
            continue
        if p.ndim == 3:
            n_tok, n_trees, n_leaves = p.shape
            p = p.reshape(n_tok * n_trees, n_leaves)
        chunks.append(p)
    if not chunks:
        raise RuntimeError("no leaf probs cached — run a soft student forward first")
    return torch.cat(chunks, dim=0)


def _first_param_device(module: Any, fallback: torch.device) -> torch.device:
    """Device of the first non-meta parameter, else ``fallback``."""
    try:
        for p in module.parameters():
            if p.device.type != "meta":
                return p.device
    except StopIteration:
        pass
    return fallback


def resolve_teacher_backbone(teacher: Any) -> Any:
    """Inner transformer that emits ``last_hidden_state`` (no LM head).

    HuggingFace CausalLM ``forward`` always runs ``lm_head`` over the full
    sequence and allocates ``(B, T, V)`` logits. Phase-1 extraction must call
    this backbone instead.
    """
    for name in ("model", "transformer", "language_model"):
        inner = getattr(teacher, name, None)
        if inner is not None and inner is not teacher and callable(inner):
            return inner
    getter = getattr(teacher, "get_decoder", None)
    if callable(getter):
        decoder = getter()
        if decoder is not None and callable(decoder):
            return decoder
    raise RuntimeError(
        "teacher has no .model / .transformer backbone; cannot skip the LM head"
    )


def resolve_teacher_lm_head(teacher: Any) -> Any:
    """Output projection ``(H → V)`` (``lm_head`` or ``get_output_embeddings``)."""
    head = getattr(teacher, "lm_head", None)
    if head is None:
        getter = getattr(teacher, "get_output_embeddings", None)
        if callable(getter):
            head = getter()
    if head is None or not callable(head):
        raise RuntimeError("teacher has no lm_head / output embeddings")
    return head


@torch.no_grad()
def forward_teacher_hidden(
    teacher: Any,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Backbone-only forward → ``last_hidden_state`` of shape ``(B, T, H)``.

    Does **not** run the LM head, so VRAM stays ``O(B·T·H)`` (~20 MiB at
    ``B=1, T=2048, H=5120`` bf16) instead of ``O(B·T·V)``.
    """
    backbone = resolve_teacher_backbone(teacher)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": False,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        out = backbone(**kwargs)
    except TypeError:
        kwargs.pop("return_dict", None)
        out = backbone(**kwargs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0] if isinstance(out, (tuple, list)) else out
    del out
    return hidden


@torch.no_grad()
def extract_topk_via_chunked_lm_head(
    hidden: Tensor,
    lm_head: Any,
    *,
    k: int = 50,
    vocab_limit: int | None = None,
    seq_chunk: int = TEACHER_TOPK_SEQ_CHUNK,
    value_dtype: torch.dtype = torch.float16,
) -> tuple[Tensor, Tensor]:
    """Project ``(B, T, H)`` through ``lm_head`` in sequence chunks, then Top-K.

    Each chunk allocates at most ``(B, seq_chunk, V)`` logits (~78 MiB bf16 at
    ``B=1, chunk=256, V=152064``), runs ``torch.topk`` in native precision,
    copies ``(B, chunk, K)`` to CPU, and frees the logits before the next
    slice. Never materializes ``(B, T, V)``.

    Parameters
    ----------
    hidden:
        ``(B, T, H)`` last hidden state on GPU (or CPU).
    lm_head:
        Linear (or embedding) mapping ``H → V``.
    k:
        Top-K along vocab.
    vocab_limit:
        Optional vocab cap ``v <= V``.
    seq_chunk:
        Tokens per LM-head slice (default 256).
    value_dtype:
        CPU storage dtype for Top-K values.

    Returns
    -------
    topk_indices:
        ``(B, T, K)`` int32 on CPU.
    topk_values:
        ``(B, T, K)`` ``value_dtype`` on CPU.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be (B, T, H), got {tuple(hidden.shape)}")
    _batch, seq_len, _hidden = hidden.shape
    chunk = max(int(seq_chunk), 1)
    k_req = max(int(k), 1)
    head_device = _first_param_device(lm_head, hidden.device)

    idx_cpu: list[Tensor] = []
    val_cpu: list[Tensor] = []
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        hidden_chunk = hidden[:, start:end, :]
        if hidden_chunk.device != head_device:
            hidden_chunk = hidden_chunk.to(device=head_device)
        logits = lm_head(hidden_chunk)
        del hidden_chunk
        if vocab_limit is not None:
            v = min(int(vocab_limit), int(logits.size(-1)))
            logits = logits[..., :v]
        k_eff = min(k_req, int(logits.size(-1)))
        topk_val, topk_idx = torch.topk(logits, k=k_eff, dim=-1)
        del logits
        idx_cpu.append(topk_idx.to(dtype=torch.int32, device="cpu"))
        val_cpu.append(topk_val.to(dtype=value_dtype, device="cpu"))
        del topk_val, topk_idx

    indices = torch.cat(idx_cpu, dim=1)
    values = torch.cat(val_cpu, dim=1)
    del idx_cpu, val_cpu
    return indices, values


def _release_cuda_cache() -> None:
    """Drop orphaned CUDA blocks after a large logits tensor is freed."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _topk_vocab_native(logits_chunk: Tensor, k_eff: int) -> tuple[Tensor, Tensor]:
    """``torch.topk`` in the chunk's native dtype; cast only ``(..., K)`` values.

    Parameters
    ----------
    logits_chunk:
        ``(..., V)`` scores (bf16 / fp16 / fp32). Not cast to FP32.
    k_eff:
        ``K`` along the last (vocab) axis.

    Returns
    -------
    topk_val:
        ``(..., K)`` float32 values.
    topk_idx:
        ``(..., K)`` int64 indices.
    """
    values, indices = torch.topk(logits_chunk, k=k_eff, dim=-1)
    out_val = values.float()
    del values
    return out_val, indices


@torch.no_grad()
def extract_teacher_topk(
    teacher_logits: Tensor,
    *,
    k: int = 200,
    vocab_limit: int | None = None,
    seq_chunk: int = TEACHER_TOPK_SEQ_CHUNK,
) -> tuple[Tensor, Tensor]:
    """Select teacher Top-K logits / indices without an FP32 ``(B, T, V)`` copy.

    ``torch.topk`` runs on ``teacher_logits[..., :v]`` in native bf16/fp16.
    Only the compact ``(..., K)`` values are cast to float32. When the
    sequence axis ``T`` (dim ``-2`` for rank ≥ 3, dim ``0`` for rank 2)
    exceeds ``seq_chunk`` (default 256), extraction loops over sequence
    chunks so peak extra VRAM is ``O(B · chunk · V)`` instead of
    ``O(B · T · V)`` in FP32 (~1.24 GiB at ``T=2048``, ``V=152064``).

    Parameters
    ----------
    teacher_logits:
        ``(B, T, V)`` or ``(N, V)`` teacher scores.
    k:
        Top-K along the vocab axis.
    vocab_limit:
        Optional vocab cap ``v <= V`` (e.g. student vocab).
    seq_chunk:
        Sequence-chunk length. ``T > seq_chunk`` enables chunking;
        ``seq_chunk <= 0`` disables it.

    Returns
    -------
    topk_val:
        ``(B, T, K)`` or ``(N, K)`` float32 values.
    topk_idx:
        Matching int64 indices.
    """
    v = int(teacher_logits.size(-1) if vocab_limit is None else vocab_limit)
    v = min(v, int(teacher_logits.size(-1)))
    k_eff = min(int(k), v)
    logits = teacher_logits[..., :v]
    chunk = int(seq_chunk)

    seq_dim: int | None = None
    seq_len = 0
    if logits.ndim >= 3:
        seq_dim = -2
        seq_len = int(logits.size(-2))
    elif logits.ndim == 2:
        seq_dim = 0
        seq_len = int(logits.size(0))

    if seq_dim is None or chunk <= 0 or seq_len <= chunk:
        topk_val, topk_idx = _topk_vocab_native(logits, k_eff)
        del logits
        return topk_val, topk_idx

    val_chunks: list[Tensor] = []
    idx_chunks: list[Tensor] = []
    for start in range(0, seq_len, chunk):
        length = min(chunk, seq_len - start)
        piece = logits.narrow(seq_dim, start, length)
        val_p, idx_p = _topk_vocab_native(piece, k_eff)
        val_chunks.append(val_p)
        idx_chunks.append(idx_p)
        del piece, val_p, idx_p

    del logits
    topk_val = torch.cat(val_chunks, dim=seq_dim)
    topk_idx = torch.cat(idx_chunks, dim=seq_dim)
    del val_chunks, idx_chunks
    _release_cuda_cache()
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
    default_model: str = "Qwen/Qwen3.5-2B",
    max_context_length: int = CONTEXT_LENGTH_256K,
    smart_init_only: bool = False,
    noise_std: float = 1e-3,
) -> tuple[Any, dict[str, Any]]:
    """Scaffold HF CausalLM → inject FFF SwiGLU → load ``student_state_dict``.

    ``smart_init_only=True`` keeps the dense-MLP slice init and **does not**
    load trained FFF weights. Use when a short-context distill checkpoint
    has collapsed language modeling.
    """
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = model_name or str(ckpt.get("model_name", default_model))
    cfg = ckpt.get("config") or {}
    fff_depth = int(
        ckpt.get(
            "depth_per_tree",
            ckpt.get("fff_depth", cfg.get("fff_depth", 4)),
        )
    )
    init_tau = float(cfg.get("init_tau", 1.0))
    n_ctx = int(ckpt.get("max_context_length", max_context_length))

    print(f"Loading student scaffold `{name}` (float32 init) ...")
    student = load_hf_causal_lm(name, dtype=torch.float32)
    apply_context_length(student, n_ctx=n_ctx)
    print(f"  context_length={model_context_length(student):,}")
    print(f"Injecting FFF SwiGLU (depth={fff_depth}) ...")
    n = patch_model_with_fff_swiglu(
        student, fff_depth=fff_depth, init_temp=init_tau, noise_std=noise_std,
        num_trees=int(ckpt.get("num_trees", cfg.get("num_trees", 1))),
    )
    if smart_init_only:
        print(
            "  smart_init_only=ON — skipping trained FFF weights "
            "(uses original Qwen MLP slices)"
        )
    else:
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
