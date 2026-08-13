#!/usr/bin/env python3
"""Pre-compute teacher Top-K logits for offline FFF distillation (Phase 1).

Loads **only** the teacher (default ``Qwen/Qwen2.5-14B-Instruct``), tokenizes
the multi-domain ChatML mixture (CoT / Agent / Code / Thai / English) at
``max_length=2048``, and writes compressed ``.pt`` shards::

    data/logits_cache/cot_logits.pt
    data/logits_cache/agent_logits.pt
    data/logits_cache/code_logits.pt
    data/logits_cache/thai_logits.pt
    data/logits_cache/english_logits.pt

Each shard stores, per token:

* ``topk_indices`` — Top-``K=50`` teacher vocab ids (int32), shape ``(N, T, K)``
* ``topk_values``  — matching logit values (fp16/bf16), shape ``(N, T, K)``

plus ``input_ids`` / ``attention_mask`` so Phase 2 never reloads the teacher.

14B on RTX 3060 12GB (full GPU, default)
---------------------------------------
``Qwen/Qwen2.5-14B-Instruct`` 4-bit NF4 is ~8.5GB and fits entirely on the
RTX 3060 12GB. Load with ``device_map={"": 0}`` (whole model on ``cuda:0``).
bitsandbytes 4-bit is **incompatible** with Accelerate CPU-offload hooks
(``Cannot copy out of meta tensor``), so 14B never uses ``max_memory`` or
``offload_folder``. Sequences are **left-truncated** to ``--max_length``
(default 2048) with **no global padding**. Each domain is sorted by length
and micro-batched (default 4, auto up to 8) with padding only to that
batch's max length (``pad_to_multiple_of=8``). ``attention_mask`` is passed
to the teacher so SDPA ignores pads.

32B (optional, CPU offload — fragile)
-------------------------------------
``Qwen/Qwen2.5-32B-Instruct`` still needs CPU offload on 12GB and may hit
the bitsandbytes/Accelerate meta-tensor bug. Prefer the 14B default.

Example
-------
    python extract_teacher_logits.py --device cuda
    python extract_teacher_logits.py --teacher_name Qwen/Qwen2.5-14B-Instruct
    python extract_teacher_logits.py --teacher_name Qwen/Qwen2.5-32B-Instruct
    python extract_teacher_logits.py --kl-topk 50 --max-length 2048 --batch-size 4
    python extract_teacher_logits.py --smoke-test --smoke-synthetic-data
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import Tensor
from tqdm import tqdm

from data.dataset_loader import (
    LOGITS_CACHE_DIR,
    format_byte_count,
    estimate_logits_cache_bytes,
    logits_cache_dir_size,
    logits_cache_path,
    save_teacher_logits_cache,
)
from device_utils import (
    apply_hardware_optimizations,
    print_device_info,
    resolve_device,
)
from fff_hf import (
    apply_context_length,
    attn_implementation_name,
    bf16_autocast,
    enable_fast_sdpa,
    enable_model_sdpa,
    encode_truncate_left,
    extract_teacher_topk,
    model_vocab_size,
    resolve_compute_dtype,
)
from train_fff_agent import (
    DEFAULT_SYSTEM,
    DOMAINS,
    load_domain_message_lists,
    messages_to_chatml,
)

DEFAULT_TEACHER: str = "Qwen/Qwen2.5-14B-Instruct"
LARGE_TEACHER: str = "Qwen/Qwen2.5-32B-Instruct"
# Optional Qwen3.6 Hub id (image-text; not the Phase-1 default).
HUB_TEACHER_35B: str = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_KL_TOPK: int = 50
DEFAULT_MAX_LENGTH: int = 2048
DEFAULT_MAX_SAMPLES: int = 8_000
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_MAX_BATCH_SIZE: int = 8
PAD_TO_MULTIPLE: int = 8
DEFAULT_GPU_MAX_MEMORY: str = "6GiB"
DEFAULT_CPU_MAX_MEMORY: str = "48GiB"
DEFAULT_OFFLOAD_FOLDER: str = "offload_cache"

# Friendly CLI names → actual HuggingFace repo.
_TEACHER_HUB_ALIASES: dict[str, str] = {
    "qwen/qwen2.5-14b-instruct": DEFAULT_TEACHER,
    "qwen/qwen2.5-14b": DEFAULT_TEACHER,
    "qwen2.5-14b-instruct": DEFAULT_TEACHER,
    "qwen2.5-14b": DEFAULT_TEACHER,
    "qwen/qwen2.5-32b-instruct": LARGE_TEACHER,
    "qwen/qwen2.5-32b": LARGE_TEACHER,
    "qwen2.5-32b-instruct": LARGE_TEACHER,
    "qwen2.5-32b": LARGE_TEACHER,
    "qwen/qwen3.6-35b-instruct": HUB_TEACHER_35B,
    "qwen/qwen3.6-35b": HUB_TEACHER_35B,
    "qwen/qwen3.6-35b-a3b-instruct": HUB_TEACHER_35B,
    "qwen3.6-35b-instruct": HUB_TEACHER_35B,
    "qwen3.6-35b-a3b": HUB_TEACHER_35B,
}


def resolve_teacher_hub_id(name: str) -> str:
    """Map ``--teacher_name`` aliases to a loadable HuggingFace repo id.

    Default teacher is ``Qwen/Qwen2.5-14B-Instruct`` (4-bit, full GPU).
    ``Qwen/Qwen2.5-32B-Instruct`` aliases map to the optional 32B checkpoint.
    Qwen3.6 ``*-Instruct`` aliases still map to ``Qwen/Qwen3.6-35B-A3B``.
    """
    raw = str(name).strip()
    mapped = _TEACHER_HUB_ALIASES.get(raw.lower())
    if mapped is None:
        return raw
    if mapped != raw:
        print(f"teacher alias `{raw}` → Hub checkpoint `{mapped}`")
    return mapped


def is_full_gpu_4bit_teacher(model_name: str) -> bool:
    """True when 4-bit weights fit on RTX 3060 12GB without CPU offload.

    ``Qwen/Qwen2.5-14B-Instruct`` is ~8.5 GiB in NF4, so it can sit entirely
    on ``cuda:0``. Larger teachers (32B) still need Accelerate CPU offload.
    """
    slug = str(model_name).lower().replace("_", "-")
    return "-14b" in slug or slug.endswith("14b")


def _require_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoTokenizer


def load_teacher_tokenizer(hub_id: str) -> Any:
    """Load tokenizer for a CausalLM teacher (Qwen2.5-14B-Instruct by default).

    Uses ``AutoTokenizer`` first. Falls back to ``AutoProcessor.tokenizer``
    only if the checkpoint has no standalone tokenizer.
    """
    AutoTokenizer = _require_tokenizer()
    try:
        tokenizer = AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"  AutoTokenizer failed ({exc.__class__.__name__}); "
            "trying AutoProcessor ..."
        )
        try:
            from transformers import AutoProcessor
        except ImportError as extra:
            raise SystemExit(
                f"Failed to load tokenizer for `{hub_id}`: {exc}"
            ) from extra
        processor = AutoProcessor.from_pretrained(hub_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "right"
    if getattr(tokenizer, "truncation_side", None) is not None:
        tokenizer.truncation_side = "left"
    return tokenizer


def _as_torch_device(spec: Any, fallback: torch.device) -> torch.device:
    """Convert an Accelerate ``hf_device_map`` value to ``torch.device``."""
    if isinstance(spec, torch.device):
        return spec
    if isinstance(spec, int):
        return torch.device(f"cuda:{spec}")
    text = str(spec).strip().lower()
    if text.isdigit():
        return torch.device(f"cuda:{int(text)}")
    if text in {"cpu", "disk", "meta"}:
        return torch.device("cpu") if text == "cpu" else fallback
    if text.startswith("cuda"):
        return torch.device(text)
    return fallback


def teacher_input_device(teacher: Any, fallback: torch.device) -> torch.device:
    """Device that should receive ``input_ids`` for a possibly sharded teacher.

    With ``device_map='auto'`` embeddings usually live on ``cuda:0``; remaining
    layers may sit on CPU RAM. Inputs must match the first mapped module.
    """
    dmap = getattr(teacher, "hf_device_map", None)
    if isinstance(dmap, dict) and dmap:
        for key in (
            "model.embed_tokens",
            "model.language_model.embed_tokens",
            "model.model.language_model.embed_tokens",
            "language_model.embed_tokens",
            "transformer.wte",
            "model.embed_tokens.weight",
        ):
            if key in dmap:
                return _as_torch_device(dmap[key], fallback)
        return _as_torch_device(next(iter(dmap.values())), fallback)
    try:
        param = next(teacher.parameters())
        if param.device.type != "meta":
            return param.device
    except StopIteration:
        pass
    return fallback


def summarize_device_map(teacher: Any) -> str:
    """Compact ``hf_device_map`` histogram for logs (e.g. ``cuda:0=28 cpu=36``)."""
    dmap = getattr(teacher, "hf_device_map", None)
    if not isinstance(dmap, dict) or not dmap:
        return "single-device"
    counts = Counter(str(v) for v in dmap.values())
    return " ".join(f"{dev}={n}" for dev, n in sorted(counts.items()))


def _require_causal_lm() -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoModelForCausalLM


def _from_pretrained_causal_lm(model_name: str, **kwargs: Any) -> Any:
    """Load via ``AutoModelForCausalLM.from_pretrained`` only.

    Never tries ``AutoModelForImageTextToText`` (that path materializes meta
    tensors and breaks bitsandbytes 4-bit offload with
    ``Tensor.item() cannot be called on meta tensors``).
    """
    AutoModelForCausalLM = _require_causal_lm()
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("low_cpu_mem_usage", True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        if kwargs.get("attn_implementation") != "sdpa":
            raise
        print("  SDPA attn_implementation rejected; retrying default attention ...")
        fallback = {k: v for k, v in kwargs.items() if k != "attn_implementation"}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return AutoModelForCausalLM.from_pretrained(model_name, **fallback)


def _freeze_eval(teacher: Any) -> Any:
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    return teacher


def load_extract_teacher(
    model_name: str,
    device: torch.device,
    *,
    use_4bit: bool,
    compute_dtype: torch.dtype,
    gpu_max_memory: str,
    cpu_max_memory: str,
    offload_folder: Path | str | None = None,
) -> Any:
    """Load the extract teacher with 4-bit NF4 on CUDA.

    14B CUDA (full GPU, default)
    ----------------------------
    ``Qwen/Qwen2.5-14B-Instruct`` 4-bit (~8.5 GiB) fits on RTX 3060 12GB.
    Load with ``device_map={"": 0}`` so the whole model sits on GPU 0.
    Do **not** pass ``max_memory`` or ``offload_folder`` — bitsandbytes 4-bit
    cannot run under Accelerate CPU-offload hooks (meta-tensor copy errors).

    32B CUDA (optional CPU offload)
    -------------------------------
    Larger teachers still use ``device_map='auto'`` + ``max_memory``; this
    path is known-fragile with bitsandbytes.

    MPS / CPU path
    --------------
    Dense ``bfloat16`` / ``float16`` (4-bit NF4 is CUDA-only).
    """
    if not use_4bit or device.type != "cuda" or not torch.cuda.is_available():
        print(
            f"Loading dense teacher `{model_name}` ({compute_dtype}) "
            f"on {device} (no 4-bit path)."
        )
        if any(tag in model_name.lower() for tag in ("14b", "32b", "35b")) and device.type != "cuda":
            print(
                "warning: Qwen2.5 14B extraction is intended for CUDA "
                "(RTX 3060 12GB). Dense weights will likely OOM here."
            )
        teacher = _from_pretrained_causal_lm(
            model_name,
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        return _freeze_eval(teacher.to(device=device, dtype=compute_dtype))

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "BitsAndBytesConfig missing — upgrade transformers"
        ) from exc
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as extra:
        raise SystemExit(
            "bitsandbytes is required for 4-bit extraction:\n"
            "  pip install bitsandbytes"
        ) from extra

    gc.collect()
    torch.cuda.empty_cache()

    gpu_index = int(device.index) if device.index is not None else 0
    full_gpu = is_full_gpu_4bit_teacher(model_name)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=not full_gpu,
    )
    extra: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }

    if full_gpu:
        device_map: Any = {"": gpu_index}
        print(
            f"Loading 4-bit teacher `{model_name}` on device_map={{'': {gpu_index}}} "
            "(~8.5GiB NF4, no CPU offload / no max_memory / no offload_folder) ..."
        )
        teacher = _from_pretrained_causal_lm(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
            **extra,
        )
        teacher = _freeze_eval(teacher)
        print(f"  hf_device_map: {summarize_device_map(teacher)}")
        return teacher

    print(
        "warning: bitsandbytes 4-bit + Accelerate CPU offload is known to fail "
        "with meta-tensor copy errors. Prefer Qwen/Qwen2.5-14B-Instruct."
    )
    max_memory: dict[Any, str] = {
        gpu_index: str(gpu_max_memory),
        "cpu": str(cpu_max_memory),
    }
    offload_dir = Path(offload_folder) if offload_folder is not None else Path(DEFAULT_OFFLOAD_FOLDER)
    offload_dir.mkdir(parents=True, exist_ok=True)
    extra["offload_folder"] = str(offload_dir)
    extra["offload_state_dict"] = True

    print(
        f"Loading 4-bit teacher `{model_name}` with device_map='auto' "
        f"max_memory={max_memory} offload_folder={offload_dir} "
        "(bnb_4bit_use_double_quant=False) ..."
    )
    teacher = _from_pretrained_causal_lm(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        max_memory=max_memory,
        **extra,
    )
    teacher = _freeze_eval(teacher)
    print(f"  hf_device_map: {summarize_device_map(teacher)}")
    return teacher


def tokenize_domain(
    messages_list: list[list[dict[str, str]]],
    tokenizer: Any,
    *,
    max_length: int,
    pad_id: int,
    default_system: str = DEFAULT_SYSTEM,
) -> list[list[int]]:
    """Encode ChatML rows **without** padding; left-truncate to ``max_length``.

    Uses ``truncation=True, max_length=..., padding=False``. Empty encodings
    are dropped. Returned sequences are variable-length (not stacked).

    ``pad_id`` is unused here (collate pads later); kept for call-site stability.
    """
    del pad_id
    max_length = max(int(max_length), 1)
    texts = [
        messages_to_chatml(msgs, default_system=default_system)
        for msgs in messages_list
    ]
    prev_side = getattr(tokenizer, "truncation_side", "right")
    seqs: list[list[int]] = []
    try:
        tokenizer.truncation_side = "left"
        try:
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            seqs = [list(ids) for ids in enc["input_ids"] if ids]
        except TypeError:
            for text in texts:
                ids = encode_truncate_left(tokenizer, text, max_length=max_length)
                if ids:
                    seqs.append(ids)
    finally:
        try:
            tokenizer.truncation_side = prev_side
        except Exception:
            pass
    if not seqs:
        raise ValueError("domain produced zero tokenized samples")
    return seqs


def sort_sequences_by_length(
    sequences: list[list[int]],
    *,
    descending: bool = True,
) -> list[list[int]]:
    """Sort tokenized rows by ``len(input_ids)`` so batches share similar T."""
    return sorted(sequences, key=len, reverse=bool(descending))


def _ceil_to_multiple(length: int, multiple: int, cap: int) -> int:
    """Round ``length`` up to ``multiple``, then clamp to ``[1, cap]``."""
    length = max(int(length), 1)
    cap = max(int(cap), 1)
    multiple = max(int(multiple), 1)
    padded = ((length + multiple - 1) // multiple) * multiple
    return min(max(padded, 1), cap)


def collate_dynamic_batch(
    sequences: list[list[int]],
    *,
    pad_id: int,
    max_length: int,
    pad_multiple: int = PAD_TO_MULTIPLE,
) -> tuple[Tensor, Tensor]:
    """Right-pad a micro-batch to its own max length (``pad_to_multiple_of``).

    Returns
    -------
    input_ids:
        ``(B, T_batch)`` int64, ``T_batch = min(cap, ceil(max_len / multiple) * multiple)``.
    attention_mask:
        ``(B, T_batch)`` int64 0/1 for SDPA (padded positions are 0).
    """
    if not sequences:
        raise ValueError("cannot collate an empty batch")
    raw_max = max(len(s) for s in sequences)
    seq_len = _ceil_to_multiple(raw_max, pad_multiple, max_length)
    batch = len(sequences)
    input_ids = torch.full((batch, seq_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, seq_len), dtype=torch.long)
    for i, ids in enumerate(sequences):
        n = min(len(ids), seq_len)
        if n <= 0:
            continue
        input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
        attention_mask[i, :n] = 1
    return input_ids, attention_mask


def _plan_micro_batch(
    sequences: list[list[int]],
    start: int,
    *,
    max_batch: int,
    token_budget: int,
    max_length: int,
    pad_multiple: int,
) -> int:
    """Grow a batch while ``T_pad * B <= token_budget`` (and ``B <= max_batch``)."""
    n_seq = len(sequences)
    if start >= n_seq:
        return 0
    max_batch = max(int(max_batch), 1)
    token_budget = max(int(token_budget), 1)
    batch_max = len(sequences[start])
    size = 1
    while start + size < n_seq and size < max_batch:
        nxt = len(sequences[start + size])
        pad_t = _ceil_to_multiple(max(batch_max, nxt), pad_multiple, max_length)
        if pad_t * (size + 1) > token_budget:
            break
        batch_max = max(batch_max, nxt)
        size += 1
    return size


def _right_pad_seq_dim(tensor: Tensor, target_t: int, *, fill: int | float) -> Tensor:
    """Right-pad dim 1 of ``(B, T, ...)`` to ``target_t``."""
    cur = int(tensor.size(1))
    target_t = max(int(target_t), 1)
    if cur == target_t:
        return tensor
    if cur > target_t:
        return tensor[:, :target_t, ...].contiguous()
    shape = list(tensor.shape)
    shape[1] = target_t
    out = tensor.new_full(tuple(shape), fill)
    out[:, :cur, ...] = tensor
    return out


def length_stats(sequences: list[list[int]]) -> str:
    """Compact min / median / mean / max length string for logs."""
    lengths = sorted(len(s) for s in sequences)
    n = len(lengths)
    if n == 0:
        return "n=0"
    mean = sum(lengths) / n
    mid = lengths[n // 2]
    return (
        f"n={n:,} len[min={lengths[0]} median={mid} mean={mean:.0f} "
        f"max={lengths[-1]}]"
    )


def _is_oom(exc: BaseException) -> bool:
    """True for CUDA / MPS allocator failures (safe to retry a smaller batch)."""
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "not enough memory" in msg
        or name in {"cudaoutofmemoryerror", "outofmemoryerror"}
    )


@torch.no_grad()
def _forward_topk_batch(
    teacher: Any,
    ids: Tensor,
    mask: Tensor,
    *,
    autocast_device: torch.device,
    k: int,
    vocab_limit: int | None,
    value_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """One teacher micro-batch → CPU Top-K ``(B, T, K)`` indices / values."""
    with bf16_autocast(autocast_device):
        out = teacher(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
        )
        logits = out.logits
        del out
        topk_val, topk_idx = extract_teacher_topk(
            logits, k=k, vocab_limit=vocab_limit
        )
        del logits
    idx_cpu = topk_idx.to(dtype=torch.int32, device="cpu")
    val_cpu = topk_val.to(dtype=value_dtype, device="cpu")
    del topk_val, topk_idx
    return idx_cpu, val_cpu


@torch.no_grad()
def extract_topk_for_ids(
    teacher: Any,
    sequences: list[list[int]],
    *,
    pad_id: int,
    device: torch.device,
    kl_topk: int,
    batch_size: int,
    max_batch_size: int,
    max_length: int,
    value_dtype: torch.dtype,
    vocab_limit: int | None = None,
    pad_multiple: int = PAD_TO_MULTIPLE,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Teacher forward over variable-length rows with dynamic micro-batch padding.

    Sequences must already be **sorted by length**. Each GPU micro-batch is
    padded only to that batch's max length (rounded up to ``pad_multiple``),
    never to a global ``T=2048``. ``attention_mask`` is passed into
    ``teacher()`` so SDPA ignores pads. Results are right-padded to
    ``max_length`` on CPU so Phase 2 still sees rectangular ``(N, T, K)``.

    Micro-batch size starts from ``batch_size`` (default 4) and auto-grows
    up to ``max_batch_size`` (default 8) while ``T_pad * B`` stays under a
    token budget of ``batch_size * max_length``. CUDA OOM halves the budget.

    Returns
    -------
    input_ids:
        ``(N, T)`` int32 on CPU, ``T = max_length``.
    attention_mask:
        ``(N, T)`` uint8 on CPU.
    topk_indices:
        ``(N, T, K)`` int32 on CPU.
    topk_values:
        ``(N, T, K)`` ``value_dtype`` on CPU.
    """
    n_samples = len(sequences)
    if n_samples == 0:
        raise ValueError("no sequences to extract")
    cache_t = max(int(max_length), 1)
    k = max(int(kl_topk), 1)
    max_bs = max(int(max_batch_size), 1)
    token_budget = max(int(batch_size), 1) * cache_t
    v_limit = vocab_limit
    input_dev = teacher_input_device(teacher, device)
    pad_multiple = max(int(pad_multiple), 1)

    id_chunks: list[Tensor] = []
    mask_chunks: list[Tensor] = []
    idx_chunks: list[Tensor] = []
    val_chunks: list[Tensor] = []

    pbar = tqdm(
        total=n_samples,
        desc="  teacher Top-K",
        unit="ex",
        leave=False,
        dynamic_ncols=True,
    )
    start = 0
    while start < n_samples:
        micro = _plan_micro_batch(
            sequences,
            start,
            max_batch=max_bs,
            token_budget=token_budget,
            max_length=cache_t,
            pad_multiple=pad_multiple,
        )
        end = min(start + micro, n_samples)
        batch_seqs = sequences[start:end]
        ids_cpu, mask_cpu = collate_dynamic_batch(
            batch_seqs,
            pad_id=pad_id,
            max_length=cache_t,
            pad_multiple=pad_multiple,
        )
        ids = ids_cpu.to(device=input_dev, dtype=torch.long)
        mask = mask_cpu.to(device=input_dev, dtype=torch.long)
        gpu_t = int(ids.size(1))
        if v_limit is not None:
            ids = ids.clamp(0, int(v_limit) - 1)
        try:
            idx_b, val_b = _forward_topk_batch(
                teacher,
                ids,
                mask,
                autocast_device=device if device.type == "cuda" else input_dev,
                k=k,
                vocab_limit=v_limit,
                value_dtype=value_dtype,
            )
        except Exception as exc:
            del ids, mask, ids_cpu, mask_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if _is_oom(exc) and micro > 1:
                pad_t = _ceil_to_multiple(
                    max(len(s) for s in batch_seqs), pad_multiple, cache_t
                )
                token_budget = max(pad_t * max(micro // 2, 1), pad_t)
                print(
                    f"  OOM at B={micro} T={pad_t}; retrying with "
                    f"token_budget={token_budget}"
                )
                continue
            raise
        del ids, mask
        idx_cpu = _right_pad_seq_dim(idx_b, cache_t, fill=0)
        val_cpu = _right_pad_seq_dim(val_b, cache_t, fill=0)
        ids_out = _right_pad_seq_dim(ids_cpu, cache_t, fill=int(pad_id)).to(
            dtype=torch.int32
        )
        mask_out = _right_pad_seq_dim(mask_cpu, cache_t, fill=0).to(dtype=torch.uint8)
        del idx_b, val_b, ids_cpu, mask_cpu
        id_chunks.append(ids_out)
        mask_chunks.append(mask_out)
        idx_chunks.append(idx_cpu)
        val_chunks.append(val_cpu)
        del ids_out, mask_out, idx_cpu, val_cpu
        pbar.update(end - start)
        pbar.set_postfix_str(
            f"{end}/{n_samples} B={micro} T={gpu_t} budget={token_budget}",
            refresh=False,
        )
        start = end
    pbar.close()

    input_ids = torch.cat(id_chunks, dim=0)
    attention_mask = torch.cat(mask_chunks, dim=0)
    topk_indices = torch.cat(idx_chunks, dim=0)
    topk_values = torch.cat(val_chunks, dim=0)
    del id_chunks, mask_chunks, idx_chunks, val_chunks
    expected_nt = (n_samples, cache_t)
    if tuple(topk_indices.shape[:2]) != expected_nt:
        raise RuntimeError(
            f"unexpected topk_indices shape {tuple(topk_indices.shape)} "
            f"(expected {expected_nt + ('K',)})"
        )
    return input_ids, attention_mask, topk_indices, topk_values


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pre-compute teacher Top-K logits for offline distillation "
            "(14B 4-bit on cuda:0; optional 32B CPU offload)"
        )
    )
    p.add_argument(
        "--teacher-name",
        "--teacher_name",
        dest="teacher_name",
        type=str,
        default=DEFAULT_TEACHER,
        help=(
            "Teacher HF id (default: Qwen/Qwen2.5-14B-Instruct; "
            "optional: Qwen/Qwen2.5-32B-Instruct)"
        ),
    )
    p.add_argument(
        "--kl-topk",
        type=int,
        default=DEFAULT_KL_TOPK,
        help="K for teacher Top-K (default 50)",
    )
    p.add_argument(
        "--max-length",
        "--max_length",
        dest="max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Cap / left-truncate length T (default 2048); GPU batches pad dynamically below this",
    )
    p.add_argument("--max-samples-per-domain", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument(
        "--batch-size",
        "--micro-batch-size",
        "--micro_batch_size",
        dest="batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Teacher micro-batch (default 4; auto-grows with short sequences)",
    )
    p.add_argument(
        "--max-batch-size",
        "--max_batch_size",
        dest="max_batch_size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Upper bound for auto-tuned micro-batch (default 8)",
    )
    p.add_argument(
        "--gpu-max-memory",
        type=str,
        default=DEFAULT_GPU_MAX_MEMORY,
        help="Accelerate GPU cap for device_map=auto (default 6GiB on RTX 3060)",
    )
    p.add_argument(
        "--cpu-max-memory",
        type=str,
        default=DEFAULT_CPU_MAX_MEMORY,
        help="Accelerate CPU RAM budget for offloaded layers (default 48GiB)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (cuda>mps>cpu) | cuda | mps | cpu",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(LOGITS_CACHE_DIR),
        help="Directory for {domain}_logits.pt shards",
    )
    p.add_argument(
        "--domains",
        type=str,
        default=",".join(DOMAINS),
        help="Comma-separated domains to extract",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp32", action="store_true", help="Dense teacher in FP32")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit even on CUDA")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing {domain}_logits.pt shards",
    )
    p.add_argument("--smoke-synthetic-data", action="store_true")
    p.add_argument(
        "--smoke-test",
        "--fast-dev",
        "--smoke_test",
        "--fast_dev",
        dest="smoke_test",
        action="store_true",
        help="Cap each domain at 200 samples (pipeline check)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    domains = tuple(
        d.strip().lower() for d in str(args.domains).split(",") if d.strip()
    )
    if not domains:
        raise SystemExit("no domains specified")
    unknown = [d for d in domains if d not in DOMAINS]
    if unknown:
        raise SystemExit(f"unknown domains {unknown}; expected subset of {list(DOMAINS)}")

    max_samples = int(args.max_samples_per_domain)
    if args.smoke_test:
        max_samples = min(max_samples, 200)
    if args.smoke_synthetic_data:
        max_samples = min(max_samples, 64)

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device(args.device)
    apply_hardware_optimizations(device)
    enable_fast_sdpa()

    teacher_4bit = not bool(args.no_4bit)
    if teacher_4bit and not torch.cuda.is_available():
        teacher_4bit = False
        host = "macOS detected" if sys.platform == "darwin" else "no NVIDIA GPU"
        print(
            f"CUDA not available ({host}). Auto-disabling 4-bit teacher "
            "quantization and falling back to native bfloat16/float16."
        )

    compute_dtype = resolve_compute_dtype(device, use_bf16=not args.fp32)
    value_dtype = (
        torch.bfloat16
        if compute_dtype == torch.bfloat16
        else torch.float16
    )
    cache_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Extract teacher Top-K logits — Phase 1 (14B 4-bit on GPU 0)")
    print("=" * 72)
    print_device_info(device)

    requested_teacher = str(args.teacher_name)
    hub_teacher = resolve_teacher_hub_id(requested_teacher)
    full_gpu = teacher_4bit and is_full_gpu_4bit_teacher(hub_teacher)
    print(
        f"teacher={requested_teacher} | hub={hub_teacher} | 4bit={teacher_4bit} | "
        f"dtype={compute_dtype} | store_values={value_dtype}"
    )
    print(
        f"K={int(args.kl_topk)} | T_cap={int(args.max_length)} | "
        f"samples/domain={max_samples} | micro-batch={int(args.batch_size)} "
        f"(max {int(args.max_batch_size)}, dynamic pad x{PAD_TO_MULTIPLE})"
    )
    if teacher_4bit and full_gpu:
        print(
            'device_map={"": 0} (14B 4-bit ~8.5GiB on RTX 3060; no CPU offload)'
        )
    elif teacher_4bit:
        print(
            f"max_memory GPU={args.gpu_max_memory} CPU={args.cpu_max_memory} "
            f"offload_folder={DEFAULT_OFFLOAD_FOLDER} "
            "(device_map=auto, bnb_4bit_use_double_quant=False)"
        )
    print(f"output_dir={cache_dir}")
    n_dom = max(len(domains), 1)
    est = estimate_logits_cache_bytes(
        n_samples=max_samples,
        max_length=int(args.max_length),
        kl_topk=int(args.kl_topk),
        n_domains=n_dom,
    )
    print(
        f"est. uncompressed Top-K cache ≈ {format_byte_count(est)} "
        f"({n_dom} domains × {max_samples:,} × T={int(args.max_length)} × K={int(args.kl_topk)}; "
        "zip .pt will be smaller)"
    )

    print("\nLoading teacher tokenizer ...")
    tokenizer = load_teacher_tokenizer(hub_teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)
    apply_context_length(tokenizer=tokenizer, n_ctx=int(args.max_length))

    print("\nLoading teacher (student is NOT loaded) ...")
    teacher = load_extract_teacher(
        hub_teacher,
        device,
        use_4bit=teacher_4bit,
        compute_dtype=compute_dtype,
        gpu_max_memory=str(args.gpu_max_memory),
        cpu_max_memory=str(args.cpu_max_memory),
        offload_folder=None if full_gpu else DEFAULT_OFFLOAD_FOLDER,
    )
    enable_model_sdpa(teacher)
    teacher.eval()
    vocab_t = model_vocab_size(teacher)
    in_dev = teacher_input_device(teacher, device)
    print(
        f"attention=SDPA | teacher={attn_implementation_name(teacher)} | "
        f"vocab={vocab_t:,} | input_device={in_dev}"
    )

    t0 = time.perf_counter()
    written: list[Path] = []
    for domain in domains:
        dest = logits_cache_path(domain, cache_dir)
        if dest.exists() and not args.force:
            print(f"\n[{domain}] skip existing {dest.name} (pass --force to overwrite)")
            written.append(dest)
            continue

        print(f"\n[{domain}] loading ChatML samples ...")
        messages_list = load_domain_message_lists(
            domain,
            max_samples=max_samples,
            smoke_synthetic=bool(args.smoke_synthetic_data),
        )
        sequences = tokenize_domain(
            messages_list,
            tokenizer,
            max_length=int(args.max_length),
            pad_id=pad_id,
        )
        sequences = sort_sequences_by_length(sequences, descending=True)
        n_tok = sum(len(s) for s in sequences)
        print(f"[{domain}] tokenized {length_stats(sequences)} tokens={n_tok:,}")

        input_ids, attention_mask, topk_indices, topk_values = extract_topk_for_ids(
            teacher,
            sequences,
            pad_id=pad_id,
            device=device,
            kl_topk=int(args.kl_topk),
            batch_size=int(args.batch_size),
            max_batch_size=int(args.max_batch_size),
            max_length=int(args.max_length),
            value_dtype=value_dtype,
            vocab_limit=vocab_t,
        )
        del sequences
        save_teacher_logits_cache(
            dest,
            domain=domain,
            input_ids=input_ids,
            attention_mask=attention_mask,
            topk_indices=topk_indices,
            topk_values=topk_values,
            teacher_name=hub_teacher,
            pad_id=pad_id,
            extra_meta={
                "kl_topk": int(args.kl_topk),
                "max_length": int(args.max_length),
                "value_dtype": str(value_dtype).replace("torch.", ""),
                "teacher_4bit": bool(teacher_4bit),
                "teacher_requested": requested_teacher,
                "device_map": (
                    {"": 0} if full_gpu else ("auto" if teacher_4bit else "none")
                ),
                "gpu_max_memory": str(args.gpu_max_memory) if teacher_4bit and not full_gpu else None,
                "cpu_max_memory": str(args.cpu_max_memory) if teacher_4bit and not full_gpu else None,
                "bnb_4bit_use_double_quant": False,
                "offload_folder": None if full_gpu else DEFAULT_OFFLOAD_FOLDER,
                "n_non_pad": n_tok,
                "dynamic_padding": True,
                "pad_multiple": PAD_TO_MULTIPLE,
                "batch_size": int(args.batch_size),
                "max_batch_size": int(args.max_batch_size),
            },
        )
        size = dest.stat().st_size
        print(
            f"[{domain}] wrote {dest} ({format_byte_count(size)}) | "
            f"idx={tuple(topk_indices.shape)} {topk_indices.dtype} | "
            f"val={tuple(topk_values.shape)} {topk_values.dtype}"
        )
        written.append(dest)
        del input_ids, attention_mask, topk_indices, topk_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    total = logits_cache_dir_size(cache_dir)
    print("\n" + "=" * 72)
    print(f"Done in {elapsed / 60.0:.1f} min | shards={len(written)}")
    for path in written:
        if path.exists():
            print(f"  {path.name}: {format_byte_count(path.stat().st_size)}")
    print(f"Total disk used under {cache_dir}: {format_byte_count(total)}")
    print("Train the student with: python train_offline_distill.py")


if __name__ == "__main__":
    main()
