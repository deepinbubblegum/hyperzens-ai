#!/usr/bin/env python3
"""Pre-compute teacher Top-K logits for offline FFF distillation (Phase 1).

Loads **only** the teacher (default ``Qwen/Qwen3.6-35B-Instruct``, resolved to
the Hub checkpoint ``Qwen/Qwen3.6-35B-A3B``), tokenizes the multi-domain
ChatML mixture (CoT / Agent / Code / Thai / English) at ``max_length=2048``,
and writes compressed ``.pt`` shards::

    data/logits_cache/cot_logits.pt
    data/logits_cache/agent_logits.pt
    data/logits_cache/code_logits.pt
    data/logits_cache/thai_logits.pt
    data/logits_cache/english_logits.pt

Each shard stores, per token:

* ``topk_indices`` — Top-``K=50`` teacher vocab ids (int32), shape ``(N, T, K)``
* ``topk_values``  — matching logit values (fp16/bf16), shape ``(N, T, K)``

plus ``input_ids`` / ``attention_mask`` so Phase 2 never reloads the teacher.

35B on RTX 3060 12GB
--------------------
Qwen3.6-35B is a multimodal MoE checkpoint (~18GB in 4-bit). bitsandbytes
NF4 + Accelerate ``device_map="auto"`` with
``max_memory={0: "6GiB", "cpu": "48GiB"}`` keeps ~6GB of 4-bit weights on the
RTX 3060 and leaves ~5.6GB VRAM free for activations and CUDA allocator
overhead. Remaining 4-bit layers offload to system RAM.
Default micro-batch is 1 at ``T=2048`` (raise to 2 if VRAM allows).

Example
-------
    python extract_teacher_logits.py --device cuda
    python extract_teacher_logits.py --teacher_name Qwen/Qwen3.6-35B-Instruct
    python extract_teacher_logits.py --kl-topk 50 --max-length 2048 --batch-size 1
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
    load_dense_teacher,
    load_hf_causal_lm,
    model_vocab_size,
    resolve_compute_dtype,
)
from train_fff_agent import (
    DEFAULT_SYSTEM,
    DOMAINS,
    load_domain_message_lists,
    messages_to_chatml,
)

DEFAULT_TEACHER: str = "Qwen/Qwen3.6-35B-Instruct"
# Official Hub id for the 35B Instruct / thinking checkpoint (MoE 35B / 3B active).
HUB_TEACHER_35B: str = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_KL_TOPK: int = 50
DEFAULT_MAX_LENGTH: int = 2048
DEFAULT_MAX_SAMPLES: int = 8_000
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_GPU_MAX_MEMORY: str = "6GiB"
DEFAULT_CPU_MAX_MEMORY: str = "48GiB"

# Friendly CLI names → actual HuggingFace repo (Qwen3.6 has no *-Instruct repo).
_TEACHER_HUB_ALIASES: dict[str, str] = {
    "qwen/qwen3.6-35b-instruct": HUB_TEACHER_35B,
    "qwen/qwen3.6-35b": HUB_TEACHER_35B,
    "qwen/qwen3.6-35b-a3b-instruct": HUB_TEACHER_35B,
    "qwen3.6-35b-instruct": HUB_TEACHER_35B,
    "qwen3.6-35b-a3b": HUB_TEACHER_35B,
}


def resolve_teacher_hub_id(name: str) -> str:
    """Map ``--teacher_name`` aliases to a loadable HuggingFace repo id.

    ``Qwen/Qwen3.6-35B-Instruct`` is the Phase-1 display name; weights live at
    ``Qwen/Qwen3.6-35B-A3B`` (image-text Instruct checkpoint).
    """
    raw = str(name).strip()
    mapped = _TEACHER_HUB_ALIASES.get(raw.lower())
    if mapped is None:
        return raw
    if mapped != raw:
        print(f"teacher alias `{raw}` → Hub checkpoint `{mapped}`")
    return mapped


def _require_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoTokenizer


def load_teacher_tokenizer(hub_id: str) -> Any:
    """Load tokenizer for a Qwen3.6 (or CausalLM) teacher checkpoint.

    Tries ``AutoTokenizer`` first, then ``AutoProcessor.tokenizer`` for
    image-text Instruct wrappers that do not expose a standalone tokenizer.
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


def load_extract_teacher(
    model_name: str,
    device: torch.device,
    *,
    use_4bit: bool,
    compute_dtype: torch.dtype,
    gpu_max_memory: str,
    cpu_max_memory: str,
    offload_folder: Path | None = None,
) -> Any:
    """Load the extract teacher with 4-bit + GPU/CPU split on CUDA.

    CUDA path
    ---------
    ``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16)``
    plus Accelerate ``device_map='auto'`` and::

        max_memory = {0: "6GiB", "cpu": "48GiB"}

    so RTX 3060 12GB keeps ~6GB of 4-bit weights on GPU and ~5.6GB VRAM
    free for activations / CUDA caching allocator. Remaining 35B 4-bit
    layers offload to system RAM (PCIe during the forward).

    MPS / CPU path
    --------------
    Dense ``bfloat16`` / ``float16`` (4-bit NF4 is CUDA-only). A 35B dense
    teacher will not fit on a laptop — use CUDA for Phase 1 extraction.
    """
    if not use_4bit or device.type != "cuda" or not torch.cuda.is_available():
        print(
            f"Loading dense teacher `{model_name}` ({compute_dtype}) "
            f"on {device} (no 4-bit CPU-offload path)."
        )
        if any(tag in model_name.lower() for tag in ("32b", "35b")) and device.type != "cuda":
            print(
                "warning: Qwen3.6-35B extraction is intended for CUDA "
                "(RTX 3060 12GB + ~48GiB RAM). Dense 35B will likely OOM here."
            )
        return load_dense_teacher(model_name, device, dtype=compute_dtype)

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
            "bitsandbytes is required for 4-bit 35B extraction:\n"
            "  pip install bitsandbytes"
        ) from extra

    gpu_index = int(device.index) if device.index is not None else 0
    max_memory: dict[Any, str] = {
        gpu_index: str(gpu_max_memory),
        "cpu": str(cpu_max_memory),
    }
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    extra: dict[str, Any] = {}
    if offload_folder is not None:
        offload_folder.mkdir(parents=True, exist_ok=True)
        extra["offload_folder"] = str(offload_folder)
        extra["offload_state_dict"] = True

    print(
        f"Loading 4-bit teacher `{model_name}` with device_map='auto' "
        f"max_memory={max_memory} (bnb_4bit_compute_dtype=bfloat16) ..."
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    teacher = load_hf_causal_lm(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        max_memory=max_memory,
        extra_from_pretrained=extra or None,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    print(f"  hf_device_map: {summarize_device_map(teacher)}")
    return teacher


def tokenize_domain(
    messages_list: list[list[dict[str, str]]],
    tokenizer: Any,
    *,
    max_length: int,
    pad_id: int,
    default_system: str = DEFAULT_SYSTEM,
) -> tuple[Tensor, Tensor]:
    """Encode ChatML rows to right-padded ``(N, T)`` int32 ids + uint8 mask.

    Left-truncates long traces so the assistant tail is kept
    (same contract as ``MultiDomainChatMixture``).
    """
    max_length = max(int(max_length), 1)
    rows: list[Tensor] = []
    for msgs in messages_list:
        text = messages_to_chatml(msgs, default_system=default_system)
        ids = encode_truncate_left(tokenizer, text, max_length=max_length)
        if not ids:
            continue
        row = torch.full((max_length,), int(pad_id), dtype=torch.int32)
        n = min(len(ids), max_length)
        row[:n] = torch.tensor(ids[:n], dtype=torch.int32)
        rows.append(row)
    if not rows:
        raise ValueError("domain produced zero tokenized samples")
    input_ids = torch.stack(rows, dim=0)
    attention_mask = (input_ids != int(pad_id)).to(dtype=torch.uint8)
    return input_ids, attention_mask


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
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    device: torch.device,
    kl_topk: int,
    batch_size: int,
    value_dtype: torch.dtype,
    vocab_limit: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Teacher forward over ``(N, T)`` ids → compact Top-K tensors on CPU.

    Uses a small micro-batch (default 1 at T=2048) so a 4-bit 35B teacher with CPU
    offload stays under the 6GiB weight cap. On CUDA OOM the batch size is
    halved automatically down to 1.

    Parameters
    ----------
    input_ids:
        ``(N, T)`` int token ids (CPU or device).
    attention_mask:
        ``(N, T)`` 0/1 mask.
    kl_topk:
        ``K`` in Top-K over the teacher vocab axis.
    value_dtype:
        Storage dtype for logit values (``float16`` or ``bfloat16``).

    Returns
    -------
    topk_indices:
        ``(N, T, K)`` int32 on CPU.
    topk_values:
        ``(N, T, K)`` ``value_dtype`` on CPU.
    """
    n_samples = int(input_ids.size(0))
    seq_len = int(input_ids.size(1))
    k = max(int(kl_topk), 1)
    micro = max(int(batch_size), 1)
    v_limit = vocab_limit
    input_dev = teacher_input_device(teacher, device)

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
        end = min(start + micro, n_samples)
        ids = input_ids[start:end].to(device=input_dev, dtype=torch.long)
        mask = attention_mask[start:end].to(device=input_dev, dtype=torch.long)
        if v_limit is not None:
            ids = ids.clamp(0, int(v_limit) - 1)
        try:
            idx_cpu, val_cpu = _forward_topk_batch(
                teacher,
                ids,
                mask,
                autocast_device=device if device.type == "cuda" else input_dev,
                k=k,
                vocab_limit=v_limit,
                value_dtype=value_dtype,
            )
        except Exception as exc:
            del ids, mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _is_oom(exc) and micro > 1:
                micro = max(micro // 2, 1)
                print(
                    f"  OOM at batch_size>{micro}; retrying with micro-batch={micro}"
                )
                continue
            raise
        del ids, mask
        idx_chunks.append(idx_cpu)
        val_chunks.append(val_cpu)
        del idx_cpu, val_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pbar.update(end - start)
        pbar.set_postfix_str(f"{end}/{n_samples} bs={micro}", refresh=False)
        start = end
    pbar.close()

    topk_indices = torch.cat(idx_chunks, dim=0)
    topk_values = torch.cat(val_chunks, dim=0)
    if tuple(topk_indices.shape) != (n_samples, seq_len, int(topk_indices.size(-1))):
        raise RuntimeError(
            f"unexpected topk_indices shape {tuple(topk_indices.shape)} "
            f"(expected {(n_samples, seq_len, 'K')})"
        )
    return topk_indices, topk_values


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pre-compute teacher Top-K logits for offline distillation "
            "(35B 4-bit + CPU offload)"
        )
    )
    p.add_argument(
        "--teacher-name",
        "--teacher_name",
        dest="teacher_name",
        type=str,
        default=DEFAULT_TEACHER,
        help=(
            "Teacher HF id or alias (default: Qwen/Qwen3.6-35B-Instruct → "
            "Qwen/Qwen3.6-35B-A3B)"
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
        help="Sequence length T (default 2048)",
    )
    p.add_argument("--max-samples-per-domain", type=int, default=DEFAULT_MAX_SAMPLES)
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Teacher micro-batch (default 1 at T=2048; try 2 if VRAM allows)",
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
    print("Extract teacher Top-K logits — Phase 1 (35B 4-bit + CPU offload)")
    print("=" * 72)
    print_device_info(device)

    requested_teacher = str(args.teacher_name)
    hub_teacher = resolve_teacher_hub_id(requested_teacher)
    print(
        f"teacher={requested_teacher} | hub={hub_teacher} | 4bit={teacher_4bit} | "
        f"dtype={compute_dtype} | store_values={value_dtype}"
    )
    print(
        f"K={int(args.kl_topk)} | T={int(args.max_length)} | "
        f"samples/domain={max_samples} | micro-batch={int(args.batch_size)}"
    )
    if teacher_4bit:
        print(
            f"max_memory GPU={args.gpu_max_memory} CPU={args.cpu_max_memory} "
            "(device_map=auto, ~6GiB weights on GPU, ~5.6GiB free for activations)"
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
        offload_folder=cache_dir / ".hf_offload",
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
        input_ids, attention_mask = tokenize_domain(
            messages_list,
            tokenizer,
            max_length=int(args.max_length),
            pad_id=pad_id,
        )
        n_tok = int(attention_mask.sum().item())
        print(
            f"[{domain}] tokenized N={int(input_ids.size(0)):,} "
            f"T={int(input_ids.size(1))} non-pad tokens={n_tok:,}"
        )

        topk_indices, topk_values = extract_topk_for_ids(
            teacher,
            input_ids,
            attention_mask,
            device=device,
            kl_topk=int(args.kl_topk),
            batch_size=int(args.batch_size),
            value_dtype=value_dtype,
            vocab_limit=vocab_t,
        )
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
                "device_map": "auto" if teacher_4bit else "none",
                "gpu_max_memory": str(args.gpu_max_memory),
                "cpu_max_memory": str(args.cpu_max_memory),
                "n_non_pad": n_tok,
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
