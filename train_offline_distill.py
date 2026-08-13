#!/usr/bin/env python3
"""Ultra-fast **offline** FFF student distillation (Phase 2, no teacher in RAM).

Reads precomputed Top-K teacher logits from ``data/logits_cache/``
(``extract_teacher_logits.py``, typically ``T=2048``, ``K=50``) and trains
**only** the FFF-SwiGLU student (``Qwen/Qwen3.5-7B``). Loss is Top-K KL + CE
− leaf entropy against cached ``topk_indices`` / ``topk_values``.

Student / optimizer
-------------------
* Base: ``Qwen/Qwen3.5-7B`` (hidden/MLP widths auto-detected from ``text_config``)
* CMM FFF: ``num_trees=12``, ``depth_per_tree=5`` (32 leaves / tree)
* STE soft routing, τ anneal ``1.0 → 0.10``
* Freeze non-FFF backbone + gradient checkpointing
* AdamW8bit on CUDA (AdamW FP32 on MPS/CPU)
* CUDA 12GB: 10GiB VRAM cap + CPU offload of frozen layers
* ``lr_leaf=2e-4``, ``lr_router=5e-5``, Cosine ``eta_min=2e-5``
* ``batch_size=1``, ``grad_accum_steps=4``, ``max_steps=4000``
* Device ``auto`` → CUDA / MPS / CPU

Pipeline
--------
    python extract_teacher_logits.py --device cuda --max-length 2048
    python train_offline_distill.py --device cuda
    python train_offline_distill.py --student_name Qwen/Qwen3.5-7B --smoke-test
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import (
    LOGITS_CACHE_DIR,
    OfflineDistillDataset,
    format_byte_count,
    logits_cache_dir_size,
)
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
    attn_implementation_name,
    bf16_autocast,
    build_fff_student,
    build_student_optimizer,
    build_student_param_groups,
    enable_fast_sdpa,
    enable_model_sdpa,
    gather_student_leaf_probs,
    infer_model_input_device,
    model_context_length,
    model_vocab_size,
    resolve_compute_dtype,
    topk_kl_distill_loss,
)
from fff_swiglu import (
    adapt_fff_swiglu_dims,
    get_text_config,
    iter_fff_swiglu_blocks,
    set_fff_temperature,
)

CHECKPOINT_NAME = "fff_offline_distill.pt"
DEFAULT_STUDENT = "Qwen/Qwen3.5-7B"
DEFAULT_NUM_TREES = 12
DEFAULT_DEPTH_PER_TREE = 5


@dataclass
class OfflineDistillConfig:
    """Hyperparameters matching online ``UltimateConfig`` (minus teacher)."""

    student_name: str = DEFAULT_STUDENT
    fff_depth: int = DEFAULT_DEPTH_PER_TREE
    num_trees: int = DEFAULT_NUM_TREES
    init_tau: float = 1.0
    min_tau: float = 0.10
    kl_temperature: float = 2.0
    entropy_coef: float = 0.01
    ce_coef: float = 0.1
    kl_topk: int = 50
    max_context_length: int = CONTEXT_LENGTH_256K
    batch_size: int = 1
    grad_accum_steps: int = 4
    lr_leaf: float = 2e-4
    lr_router: float = 5e-5
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 4000
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    device: str = "auto"
    checkpoint: str = CHECKPOINT_NAME
    logits_cache: str = str(LOGITS_CACHE_DIR)
    use_bf16: bool = True
    freeze_backbone: bool = True
    gradient_checkpointing: bool = True
    adam_8bit: bool = True
    equal_weight: bool = True
    smoke_test: bool = False
    gpu_max_memory: str = "10GiB"
    cpu_max_memory: str = "32GiB"


def build_argparser() -> argparse.ArgumentParser:
    d = OfflineDistillConfig()
    p = argparse.ArgumentParser(
        description="Offline Top-K KL distillation (student only, cached teacher logits)"
    )
    p.add_argument(
        "--student-name",
        "--student_name",
        dest="student_name",
        type=str,
        default=d.student_name,
        help="Student HF id (default: Qwen/Qwen3.5-7B)",
    )
    p.add_argument(
        "--fff-depth",
        "--depth-per-tree",
        "--depth_per_tree",
        dest="fff_depth",
        type=int,
        default=d.fff_depth,
        help="Per-tree FFF depth d (default 5 → 32 leaves/tree)",
    )
    p.add_argument(
        "--num-trees",
        "--num_trees",
        dest="num_trees",
        type=int,
        default=d.num_trees,
        help="CMM forest size K (default 12)",
    )
    p.add_argument("--init-tau", type=float, default=d.init_tau)
    p.add_argument("--min-tau", type=float, default=d.min_tau)
    p.add_argument("--kl-temperature", type=float, default=d.kl_temperature)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--ce-coef", type=float, default=d.ce_coef)
    p.add_argument(
        "--kl-topk",
        type=int,
        default=d.kl_topk,
        help="Must match K stored in the cache (default 50)",
    )
    p.add_argument("--max-context-length", type=int, default=d.max_context_length)
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
        "--device",
        type=str,
        default=d.device,
        help="auto (cuda>mps>cpu) | cuda | mps | cpu",
    )
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument(
        "--logits-cache",
        type=str,
        default=d.logits_cache,
        help="Directory of {domain}_logits.pt shards",
    )
    p.add_argument(
        "--gpu-max-memory",
        type=str,
        default=d.gpu_max_memory,
        help="CUDA VRAM cap for 7B student offload (default 10GiB)",
    )
    p.add_argument(
        "--cpu-max-memory",
        type=str,
        default=d.cpu_max_memory,
        help="CPU RAM budget for offloaded frozen layers (default 32GiB)",
    )
    p.add_argument("--fp32", action="store_true")
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
    p.add_argument(
        "--length-weighted",
        action="store_true",
        help="Concatenate shards instead of equal-weight domain mixing",
    )
    p.add_argument(
        "--smoke-test",
        "--fast-dev",
        "--smoke_test",
        "--fast_dev",
        dest="smoke_test",
        action="store_true",
        help="50 steps, log/save every 10 (pipeline check)",
    )
    return p


def _cycle_loader(loader: DataLoader) -> Iterator[dict[str, Any]]:
    """Infinite iterator over a finite ``DataLoader``."""
    while True:
        for batch in loader:
            yield batch


def _move_offline_batch(
    batch: dict[str, Any],
    device: torch.device,
    *,
    vocab_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Host → device: ``(B, T)`` ids/mask and ``(B, T, K)`` teacher Top-K.

    ``topk_indices`` are cast to int64 for ``torch.gather``; values to float32
    for the KL softmax. Indices are clamped to the student vocab.
    """
    non_blocking = pin_memory_for(device)
    input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
    attention_mask = batch["attention_mask"].to(device, non_blocking=non_blocking)
    topk_idx = batch["topk_indices"].to(
        device=device, dtype=torch.long, non_blocking=non_blocking
    )
    topk_val = batch["topk_values"].to(
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    input_ids = input_ids.clamp(0, vocab_size - 1)
    topk_idx = topk_idx.clamp(0, vocab_size - 1)
    return input_ids, attention_mask, topk_idx, topk_val


def main() -> None:
    args = build_argparser().parse_args()
    cfg = OfflineDistillConfig(
        student_name=args.student_name,
        fff_depth=args.fff_depth,
        num_trees=args.num_trees,
        init_tau=args.init_tau,
        min_tau=args.min_tau,
        kl_temperature=args.kl_temperature,
        entropy_coef=args.entropy_coef,
        ce_coef=args.ce_coef,
        kl_topk=args.kl_topk,
        max_context_length=args.max_context_length,
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
        device=args.device,
        checkpoint=args.checkpoint,
        logits_cache=args.logits_cache,
        use_bf16=not args.fp32,
        freeze_backbone=not args.unfreeze_backbone,
        gradient_checkpointing=True,
        adam_8bit=not args.adam_fp32,
        equal_weight=not args.length_weighted,
        smoke_test=bool(args.smoke_test),
        gpu_max_memory=str(args.gpu_max_memory),
        cpu_max_memory=str(args.cpu_max_memory),
    )
    if cfg.smoke_test:
        cfg.max_steps = 50
        cfg.save_every = 10
        cfg.log_every = 10
        print(
            "smoke_test / fast_dev: "
            f"max_steps={cfg.max_steps} log_every={cfg.log_every} "
            f"save_every={cfg.save_every}"
        )

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    enable_fast_sdpa()

    if cfg.adam_8bit and not torch.cuda.is_available():
        cfg.adam_8bit = False
        print(
            "CUDA not available. Auto-disabling 8-bit AdamW "
            "(using full-precision AdamW)."
        )

    compute_dtype = resolve_compute_dtype(device, use_bf16=cfg.use_bf16)
    n_leaves = 1 << cfg.fff_depth
    h_uniform = math.log(float(n_leaves))

    print("=" * 72)
    print("Offline FFF Distill — Qwen3.5-7B CMM student (cached teacher Top-K)")
    print("=" * 72)
    print_device_info(device)
    print(f"config: {asdict(cfg)}")
    print(
        f"student_dtype={compute_dtype} | kl_topk={cfg.kl_topk} | "
        f"grad_accum={cfg.grad_accum_steps} | "
        f"CMM K={cfg.num_trees} d={cfg.fff_depth} leaves/tree={n_leaves}"
    )
    print(
        f"target leaf entropy H_uniform=log({n_leaves})={h_uniform:.4f} "
        f"(per tree; freeze_backbone={cfg.freeze_backbone} "
        f"grad_ckpt={cfg.gradient_checkpointing})"
    )

    cache_dir = Path(cfg.logits_cache)
    cache_bytes = logits_cache_dir_size(cache_dir)
    print(
        f"\nLoading OfflineDistillDataset from {cache_dir} "
        f"({format_byte_count(cache_bytes)} on disk) ..."
    )
    dataset = OfflineDistillDataset(
        cache_dir,
        equal_weight=cfg.equal_weight,
    )
    if dataset.kl_topk is not None and int(dataset.kl_topk) != int(cfg.kl_topk):
        print(
            f"warning: cache K={dataset.kl_topk} != --kl-topk {cfg.kl_topk}; "
            "using cached K for gather"
        )
        cfg.kl_topk = int(dataset.kl_topk)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory_for(device),
        drop_last=False,
    )
    print(
        f"dataset len={len(dataset):,} (raw={dataset.num_raw_samples:,}) | "
        f"domains={dataset.domains} | T={dataset.max_length} | K={dataset.kl_topk}"
    )

    print("\nLoading student only (teacher is NOT loaded) ...")
    student = build_fff_student(
        cfg.student_name,
        device,
        fff_depth=cfg.fff_depth,
        num_trees=cfg.num_trees,
        init_temp=cfg.init_tau,
        compute_dtype=compute_dtype,
        freeze_backbone=cfg.freeze_backbone,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gpu_max_memory=cfg.gpu_max_memory,
        cpu_max_memory=cfg.cpu_max_memory,
    )
    apply_context_length(student, n_ctx=cfg.max_context_length)
    enable_model_sdpa(student)
    input_device = infer_model_input_device(student, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    text_cfg = get_text_config(student)
    d_model, used_i, n_leaves_chk, slice_size = adapt_fff_swiglu_dims(
        int(text_cfg.hidden_size),
        int(text_cfg.intermediate_size),
        num_trees=cfg.num_trees,
        depth_per_tree=cfg.fff_depth,
    )
    print(
        f"context_length={model_context_length(student):,} | "
        f"attention=SDPA student={attn_implementation_name(student)} | "
        f"input_device={input_device}"
    )
    print(
        f"FFF dims: D={d_model} I={int(text_cfg.intermediate_size)}→{used_i} "
        f"K={cfg.num_trees} d={cfg.fff_depth} "
        f"leaves/tree={n_leaves_chk} slice={slice_size}"
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

    vocab_s = model_vocab_size(student)
    pad_id = int(dataset.pad_id)
    data_iter = _cycle_loader(loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running: dict[str, float] = {
        "kl": 0.0,
        "entropy": 0.0,
        "ce": 0.0,
        "total": 0.0,
    }
    log_count = 0
    ckpt_path = Path(cfg.checkpoint)

    print(
        f"lr_leaf={cfg.lr_leaf:.2e} lr_router={cfg.lr_router:.2e} | "
        f"CosineAnnealingLR T_max={n_opt_steps} eta_min={cfg.min_lr:.2e}"
    )

    pbar = tqdm(range(cfg.max_steps), desc="offline-distill", dynamic_ncols=True)
    for step in pbar:
        tau = annealed_tau(step, cfg.max_steps, cfg.init_tau, cfg.min_tau)
        set_fff_temperature(student, tau)
        lr_leaf = float(optimizer.param_groups[0]["lr"])
        lr_router = float(optimizer.param_groups[-1]["lr"])

        batch = next(data_iter)
        input_ids, attention_mask, topk_idx, topk_val = _move_offline_batch(
            batch, input_device, vocab_size=vocab_s
        )

        with bf16_autocast(device):
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            student_logits = student_out.logits
            del student_out
            routing = gather_student_leaf_probs(student)
            loss, parts = topk_kl_distill_loss(
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
        parts = {k: v.detach() for k, v in parts.items()}
        del student_logits, loss, topk_val, topk_idx, routing, input_ids, attention_mask

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
                "num_trees": cfg.num_trees,
                "depth_per_tree": cfg.fff_depth,
                "model_name": cfg.student_name,
                "teacher_name": dataset.teacher_name,
                "architecture": "fff_swiglu_offline_distill",
                "domains": list(dataset.domains),
                "compute_dtype": str(compute_dtype),
                "max_context_length": int(cfg.max_context_length),
                "offline": True,
                "kl_topk": int(cfg.kl_topk),
            }
            torch.save(payload, ckpt_path)
            tqdm.write(f"saved checkpoint → {ckpt_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60.0:.1f} min. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
