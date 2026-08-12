#!/usr/bin/env python3
"""Knowledge distillation: Gemma (GeGLU) → FFF-GeGLU Student.

Teacher
-------
Frozen HuggingFace Gemma CausalLM (e.g. ``google/gemma-2-2b``). Gemma uses
GeGLU MLPs: ``y = (GELU_tanh(x W_gate) ⊙ (x W_up)) W_down``.

Student
-------
Same trunk with each ``mlp`` replaced by :class:`FFFGemmaBlock`
(smart-init from Teacher GeGLU slices + learnable ``output_scale ≈ √L``).
Training uses **STE** hard-aware soft routing (depth=4 → 16 leaves).

Init safety
-----------
Leaf / router init runs in **float32** (avoids LAPACK ``geqrf`` on Half),
then teacher + student are cast to **bfloat16** for the CUDA distill loop.

Loss / schedule
---------------
Same objective as Qwen distill:
``L = KL(student ‖ teacher)·T² + λ_feat · MSE(RMSNorm_h) − λ_H · H(leaf)``
with Cosine Annealing, ``lr_leaf=2e-4``, ``lr_router=5e-5``, ``max_steps=3000``.

Checkpoint
----------
``gemma_fff_distill.pt``

Example
-------
    python distill_gemma.py --device cuda --max-steps 3000
    python distill_gemma.py --model-name google/gemma-2-2b --max-steps 50
    # Gated models require: huggingface-cli login
"""

from __future__ import annotations

import argparse
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from device_utils import (
    apply_hardware_optimizations,
    pin_memory_for,
    print_device_info,
    resolve_device,
)
from distill_modern_llm import (
    DistillLossWeights,
    _RMSNormCapture,
    bf16_autocast,
    build_student_param_groups,
    distillation_loss,
    gather_student_leaf_probs,
    resolve_compute_dtype,
    student_forward_capture,
    teacher_forward_capture,
)
from fff_distill import (
    TokenChunkDataset,
    annealed_tau,
    load_wikitext_tokens,
)
from fff_modern_llm import (
    iter_fff_swiglu_blocks,
    patch_model_with_fff_gemma,
    set_fff_routing_mode,
    set_fff_temperature,
)

CHECKPOINT_NAME = "gemma_fff_distill.pt"
DEFAULT_TEACHER = "google/gemma-2-2b"


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_gemma_teacher(model_name: str, device: torch.device) -> Any:
    """Load Gemma CausalLM in FP32, freeze, eval (caller casts dtype/device)."""
    AutoModelForCausalLM, _ = _require_transformers()
    print(f"Loading teacher `{model_name}` (float32, frozen) ...")
    print("  note: gated Gemma weights need `huggingface-cli login`")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_fff_student_from_gemma(
    teacher: Any,
    *,
    fff_depth: int = 4,
    init_temp: float = 1.0,
) -> Any:
    """Deep-copy Teacher, unfreeze, replace every ``mlp`` with FFF GeGLU.

    Patching / smart-init run in **float32** (safe ``orthogonal_`` / slice).
    Caller casts to BF16 after this returns when training on CUDA.
    """
    print("Cloning teacher → student + FFF GeGLU (FFFGemmaBlock) injection ...")
    student = deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    n = patch_model_with_fff_gemma(
        student,
        fff_depth=fff_depth,
        init_temp=init_temp,
    )
    n_leaves = 1 << fff_depth
    intermediate = int(teacher.config.intermediate_size)
    slice_dim = intermediate // n_leaves
    scale0 = math.sqrt(float(n_leaves))
    print(
        f"  patched {n} MLPs | STE soft train | "
        f"slice_dim=I/{n_leaves}={slice_dim} | "
        f"output_scale init=√L={scale0:.4f}"
    )
    set_fff_routing_mode(student, "soft")
    return student


@dataclass
class DistillConfig:
    model_name: str = DEFAULT_TEACHER
    fff_depth: int = 4
    init_tau: float = 1.0
    min_tau: float = 0.10
    kl_temperature: float = 2.0
    feature_coef: float = 0.5
    entropy_coef: float = 0.01
    block_size: int = 128
    batch_size: int = 1
    grad_accum_steps: int = 8
    lr_leaf: float = 2e-4
    lr_router: float = 5e-5
    min_lr: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 3000
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    max_train_chars: int = 1_500_000
    device: str = "cuda"
    checkpoint: str = CHECKPOINT_NAME
    use_bf16: bool = True


def build_argparser() -> argparse.ArgumentParser:
    d = DistillConfig()
    p = argparse.ArgumentParser(
        description="Gemma → FFF-GeGLU knowledge distillation"
    )
    p.add_argument("--model-name", type=str, default=d.model_name)
    p.add_argument("--fff-depth", type=int, default=d.fff_depth)
    p.add_argument("--init-tau", type=float, default=d.init_tau)
    p.add_argument("--min-tau", type=float, default=d.min_tau)
    p.add_argument("--kl-temperature", type=float, default=d.kl_temperature)
    p.add_argument("--feature-coef", type=float, default=d.feature_coef)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--block-size", type=int, default=d.block_size)
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
    p.add_argument("--max-train-chars", type=int, default=d.max_train_chars)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument(
        "--fp32",
        action="store_true",
        help="Disable BF16 (train in float32)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DistillConfig(
        model_name=args.model_name,
        fff_depth=args.fff_depth,
        init_tau=args.init_tau,
        min_tau=args.min_tau,
        kl_temperature=args.kl_temperature,
        feature_coef=args.feature_coef,
        entropy_coef=args.entropy_coef,
        block_size=args.block_size,
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
        max_train_chars=args.max_train_chars,
        device=args.device,
        checkpoint=args.checkpoint,
        use_bf16=not args.fp32,
    )

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    compute_dtype = resolve_compute_dtype(device, use_bf16=cfg.use_bf16)

    n_leaves = 1 << cfg.fff_depth
    h_uniform = math.log(float(n_leaves))
    print("=" * 72)
    print("Gemma Distill — Teacher → FFF-GeGLU Student (STE)")
    print("=" * 72)
    print_device_info(device)
    print(f"config: {asdict(cfg)}")
    print(f"compute_dtype={compute_dtype} (no GradScaler)")
    print(f"target leaf entropy H_uniform=log({n_leaves})={h_uniform:.4f}")

    _, AutoTokenizer = _require_transformers()
    print("\nLoading tokenizer / WikiText-2 ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = load_wikitext_tokens(
        tokenizer, split="train", max_chars=cfg.max_train_chars
    )
    dataset = TokenChunkDataset(tokens, cfg.block_size)
    if len(dataset) == 0:
        raise SystemExit("empty dataset — increase --max-train-chars")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory_for(device),
    )
    print(f"train chunks={len(dataset)} block_size={cfg.block_size}")

    teacher = load_gemma_teacher(cfg.model_name, device)
    student = build_fff_student_from_gemma(
        teacher,
        fff_depth=cfg.fff_depth,
        init_temp=cfg.init_tau,
    )

    # Cast AFTER FFF FP32 init (avoids Half/BF16 geqrf during orthogonal_).
    if compute_dtype != torch.float32:
        print(f"Casting teacher/student → {compute_dtype} ...")
        teacher = teacher.to(dtype=compute_dtype)
        student = student.to(dtype=compute_dtype)
    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    student.train()
    set_fff_routing_mode(student, "soft")

    n_fff = sum(1 for _ in iter_fff_swiglu_blocks(student))
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"FFF GeGLU blocks={n_fff} | student trainable={n_train:,}")

    loss_w = DistillLossWeights(
        kl_temperature=cfg.kl_temperature,
        feature_coef=cfg.feature_coef,
        entropy_coef=cfg.entropy_coef,
    )
    teacher_cap = _RMSNormCapture(teacher)
    student_cap = _RMSNormCapture(student)

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
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running: dict[str, float] = {
        "kl": 0.0,
        "feature": 0.0,
        "entropy": 0.0,
        "total": 0.0,
    }
    log_count = 0

    print(
        f"optimizer: AdamW leaf/other lr={cfg.lr_leaf:.2e} "
        f"router lr={cfg.lr_router:.2e} | "
        f"CosineAnnealingLR T_max={n_opt_steps} (opt steps) eta_min={cfg.min_lr:.2e}"
    )

    pbar = tqdm(range(cfg.max_steps), desc="distill-gemma", dynamic_ncols=True)
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

        vocab = int(student.config.vocab_size)
        input_ids = input_ids.clamp(0, vocab - 1)

        with bf16_autocast(device):
            teacher_logits, teacher_rms = teacher_forward_capture(
                teacher, input_ids, teacher_cap
            )
            student_logits, student_rms = student_forward_capture(
                student, input_ids, student_cap
            )
            routing = gather_student_leaf_probs(student)
            loss, parts = distillation_loss(
                student_logits,
                teacher_logits.detach(),
                student_rms,
                teacher_rms.detach(),
                routing,
                loss_w,
            )
            loss = loss / float(cfg.grad_accum_steps)

        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            running[k] += float(parts[k].detach().item())
        log_count += 1

        if (step + 1) % cfg.log_every == 0 or step == 0:
            inv = 1.0 / max(log_count, 1)
            msg = (
                f"step={step + 1}/{cfg.max_steps} "
                f"loss={running['total'] * inv:.4f} "
                f"kl={running['kl'] * inv:.4f} "
                f"feat={running['feature'] * inv:.4f} "
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
                "model_name": cfg.model_name,
                "architecture": "fff_geglu_gemma",
                "compute_dtype": str(compute_dtype),
            }
            torch.save(payload, ckpt_path)
            tqdm.write(f"saved checkpoint → {ckpt_path}")

    teacher_cap.close()
    student_cap.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60.0:.1f} min. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
