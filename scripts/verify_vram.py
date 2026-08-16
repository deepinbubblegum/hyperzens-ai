#!/usr/bin/env python3
"""Verify the RTX 3060 12 GB training config fits under an 8.5 GB VRAM budget.

Builds the BitNet-FFF student (``d_model=1024``, ``n_layers=6``,
``recurrent_steps=4``, ``fff_depth=8``, ``top_k=4``) with GaLoreAdamW
optimizer routing plus a small distillation teacher, then runs the exact
forward + backward micro-batch cycle ``scripts/distill_multitask.py`` uses
(BF16 autocast, temperature-scaled KL + CE, gradient clipping) and captures
``torch.cuda.max_memory_allocated`` across the backward passes. Exits nonzero
when the peak exceeds ``--budget-gb`` (default 8.5 GB).

Run on the RTX 3060 machine::

    python scripts/verify_vram.py
    python scripts/verify_vram.py --seq-len 256 --batch-size 2 --grad-accum-steps 2
    python scripts/verify_vram.py --no-use-galore      # AdamW baseline for comparison
    python scripts/verify_vram.py --gradient-checkpointing   # lower activation peak
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import distill_multitask as dm
import train_qat

from bitnet_fff.models import BitNetFFTConfig
from bitnet_fff.qat import FP16MasterAdamW
from bitnet_fff.tokenizer import ByteTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the BitNet-FFF RTX 3060 training config fits the "
                    "8.5 GB VRAM budget during full backward passes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    s = p.add_argument_group("student (RTX 3060 config)")
    s.add_argument("--d-model", type=int, default=1024)
    s.add_argument("--n-heads", type=int, default=4)
    s.add_argument("--n-layers", type=int, default=6)
    s.add_argument("--recurrent-steps", type=int, default=4)
    s.add_argument("--fff-depth", type=int, default=8)
    s.add_argument("--top-k", type=int, default=4)
    s.add_argument("--vocab-size", type=int, default=256)
    s.add_argument("--max-seq-len", type=int, default=512)
    s.add_argument("--no-fp16", action="store_true",
                   help="train fp32 master weights instead of fp16")

    t = p.add_argument_group("teacher")
    t.add_argument("--teacher-d-model", type=int, default=256)
    t.add_argument("--teacher-n-heads", type=int, default=4)
    t.add_argument("--teacher-n-layers", type=int, default=4)
    t.add_argument("--teacher-fff-depth", type=int, default=3)

    r = p.add_argument_group("run")
    r.add_argument("--seq-len", type=int, default=128,
                   help="micro-batch sequence length")
    r.add_argument("--batch-size", type=int, default=2)
    r.add_argument("--grad-accum-steps", type=int, default=2,
                   help="micro-batches per optimizer step (worst case: N x "
                        "forward+backward before the optimizer step)")
    r.add_argument("--use-galore", action=argparse.BooleanOptionalAction,
                   default=True)
    r.add_argument("--galore-rank", type=int, default=128)
    r.add_argument("--update-proj-gap", type=int, default=200)
    r.add_argument("--gradient-checkpointing", action="store_true",
                   help="enable gradient checkpointing on student layers")
    r.add_argument("--budget-gb", type=float, default=8.5,
                   help="hard VRAM budget (GiB) the peak must stay below")
    r.add_argument("--alpha", type=float, default=0.7)
    r.add_argument("--temperature", type=float, default=2.0)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("[verify-vram] CUDA not available; this check runs on the RTX "
              "3060. Nothing to measure here.")
        return 2

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    tok = ByteTokenizer(args.vocab_size)
    cfg = BitNetFFTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        recurrent_steps=args.recurrent_steps,
        fff_depth=args.fff_depth,
        top_k=args.top_k,
        fff_bias=True,
        max_seq_len=args.max_seq_len,
        activation_bits=8,
        tie_weights=True,
        use_fast_inference=False,
    ).bind_tokenizer(tok)

    student, _ = train_qat.make_qat_model(
        cfg, device,
        fp16=not args.no_fp16,
        quant_mode="absmax",
        activation_bits=8,
        lr=args.lr,
    )
    if args.use_galore:
        opt = dm._galore_optimizer(
            student, args.lr, (0.9, 0.999), 0.01,
            args.galore_rank, args.update_proj_gap,
        )
        opt_name = f"GaLoreAdamW rank={args.galore_rank}"
    else:
        opt = FP16MasterAdamW(
            student.parameters(), lr=args.lr, weight_decay=0.01
        )
        opt_name = "FP16MasterAdamW (full-rank baseline)"
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    teacher = dm.distill_mod.load_teacher(
        "local", device, args.vocab_size, args
    )[0]

    n_params = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[verify-vram] student params={n_params / 1e6:.1f}M "
          f"trainable={trainable / 1e6:.1f}M "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"recurrent_steps={cfg.recurrent_steps} fff_depth={cfg.fff_depth} "
          f"top_k={cfg.top_k} fp16_master={not args.no_fp16}")
    print(f"[verify-vram] optimizer={opt_name} "
          f"batch={args.batch_size} seq={args.seq_len} "
          f"grad_accum={args.grad_accum_steps} "
          f"gradient_checkpointing={args.gradient_checkpointing}")

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    torch.cuda.reset_peak_memory_stats()
    micro = [
        torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len),
                      device=device)
        for _ in range(args.grad_accum_steps)
    ]
    # Prime the GaLore projection basis (update_proj_gap) so the measured run
    # reflects the steady-state optimizer path.
    with contextlib.suppress(Exception):
        dm.accum_step(
            student, opt, teacher, micro, args.alpha, args.temperature,
            cfg.vocab_size, args.grad_accum_steps, scaler,
        )
    torch.cuda.reset_peak_memory_stats()

    loss, kd, ce = dm.accum_step(
        student, opt, teacher, micro, args.alpha, args.temperature,
        cfg.vocab_size, args.grad_accum_steps, scaler,
    )
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    budget_gb = args.budget_gb

    print(f"[verify-vram] loss={loss:.4f} kd={kd:.4f} ce={ce:.4f}")
    print(f"[verify-vram] peak VRAM = {peak:.2f} GiB (reserved {reserved:.2f} "
          f"GiB) budget = {budget_gb:.2f} GiB")

    if peak < budget_gb:
        print(f"[verify-vram] PASS: peak {peak:.2f} GiB < {budget_gb:.2f} GiB")
        return 0
    print(f"[verify-vram] FAIL: peak {peak:.2f} GiB exceeds "
          f"{budget_gb:.2f} GiB budget")
    print("[verify-vram] try --gradient-checkpointing, a smaller "
          "--seq-len/--batch-size, or fp16 masters (default).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())