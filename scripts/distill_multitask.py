#!/usr/bin/env python3
"""Multi-task data-mixture distillation for an all-rounder BitNet-FFF student.

Distills logits from a frozen Teacher into a small QAT-trained
:class:`BitNetFFTTransformer` student on a mixture of skill areas streamed by
:class:`~bitnet_fff.dataset.MultiTaskStreamer`:

* Reasoning/Math (40%): ``open-r1/OpenR1-Math-220k``
* General instruction (40%): ``Open-Orca/SlimOrca-Dedup``
* Code/Logic (20%): ``m-a-p/CodeFeedback``

The loss blends temperature-scaled KL divergence (teacher) with hard
next-token cross-entropy, as in ``distill.py``, but the KD term is re-scaled by
``alpha * T^2 / vocab_size`` (see :func:`accum_step`) so the total loss stays
in a normal ~1-20 range, and gradients are clipped to norm 1.0 before each
optimizer step. Three extra levers are built in for memory-frugal all-rounder
training:

* ``--tie-weights`` is **on by default** (the output head shares the input
  embedding), cutting the student's trainable parameter count.
* ``--grad-accum-steps`` accumulates gradients over N micro-batches before a
  single optimizer step, so an effective batch size of 32-64 is reached
  without ever holding that many forward activations on the MPS GPU.
* Live per-task validation loss is reported every ``--val-every`` steps
  (default 500), each task streamed fresh from its own split
  (``validation -> test -> train`` fallback).

Usage:
    python scripts/distill_multitask.py --device mps --grad-accum-steps 4 \
        --batch-size 8 --val-every 500 --checkpoint ckpts/mt_student.pt
    python scripts/distill_multitask.py --teacher hf://Qwen/Qwen2.5-0.5B \
        --tasks math,code --weights 0.6,0.4 --steps 20000 --device mps
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_qat
import distill as distill_mod

from bitnet_fff.dataset import MultiTaskStreamer, multi_task_recipe
from bitnet_fff.models import BitNetFFTConfig
from bitnet_fff.mps_utils import (
    is_mps_available,
    mps_driver_allocated_bytes,
    mps_empty_cache,
)
from bitnet_fff.tokenizer import TikTokenizer

from tqdm import tqdm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task data-mixture distillation for BitNet-FFF "
                    "(KL + CE, weight-tied student, gradient accumulation, "
                    "per-task validation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m = p.add_argument_group("student model")
    m.add_argument("--d-model", type=int, default=128)
    m.add_argument("--n-heads", type=int, default=4)
    m.add_argument("--n-layers", type=int, default=2)
    m.add_argument("--fff-depth", type=int, default=3)
    m.add_argument("--vocab-size", type=int, default=256)
    m.add_argument("--max-seq-len", type=int, default=64)
    m.add_argument("--activation-bits", type=int, default=8)
    m.add_argument("--attention-activation-bits", type=int, default=None)
    m.add_argument("--router-rank", choices=("full", "r1"), default="full")
    m.add_argument("--no-fff-bias", action="store_true")
    m.add_argument("--tie-weights", action="store_true", default=True,
                   help="tie the output head to the input embedding "
                        "(enabled by default)")
    m.add_argument("--no-tie-weights", action="store_true",
                   help="disable weight tying (overrides --tie-weights)")

    tm = p.add_argument_group("teacher")
    tm.add_argument("--teacher", default="local",
                    help="hf://<name>, a train_qat checkpoint path, or 'local'")
    tm.add_argument("--teacher-d-model", type=int, default=256)
    tm.add_argument("--teacher-n-heads", type=int, default=4)
    tm.add_argument("--teacher-n-layers", type=int, default=4)
    tm.add_argument("--teacher-fff-depth", type=int, default=3)

    k = p.add_argument_group("distillation")
    k.add_argument("--alpha", type=float, default=0.7,
                   help="KL weight (1-alpha is the hard-label CE weight)")
    k.add_argument("--temperature", type=float, default=2.0)
    k.add_argument("--no-fp16", action="store_true")

    o = p.add_argument_group("optimizer / QAT")
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--warmup-steps", type=int, default=100,
                   help="Number of warmup steps (linear LR ramp 0 -> --lr)")
    o.add_argument("--beta1", type=float, default=0.9)
    o.add_argument("--beta2", type=float, default=0.999)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--clip-grad-norm", type=float, default=1.0)
    o.add_argument("--quant-mode", choices=("absmax", "per_channel", "ema", "learned"),
                   default="absmax")

    d = p.add_argument_group("data mixture")
    d.add_argument("--tasks", default="math,instruct,code",
                   help="comma-separated task subset of "
                        "math|instruct|code (or 'all')")
    d.add_argument("--weights", default=None,
                   help="comma-separated sampling ratios matching --tasks "
                        "(defaults to the recipe: 0.4,0.4,0.2)")
    d.add_argument("--tokenizer", default=None,
                   help="tokenizer name (default gpt2 BPE; 'bytes' for the "
                        "byte-level fallback; 'o200k_base' for the fixed-vocab "
                        "tiktoken encoding)")
    d.add_argument("--seq-len", type=int, default=64)
    d.add_argument("--batch-size", type=int, default=8,
                   help="micro-batch size per forward")
    d.add_argument("--grad-accum-steps", type=int, default=4,
                   help="micro-batches per optimizer step (effective batch = "
                        "batch_size * grad_accum_steps)")
    d.add_argument("--max-examples", type=int, default=None,
                   help="cap streamed examples per task")

    v = p.add_argument_group("validation")
    v.add_argument("--val-every", type=int, default=500,
                   help="run per-task validation every N steps (0 disables)")
    v.add_argument("--val-split", default="validation",
                   help="validation split (falls back test -> train)")
    v.add_argument("--val-batches", type=int, default=8,
                   help="max validation batches per task per run")
    v.add_argument("--val-batch-size", type=int, default=None,
                   help="validation batch size (defaults to --batch-size)")

    r = p.add_argument_group("run")
    r.add_argument("--steps", type=int, default=2000)
    r.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--compile", action="store_true",
                   help="wrap the student with torch.compile before training")
    r.add_argument("--log-every", type=int, default=10)
    r.add_argument("--save-every", type=int, default=500)
    r.add_argument("--empty-cache-every", type=int, default=100,
                   help="release MPS cached blocks every N steps")
    r.add_argument("--checkpoint", default=None,
                   help="where to save the distilled student (torch.save)")
    r.add_argument("--resume", default=None,
                   help="checkpoint to resume from (step, optimizer, best state)")
    return p.parse_args(argv)


def _resolve_device(device: str | None) -> torch.device:
    """Pick the compute device: explicit flag, else CUDA -> MPS -> CPU."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_autocast(device: torch.device) -> object:
    """BF16 autocast context for CUDA; a no-op on MPS/CPU.

    Uses ``torch.amp.autocast('cuda', dtype=torch.bfloat16)`` so the forward pass
    and loss computation run in native bfloat16 on CUDA (Ampere Tensor Cores),
    while MPS/CPU keep their existing FP32/FP16-master path untouched.
    """
    if getattr(device, "type", None) == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _amp_step(scaler, optimizer, model, max_norm: float = 1.0) -> None:
    """Unscale -> clip -> step with GradScaler semantics (or direct step for BF16/CPU/MPS).

    For BF16 on CUDA, or CPU/MPS, GradScaler is disabled because BF16 shares the
    same 8-bit dynamic range as FP32 (no scaling needed). Gradients are clipped
    and the optimizer steps directly.
    """
    params = list(model.parameters()) if hasattr(model, "parameters") else list(model)
    if scaler is None or not scaler.is_enabled():
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        optimizer.step()
        return

    if hasattr(scaler, "unscale_"):
        scaler.unscale_(optimizer)
    elif hasattr(scaler, "get_scale"):
        scale = scaler.get_scale()
        inv_scale = 1.0 / scale
        found_inf = False
        for p in params:
            if p.grad is not None:
                p.grad.mul_(inv_scale)
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    found_inf = True
        if found_inf:
            scaler.update(new_scale=scale * 0.5)
            return

    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    if hasattr(scaler, "step"):
        scaler.step(optimizer)
    else:
        optimizer.step()

    if hasattr(scaler, "update"):
        scaler.update()


def accum_step(
    student,
    optimizer,
    teacher: object,
    batches: list[torch.Tensor],
    alpha: float,
    temperature: float,
    vocab: int,
    grad_accum: int,
    scaler=None,
) -> tuple[float, float, float]:
    """One optimizer step over ``grad_accum`` micro-batches.

    Gradients from each micro-batch are scaled by ``1 / grad_accum`` and
    accumulated (only one micro-batch's activations live at a time), giving an
    effective batch of ``len(batches) * batch_size`` without the corresponding
    MPS VRAM spike. Returns ``(loss, kd, ce)`` averaged over the micro-batches.

    **Loss scaling** — ``distill.distillation_loss`` returns a temperature-
    squared batchmean KL (``kd = KL * T^2``); that is re-scaled by
    ``alpha * T^2 / vocab_size`` before blending with the hard-label CE, so the
    KD term lands on the same order of magnitude as the CE and the total loss
    stays in a normal ~1-20 range instead of being dominated by the KL.

    **AMP** — on CUDA the forward pass and loss computation run under BF16
    ``autocast`` (GradScaler disabled as BF16 matches FP32 dynamic range); on
    MPS/CPU the step is a plain clip-then-step.
    """
    if not batches:
        raise RuntimeError("accum_step requires at least one micro-batch")

    # zero_grad is called only at the start of the accumulation cycle
    optimizer.zero_grad()
    loss_acc = kd_acc = ce_acc = 0.0
    for batch in batches:
        with _amp_autocast(batch.device):
            t_logits = distill_mod.teacher_logits(teacher, batch)
            s_logits = student(batch)
            _, kd, ce = distill_mod.distillation_loss(
                s_logits, t_logits, batch, alpha, temperature, vocab
            )
            # distillation_loss returns kd = KL_batchmean * T^2, so the scaled KD
            # term (alpha * T^2 / vocab) * KL_batchmean == (alpha / vocab) * kd.
            kd = (alpha / vocab) * kd
            loss = kd + (1.0 - alpha) * ce
        scaled_loss = loss / grad_accum
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        loss_acc += float(loss.item())
        kd_acc += float(kd.item())
        ce_acc += float(ce.item())

    # _amp_step is called only at the end of the gradient accumulation cycle
    params = list(student.parameters())
    _amp_step(scaler, optimizer, params, max_norm=1.0)
    n = len(batches)
    return loss_acc / n, kd_acc / n, ce_acc / n


def validate_tasks(
    student,
    val_streamers: dict[str, MultiTaskStreamer],
    device: torch.device,
    max_batches: int,
    weights: dict[str, float],
) -> tuple[dict[str, float | None], float]:
    """Per-task mean next-token CE over each task's validation stream.

    Returns ``(metrics, blended)`` where ``blended`` is the weight-normalized
    mean over tasks that yielded at least one batch (``inf`` if none did).
    """
    was_training = student.training
    student.eval()
    try:
        metrics: dict[str, float | None] = {}
        for name, streamer in val_streamers.items():
            metrics[name] = train_qat.validate(
                student, streamer, device, max_batches
            )
    finally:
        student.train(was_training)

    present = {k: v for k, v in metrics.items() if v is not None}
    if not present:
        return metrics, float("inf")
    blended = sum(weights[k] * present[k] for k in present) / sum(
        weights[k] for k in present
    )
    return metrics, float(blended)


def _select_tasks(args: argparse.Namespace) -> list:
    recipe = {t.name: t for t in multi_task_recipe()}
    if args.tasks.strip().lower() == "all":
        names = list(recipe)
    else:
        names = [n.strip() for n in args.tasks.split(",") if n.strip()]
    bad = [n for n in names if n not in recipe]
    if bad:
        raise SystemExit(
            f"unknown task(s) {bad}; available: {sorted(recipe)}"
        )
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(names):
            raise SystemExit(
                f"--weights ({len(weights)} values) must match --tasks "
                f"({len(names)} tasks)"
            )
        tasks = [
            dataclasses.replace(recipe[n], weight=w)
            for n, w in zip(names, weights)
        ]
    else:
        tasks = [recipe[n] for n in names]
    if not tasks:
        raise SystemExit("no tasks selected")
    return tasks


class LinearWarmupScheduler:
    """Linear LR warmup for the custom FP16-master AdamW optimizer.

    ``torch.optim.lr_scheduler.LambdaLR`` cannot drive the optimizer returned
    by :func:`train_qat.make_qat_model` (``FP16MasterAdamW`` is not a
    ``torch.optim.Optimizer`` subclass: it has no ``param_groups``/``defaults``),
    so this mirrors LambdaLR's linear warmup by scaling the optimizer's ``lr``
    attribute directly: the learning rate ramps linearly from 0 to the base
    ``--lr`` over ``warmup_steps`` optimizer steps and then stays constant.
    Call :meth:`step` once per optimizer step; the state is checkpointed so a
    resumed run continues the warmup instead of restarting it.
    """

    def __init__(self, optimizer, warmup_steps: int, last_step: int = 0) -> None:
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.last_step = int(last_step)
        optimizer.lr = self._lr_for(self.last_step)

    def step(self) -> None:
        self.last_step += 1
        self.optimizer.lr = self._lr_for(self.last_step)

    def get_last_lr(self) -> list[float]:
        return [self.optimizer.lr]

    def _lr_for(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.base_lr
        return self.base_lr * min(1.0, step / self.warmup_steps)

    def state_dict(self) -> dict:
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.last_step = int(state.get("last_step", self.last_step))
        self.optimizer.lr = self._lr_for(self.last_step)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1")
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be >= 2 (distillation needs labels)")
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    tok = train_qat.load_tokenizer(args.tokenizer, args.vocab_size)
    if isinstance(tok, train_qat.BPETokenizer):
        print(f"[tokenizer] BPE {tok.name} vocab={tok.vocab_size}")
    elif isinstance(tok, TikTokenizer):
        print(f"[tokenizer] tiktoken {tok.name} vocab={tok.vocab_size}")
    else:
        print(f"[tokenizer] byte-level fallback vocab={tok.vocab_size}")

    student_vocab = args.vocab_size
    if isinstance(tok, (train_qat.BPETokenizer, TikTokenizer)):
        student_vocab = int(tok.vocab_size)
    teacher, teacher_meta = distill_mod.load_teacher(
        args.teacher, device, student_vocab, args
    )
    print(f"[teacher] {teacher_meta.get('type')}: "
          f"{teacher_meta.get('name', teacher_meta.get('path', 'local'))}")
    if teacher_meta.get("type") == "hf":
        student_vocab = teacher_meta["vocab_size"]
    if args.vocab_size != student_vocab:
        print(f"[student] forcing vocab_size={student_vocab}")
        args.vocab_size = student_vocab

    tie = args.tie_weights and not args.no_tie_weights
    cfg = BitNetFFTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        fff_depth=args.fff_depth,
        fff_bias=not args.no_fff_bias,
        max_seq_len=args.max_seq_len,
        activation_bits=args.activation_bits,
        attention_activation_bits=args.attention_activation_bits,
        router_rank=args.router_rank,
        tie_weights=tie,
        use_fast_inference=False,
    ).bind_tokenizer(tok)
    student, opt = train_qat.make_qat_model(
        cfg, device,
        fp16=not args.no_fp16,
        quant_mode=args.quant_mode,
        activation_bits=args.activation_bits,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
    )
    scheduler = LinearWarmupScheduler(opt, args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    if args.compile:
        student = torch.compile(student)
        print("[student] torch.compile enabled")
    print(f"[student] d_model={cfg.d_model} n_heads={cfg.n_heads} "
          f"n_layers={cfg.n_layers} fff_depth={cfg.fff_depth} "
          f"vocab={cfg.vocab_size} tie_weights={tie} alpha={args.alpha} "
          f"T={args.temperature} device={device}")
    if args.warmup_steps > 0:
        print(f"[sched] linear warmup 0 -> {args.lr:g} over "
              f"{args.warmup_steps} steps")
    if device.type == "cuda":
        print("[amp] cuda bf16 autocast (GradScaler disabled for BF16)")

    tasks = _select_tasks(args)
    weights = {t.name: t.weight for t in tasks}
    print(f"[mixture] " + ", ".join(
        f"{t.name}={t.weight:.0%} ({t.dataset})" for t in tasks
    ))
    print(f"[run] effective batch = {args.batch_size} x "
          f"{args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps}")

    loader = MultiTaskStreamer(
        tasks, tok, cfg.vocab_size, args.seq_len, args.batch_size,
        bos_token_id=cfg.bos_token_id, eos_token_id=cfg.eos_token_id,
        max_examples=args.max_examples, seed=args.seed,
    )

    def _batches():
        while True:
            for batch in loader:
                yield batch

    stream = _batches()

    val_streamers: dict[str, MultiTaskStreamer] = {}
    if args.val_every > 0:
        for spec in tasks:
            val_streamers[spec.name] = MultiTaskStreamer(
                [dataclasses.replace(spec, weight=1.0)],
                tok, cfg.vocab_size, args.seq_len,
                args.val_batch_size or args.batch_size,
                bos_token_id=cfg.bos_token_id, eos_token_id=cfg.eos_token_id,
                split=args.val_split, seed=args.seed,
            )
        print(f"[val] per-task validation every {args.val_every} steps "
              f"(split={args.val_split!r}, batches={args.val_batches})")

    step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = train_qat.load_checkpoint(args.resume)
        student.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[distill-mt] resumed from step {step} "
              f"(best_loss={best_loss:.4f})")

    best_path = train_qat.best_checkpoint_path(args.checkpoint)
    ema: float | None = None
    loss = kd = ce = 0.0
    bar = tqdm(
        total=args.steps,
        initial=step,
        desc="[distill-mt]",
        unit="it",
        dynamic_ncols=True,
        disable=None,
    )
    try:
        while step < args.steps:
            micro = [next(stream).to(device) for _ in range(args.grad_accum_steps)]
            loss, kd, ce = accum_step(
                student, opt, teacher, micro, args.alpha,
                args.temperature, cfg.vocab_size, args.grad_accum_steps,
                scaler,
            )
            scheduler.step()
            ema = loss if ema is None else 0.9 * ema + 0.1 * loss
            step += 1

            mem = mps_driver_allocated_bytes()
            mem_mb = f"{mem / 1e6:.0f}MB" if mem else ""
            mem_str = f" mem={mem_mb}" if mem_mb else ""
            bar.set_postfix(
                loss=f"{loss:.4f}",
                ema=f"{ema:.4f}",
                mem=mem_mb or "-",
                refresh=False,
            )
            bar.update(1)

            if args.val_every > 0 and step % args.val_every == 0:
                vmetrics, blended = validate_tasks(
                    student, val_streamers, device, args.val_batches, weights
                )
                line = " ".join(
                    f"{k}={v:.4f}" for k, v in sorted(vmetrics.items())
                    if v is not None
                )
                print(f"[val] step {step} | {line} | blended={blended:.4f}")
                if blended < best_loss:
                    best_loss = blended
                    if best_path:
                        train_qat.save_checkpoint(
                            best_path, cfg, student.module.state_dict(),
                            opt.state_dict(), step, loss,
                            extra={
                                "teacher": teacher_meta,
                                "recipe": weights,
                                "alpha": args.alpha,
                                "temperature": args.temperature,
                                "grad_accum_steps": args.grad_accum_steps,
                                "scheduler": scheduler.state_dict(),
                                "best_loss": best_loss,
                                "kind": "best",
                            },
                        )
                        print(f"[distill-mt] best checkpoint -> {best_path} "
                              f"(best_loss={best_loss:.4f})")

            if step % args.log_every == 0 or step >= args.steps:
                print(f"[distill-mt] step {step}/{args.steps} loss={loss:.4f} "
                      f"kd={kd:.4f} ce={ce:.4f} ema={ema:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.3g}{mem_str}")

            if step % args.empty_cache_every == 0:
                mps_empty_cache()

            if args.checkpoint and step % args.save_every == 0:
                train_qat.save_checkpoint(
                    args.checkpoint, cfg, student.module.state_dict(),
                    opt.state_dict(), step, loss,
                    extra={
                        "teacher": teacher_meta,
                        "recipe": weights,
                        "alpha": args.alpha,
                        "temperature": args.temperature,
                        "grad_accum_steps": args.grad_accum_steps,
                        "scheduler": scheduler.state_dict(),
                        "best_loss": best_loss,
                    },
                )
                print(f"[distill-mt] checkpoint -> {args.checkpoint}")
    finally:
        bar.close()

    if args.checkpoint:
        train_qat.save_checkpoint(
            args.checkpoint, cfg, student.module.state_dict(),
            opt.state_dict(), step, loss,
            extra={
                "teacher": teacher_meta,
                "recipe": weights,
                "alpha": args.alpha,
                "temperature": args.temperature,
                "grad_accum_steps": args.grad_accum_steps,
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
            },
        )
        print(f"[distill-mt] student checkpoint -> {args.checkpoint}")
    print(f"[distill-mt] done: {step} steps, final loss={loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
