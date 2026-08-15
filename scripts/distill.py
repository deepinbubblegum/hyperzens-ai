#!/usr/bin/env python3
"""Teacher-Student Knowledge Distillation for BitNet-FFF.

Distills logits from a Teacher decoder into a QAT-trained BitNet-FFF student.
The teacher is a frozen model: either a HuggingFace decoder
(``--teacher hf://Qwen/Qwen2.5-0.5B`` / any Llama-family model, requires
``transformers``), a local BitNet-FFF checkpoint produced by ``train_qat.py``,
or a larger randomly-initialized local BitNet-FFF transformer (default).

The loss blends hard next-token cross-entropy with temperature-scaled
Kullback-Leibler divergence between teacher and student logits:

    loss = alpha * T**2 * KL(softmax(s/T) || softmax(t/T)) + (1 - alpha) * CE(s)

Checkpoints (``torch.save``) store the student config, weights, optimizer state
and a teacher descriptor so runs are fully resumable (``--resume``).

Data streams lazily from HuggingFace datasets or local text through
:class:`~scripts.train_qat.StreamingTextDataloader` (token-buffer packing, no
padding). A validation loop (``--val-data`` / ``--val-every``) saves the best
student to ``checkpoint_best.pt``; step logs include MPS Unified Memory usage.

Usage:
    python scripts/distill.py --data data/ --teacher hf://Qwen/Qwen2.5-0.5B \
        --checkpoint ckpts/distill_student.pt --steps 3000 --device mps
    python scripts/distill.py --data hf://roneneldan/TinyStories \
        --teacher teacher.pt --val-every 500 --checkpoint ckpts/distill.pt
    python scripts/distill.py --data data/ --teacher teacher.pt \
        --alpha 0.7 --temperature 2.0 --steps 1000
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.mps_utils import is_mps_available, mps_driver_allocated_bytes
from bitnet_fff.qat import BitNetQAT

import train_qat

HF_PREFIX = "hf://"


def load_teacher(
    spec: str,
    device: torch.device,
    student_vocab: int,
    teacher_args: argparse.Namespace,
) -> tuple[object, dict]:
    """Load and freeze the teacher; returns ``(model, meta)``.

    ``spec`` may be ``hf://<name>`` (HuggingFace decoder), a checkpoint path
    (dict with ``config``/``model`` from :func:`train_qat.save_checkpoint`), or
    the default ``"local"`` which builds a larger BitNet-FFF transformer.
    """
    if spec.startswith(HF_PREFIX):
        name = spec[len(HF_PREFIX):]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise SystemExit(
                f"'{name}' requires the 'transformers' package; use a local "
                "teacher (checkpoint path) instead"
            )
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        meta = {"type": "hf", "name": name,
                "vocab_size": int(model.config.vocab_size)}
        return model, meta

    if spec != "local":
        ckpt = train_qat.load_checkpoint(spec)
        cfg = BitNetFFTConfig(**ckpt["config"])
        if cfg.vocab_size != student_vocab:
            print(
                f"[teacher] warning: checkpoint vocab {cfg.vocab_size} != "
                f"student/tokenizer vocab {student_vocab}; logits will not "
                "align for distillation",
                file=sys.stderr,
            )
        model = BitNetFFTTransformer(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        meta = {"type": "checkpoint", "path": spec, "config": ckpt["config"]}
    else:
        cfg = BitNetFFTConfig(
            vocab_size=student_vocab,
            d_model=teacher_args.teacher_d_model,
            n_heads=teacher_args.teacher_n_heads,
            n_layers=teacher_args.teacher_n_layers,
            fff_depth=teacher_args.teacher_fff_depth,
            max_seq_len=teacher_args.max_seq_len,
            activation_bits=teacher_args.activation_bits,
            use_fast_inference=False,
        )
        model = BitNetFFTTransformer(cfg).to(device)
        meta = {"type": "local", "config": dataclasses.asdict(cfg)}

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def teacher_logits(teacher: object, batch: torch.Tensor) -> torch.Tensor:
    """Forward ``batch`` through the teacher and return raw logits ``(B, S, V)``."""
    with torch.no_grad():
        out = teacher(batch)
    if isinstance(out, torch.Tensor):
        return out
    return out.logits


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits_: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
    vocab: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, kd, ce)`` with temperature-scaled KL distillation."""
    s = student_logits.float()[:, :-1]
    t = teacher_logits_.float()[:, :-1]
    labels = labels[:, 1:]
    ce = F.cross_entropy(s.reshape(-1, vocab), labels.reshape(-1))
    T = temperature
    kd = (
        F.kl_div(
            F.log_softmax(s / T, dim=-1),
            F.softmax(t / T, dim=-1),
            reduction="batchmean",
        )
        * T * T
    )
    loss = alpha * kd + (1.0 - alpha) * ce
    return loss, kd, ce


def distill_step(
    student: BitNetQAT,
    optimizer,
    teacher: object,
    batch: torch.Tensor,
    alpha: float,
    temperature: float,
    vocab: int,
) -> tuple[float, float, float]:
    optimizer.zero_grad()
    t_logits = teacher_logits(teacher, batch)
    s_logits = student(batch)
    loss, kd, ce = distillation_loss(s_logits, t_logits, batch, alpha, temperature, vocab)
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(kd.item()), float(ce.item())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teacher-Student Knowledge Distillation for BitNet-FFF.",
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

    o = p.add_argument_group("optimizer")
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--clip-grad-norm", type=float, default=1.0)

    d = p.add_argument_group("data")
    d.add_argument("--data", required=True,
                   help="text file, directory of .txt files, or hf://<dataset>")
    d.add_argument("--tokenizer", default=None,
                   help="HuggingFace tokenizer name (default gpt2 BPE; "
                        "'bytes' for the byte-level fallback)")
    d.add_argument("--seq-len", type=int, default=64)
    d.add_argument("--batch-size", type=int, default=8)
    d.add_argument("--max-examples", type=int, default=None)

    v = p.add_argument_group("validation")
    v.add_argument("--val-data", default=None,
                   help="validation source (hf://<ds> or local .txt/dir); "
                        "defaults to the 'validation' split of --data")
    v.add_argument("--val-every", type=int, default=200,
                   help="compute validation loss every N steps (0 disables)")
    v.add_argument("--val-batches", type=int, default=10,
                   help="max validation batches per validation run")
    v.add_argument("--val-batch-size", type=int, default=None,
                   help="validation batch size (defaults to --batch-size)")

    r = p.add_argument_group("run")
    r.add_argument("--steps", type=int, default=2000)
    r.add_argument("--device", choices=("cpu", "mps"), default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--log-every", type=int, default=10)
    r.add_argument("--checkpoint", default=None,
                   help="where to save the distilled student (torch.save)")
    r.add_argument("--save-every", type=int, default=500)
    r.add_argument("--resume", default=None,
                   help="checkpoint to resume from (auto-resumes step, "
                        "optimizer, and best-validation state)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("mps" if is_mps_available() else "cpu"))
    torch.manual_seed(args.seed)

    tok = train_qat.load_tokenizer(args.tokenizer, args.vocab_size)
    student_vocab = args.vocab_size
    if isinstance(tok, train_qat.BPETokenizer):
        student_vocab = int(tok.vocab_size)
        print(f"[tokenizer] BPE {tok.name} vocab={student_vocab}")
    teacher, teacher_meta = load_teacher(args.teacher, device, student_vocab, args)
    print(f"[teacher] {teacher_meta.get('type')}: "
          f"{teacher_meta.get('name', teacher_meta.get('path', 'local'))}")
    if teacher_meta.get("type") == "hf":
        student_vocab = teacher_meta["vocab_size"]
    if args.vocab_size != student_vocab:
        print(f"[student] forcing vocab_size={student_vocab}")
        args.vocab_size = student_vocab

    student_cfg = train_qat.build_cfg(args)
    student, opt = train_qat.make_qat_model(
        student_cfg, device, fp16=not args.no_fp16, lr=args.lr,
        weight_decay=args.weight_decay, clip_grad_norm=args.clip_grad_norm,
    )
    print(f"[student] d_model={student_cfg.d_model} n_heads={student_cfg.n_heads} "
          f"n_layers={student_cfg.n_layers} fff_depth={student_cfg.fff_depth} "
          f"vocab={student_cfg.vocab_size} alpha={args.alpha} "
          f"T={args.temperature} device={device}")

    step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = train_qat.load_checkpoint(args.resume)
        student.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"[distill] resumed from step {step} "
              f"(best_loss={best_loss:.4f})")

    loader = train_qat.StreamingTextDataloader(
        args.data, tok, student_vocab, args.seq_len, args.batch_size,
        max_examples=args.max_examples, seed=args.seed,
    )

    val_loader: train_qat.StreamingTextDataloader | None = None
    val_every = args.val_every
    if val_every > 0:
        val_source = args.val_data or (
            args.data if args.data.startswith(train_qat.HF_STREAM) else None
        )
        if val_source:
            val_loader = train_qat.StreamingTextDataloader(
                val_source, tok, student_vocab, args.seq_len,
                args.val_batch_size or args.batch_size,
                split="validation", seed=args.seed,
            )
            print(f"[val] streaming validation from {val_source} "
                  f"(split=validation, batch={args.val_batch_size or args.batch_size}, "
                  f"every {val_every} steps)")
        else:
            val_every = 0
            print("[val] no validation source; pass --val-data or --data hf://<ds>")

    best_path = train_qat.best_checkpoint_path(args.checkpoint)
    ema: float | None = None
    loss = kd = ce = None
    for batch in loader:
        loss, kd, ce = distill_step(
            student, opt, teacher, batch.to(device), args.alpha,
            args.temperature, student_cfg.vocab_size,
        )
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        step += 1

        candidate: float | None = None
        if val_every and val_loader is not None and step % val_every == 0:
            val_loss = train_qat.validate(
                student, val_loader, device, args.val_batches
            )
            if val_loss is not None:
                print(f"[val] step {step} val_loss={val_loss:.4f}")
                candidate = val_loss
            else:
                print(f"[val] step {step} validation stream empty; "
                      "disabling further validation")
                val_every = 0
        elif val_every == 0 and ema is not None and step % args.save_every == 0:
            candidate = ema

        if candidate is not None and candidate < best_loss:
            best_loss = candidate
            if best_path:
                train_qat.save_checkpoint(
                    best_path, student_cfg, student.module.state_dict(),
                    opt.state_dict(), step, loss,
                    extra={"teacher": teacher_meta, "alpha": args.alpha,
                           "temperature": args.temperature,
                           "best_loss": best_loss, "kind": "best"},
                )
                print(f"[distill] best checkpoint -> {best_path} "
                      f"(best_loss={best_loss:.4f})")

        if step % args.log_every == 0 or step >= args.steps:
            mem = mps_driver_allocated_bytes()
            mem_str = f" mem={mem / 1e6:.0f}MB" if mem else ""
            print(f"[distill] step {step}/{args.steps} loss={loss:.4f} "
                  f"kd={kd:.4f} ce={ce:.4f} ema={ema:.4f}{mem_str}")
        if args.checkpoint and step % args.save_every == 0:
            train_qat.save_checkpoint(
                args.checkpoint, student_cfg, student.module.state_dict(),
                opt.state_dict(), step, loss,
                extra={"teacher": teacher_meta, "alpha": args.alpha,
                       "temperature": args.temperature,
                       "best_loss": best_loss},
            )
        if step >= args.steps:
            break
    if args.checkpoint:
        train_qat.save_checkpoint(
            args.checkpoint, student_cfg, student.module.state_dict(),
            opt.state_dict(), step, loss,
            extra={"teacher": teacher_meta, "alpha": args.alpha,
                   "temperature": args.temperature,
                   "best_loss": best_loss},
        )
        print(f"[distill] student checkpoint -> {args.checkpoint}")
    if step == 0:
        print("[distill] no batches consumed; is --data empty or shorter than "
              "--seq-len?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
