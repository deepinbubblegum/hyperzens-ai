#!/usr/bin/env python3
"""Quantization-Aware Training (QAT) for BitNet-FFF on MPS/CPU.

Trains the BitNet ternary leaf weights and routing decision nodes of a
:class:`BitNetFFTTransformer` under dynamic activation scaling
(:class:`ActivationQuantizer`, AbsMax by default) with FP16 master weights
held by :class:`FP16MasterAdamW` (all Adam math in FP32). The transformer's
internal straight-through quantizers do the real BitNet work: per-token AbsMax
activation scaling on attention/FFF inputs, and STE ternary leaf weights.

Data can be streamed either from HuggingFace datasets
(``--data hf://roneneldan/TinyStories``) or from local ``.txt`` files/dirs.
Text is tokenized as bytes (GPT-2 style) when ``transformers`` is unavailable.

Usage:
    python scripts/train_qat.py --data data/ --seq-len 64 --batch-size 8 --steps 2000
    python scripts/train_qat.py --data hf://roneneldan/TinyStories --steps 5000 \
        --d-model 256 --n-layers 4 --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.mps_utils import is_mps_available
from bitnet_fff.qat import BitNetQAT

HF_STREAM = "hf://"


class ByteTokenizer:
    """Byte-level tokenizer (GPT-2 style: id <-> raw UTF-8 byte)."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if any(i >= self.vocab_size for i in ids):
            raise ValueError(
                f"text byte {max(ids)} >= vocab_size {self.vocab_size}; "
                "raise --vocab-size (>= 256)"
            )
        return ids

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def load_tokenizer(name: str | None, vocab_size: int):
    """Return a tokenizer object with ``encode(str) -> list[int]`` / ``decode``.

    Uses a HuggingFace ``AutoTokenizer`` when ``name`` is given and
    ``transformers`` is installed, otherwise the byte-level fallback.
    """
    if not name:
        return ByteTokenizer(vocab_size)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "[tokenizer] 'transformers' not installed; byte-level fallback",
            file=sys.stderr,
        )
        return ByteTokenizer(vocab_size)
    return AutoTokenizer.from_pretrained(name)


def _encode_clamped(tokenizer, text: str, vocab_size: int) -> list[int]:
    raw = tokenizer.encode(text)
    if any(i >= vocab_size for i in raw):
        raw = [min(i, vocab_size - 1) for i in raw]
    return raw


def tokenize_stream(
    source: str, tokenizer, vocab_size: int, max_examples: int | None = None
):
    """Yield lists of token ids from a text source.

    ``source`` is a ``.txt`` file, a directory of ``.txt`` files, or an
    ``hf://`` dataset name (requires ``datasets``).
    """
    count = 0

    def _yield(text: str):
        nonlocal count
        if max_examples is not None and count >= max_examples:
            return
        ids = _encode_clamped(tokenizer, text, vocab_size)
        if ids:
            count += 1
            yield ids

    if source.startswith(HF_STREAM):
        name = source[len(HF_STREAM):]
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                f"'{name}' requires the 'datasets' package; install it or use "
                "a local text file/dir for --data"
            )
        ds = load_dataset(name, split="train", streaming=True)
        for example in ds:
            if max_examples is not None and count >= max_examples:
                break
            text = example.get("text") or example.get("content") or ""
            yield from _yield(text)
        return

    paths: list[str] = []
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.txt")))
    elif os.path.isfile(source):
        paths = [source]
    else:
        raise SystemExit(f"--data {source!r} not found")
    if not paths:
        raise SystemExit(f"no *.txt files under {source!r}")
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from _yield(f.read())


def iter_batches(token_stream, seq_len: int, batch_size: int, vocab_size: int):
    """Chunk a token stream into ``(batch_size, seq_len)`` long tensors.

    Non-overlapping ``seq_len`` windows are grouped into batches; a trailing
    partial row and an incomplete final batch are dropped.
    """
    buf: list[int] = []
    rows: list[list[int]] = []
    for ids in token_stream:
        buf.extend(ids)
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            del buf[:seq_len]
            if len(rows) == batch_size:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []


def build_cfg(args: argparse.Namespace) -> BitNetFFTConfig:
    def g(name: str, default):
        return getattr(args, name, default)

    if g("seq_len", 64) > g("max_seq_len", 64):
        raise ValueError(
            f"--seq-len {g('seq_len', 64)} > --max-seq-len {g('max_seq_len', 64)}"
        )
    return BitNetFFTConfig(
        vocab_size=g("vocab_size", 256),
        d_model=g("d_model", 128),
        n_heads=g("n_heads", 4),
        n_layers=g("n_layers", 2),
        fff_depth=g("fff_depth", 3),
        max_seq_len=g("max_seq_len", 64),
        activation_bits=g("activation_bits", 8),
        attention_activation_bits=g("attention_activation_bits", None),
        router_rank=g("router_rank", "full"),
        fff_bias=not g("no_fff_bias", False),
        use_fast_inference=False,
    )


def make_qat_model(
    cfg: BitNetFFTConfig,
    device: torch.device,
    fp16: bool = True,
    quant_mode: str = "absmax",
    activation_bits: int = 8,
    threshold_scale: float = 1.0,
    lr: float = 1e-3,
    **opt_kwargs,
) -> tuple[BitNetQAT, torch.optim.Optimizer]:
    """Build a QAT-wrapped transformer and its FP16-master AdamW optimizer."""
    model = BitNetFFTTransformer(cfg).to(device)
    qat = BitNetQAT(
        model,
        activation_bits=activation_bits,
        quant_mode=quant_mode,
        threshold_scale=threshold_scale,
    )
    if fp16:
        qat.enable_fp16_master()
    optimizer = qat.optimizer(lr=lr, **opt_kwargs)
    return qat, optimizer


def train_step(qat: BitNetQAT, optimizer, input_ids: torch.Tensor) -> float:
    """One next-token-prediction QAT step; returns the CE loss."""
    optimizer.zero_grad()
    logits = qat(input_ids)
    vocab = logits.shape[-1]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab).float(), input_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    path: str,
    cfg: BitNetFFTConfig,
    model_state: dict,
    optimizer_state: dict | None = None,
    step: int = 0,
    loss: float | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": dataclasses.asdict(cfg),
        "model": model_state,
        "optimizer": optimizer_state,
        "step": step,
        "loss": loss,
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantization-Aware Training for BitNet-FFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m = p.add_argument_group("model")
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

    q = p.add_argument_group("QAT / optimizer")
    q.add_argument("--quant-mode", choices=("absmax", "per_channel", "ema", "learned"),
                   default="absmax")
    q.add_argument("--threshold-scale", type=float, default=1.0)
    q.add_argument("--no-fp16", action="store_true",
                   help="train fp32 master weights instead of fp16")
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--beta1", type=float, default=0.9)
    q.add_argument("--beta2", type=float, default=0.999)
    q.add_argument("--weight-decay", type=float, default=0.01)
    q.add_argument("--clip-grad-norm", type=float, default=1.0)

    d = p.add_argument_group("data")
    d.add_argument("--data", required=True,
                   help="text file, directory of .txt files, or hf://<dataset>")
    d.add_argument("--tokenizer", default=None,
                   help="HuggingFace tokenizer name (byte fallback if unset)")
    d.add_argument("--seq-len", type=int, default=64)
    d.add_argument("--batch-size", type=int, default=8)
    d.add_argument("--max-examples", type=int, default=None,
                   help="cap streamed examples")

    t = p.add_argument_group("run")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--device", choices=("cpu", "mps"), default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=500)
    t.add_argument("--checkpoint", default=None, help="output checkpoint path")
    t.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("mps" if is_mps_available() else "cpu"))
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.tokenizer, args.vocab_size)
    cfg = build_cfg(args)
    qat, opt = make_qat_model(
        cfg, device,
        fp16=not args.no_fp16,
        quant_mode=args.quant_mode,
        activation_bits=args.activation_bits,
        threshold_scale=args.threshold_scale,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
    )
    print(f"[qat] {cfg.d_model}/{cfg.n_heads}/{cfg.n_layers} fff_depth={cfg.fff_depth} "
          f"vocab={cfg.vocab_size} fp16_master={not args.no_fp16} "
          f"quant_mode={args.quant_mode} device={device}")

    step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        qat.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        print(f"[qat] resumed from step {step}")

    stream = iter_batches(
        tokenize_stream(args.data, tok, args.vocab_size, args.max_examples),
        args.seq_len, args.batch_size, args.vocab_size,
    )
    losses: list[float] = []
    ema: float | None = None
    for batch in stream:
        loss = train_step(qat, opt, batch.to(device))
        losses.append(loss)
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        step += 1
        if step % args.log_every == 0 or step >= args.steps:
            print(f"[qat] step {step}/{args.steps} loss={loss:.4f} ema={ema:.4f}")
        if args.checkpoint and step % args.save_every == 0:
            save_checkpoint(args.checkpoint, cfg, qat.module.state_dict(),
                            opt.state_dict(), step, loss)
        if step >= args.steps:
            break
    if args.checkpoint:
        save_checkpoint(args.checkpoint, cfg, qat.module.state_dict(),
                        opt.state_dict(), step, losses[-1] if losses else None)
        print(f"[qat] checkpoint -> {args.checkpoint}")
    final_loss = losses[-1] if losses else None
    print(f"[qat] done: {step} steps, "
          f"final loss={final_loss:.4f}" if final_loss is not None
          else f"[qat] done: {step} steps (no batches consumed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
