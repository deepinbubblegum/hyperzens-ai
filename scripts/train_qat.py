#!/usr/bin/env python3
"""Quantization-Aware Training (QAT) for BitNet-FFF on MPS/CPU.

Trains the BitNet ternary leaf weights and routing decision nodes of a
:class:`BitNetFFTTransformer` under dynamic activation scaling
(:class:`ActivationQuantizer`, AbsMax by default) with FP16 master weights
held by :class:`FP16MasterAdamW` (all Adam math in FP32). The transformer's
internal straight-through quantizers do the real BitNet work: per-token AbsMax
activation scaling on attention/FFF inputs, and STE ternary leaf weights.

Data is streamed lazily through :class:`StreamingTextDataloader` from
HuggingFace datasets (``--data hf://roneneldan/TinyStories``,
``--data hf://openwebtext``) or local ``.txt`` files/dirs, with token-buffer
packing into fixed ``seq_len`` windows (no padding). A validation loop
(``--val-data`` / ``--val-every``) tracks the best model into
``checkpoint_best.pt``; ``--resume`` auto-continues from any saved checkpoint.
Training-step logs include the MPS Unified Memory usage
(``mps_driver_allocated_bytes``).

Usage:
    python scripts/train_qat.py --data data/ --seq-len 64 --batch-size 8 --steps 2000
    python scripts/train_qat.py --data hf://roneneldan/TinyStories --steps 5000 \
        --d-model 256 --n-layers 4 --device mps
    python scripts/train_qat.py --data hf://openwebtext --val-every 500 \
        --checkpoint runs/qat.pt --resume runs/qat.pt
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
from bitnet_fff.mps_utils import is_mps_available, mps_driver_allocated_bytes
from bitnet_fff.qat import BitNetQAT
from bitnet_fff.tokenizer import ByteTokenizer, BPETokenizer, load_tokenizer

HF_STREAM = "hf://"


def _encode_clamped(tokenizer, text: str, vocab_size: int) -> list[int]:
    raw = tokenizer.encode(text)
    if any(i >= vocab_size for i in raw):
        raw = [min(i, vocab_size - 1) for i in raw]
    return raw


def tokenize_stream(
    source: str,
    tokenizer,
    vocab_size: int,
    max_examples: int | None = None,
    split: str = "train",
    seed: int = 0,
):
    """Yield lists of token ids from a text source.

    ``source`` is a ``.txt`` file, a directory of ``.txt`` files, or an
    ``hf://`` dataset name streamed with ``datasets.load_dataset(...,
    streaming=True)``. For HuggingFace sources ``split`` selects the stream
    (``"train"`` / ``"validation"`` / ``"test"``); a validation split that does
    not exist automatically falls back to ``"test"`` so the same dataset can
    serve both training and validation.
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
        attempts = [split] + (["test"] if split != "train" else [])
        ds = None
        last_err: Exception | None = None
        for s in attempts:
            try:
                ds = load_dataset(name, split=s, streaming=True)
                break
            except Exception as e:  # missing/unsupported split, bad shard, ...
                last_err = e
        if ds is None:
            raise SystemExit(
                f"dataset {name!r}: no streamable split in {attempts} "
                f"({last_err})"
            )
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


class StreamingTextDataloader:
    """Stream text chunks and pack tokens into fixed-window batches.

    A :term:`streaming dataloader`: text is pulled lazily from the source (a
    HuggingFace dataset via ``load_dataset(..., streaming=True)``, a ``.txt``
    file, or a directory of ``.txt`` files), tokenized, and **packed** into
    non-overlapping ``seq_len`` windows grouped into ``(batch_size, seq_len)``
    long tensors. Windows never straddle the batch boundary and no padding is
    added — a trailing partial window or an incomplete final batch is dropped.
    Re-iterating a loader starts a fresh stream, so the same instance can be
    used for repeated validation passes.

    Args:
        source: ``hf://<dataset>``, a ``.txt`` file, or a directory.
        tokenizer: object with ``encode(str) -> list[int]``.
        vocab_size: clamp token ids below this bound.
        seq_len: fixed sequence-window length per row.
        batch_size: rows per yielded batch.
        split: dataset split to stream (ignored for local files).
        max_examples: cap the number of tokenized text examples.
        seed: reserved for reproducibility (streams are consumed in order).
    """

    def __init__(
        self,
        source: str,
        tokenizer,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        split: str = "train",
        max_examples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.source = source
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.split = split
        self.max_examples = max_examples
        self.seed = seed

    def __iter__(self):
        stream = tokenize_stream(
            self.source,
            self.tokenizer,
            self.vocab_size,
            self.max_examples,
            split=self.split,
            seed=self.seed,
        )
        yield from iter_batches(
            stream, self.seq_len, self.batch_size, self.vocab_size
        )


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
        top_k=g("top_k", None),
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


def next_token_loss(qat: BitNetQAT, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy over the logits vs the shifted ids."""
    logits = qat(input_ids)
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab).float(), input_ids[:, 1:].reshape(-1)
    )


def train_step(qat: BitNetQAT, optimizer, input_ids: torch.Tensor) -> float:
    """One next-token-prediction QAT step; returns the CE loss."""
    optimizer.zero_grad()
    loss = next_token_loss(qat, input_ids)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def validate(
    qat: BitNetQAT,
    dataloader: StreamingTextDataloader,
    device: torch.device,
    max_batches: int = 10,
) -> float | None:
    """Mean next-token CE loss over a validation stream (no grads).

    Returns ``None`` when the validation stream yields no batches (e.g. the
    requested split does not exist), letting the caller disable validation.
    """
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            total += float(next_token_loss(qat, batch.to(device)).item())
            count += 1
    return total / count if count else None


def best_checkpoint_path(checkpoint: str | None) -> str | None:
    """Path for ``checkpoint_best.pt`` next to ``checkpoint`` (or ``None``)."""
    if not checkpoint:
        return None
    return os.path.join(
        os.path.dirname(os.path.abspath(checkpoint)) or ".", "checkpoint_best.pt"
    )


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
    m.add_argument("--top-k", type=int, default=None,
                   help="activate top-k leaves per token (BitNetUFF); "
                        "None = classic single-leaf FFF. Pair with deep "
                        "--fff-depth 10/12 for ultra-sparse compute")
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
                   help="HuggingFace tokenizer name (default gpt2 BPE; "
                        "'bytes' for the byte-level fallback)")
    d.add_argument("--seq-len", type=int, default=64)
    d.add_argument("--batch-size", type=int, default=8)
    d.add_argument("--max-examples", type=int, default=None,
                   help="cap streamed examples")

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
    v.add_argument("--val-max-examples", type=int, default=None,
                   help="cap streamed validation examples")

    t = p.add_argument_group("run")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--device", choices=("cpu", "mps"), default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=500)
    t.add_argument("--checkpoint", default=None, help="output checkpoint path")
    t.add_argument("--resume", default=None,
                   help="checkpoint to resume from (auto-resumes step, "
                        "optimizer, and best-validation state)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("mps" if is_mps_available() else "cpu"))
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.tokenizer, args.vocab_size)
    cfg = build_cfg(args).bind_tokenizer(tok)
    if isinstance(tok, BPETokenizer):
        print(f"[tokenizer] BPE {tok.name} vocab={cfg.vocab_size}")
    else:
        print(f"[tokenizer] byte-level fallback vocab={cfg.vocab_size}")
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
          f"top_k={cfg.top_k} vocab={cfg.vocab_size} fp16_master={not args.no_fp16} "
          f"quant_mode={args.quant_mode} device={device}")

    step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        qat.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"[qat] resumed from step {step} "
              f"(best_loss={best_loss:.4f})")

    loader = StreamingTextDataloader(
        args.data, tok, cfg.vocab_size, args.seq_len, args.batch_size,
        max_examples=args.max_examples, seed=args.seed,
    )

    val_loader: StreamingTextDataloader | None = None
    val_every = args.val_every
    if val_every > 0:
        val_source = args.val_data or (
            args.data if args.data.startswith(HF_STREAM) else None
        )
        if val_source:
            val_loader = StreamingTextDataloader(
                val_source, tok, cfg.vocab_size, args.seq_len,
                args.val_batch_size or args.batch_size,
                split="validation",
                max_examples=args.val_max_examples, seed=args.seed,
            )
            print(f"[val] streaming validation from {val_source} "
                  f"(split=validation, batch={args.val_batch_size or args.batch_size}, "
                  f"every {val_every} steps)")
        else:
            val_every = 0
            print("[val] no validation source; pass --val-data or --data hf://<ds>")

    best_path = best_checkpoint_path(args.checkpoint)
    losses: list[float] = []
    ema: float | None = None
    for batch in loader:
        loss = train_step(qat, opt, batch.to(device))
        losses.append(loss)
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        step += 1

        candidate: float | None = None
        if val_every and val_loader is not None and step % val_every == 0:
            val_loss = validate(qat, val_loader, device, args.val_batches)
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
                save_checkpoint(
                    best_path, cfg, qat.module.state_dict(), opt.state_dict(),
                    step, loss, extra={"best_loss": best_loss, "kind": "best"},
                )
                print(f"[qat] best checkpoint -> {best_path} "
                      f"(best_loss={best_loss:.4f})")

        if step % args.log_every == 0 or step >= args.steps:
            mem = mps_driver_allocated_bytes()
            mem_str = f" mem={mem / 1e6:.0f}MB" if mem else ""
            print(f"[qat] step {step}/{args.steps} loss={loss:.4f} "
                  f"ema={ema:.4f}{mem_str}")
        if args.checkpoint and step % args.save_every == 0:
            save_checkpoint(args.checkpoint, cfg, qat.module.state_dict(),
                            opt.state_dict(), step, loss,
                            extra={"best_loss": best_loss})
        if step >= args.steps:
            break
    if args.checkpoint:
        save_checkpoint(args.checkpoint, cfg, qat.module.state_dict(),
                        opt.state_dict(), step, losses[-1] if losses else None,
                        extra={"best_loss": best_loss})
        print(f"[qat] checkpoint -> {args.checkpoint}")
    final_loss = losses[-1] if losses else None
    print(f"[qat] done: {step} steps, "
          f"final loss={final_loss:.4f}" if final_loss is not None
          else f"[qat] done: {step} steps (no batches consumed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
