#!/usr/bin/env python3
"""Generate text with a BitNetFFTTransformer using the KV-cache fast path.

Builds (or loads) a BitNet FFT transformer, tokenizes the prompt with a
HuggingFace tokenizer when ``transformers`` is installed (falling back to a
byte-level vocabulary otherwise), prefills the KV cache on the full prompt,
decodes one token per step with the fused single-pass FFF kernel, and
benchmarks the decode phase.

Usage:
    python scripts/generate.py --prompt "The quick brown fox jumps" --max-new-tokens 32
    python scripts/generate.py --prompt "Once upon a time" --temperature 0.8 \
        --top-k 40 --top-p 0.9 --eos-token-id 50256 --device mps
    python scripts/generate.py --checkpoint runs/checkpoint.pt --device mps
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_fff.models import (
    BitNetFFTConfig,
    BitNetFFTTransformer,
    KVCache,
    _sample_from_logits,
)
from bitnet_fff.mps_utils import is_mps_available, mps_synchronize


class ByteTokenizer:
    """Byte-level fallback tokenizer (GPT-2 style: id <-> raw UTF-8 byte).

    Used when ``transformers`` is not installed or ``--tokenizer`` is omitted.
    Each id is a byte (0..255), so ``vocab_size`` must be at least 256 for
    arbitrary UTF-8 text; shorter vocabularies truncate per-id silently via the
    caller's clamping.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if any(i >= self.vocab_size for i in ids):
            raise ValueError(
                f"text byte {max(ids)} >= vocab_size {self.vocab_size}; "
                "raise --vocab-size (>= 256) or pass --tokenizer"
            )
        return ids

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={self.vocab_size})"


def load_tokenizer(
    name: str | None, vocab_size: int
) -> tuple[object, bool]:
    """Load a HuggingFace tokenizer or fall back to the byte-level one.

    Returns ``(tokenizer, is_hf)`` where ``is_hf`` is ``True`` only when a real
    ``transformers`` tokenizer was loaded.
    """
    if not name:
        return ByteTokenizer(vocab_size), False
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "[tokenizer] 'transformers' is not installed; "
            "falling back to the byte-level tokenizer",
            file=sys.stderr,
        )
        return ByteTokenizer(vocab_size), False
    tok = AutoTokenizer.from_pretrained(name)
    return tok, True


def build_model(
    cfg: BitNetFFTConfig,
    device: torch.device,
    checkpoint: str | None = None,
) -> BitNetFFTTransformer:
    model = BitNetFFTTransformer(cfg).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        state = state.get("model", state)
        model.load_state_dict(state)
        print(f"[model] loaded checkpoint {checkpoint}")
    model.eval()
    return model


def _fresh_caches(
    model: BitNetFFTTransformer, batch: int, device: torch.device
) -> list[KVCache]:
    head_dim = model.cfg.d_model // model.cfg.n_heads
    dtype = next(model.parameters()).dtype
    return [
        KVCache.preallocate(
            batch, model.cfg.n_heads, model.cfg.max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        for _ in model.layers
    ]


def benchmark_decode(
    model: BitNetFFTTransformer,
    prompt_ids: torch.Tensor,
    max_seq_len: int,
    iters: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
    warmup: int = 3,
) -> tuple[float, float]:
    """Mean decode-step latency (ms/token) and throughput (tokens/sec).

    The prompt is prefilled once (untimed), then each step processes a single
    cached token through the model plus token selection -- exactly the decode
    phase of :meth:`generate`. ``mps_synchronize`` is called around every step
    so the reported wall time includes the actual MPS execution.
    """
    batch, prompt_len = prompt_ids.shape
    budget = max_seq_len - prompt_len - warmup
    iters = min(iters, budget)
    if iters < 1:
        raise ValueError(
            f"prompt_len + warmup ({prompt_len} + {warmup}) leaves no room for "
            f"{max_seq_len} - budget"
        )

    caches = _fresh_caches(model, batch, device)
    model(prompt_ids, kv_cache=caches)
    last = prompt_ids[:, -1:]
    past = prompt_len

    for _ in range(warmup):
        logits = model(last, kv_cache=caches, past_length=past)
        last = _sample_from_logits(logits[:, -1], temperature, top_k, top_p)
        past += 1

    mps_synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        logits = model(last, kv_cache=caches, past_length=past)
        last = _sample_from_logits(logits[:, -1], temperature, top_k, top_p)
        past += 1
        mps_synchronize()
        times.append(time.perf_counter() - t0)

    mean_s = statistics.mean(times)
    ms_per_token = mean_s * 1e3
    tokens_per_s = 1e3 / ms_per_token if ms_per_token > 0 else float("nan")
    return ms_per_token, tokens_per_s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text with a BitNet FFT transformer (KV-cache decode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model config")
    g.add_argument("--d-model", type=int, default=128)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument("--fff-depth", type=int, default=3)
    g.add_argument("--vocab-size", type=int, default=256)
    g.add_argument("--max-seq-len", type=int, default=128)
    g.add_argument("--activation-bits", type=int, default=8)
    g.add_argument("--attention-activation-bits", type=int, default=None)
    g.add_argument("--router-rank", choices=("full", "r1"), default="full")
    g.add_argument("--no-fff-bias", action="store_true", help="disable FFF leaf biases")
    g.add_argument("--no-fast-inference", action="store_true",
                   help="disable the packed-ternary fused kernel (use the FP32 path)")
    g.add_argument("--tie-weights", action="store_true")
    g.add_argument("--checkpoint", default=None, help="state-dict path to load")

    d = p.add_argument_group("device / rng")
    d.add_argument("--device", choices=("cpu", "mps"), default=None)
    d.add_argument("--seed", type=int, default=0)

    t = p.add_argument_group("tokenizer")
    t.add_argument("--tokenizer", default=None,
                   help="HuggingFace tokenizer name (e.g. 'gpt2'); byte fallback if unset")

    s = p.add_argument_group("generation")
    s.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    s.add_argument("--max-new-tokens", type=int, default=32)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--top-p", type=float, default=0.9)
    s.add_argument("--eos-token-id", type=int, default=None)

    b = p.add_argument_group("decode-phase benchmark")
    b.add_argument("--bench-iters", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(
        args.device or ("mps" if is_mps_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    tok, is_hf = load_tokenizer(args.tokenizer, args.vocab_size)
    print(f"[tokenizer] {'HF:' + str(args.tokenizer) if is_hf else repr(tok)}")

    cfg = BitNetFFTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        fff_depth=args.fff_depth,
        max_seq_len=args.max_seq_len,
        activation_bits=args.activation_bits,
        attention_activation_bits=args.attention_activation_bits,
        router_rank=args.router_rank,
        fff_bias=not args.no_fff_bias,
        use_fast_inference=not args.no_fast_inference,
        tie_weights=args.tie_weights,
    )
    model = build_model(cfg, device, args.checkpoint)

    eos = args.eos_token_id
    if eos is None and is_hf and getattr(tok, "eos_token_id", None) is not None:
        eos = int(tok.eos_token_id)

    raw = tok.encode(args.prompt)
    if any(i >= args.vocab_size for i in raw):
        print(
            f"[tokenizer] clipping {sum(i >= args.vocab_size for i in raw)} "
            f"ids to vocab_size {args.vocab_size}",
            file=sys.stderr,
        )
    ids = [min(i, args.vocab_size - 1) for i in raw]
    if not ids:
        raise SystemExit("empty prompt after tokenization")
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    if args.prompt and args.prompt.strip() != tok.decode(raw):
        print(f"[prompt] tokenized as {raw}")

    print(f"[model] d_model={cfg.d_model} n_heads={cfg.n_heads} "
          f"n_layers={cfg.n_layers} fff_depth={cfg.fff_depth} "
          f"params={sum(p.numel() for p in model.parameters()):,} "
          f"device={device}")
    if eos is not None:
        print(f"[gen] eos_token_id={eos}")

    t0 = time.perf_counter()
    out = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos,
    )
    wall_ms = (time.perf_counter() - t0) * 1e3
    mps_synchronize()

    prompt_len = prompt.shape[1]
    generated = out[0][prompt_len:]
    print(f"[gen] wall time (prefill+decode) = {wall_ms:.1f} ms")
    print(f"[gen] generated {generated.shape[0]}/{args.max_new_tokens} tokens")
    print()
    print(tok.decode(out[0].tolist()))
    print()

    ms_per_token, tokens_per_s = benchmark_decode(
        model, prompt, cfg.max_seq_len, args.bench_iters,
        args.temperature, args.top_k, args.top_p, device,
    )
    print(f"[benchmark] decode phase (device={device}, batch=1, "
          f"iters={args.bench_iters})")
    print(f"  latency : {ms_per_token:.2f} ms/token")
    print(f"  speed   : {tokens_per_s:,.1f} tokens/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
