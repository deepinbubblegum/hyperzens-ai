#!/usr/bin/env python3
"""Real-time interactive terminal chat for a BitNetFFTTransformer.

Streams the model's reply token-by-token to stdout (typewriter effect) via
:meth:`bitnet_fff.models.BitNetFFTTransformer.stream_generate`, which drives
the KV-cache + fused single-pass FFF kernel decode path. Multi-turn dialogue
history is kept inside the ``max_seq_len`` window by evicting the oldest turns
until the assembled prompt fits (clipping only as a last resort).

Usage:
    python scripts/chat.py --device mps
    python scripts/chat.py --checkpoint runs/model.pt --temperature 0.8 \
        --max-new-tokens 96 --system "You are a helpful assistant."
    python scripts/chat.py --device cpu --tokenizer bytes  # offline fallback

When ``--checkpoint`` points to a checkpoint saved by ``train_qat`` (which
embeds the full ``BitNetFFTConfig`` as ``"config"``), the architecture is
restored from the checkpoint automatically, so no ``--d-model``/``--n-layers``/
``--fff-depth`` flags are needed. Explicit CLI architecture flags are then
ignored; ``--no-fast-inference`` still overrides the fused-kernel path.

Commands inside the chat: ``/help``, ``/reset``, ``/quit`` (or ``/exit``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.mps_utils import is_mps_available, mps_synchronize
from bitnet_fff.tokenizer import TikTokenizer, load_tokenizer

_HELP = """Commands:
  /help            show this help
  /reset           clear the dialogue history
  /quit, /exit     leave the chat
Enter text to talk to the model."""


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
    """FP16 autocast context for CUDA / MPS; a no-op on CPU.

    Uses ``torch.amp.autocast(device_type=device.type, dtype=torch.float16)`` so generated forward
    passes run in mixed precision on CUDA and Apple Silicon (MPS), while CPU keeps the existing
    FP32 path.
    """
    dev_type = getattr(device, "type", None)
    if dev_type in ("cuda", "mps"):
        return torch.amp.autocast(device_type=dev_type, dtype=torch.float16)
    return contextlib.nullcontext()


def build_model(
    cfg: BitNetFFTConfig,
    device: torch.device,
    checkpoint: str | None = None,
    state: dict | None = None,
) -> BitNetFFTTransformer:
    model = BitNetFFTTransformer(cfg).to(device)
    if checkpoint or state is not None:
        if state is None:
            state = torch.load(checkpoint, map_location=device)
        state = state.get("model", state)
        new_len = _resize_position_embedding(state, model)
        if new_len is not None:
            print(f"[model] resized pos embedding {new_len} rows "
                  f"(max_seq_len={model.cfg.max_seq_len})")
        model.load_state_dict(state)
        print(f"[model] loaded checkpoint {checkpoint}")
    model.eval()
    return model


def _resize_position_embedding(
    state: dict,
    model: BitNetFFTTransformer,
) -> int | None:
    """Resize a checkpoint's ``pos.weight`` to the model's ``max_seq_len``.

    Position embeddings are learned tables sized by ``max_seq_len``, so
    overriding ``--max-seq-len`` (e.g. ``512`` for a checkpoint saved at a
    shorter window) would otherwise fail ``load_state_dict`` with a shape
    mismatch. The overlapping prefix is preserved and any new rows keep the
    freshly initialized values. Returns the resized length, else ``None``.
    """
    saved = state.get("pos.weight")
    if saved is None or model.pos is None:
        return None
    target = model.pos.weight.shape[0]
    if saved.shape[0] == target:
        return None
    resized = model.pos.weight.detach().clone()
    if saved.shape[1] == resized.shape[1]:
        n = min(saved.shape[0], target)
        resized[:n] = saved[:n]
    state["pos.weight"] = resized
    return target


def _checkpoint_config(state: dict) -> BitNetFFTConfig | None:
    """Rebuild a :class:`BitNetFFTConfig` from a checkpoint's ``"config"`` dict.

    Checkpoints saved by ``train_qat.save_checkpoint`` store the full config
    (``dataclasses.asdict``), so the architecture (``d_model``, ``n_heads``,
    ``n_layers``, ``fff_depth``, ``top_k``, ``max_seq_len``, ...) can be
    recovered without passing CLI flags. Unknown or ``None`` keys are dropped
    so older checkpoints stay compatible; the pre-``top_k`` config key
    ``fff_k`` (BitNet-UFF active leaves) is migrated to ``top_k``.
    """
    saved = state.get("config")
    if not isinstance(saved, dict):
        return None
    saved = dict(saved)
    if "fff_k" in saved and "top_k" not in saved:
        saved["top_k"] = saved.pop("fff_k")
    allowed = set(BitNetFFTConfig.__dataclass_fields__)
    known = {k: v for k, v in saved.items() if k in allowed and v is not None}
    if not known:
        return None
    return BitNetFFTConfig(**known)


def _cli_arch_overrides(
    args: argparse.Namespace,
    defaults: argparse.Namespace,
) -> tuple[dict, list[str]]:
    """Architecture flags the user explicitly supplied (vs parser defaults).

    Every architecture-related argparse value is compared against its parser
    default, so a flag that appears on the command line overrides the
    checkpoint metadata while the plain defaults keep the checkpoint's values.
    Returns ``(cfg_kwargs, names)`` ready for ``dataclasses.replace(cfg, **cfg_kwargs)``.
    """
    mapping = [
        ("d_model", "d_model"),
        ("n_heads", "n_heads"),
        ("n_layers", "n_layers"),
        ("fff_depth", "fff_depth"),
        ("uff_k", "top_k"),
        ("vocab_size", "vocab_size"),
        ("max_seq_len", "max_seq_len"),
        ("activation_bits", "activation_bits"),
        ("attention_activation_bits", "attention_activation_bits"),
        ("router_rank", "router_rank"),
        ("tie_weights", "tie_weights"),
        ("no_fff_bias", "fff_bias"),
    ]
    kwargs: dict = {}
    names: list[str] = []
    for arg_name, cfg_name in mapping:
        if getattr(args, arg_name) != getattr(defaults, arg_name):
            value = getattr(args, arg_name)
            if arg_name == "no_fff_bias":
                value = not value
            kwargs[cfg_name] = value
            names.append(arg_name.replace("_", "-"))
    return kwargs, names


def build_history_prompt(
    messages: list[tuple[str, str]],
    tokenizer,
    max_tokens: int,
    vocab_size: int,
) -> list[int]:
    """Assemble ``(role, text)`` turns into one prompt within ``max_tokens`` ids.

    Turns are encoded as ``"{role} > {text}\\n"``. When the transcript exceeds
    the window, the oldest turns are evicted (the most recent user turn is
    always kept); if a single turn alone is still too large it is truncated to
    the last ``max_tokens`` ids so the model always sees the tail of the latest
    message.
    """
    encoded = [tokenizer.encode(f"{role} > {text}\n") for role, text in messages]
    while len(encoded) > 1 and sum(map(len, encoded)) > max_tokens:
        encoded.pop(0)
    flat = [min(i, vocab_size - 1) for chunk in encoded for i in chunk]
    if len(flat) > max_tokens:
        flat = flat[-max_tokens:]
    return flat


def run_turn(
    model: BitNetFFTTransformer,
    tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int | None,
    repetition_penalty: float = 1.0,
    flush_every: int = 1,
) -> tuple[str, dict]:
    """Stream one assistant reply; returns ``(text, stats)``.

    Each generated token is written to stdout and flushed immediately for a
    zero-latency typewriter effect. ``stats`` holds ``tokens``, ``seconds``,
    ``tokens_per_s`` and ``ms_per_token`` for the whole decode phase.
    """
    if hasattr(tokenizer, "reset_stream"):
        tokenizer.reset_stream()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    pieces: list[str] = []
    n = 0
    t0 = time.perf_counter()
    decode_token = getattr(tokenizer, "decode_step", None)
    model.eval()
    with torch.no_grad(), _amp_autocast(device):
        for piece in model.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            decode_token=decode_token,
        ):
            sys.stdout.write(piece)
            if flush_every <= 1 or n % flush_every == flush_every - 1:
                sys.stdout.flush()
            pieces.append(piece)
            n += 1
    mps_synchronize()
    seconds = time.perf_counter() - t0

    if hasattr(tokenizer, "reset_stream"):
        tail = tokenizer.reset_stream()
        if tail:
            sys.stdout.write(tail)
            pieces.append(tail)
    sys.stdout.flush()

    stats = {
        "tokens": n,
        "seconds": seconds,
        "tokens_per_s": n / seconds if seconds > 0 else float("nan"),
        "ms_per_token": seconds * 1e3 / n if n else 0.0,
    }
    return "".join(pieces), stats


def format_stats(stats: dict) -> str:
    return (
        f"[stats] {stats['tokens']} tokens | "
        f"{stats['tokens_per_s']:,.1f} tok/s | "
        f"{stats['ms_per_token']:.1f} ms/token | "
        f"{stats['seconds']:.2f} s"
    )


def chat_loop(
    model: BitNetFFTTransformer,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    messages: list[tuple[str, str]] = []
    if args.system:
        messages.append(("System", args.system))
        print(f"[system] {args.system}")
    print(f"[chat] d_model={model.cfg.d_model} n_heads={model.cfg.n_heads} "
          f"n_layers={model.cfg.n_layers} fff_depth={model.cfg.fff_depth} "
          f"max_seq_len={model.cfg.max_seq_len} device={device}")
    print(_HELP)

    history_budget = model.cfg.max_seq_len - args.max_new_tokens
    while True:
        try:
            raw = input("User > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = raw.strip()
        if not line:
            continue
        if line == "/help":
            print(_HELP)
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/reset":
            messages = [("System", args.system)] if args.system else []
            print("[chat] history cleared")
            continue

        messages.append(("User", line))
        prompt_ids = build_history_prompt(
            messages, tokenizer, history_budget, model.cfg.vocab_size
        )
        if len(prompt_ids) + args.max_new_tokens > model.cfg.max_seq_len:
            raise SystemExit(
                f"prompt_tokens ({len(prompt_ids)}) + max_new_tokens "
                f"({args.max_new_tokens}) exceeds max_seq_len "
                f"({model.cfg.max_seq_len})"
            )
        sys.stdout.write("Assistant > ")
        sys.stdout.flush()
        text, stats = run_turn(
            model, tokenizer, prompt_ids, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=args.eos_token_id,
            repetition_penalty=args.repetition_penalty,
        )
        sys.stdout.write("\n")
        print(format_stats(stats))
        if text:
            messages.append(("Assistant", text))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive chat with a BitNet FFT transformer "
                    "(streaming typewriter decode).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model config")
    g.add_argument("--d-model", type=int, default=128)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument("--fff-depth", type=int, default=3)
    g.add_argument("--uff-k", type=int, default=None,
                   help="activate top-k leaves per token in BitNet-UFF "
                        "(--top-k is taken by sampling); None = classic "
                        "single-leaf FFF. Pair with deep --fff-depth 10/12 "
                        "for ultra-sparse compute")
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
    d.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--compile", action="store_true",
                   help="wrap the model with torch.compile before chatting")

    t = p.add_argument_group("tokenizer")
    t.add_argument("--tokenizer", default=None,
                   help="tokenizer name (default gpt2 BPE; 'bytes' for the "
                        "byte-level fallback; 'o200k_base' for the fixed-vocab "
                        "tiktoken encoding)")

    s = p.add_argument_group("generation")
    s.add_argument("--max-new-tokens", type=int, default=64)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--top-p", type=float, default=0.9)
    s.add_argument("--eos-token-id", type=int, default=None)
    s.add_argument("--repetition-penalty", type=float, default=1.2,
                   help="discount logits of already-sampled tokens "
                        "(1.0 disables)")
    s.add_argument("--system", default=None,
                   help="optional system prompt prepended to the history")
    s.add_argument("--flush-every", type=int, default=1,
                   help="flush stdout every N generated tokens (1 = per token)")
    return p.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace, tok) -> BitNetFFTConfig:
    return BitNetFFTConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        fff_depth=args.fff_depth,
        top_k=args.uff_k,
        max_seq_len=args.max_seq_len,
        activation_bits=args.activation_bits,
        attention_activation_bits=args.attention_activation_bits,
        router_rank=args.router_rank,
        fff_bias=not args.no_fff_bias,
        use_fast_inference=not args.no_fast_inference,
        tie_weights=args.tie_weights,
    ).bind_tokenizer(tok)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    tok = load_tokenizer(args.tokenizer, args.vocab_size)
    if isinstance(tok, TikTokenizer):
        print(f"[tokenizer] tiktoken {tok.name} vocab={tok.vocab_size}")
        args.vocab_size = int(tok.vocab_size)
    elif hasattr(tok, "name"):
        print(f"[tokenizer] BPE {tok.name} vocab={tok.vocab_size}")
        args.vocab_size = int(tok.vocab_size)
    else:
        print(f"[tokenizer] {tok}")

    ckpt_state: dict | None = None
    cfg: BitNetFFTConfig | None = None
    if args.checkpoint:
        ckpt_state = torch.load(args.checkpoint, map_location=device)
        cfg = _checkpoint_config(ckpt_state)
        if cfg is not None:
            saved_vocab = cfg.vocab_size
            overrides, names = _cli_arch_overrides(args, parse_args([]))
            if overrides:
                cfg = dataclasses.replace(cfg, **overrides)
            cfg.use_fast_inference = not args.no_fast_inference
            cfg = cfg.bind_tokenizer(tok)
            print(f"[cfg] architecture loaded from checkpoint "
                  f"(d_model={cfg.d_model} n_heads={cfg.n_heads} "
                  f"n_layers={cfg.n_layers} fff_depth={cfg.fff_depth} "
                  f"top_k={cfg.top_k} "
                  f"max_seq_len={cfg.max_seq_len} vocab={cfg.vocab_size} "
                  f"tie_weights={cfg.tie_weights})")
            if names:
                print(f"[cfg] CLI overrides applied: {', '.join(names)}")
            if saved_vocab != tok.vocab_size:
                print(f"[cfg] WARNING: checkpoint vocab_size={saved_vocab} "
                      f"!= tokenizer vocab_size={tok.vocab_size}")
    if cfg is None:
        cfg = _cfg_from_args(args, tok)

    if args.max_new_tokens >= cfg.max_seq_len:
        raise SystemExit(
            f"prompt_tokens (>= 1) + max_new_tokens ({args.max_new_tokens}) "
            f"exceeds max_seq_len ({cfg.max_seq_len})"
        )

    model = build_model(cfg, device, args.checkpoint, state=ckpt_state)
    if args.compile:
        model = torch.compile(model)
        print("[model] torch.compile enabled")
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    if args.eos_token_id is None:
        args.eos_token_id = cfg.eos_token_id
    if args.eos_token_id is not None:
        print(f"[gen] eos_token_id={args.eos_token_id}")

    return chat_loop(model, tok, device, args)


if __name__ == "__main__":
    raise SystemExit(main())
