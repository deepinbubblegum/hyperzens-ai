#!/usr/bin/env python3
"""Sanity checks: Soft CMM (PyTorch) vs Hard CMM (C++), plus agent-loop smoke.

Run
---
    python sanity_cmm.py
    python sanity_cmm.py --seed 0 --n-embd 64 --n-layer 2

Checks
------
1. Layer-level: peaked routers, ``τ → 0`` — Soft ≈ Hard PyTorch ≈ ``cmm_hard_forward_cpu``.
2. Transformer-level: same agreement on ``FFFTransformer`` logits.
3. Agent loop: ``chat_fff_agent.run_agent_turn`` on a tiny HF-shaped CMM model
   (no checkpoint / no Qwen weights) — streaming tokens without shape errors.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from models.fff_layer import (
    FastFeedforwardLinear,
    is_cmm_cpp_available,
    is_fff_cpp_available,
)
from models.transformer import FFFConfig, FFFTransformer


@dataclass
class CheckResult:
    """One numeric agreement check."""

    name: str
    max_abs: float
    ok: bool
    detail: str = ""


def _peak_routers(module: nn.Module, scale: float = 50.0) -> None:
    """Enlarge router logits so the soft mixture is effectively one-hot."""
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, FastFeedforwardLinear):
                m.router_weights.mul_(scale)
                m.set_temperature(1e-4)


def check_layer_soft_vs_hard_cpp(
    *,
    in_features: int = 64,
    out_features: int = 64,
    num_trees: int = 8,
    depth: int = 4,
    batch: int = 6,
    seed: int = 0,
) -> list[CheckResult]:
    """Compare Soft / Hard / C++ CMM on one :class:`FastFeedforwardLinear`."""
    torch.manual_seed(seed)
    layer = FastFeedforwardLinear(
        in_features,
        out_features,
        depth=depth,
        num_trees=num_trees,
        init_temp=1.0,
    )
    layer.eval()
    _peak_routers(layer)
    x = torch.randn(batch, in_features)

    y_soft = layer.forward_soft(x)
    y_hard = layer.forward_hard(x)
    y_cpp = layer.forward_hard_cpp(x)
    y_seq = layer.forward_hard_sequential(x[0])

    results = [
        CheckResult(
            name="layer sequential vs batched hard",
            max_abs=float((y_seq - y_hard[0]).abs().max().item()),
            ok=torch.allclose(y_seq, y_hard[0], atol=1e-5, rtol=1e-5),
        ),
        CheckResult(
            name="layer hard vs C++ hard",
            max_abs=float((y_hard - y_cpp).abs().max().item()),
            ok=torch.allclose(y_hard, y_cpp, atol=1e-4, rtol=1e-4),
            detail=(
                "cmm_hard_forward_cpu"
                if is_cmm_cpp_available()
                else "C++ CMM unavailable — PyTorch fallback"
            ),
        ),
        CheckResult(
            name="layer soft (τ→0) vs C++ hard",
            max_abs=float((y_soft - y_cpp).abs().max().item()),
            ok=torch.allclose(y_soft, y_cpp, atol=2e-3, rtol=2e-3),
        ),
        CheckResult(
            name="layer soft (τ→0) vs PyTorch hard",
            max_abs=float((y_soft - y_hard).abs().max().item()),
            ok=torch.allclose(y_soft, y_hard, atol=2e-3, rtol=2e-3),
        ),
    ]
    return results


@torch.no_grad()
def check_transformer_soft_vs_hard_cpp(
    *,
    n_embd: int = 64,
    n_layer: int = 2,
    n_head: int = 4,
    num_trees: int = 8,
    depth: int = 4,
    seq_len: int = 8,
    seed: int = 0,
) -> list[CheckResult]:
    """Compare Soft vs Hard vs C++ logits on a tiny :class:`FFFTransformer`."""
    torch.manual_seed(seed)
    cfg = FFFConfig(
        vocab_size=128,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=32,
        dropout=0.0,
        fff_depth=depth,
        num_trees=num_trees,
        init_temp=1.0,
        tie_weights=True,
        bias=False,
    )
    model = FFFTransformer(cfg)
    model.eval()
    _peak_routers(model)
    ids = torch.randint(0, cfg.vocab_size, (2, seq_len))

    logits_soft, _ = model(ids, mode="soft")
    logits_hard, _ = model(ids, mode="hard")
    logits_cpp, _ = model(ids, mode="hard_cpp")

    return [
        CheckResult(
            name="transformer hard vs C++ hard",
            max_abs=float((logits_hard - logits_cpp).abs().max().item()),
            ok=torch.allclose(logits_hard, logits_cpp, atol=1e-4, rtol=1e-4),
        ),
        CheckResult(
            name="transformer soft (τ→0) vs C++ hard",
            max_abs=float((logits_soft - logits_cpp).abs().max().item()),
            ok=torch.allclose(logits_soft, logits_cpp, atol=5e-3, rtol=5e-3),
        ),
    ]


class _CausalLMOutput:
    """Minimal HF-style forward output (``logits`` + ``past_key_values``)."""

    def __init__(self, logits: Tensor, past_key_values: Any | None) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class TinyCmmCausalLM(nn.Module):
    """Tiny decoder that looks like a HF CausalLM and uses Multi-Tree CMM.

    Used only to exercise :func:`chat_fff_agent.run_agent_turn` without
    downloading Qwen weights. KV cache is a dummy tuple so decode steps
    pass ``input_ids[:, -1:]`` after the first forward.
    """

    def __init__(self, vocab_size: int = 64, n_embd: int = 32, num_trees: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            max_position_embeddings=128,
            use_cache=True,
        )
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.fff = FastFeedforwardLinear(
            n_embd, n_embd, depth=2, num_trees=num_trees, init_temp=1.0
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        **_kwargs: Any,
    ) -> _CausalLMOutput:
        """``(B, T) → logits (B, T, V)`` via one CMM layer (hard routing)."""
        del attention_mask
        x = self.embed(input_ids)
        x = self.fff(x, mode="hard")
        logits = self.lm_head(x)
        past = (logits.new_zeros(1),) if use_cache else past_key_values
        return _CausalLMOutput(logits, past)


class TinyTokenizer:
    """Whitespace / char tokenizer with a tiny vocab for agent-loop smoke."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Map characters to ids in ``[1, vocab_size)``."""
        del add_special_tokens, truncation
        ids = [(ord(ch) % (self.vocab_size - 1)) + 1 for ch in text]
        if max_length is not None:
            ids = ids[-int(max_length) :]
        return ids or [1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Inverse of :meth:`encode` (lossy ASCII round-trip)."""
        del skip_special_tokens
        chars: list[str] = []
        for i in ids:
            if int(i) <= 0:
                continue
            chars.append(chr(int(i) % 128) if 32 <= int(i) % 128 < 127 else "x")
        return "".join(chars)

    def convert_tokens_to_ids(self, token: str) -> int:
        """Unknown special tokens → ``-1`` (ignored by the agent)."""
        del token
        return -1

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **_kwargs: Any,
    ) -> list[int] | str:
        """Flatten ChatML messages to ids (or a string)."""
        parts = [m.get("content", "") for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        text = "\n".join(parts)
        if tokenize:
            return self.encode(text)
        return text


def check_agent_loop(*, max_new_tokens: int = 8, seed: int = 0) -> CheckResult:
    """Run one ``run_agent_turn`` on :class:`TinyCmmCausalLM` (no HF checkpoint)."""
    from chat_fff_agent import run_agent_turn

    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = TinyCmmCausalLM().to(device)
    model.eval()
    tokenizer = TinyTokenizer()
    history: list[dict[str, str]] = []
    try:
        reply, tok_s, ms_tok, n_new = run_agent_turn(
            model,
            tokenizer,
            device,
            history,
            "ping",
            system="You are a test agent.",
            tools=None,
            max_tool_rounds=0,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=64,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
        )
    except Exception as exc:  # noqa: BLE001 — report as a failed check
        return CheckResult(
            name="agent run_agent_turn (TinyCMM)",
            max_abs=float("inf"),
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )

    ok = n_new > 0 and len(history) == 2
    return CheckResult(
        name="agent run_agent_turn (TinyCMM)",
        max_abs=0.0,
        ok=ok,
        detail=(
            f"n_new={n_new} tok/s={tok_s:.1f} ms/tok={ms_tok:.2f} "
            f"hist={len(history)} reply_len={len(reply)}"
        ),
    )


def _print_results(title: str, results: list[CheckResult]) -> bool:
    print(f"\n{title}")
    print("-" * 72)
    all_ok = True
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        extra = f"  ({r.detail})" if r.detail else ""
        print(f"  [{status}] {r.name}: max|Δ|={r.max_abs:.3e}{extra}")
        all_ok = all_ok and r.ok
    return all_ok


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soft CMM vs Hard C++ CMM sanity checks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--num-trees", type=int, default=8)
    p.add_argument("--depth-per-tree", type=int, default=4)
    return p


def main() -> int:
    args = build_argparser().parse_args()
    print("=" * 72)
    print("CMM sanity — Soft (PyTorch) vs Hard (C++ cmm_hard_forward_cpu)")
    print("=" * 72)
    print(f"  C++ fff_hard: {'yes' if is_fff_cpp_available() else 'NO'}")
    print(f"  C++ cmm_hard_forward_cpu: {'yes' if is_cmm_cpp_available() else 'NO'}")
    print(
        f"  K={args.num_trees}  d_sub={args.depth_per_tree}  "
        f"D={args.n_embd}  L={args.n_layer}"
    )

    layer_ok = _print_results(
        "1) Layer CMM",
        check_layer_soft_vs_hard_cpp(
            in_features=args.n_embd,
            out_features=args.n_embd,
            num_trees=args.num_trees,
            depth=args.depth_per_tree,
            seed=args.seed,
        ),
    )
    xf_ok = _print_results(
        "2) Transformer CMM",
        check_transformer_soft_vs_hard_cpp(
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=max(1, args.n_embd // 16),
            num_trees=args.num_trees,
            depth=args.depth_per_tree,
            seed=args.seed,
        ),
    )
    agent = check_agent_loop(seed=args.seed)
    agent_ok = _print_results("3) Agent loop (chat_fff_agent.run_agent_turn)", [agent])

    all_ok = layer_ok and xf_ok and agent_ok
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
