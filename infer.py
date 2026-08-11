"""CPU hard-routing inference for a trained FFFTransformer.

Loads ``fff_checkpoint.pt``, generates text with ``mode="hard"``, reports
ms/token and FFF active-vs-total parameter counts, and verifies that hard
routing evaluates only the selected tree path (skips unselected branches).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from models.transformer import FFFConfig, FFFTransformer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "fff_checkpoint.pt"


@dataclass
class CharTokenizer:
    """Minimal char tokenizer restored from checkpoint metadata."""

    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size: int = len(self.chars)

    def encode(self, s: str) -> list[int]:
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise ValueError(
                f"prompt contains characters outside the training vocab: {unknown!r}"
            )
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


@dataclass
class ParamStats:
    """Parameter accounting for hard FFF inference."""

    total_model_params: int
    total_fff_params: int
    fff_active_per_token: int  # across all layers, one forward token position
    fff_active_fraction: float
    per_layer_active: int
    per_layer_stored: int
    n_fff_layers: int
    fff_depth: int
    num_leaves: int


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[FFFTransformer, CharTokenizer, dict[str, Any]]:
    """Load model + tokenizer from ``fff_checkpoint.pt`` onto CPU by default."""
    if device is None:
        device = torch.device("cpu")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first with `python train.py`."
        )

    # weights_only=False: checkpoint includes tokenizer metadata / configs.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = FFFConfig(**ckpt["model_config"])
    model = FFFTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tok_meta = ckpt["tokenizer"]
    tokenizer = CharTokenizer(chars=list(tok_meta["chars"]))
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({model_cfg.vocab_size})"
        )
    return model, tokenizer, ckpt


def compute_param_stats(model: FFFTransformer) -> ParamStats:
    """Compare hard-active FFF params/token vs stored model / FFF totals."""
    total_model = sum(p.numel() for p in model.parameters())
    fff_layers = list(model.fff_layers())
    if not fff_layers:
        raise RuntimeError("model has no FastFeedforwardLinear layers")

    per_layer = fff_layers[0].active_params_per_token()
    n_layers = len(fff_layers)
    # Every FFF layer has the same geometry under FFFConfig.
    total_fff = sum(
        layer.router_weights.numel()
        + layer.router_biases.numel()
        + layer.leaf_weights.numel()
        + layer.leaf_biases.numel()
        for layer in fff_layers
    )
    active = per_layer["hard_active_per_token"] * n_layers
    return ParamStats(
        total_model_params=total_model,
        total_fff_params=total_fff,
        fff_active_per_token=active,
        fff_active_fraction=active / max(total_fff, 1),
        per_layer_active=per_layer["hard_active_per_token"],
        per_layer_stored=per_layer["stored_total"],
        n_fff_layers=n_layers,
        fff_depth=fff_layers[0].depth,
        num_leaves=fff_layers[0].num_leaves,
    )


@torch.no_grad()
def verify_hard_skips_unselected_branches(
    model: FFFTransformer,
    sample: Tensor | None = None,
) -> dict[str, Any]:
    """Prove hard routing never reads unselected leaves / off-path routers.

    Method
    ------
    1. Trace the hard path (``depth`` routers + 1 leaf) for a probe vector.
    2. Assert path length / uniqueness invariants.
    3. **Poison test**: scramble every *unselected* leaf (and off-path router).
       Hard output must be unchanged; soft output must change for the same input.
    4. Scramble the *selected* leaf → hard output must change.

    Parameters
    ----------
    model:
        Model under test (mutates FFF weights temporarily, then restores).
    sample:
        Optional probe ``(D,)``. Defaults to a fixed RNG vector on CPU.
    """
    device = next(model.parameters()).device
    layer = next(iter(model.fff_layers()))
    d_model = layer.in_features
    if sample is None:
        g = torch.Generator(device="cpu").manual_seed(0)
        sample = torch.randn(d_model, generator=g).to(device)
    else:
        sample = sample.to(device)

    # --- structural path check ---
    path = layer.trace_hard_path(sample)
    router_ids = path["router_ids"]
    leaf_id = int(path["leaf_id"])
    assert isinstance(router_ids, list)
    if len(router_ids) != layer.depth:
        raise AssertionError(
            f"expected {layer.depth} routers on path, got {len(router_ids)}"
        )
    if len(set(router_ids)) != layer.depth:
        raise AssertionError("hard path revisited a router (invalid tree walk)")
    if leaf_id < 0 or leaf_id >= layer.num_leaves:
        raise AssertionError(f"leaf_id {leaf_id} out of range")
    if len(path["skipped_leaf_ids"]) != layer.num_leaves - 1:
        raise AssertionError("skipped leaf count mismatch")
    if len(path["skipped_router_ids"]) != layer.num_routers - layer.depth:
        raise AssertionError("skipped router count mismatch")

    # Backup weights
    leaf_w = layer.leaf_weights.data.clone()
    leaf_b = layer.leaf_biases.data.clone()
    router_w = layer.router_weights.data.clone()
    router_b = layer.router_biases.data.clone()

    x = sample.unsqueeze(0)  # (1, D)
    y_hard_ref = layer.forward_hard(x).clone()
    y_soft_ref = layer.forward_soft(x).clone()

    try:
        # Poison all unselected leaves + off-path routers with large noise.
        noise = 10.0
        skipped_leaves = path["skipped_leaf_ids"]
        skipped_routers = path["skipped_router_ids"]
        assert isinstance(skipped_leaves, list) and isinstance(skipped_routers, list)

        layer.leaf_weights.data[skipped_leaves] = (
            leaf_w[skipped_leaves] + noise * torch.randn_like(leaf_w[skipped_leaves])
        )
        layer.leaf_biases.data[skipped_leaves] = (
            leaf_b[skipped_leaves] + noise * torch.randn_like(leaf_b[skipped_leaves])
        )
        if skipped_routers:
            layer.router_weights.data[skipped_routers] = (
                router_w[skipped_routers]
                + noise * torch.randn_like(router_w[skipped_routers])
            )
            layer.router_biases.data[skipped_routers] = (
                router_b[skipped_routers]
                + noise * torch.randn_like(router_b[skipped_routers])
            )

        y_hard_poison_other = layer.forward_hard(x)
        y_soft_poison_other = layer.forward_soft(x)

        if not torch.allclose(y_hard_poison_other, y_hard_ref, atol=1e-5, rtol=1e-5):
            raise AssertionError(
                "hard output changed after poisoning unselected branches — "
                "unselected leaves/routers were incorrectly evaluated"
            )
        # Soft mixture must depend on poisoned leaves (with overwhelming probability).
        if torch.allclose(y_soft_poison_other, y_soft_ref, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                "soft output unchanged after poisoning other leaves — "
                "unexpected; soft mode should mix all leaves"
            )

        # Restore, then poison the selected leaf only.
        layer.leaf_weights.data.copy_(leaf_w)
        layer.leaf_biases.data.copy_(leaf_b)
        layer.router_weights.data.copy_(router_w)
        layer.router_biases.data.copy_(router_b)

        layer.leaf_weights.data[leaf_id] = (
            leaf_w[leaf_id] + noise * torch.randn_like(leaf_w[leaf_id])
        )
        layer.leaf_biases.data[leaf_id] = (
            leaf_b[leaf_id] + noise * torch.randn_like(leaf_b[leaf_id])
        )
        y_hard_poison_selected = layer.forward_hard(x)
        if torch.allclose(y_hard_poison_selected, y_hard_ref, atol=1e-5, rtol=1e-5):
            raise AssertionError(
                "hard output unchanged after poisoning the selected leaf — "
                "selected leaf was not evaluated"
            )
    finally:
        layer.leaf_weights.data.copy_(leaf_w)
        layer.leaf_biases.data.copy_(leaf_b)
        layer.router_weights.data.copy_(router_w)
        layer.router_biases.data.copy_(router_b)

    # Batched hard must match sequential if-else on the same vector.
    y_seq = layer.forward_hard_sequential(sample)
    y_batch = layer.forward_hard(sample.unsqueeze(0)).squeeze(0)
    if not torch.allclose(y_seq, y_batch, atol=1e-5, rtol=1e-5):
        raise AssertionError("forward_hard and forward_hard_sequential disagree")

    return {
        "ok": True,
        "router_ids": router_ids,
        "leaf_id": leaf_id,
        "num_routers_evaluated": layer.depth,
        "num_routers_skipped": layer.num_routers - layer.depth,
        "num_leaves_evaluated": 1,
        "num_leaves_skipped": layer.num_leaves - 1,
        "layers_checked": 1,
    }


class FFFGenerator:
    """Hard-routing text generator bound to a loaded checkpoint."""

    def __init__(
        self,
        model: FFFTransformer,
        tokenizer: CharTokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        self.param_stats = compute_param_stats(model)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Autoregressive generation with ``mode="hard"`` FFF routing.

        Parameters
        ----------
        prompt:
            Conditioning text (must use training vocabulary characters).
        max_new_tokens:
            Number of new characters to sample.
        temperature:
            Softmax temperature over LM logits (not FFF tree temperature).
        top_k:
            Optional top-k filtering on LM logits.

        Returns
        -------
        str
            ``prompt + generated continuation``.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        ids = self.tokenizer.encode(prompt)
        if not ids:
            # Empty prompt → start from newline if available, else first char.
            bos = self.tokenizer.stoi.get("\n", 0)
            ids = [bos]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.model.config.block_size
                else input_ids[:, -self.model.config.block_size :]
            )
            # Strict hard routing for every FFF block.
            logits, _ = self.model(idx_cond, mode="hard")
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

        return self.tokenizer.decode(input_ids[0])

    @torch.no_grad()
    def generate_with_stats(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        warmup: int = 2,
    ) -> tuple[str, dict[str, float | int]]:
        """Generate and measure wall-clock ms/token on CPU (hard mode)."""
        # Warmup (not timed) to stabilize allocations / one-time costs.
        if warmup > 0 and max_new_tokens > 0:
            self.generate(prompt, max_new_tokens=min(warmup, max_new_tokens), temperature=temperature, top_k=top_k)

        t0 = time.perf_counter()
        text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed_s = time.perf_counter() - t0
        ms_per_token = (elapsed_s * 1000.0) / max(max_new_tokens, 1)
        stats: dict[str, float | int] = {
            "max_new_tokens": max_new_tokens,
            "elapsed_s": elapsed_s,
            "ms_per_token": ms_per_token,
            "tokens_per_s": max_new_tokens / max(elapsed_s, 1e-12),
            "total_model_params": self.param_stats.total_model_params,
            "total_fff_params": self.param_stats.total_fff_params,
            "fff_active_params_per_token": self.param_stats.fff_active_per_token,
            "fff_active_fraction": self.param_stats.fff_active_fraction,
            "fff_depth": self.param_stats.fff_depth,
            "num_leaves": self.param_stats.num_leaves,
            "n_fff_layers": self.param_stats.n_fff_layers,
        }
        return text, stats


def print_stats(stats: dict[str, float | int], param_stats: ParamStats) -> None:
    """Pretty-print timing and hard-routing parameter accounting."""
    print("\n=== Hard-routing inference stats (CPU) ===")
    print(f"Time per token:     {stats['ms_per_token']:.3f} ms/token")
    print(f"Throughput:         {stats['tokens_per_s']:.2f} tokens/s")
    print(f"Generated tokens:   {stats['max_new_tokens']}")
    print(f"Wall time:          {stats['elapsed_s']:.3f} s")
    print("--- Parameters ---")
    print(f"Total model params: {param_stats.total_model_params:,}")
    print(f"Total FFF params:   {param_stats.total_fff_params:,}")
    print(
        f"FFF evaluated/token:{param_stats.fff_active_per_token:,} "
        f"({100.0 * param_stats.fff_active_fraction:.2f}% of FFF params)"
    )
    print(
        f"  per FFF layer:    {param_stats.per_layer_active:,} active / "
        f"{param_stats.per_layer_stored:,} stored "
        f"(depth={param_stats.fff_depth}, leaves={param_stats.num_leaves}, "
        f"active = {param_stats.fff_depth} routers + 1 leaf)"
    )
    print(
        f"  across {param_stats.n_fff_layers} layers: "
        f"skips {(param_stats.num_leaves - 1) * param_stats.n_fff_layers} leaves/token"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hard-routing CPU inference for FFFTransformer")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to fff_checkpoint.pt",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:\n",
        help="Text prompt (characters must be in the training vocab)",
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip hard-routing branch-skip verification",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"Loading checkpoint (CPU): {args.checkpoint}")
    model, tokenizer, ckpt = load_checkpoint(Path(args.checkpoint), device=device)
    step = ckpt.get("step", "?")
    print(
        f"Loaded step={step} | vocab={tokenizer.vocab_size} | "
        f"layers={model.config.n_layer} | d_model={model.config.n_embd} | "
        f"fff_depth={model.config.fff_depth}"
    )

    generator = FFFGenerator(model, tokenizer, device=device)

    if not args.skip_verify:
        print("\nVerifying hard routing skips unselected branches...")
        result = verify_hard_skips_unselected_branches(model)
        print(
            f"OK — evaluated {result['num_routers_evaluated']} routers + "
            f"{result['num_leaves_evaluated']} leaf; "
            f"skipped {result['num_routers_skipped']} routers + "
            f"{result['num_leaves_skipped']} leaves "
            f"(path routers={result['router_ids']}, leaf={result['leaf_id']})"
        )

    text, stats = generator.generate_with_stats(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n=== Generated text ===")
    print(text)
    print_stats(stats, generator.param_stats)


# Module-level convenience matching the requested API when used as a library.
_default_generator: FFFGenerator | None = None


def generate(prompt: str, max_new_tokens: int = 100) -> str:
    """Generate with hard routing using the default ``fff_checkpoint.pt`` on CPU."""
    global _default_generator
    if _default_generator is None:
        model, tokenizer, _ = load_checkpoint(DEFAULT_CHECKPOINT, device=torch.device("cpu"))
        _default_generator = FFFGenerator(model, tokenizer)
    return _default_generator.generate(prompt, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    main()
