"""Fast Feedforward + BitNet b1.58 hybrid layer.

A balanced binary decision tree routes every token to exactly one of
``2**depth`` leaves by evaluating only ``depth == log2(num_leaves)`` decision
nodes per token. Each leaf is a BitNet b1.58 projection (ternary weights,
8-bit activations) of width ``d_out``, so the layer's total capacity is
``2**depth * d_out`` but only a ``1 / 2**depth`` fraction of the leaf FLOPs is
executed. With ``depth = log2(d_out)`` (the default) there are exactly
``log2(d_out)`` decision paths evaluated per token.

Reference: Belcak & Wattenhofer, "Fast Feedforward Networks" (NeurIPS 2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .bitlinear import absmax_quantize, ste_ternarize

__all__ = ["FastFeedForwardBitNet"]


class FastFeedForwardBitNet(nn.Module):
    """Conditionally-executing FFN with ternary BitNet leaves.

    Routing:
        A heap-indexed binary tree over ``num_leaves - 1`` decision nodes.
        Node ``n`` has children ``2n+1`` (left) and ``2n+2`` (right). At each
        level only the node reached by the running path is evaluated:
        ``r = sigmoid(<x, w_n> + b_n)`` and the token descends right when
        ``r >= 0.5``. After ``depth`` steps the token owns a leaf index.

        Routing is hard at forward time; in training mode a straight-through
        term re-injects gradient into the traversed decision logits via the
        path probability.

    Leaves:
        Each leaf is a BitNet b1.58 projection: weights quantized to
        {-1, 0, +1} by AbsMean (STE) applied to AbsMax-quantized inputs
        through a single per-token ``bmm``. Only the selected leaf rows are
        gathered, so only one leaf path (``log2(d_out)`` decision nodes when
        ``depth = log2(d_out)``) is executed per token.

    Args:
        d_in: input feature dimension.
        d_out: output feature dimension (equal to the per-leaf width).
        depth: tree depth; defaults to ``log2(d_out)`` when ``d_out`` is a
            power of two.
        bias: use leaf biases.
        activation_bits: BitNet activation quantization width.
        router_rank: ``"full"`` for a dense decision node or ``"r1"`` for the
            rank-1 factorization ``(u^T x)(v^T x)``.
        chunk_size: if set, the leaf gather + ``bmm`` is processed in batch
            chunks to bound peak unified-memory footprint.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int | None = None,
        depth: int | None = None,
        bias: bool = True,
        activation_bits: int = 8,
        router_rank: str = "full",
        chunk_size: int | None = None,
        eps: float = 1e-8,
        ternarize_threshold_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if router_rank not in ("full", "r1"):
            raise ValueError(f"router_rank must be 'full' or 'r1', got {router_rank!r}")

        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        if ternarize_threshold_scale <= 0:
            raise ValueError(
                f"ternarize_threshold_scale must be positive, got {ternarize_threshold_scale}"
            )

        if depth is None:
            if self.d_out & (self.d_out - 1):
                raise ValueError(
                    "d_out must be a power of two when depth is inferred, "
                    f"got d_out={self.d_out}"
                )
            depth = int(math.log2(self.d_out))
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")

        self.depth = depth
        self.num_leaves = 1 << depth
        self.activation_bits = activation_bits
        self.router_rank = router_rank
        self.chunk_size = chunk_size
        self.eps = eps
        self.ternarize_threshold_scale = ternarize_threshold_scale

        self.leaf_weight = nn.Parameter(
            torch.empty(self.num_leaves, self.d_out, self.d_in)
        )
        if bias:
            self.leaf_bias = nn.Parameter(torch.empty(self.num_leaves, self.d_out))
        else:
            self.register_parameter("leaf_bias", None)

        num_nodes = self.num_leaves - 1
        if router_rank == "full":
            self.router_weight = nn.Parameter(torch.empty(num_nodes, self.d_in))
            self.router_bias = nn.Parameter(torch.empty(num_nodes))
        else:
            self.router_u = nn.Parameter(torch.empty(num_nodes, self.d_in))
            self.router_v = nn.Parameter(torch.empty(num_nodes, self.d_in))
            self.register_parameter("router_weight", None)
            self.register_parameter("router_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.leaf_weight, std=0.02)
        if self.leaf_bias is not None:
            nn.init.zeros_(self.leaf_bias)
        if self.router_rank == "full":
            nn.init.normal_(self.router_weight, std=0.02)
            nn.init.zeros_(self.router_bias)
        else:
            nn.init.normal_(self.router_u, std=0.02)
            nn.init.normal_(self.router_v, std=0.02)

    @property
    def n_routing_nodes(self) -> int:
        return self.num_leaves - 1

    def leaf_gather_temp_bytes(self, batch: int) -> int:
        """Worst-case bytes of the gathered leaf weight intermediate for a batch."""
        rows = batch if self.chunk_size is None else min(batch, self.chunk_size)
        return rows * self.d_out * self.d_in * 4

    def to_dense(self) -> nn.Linear:
        """Reference dense FFN with the same leaf capacity (``num_leaves*d_out``)."""
        hidden = self.num_leaves * self.d_out
        dense = nn.Linear(self.d_in, hidden, bias=self.leaf_bias is not None)
        with torch.no_grad():
            dense.weight.copy_(self.leaf_weight.detach().reshape(hidden, self.d_in))
            if self.leaf_bias is not None:
                dense.bias.copy_(self.leaf_bias.detach().reshape(hidden))
        return dense

    def _routing_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard routing -> (leaf_index per token, path probability per token)."""
        device, dtype = x.device, x.dtype
        batch = x.shape[0]
        node = torch.zeros(batch, dtype=torch.long, device=device)
        path_prob = torch.ones(batch, dtype=dtype, device=device)
        for _ in range(self.depth):
            if self.router_rank == "r1":
                u = self.router_u[node]
                v = self.router_v[node]
                logit = (x * u).sum(-1) * (x * v).sum(-1)
            else:
                w = self.router_weight[node]
                logit = (x * w).sum(-1) + self.router_bias[node]
            r = torch.sigmoid(logit)
            go_right = r >= 0.5
            path_prob = path_prob * torch.where(go_right, r, 1.0 - r)
            node = 2 * node + 1 + go_right.to(torch.long)
        return node - (self.num_leaves - 1), path_prob

    def _leaf_projection(self, x: torch.Tensor, leaf_idx: torch.Tensor) -> torch.Tensor:
        """Gather only the selected leaves and apply the ternary projection."""
        batch = x.shape[0]
        wq = ste_ternarize(
            self.leaf_weight,
            eps=self.eps,
            threshold_scale=self.ternarize_threshold_scale,
        )
        if self.activation_bits is None or self.activation_bits < 32:
            xq = absmax_quantize(x, bits=self.activation_bits or 32, eps=self.eps)
        else:
            xq = x
        if self.chunk_size is None or batch <= self.chunk_size:
            w_sel = wq[leaf_idx]
            out = torch.bmm(xq.unsqueeze(1), w_sel.transpose(-1, -2)).squeeze(1)
            if self.leaf_bias is not None:
                out = out + self.leaf_bias[leaf_idx]
            return out
        out = torch.empty((batch, self.d_out), device=x.device, dtype=x.dtype)
        for start in range(0, batch, self.chunk_size):
            end = min(start + self.chunk_size, batch)
            w_sel = wq[leaf_idx[start:end]]
            piece = torch.bmm(
                xq[start:end].unsqueeze(1), w_sel.transpose(-1, -2)
            ).squeeze(1)
            if self.leaf_bias is not None:
                piece = piece + self.leaf_bias[leaf_idx[start:end]]
            out[start:end] = piece
        return out

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        leaf_idx, path_prob = self._routing_forward(x)
        out = self._leaf_projection(x, leaf_idx)
        if self.training:
            gate = (path_prob - path_prob.detach()).unsqueeze(-1)
            out = out + gate * out.detach()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            out = self._forward(x.reshape(-1, shape[-1]))
            return out.reshape(*shape[:-1], self.d_out)
        return self._forward(x)

    def pack(self, device: torch.device | str | None = None) -> "PackedFFFEvaluator":
        """Build a packed-ternary evaluator for fast inference on ``device``."""
        from .fast_inference import PackedFFFEvaluator

        return PackedFFFEvaluator(self, device=device)

    def fast_forward(
        self,
        x: torch.Tensor,
        chunk_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Zero-gather inference: routes, then evaluates only the selected leaves.

        The first call packs + uploads the ternary weights (lazy); subsequent
        calls reuse them. Falls back to ``_forward`` if the native extension is
        unavailable.
        """
        from .fast_inference import extension_available

        if not extension_available():
            return self.forward(x)
        evaluator = getattr(self, "_packed_eval", None)
        if evaluator is None or evaluator.device != torch.device(x.device):
            evaluator = self.pack(device=x.device)
            evaluator.chunk_size = chunk_size
            self._packed_eval = evaluator
        if x.dim() > 2:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            leaf_idx, _ = self._routing_forward(flat)
            out = evaluator(flat, leaf_idx)
            return out.reshape(*shape[:-1], self.d_out)
        leaf_idx, _ = self._routing_forward(x)
        return evaluator(x, leaf_idx)

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, depth={self.depth}, "
            f"num_leaves={self.num_leaves}, router_rank={self.router_rank!r}, "
            f"chunk_size={self.chunk_size}, "
            f"ternarize_threshold_scale={self.ternarize_threshold_scale}"
        )
