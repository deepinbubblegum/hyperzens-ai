"""BitNet-UFF: Ultra-Fast Feedforward with Vectorized Flat Indexing.

``BitNetUFFLayer`` is the deep-tree, ultra-sparse successor to
:class:`~bitnet_fff.fast_fff.FastFeedForwardBitNet`. It keeps the same
bitnet-style 1.58-bit ternary parameters but replaces the classic FFF's
sequential (per-level) tree walk with a **single flat pass** over all routing
nodes:

1. **One flat matmul routes every token.** ``logits = x @ W_router^T +
   b_router`` scores all ``2**depth - 1`` decision nodes in a single dense
   ``(batch, nodes)`` operation — there is no loop over ``depth`` and no
   recursion. With ``--fff-depth 12`` that is 4095 node logits computed
   together.

2. **Vectorized flat gather/scatter over the heap.** Two precomputed constant
   index tables — ``path_nodes[leaf, level]`` (the ``(num_leaves, depth)``
   heap-id table) and ``path_sign[leaf, level]`` (the ``±1`` direction each
   leaf's path needs) — turn the flat node logits into every leaf's path
   log-probability in one advanced-index gather:
   ``leaf_logp = -softplus(-sign * logits[:, path_nodes]).reshape(batch,
   leaves, depth).sum(-1)``. This is the exact product of the router's
   Bernoulli decision probabilities along each leaf's path (sum of log-probs),
   with no per-token branching and no per-level gather loop. As the router
   becomes confident (logits -> +/-inf) the top-1 leaf converges to the
   classic hard-routed leaf, so ``k=1`` reproduces single-leaf FFF routing.

3. **Top-k sparse activation.** Per token only the ``k`` most probable leaves
   (default ``k=8``) are activated. Of ``num_leaves = 2**depth`` leaves,
   ``k / num_leaves`` rows execute: 8 of 4096 leaves at ``depth=12`` is
   ``~0.2%`` of leaf capacity active. Leaf projections are gathered by flat
   indexing (the backward pass is an ``index_add`` scatter, so only the
   selected rows receive gradients) and combined with a softmax path
   weight, which gives the router a differentiable training signal.

4. **Bounded activation memory.** The two flat gathers — the ``(k, d_out,
   d_in)`` leaf-weight slice per token and the ``(num_leaves, depth)``
   path-node slice — are processed in mini-chunks of ``_DEFAULT_CHUNK``
   (256) tokens, and the chunk is always clamped so the worst-case single
   gather stays under ``_GATHER_BUDGET_BYTES`` (256 MB). A full
   ``(B*S, k, d_out, d_in)`` 4D tensor is therefore never allocated, keeping
   per-layer forward activation memory well under 500 MB during training.

5. **BitNet 1.58-bit arithmetic.** Router ("node") projections and leaf
   weights are AbsMean-ternarized to {-1, 0, +1} with a straight-through
   estimator; activations use per-token AbsMax k-bit quantization on the leaf
   path, mirroring :class:`FastFeedForwardBitNet`.

All indexing, matmul, ``topk`` and ``softmax`` operations are native torch
ops that run under CUDA autocast (fp16/bf16) and on MPS devices.

Reference: Belcak & Wattenhofer, "Fast Feedforward Networks" (NeurIPS 2023),
with the flat-indexing / top-k extension of "Ultra-Fast Feedforward" layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import absmax_quantize, ste_ternarize

__all__ = ["BitNetUFFLayer"]


class BitNetUFFLayer(nn.Module):
    """Ultra-sparse feedforward with a single flat routing matmul + top-k leaves.

    Args:
        d_in: input feature dimension.
        d_out: output feature dimension (equal to the per-leaf width).
        depth: tree depth; ``num_leaves = 2**depth``. Deep trees (``10`` or
            ``12`` -> 1024 / 4096 leaves) are the intended regime; only ``k``
            leaves execute per token so compute stays ultra-sparse.
        k: number of top-k leaves activated per token (default 8).
        bias: use leaf biases.
        activation_bits: BitNet activation quantization width applied to the
            leaf-projection input (``None`` or ``>= 32`` disables it).
        chunk_size: maximum tokens processed per chunk in the routing / leaf
            projection pass. If unset, ``_DEFAULT_CHUNK`` (256) is used, and
            the chunk is always clamped so the worst-case single gather stays
            under ``_GATHER_BUDGET_BYTES`` (256 MB), keeping per-layer forward
            activation memory well under 500 MB during training.
        eps: numerical stability for quantizers.
        ternarize_threshold_scale: multiplier on the AbsMean ternary threshold
            (see :func:`absmean_ternarize`).

    Weight layout (heap tree):
        ``router_weight (num_leaves - 1, d_in)``, ``router_bias
        (num_leaves - 1)`` for the flat routing matmul, and ``leaf_weight
        (num_leaves, d_out, d_in)`` (+ optional ``leaf_bias``) for the leaf
        projections. The heap-id path table is a non-persistent ``long``
        buffer that travels with the module across devices.
    """

    # Token mini-chunk and single-gather byte budget used to bound activation
    # memory (see ``_chunk_size``). Tune for smaller GPUs as needed.
    _DEFAULT_CHUNK = 256
    _GATHER_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        d_in: int,
        d_out: int | None = None,
        depth: int = 10,
        k: int = 8,
        bias: bool = True,
        activation_bits: int = 8,
        chunk_size: int | None = None,
        eps: float = 1e-8,
        ternarize_threshold_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if ternarize_threshold_scale <= 0:
            raise ValueError(
                f"ternarize_threshold_scale must be positive, got "
                f"{ternarize_threshold_scale}"
            )

        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.depth = int(depth)
        self.num_leaves = 1 << self.depth
        self.k = min(int(k), self.num_leaves)
        self.activation_bits = activation_bits
        self.chunk_size = chunk_size
        self.eps = eps
        self.ternarize_threshold_scale = ternarize_threshold_scale

        num_nodes = self.num_leaves - 1
        self.router_weight = nn.Parameter(torch.empty(num_nodes, d_in))
        self.router_bias = nn.Parameter(torch.empty(num_nodes))
        self.leaf_weight = nn.Parameter(
            torch.empty(self.num_leaves, self.d_out, self.d_in)
        )
        if bias:
            self.leaf_bias = nn.Parameter(
                torch.empty(self.num_leaves, self.d_out)
            )
        else:
            self.register_parameter("leaf_bias", None)

        # Constant heap-id table: node at level i on the path to leaf l is
        # (2**i - 1) + (l >> (depth - i)); the matching path direction is the
        # bit at position (depth - 1 - i) of l. Both are flattened/signed so a
        # single gather yields every leaf's path score.
        levels = torch.arange(self.depth, dtype=torch.long)
        offsets = (1 << levels) - 1
        leaf_ids = torch.arange(self.num_leaves, dtype=torch.long)[:, None]
        path_nodes = offsets[None, :] + (leaf_ids >> (self.depth - levels))
        bits = (leaf_ids >> (self.depth - 1 - levels)) & 1  # (leaves, depth)
        path_sign = bits.to(torch.float) * 2 - 1
        self.register_buffer(
            "_path_nodes", path_nodes.reshape(-1), persistent=False
        )
        self.register_buffer("_path_sign", path_sign, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router_weight, std=0.02)
        nn.init.zeros_(self.router_bias)
        nn.init.normal_(self.leaf_weight, std=0.02)
        if self.leaf_bias is not None:
            nn.init.zeros_(self.leaf_bias)

    @property
    def n_routing_nodes(self) -> int:
        return self.num_leaves - 1

    @property
    def active_fraction(self) -> float:
        """Fraction of leaf rows executed per token (``k / num_leaves``)."""
        return self.k / self.num_leaves

    def leaf_gather_temp_bytes(self, batch: int) -> int:
        """Worst-case bytes of the gathered intermediates for a batch."""
        rows = self._chunk_size(batch)
        per_row = (
            self.k * self.d_out * self.d_in + self.num_leaves * self.depth
        )
        return rows * per_row * 4

    def _chunk_size(self, batch: int, elem_bytes: int = 4) -> int:
        """Per-chunk token count that bounds this layer's activation spikes.

        The flat UFF design materializes two activation-heavy gathers per
        token: the ``(k, d_out, d_in)`` leaf-weight slice and the
        ``(num_leaves * depth)`` path-node slice. ``_chunk_size`` returns the
        largest chunk (``chunk_size`` if set, else ``_DEFAULT_CHUNK``) that
        keeps the worst-case single gather under ``_GATHER_BUDGET_BYTES``, so
        a full ``(B*S, k, d_out, d_in)`` 4D tensor is never allocated.
        """
        per_row = max(
            self.k * self.d_out * self.d_in,  # 4D leaf-weight gather
            self.num_leaves * self.depth,  # flat path-node gather (routing)
        )
        budget_rows = max(1, self._GATHER_BUDGET_BYTES // (per_row * elem_bytes))
        chunk = self.chunk_size or self._DEFAULT_CHUNK
        return min(chunk, budget_rows, batch)

    def _routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat single-pass routing -> ``(top-k leaf ids, softmax weights)``.

        Computes every node logit with one dense matmul, then gathers all
        ``num_leaves`` leaf path log-probabilities through the flattened
        heap-id / direction-sign tables in a single advanced-index op:
        ``leaf_logp[l] = sum_level -softplus(-sign(l, level) * logit[path_node(
        l, level)])``. This is the router's Bernoulli path probability for
        every leaf (``log P(left) = -softplus(+logit)``,
        ``log P(right) = -softplus(-logit)``), so the top ``k`` leaves are the
        k most probable routes under the router's decisions.

        Returns ``top_idx`` as ``(batch, k)`` long ids and ``w`` as the
        ``(batch, k)`` softmax of the selected path log-probabilities.
        """
        batch = x.shape[0]

        wq_router = ste_ternarize(
            self.router_weight,
            eps=self.eps,
            threshold_scale=self.ternarize_threshold_scale,
        )
        logits = F.linear(x, wq_router, self.router_bias)  # (batch, nodes)

        logits_g = logits[:, self._path_nodes]  # (batch, leaves * depth)
        logits_g = logits_g.reshape(batch, self.num_leaves, self.depth)
        leaf_logp = -F.softplus(-self._path_sign * logits_g).sum(dim=-1)

        top_logp, top_idx = torch.topk(leaf_logp, self.k, dim=-1)
        w = torch.softmax(top_logp, dim=-1)
        return top_idx, w

    def _leaf_projection(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        w: torch.Tensor,
        wq_leaf: torch.Tensor,
    ) -> torch.Tensor:
        """Gather only the top-k leaves and combine their ternary projections.

        Tokens are processed in mini-chunks (≤ ``_DEFAULT_CHUNK`` and always
        clamped by ``_GATHER_BUDGET_BYTES``), so the gathered
        ``(chunk, k, d_out, d_in)`` weight tensor — which autograd also saves
        for the matmul backward during training — never grows with ``B*S``.
        """
        batch = x.shape[0]
        device, dtype = x.device, x.dtype
        if self.activation_bits is not None and self.activation_bits < 32:
            xq = absmax_quantize(x, bits=self.activation_bits, eps=self.eps)
        else:
            xq = x

        out = torch.empty((batch, self.d_out), device=device, dtype=dtype)
        chunk = self._chunk_size(batch, wq_leaf.element_size())
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            w_sel = wq_leaf[top_idx[start:end]]  # (c, k, d_out, d_in)
            proj = torch.matmul(
                w_sel.transpose(-1, -2), xq[start:end, None, :, None]
            ).squeeze(-1)  # (c, k, d_out, 1) -> (c, k, d_out)
            if self.leaf_bias is not None:
                proj = proj + self.leaf_bias[top_idx[start:end]]
            out[start:end] = (proj * w[start:end].unsqueeze(-1)).sum(dim=1)
        return out

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        wq_leaf = ste_ternarize(
            self.leaf_weight,
            eps=self.eps,
            threshold_scale=self.ternarize_threshold_scale,
        )
        batch = x.shape[0]
        chunk = self._chunk_size(batch, x.element_size())
        if batch <= chunk:
            top_idx, w = self._routing(x)
            return self._leaf_projection(x, top_idx, w, wq_leaf)
        out = torch.empty((batch, self.d_out), device=x.device, dtype=x.dtype)
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            top_idx, w = self._routing(x[start:end])
            out[start:end] = self._leaf_projection(
                x[start:end], top_idx, w, wq_leaf
            )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            shape = x.shape
            out = self._forward(x.reshape(-1, shape[-1]))
            return out.reshape(*shape[:-1], self.d_out)
        return self._forward(x)

    def fast_forward(
        self,
        x: torch.Tensor,
        chunk_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Inference entry point (interface-compatible with the classic FFF).

        UFF has no separate packed evaluator: the flat routing matmul is
        already a single native call, so this just runs the reference path.
        """
        return self.forward(x)

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, depth={self.depth}, "
            f"num_leaves={self.num_leaves}, k={self.k} "
            f"({self.active_fraction:.2%} active), "
            f"chunk_size={self.chunk_size}, "
            f"ternarize_threshold_scale={self.ternarize_threshold_scale}"
        )
