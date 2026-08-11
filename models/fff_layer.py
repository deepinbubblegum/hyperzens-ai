"""Fast Feedforward (FFF) linear layer — Belcak & Wattenhofer.

A complete binary decision tree of ``depth`` replaces a dense linear map.
Internal nodes route; leaves apply local affine transforms.

Tree indexing (heap / breadth-first, 0-based):
    - Root: 0
    - Left child of ``i``: ``2*i + 1``
    - Right child of ``i``: ``2*i + 2``
    - Leaf heap indices: ``[2^d - 1, 2^{d+1} - 2]``
    - Contiguous leaf id: ``heap_index - (2^d - 1)``
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


RoutingMode = Literal["soft", "hard"]


class FastFeedforwardLinear(nn.Module):
    """Fast Feedforward linear layer with soft (train) and hard (infer) routing.

    Replaces ``y = x W + b`` with a depth-``d`` binary tree:
        - ``2^d - 1`` router (internal) nodes
        - ``2^d`` leaf affine maps

    Soft mode (training)
    --------------------
    At router ``n``, the right-child probability is
        ``c_n(x) = σ( (w_nᵀ x + b_n) / τ )``
    Path probability of leaf ``ℓ`` is the product of ``c`` / ``(1-c)`` along
    the unique root→leaf path. Output is the mixture
        ``y = Σ_ℓ P(ℓ | x) · (x W_ℓ + b_ℓ)``

    Hard mode (inference)
    ---------------------
    Traverse with the threshold ``(w_nᵀ x + b_n) > 0`` (right if true, else
    left). Only the selected leaf is evaluated — ``O(depth)`` routers + 1 leaf.

    Parameters
    ----------
    in_features:
        Input dim ``D_in``.
    out_features:
        Output dim ``D_out``.
    depth:
        Tree depth ``d``. Leaves = ``2^d``, routers = ``2^d - 1``.
    init_temp:
        Initial temperature ``τ`` for soft sigmoid decisions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 4,
        init_temp: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        if init_temp <= 0.0:
            raise ValueError(f"init_temp must be > 0, got {init_temp}")

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.depth: int = depth
        self.num_leaves: int = 1 << depth  # 2^depth
        self.num_routers: int = self.num_leaves - 1  # 2^depth - 1

        # Routers: (R, D_in), biases (R,)
        self.router_weights = nn.Parameter(torch.empty(self.num_routers, in_features))
        self.router_biases = nn.Parameter(torch.empty(self.num_routers))

        # Leaves: (L, D_in, D_out), biases (L, D_out)
        self.leaf_weights = nn.Parameter(
            torch.empty(self.num_leaves, in_features, out_features)
        )
        self.leaf_biases = nn.Parameter(torch.empty(self.num_leaves, out_features))

        # Not a Parameter — annealing is driven externally via set_temperature.
        self.register_buffer(
            "temperature",
            torch.tensor(float(init_temp)),
            persistent=True,
        )

        # Cached soft-routing stats for compute_balance_loss() after forward_soft.
        # _last_node_decisions: (N, R) — c_n(x) for every router (batched).
        # _last_reach_probs:    (N, R) — prob of visiting each router.
        # _last_leaf_probs:     (N, L) — mixture weights over leaves.
        self._last_node_decisions: Tensor | None = None
        self._last_reach_probs: Tensor | None = None
        self._last_leaf_probs: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize routers near 0 (≈ uniform soft splits) and leaves like Linear.

        Router weights MUST stay tiny. Large ``‖w‖`` makes ``σ(wᵀx/τ)`` saturate to
        0/1 at step 0 → instant tree collapse before leaves can learn.
        """
        # Explicit tiny Gaussian: std=1e-3 (not Xavier / not 0.01).
        nn.init.normal_(self.router_weights, mean=0.0, std=1e-3)
        nn.init.zeros_(self.router_biases)

        # Per-leaf Xavier, matching nn.Linear fan-in/fan-out on (D_in, D_out).
        for leaf in range(self.num_leaves):
            nn.init.xavier_uniform_(self.leaf_weights[leaf])
        nn.init.zeros_(self.leaf_biases)

    def set_temperature(self, temp: float) -> None:
        """Set soft-routing temperature ``τ`` (anneal toward 0 during training).

        Parameters
        ----------
        temp:
            New temperature. Must be ``> 0``.
        """
        if temp <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temp}")
        self.temperature.fill_(float(temp))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"depth={self.depth}, num_routers={self.num_routers}, "
            f"num_leaves={self.num_leaves}, temperature={float(self.temperature):.4g}"
        )

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, mode: RoutingMode = "soft") -> Tensor:
        """Apply FFF transform.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., D_in)``.
        mode:
            ``"soft"`` — differentiable path mixture (training).
            ``"hard"`` — discrete tree walk, single leaf (inference).

        Returns
        -------
        Tensor
            Output of shape ``(..., D_out)``.
        """
        if mode == "soft":
            return self.forward_soft(x)
        if mode == "hard":
            return self.forward_hard(x)
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

    # ------------------------------------------------------------------
    # Soft routing (training)
    # ------------------------------------------------------------------

    def forward_soft(self, x: Tensor) -> Tensor:
        """Soft mixture over all leaves via **log-space** path products.

        Math
        ----
        ``c_n = σ((w_nᵀ x + b_n) / τ)`` — P(go right | node n).
        Path probabilities are accumulated as
            ``log m_left ← log m + log(1-c)``, ``log m_right ← log m + log c``
        using ``logsigmoid`` (avoids ``∏ p_k → 0`` underflow that kills grads).
        Then ``y = Σ_ℓ m_ℓ (x W_ℓ + b_ℓ)`` with ``m = exp(log m)``.

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``

        Caches node decisions / reach probs / leaf mixture for
        :meth:`compute_balance_loss` and :meth:`get_routing_diagnostics`.
        """
        flat, leading = self._flatten_input(x)
        # flat: (N, D_in)

        leaf_probs, node_decisions, reach_probs = self._soft_leaf_probs(flat)
        # leaf_probs:     (N, L)
        # node_decisions: (N, R)
        # reach_probs:    (N, R)

        self._last_leaf_probs = leaf_probs
        self._last_node_decisions = node_decisions
        self._last_reach_probs = reach_probs

        # Per-leaf affine: (N, L, D_out)
        leaf_out = torch.einsum("ni,lio->nlo", flat, self.leaf_weights)
        leaf_out = leaf_out + self.leaf_biases.unsqueeze(0)

        # Soft-weighted sum over leaves → (N, D_out)
        y = torch.einsum("nl,nlo->no", leaf_probs, leaf_out)
        return y.view(*leading, self.out_features)

    def _soft_leaf_probs(
        self, flat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute leaf mixture via log-space path products (numerically stable).

        Parameters
        ----------
        flat:
            ``(N, D_in)``

        Returns
        -------
        leaf_probs:
            ``(N, L)`` — Σ_ℓ leaf_probs[n, ℓ] = 1 (up to float error, then renorm).
        node_decisions:
            ``(N, R)`` — clamped right-child sigmoid at every router.
        reach_probs:
            ``(N, R)`` — probability of visiting each router under soft paths.
        """
        n_batch = flat.shape[0]
        device = flat.device
        dtype = flat.dtype
        eps = 1e-7

        # All router logits at once: (N, R)
        router_logits = (
            F.linear(flat, self.router_weights, self.router_biases)
            / self.temperature
        )
        # Clamp probs for diagnostics / balance; path uses logsigmoid on logits.
        node_decisions = torch.sigmoid(router_logits).clamp(min=eps, max=1.0 - eps)

        # log_mixture: (N, 1) starts at log(1)=0
        log_mixture = torch.zeros(n_batch, 1, device=device, dtype=dtype)
        reach_chunks: list[Tensor] = []
        node_offset = 0

        for level in range(self.depth):
            n_level = 1 << level
            # Soft mass at this frontier (for reach / balance diagnostics)
            reach_chunks.append(log_mixture.exp())

            level_logits = router_logits[:, node_offset : node_offset + n_level]
            # log σ(z) and log(1-σ(z)) = log σ(-z) — stable for deep trees
            log_c = F.logsigmoid(level_logits)
            log_not_c = F.logsigmoid(-level_logits)

            log_left = log_mixture + log_not_c
            log_right = log_mixture + log_c
            log_mixture = torch.stack((log_left, log_right), dim=-1).reshape(
                n_batch, -1
            )
            node_offset += n_level

        reach_probs = torch.cat(reach_chunks, dim=-1)  # (N, R)
        # Exp + renorm guards residual float drift across depth levels.
        leaf_probs = torch.exp(log_mixture)
        leaf_probs = leaf_probs / leaf_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        return leaf_probs, node_decisions, reach_probs

    # ------------------------------------------------------------------
    # Hard routing (inference)
    # ------------------------------------------------------------------

    def forward_hard(self, x: Tensor) -> Tensor:
        """Hard tree traversal — evaluate only the selected leaf.

        For each sample, walk ``depth`` routers with the rule
            ``go_right = (w_nᵀ x + b_n) > 0``
        then apply ``y = x W_ℓ* + b_ℓ*`` for the chosen leaf ``ℓ*``.

        Cost per sample: ``depth`` router dots + 1 leaf matvec (``O(log L)``),
        not ``O(L)``.

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``
        """
        flat, leading = self._flatten_input(x)
        n_batch = flat.shape[0]

        # Batched path indices (vectorized if-else equivalent).
        # node_ids start at root 0; after `depth` steps they are leaf heap ids.
        node_ids = torch.zeros(n_batch, dtype=torch.long, device=flat.device)

        for _ in range(self.depth):
            # Gather active router params: (N, D_in), (N,)
            w = self.router_weights[node_ids]
            b = self.router_biases[node_ids]
            # logit_n = w_nᵀ x + b_n
            logits = torch.einsum("ni,ni->n", flat, w) + b
            go_right = logits > 0  # hard threshold (matches τ → 0 soft limit)
            # left: 2i+1, right: 2i+2
            node_ids = (node_ids << 1) + 1 + go_right.to(torch.long)

        # Map heap leaf index → contiguous leaf id in [0, L)
        leaf_ids = node_ids - (self.num_leaves - 1)  # (N,)

        # Selected leaf transforms: (N, D_in, D_out), (N, D_out)
        w_leaf = self.leaf_weights[leaf_ids]
        b_leaf = self.leaf_biases[leaf_ids]
        y = torch.einsum("ni,nio->no", flat, w_leaf) + b_leaf
        return y.view(*leading, self.out_features)

    def forward_hard_sequential(self, x: Tensor) -> Tensor:
        """Hard routing with explicit Python if-else (single-vector / debug).

        Preferred reference for CPU single-token inference semantics.
        For batched tensors prefers :meth:`forward_hard` (same decisions).

        Parameters
        ----------
        x:
            Shape ``(D_in,)`` or ``(1, D_in)``.

        Returns
        -------
        Tensor
            Shape ``(D_out,)``.
        """
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError(
                    "forward_hard_sequential expects a single vector; "
                    "use forward_hard for batches"
                )
            x = x.squeeze(0)
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError(
                f"expected x shape ({self.in_features},), got {tuple(x.shape)}"
            )

        node = 0
        for _ in range(self.depth):
            logit = torch.dot(self.router_weights[node], x) + self.router_biases[node]
            if bool(logit > 0):
                node = 2 * node + 2  # right
            else:
                node = 2 * node + 1  # left

        leaf = node - (self.num_leaves - 1)
        return x @ self.leaf_weights[leaf] + self.leaf_biases[leaf]

    @torch.no_grad()
    def trace_hard_path(self, x: Tensor) -> dict[str, list[int] | int]:
        """Trace the hard if-else path for a single token vector.

        Parameters
        ----------
        x:
            ``(D_in,)`` or ``(1, D_in)``.

        Returns
        -------
        dict
            ``router_ids``: length ``depth`` (heap indices visited).
            ``leaf_id``: contiguous leaf in ``[0, L)``.
            ``skipped_router_ids``: routers not on the path.
            ``skipped_leaf_ids``: leaves not selected.
        """
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("trace_hard_path expects a single vector")
            x = x.squeeze(0)
        if x.ndim != 1 or x.shape[0] != self.in_features:
            raise ValueError(
                f"expected x shape ({self.in_features},), got {tuple(x.shape)}"
            )

        router_ids: list[int] = []
        node = 0
        for _ in range(self.depth):
            router_ids.append(node)
            logit = torch.dot(self.router_weights[node], x) + self.router_biases[node]
            if bool(logit > 0):
                node = 2 * node + 2
            else:
                node = 2 * node + 1

        leaf_id = node - (self.num_leaves - 1)
        all_routers = set(range(self.num_routers))
        all_leaves = set(range(self.num_leaves))
        return {
            "router_ids": router_ids,
            "leaf_id": leaf_id,
            "skipped_router_ids": sorted(all_routers - set(router_ids)),
            "skipped_leaf_ids": sorted(all_leaves - {leaf_id}),
        }

    def active_params_per_token(self) -> dict[str, int]:
        """Stored vs hard-active parameter counts for this layer."""
        return self.count_active_params(
            self.depth, self.in_features, self.out_features
        )

    # ------------------------------------------------------------------
    # Load balancing + diagnostics
    # ------------------------------------------------------------------

    def compute_balance_loss(self, eps: float = 1e-7) -> Tensor:
        """Entropy / uniform load-balancing loss (anti tree-collapse).

        Uses statistics cached by the latest :meth:`forward_soft` call.

        Router MSE (fair split)
        -----------------------
        ``p_avg[n] = mean_batch σ((w_nᵀx+b_n)/τ)``
        ``L_mse = Σ_n (p_avg[n] - 1/2)²``

        Router entropy maximization
        ---------------------------
        Binary entropy ``H(p) = -p log p - (1-p) log(1-p)`` (max at p=0.5).
        ``L_ent = -mean_n H(p_avg[n])``  (minimize ⇒ push splits toward 0.5)

        Leaf uniformity
        ---------------
        ``L_leaf = Σ_ℓ (E[m_ℓ] - 1/L)²``

        Returns
        -------
        Tensor
            Scalar ``L_mse + L_ent + L_leaf``.
        """
        if (
            self._last_node_decisions is None
            or self._last_leaf_probs is None
        ):
            raise RuntimeError(
                "compute_balance_loss() requires a prior forward_soft() call "
                "to cache routing statistics"
            )

        decisions = self._last_node_decisions  # (N, R)
        leaf_probs = self._last_leaf_probs  # (N, L)

        # Batch-average right-child probability per router: (R,)
        p_avg = decisions.mean(dim=0).clamp(min=eps, max=1.0 - eps)
        node_mse = (p_avg - 0.5).square().sum()

        # Maximize binary entropy at each router (equiv. minimize -H).
        entropy = -(
            p_avg * p_avg.log() + (1.0 - p_avg) * (1.0 - p_avg).log()
        )
        node_entropy_loss = -entropy.mean()

        # Leaf occupancy vs uniform 1/L
        mean_leaf = leaf_probs.mean(dim=0)  # (L,)
        uniform = 1.0 / float(self.num_leaves)
        leaf_loss = (mean_leaf - uniform).square().sum()

        return node_mse + node_entropy_loss + leaf_loss

    def get_routing_diagnostics(
        self,
        active_leaf_threshold: float | None = None,
        eps: float = 1e-8,
    ) -> dict[str, float | list[float]]:
        """Routing health metrics from the latest soft forward (+ optional grads).

        Metrics
        -------
        leaf_utilization_pct:
            % of leaves with mean batch mass ``> threshold``
            (default threshold ``0.5 / L``). Target early training: ``> 80``.
        router_split_ratio_mean / router_split_ratios:
            Batch-mean ``p = σ(·)`` per router (and global mean). Target ≈ 0.5.
        router_grad_norm / leaf_grad_norm:
            ``‖∇‖₂`` of router vs leaf parameters (0 if grads missing).
        leaf_entropy:
            Entropy of mean leaf occupancy (nats); higher ⇒ less collapse.
        balance_loss:
            Current :meth:`compute_balance_loss` value.
        """
        if self._last_node_decisions is None or self._last_leaf_probs is None:
            raise RuntimeError(
                "get_routing_diagnostics() requires a prior forward_soft() call"
            )

        if active_leaf_threshold is None:
            active_leaf_threshold = 0.5 / float(self.num_leaves)

        mean_leaf = self._last_leaf_probs.mean(dim=0)  # (L,)
        active = mean_leaf > active_leaf_threshold
        leaf_utilization_pct = float(active.float().mean().item() * 100.0)

        p_avg = self._last_node_decisions.mean(dim=0)  # (R,)
        router_split_ratios = [float(v) for v in p_avg.detach().cpu().tolist()]
        router_split_ratio_mean = float(p_avg.mean().item())
        router_split_ratio_std = float(p_avg.std(unbiased=False).item())

        # Leaf distribution entropy: -Σ p log p
        p_leaf = mean_leaf.clamp_min(eps)
        p_leaf = p_leaf / p_leaf.sum()
        leaf_entropy = float(-(p_leaf * p_leaf.log()).sum().item())
        max_leaf_entropy = math.log(float(self.num_leaves))
        leaf_entropy_norm = leaf_entropy / max(max_leaf_entropy, eps)

        router_grad_norm = self._param_grad_norm(
            (self.router_weights, self.router_biases)
        )
        leaf_grad_norm = self._param_grad_norm(
            (self.leaf_weights, self.leaf_biases)
        )

        balance = float(self.compute_balance_loss().detach().item())

        return {
            "leaf_utilization_pct": leaf_utilization_pct,
            "router_split_ratio_mean": router_split_ratio_mean,
            "router_split_ratio_std": router_split_ratio_std,
            "router_split_ratios": router_split_ratios,
            "router_grad_norm": router_grad_norm,
            "leaf_grad_norm": leaf_grad_norm,
            "leaf_entropy": leaf_entropy,
            "leaf_entropy_norm": leaf_entropy_norm,
            "balance_loss": balance,
            "num_leaves": float(self.num_leaves),
            "num_active_leaves": float(active.sum().item()),
            "temperature": float(self.temperature.item()),
        }

    @staticmethod
    def _param_grad_norm(params: tuple[Tensor, ...]) -> float:
        """L2 norm of gradients over ``params`` (0 if no grads yet)."""
        sq = 0.0
        found = False
        for p in params:
            if p.grad is None:
                continue
            found = True
            sq += float(p.grad.detach().pow(2).sum().item())
        return math.sqrt(sq) if found else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_input(self, x: Tensor) -> tuple[Tensor, torch.Size]:
        """Collapse leading dims to a batch: ``(..., D_in) → (N, D_in)``."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim in_features={self.in_features}, "
                f"got {x.shape[-1]} (full shape {tuple(x.shape)})"
            )
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        return flat, leading

    def leaf_usage(self) -> Tensor:
        """Mean soft leaf occupancy from the last soft forward — shape ``(L,)``."""
        if self._last_leaf_probs is None:
            raise RuntimeError("leaf_usage() requires a prior forward_soft() call")
        return self._last_leaf_probs.mean(dim=0)

    @staticmethod
    def count_active_params(depth: int, in_features: int, out_features: int) -> dict[str, int]:
        """Parameter counts for soft (all) vs hard (one path) evaluation.

        Soft / stored: all routers + all leaves.
        Hard / active per token: ``depth`` routers + 1 leaf.
        """
        num_leaves = 1 << depth
        num_routers = num_leaves - 1
        router_params = num_routers * (in_features + 1)
        leaf_params = num_leaves * (in_features * out_features + out_features)
        hard_routers = depth * (in_features + 1)
        hard_leaf = in_features * out_features + out_features
        return {
            "stored_total": router_params + leaf_params,
            "hard_active_per_token": hard_routers + hard_leaf,
            "dense_linear": in_features * out_features + out_features,
            "speedup_vs_dense_flops_proxy": (in_features * out_features + out_features)
            / max(hard_routers + hard_leaf, 1),
            "num_routers": num_routers,
            "num_leaves": num_leaves,
        }


def _demo_shapes() -> None:
    """Sanity check used for local smoke tests (not run on import)."""
    layer = FastFeedforwardLinear(32, 64, depth=3, init_temp=1.0)
    x = torch.randn(8, 16, 32)
    y_soft = layer(x, mode="soft")
    y_hard = layer(x, mode="hard")
    loss = layer.compute_balance_loss()
    assert y_soft.shape == (8, 16, 64)
    assert y_hard.shape == (8, 16, 64)
    assert loss.ndim == 0
    # Hard sequential matches batched hard for one vector
    x0 = x[0, 0]
    y_seq = layer.forward_hard_sequential(x0)
    y_b = layer.forward_hard(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_seq, y_b), (y_seq - y_b).abs().max()
    # Soft → hard agreement at low temperature (near one-hot mixture)
    layer.set_temperature(1e-4)
    y_s = layer.forward_soft(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_s, y_b, atol=1e-3), (y_s - y_b).abs().max()
