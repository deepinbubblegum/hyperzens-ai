"""Fast Feedforward (FFF) linear layer — Belcak & Wattenhofer.

A complete binary decision tree of ``depth`` replaces a dense linear map.
Internal nodes route; leaves apply local affine transforms.

Tree indexing (heap / breadth-first, 0-based):
    - Root: 0
    - Left child of ``i``: ``2*i + 1``
    - Right child of ``i``: ``2*i + 2``
    - Leaf heap indices: ``[2^d - 1, 2^{d+1} - 2]``
    - Contiguous leaf id: ``heap_index - (2^d - 1)``

Hard routing on CPU can use a JIT-compiled C++ extension
(``csrc/fff_hard.cpp``) via :meth:`forward_hard_cpp`, with automatic
fallback to the pure-PyTorch path if the toolchain is unavailable.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


RoutingMode = Literal["soft", "hard", "hard_cpp", "triton"]

# ---------------------------------------------------------------------------
# Optional C++ hard-routing extension (JIT via torch.utils.cpp_extension)
# ---------------------------------------------------------------------------

_CSRC_DIR = Path(__file__).resolve().parent.parent / "csrc"
_FFF_HARD_SRC = _CSRC_DIR / "fff_hard.cpp"

_fff_hard_ext: Any | None = None
_fff_hard_load_attempted: bool = False
_fff_hard_load_error: BaseException | None = None


def _load_fff_hard_ext(*, verbose: bool = False) -> Any | None:
    """Load ``fff_hard_cpp`` — prefer an installed wheel, else JIT-compile.

    Compile flags come from ``csrc/compiler_flags.py`` (same as ``setup.py``):
    macOS → ``-O3 -std=c++17`` (optional OpenMP); Linux → ``-O3 -march=native
    -fopenmp -funroll-loops``. Retries without OpenMP if needed.
    """
    global _fff_hard_ext, _fff_hard_load_attempted, _fff_hard_load_error
    if _fff_hard_ext is not None:
        return _fff_hard_ext
    if _fff_hard_load_attempted:
        return None
    _fff_hard_load_attempted = True

    # 1) Prefer a setuptools-built extension (`pip install -e .` / setup.py).
    try:
        import fff_hard_cpp as installed  # type: ignore[import-not-found]

        _fff_hard_ext = installed
        return _fff_hard_ext
    except ImportError:
        pass

    if not _FFF_HARD_SRC.is_file():
        _fff_hard_load_error = FileNotFoundError(_FFF_HARD_SRC)
        warnings.warn(
            f"FFF C++ hard kernel source missing: {_FFF_HARD_SRC}",
            stacklevel=2,
        )
        return None

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:  # pragma: no cover
        _fff_hard_load_error = exc
        warnings.warn(
            f"torch.utils.cpp_extension unavailable ({exc}); "
            "using PyTorch hard routing",
            stacklevel=2,
        )
        return None

    # Shared aggressive flags with setup.py
    try:
        import sys as _sys

        csrc_path = str(_CSRC_DIR)
        if csrc_path not in _sys.path:
            _sys.path.insert(0, csrc_path)
        from compiler_flags import (  # type: ignore[import-not-found]
            fff_hard_jit_cflags,
            fff_hard_jit_cflags_no_openmp,
            fff_hard_jit_ldflags,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not import csrc/compiler_flags ({exc}); using -O3 only",
            stacklevel=2,
        )
        fff_hard_jit_cflags = lambda: ["-O3"]  # type: ignore[assignment,misc]
        fff_hard_jit_cflags_no_openmp = lambda: ["-O3"]  # type: ignore[assignment,misc]
        fff_hard_jit_ldflags = lambda: []  # type: ignore[assignment,misc]

    # Keep build artifacts inside the repo (avoids home-cache permission issues).
    build_dir = _CSRC_DIR / ".build" / "fff_hard_cpp"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Ensure venv ``ninja`` is on PATH (PyTorch looks up the binary by name).
    import os
    import sys

    venv_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in path_parts:
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

    # Avoid hard-abort when both torch and the extension see an OpenMP runtime.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    flag_sets: list[tuple[str, list[str], list[str]]] = [
        ("openmp", fff_hard_jit_cflags(), fff_hard_jit_ldflags()),
        ("no-openmp", fff_hard_jit_cflags_no_openmp(), []),
    ]
    last_exc: BaseException | None = None
    for label, cflags, ldflags in flag_sets:
        try:
            if verbose:
                print(f"[fff_hard] JIT compile attempt ({label}): cflags={cflags} ldflags={ldflags}")
            _fff_hard_ext = load(
                name="fff_hard_cpp",
                sources=[str(_FFF_HARD_SRC)],
                extra_cflags=cflags,
                extra_ldflags=ldflags,
                build_directory=str(build_dir),
                verbose=verbose,
            )
            return _fff_hard_ext
        except Exception as exc:  # noqa: BLE001 — toolchain / compile failures
            last_exc = exc
            if verbose:
                print(f"[fff_hard] JIT attempt ({label}) failed: {exc}")
            continue

    _fff_hard_load_error = last_exc
    warnings.warn(
        f"FFF C++ hard kernel failed to compile ({type(last_exc).__name__}: {last_exc}); "
        "falling back to PyTorch hard routing",
        stacklevel=2,
    )
    return None


def is_fff_cpp_available() -> bool:
    """Return ``True`` if the C++ hard-routing extension loaded successfully."""
    return _load_fff_hard_ext() is not None


def fff_cpp_load_error() -> BaseException | None:
    """Last C++ extension load error (if any), else ``None``."""
    _load_fff_hard_ext()
    return _fff_hard_load_error


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

        # Precomputed path tables for fully vectorized soft routing (no depth loop).
        # path_router_idx[ℓ, k] = heap router index on leaf ℓ's path at level k.
        # path_go_right[ℓ, k]   = True iff leaf ℓ takes the right child at level k.
        # leaf_router_inc[ℓ, r] = 1 if leaf ℓ's path visits router r.
        path_router_idx, path_go_right, leaf_router_inc = self._build_path_tables(
            depth, self.num_leaves, self.num_routers
        )
        self.register_buffer("path_router_idx", path_router_idx, persistent=False)
        self.register_buffer("path_go_right", path_go_right, persistent=False)
        self.register_buffer("leaf_router_inc", leaf_router_inc, persistent=False)

        # Cached soft-routing stats for compute_balance_loss() after forward_soft.
        # _last_node_decisions: (N, R) — c_n(x) for every router (batched).
        # _last_reach_probs:    (N, R) — prob of visiting each router.
        # _last_leaf_probs:     (N, L) — mixture weights over leaves.
        self._last_node_decisions: Tensor | None = None
        self._last_reach_probs: Tensor | None = None
        self._last_leaf_probs: Tensor | None = None

        self.reset_parameters()

    @staticmethod
    def _build_path_tables(
        depth: int, num_leaves: int, num_routers: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build leaf→path index tables (CPU, copied to device with the module).

        Leaf index bit order matches the soft mixture layout
        (MSB = root decision, LSB = deepest decision):
            leaf 0 = all-left, leaf ``2^d-1`` = all-right.
        """
        path_router_idx = torch.empty(num_leaves, depth, dtype=torch.long)
        path_go_right = torch.empty(num_leaves, depth, dtype=torch.bool)
        leaf_router_inc = torch.zeros(num_leaves, num_routers, dtype=torch.float32)

        for leaf in range(num_leaves):
            node = 0
            for level in range(depth):
                path_router_idx[leaf, level] = node
                leaf_router_inc[leaf, node] = 1.0
                go_right = bool((leaf >> (depth - 1 - level)) & 1)
                path_go_right[leaf, level] = go_right
                node = (node << 1) + 1 + int(go_right)

        return path_router_idx, path_go_right, leaf_router_inc

    def reset_parameters(self) -> None:
        """Initialize routers near 0 (≈ uniform soft splits) and leaves like Linear.

        Router weights MUST stay tiny. Large ``‖w‖`` makes ``σ(wᵀx/τ)`` saturate to
        0/1 at step 0 → instant tree collapse before leaves can learn.
        """
        # Explicit tiny Gaussian: std=1e-3 (not Xavier / not 0.01).
        nn.init.normal_(self.router_weights, mean=0.0, std=1e-3)
        nn.init.zeros_(self.router_biases)

        # Vectorized Xavier-uniform over all leaves (same fan-in/out per leaf).
        # a = sqrt(6 / (fan_in + fan_out))
        fan_in, fan_out = self.in_features, self.out_features
        a = math.sqrt(6.0 / float(fan_in + fan_out))
        nn.init.uniform_(self.leaf_weights, -a, a)
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
            ``"hard"`` — discrete tree walk, single leaf (PyTorch).
            ``"hard_cpp"`` — same semantics via CPU C++ kernel (falls back
            to PyTorch if the extension is unavailable or ``x`` is not CPU).
            ``"triton"`` — fused CUDA Triton kernel (falls back to PyTorch hard
            if Triton/CUDA is unavailable).

        Returns
        -------
        Tensor
            Output of shape ``(..., D_out)``.
        """
        if mode == "soft":
            return self.forward_soft(x)
        if mode == "hard":
            return self.forward_hard(x)
        if mode == "hard_cpp":
            return self.forward_hard_cpp(x)
        if mode == "triton":
            return self.forward_hard_triton(x)
        raise ValueError(
            f"mode must be 'soft', 'hard', 'hard_cpp', or 'triton', got {mode!r}"
        )

    # ------------------------------------------------------------------
    # Soft routing (training)
    # ------------------------------------------------------------------

    def forward_soft(self, x: Tensor) -> Tensor:
        """Soft mixture over all leaves via **vectorized log-space** path products.

        Math
        ----
        ``c_n = σ((w_nᵀ x + b_n) / τ)`` — P(go right | node n).
        For every leaf ``ℓ`` in parallel:
            ``log P(ℓ|x) = Σ_{k∈path(ℓ)} logsigmoid(±z_{r_k})``
        then ``y = Σ_ℓ P(ℓ|x) · (x W_ℓ + b_ℓ)``.

        Implementation is fully batched (no Python depth/leaf loops): gather path
        logits with index tables, sum in log-space, ``softmax`` over leaves.

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``
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

        # Batched leaf affine via bmm: (L, N, D_in) @ (L, D_in, D_out) → (L, N, D_out)
        flat_b = flat.unsqueeze(0).expand(self.num_leaves, -1, -1)
        leaf_out = torch.bmm(flat_b, self.leaf_weights).transpose(0, 1)
        # leaf_out: (N, L, D_out)
        leaf_out = leaf_out + self.leaf_biases.unsqueeze(0)

        # Soft-weighted sum over leaves → (N, D_out)
        y = torch.bmm(leaf_probs.unsqueeze(1), leaf_out).squeeze(1)
        return y.view(*leading, self.out_features)

    def _soft_leaf_probs(
        self, flat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Vectorized leaf mixture: parallel log-path products over the batch.

        Parameters
        ----------
        flat:
            ``(N, D_in)``

        Returns
        -------
        leaf_probs:
            ``(N, L)`` — row-stochastic mixture weights.
        node_decisions:
            ``(N, R)`` — clamped right-child sigmoid at every router.
        reach_probs:
            ``(N, R)`` — soft visit mass per router (= ``leaf_probs @ incidence``).
        """
        eps = 1e-7

        # All router logits at once: (N, R) — batch matmul via F.linear
        router_logits = (
            F.linear(flat, self.router_weights, self.router_biases)
            / self.temperature
        )
        node_decisions = torch.sigmoid(router_logits).clamp(min=eps, max=1.0 - eps)

        # Log-space edge weights for every router (N, R)
        log_c = F.logsigmoid(router_logits)
        log_not_c = F.logsigmoid(-router_logits)

        # Gather path routers for all leaves in one shot: (N, L, depth)
        # path_router_idx: (L, depth)
        idx = self.path_router_idx
        log_c_path = log_c[:, idx]
        log_not_c_path = log_not_c[:, idx]

        # Select left/right log-prob along each leaf's path — still (N, L, depth)
        go_right = self.path_go_right.unsqueeze(0)  # (1, L, depth)
        log_edges = torch.where(go_right, log_c_path, log_not_c_path)

        # Parallel path product in log-space → (N, L)
        log_leaf = log_edges.sum(dim=-1)
        leaf_probs = torch.softmax(log_leaf, dim=-1)

        # Reach mass: which routers soft-mass visits (N, R)
        # leaf_router_inc: (L, R)
        reach_probs = leaf_probs @ self.leaf_router_inc

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

    def forward_hard_cpp(self, x: Tensor) -> Tensor:
        """Hard routing via the JIT C++ CPU kernel (exact PyTorch semantics).

        Falls back to :meth:`forward_hard` when:
        * the C++ extension failed to compile / load, or
        * ``x`` is not on CPU (GPU callers keep the PyTorch path).

        Inputs are forced contiguous float32 for the native kernel.
        """
        flat, leading = self._flatten_input(x)
        if flat.device.type != "cpu":
            return self.forward_hard(x)

        ext = _load_fff_hard_ext()
        if ext is None:
            return self.forward_hard(x)

        # Match C++ contract: float32 contiguous (N, D_in).
        flat_f = flat.float().contiguous()
        y = ext.fff_hard_forward_cpu(
            flat_f,
            self.router_weights.detach().float().contiguous(),
            self.router_biases.detach().float().contiguous(),
            self.leaf_weights.detach().float().contiguous(),
            self.leaf_biases.detach().float().contiguous(),
            int(self.depth),
        )
        return y.to(dtype=flat.dtype).view(*leading, self.out_features)

    def forward_hard_triton(self, x: Tensor) -> Tensor:
        """Hard routing via hybrid Triton CUDA kernels (fused or leaf-sorted).

        Falls back to :meth:`forward_hard` when Triton is missing or ``x`` is
        not on CUDA. Hybrid dispatch uses the leading batch size when ``x`` is
        ``(B, T, D)`` so small batches stay on the fused kernel even if
        ``N = B·T > 4``.
        """
        flat, leading = self._flatten_input(x)
        if flat.device.type != "cuda":
            return self.forward_hard(x)

        try:
            from models.fff_hard_triton import (
                _as_compute_tensor,
                fff_hard_forward_triton,
                is_triton_available,
            )
        except ImportError:
            return self.forward_hard(x)

        if not is_triton_available():
            return self.forward_hard(x)

        # Match activation dtype without re-casting when already half/BF16.
        # On Triton failure, fall back to PyTorch hard in the same dtype.
        dt = flat.dtype
        dispatch_n = int(x.shape[0]) if x.ndim >= 2 else int(flat.shape[0])
        try:
            y = fff_hard_forward_triton(
                flat if flat.is_contiguous() else flat.contiguous(),
                _as_compute_tensor(self.router_weights.detach(), dt),
                _as_compute_tensor(self.router_biases.detach(), dt),
                _as_compute_tensor(self.leaf_weights.detach(), dt),
                _as_compute_tensor(self.leaf_biases.detach(), dt),
                int(self.depth),
                dispatch_n=dispatch_n,
            )
        except (RuntimeError, TypeError, ValueError):
            return self.forward_hard(x)
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
    y_cpp = layer(x, mode="hard_cpp")
    loss = layer.compute_balance_loss()
    assert y_soft.shape == (8, 16, 64)
    assert y_hard.shape == (8, 16, 64)
    assert y_cpp.shape == (8, 16, 64)
    assert loss.ndim == 0
    # Hard sequential matches batched hard for one vector
    x0 = x[0, 0]
    y_seq = layer.forward_hard_sequential(x0)
    y_b = layer.forward_hard(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_seq, y_b), (y_seq - y_b).abs().max()
    # C++ hard must match PyTorch hard (or fall back identically).
    assert torch.allclose(y_hard, y_cpp, atol=1e-5, rtol=1e-5), (
        (y_hard - y_cpp).abs().max().item()
    )
    # Soft → hard agreement at low temperature (near one-hot mixture)
    layer.set_temperature(1e-4)
    y_s = layer.forward_soft(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_s, y_b, atol=1e-3), (y_s - y_b).abs().max()
    print(
        f"OK — soft/hard/cpp shapes agree; cpp_available={is_fff_cpp_available()}"
    )


if __name__ == "__main__":
    _demo_shapes()
