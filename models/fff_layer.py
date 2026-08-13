"""Fast Feedforward / Multi-Tree CMM linear layer — Belcak & Wattenhofer.

Replaces a dense linear map with ``K`` independent binary trees of depth
``d`` (UltraFastBERT / Conditional Matrix Multiplication):

    y = Σ_{k=1}^{K} (x W_{k, ℓ_k} + b_{k, ℓ_k})

``K = 1`` is the original single-tree FFF (checkpoint-compatible).

Tree indexing (heap / breadth-first, 0-based, **per tree**):
    - Root: 0
    - Left child of ``i``: ``2*i + 1``
    - Right child of ``i``: ``2*i + 2``
    - Leaf heap indices: ``[2^d - 1, 2^{d+1} - 2]``
    - Contiguous leaf id: ``heap_index - (2^d - 1)``

Parameter layout (tree-major, matches ``csrc/fff_hard.cpp``)::

    router_weights : (K, R, D_in)      R = 2^d - 1
    router_biases  : (K, R)
    leaf_weights   : (K, L, D_in, D_out)   L = 2^d
    leaf_biases    : (K, L, D_out)

Hard routing on CPU uses ``cmm_hard_forward_cpu`` (falls back to the
legacy ``fff_hard_forward_cpu`` when ``K = 1`` and the symbol is missing).
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


RoutingMode = Literal["soft", "hard", "hard_cpp", "triton", "triton_int8", "triton_int4"]

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

    # Prefer a setuptools-built extension, but skip stale single-tree builds
    # that predate ``cmm_hard_forward_cpu``.
    installed: Any | None = None
    try:
        import fff_hard_cpp as installed_mod  # type: ignore[import-not-found]

        installed = installed_mod
        if callable(getattr(installed, "cmm_hard_forward_cpu", None)):
            _fff_hard_ext = installed
            return _fff_hard_ext
        warnings.warn(
            "Installed fff_hard_cpp lacks cmm_hard_forward_cpu; "
            "JIT-compiling Multi-Tree CMM from csrc/fff_hard.cpp. "
            "Rebuild permanently with: pip install -e . --no-build-isolation",
            stacklevel=2,
        )
    except ImportError:
        pass

    if not _FFF_HARD_SRC.is_file():
        _fff_hard_load_error = FileNotFoundError(_FFF_HARD_SRC)
        warnings.warn(
            f"FFF C++ hard kernel source missing: {_FFF_HARD_SRC}",
            stacklevel=2,
        )
        if installed is not None:
            _fff_hard_ext = installed
            return _fff_hard_ext
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
        if installed is not None:
            _fff_hard_ext = installed
            return _fff_hard_ext
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
    jit_name = "fff_hard_cmm" if installed is not None else "fff_hard_cpp"
    build_dir = _CSRC_DIR / ".build" / jit_name
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
                name=jit_name,
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
    if installed is not None:
        _fff_hard_ext = installed
        return _fff_hard_ext
    return None


def is_fff_cpp_available() -> bool:
    """Return ``True`` if the C++ hard-routing extension loaded successfully."""
    return _load_fff_hard_ext() is not None


def is_cmm_cpp_available() -> bool:
    """Return ``True`` if the native module exports ``cmm_hard_forward_cpu``."""
    ext = _load_fff_hard_ext()
    return callable(getattr(ext, "cmm_hard_forward_cpu", None))


def fff_cpp_load_error() -> BaseException | None:
    """Last C++ extension load error (if any), else ``None``."""
    _load_fff_hard_ext()
    return _fff_hard_load_error


def _infer_cmm_hparams_from_state(
    ckpt: dict[str, Any],
) -> tuple[int, int] | None:
    """Infer ``(K, d)`` from the first ``router_weights`` tensor in a checkpoint.

    Tree-major CMM stores ``(K, R, D_in)`` with ``R = 2^d - 1``. Legacy
    single-tree FFF stores ``(R, D_in)`` (``K = 1``).
    """
    sd = ckpt.get("student_state_dict")
    if not isinstance(sd, dict):
        sd = ckpt.get("state_dict")
    if not isinstance(sd, dict):
        return None
    for key, tensor in sd.items():
        if not str(key).endswith("router_weights"):
            continue
        if not hasattr(tensor, "ndim") or not hasattr(tensor, "shape"):
            continue
        if tensor.ndim == 3:
            num_trees, num_routers, _ = (int(s) for s in tensor.shape)
        elif tensor.ndim == 2:
            num_trees, num_routers = 1, int(tensor.shape[0])
        else:
            continue
        depth = int(round(math.log2(num_routers + 1)))
        if (1 << depth) - 1 == num_routers and depth >= 1:
            return max(num_trees, 1), depth
    return None


def cmm_hparams_from_checkpoint(ckpt: dict[str, Any]) -> tuple[int, int]:
    """Read ``(num_trees, depth_per_tree)`` from a GPT-2 FFF / CMM checkpoint.

    Accepts both new keys (``num_trees``, ``depth_per_tree``) and legacy
    ``fff_depth`` (treated as ``depth_per_tree`` with ``num_trees=1``).
    Nested ``config`` dicts are consulted as a fallback. When metadata is
    missing, shapes of ``router_weights`` in ``student_state_dict`` are used.
    """
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    cfg = cfg or {}
    num_trees = ckpt.get("num_trees", cfg.get("num_trees"))
    depth = ckpt.get("depth_per_tree", cfg.get("depth_per_tree"))
    if depth is None:
        depth = ckpt.get("fff_depth", cfg.get("fff_depth"))

    inferred = _infer_cmm_hparams_from_state(ckpt)
    if inferred is not None:
        inf_k, inf_d = inferred
        if num_trees is None:
            num_trees = inf_k
        if depth is None:
            depth = inf_d

    if num_trees is None:
        num_trees = 1
    if depth is None:
        depth = 4
    return max(int(num_trees), 1), max(int(depth), 1)


class FastFeedforwardLinear(nn.Module):
    """Multi-tree CMM layer with soft (train) and hard (infer) routing.

    ``K`` independent trees of depth ``d`` replace ``y = x W + b``. Each tree
    has ``2^d - 1`` routers and ``2^d`` leaf affines. The layer output is the
    **sum** of the K selected (hard) or mixed (soft) leaf maps, so
    ``D_out`` stays aligned with GPT-2 / Transformer residuals.

    Soft mode (training)
    --------------------
    Tree ``k`` uses a differentiable path mixture
        ``c_{k,n}(x) = σ( (w_{k,n}ᵀ x + b_{k,n}) / τ )``
        ``P_k(ℓ | x) = Π_{n ∈ path(ℓ)} c or (1-c)``
        ``y = Σ_k Σ_ℓ P_k(ℓ|x) · (x W_{k,ℓ} + b_{k,ℓ})``
    STE (:meth:`forward_soft_ste`) uses a hard one-hot per tree in forward
    and ``∂softmax`` in backward (Hard / C++-aligned).

    Hard mode (inference)
    ---------------------
    Each tree walks with ``(wᵀ x + b) > 0`` (right if true). Only the
    winning leaf of each tree is evaluated — ``K · (d`` router dots ``+ 1``
    leaf GEMV). CPU delegates to ``cmm_hard_forward_cpu``.

    Parameters
    ----------
    in_features:
        Input dim ``D_in``.
    out_features:
        Output dim ``D_out`` (after summing the K trees).
    depth:
        Per-tree depth ``d``. Alias of ``depth_per_tree`` (legacy name).
    init_temp:
        Initial temperature ``τ`` for soft sigmoid decisions.
    num_trees:
        Number of parallel trees ``K`` (default ``1`` = classic FFF).
    depth_per_tree:
        If set, overrides ``depth``. Must match ``depth`` when both given
        and ``depth`` is not the default used only as a positional alias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 4,
        init_temp: float = 1.0,
        *,
        num_trees: int = 1,
        depth_per_tree: int | None = None,
    ) -> None:
        super().__init__()
        if depth_per_tree is not None:
            if int(depth_per_tree) < 1:
                raise ValueError(f"depth_per_tree must be >= 1, got {depth_per_tree}")
            depth = int(depth_per_tree)
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if num_trees < 1:
            raise ValueError(f"num_trees must be >= 1, got {num_trees}")
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be >= 1")
        if init_temp <= 0.0:
            raise ValueError(f"init_temp must be > 0, got {init_temp}")

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_trees: int = int(num_trees)
        self.depth: int = int(depth)
        self.depth_per_tree: int = self.depth
        self.num_leaves: int = 1 << self.depth  # 2^d per tree
        self.num_routers: int = self.num_leaves - 1  # 2^d - 1 per tree

        # Tree-major layout — matches csrc/fff_hard.cpp CmmLayout.
        self.router_weights = nn.Parameter(
            torch.empty(self.num_trees, self.num_routers, in_features)
        )
        self.router_biases = nn.Parameter(
            torch.empty(self.num_trees, self.num_routers)
        )
        self.leaf_weights = nn.Parameter(
            torch.empty(
                self.num_trees, self.num_leaves, in_features, out_features
            )
        )
        self.leaf_biases = nn.Parameter(
            torch.empty(self.num_trees, self.num_leaves, out_features)
        )

        self.register_buffer(
            "temperature",
            torch.tensor(float(init_temp)),
            persistent=True,
        )

        # Path tables are identical across trees (same complete binary shape).
        path_router_idx, path_go_right, leaf_router_inc = self._build_path_tables(
            self.depth, self.num_leaves, self.num_routers
        )
        self.register_buffer("path_router_idx", path_router_idx, persistent=False)
        self.register_buffer("path_go_right", path_go_right, persistent=False)
        self.register_buffer("leaf_router_inc", leaf_router_inc, persistent=False)

        # Soft-routing caches. K=1 squeezes the tree axis so legacy callers
        # still see ``(N, L)`` / ``(N, R)``.
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
            f"num_trees={self.num_trees}, depth_per_tree={self.depth}, "
            f"num_routers={self.num_routers}, num_leaves={self.num_leaves}, "
            f"temperature={float(self.temperature):.4g}"
        )

    def _cache_soft_stats(
        self,
        leaf_probs: Tensor,
        node_decisions: Tensor,
        reach_probs: Tensor,
    ) -> None:
        """Store mixture stats; squeeze the tree axis when ``K = 1``."""
        if self.num_trees == 1:
            self._last_leaf_probs = leaf_probs.squeeze(1)
            self._last_node_decisions = node_decisions.squeeze(1)
            self._last_reach_probs = reach_probs.squeeze(1)
        else:
            self._last_leaf_probs = leaf_probs
            self._last_node_decisions = node_decisions
            self._last_reach_probs = reach_probs

    def _flat_balance_stats(self) -> tuple[Tensor, Tensor]:
        """Return ``(node_decisions, leaf_probs)`` as ``(N*, R)`` / ``(N*, L)``."""
        if self._last_node_decisions is None or self._last_leaf_probs is None:
            raise RuntimeError(
                "compute_balance_loss() requires a prior forward_soft() call "
                "to cache routing statistics"
            )
        decisions = self._last_node_decisions
        leaf_probs = self._last_leaf_probs
        if decisions.ndim == 3:
            decisions = decisions.reshape(-1, self.num_routers)
        if leaf_probs.ndim == 3:
            leaf_probs = leaf_probs.reshape(-1, self.num_leaves)
        return decisions, leaf_probs

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
            ``"hard_cpp"`` — discrete CMM via CPU C++ (``cmm_hard_forward_cpu``).
            ``"triton"`` — fused CUDA Triton kernel (falls back to PyTorch hard
            if Triton/CUDA is unavailable).
            ``"triton_int8"`` / ``"triton_int4"`` — Triton hard routing with
            quantized leaf weights (requires ``leaf_qstate`` attached).

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
        if mode == "triton_int8":
            return self.forward_hard_triton_quant(x, expect_mode="int8")
        if mode == "triton_int4":
            return self.forward_hard_triton_quant(x, expect_mode="int4")
        raise ValueError(
            "mode must be 'soft', 'hard', 'hard_cpp', 'triton', "
            f"'triton_int8', or 'triton_int4', got {mode!r}"
        )

    # ------------------------------------------------------------------
    # Soft routing (training)
    # ------------------------------------------------------------------

    def forward_soft(self, x: Tensor) -> Tensor:
        """Soft multi-tree CMM — independent mixture per tree, then sum.

        Math
        ----
        For each tree ``k`` independently:
            ``c_{k,n} = σ((w_{k,n}ᵀ x + b_{k,n}) / τ)``
            ``log P_k(ℓ|x) = Σ_{n∈path(ℓ)} logsigmoid(±z_{k,n})``
            ``y_k = Σ_ℓ P_k(ℓ|x) · (x W_{k,ℓ} + b_{k,ℓ})``
        ``y = Σ_k y_k``.

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``
        """
        flat, leading = self._flatten_input(x)
        leaf_probs, node_decisions, reach_probs = self._soft_leaf_probs(flat)
        self._cache_soft_stats(leaf_probs, node_decisions, reach_probs)

        # leaf_out: (N, K, L, D_out) — contract D_in (`i`) across every leaf.
        leaf_out = torch.einsum("ni,klio->nklo", flat, self.leaf_weights)
        leaf_out = leaf_out + self.leaf_biases.unsqueeze(0)
        y = torch.einsum("nkl,nklo->no", leaf_probs, leaf_out)
        return y.view(*leading, self.out_features)

    def forward_soft_ste(self, x: Tensor) -> Tensor:
        """STE hard-aware CMM: one winning leaf **per tree** in forward.

        Backward treats the per-tree mask as soft mixture probabilities so
        router gradients still flow. Matches hard / C++ inference as ``τ → 0``.

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``
        """
        flat, leading = self._flatten_input(x)
        log_leaf, node_decisions = self._soft_leaf_logits(flat)
        # log_leaf: (N, K, L)
        prob_soft = F.softmax(log_leaf, dim=-1)
        reach_probs = torch.matmul(prob_soft, self.leaf_router_inc)
        self._cache_soft_stats(prob_soft, node_decisions, reach_probs)

        mask_hard = F.one_hot(
            torch.argmax(log_leaf, dim=-1),
            num_classes=self.num_leaves,
        ).to(dtype=prob_soft.dtype)
        mask = (mask_hard - prob_soft).detach() + prob_soft

        # masked_in: (N, K, L, D_in) — only the winning leaf of each tree
        # is nonzero in forward; grads flow through ``mask``.
        masked_in = flat.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(-1)
        leaf_out = torch.einsum("nkli,klio->nklo", masked_in, self.leaf_weights)
        leaf_out = leaf_out + self.leaf_biases.unsqueeze(0) * mask.unsqueeze(-1)
        y = leaf_out.sum(dim=(1, 2))
        return y.view(*leading, self.out_features)

    def _soft_leaf_logits(self, flat: Tensor) -> tuple[Tensor, Tensor]:
        """Log-path scores and router decisions for every tree.

        Parameters
        ----------
        flat:
            ``(N, D_in)``

        Returns
        -------
        log_leaf:
            ``(N, K, L)`` — unnormalized log-path scores (routers already ``/ τ``).
        node_decisions:
            ``(N, K, R)`` — clamped right-child sigmoid at every router.
        """
        eps = 1e-7
        router_logits = (
            torch.einsum("nd,krd->nkr", flat, self.router_weights)
            + self.router_biases.unsqueeze(0)
        ) / self.temperature
        node_decisions = torch.sigmoid(router_logits).clamp(min=eps, max=1.0 - eps)

        log_c = F.logsigmoid(router_logits)
        log_not_c = F.logsigmoid(-router_logits)
        idx = self.path_router_idx  # (L, depth)
        log_c_path = log_c[:, :, idx]  # (N, K, L, depth)
        log_not_c_path = log_not_c[:, :, idx]
        go_right = self.path_go_right.view(1, 1, self.num_leaves, self.depth)
        log_edges = torch.where(go_right, log_c_path, log_not_c_path)
        log_leaf = log_edges.sum(dim=-1)
        return log_leaf, node_decisions

    def _soft_leaf_probs(
        self, flat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Vectorized per-tree leaf mixtures.

        Returns
        -------
        leaf_probs:
            ``(N, K, L)`` — row-stochastic over leaves, independently per tree.
        node_decisions:
            ``(N, K, R)``.
        reach_probs:
            ``(N, K, R)`` — ``leaf_probs @ incidence``.
        """
        log_leaf, node_decisions = self._soft_leaf_logits(flat)
        leaf_probs = torch.softmax(log_leaf, dim=-1)
        reach_probs = torch.matmul(leaf_probs, self.leaf_router_inc)
        return leaf_probs, node_decisions, reach_probs

    # ------------------------------------------------------------------
    # Hard routing (inference)
    # ------------------------------------------------------------------

    def forward_hard(self, x: Tensor) -> Tensor:
        """Hard multi-tree CMM — one leaf per tree, then sum.

        For each tree ``k`` walk ``depth`` routers with
            ``go_right = (w_{k,n}ᵀ x + b_{k,n}) > 0``
        then ``y = Σ_k (x W_{k,ℓ_k} + b_{k,ℓ_k})``.

        Cost per token: ``K · (depth`` router dots ``+ 1`` leaf GEMV).

        Shapes
        ------
        x: ``(..., D_in)``
        returns: ``(..., D_out)``
        """
        flat, leading = self._flatten_input(x)
        n_batch = flat.shape[0]
        k_idx = (
            torch.arange(self.num_trees, device=flat.device)
            .view(1, -1)
            .expand(n_batch, self.num_trees)
            .contiguous()
        )
        node_ids = torch.zeros(
            n_batch, self.num_trees, dtype=torch.long, device=flat.device
        )

        for _ in range(self.depth):
            w = self.router_weights[k_idx, node_ids]  # (N, K, D_in)
            b = self.router_biases[k_idx, node_ids]  # (N, K)
            logits = torch.einsum("ni,nki->nk", flat, w) + b
            go_right = logits > 0
            node_ids = (node_ids << 1) + 1 + go_right.to(torch.long)

        leaf_ids = node_ids - (self.num_leaves - 1)
        w_leaf = self.leaf_weights[k_idx, leaf_ids]  # (N, K, D_in, D_out)
        b_leaf = self.leaf_biases[k_idx, leaf_ids]  # (N, K, D_out)
        # Contract D_in (`i`): y[n,k] = x[n] @ W[n,k]  (not a reduction over i).
        y_k = torch.einsum("ni,nkio->nko", flat, w_leaf) + b_leaf
        y = y_k.sum(dim=1)
        return y.view(*leading, self.out_features)

    def forward_hard_cpp(self, x: Tensor) -> Tensor:
        """Hard CMM via the JIT C++ CPU kernel (exact PyTorch hard semantics).

        Prefers ``cmm_hard_forward_cpu`` (tree-major ``(K, …)`` tensors,
        ``reduce=sum``). Falls back to legacy ``fff_hard_forward_cpu`` when
        ``K = 1`` and the new symbol is missing, then to :meth:`forward_hard`.
        """
        flat, leading = self._flatten_input(x)
        if flat.device.type != "cpu":
            return self.forward_hard(x)

        ext = _load_fff_hard_ext()
        if ext is None:
            return self.forward_hard(x)

        flat_f = flat.float().contiguous()
        wr = self.router_weights.detach().float().contiguous()
        br = self.router_biases.detach().float().contiguous()
        wl = self.leaf_weights.detach().float().contiguous()
        bl = self.leaf_biases.detach().float().contiguous()
        depth = int(self.depth)

        cmm_fn = getattr(ext, "cmm_hard_forward_cpu", None)
        try:
            if cmm_fn is not None:
                y = cmm_fn(flat_f, wr, br, wl, bl, depth, 0)
            elif self.num_trees == 1 and hasattr(ext, "fff_hard_forward_cpu"):
                y = ext.fff_hard_forward_cpu(
                    flat_f, wr[0], br[0], wl[0], bl[0], depth
                )
            else:
                return self.forward_hard(x)
        except Exception:
            return self.forward_hard(x)
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
            y = self._triton_hard_sum(
                fff_hard_forward_triton,
                flat if flat.is_contiguous() else flat.contiguous(),
                dt,
                dispatch_n=dispatch_n,
            )
        except (RuntimeError, TypeError, ValueError):
            return self.forward_hard(x)
        return y.view(*leading, self.out_features)

    def _triton_hard_sum(
        self,
        kernel_fn: Any,
        flat: Tensor,
        dt: torch.dtype,
        *,
        dispatch_n: int,
        qstate: Any | None = None,
    ) -> Tensor:
        """Run the single-tree Triton kernel once per CMM tree and sum.

        Existing Triton kernels expect 2-D routers / 3-D leaves. Each tree
        is a contiguous slab in the tree-major layout, so we slice ``k``
        without a copy when the storage is already contiguous.
        """
        from models.fff_hard_triton import _as_compute_tensor

        y_acc: Tensor | None = None
        for k in range(self.num_trees):
            wr = _as_compute_tensor(self.router_weights[k].detach(), dt)
            br = _as_compute_tensor(self.router_biases[k].detach(), dt)
            bl = _as_compute_tensor(self.leaf_biases[k].detach(), dt)
            if qstate is None:
                wl = _as_compute_tensor(self.leaf_weights[k].detach(), dt)
                y_k = kernel_fn(
                    flat,
                    wr,
                    br,
                    wl,
                    bl,
                    int(self.depth),
                    dispatch_n=dispatch_n,
                )
            else:
                # Quant packs are single-tree today; K>1 falls back upstream.
                y_k = kernel_fn(
                    flat,
                    wr,
                    br,
                    qstate,
                    bl,
                    int(self.depth),
                    dispatch_n=dispatch_n,
                )
            y_acc = y_k if y_acc is None else y_acc + y_k
        assert y_acc is not None
        return y_acc

    def forward_hard_triton_quant(
        self,
        x: Tensor,
        *,
        expect_mode: str | None = None,
    ) -> Tensor:
        """Hard routing via Triton with INT8/INT4 leaf dequant (``leaf_qstate``).

        Attach a :class:`~models.fff_quant.QuantizedLeafWeights` as
        ``self.leaf_qstate`` before calling. Falls back to FP16 dequant +
        :meth:`forward_hard` when Triton/CUDA is unavailable.
        """
        qstate = getattr(self, "leaf_qstate", None)
        if qstate is None:
            raise RuntimeError(
                "forward_hard_triton_quant requires self.leaf_qstate "
                "(run quantize_fff_model_leaves / attach_leaf_qstates)"
            )
        if expect_mode is not None and qstate.mode != expect_mode:
            raise ValueError(
                f"leaf_qstate.mode={qstate.mode!r} but mode expects {expect_mode!r}"
            )

        flat, leading = self._flatten_input(x)
        if flat.device.type != "cuda":
            from models.fff_quant import dequantize_leaf_weights

            w_fp = dequantize_leaf_weights(qstate).to(device=flat.device, dtype=flat.dtype)
            # Temporary weight swap for PyTorch hard fallback.
            saved = self.leaf_weights.data
            try:
                self.leaf_weights.data = w_fp
                return self.forward_hard(x)
            finally:
                self.leaf_weights.data = saved

        try:
            from models.fff_hard_triton import (
                _as_compute_tensor,
                fff_hard_forward_triton_quant,
                is_triton_available,
            )
        except ImportError:
            from models.fff_quant import dequantize_leaf_weights

            w_fp = dequantize_leaf_weights(qstate).to(dtype=flat.dtype)
            saved = self.leaf_weights.data
            try:
                self.leaf_weights.data = w_fp
                return self.forward_hard(x)
            finally:
                self.leaf_weights.data = saved

        if not is_triton_available():
            from models.fff_quant import dequantize_leaf_weights

            w_fp = dequantize_leaf_weights(qstate).to(dtype=flat.dtype)
            saved = self.leaf_weights.data
            try:
                self.leaf_weights.data = w_fp
                return self.forward_hard(x)
            finally:
                self.leaf_weights.data = saved

        dt = flat.dtype
        dispatch_n = int(x.shape[0]) if x.ndim >= 2 else int(flat.shape[0])
        if self.num_trees != 1:
            # Quant packs are single-tree; CMM K>1 uses dequant + PyTorch hard.
            from models.fff_quant import dequantize_leaf_weights

            w_fp = dequantize_leaf_weights(qstate).to(
                device=flat.device, dtype=flat.dtype
            )
            saved = self.leaf_weights.data
            try:
                self.leaf_weights.data = w_fp
                return self.forward_hard(x)
            finally:
                self.leaf_weights.data = saved
        try:
            y = fff_hard_forward_triton_quant(
                flat if flat.is_contiguous() else flat.contiguous(),
                _as_compute_tensor(self.router_weights[0].detach(), dt),
                _as_compute_tensor(self.router_biases[0].detach(), dt),
                qstate,
                _as_compute_tensor(self.leaf_biases[0].detach(), dt),
                int(self.depth),
                dispatch_n=dispatch_n,
            )
        except (RuntimeError, TypeError, ValueError):
            from models.fff_quant import dequantize_leaf_weights

            w_fp = dequantize_leaf_weights(qstate).to(dtype=flat.dtype)
            saved = self.leaf_weights.data
            try:
                self.leaf_weights.data = w_fp
                return self.forward_hard(x)
            finally:
                self.leaf_weights.data = saved
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

        y = x.new_zeros(self.out_features)
        for k in range(self.num_trees):
            node = 0
            wr = self.router_weights[k]
            br = self.router_biases[k]
            for _ in range(self.depth):
                logit = torch.dot(wr[node], x) + br[node]
                if bool(logit > 0):
                    node = 2 * node + 2
                else:
                    node = 2 * node + 1
            leaf = node - (self.num_leaves - 1)
            y = y + x @ self.leaf_weights[k, leaf] + self.leaf_biases[k, leaf]
        return y

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

        per_tree: list[dict[str, list[int] | int]] = []
        all_routers = set(range(self.num_routers))
        all_leaves = set(range(self.num_leaves))
        for k in range(self.num_trees):
            router_ids: list[int] = []
            node = 0
            wr = self.router_weights[k]
            br = self.router_biases[k]
            for _ in range(self.depth):
                router_ids.append(node)
                logit = torch.dot(wr[node], x) + br[node]
                if bool(logit > 0):
                    node = 2 * node + 2
                else:
                    node = 2 * node + 1
            leaf_id = node - (self.num_leaves - 1)
            per_tree.append(
                {
                    "router_ids": router_ids,
                    "leaf_id": leaf_id,
                    "skipped_router_ids": sorted(all_routers - set(router_ids)),
                    "skipped_leaf_ids": sorted(all_leaves - {leaf_id}),
                }
            )
        # K=1 keeps the historical flat keys so infer.py verification still works.
        out: dict[str, list[int] | int] = dict(per_tree[0])
        if self.num_trees > 1:
            out["num_trees"] = self.num_trees  # type: ignore[assignment]
        return out

    def active_params_per_token(self) -> dict[str, int]:
        """Stored vs hard-active parameter counts for this layer."""
        return self.count_active_params(
            self.depth,
            self.in_features,
            self.out_features,
            num_trees=self.num_trees,
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
        decisions, leaf_probs = self._flat_balance_stats()

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

        decisions, leaf_probs = self._flat_balance_stats()
        mean_leaf = leaf_probs.mean(dim=0)  # (L,)
        active = mean_leaf > active_leaf_threshold
        leaf_utilization_pct = float(active.float().mean().item() * 100.0)

        p_avg = decisions.mean(dim=0)  # (R,)
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
            "num_trees": float(self.num_trees),
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

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Upgrade legacy single-tree 2-D/3-D weights to tree-major ``(K, …)``.

        Old checkpoints stored ``router_weights`` as ``(R, D_in)`` and
        ``leaf_weights`` as ``(L, D_in, D_out)``. When this module has
        ``K = 1``, unsqueeze a leading tree axis so ``load_state_dict`` matches.
        """
        rw = prefix + "router_weights"
        if rw in state_dict and state_dict[rw].ndim == 2 and self.router_weights.ndim == 3:
            if self.num_trees != 1:
                error_msgs.append(
                    f"{rw}: legacy 2-D FFF weights cannot load into num_trees="
                    f"{self.num_trees} (expected K=1 unsqueeze)"
                )
            else:
                state_dict[rw] = state_dict[rw].unsqueeze(0)
                rb = prefix + "router_biases"
                lw = prefix + "leaf_weights"
                lb = prefix + "leaf_biases"
                if rb in state_dict and state_dict[rb].ndim == 1:
                    state_dict[rb] = state_dict[rb].unsqueeze(0)
                if lw in state_dict and state_dict[lw].ndim == 3:
                    state_dict[lw] = state_dict[lw].unsqueeze(0)
                if lb in state_dict and state_dict[lb].ndim == 2:
                    state_dict[lb] = state_dict[lb].unsqueeze(0)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        """Mean soft leaf occupancy from the last soft forward — shape ``(L,)``.

        Averages over the batch and, when ``K > 1``, over trees.
        """
        if self._last_leaf_probs is None:
            raise RuntimeError("leaf_usage() requires a prior forward_soft() call")
        _, leaf_probs = self._flat_balance_stats()
        return leaf_probs.mean(dim=0)

    @staticmethod
    def count_active_params(
        depth: int,
        in_features: int,
        out_features: int,
        num_trees: int = 1,
    ) -> dict[str, int]:
        """Parameter counts for soft (all) vs hard (one path per tree).

        Soft / stored: all routers + all leaves across ``K`` trees.
        Hard / active per token: ``K · (depth`` routers ``+ 1`` leaf).
        """
        num_trees = max(int(num_trees), 1)
        num_leaves = 1 << depth
        num_routers = num_leaves - 1
        router_params = num_trees * num_routers * (in_features + 1)
        leaf_params = num_trees * num_leaves * (
            in_features * out_features + out_features
        )
        hard_routers = num_trees * depth * (in_features + 1)
        hard_leaf = num_trees * (in_features * out_features + out_features)
        return {
            "stored_total": router_params + leaf_params,
            "hard_active_per_token": hard_routers + hard_leaf,
            "dense_linear": in_features * out_features + out_features,
            "speedup_vs_dense_flops_proxy": (in_features * out_features + out_features)
            / max(hard_routers + hard_leaf, 1),
            "num_routers": num_routers,
            "num_leaves": num_leaves,
            "num_trees": num_trees,
            "num_leaves_total": num_trees * num_leaves,
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
    # Soft → hard agreement at low temperature (peak routers so MAP = greedy).
    with torch.no_grad():
        layer.router_weights.mul_(50.0)
    layer.set_temperature(1e-4)
    y_s = layer.forward_soft(x0.unsqueeze(0)).squeeze(0)
    y_b = layer.forward_hard(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_s, y_b, atol=1e-3), (y_s - y_b).abs().max()
    print(
        f"OK — K=1 soft/hard/cpp shapes agree; "
        f"cpp_available={is_fff_cpp_available()} "
        f"cmm_cpp={is_cmm_cpp_available()}"
    )

    forest = FastFeedforwardLinear(32, 64, depth=3, num_trees=4, init_temp=1.0)
    y_f_soft = forest(x, mode="soft")
    y_f_hard = forest(x, mode="hard")
    y_f_cpp = forest(x, mode="hard_cpp")
    assert y_f_soft.shape == (8, 16, 64)
    assert y_f_hard.shape == y_f_cpp.shape == (8, 16, 64)
    y_f_seq = forest.forward_hard_sequential(x0)
    y_f_b = forest.forward_hard(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_f_seq, y_f_b, atol=1e-5), (y_f_seq - y_f_b).abs().max()
    assert torch.allclose(y_f_hard, y_f_cpp, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        forest.router_weights.mul_(50.0)
    forest.set_temperature(1e-4)
    y_f_s = forest.forward_soft(x0.unsqueeze(0)).squeeze(0)
    y_f_b = forest.forward_hard(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_f_s, y_f_b, atol=1e-3), (y_f_s - y_f_b).abs().max()
    y_f_ste = forest.forward_soft_ste(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(y_f_ste, y_f_b, atol=1e-3), (y_f_ste - y_f_b).abs().max()
    print("OK — CMM K=4 d=3 soft/STE/hard/cpp agree")


if __name__ == "__main__":
    _demo_shapes()
