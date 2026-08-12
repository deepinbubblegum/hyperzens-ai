"""Triton CUDA FFF hard routing with hybrid fused / leaf-sorted paths.

Hybrid dispatch
---------------
Routing uses a **dispatch count** ``D`` (default: flattened ``N = x.shape[0]``;
transformer callers pass ``dispatch_n=B`` so full-context decode still keys
off microbatch size):

* **FP16 / BF16:** fused while ``D <= 16``; sorted for ``D > 16``.
* **FP32:** fused while ``D <= 4``; sorted for ``D > 4``.

Fused path: single-pass ``fff_hard_forward_triton_kernel`` (no sort overhead).
Sorted path: route → ``torch.sort`` by leaf → coalesced ``torch.mm`` per leaf.

Buffers for ``y``, ``leaf_ids``, sort indices, and sorted ``x``/``y`` are
pooled and reused to avoid allocator churn / fragmentation on RTX 3060.

Precision (Ampere / sm_86 Tensor Cores)
---------------------------------------
* Activations and weights may be ``float16``, ``bfloat16``, or ``float32``.
* Triton kernels accumulate ``score`` / ``acc`` in ``tl.float32`` and cast to
  the destination dtype only on store into ``y``.
* Autotune uses a 3-config Ampere (sm_86) shortlist with power-of-two
  tiles ``>= 64`` so FP16 warmup stays fast and Tensor Core–aligned.
* Neither hybrid path upcasts tensors to FP32 for the whole forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# Native compute dtypes for Triton / cuBLAS Tensor Core paths.
SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float16, torch.bfloat16, torch.float32}
)

_TRITON_IMPORT_ERROR: BaseException | None = None
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError as _exc:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = _exc


def is_triton_available() -> bool:
    """Return ``True`` if the ``triton`` package imports successfully."""
    return _HAS_TRITON


def triton_import_error() -> BaseException | None:
    return _TRITON_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Reusable CUDA buffer pool (eliminates per-call empty() churn)
# ---------------------------------------------------------------------------


@dataclass
class _FFFTritonBuffers:
    """Device buffers sized for up to ``cap_n`` tokens."""

    cap_n: int
    d_in: int
    d_out: int
    device: torch.device
    dtype: torch.dtype
    y: Tensor
    leaf_ids: Tensor
    leaf_sorted: Tensor
    order: Tensor
    x_sorted: Tensor
    y_sorted: Tensor


_BUFFER_POOL: dict[tuple[Any, ...], _FFFTritonBuffers] = {}


def _pool_key(
    device: torch.device,
    dtype: torch.dtype,
    d_in: int,
    d_out: int,
) -> tuple[Any, ...]:
    index = device.index if device.index is not None else -1
    return (device.type, index, dtype, d_in, d_out)


def _get_buffers(
    n: int,
    d_in: int,
    d_out: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _FFFTritonBuffers:
    """Return pooled buffers, growing capacity with slack when needed."""
    key = _pool_key(device, dtype, d_in, d_out)
    buf = _BUFFER_POOL.get(key)
    need = n
    if buf is not None and buf.cap_n >= need:
        return buf

    # Grow with 25% slack to absorb sequence-length jitter without realloc.
    cap = max(need, int(need * 1.25) + 8, 16 if buf is None else int(buf.cap_n * 1.5))
    cap = int(cap)
    buf = _FFFTritonBuffers(
        cap_n=cap,
        d_in=d_in,
        d_out=d_out,
        device=device,
        dtype=dtype,
        y=torch.empty((cap, d_out), device=device, dtype=dtype),
        leaf_ids=torch.empty((cap,), device=device, dtype=torch.int32),
        leaf_sorted=torch.empty((cap,), device=device, dtype=torch.int32),
        order=torch.empty((cap,), device=device, dtype=torch.int64),
        x_sorted=torch.empty((cap, d_in), device=device, dtype=dtype),
        y_sorted=torch.empty((cap, d_out), device=device, dtype=dtype),
    )
    _BUFFER_POOL[key] = buf
    return buf


def clear_triton_buffer_pool(*, empty_cache: bool = False) -> None:
    """Drop pooled buffers (e.g. after OOM). Optionally ``empty_cache()``."""
    _BUFFER_POOL.clear()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


# Fast-path (fused) thresholds. Half-precision keeps fused longer — sorting
# overhead dominates until medium microbatches on Ampere Tensor Cores.
FUSED_MAX_N: int = 4
FUSED_MAX_N_HALF: int = 16


def fused_dispatch_limit(dtype: torch.dtype) -> int:
    """Return fused-path max dispatch count for ``dtype`` (tokens or batch hint)."""
    if dtype in (torch.float16, torch.bfloat16):
        return FUSED_MAX_N_HALF
    return FUSED_MAX_N


def _dtype_code(dtype: torch.dtype) -> int:
    """Compact dtype tag for Triton autotune keys (0=fp32, 1=fp16, 2=bf16)."""
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def _as_compute_tensor(t: Tensor, dtype: torch.dtype) -> Tensor:
    """Return ``t`` in ``dtype`` without redundant casts/copies.

    Same-dtype contiguous tensors are returned as-is (avoids FP16 cast loops
    on the hybrid hot path when the model is already ``.half()`` / BF16).
    """
    if t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t if t.is_contiguous() else t.contiguous()


def _ensure_compute_dtype(*tensors: Tensor) -> torch.dtype:
    """Require a shared supported dtype across activations / weights."""
    dtype = tensors[0].dtype
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"unsupported dtype {dtype}; expected one of "
            f"{sorted(str(d) for d in SUPPORTED_DTYPES)}"
        )
    for t in tensors[1:]:
        if t.dtype != dtype:
            raise TypeError(
                f"dtype mismatch: expected {dtype}, got {t.dtype} "
                f"(cast weights to activation dtype before calling)"
            )
    return dtype


if _HAS_TRITON:

    # ------------------------------------------------------------------
    # Autotune configs — fixed Ampere (sm_86 / RTX 3060) shortlist.
    # Tile dims are powers of 2 and >= 16 (Tensor Core alignment).
    # Keep ≤3 configs so FP16 warmup does not hang on a huge search space.
    # ------------------------------------------------------------------

    def _route_autotune_configs() -> list:
        return [
            triton.Config({"BLOCK_SIZE_I": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE_I": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_I": 256}, num_warps=8, num_stages=3),
        ]

    def _fused_autotune_configs() -> list:
        # Powers of 2 >= 64 — Ampere Tensor Core–friendly sm_86 shortlist.
        return [
            triton.Config(
                {"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 128},
                num_warps=4,
                num_stages=3,
            ),
            triton.Config(
                {"BLOCK_SIZE_I": 128, "BLOCK_SIZE_O": 64},
                num_warps=4,
                num_stages=2,
            ),
            triton.Config(
                {"BLOCK_SIZE_I": 256, "BLOCK_SIZE_O": 128},
                num_warps=8,
                num_stages=3,
            ),
        ]

    def _decorate_autotune(configs: list, key: list[str]):  # type: ignore[no-untyped-def]
        def deco(fn):  # type: ignore[no-untyped-def]
            # Low warmup/rep: 3 configs × few reps is enough for sm_86.
            kwargs: dict[str, Any] = {"configs": configs, "key": key}
            try:
                return triton.autotune(**kwargs, warmup=1, rep=3)(fn)
            except TypeError:
                return triton.autotune(**kwargs)(fn)

        return deco

    # ------------------------------------------------------------------
    # Pass 1: route-only kernel → leaf_ids
    # ------------------------------------------------------------------

    @_decorate_autotune(
        _route_autotune_configs(),
        ["D_in", "DEPTH", "MAX_DIN", "DTYPE_CODE"],
    )
    @triton.jit
    def fff_route_leaf_ids_kernel(
        x_ptr,
        wr_ptr,
        br_ptr,
        leaf_ptr,
        N,
        D_in,
        num_leaves,
        stride_x_n,
        stride_x_d,
        stride_wr_r,
        stride_wr_d,
        DEPTH: tl.constexpr,
        MAX_DIN: tl.constexpr,
        BLOCK_SIZE_I: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
    ):
        """Walk the FFF tree for each token; store contiguous ``leaf_id``.

        Dot products accumulate in ``tl.float32`` regardless of input dtype
        (fp16 / bf16 / fp32) for stable router decisions.
        """
        n = tl.program_id(0)
        if n >= N:
            return

        node = tl.zeros((), dtype=tl.int32)
        for _d in tl.static_range(DEPTH):
            score = tl.zeros((), dtype=tl.float32)
            for i_start in tl.static_range(0, MAX_DIN, BLOCK_SIZE_I):
                i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
                i_mask = i_offsets < D_in
                x_tile = tl.load(
                    x_ptr + n * stride_x_n + i_offsets * stride_x_d,
                    mask=i_mask,
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    wr_ptr + node * stride_wr_r + i_offsets * stride_wr_d,
                    mask=i_mask,
                    other=0.0,
                ).to(tl.float32)
                score += tl.sum(x_tile * w_tile)
            score += tl.load(br_ptr + node).to(tl.float32)
            node = tl.where(score > 0.0, 2 * node + 2, 2 * node + 1)

        leaf_id = node - (num_leaves - 1)
        leaf_id = tl.maximum(tl.minimum(leaf_id, num_leaves - 1), 0)
        tl.store(leaf_ptr + n, leaf_id.to(tl.int32))

    # ------------------------------------------------------------------
    # Fused path (dispatch <= fused_dispatch_limit): tree walk + leaf GEMV
    # ------------------------------------------------------------------

    @_decorate_autotune(
        _fused_autotune_configs(),
        ["D_in", "D_out", "DEPTH", "MAX_DIN", "DTYPE_CODE"],
    )
    @triton.jit
    def fff_hard_forward_triton_kernel(
        x_ptr,
        wr_ptr,
        br_ptr,
        wl_ptr,
        bl_ptr,
        y_ptr,
        N,
        D_in,
        D_out,
        num_leaves,
        stride_x_n,
        stride_x_d,
        stride_wr_r,
        stride_wr_d,
        stride_wl_l,
        stride_wl_i,
        stride_wl_o,
        stride_bl_l,
        stride_bl_o,
        stride_y_n,
        stride_y_o,
        DEPTH: tl.constexpr,
        MAX_DIN: tl.constexpr,
        BLOCK_SIZE_O: tl.constexpr,
        BLOCK_SIZE_I: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
    ):
        """Fused hard routing for small batches; FP32 accumulate, store as ``y`` dtype."""
        n = tl.program_id(0)
        o_tile = tl.program_id(1)
        if n >= N:
            return

        o_offsets = o_tile * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < D_out

        node = tl.zeros((), dtype=tl.int32)
        for _d in tl.static_range(DEPTH):
            score = tl.zeros((), dtype=tl.float32)
            for i_start in tl.static_range(0, MAX_DIN, BLOCK_SIZE_I):
                i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
                i_mask = i_offsets < D_in
                x_tile = tl.load(
                    x_ptr + n * stride_x_n + i_offsets * stride_x_d,
                    mask=i_mask,
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    wr_ptr + node * stride_wr_r + i_offsets * stride_wr_d,
                    mask=i_mask,
                    other=0.0,
                ).to(tl.float32)
                score += tl.sum(x_tile * w_tile)
            score += tl.load(br_ptr + node).to(tl.float32)
            node = tl.where(score > 0.0, 2 * node + 2, 2 * node + 1)

        leaf_id = node - (num_leaves - 1)
        leaf_id = tl.maximum(tl.minimum(leaf_id, num_leaves - 1), 0)

        acc = tl.load(
            bl_ptr + leaf_id * stride_bl_l + o_offsets * stride_bl_o,
            mask=o_mask,
            other=0.0,
        ).to(tl.float32)
        for i_start in tl.static_range(0, MAX_DIN, BLOCK_SIZE_I):
            i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
            i_mask = i_offsets < D_in
            x_tile = tl.load(
                x_ptr + n * stride_x_n + i_offsets * stride_x_d,
                mask=i_mask,
                other=0.0,
            ).to(tl.float32)
            w_ptrs = (
                wl_ptr
                + leaf_id * stride_wl_l
                + i_offsets[:, None] * stride_wl_i
                + o_offsets[None, :] * stride_wl_o
            )
            w_tile = tl.load(
                w_ptrs,
                mask=i_mask[:, None] & o_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x_tile[:, None] * w_tile, axis=0)

        # Cast FP32 accumulator to destination dtype (fp16/bf16/fp32) on writeback.
        tl.store(
            y_ptr + n * stride_y_n + o_offsets * stride_y_o,
            acc.to(y_ptr.dtype.element_ty),
            mask=o_mask,
        )


def _launch_route_leaf_ids(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    leaf_ids: Tensor,
    depth: int,
) -> None:
    assert _HAS_TRITON
    n, d_in = x.shape
    num_leaves = 1 << depth
    max_din = triton.next_power_of_2(max(d_in, 1))
    dtype_code = _dtype_code(x.dtype)

    def grid(_meta: dict[str, Any]) -> tuple[int]:
        return (n,)

    fff_route_leaf_ids_kernel[grid](
        x,
        w_router,
        b_router,
        leaf_ids,
        n,
        d_in,
        num_leaves,
        x.stride(0),
        x.stride(1),
        w_router.stride(0),
        w_router.stride(1),
        DEPTH=depth,
        MAX_DIN=max_din,
        DTYPE_CODE=dtype_code,
    )


def _launch_fused(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    depth: int,
    y: Tensor,
) -> None:
    assert _HAS_TRITON
    n, d_in = x.shape
    d_out = w_leaf.shape[-1]
    num_leaves = 1 << depth
    max_din = triton.next_power_of_2(max(d_in, 1))
    dtype_code = _dtype_code(x.dtype)

    def grid(meta: dict[str, Any]) -> tuple[int, int]:
        return (n, triton.cdiv(d_out, meta["BLOCK_SIZE_O"]))

    fff_hard_forward_triton_kernel[grid](
        x,
        w_router,
        b_router,
        w_leaf,
        b_leaf,
        y,
        n,
        d_in,
        d_out,
        num_leaves,
        x.stride(0),
        x.stride(1),
        w_router.stride(0),
        w_router.stride(1),
        w_leaf.stride(0),
        w_leaf.stride(1),
        w_leaf.stride(2),
        b_leaf.stride(0),
        b_leaf.stride(1),
        y.stride(0),
        y.stride(1),
        DEPTH=depth,
        MAX_DIN=max_din,
        DTYPE_CODE=dtype_code,
    )


def _coalesced_leaf_gemm(
    x: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    leaf_ids: Tensor,
    y: Tensor,
    buf: _FFFTritonBuffers,
) -> Tensor:
    """Pass 2–3: sort by leaf_id, batched GEMM per unique leaf, scatter to ``y``.

    ``w_leaf[k]`` is loaded once per occupied leaf and reused across all tokens
    routed to that leaf. Uses ``torch.mm`` in the input dtype (fp16/bf16/fp32)
    so Ampere Tensor Cores are eligible for half-precision GEMMs — no FP32 upcast.
    """
    n = x.shape[0]
    leaf_view = leaf_ids[:n]
    leaf_sorted = buf.leaf_sorted[:n]
    order = buf.order[:n]
    x_sorted = buf.x_sorted[:n]
    y_sorted = buf.y_sorted[:n]

    # Stable sort keeps temporal locality within a leaf bucket.
    torch.sort(leaf_view, stable=True, out=(leaf_sorted, order))

    # Gather tokens into leaf-major order (writes into preallocated x_sorted).
    x_sorted.copy_(x.index_select(0, order))

    uniq, counts = torch.unique_consecutive(leaf_sorted, return_counts=True)
    # Single host sync for ≤ num_leaves entries (typically ≤ 64).
    uniq_list = uniq.tolist()
    counts_list = counts.tolist()

    offset = 0
    for leaf_k, cnt in zip(uniq_list, counts_list):
        sl = slice(offset, offset + cnt)
        # y = x @ W[leaf] + b[leaf] in-place in ``x.dtype`` (Tensor Core path).
        torch.mm(x_sorted[sl], w_leaf[leaf_k], out=y_sorted[sl])
        y_sorted[sl].add_(b_leaf[leaf_k])
        offset += cnt

    # Scatter back to original token order into preallocated y[:n].
    y_out = y[:n]
    y_out.index_copy_(0, order, y_sorted)
    return y_out


def fff_hard_forward_triton(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    depth: int,
    *,
    out: Tensor | None = None,
    dispatch_n: int | None = None,
) -> Tensor:
    """Hard-routing forward with hybrid fused / sorted paths (fp16/bf16/fp32).

    Parameters
    ----------
    x:
        ``(N, D_in)`` CUDA contiguous (``N`` may be ``B·T`` after flatten).
        Supported dtypes: ``float16``, ``bfloat16``, ``float32``.
    w_router, b_router, w_leaf, b_leaf, depth:
        FFF tree parameters; must share ``x.dtype`` (no silent FP32 upcast).
    out:
        Optional ``(N, D_out)`` preallocated output (else pooled buffer is used).
    dispatch_n:
        Count used for hybrid selection. Defaults to ``N = x.shape[0]``.
        Pass the microbatch size ``B`` from ``(B, T, D)`` callers so small
        batches stay fused even when ``N = B·T`` is large. FP16/BF16 use
        ``FUSED_MAX_N_HALF`` (16); FP32 uses ``FUSED_MAX_N`` (4).
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "triton is not installed. pip install triton  "
            f"(import error: {_TRITON_IMPORT_ERROR})"
        )
    if x.device.type != "cuda":
        raise RuntimeError("fff_hard_forward_triton requires CUDA tensors")
    if depth < 1 or depth > 12:
        raise ValueError(f"depth must be in [1, 12], got {depth}")

    x = x.contiguous()
    # Align weights to activation dtype without promoting to FP32 or re-casting
    # tensors that are already in the correct half/BF16/FP32 dtype.
    compute_dtype = x.dtype
    if compute_dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"unsupported x.dtype={compute_dtype}; expected one of "
            f"{sorted(str(d) for d in SUPPORTED_DTYPES)}"
        )
    wr = _as_compute_tensor(w_router, compute_dtype)
    br = _as_compute_tensor(b_router, compute_dtype)
    wl = _as_compute_tensor(w_leaf, compute_dtype)
    bl = _as_compute_tensor(b_leaf, compute_dtype)
    _ensure_compute_dtype(x, wr, br, wl, bl)

    if x.ndim != 2:
        raise ValueError(f"x must be (N, D_in), got {tuple(x.shape)}")
    n, d_in = x.shape
    # Dispatch count: explicit batch hint, else total flattened tokens N.
    route_n = int(dispatch_n) if dispatch_n is not None else n
    if route_n < 1:
        raise ValueError(f"dispatch_n must be >= 1, got {route_n}")
    fused_limit = fused_dispatch_limit(compute_dtype)
    num_leaves = 1 << depth
    num_routers = num_leaves - 1
    if wr.shape != (num_routers, d_in):
        raise ValueError(
            f"w_router shape {tuple(wr.shape)} != ({num_routers}, {d_in})"
        )
    if br.shape != (num_routers,):
        raise ValueError(f"b_router shape {tuple(br.shape)} != ({num_routers},)")
    if wl.ndim != 3 or wl.shape[0] != num_leaves or wl.shape[1] != d_in:
        raise ValueError(
            f"w_leaf shape {tuple(wl.shape)} incompatible with depth/D_in"
        )
    d_out = wl.shape[2]
    if bl.shape != (num_leaves, d_out):
        raise ValueError(f"b_leaf shape {tuple(bl.shape)} != ({num_leaves}, {d_out})")
    if d_in > 4096:
        raise ValueError(f"D_in={d_in} exceeds supported max (4096)")

    try:
        buf = _get_buffers(n, d_in, d_out, x.device, compute_dtype)
        if out is None:
            y = buf.y
        else:
            if out.shape != (n, d_out) or out.device != x.device or out.dtype != compute_dtype:
                raise ValueError(
                    f"out must be ({n}, {d_out}) {compute_dtype} on {x.device}, "
                    f"got {tuple(out.shape)} {out.dtype} on {out.device}"
                )
            y = out.contiguous()

        # FP16/BF16: fused for route_n <= 16; FP32: <= 4. Else sorted GEMM.
        if route_n <= fused_limit:
            _launch_fused(x, wr, br, wl, bl, depth, y[:n])
            return y[:n]

        # Large batch: Pass 1 route → Pass 2–3 sort + coalesced GEMM per leaf.
        _launch_route_leaf_ids(x, wr, br, buf.leaf_ids[:n], depth)
        return _coalesced_leaf_gemm(x, wl, bl, buf.leaf_ids, y, buf)

    except torch.cuda.OutOfMemoryError:
        clear_triton_buffer_pool(empty_cache=True)
        raise


@torch.no_grad()
def warmup_fff_hard_triton(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    depth: int,
    *,
    n_iters: int = 5,
) -> None:
    """Force Triton autotune + buffer pool init before benchmark timing."""
    if not _HAS_TRITON or x.device.type != "cuda":
        return
    # Warm fused (dispatch <= limit) and sorted (dispatch > limit) paths.
    limit = fused_dispatch_limit(x.dtype)
    for _ in range(max(n_iters, 1)):
        fff_hard_forward_triton(
            x, w_router, b_router, w_leaf, b_leaf, depth, dispatch_n=1
        )
        fff_hard_forward_triton(
            x,
            w_router,
            b_router,
            w_leaf,
            b_leaf,
            depth,
            dispatch_n=limit + 1,
        )
    torch.cuda.synchronize(x.device)


@torch.no_grad()
def warmup_fff_model_triton(
    model: Any,
    sample_tokens: int = 32,
    n_iters: int = 5,
) -> None:
    """Warmup every FFF layer (fused small-batch + sorted large-batch paths)."""
    if not _HAS_TRITON:
        return
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    layers = list(model.fff_layers()) if hasattr(model, "fff_layers") else []
    if not layers:
        return

    d_model = int(layers[0].in_features)
    dtype = next(model.parameters()).dtype
    if dtype not in SUPPORTED_DTYPES:
        dtype = torch.float32
    n = max(int(sample_tokens), fused_dispatch_limit(dtype) + 1)
    x = torch.randn(n, d_model, device=device, dtype=dtype)
    for layer in layers:
        warmup_fff_hard_triton(
            x,
            layer.router_weights.detach(),
            layer.router_biases.detach(),
            layer.leaf_weights.detach(),
            layer.leaf_biases.detach(),
            int(layer.depth),
            n_iters=n_iters,
        )
    torch.cuda.synchronize(device)


def estimate_triton_occupancy(d_out: int, block_size_o: int = 64) -> int:
    """Helper: number of output tiles for a given ``D_out`` (debug / logging)."""
    return int(math.ceil(d_out / float(block_size_o)))
