"""Fused Triton CUDA kernel for FFF hard routing (tree walk + leaf GEMV).

Designed for NVIDIA Ampere (RTX 3060 / ``sm_86``) GPU inference. Each program
instance owns one token ``n`` and a tile of output features; the binary-tree
path is resolved in registers, then a single leaf GEMV writes ``y[n, :]``.

Falls back gracefully when Triton is not installed or the tensors are not on CUDA
(callers should use :meth:`models.fff_layer.FastFeedforwardLinear.forward_hard`).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

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


if _HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_O": 64, "BLOCK_SIZE_I": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_O": 128, "BLOCK_SIZE_I": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_O": 64, "BLOCK_SIZE_I": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE_O": 32, "BLOCK_SIZE_I": 64}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_SIZE_O": 128, "BLOCK_SIZE_I": 64}, num_warps=8, num_stages=3),
        ],
        key=["D_in", "D_out", "DEPTH", "MAX_DIN"],
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
    ):
        """Fused FFF hard routing: tree walk then leaf GEMV (one launch)."""
        n = tl.program_id(0)
        o_tile = tl.program_id(1)

        if n >= N:
            return

        o_offsets = o_tile * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < D_out

        # ------------------------------------------------------------------
        # Step 1 — tree traversal (path entirely in registers)
        # ------------------------------------------------------------------
        node = tl.zeros((), dtype=tl.int32)
        for _d in tl.static_range(DEPTH):
            score = tl.zeros((), dtype=tl.float32)
            # Dot(x[n], w_router[node]) tiled over D_in
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
            # s > 0 → right (2i+2), else left (2i+1)
            node = tl.where(score > 0.0, 2 * node + 2, 2 * node + 1)

        leaf_base = num_leaves - 1
        leaf_id = node - leaf_base
        leaf_id = tl.maximum(tl.minimum(leaf_id, num_leaves - 1), 0)

        # ------------------------------------------------------------------
        # Step 2 — leaf GEMV for this output tile
        # ------------------------------------------------------------------
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
            w_mask = i_mask[:, None] & o_mask[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            acc += tl.sum(x_tile[:, None] * w_tile, axis=0)

        tl.store(
            y_ptr + n * stride_y_n + o_offsets * stride_y_o,
            acc,
            mask=o_mask,
        )


def _launch_fff_hard_triton(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    depth: int,
    y: Tensor,
) -> Tensor:
    assert _HAS_TRITON
    N, D_in = x.shape
    D_out = w_leaf.shape[-1]
    num_leaves = 1 << depth

    # Grid: one program per (token, output-tile). BLOCK_SIZE_O comes from autotune.
    # Use a conservative meta default for grid sizing; autotune picks real BLOCK_SIZE_O.
    def grid(meta: dict[str, Any]):
        return (
            N,
            triton.cdiv(D_out, meta["BLOCK_SIZE_O"]),
        )

    fff_hard_forward_triton_kernel[grid](
        x,
        w_router,
        b_router,
        w_leaf,
        b_leaf,
        y,
        N,
        D_in,
        D_out,
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
        MAX_DIN=triton.next_power_of_2(max(D_in, 1)),
    )
    return y


def fff_hard_forward_triton(
    x: Tensor,
    w_router: Tensor,
    b_router: Tensor,
    w_leaf: Tensor,
    b_leaf: Tensor,
    depth: int,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """Launch the fused hard-routing Triton kernel.

    Parameters
    ----------
    x:
        ``(N, D_in)`` CUDA contiguous float16/float32 (or castable).
    w_router:
        ``(num_routers, D_in)``
    b_router:
        ``(num_routers,)``
    w_leaf:
        ``(num_leaves, D_in, D_out)``
    b_leaf:
        ``(num_leaves, D_out)``
    depth:
        Tree depth ``d`` (``num_leaves = 2^d``).
    out:
        Optional preallocated ``(N, D_out)`` on the same device/dtype as ``x``.

    Returns
    -------
    Tensor
        ``y`` of shape ``(N, D_out)`` (``out`` if provided).
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
    w_router = w_router.contiguous()
    b_router = b_router.contiguous()
    w_leaf = w_leaf.contiguous()
    b_leaf = b_leaf.contiguous()

    if x.ndim != 2:
        raise ValueError(f"x must be (N, D_in), got {tuple(x.shape)}")
    N, D_in = x.shape
    num_leaves = 1 << depth
    num_routers = num_leaves - 1
    if w_router.shape != (num_routers, D_in):
        raise ValueError(
            f"w_router shape {tuple(w_router.shape)} != ({num_routers}, {D_in})"
        )
    if b_router.shape != (num_routers,):
        raise ValueError(f"b_router shape {tuple(b_router.shape)} != ({num_routers},)")
    if w_leaf.ndim != 3 or w_leaf.shape[0] != num_leaves or w_leaf.shape[1] != D_in:
        raise ValueError(
            f"w_leaf shape {tuple(w_leaf.shape)} incompatible with depth/D_in"
        )
    D_out = w_leaf.shape[2]
    if b_leaf.shape != (num_leaves, D_out):
        raise ValueError(f"b_leaf shape {tuple(b_leaf.shape)} != ({num_leaves}, {D_out})")
    if D_in > 4096:
        raise ValueError(f"D_in={D_in} exceeds supported max (4096) for Triton kernel")

    # Pre-allocate to avoid allocator churn / fragmentation under repeated infer.
    if out is None:
        y = torch.empty((N, D_out), device=x.device, dtype=x.dtype)
    else:
        if out.shape != (N, D_out) or out.device != x.device or out.dtype != x.dtype:
            raise ValueError(
                f"out must be ({N}, {D_out}) {x.dtype} on {x.device}, "
                f"got {tuple(out.shape)} {out.dtype} on {out.device}"
            )
        y = out.contiguous()

    # Kernel accumulates in fp32; store casts back via tl.store into y dtype.
    # Match dtypes across operands (promote to fp32 weights if x is fp16 for stability,
    # but keep x/y dtype for bandwidth — cast loads inside kernel to fp32).
    _launch_fff_hard_triton(
        x,
        w_router.to(dtype=x.dtype),
        b_router.to(dtype=x.dtype),
        w_leaf.to(dtype=x.dtype),
        b_leaf.to(dtype=x.dtype),
        depth,
        y,
    )
    return y


def estimate_triton_occupancy(d_out: int, block_size_o: int = 64) -> int:
    """Helper: number of output tiles for a given ``D_out`` (debug / logging)."""
    return int(math.ceil(d_out / float(block_size_o)))
