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

    def _ampere_autotune_configs() -> list:
        """Autotune grid for RTX 3060 (sm_86 / Ampere).

        Sweeps ``BLOCK_SIZE_I × BLOCK_SIZE_O × num_warps × num_stages`` while
        dropping tiles that put excessive register / SRAM pressure on the fused
        tree-walk + leaf GEMV (``w_leaf`` tile is ``BLOCK_I × BLOCK_O``).
        """
        configs: list = []
        for block_i in (64, 128, 256, 512):
            for block_o in (32, 64, 128, 256):
                # Bound the GEMV tile: float32 tile ≈ 4 * Bi * Bo bytes (+ x/b).
                # Keep Bi*Bo ≤ 16K elements (~64 KiB) to leave headroom for
                # tree-walk registers and software-pipelined stages.
                if block_i * block_o > 16_384:
                    continue
                for num_warps in (2, 4, 8):
                    # Large tiles need enough warps to hide latency.
                    if block_i * block_o >= 8_192 and num_warps < 4:
                        continue
                    if block_i >= 512 and num_warps < 4:
                        continue
                    for num_stages in (2, 3, 4):
                        # Deep pipelining + huge tiles → register spills on Ampere.
                        if num_stages >= 4 and block_i * block_o >= 8_192:
                            continue
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_I": block_i,
                                    "BLOCK_SIZE_O": block_o,
                                },
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
        return configs

    _AUTOTUNE_CONFIGS = _ampere_autotune_configs()

    def _decorate_autotune(fn):  # type: ignore[no-untyped-def]
        """Apply ``@triton.autotune`` with warmup/rep when the installed Triton supports them."""
        kwargs: dict[str, Any] = {
            "configs": _AUTOTUNE_CONFIGS,
            "key": ["D_in", "D_out", "DEPTH", "MAX_DIN"],
        }
        # Triton 2.x accepts warmup/rep; some 3.x builds may not — degrade gracefully.
        try:
            return triton.autotune(**kwargs, warmup=5, rep=25)(fn)
        except TypeError:
            return triton.autotune(**kwargs)(fn)

    @_decorate_autotune
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
        """Fused FFF hard routing: tree walk then leaf GEMV (one launch).

        Autotune favors smaller ``BLOCK_SIZE_O`` when register pressure from the
        depth-loop is high, and larger ``BLOCK_SIZE_I`` to reuse ``x`` / ``w_leaf``
        rows from SRAM across the output tile.
        """
        n = tl.program_id(0)
        o_tile = tl.program_id(1)

        if n >= N:
            return

        o_offsets = o_tile * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < D_out

        # ------------------------------------------------------------------
        # Step 1 — tree traversal (path entirely in registers)
        # Keep temporaries scalar / small tiles to limit register pressure.
        # ------------------------------------------------------------------
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

        leaf_base = num_leaves - 1
        leaf_id = node - leaf_base
        leaf_id = tl.maximum(tl.minimum(leaf_id, num_leaves - 1), 0)

        # ------------------------------------------------------------------
        # Step 2 — leaf GEMV: stream w_leaf[leaf] rows (SRAM-friendly Bi×Bo)
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
    """Force Triton autotune + CUDA compile before benchmark timing.

    Runs ``n_iters`` launches for the given shape key
    ``(D_in, D_out, DEPTH, MAX_DIN)`` then synchronizes so subsequent timed
    calls hit the cached best config (autotune cost excluded from latency).
    """
    if not _HAS_TRITON:
        return
    if x.device.type != "cuda":
        return
    out = torch.empty(
        (x.shape[0], w_leaf.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    for _ in range(max(n_iters, 1)):
        fff_hard_forward_triton(
            x,
            w_router,
            b_router,
            w_leaf,
            b_leaf,
            depth,
            out=out,
        )
    torch.cuda.synchronize(x.device)


@torch.no_grad()
def warmup_fff_model_triton(
    model: Any,
    sample_tokens: int = 32,
    n_iters: int = 5,
) -> None:
    """Warmup every FFF layer in an ``FFFTransformer`` (or ModuleList of FFFs).

    Builds a representative ``(sample_tokens, d_model)`` activation so autotune
    keys match inference-time shapes.
    """
    if not _HAS_TRITON:
        return
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    layers = list(model.fff_layers()) if hasattr(model, "fff_layers") else []
    if not layers:
        return

    d_model = int(layers[0].in_features)
    x = torch.randn(sample_tokens, d_model, device=device, dtype=torch.float32)
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
