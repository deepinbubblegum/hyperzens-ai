"""Triton CUDA FFF hard routing with hybrid fused / leaf-sorted paths.

Hybrid dispatch (``N = B·T`` flattened tokens)
---------------------------------------------
* **``N <= 4`` (small / decode):** single fused Triton kernel
  (tree-walk + leaf GEMV). Avoids sort/bucket overhead — best for
  single-token and tiny-batch generation (~189+ tok/s).
* **``N > 4`` (batched inference):** multi-pass leaf-sorted path
  1. Pass 1 — route: Triton writes ``leaf_id[n]`` for all tokens.
  2. Pass 2 — sort: ``torch.sort`` buckets by ``leaf_id``.
  3. Pass 3 — coalesced leaf GEMM: one ``mm`` per unique leaf so
     ``w_leaf[k]`` is loaded once and reused (~513+ tok/s at large N).

Buffers for ``y``, ``leaf_ids``, sort indices, and sorted ``x``/``y`` are
pooled and reused to avoid allocator churn / fragmentation on RTX 3060.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


# Max token count that uses the fused kernel (no sort). Above this → grouped path.
FUSED_MAX_N: int = 4


if _HAS_TRITON:

    # ------------------------------------------------------------------
    # Autotune configs (Ampere / sm_86)
    # ------------------------------------------------------------------

    def _route_autotune_configs() -> list:
        configs: list = []
        for block_i in (64, 128, 256, 512):
            for num_warps in (2, 4, 8):
                if block_i >= 512 and num_warps < 4:
                    continue
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_I": block_i},
                        num_warps=num_warps,
                        num_stages=2,
                    )
                )
        return configs

    def _fused_autotune_configs() -> list:
        configs: list = []
        for block_i in (64, 128, 256, 512):
            for block_o in (32, 64, 128, 256):
                if block_i * block_o > 16_384:
                    continue
                for num_warps in (2, 4, 8):
                    if block_i * block_o >= 8_192 and num_warps < 4:
                        continue
                    if block_i >= 512 and num_warps < 4:
                        continue
                    for num_stages in (2, 3, 4):
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

    def _decorate_autotune(configs: list, key: list[str]):  # type: ignore[no-untyped-def]
        def deco(fn):  # type: ignore[no-untyped-def]
            kwargs: dict[str, Any] = {"configs": configs, "key": key}
            try:
                return triton.autotune(**kwargs, warmup=5, rep=25)(fn)
            except TypeError:
                return triton.autotune(**kwargs)(fn)

        return deco

    # ------------------------------------------------------------------
    # Pass 1: route-only kernel → leaf_ids
    # ------------------------------------------------------------------

    @_decorate_autotune(_route_autotune_configs(), ["D_in", "DEPTH", "MAX_DIN"])
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
    ):
        """Walk the FFF tree for each token; store contiguous ``leaf_id``."""
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
    # Fused path (N <= FUSED_MAX_N): tree walk + leaf GEMV in one launch
    # ------------------------------------------------------------------

    @_decorate_autotune(
        _fused_autotune_configs(),
        ["D_in", "D_out", "DEPTH", "MAX_DIN"],
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
        """Fused hard routing for small ``N`` (no cross-token leaf reuse)."""
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

        tl.store(
            y_ptr + n * stride_y_n + o_offsets * stride_y_o,
            acc,
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
    routed to that leaf (coalesced / cuBLAS-friendly).
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
        # y = x @ W[leaf] + b[leaf]; W is (D_in, D_out)
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
) -> Tensor:
    """Hard-routing forward with hybrid fused (``N<=4``) / sorted (``N>4``) paths.

    Parameters
    ----------
    x:
        ``(N, D_in)`` CUDA contiguous.
    w_router, b_router, w_leaf, b_leaf, depth:
        FFF tree parameters (see module docstring).
    out:
        Optional ``(N, D_out)`` preallocated output (else pooled buffer is used).
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
    n, d_in = x.shape
    num_leaves = 1 << depth
    num_routers = num_leaves - 1
    if w_router.shape != (num_routers, d_in):
        raise ValueError(
            f"w_router shape {tuple(w_router.shape)} != ({num_routers}, {d_in})"
        )
    if b_router.shape != (num_routers,):
        raise ValueError(f"b_router shape {tuple(b_router.shape)} != ({num_routers},)")
    if w_leaf.ndim != 3 or w_leaf.shape[0] != num_leaves or w_leaf.shape[1] != d_in:
        raise ValueError(
            f"w_leaf shape {tuple(w_leaf.shape)} incompatible with depth/D_in"
        )
    d_out = w_leaf.shape[2]
    if b_leaf.shape != (num_leaves, d_out):
        raise ValueError(f"b_leaf shape {tuple(b_leaf.shape)} != ({num_leaves}, {d_out})")
    if d_in > 4096:
        raise ValueError(f"D_in={d_in} exceeds supported max (4096)")

    wr = w_router.to(dtype=x.dtype)
    br = b_router.to(dtype=x.dtype)
    wl = w_leaf.to(dtype=x.dtype)
    bl = b_leaf.to(dtype=x.dtype)

    try:
        buf = _get_buffers(n, d_in, d_out, x.device, x.dtype)
        if out is None:
            y = buf.y
        else:
            if out.shape != (n, d_out) or out.device != x.device or out.dtype != x.dtype:
                raise ValueError(
                    f"out must be ({n}, {d_out}) {x.dtype} on {x.device}, "
                    f"got {tuple(out.shape)} {out.dtype} on {out.device}"
                )
            y = out.contiguous()

        # Small N: fused kernel — skip sort/bucket overhead (decode / microbatch).
        if n <= FUSED_MAX_N:
            _launch_fused(x, wr, br, wl, bl, depth, y[:n])
            return y[:n]

        # Large N: Pass 1 route → Pass 2–3 sort + coalesced GEMM per leaf.
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
    # Warm fused (N<=4) and sorted (N>4) paths when sample size allows.
    n = x.shape[0]
    x_fused = x[: min(n, FUSED_MAX_N)].contiguous()
    for _ in range(max(n_iters, 1)):
        fff_hard_forward_triton(x_fused, w_router, b_router, w_leaf, b_leaf, depth)
        if n > FUSED_MAX_N:
            fff_hard_forward_triton(x, w_router, b_router, w_leaf, b_leaf, depth)
    torch.cuda.synchronize(x.device)


@torch.no_grad()
def warmup_fff_model_triton(
    model: Any,
    sample_tokens: int = 32,
    n_iters: int = 5,
) -> None:
    """Warmup every FFF layer (fused ``N<=4`` + sorted ``N>4`` paths)."""
    if not _HAS_TRITON:
        return
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    layers = list(model.fff_layers()) if hasattr(model, "fff_layers") else []
    if not layers:
        return

    d_model = int(layers[0].in_features)
    # N > FUSED_MAX_N so both hybrid branches are compiled + autotuned.
    n = max(int(sample_tokens), FUSED_MAX_N + 1)
    x = torch.randn(n, d_model, device=device, dtype=torch.float32)
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
