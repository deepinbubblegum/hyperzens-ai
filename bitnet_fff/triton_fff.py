"""Fused Triton kernel for BitNet Fast Feedforward (FFF) layers on NVIDIA CUDA.

A balanced binary decision tree routes every token to exactly one of
``2**depth`` ternary BitNet b1.58 leaves. The reference PyTorch path
(:func:`fff_forward_ref` / :func:`fff_forward_native`) mirrors
:class:`~bitnet_fff.fast_fff.FastFeedForwardBitNet`; the Triton path
(:func:`_triton_forward`) fuses *all* of the following into a single kernel
program per block of rows, so no intermediate router-logit / gathered-weight
tensors are ever materialized in VRAM:

1. **Fused routing + leaf projection** — one program loads ``X``
   (``BLOCK_N, d_model``), computes the ``num_leaves - 1`` router decisions,
   walks the tree in registers, and evaluates the selected leaf's ternary
   ``(d_model, d_model)`` matmul before writing ``(BLOCK_N, d_model)``.
2. **BitNet ternary arithmetic** — inputs are AbsMax-scaled per token; router
   logits are ``sign``-weighted sums computed as *subtractions of masked
   reductions* (``sum(x where w>0) - sum(x where w<0)``) instead of FP32
   multiply-accumulate, exactly the fast add/sub trick for weights in
   {-1, 0, +1}.
3. **In-register tree traversal** — the ``depth``-step conditional walk uses
   the heap layout ``node = 2*node + bit`` with ``bit = logit >= 0``; the per
   node logit is read out of the register-resident logit tile by a masked
   select, so the whole path is decided without shared memory or atomics.
4. **Coalesced loads / SRAM reuse** — ``BLOCK_N`` rows load contiguous
   ``X``, the router/leaf weight tiles are loaded into SRAM-backed registers,
   and leaf weights are stored transposed (``L, d_model, d_model`` -> row-major
   ``L, K, D``) so the leaf ``tl.dot`` needs no transpose. Output is FP16.

The leaf matmul is executed as ``32`` tensor-core ``tl.dot`` tiles, each masked
to the rows that actually routed to that leaf. This computes ``num_leaves``x
the leaf FLOPs (the fused-kernel tradeoff for data-dependent leaf selection)
but runs entirely on the MMA pipe at peak throughput and keeps the kernel a
single fused launch.

Reference: Belcak & Wattenhofer, "Fast Feedforward Networks" (NeurIPS 2023),
and the repo's ``FastFeedForwardBitNet`` routing semantics (heap tree, hard
``logit >= 0`` decisions, raw-x router, AbsMax 8-bit activation quantization).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # triton is optional (CUDA only); the module still imports without it.
    import triton
    import triton.language as tl

    has_triton = True
except Exception:  # pragma: no cover - exercised only on non-CUDA machines
    triton = None
    tl = None
    has_triton = False

__all__ = [
    "has_triton",
    "fff_forward_ref",
    "fff_forward_native",
    "FFFTriton",
    "FFFTritonLayer",
]

_DEFAULT_BLOCK_N = 64


# --------------------------------------------------------------------------
# reference implementations (device/dtype agnostic)
# --------------------------------------------------------------------------


def _ternarize(
    w: torch.Tensor, eps: float = 1e-8, threshold_scale: float = 1.0
) -> torch.Tensor:
    """AbsMean ternary weights in {-1, 0, +1} (same rule as bitlinear)."""
    w = w.detach()
    threshold = w.abs().mean().clamp_min(eps) * threshold_scale
    return torch.where(w.abs() > threshold, torch.sign(w), torch.zeros_like(w))


def _absmax_quantize_rah(
    x: torch.Tensor, bits: int | None, eps: float = 1e-8
) -> torch.Tensor:
    """AbsMax k-bit activation quantization with round-half-away-from-zero.

    Rounding matches the Triton kernel's ``floor(x + 0.5)``/``ceil(x - 0.5)``
    exactly (vs ``torch.round``'s half-to-even), so the CPU fallback and the
    kernel agree bit-for-bit on the quantized values.
    """
    if bits is None or bits < 2 or bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    q = (x / scale) * qmax
    q = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5))
    q = q.clamp(-(qmax + 1), qmax)
    return (q / qmax) * scale


def _route(
    x: torch.Tensor,
    wq_router: torch.Tensor,
    b_router: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Heap-tree routing on raw ``x``: returns the leaf index per token."""
    batch = x.shape[0]
    device = x.device
    node = torch.zeros(batch, dtype=torch.long, device=device)
    for _ in range(depth):
        logit = (x * wq_router[node]).sum(-1) + b_router[node]
        node = 2 * node + 1 + (logit >= 0).to(torch.long)
    return node - (2 ** depth - 1)


def fff_forward_ref(
    x: torch.Tensor,
    wq_router: torch.Tensor,
    b_router: torch.Tensor,
    wq_leaf: torch.Tensor,
    b_leaf: torch.Tensor | None = None,
    depth: int = 5,
    activation_bits: int = 8,
) -> torch.Tensor:
    """Reference forward with *already-ternarized* weights.

    ``wq_router`` is ``(num_leaves - 1, d_model)``, ``wq_leaf`` is
    ``(num_leaves, d_model, d_model)`` in {-1, 0, +1}. Works in the input
    dtype; used for correctness testing, the CPU fallback and the "PyTorch
    native" benchmark baseline.
    """
    x = x.detach()
    batch, _ = x.shape
    depth = int(depth)
    num_leaves = 1 << depth
    leaf = _route(x, wq_router, b_router, depth)
    xq = _absmax_quantize_rah(x, activation_bits)
    w_sel = wq_leaf[leaf]  # (batch, d_model, d_model)
    out = torch.bmm(xq.unsqueeze(1), w_sel.transpose(1, 2)).squeeze(1)
    if b_leaf is not None:
        out = out + b_leaf[leaf]
    return out


def fff_forward_native(
    x: torch.Tensor,
    w_router: torch.Tensor,
    b_router: torch.Tensor,
    w_leaf: torch.Tensor,
    b_leaf: torch.Tensor | None = None,
    depth: int = 5,
    activation_bits: int = 8,
    eps: float = 1e-8,
    threshold_scale: float = 1.0,
) -> torch.Tensor:
    """PyTorch-native FFF forward (AbsMean-ternarizes the weights internally)."""
    return fff_forward_ref(
        x,
        _ternarize(w_router, eps, threshold_scale),
        b_router,
        _ternarize(w_leaf, eps, threshold_scale),
        b_leaf,
        depth,
        activation_bits,
    )


# --------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------


@triton.jit
def _fff_forward_kernel(
    X, WR, BR, WL, BL, Y,
    N,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    DEPTH: tl.constexpr,
    LEAVES: tl.constexpr,
    NODES: tl.constexpr,
    QMAX: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Fused FFF forward for one block of ``BLOCK_N`` rows.

    ``WL`` holds the leaf weights **transposed** per leaf (``L, K, D``) so the
    leaf matmul is a direct ``tl.dot(xq, wl)`` with no register transpose.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    offs_d = tl.arange(0, D)
    offs_l = tl.arange(0, LEAVES)

    # ---- load input block + BitNet AbsMax activation quantize ----
    x = tl.load(X + offs_n[:, None] * K + offs_k[None, :],
                mask=offs_n[:, None] < N, other=0.0)
    x = x.to(tl.float32)
    if QMAX > 0:
        scale = tl.max(tl.abs(x), axis=1)                 # (BLOCK_N,)
        scale = tl.maximum(scale, 1e-8)
        xq = x / scale[:, None] * QMAX
        xq = tl.where(xq >= 0, tl.floor(xq + 0.5), tl.ceil(xq - 0.5))
        xq = tl.minimum(tl.maximum(xq, -(QMAX + 1)), QMAX)
        xq = (xq / QMAX) * scale[:, None]
    else:
        xq = x
    xq = xq.to(tl.float16)

    # ---- router logits via ternary add/sub (no full FP32 multiplies) ----
    # logits[., node] = sum(x where w>0) - sum(x where w<0) + bias
    logits = tl.zeros((BLOCK_N, LEAVES), dtype=tl.float32)
    for node in tl.static_range(LEAVES):
        w = tl.load(WR + node * K + offs_k, mask=node < NODES, other=0.0)
        b = tl.load(BR + node, mask=node < NODES, other=0.0)
        pos = tl.sum(tl.where(w > 0, x, 0.0), axis=1)
        neg = tl.sum(tl.where(w < 0, x, 0.0), axis=1)
        logits += tl.where(offs_l[None, :] == node, pos - neg + b, 0.0)

    # ---- conditional tree traversal in registers ----
    leaf = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for level in tl.static_range(DEPTH):
        node_id = (1 << level) - 1 + leaf
        logit = tl.sum(
            tl.where(offs_l[None, :] == node_id[:, None], logits, 0.0), axis=1
        )
        leaf = 2 * leaf + (logit >= 0).to(tl.int32)

    # ---- target-leaf matmul: masked tensor-core dots ----
    acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    for l in tl.static_range(LEAVES):
        wl = tl.load(WL + l * K * D + offs_k[:, None] * D + offs_d[None, :])
        acc += tl.dot(xq, wl) * (leaf == l).to(tl.float16)[:, None]
    if HAS_BIAS:
        for l in tl.static_range(LEAVES):
            bl = tl.load(BL + l * D + offs_d)
            acc += (leaf == l).to(tl.float16)[:, None] * bl[None, :]

    out = acc.to(tl.float16)
    tl.store(Y + offs_n[:, None] * D + offs_d[None, :], out,
             mask=offs_n[:, None] < N)


def _triton_forward(
    x: torch.Tensor,
    wq_router: torch.Tensor,
    b_router: torch.Tensor,
    wq_leaf: torch.Tensor,
    b_leaf: torch.Tensor | None,
    depth: int,
    activation_bits: int | None,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = 8,
) -> torch.Tensor:
    """Run the fused Triton FFF kernel; inputs must be CUDA tensors."""
    if not has_triton:  # pragma: no cover - guarded by callers
        raise RuntimeError("triton is not installed")
    batch, k_dim = x.shape
    num_leaves, _, d_dim = wq_leaf.shape
    device = x.device

    wq_router_f16 = wq_router.to(torch.float16).contiguous()
    b_router_f32 = b_router.float().contiguous()
    wq_leaf_t = wq_leaf.transpose(1, 2).to(torch.float16).contiguous()
    qmax = (
        2 ** (int(activation_bits) - 1) - 1
        if (activation_bits is not None and activation_bits < 32)
        else 0
    )

    y = torch.empty((batch, d_dim), dtype=torch.float16, device=device)
    grid = (triton.cdiv(batch, block_n),)
    _fff_forward_kernel[grid](
        x, wq_router_f16, b_router_f32, wq_leaf_t,
        b_leaf.float().contiguous() if b_leaf is not None else x.new_empty(0),
        y, batch,
        BLOCK_N=block_n, K=k_dim, D=d_dim, DEPTH=depth, LEAVES=num_leaves,
        NODES=num_leaves - 1, QMAX=qmax, HAS_BIAS=b_leaf is not None,
        num_warps=num_warps,
    )
    return y


# --------------------------------------------------------------------------
# autograd wrapper (Triton forward on CUDA, torch reference elsewhere;
# backward is computed in torch with BitNet-FFF STE semantics)
# --------------------------------------------------------------------------


def _route_with_probs(
    x: torch.Tensor,
    wq_router: torch.Tensor,
    b_router: torch.Tensor,
    depth: int,
):
    """Route and return leaf, per-level ``(sigmoid r, node ids, go_right)``."""
    batch = x.shape[0]
    device = x.device
    node = torch.zeros(batch, dtype=torch.long, device=device)
    r_list: list[torch.Tensor] = []
    node_ids: list[torch.Tensor] = []
    go_right_list: list[torch.Tensor] = []
    for _ in range(depth):
        node_ids.append(node.clone())
        logit = (x * wq_router[node]).sum(-1) + b_router[node]
        r = torch.sigmoid(logit)
        go_right = logit >= 0
        r_list.append(r)
        go_right_list.append(go_right)
        node = 2 * node + 1 + go_right.to(torch.long)
    return node - (2 ** depth - 1), r_list, node_ids, go_right_list


class FFFTriton(torch.autograd.Function):
    """Fused Triton FFF with straight-through BitNet b1.58 gradients.

    Forward:
        * CUDA + triton: the fused kernel above (no intermediate VRAM).
        * otherwise: the exact-same-math torch reference (for tests/devices
          without a GPU).

    Backward (BitNet/FFF conventions, matching ``FastFeedForwardBitNet`` in
    train mode):
        * activations/weights flow through the AbsMax and ternary quantizers
          via STE (identity local gradient);
        * the hard routing itself is not differentiable, but the training path
          probability gate ``(path_prob - path_prob.detach()) * out.detach()``
          backpropagates into the traversed router logits via the sigmoid.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_router: torch.Tensor,
        b_router: torch.Tensor,
        w_leaf: torch.Tensor,
        b_leaf: torch.Tensor | None,
        depth: int = 5,
        activation_bits: int = 8,
        eps: float = 1e-8,
        threshold_scale: float = 1.0,
    ) -> torch.Tensor:
        x = x.contiguous()
        wq_router = _ternarize(w_router, eps, threshold_scale)
        wq_leaf = _ternarize(w_leaf, eps, threshold_scale)
        ctx.save_for_backward(
            x, w_router, b_router, w_leaf,
            b_leaf if b_leaf is not None else x.new_zeros(0),
        )
        ctx.wq_router = wq_router
        ctx.wq_leaf = wq_leaf
        ctx.depth = int(depth)
        ctx.activation_bits = activation_bits
        ctx.has_bias = b_leaf is not None

        if x.is_cuda and has_triton:
            out = _triton_forward(
                x, wq_router, b_router, wq_leaf, b_leaf,
                ctx.depth, activation_bits,
            )
        else:
            out = fff_forward_ref(
                x, wq_router, b_router, wq_leaf, b_leaf,
                ctx.depth, activation_bits,
            )
        needs_grad = any(
            t is not None and t.requires_grad
            for t in (x, w_router, b_router, w_leaf, b_leaf)
        )
        if needs_grad and not out.requires_grad:
            out.requires_grad_(True)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w_router, b_router, w_leaf, b_leaf_or_empty = ctx.saved_tensors
        wq_router, wq_leaf = ctx.wq_router, ctx.wq_leaf
        depth, activation_bits = ctx.depth, ctx.activation_bits
        b_leaf = b_leaf_or_empty if ctx.has_bias else None

        grad_out = grad_out.float()
        xf = x.float()
        batch, k_dim = xf.shape
        d_dim = w_leaf.shape[1]
        device = xf.device

        xq = _absmax_quantize_rah(xf, activation_bits)
        leaf, r_list, node_ids, go_right_list = _route_with_probs(
            xf, wq_router, b_router, depth
        )

        # full path probability (product of r / 1-r along each row's path)
        full = torch.ones(batch, device=device)
        for i in range(depth):
            r = r_list[i]
            full = full * torch.where(go_right_list[i], r, 1.0 - r)

        # base = xq @ wq_leaf[leaf]^T + b_leaf[leaf]  (bounded-memory loop)
        base = torch.empty((batch, d_dim), dtype=xf.dtype, device=device)
        for l in range(wq_leaf.shape[0]):
            idx = (leaf == l).nonzero().squeeze(1)
            if idx.numel():
                base[idx] = torch.mm(xq[idx], wq_leaf[l].T)
                if b_leaf is not None:
                    base[idx] = base[idx] + b_leaf[l]

        # leaf gradients (STE: quantizers are identity in backward)
        gx = torch.zeros_like(xf)
        gw_leaf = torch.zeros_like(w_leaf)
        gb_leaf = torch.zeros_like(b_leaf) if b_leaf is not None else None
        for l in range(wq_leaf.shape[0]):
            idx = (leaf == l).nonzero().squeeze(1)
            if idx.numel():
                gx[idx] = torch.mm(grad_out[idx], wq_leaf[l])
                gw_leaf[l] = torch.mm(grad_out[idx].T, xq[idx])
                if b_leaf is not None:
                    gb_leaf[l] = grad_out[idx].sum(0)

        # routing gradients via the path-probability STE gate
        g_dot = (grad_out * base.detach()).sum(dim=1)  # (batch,)
        gw_router = torch.zeros_like(w_router)
        gb_router = torch.zeros_like(b_router)
        for i in range(depth):
            nids = node_ids[i]
            r = r_list[i]
            coef = torch.where(go_right_list[i], 1.0 - r, -r)
            g_logit = g_dot * full * coef
            gw_router.index_add_(0, nids, g_logit[:, None] * xf)
            gb_router.index_add_(0, nids, g_logit)
            gx = gx + g_logit[:, None] * wq_router[nids]

        return (
            gx, gw_router, gb_router, gw_leaf, gb_leaf,
            None, None, None, None,
        )


class FFFTritonLayer(nn.Module):
    """FFF layer that uses the fused Triton kernel on CUDA, torch otherwise.

    Weight shapes match :class:`~bitnet_fff.fast_fff.FastFeedForwardBitNet`
    (``router_weight (num_leaves-1, d_model)``, ``leaf_weight
    (num_leaves, d_model, d_model)``) so checkpoints are interchangeable.
    """

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 5,
        bias: bool = True,
        activation_bits: int = 8,
        eps: float = 1e-8,
        threshold_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        self.d_model = d_model
        self.depth = depth
        self.num_leaves = 1 << depth
        self.activation_bits = activation_bits
        self.eps = eps
        self.threshold_scale = threshold_scale

        self.router_weight = nn.Parameter(
            torch.empty(self.num_leaves - 1, d_model)
        )
        self.router_bias = nn.Parameter(torch.empty(self.num_leaves - 1))
        self.leaf_weight = nn.Parameter(
            torch.empty(self.num_leaves, d_model, d_model)
        )
        if bias:
            self.leaf_bias = nn.Parameter(torch.empty(self.num_leaves, d_model))
        else:
            self.register_parameter("leaf_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.leaf_weight, std=0.02)
        nn.init.normal_(self.router_weight, std=0.02)
        nn.init.zeros_(self.router_bias)
        if self.leaf_bias is not None:
            nn.init.zeros_(self.leaf_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x if x.dim() <= 2 else x.reshape(-1, x.shape[-1])
        out = FFFTriton.apply(
            flat, self.router_weight, self.router_bias, self.leaf_weight,
            self.leaf_bias, self.depth, self.activation_bits,
            self.eps, self.threshold_scale,
        )
        if x.dim() <= 2:
            return out
        return out.reshape(*x.shape[:-1], self.d_model)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, depth={self.depth}, "
            f"num_leaves={self.num_leaves}, "
            f"activation_bits={self.activation_bits}, "
            f"bias={self.leaf_bias is not None}"
        )
