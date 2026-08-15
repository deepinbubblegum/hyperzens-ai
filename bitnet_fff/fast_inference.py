"""Packed-ternary fast-inference path for :class:`FastFeedForwardBitNet`.

BitNet b1.58 leaves are quantized to {-1, 0, +1} and bit-packed two bits per
weight (4 values per ``uint8``). The grouped matmul ``out[b] = W[leaf[b]] @ x[b]``
is executed by a native extension:

* CPU: ARM64 NEON kernel (pure vector add/sub, no FP multiplies) with token
  bucketing by leaf.
* MPS (Apple Silicon): a Metal compute kernel with one threadgroup per batch
  row, cooperative shared-memory load of the input row, and branchless
  select-add accumulation. Only the routed leaf rows are read - no per-branch
  buffers are ever materialized, so peak unified memory drops from the
  ``O(B*d_out*d_in)`` gather to ``O(B*(d_in + d_out))``.

Packing (2-bit encoding, lane ``k`` of byte ``j`` holds column ``4j+k``):
    ``0b00 -> 0``, ``0b01 -> +1``, ``0b10 -> -1``, ``0b11`` unused.

``d_in`` must be a multiple of 4 for both kernels; padding is handled here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .bitlinear import absmax_quantize, absmean_ternarize

__all__ = [
    "extension_available",
    "pack_ternary_weights",
    "unpack_ternary_weights",
    "ternary_mm",
    "PackedTernaryMM",
    "PackedFFFEvaluator",
]

_EXT_AVAILABLE: bool | None = None


def extension_available() -> bool:
    """True once the packed-ternary native extension is loadable (lazy build)."""
    global _EXT_AVAILABLE
    if _EXT_AVAILABLE is None:
        try:
            from csrc.build_ext import get_extension

            get_extension()
            _EXT_AVAILABLE = True
        except Exception:
            _EXT_AVAILABLE = False
    return _EXT_AVAILABLE


def pack_ternary_weights(weight: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, int]:
    """AbsMean-ternarize ``(L, d_out, d_in)`` weights and bit-pack to uint8.

    Returns ``(packed, d_in_padded)`` with ``packed`` of shape
    ``(L, d_out, ceil(d_in/4))``. ``d_in_padded`` is the input width the
    kernels expect (``d_in`` rounded up to a multiple of 4).
    """
    t = absmean_ternarize(weight.detach(), eps=eps)
    d_in = t.shape[-1]
    pad = (-d_in) % 4
    if pad:
        t = F.pad(t, (0, pad))
    L, d_out, K = t.shape[0], t.shape[1], (d_in + pad) // 4
    e = torch.where(
        t < 0,
        torch.full_like(t, 2.0),
        torch.where(t > 0, torch.full_like(t, 1.0), torch.zeros_like(t)),
    )
    E = e.to(torch.uint8).view(L, d_out, K, 4)
    packed = (E[..., 0] | (E[..., 1] << 2) | (E[..., 2] << 4) | (E[..., 3] << 6)).to(
        torch.uint8
    )
    return packed.contiguous(), d_in + pad


def unpack_ternary_weights(packed: torch.Tensor) -> torch.Tensor:
    """Decode ``(L, d_out, K)`` uint8 into a float ``(L, d_out, K*4)`` tensor in {-1, 0, +1}."""
    s = torch.stack([(packed >> (2 * i)) & 3 for i in range(4)], dim=-1)
    s = s.to(torch.float32)
    return torch.where(s == 1, 1.0, torch.where(s == 2, -1.0, 0.0)).flatten(-2, -1)


def ternary_mm(
    x: torch.Tensor, packed: torch.Tensor, leaf_idx: torch.Tensor
) -> torch.Tensor:
    """Run the packed-ternary grouped matmul.

    ``x`` is ``(B, d_in)`` float32, ``packed`` is ``(L, d_out, d_in/4)`` uint8,
    ``leaf_idx`` is ``(B,)`` integer (int32 required on MPS, int64/int32 on CPU).
    All tensors must already live on the target device; dtype is normalized here.
    """
    if not extension_available():
        raise RuntimeError("packed-ternary native extension is not available")
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    packed = packed.to(torch.uint8).contiguous()
    if x.device.type == "mps":
        leaf = leaf_idx.to(torch.int32).contiguous()
        return torch.ops.ternary_packed.ternary_mm(x.contiguous(), packed, leaf)
    leaf = leaf_idx.to(torch.int64).contiguous()
    return torch.ops.ternary_packed.ternary_mm(x.contiguous(), packed, leaf)


class PackedTernaryMM(torch.autograd.Function):
    """Autograd wrapper over the packed op (STE backward, for fine-tuning).

    Forward ternarizes + packs ``weight`` (the raw ``(L, d_out, d_in)``
    parameter) and runs the grouped ternary matmul. Backward flows gradients
    through the ternary quantization unchanged (STE): ``grad_x`` via ``bmm``
    with the decoded ternary weights, ``grad_w`` via ``bmm`` + ``index_add_``
    scattered back into the full parameter buffer.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        leaf_idx: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        packed, d_in_padded = pack_ternary_weights(weight.detach(), eps=eps)
        packed = packed.to(x.device)
        pad = d_in_padded - x.shape[-1]
        xp = F.pad(x, (0, pad)) if pad else x.contiguous()
        out = ternary_mm(xp.contiguous(), packed, leaf_idx)
        ctx.save_for_backward(xp, packed, leaf_idx)
        ctx.d_in_orig = weight.shape[-1]
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xp, packed, leaf_idx = ctx.saved_tensors
        wf = unpack_ternary_weights(packed)
        idx = leaf_idx.to(torch.long)
        w_sel = wf[idx]
        grad_x = torch.bmm(grad_out.unsqueeze(1), w_sel).squeeze(1)
        if grad_x.shape[-1] != ctx.d_in_orig:
            grad_x = grad_x[..., : ctx.d_in_orig]
        dw = torch.bmm(grad_out.unsqueeze(1).transpose(1, 2), xp.unsqueeze(1)).squeeze(
            1
        )
        grad_w = torch.zeros_like(wf)
        grad_w.index_add_(0, idx, dw)
        grad_w = grad_w[..., : ctx.d_in_orig]
        return grad_x, grad_w, None, None


class PackedFFFEvaluator:
    """Zero-gather, buffer-recycled inference path for a ``FastFeedForwardBitNet``.

    Packed ternary weights + leaf bias are materialized once on ``device``.
    Each ``forward(x, leaf_idx)`` runs routing externally and then evaluates only
    the selected leaf rows through the native kernel. ``chunk_size`` bounds the
    peak footprint by splitting the batch. A small per-batch buffer pool reuses
    the padded-input scratch across calls.
    """

    def __init__(
        self,
        module,
        device: torch.device | str | None = None,
        chunk_size: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        module.eval()
        self.module = module
        self.d_in = module.d_in
        self.d_out = module.d_out
        self.num_leaves = module.num_leaves
        self.activation_bits = module.activation_bits
        self.eps = eps
        self.chunk_size = chunk_size
        self.device = (
            torch.device(device)
            if device is not None
            else next(module.parameters()).device
        )
        with torch.no_grad():
            self.packed, self.d_in_padded = pack_ternary_weights(
                module.leaf_weight.data, eps=eps
            )
            self.packed = self.packed.to(self.device)
            if module.leaf_bias is not None:
                self.bias = module.leaf_bias.detach().to(self.device)
            else:
                self.bias = None
        self._pool: dict[int, torch.Tensor] = {}

    def _padded(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, torch.float32)
        if self.d_in_padded == self.d_in and x.is_contiguous():
            return x
        x = x.contiguous()
        batch = x.shape[0]
        buf = self._pool.get(batch)
        if buf is None or buf.shape[1] != self.d_in_padded:
            buf = torch.empty(
                batch, self.d_in_padded, device=self.device, dtype=torch.float32
            )
            self._pool[batch] = buf
        buf[:, : x.shape[1]].copy_(x)
        buf[:, x.shape[1] :].zero_()
        return buf

    def _leaf(self, leaf_idx: torch.Tensor) -> torch.Tensor:
        target = torch.int32 if self.device.type == "mps" else torch.int64
        return leaf_idx.to(target).contiguous()

    def forward(self, x: torch.Tensor, leaf_idx: torch.Tensor) -> torch.Tensor:
        xp = self._padded(x)
        if self.activation_bits <= 32:
            xp = absmax_quantize(xp, bits=self.activation_bits, eps=self.eps)
        out = ternary_mm(xp, self.packed, self._leaf(leaf_idx))
        if self.bias is not None:
            out = out + self.bias[leaf_idx.to(torch.long)]
        return out

    def forward_chunked(self, x: torch.Tensor, leaf_idx: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        chunk = self.chunk_size or batch
        pieces = [
            self.forward(x[s : s + chunk], leaf_idx[s : s + chunk])
            for s in range(0, batch, chunk)
        ]
        return torch.cat(pieces, dim=0)

    def __call__(self, x: torch.Tensor, leaf_idx: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return self.forward_chunked(x, leaf_idx)
        return self.forward(x, leaf_idx)
