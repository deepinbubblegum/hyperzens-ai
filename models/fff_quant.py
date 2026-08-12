"""INT8 / INT4 quantization helpers for FFF leaf affine weights.

Leaf tensors are shaped ``(L, D_in, D_out)``. Routers stay in FP16/FP32; only
leaf weight matrices are quantized for VRAM compression. Scales are stored as
``float16`` per leaf.

INT8
----
Symmetric per-leaf: ``scale = max|W| / 127``, ``q = round(W / scale) ∈ [-128, 127]``.

INT4
----
Symmetric per-leaf: ``scale = max|W| / 7``, ``q ∈ [-8, 7]``, then pack two
signed nibbles into one ``uint8`` (low nibble = even flat index, high = odd).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

QuantMode = Literal["int8", "int4"]


@dataclass
class QuantizedLeafWeights:
    """Packed leaf weights + per-leaf FP16 scales for on-the-fly dequant.

    Attributes
    ----------
    mode:
        ``\"int8\"`` or ``\"int4\"``.
    weight_q:
        INT8: ``(L, D_in, D_out)`` ``torch.int8``.
        INT4: ``(L, ceil(D_in·D_out / 2))`` ``torch.uint8`` packed.
    scale:
        ``(L,)`` ``float16`` — multiply after casting q → float.
    d_in, d_out:
        Logical leaf matrix shape (needed to unpack INT4).
    flat_numel:
        ``D_in * D_out`` (before INT4 padding).
    """

    mode: QuantMode
    weight_q: Tensor
    scale: Tensor
    d_in: int
    d_out: int
    flat_numel: int

    @property
    def num_leaves(self) -> int:
        return int(self.weight_q.shape[0])

    def nbytes(self) -> int:
        """Storage bytes for quantized weights + scales (excludes routers)."""
        return int(self.weight_q.numel() * self.weight_q.element_size()) + int(
            self.scale.numel() * self.scale.element_size()
        )


def quantize_fff_leaf_int8(w_leaf: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-leaf INT8 quantization of ``w_leaf`` ``(L, D_in, D_out)``.

    Returns
    -------
    w_int8:
        ``(L, D_in, D_out)`` ``torch.int8``.
    scale:
        ``(L,)`` ``torch.float16`` with ``W ≈ q.float() * scale``.
    """
    if w_leaf.ndim != 3:
        raise ValueError(f"w_leaf must be (L, D_in, D_out), got {tuple(w_leaf.shape)}")
    # scale = max|W| / 127 per leaf (broadcast over D_in, D_out).
    amax = w_leaf.detach().float().abs().amax(dim=(-2, -1))
    scale = (amax / 127.0).clamp(min=1e-8).to(dtype=torch.float16)
    scale_f = scale.to(dtype=torch.float32).view(-1, 1, 1)
    q = torch.round(w_leaf.detach().float() / scale_f).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_fff_leaf_int4(w_leaf: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-leaf INT4 quantization with uint8 nibble packing.

    Returns
    -------
    w_packed_uint8:
        ``(L, ceil(D_in·D_out / 2))`` ``torch.uint8``.
    scale:
        ``(L,)`` ``torch.float16`` with ``W ≈ q.float() * scale``, ``q ∈ [-8, 7]``.
    """
    if w_leaf.ndim != 3:
        raise ValueError(f"w_leaf must be (L, D_in, D_out), got {tuple(w_leaf.shape)}")
    l, d_in, d_out = w_leaf.shape
    amax = w_leaf.detach().float().abs().amax(dim=(-2, -1))
    scale = (amax / 7.0).clamp(min=1e-8).to(dtype=torch.float16)
    scale_f = scale.to(dtype=torch.float32).view(-1, 1, 1)
    q = torch.round(w_leaf.detach().float() / scale_f).clamp(-8, 7).to(torch.int8)

    flat = q.reshape(l, d_in * d_out)
    n = flat.shape[1]
    if n % 2 == 1:
        flat = torch.nn.functional.pad(flat, (0, 1), value=0)
    # Pack: low nibble = even index, high nibble = odd (signed → 4-bit two's complement).
    even = flat[:, 0::2].to(torch.int16) & 0x0F
    odd = flat[:, 1::2].to(torch.int16) & 0x0F
    packed = (even | (odd << 4)).to(torch.uint8).contiguous()
    return packed, scale.contiguous()


def _sign_extend_nibble(nibble: Tensor) -> Tensor:
    """Map unsigned 4-bit values in ``[0, 15]`` to signed ``[-8, 7]``."""
    n = nibble.to(torch.int16)
    return torch.where(n >= 8, n - 16, n).to(torch.int8)


def dequantize_fff_leaf_int8(w_int8: Tensor, scale: Tensor) -> Tensor:
    """Reference dequant INT8 → FP16 (CPU / tests only — not used on CUDA hot path).

    CUDA inference must keep ``w_int8`` packed and dequantize inside Triton so
    a full FP16 leaf tensor is never allocated in HBM.
    """
    s = scale.to(dtype=torch.float16).view(-1, 1, 1)
    return (w_int8.to(dtype=torch.float16) * s).contiguous()


def dequantize_fff_leaf_int4(
    w_packed: Tensor,
    scale: Tensor,
    d_in: int,
    d_out: int,
) -> Tensor:
    """Reference unpack INT4 → FP16 (CPU / tests only — not used on CUDA hot path)."""
    l = w_packed.shape[0]
    flat_n = d_in * d_out
    even = _sign_extend_nibble(w_packed.to(torch.int16) & 0x0F)
    odd = _sign_extend_nibble((w_packed.to(torch.int16) >> 4) & 0x0F)
    flat = torch.empty(l, even.shape[1] * 2, device=w_packed.device, dtype=torch.int8)
    flat[:, 0::2] = even
    flat[:, 1::2] = odd
    flat = flat[:, :flat_n]
    s = scale.to(dtype=torch.float16).view(-1, 1, 1)
    return (flat.to(dtype=torch.float16).view(l, d_in, d_out) * s).contiguous()


def quantize_leaf_weights(w_leaf: Tensor, mode: QuantMode) -> QuantizedLeafWeights:
    """Quantize ``(L, D_in, D_out)`` leaf weights to INT8 or packed INT4.

    Output tensors stay on ``w_leaf.device`` (no host round-trip).
    """
    d_in = int(w_leaf.shape[1])
    d_out = int(w_leaf.shape[2])
    flat_numel = d_in * d_out
    if mode == "int8":
        w_q, scale = quantize_fff_leaf_int8(w_leaf)
    elif mode == "int4":
        w_q, scale = quantize_fff_leaf_int4(w_leaf)
    else:
        raise ValueError(f"mode must be 'int8' or 'int4', got {mode!r}")
    return QuantizedLeafWeights(
        mode=mode,
        weight_q=w_q,
        scale=scale,
        d_in=d_in,
        d_out=d_out,
        flat_numel=flat_numel,
    )


def dequantize_leaf_weights(qstate: QuantizedLeafWeights) -> Tensor:
    """Full FP16 reconstruction (reference / CPU fallback only)."""
    if qstate.mode == "int8":
        return dequantize_fff_leaf_int8(qstate.weight_q, qstate.scale)
    return dequantize_fff_leaf_int4(
        qstate.weight_q, qstate.scale, qstate.d_in, qstate.d_out
    )


def estimate_leaf_vram_mb(w_leaf_fp: Tensor, mode: QuantMode | Literal["fp16"]) -> float:
    """Estimate leaf-weight storage (MB) for FP16 vs INT8 vs INT4 packing."""
    l, d_in, d_out = w_leaf_fp.shape
    if mode == "fp16":
        nbytes = l * d_in * d_out * 2
    elif mode == "int8":
        nbytes = l * d_in * d_out * 1 + l * 2  # int8 + fp16 scales
    elif mode == "int4":
        packed = (d_in * d_out + 1) // 2
        nbytes = l * packed * 1 + l * 2
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return float(nbytes) / (1024.0 * 1024.0)


@torch.no_grad()
def quantize_fff_model_leaves(
    model: torch.nn.Module,
    mode: QuantMode,
) -> list[QuantizedLeafWeights]:
    """Quantize every FFF layer's ``leaf_weights``; returns parallel list of states."""
    if not hasattr(model, "fff_layers"):
        raise TypeError("model must expose fff_layers()")
    states: list[QuantizedLeafWeights] = []
    for layer in model.fff_layers():
        states.append(quantize_leaf_weights(layer.leaf_weights.detach(), mode))
    return states


def total_quant_leaf_vram_mb(states: list[QuantizedLeafWeights]) -> float:
    """Sum quantized leaf storage across layers (MB)."""
    return sum(s.nbytes() for s in states) / (1024.0 * 1024.0)


def attach_leaf_qstates(
    model: torch.nn.Module,
    states: list[QuantizedLeafWeights],
) -> None:
    """Attach ``leaf_qstate`` on each FFF layer without redundant device copies.

    If ``weight_q`` / ``scale`` are already on the layer device, they are reused
    in-place (no FP16 expand, no extra ``.to()`` allocation).
    """
    layers = list(model.fff_layers())  # type: ignore[attr-defined]
    if len(layers) != len(states):
        raise ValueError(
            f"layer/state count mismatch: {len(layers)} vs {len(states)}"
        )
    for layer, st in zip(layers, states):
        device = layer.leaf_weights.device
        w_q = st.weight_q
        scale = st.scale
        if w_q.device != device:
            w_q = w_q.to(device=device, non_blocking=True)
        if scale.device != device:
            scale = scale.to(device=device, non_blocking=True)
        if scale.dtype != torch.float16:
            scale = scale.to(dtype=torch.float16)
        if not w_q.is_contiguous():
            w_q = w_q.contiguous()
        if not scale.is_contiguous():
            scale = scale.contiguous()
        layer.leaf_qstate = QuantizedLeafWeights(
            mode=st.mode,
            weight_q=w_q,
            scale=scale,
            d_in=st.d_in,
            d_out=st.d_out,
            flat_numel=st.flat_numel,
        )


def clear_leaf_qstates(model: torch.nn.Module) -> None:
    """Remove ``leaf_qstate`` attributes from FFF layers."""
    if not hasattr(model, "fff_layers"):
        return
    for layer in model.fff_layers():
        if hasattr(layer, "leaf_qstate"):
            delattr(layer, "leaf_qstate")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()