"""BitNet b1.58 linear projection with straight-through estimators (STE).

References:
    Wang et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
    Ma et al., "The Era of 1-bit LLMs: Training Tips on 1.58-bit LLMs"

Quantization scheme
    - Weights:  AbsMean ternary quantization -> {-1, 0, +1}
    - Activations: AbsMax k-bit quantization (default 8-bit)
    - Gradients: straight-through estimator (STE) through both quantizers so that
      the upstream gradient passes through unchanged while the forward pass runs
      on low-precision tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "absmean_ternarize",
    "ste_ternarize",
    "absmax_quantize",
    "BitLinear",
]


def absmean_ternarize(
    w: torch.Tensor,
    eps: float = 1e-8,
    threshold_scale: float = 1.0,
    threshold: float | None = None,
) -> torch.Tensor:
    """AbsMean ternary quantization returning values in {-1, 0, +1}.

    Weights with magnitude above the threshold map to ``sign(w)``, everything
    else to zero. The default threshold is the AbsMean magnitude scaled by
    ``threshold_scale`` (1.0 -> the classic BitNet b1.58 threshold); a fixed
    absolute ``threshold`` overrides both. ``threshold_scale`` is the knob the
    auto-tuner sweeps against tree depth.
    """
    if threshold is None:
        threshold = float(w.detach().abs().mean().clamp_min(eps)) * threshold_scale
    return torch.where(w.abs() > threshold, torch.sign(w), torch.zeros_like(w))


def ste_ternarize(
    w: torch.Tensor,
    eps: float = 1e-8,
    threshold_scale: float = 1.0,
    threshold: float | None = None,
) -> torch.Tensor:
    """Ternary weights with a straight-through estimator.

    Forward pass uses the ternary weight; backward pass treats the
    quantization as the identity so the gradient reaches ``w`` unchanged.
    """
    wq = absmean_ternarize(w, eps=eps, threshold_scale=threshold_scale, threshold=threshold)
    return wq + (w - w.detach())


def absmax_quantize(x: torch.Tensor, bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """AbsMax k-bit activation quantization with STE.

    Per-row scaling to ``[-2**(bits-1), 2**(bits-1)-1]``; the forward pass
    runs on the rounded values, the backward pass is the identity.
    """
    if not 1 < bits <= 32:
        raise ValueError(f"bits must be in (1, 32], got {bits}")
    max_q = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    q = (x / scale) * max_q
    q = q.round().clamp(-(max_q + 1), max_q)
    xq = (q / max_q) * scale
    return xq + (x - x.detach())


class BitLinear(nn.Module):
    """1.58-bit ternary linear layer (BitNet b1.58) with STE quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_bits: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weight(self) -> torch.Tensor:
        return ste_ternarize(self.weight, eps=self.eps)

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        return absmax_quantize(x, bits=self.activation_bits, eps=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        xq = self.quantize_activation(x)
        wq = self.quantize_weight()
        return F.linear(xq, wq, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation_bits={self.activation_bits}"
        )
