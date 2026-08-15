"""Hybrid Fast Feedforward (FFF) + BitNet b1.58 layers for Apple Silicon (MPS)."""

from __future__ import annotations

from .bitlinear import BitLinear, absmax_quantize, absmean_ternarize, ste_ternarize
from .fast_fff import FastFeedForwardBitNet
from . import mps_utils
from . import fast_inference

__all__ = [
    "BitLinear",
    "FastFeedForwardBitNet",
    "absmean_ternarize",
    "ste_ternarize",
    "absmax_quantize",
    "mps_utils",
    "fast_inference",
]

__version__ = "0.1.0"
