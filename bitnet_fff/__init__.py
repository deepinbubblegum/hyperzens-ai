"""Hybrid Fast Feedforward (FFF) + BitNet b1.58 layers for Apple Silicon (MPS)."""

from __future__ import annotations

from .bitlinear import BitLinear, absmax_quantize, absmean_ternarize, ste_ternarize
from .fast_fff import FastFeedForwardBitNet
from .uff import BitNetUFFLayer
from . import mps_utils
from . import fast_inference
from . import models
from . import qat
from . import profiler
from . import tokenizer
from . import tuning
from . import dataset
from . import triton_fff
from . import triton_ternary_mm

__all__ = [
    "BitLinear",
    "FastFeedForwardBitNet",
    "BitNetUFFLayer",
    "absmean_ternarize",
    "ste_ternarize",
    "absmax_quantize",
    "mps_utils",
    "fast_inference",
    "models",
    "qat",
    "profiler",
    "tokenizer",
    "tuning",
    "dataset",
    "triton_fff",
    "triton_ternary_mm",
]

__version__ = "0.1.0"
