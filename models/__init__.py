"""Model package for the Fast Feedforward language model."""

from __future__ import annotations

from models.fff_layer import FastFeedforwardLinear, is_fff_cpp_available, is_cmm_cpp_available, cmm_hparams_from_checkpoint
from models.fff_hard_triton import is_triton_available
from models.transformer import FFFConfig, FFFTransformer, StandardTransformer

__all__ = [
    "FastFeedforwardLinear",
    "FFFConfig",
    "FFFTransformer",
    "StandardTransformer",
    "is_fff_cpp_available",
    "is_cmm_cpp_available",
    "is_triton_available",
    "cmm_hparams_from_checkpoint",
]
