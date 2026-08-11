"""Model package for the Fast Feedforward language model."""

from __future__ import annotations

from models.fff_layer import FastFeedforwardLinear
from models.transformer import FFFConfig, FFFTransformer, StandardTransformer

__all__ = [
    "FastFeedforwardLinear",
    "FFFConfig",
    "FFFTransformer",
    "StandardTransformer",
]
