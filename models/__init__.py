"""Model package for the Fast Feedforward language model."""

from __future__ import annotations

from models.fff_layer import (
    FFFTree,
    FastFeedforwardLinear,
    MultiTreeFFFLayer,
    SwiGLULeafExpert,
    HYPERZENS_35B_HIDDEN_SIZE,
    HYPERZENS_35B_NUM_TREES,
    HYPERZENS_35B_TREE_DEPTH,
    HYPERZENS_35B_INTERMEDIATE_SIZE,
    cmm_hparams_from_checkpoint,
    is_cmm_cpp_available,
    is_fff_cpp_available,
)
from models.fff_hard_triton import is_triton_available
from models.transformer import FFFConfig, FFFTransformer, StandardTransformer

__all__ = [
    "FastFeedforwardLinear",
    "FFFTree",
    "MultiTreeFFFLayer",
    "SwiGLULeafExpert",
    "HYPERZENS_35B_HIDDEN_SIZE",
    "HYPERZENS_35B_NUM_TREES",
    "HYPERZENS_35B_TREE_DEPTH",
    "HYPERZENS_35B_INTERMEDIATE_SIZE",
    "FFFConfig",
    "FFFTransformer",
    "StandardTransformer",
    "is_fff_cpp_available",
    "is_cmm_cpp_available",
    "is_triton_available",
    "cmm_hparams_from_checkpoint",
]
