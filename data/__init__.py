"""Data package: BPE WikiText loaders and offline distill caches.

Import from ``data.dataset_loader`` directly to avoid ``python -m`` import cycles, e.g.::

    from data.dataset_loader import BPEDataset, OfflineDistillDataset, get_wikitext_data
"""

from __future__ import annotations

__all__ = [
    "BPEDataset",
    "GPT2_VOCAB_SIZE",
    "LOGITS_CACHE_DIR",
    "OfflineDistillDataset",
    "build_wikitext_datasets",
    "decode_tokens",
    "encode_text",
    "get_gpt2_encoding",
    "get_wikitext_data",
    "logits_cache_path",
    "save_teacher_logits_cache",
]


def __getattr__(name: str):
    if name in __all__:
        from data import dataset_loader as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
