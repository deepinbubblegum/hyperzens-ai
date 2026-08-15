"""Apple Silicon (MPS) memory and tensor helpers for unified-memory budgets."""

from __future__ import annotations

import torch

__all__ = [
    "is_mps_available",
    "mps_current_allocated_bytes",
    "mps_driver_allocated_bytes",
    "mps_synchronize",
    "mps_empty_cache",
    "to_device_contiguous",
    "tensor_bytes",
]


def is_mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def mps_current_allocated_bytes() -> int:
    if is_mps_available():
        try:
            return int(torch.mps.current_allocated_memory())
        except Exception:
            return 0
    return 0


def mps_driver_allocated_bytes() -> int:
    if is_mps_available():
        try:
            return int(torch.mps.driver_allocated_memory())
        except Exception:
            return 0
    return 0


def mps_synchronize() -> None:
    if is_mps_available():
        torch.mps.synchronize()


def mps_empty_cache() -> None:
    if is_mps_available():
        torch.mps.empty_cache()


def to_device_contiguous(
    x: torch.Tensor, device: torch.device | str | None = None
) -> torch.Tensor:
    """Move ``x`` to ``device`` and enforce a contiguous layout without copies."""
    if device is not None and x.device != torch.device(device):
        x = x.to(device)
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def tensor_bytes(x: torch.Tensor) -> int:
    return int(x.numel()) * x.element_size()
