"""Cross-platform device helpers for CUDA / MPS / CPU.

Used by training, inference, chat, and ``benchmark.py``.

Resolution order for ``auto``: CUDA (Linux NVIDIA) → Apple MPS → CPU.
Requesting ``cuda`` or ``mps`` when that backend is missing logs a warning
and falls back instead of raising.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch
from torch import Tensor


def mps_is_available() -> bool:
    """True when Apple Metal Performance Shaders can run tensors."""
    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:  # noqa: BLE001 — some CPU-only wheels omit MPS fully
        return False


def get_device() -> torch.device:
    """Auto-detect the best available accelerator.

    Priority: CUDA → Apple MPS → CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warn_device_fallback(requested: str, chosen: torch.device) -> None:
    """Print a visible CLI warning when the requested accelerator is missing."""
    extra = " (Apple Silicon MPS)" if chosen.type == "mps" else ""
    msg = (
        f"{requested.upper()} requested but is not available on this machine; "
        f"falling back to {chosen.type}{extra}."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
    print(f"warning: {msg}", flush=True)


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Resolve a CLI device string (``auto`` / ``cuda`` / ``mps`` / ``cpu``).

    ``auto``
        CUDA if present, else MPS on Apple Silicon, else CPU.
    ``cuda``
        Use CUDA when available; otherwise warn and fall back to MPS or CPU.
        Does **not** raise ``RuntimeError`` on macOS / CPU-only machines.
    ``mps``
        Use MPS when available; otherwise warn and fall back to CUDA or CPU.
    ``cpu``
        Always CPU.
    """
    req = (requested or "auto").lower().strip()
    if req in {"", "auto"}:
        return get_device()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        chosen = torch.device("mps") if mps_is_available() else torch.device("cpu")
        _warn_device_fallback("cuda", chosen)
        return chosen
    if req == "mps":
        if mps_is_available():
            return torch.device("mps")
        chosen = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _warn_device_fallback("mps", chosen)
        return chosen
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}; use auto|cuda|mps|cpu")


def device_label(device: torch.device | None = None) -> str:
    """Human-readable environment label for logs / banners."""
    if device is None:
        device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return f"CUDA GPU — {name} (sm_{major}{minor})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU Fallback"


def print_device_info(device: torch.device | None = None, *, prefix: str = "") -> torch.device:
    """Print clear startup environment info; return the resolved device."""
    if device is None:
        device = get_device()
    label = device_label(device)
    head = f"{prefix}Selected device: {device}"
    print(head, flush=True)
    print(f"{prefix}Environment: {label}", flush=True)
    if device.type == "cuda":
        print(f"{prefix}cuDNN benchmark: {torch.backends.cudnn.benchmark}", flush=True)
    return device


def apply_hardware_optimizations(device: torch.device) -> None:
    """Enable backend knobs appropriate for ``device`` (safe no-ops otherwise)."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def pin_memory_for(device: torch.device) -> bool:
    """Whether DataLoader should pin host memory.

    Pinned pages help CUDA H2D copies. MPS and CPU warn if ``pin_memory=True``.
    """
    return device.type == "cuda"


def amp_enabled(device: torch.device) -> bool:
    """True when mixed precision should be attempted on this device."""
    return device.type in {"cuda", "mps"}


def mps_autocast_supported() -> bool:
    """Return True if ``torch.amp.autocast('mps')`` is usable on this build."""
    if not mps_is_available():
        return False
    try:
        with torch.amp.autocast("mps"):
            _ = torch.zeros(1, device="mps") * 2
        return True
    except Exception:
        return False


@contextmanager
def amp_autocast(device: torch.device) -> Iterator[None]:
    """Context manager for CUDA / MPS autocast with graceful MPS fallback.

    On CUDA, forces ``dtype=torch.float16`` so large LM logits stay in fp16
    (lower peak VRAM than the default which may promote to bf16/fp32).
    """
    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return
    if device.type == "mps":
        if mps_autocast_supported():
            try:
                with torch.amp.autocast(device_type="mps"):
                    yield
                return
            except Exception:
                pass  # fall through to nullcontext
        with nullcontext():
            yield
        return
    with nullcontext():
        yield


def make_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    """Create a GradScaler — enabled for CUDA; disabled elsewhere (identity)."""
    # GradScaler is a CUDA training aid; MPS/CPU use enabled=False.
    return torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


def to_device(
    tensor: Tensor,
    device: torch.device,
    *,
    non_blocking: bool | None = None,
) -> Tensor:
    """Move a tensor to ``device`` with sensible ``non_blocking`` defaults."""
    if non_blocking is None:
        non_blocking = pin_memory_for(device)
    return tensor.to(device, non_blocking=non_blocking)
