"""Cross-platform compiler / linker flags for FFF native extensions.

Supported hosts
---------------
* **macOS (Darwin / Apple Silicon)** — CPU-only local dev & benchmarks.
  Flags: ``-O3 -std=c++17`` (+ optional Homebrew libomp). **No CUDA.**
* **Linux x86_64** — full execution host (RTX 3060).
  Flags: ``-O3 -march=native -fopenmp -funroll-loops -std=c++17``.
  CUDA (when available): NVCC ``sm_86`` / ``TORCH_CUDA_ARCH_LIST=8.6``.

Shared by ``setup.py`` and the JIT loader in ``models/fff_layer.py``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# RTX 3060 (Ampere GA106) — Linux CUDA builds only.
TARGET_CUDA_ARCH: str = "8.6"  # sm_86
TARGET_CUDA_GENCODE: str = "arch=compute_86,code=sm_86"


# ---------------------------------------------------------------------------
# Host / CUDA detection
# ---------------------------------------------------------------------------


def host_system() -> str:
    """Normalized OS name: ``Darwin`` | ``Linux`` | ``Windows`` | …"""
    return platform.system()


def is_macos() -> bool:
    return host_system() == "Darwin" or sys.platform == "darwin"


def is_linux() -> bool:
    return host_system() == "Linux" or sys.platform.startswith("linux")


def is_windows() -> bool:
    return host_system() == "Windows" or sys.platform == "win32"


def cuda_is_available() -> bool:
    """True when this process can see a usable CUDA device via PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch missing / broken CUDA build
        return False


def should_build_cuda() -> bool:
    """CUDA extensions are Linux-only and require a working CUDA toolkit + GPU."""
    if not is_linux():
        return False
    if os.environ.get("FFF_FORCE_CPU_ONLY", "").strip() in {"1", "true", "TRUE", "yes"}:
        return False
    return cuda_is_available()


def apply_cuda_arch_env() -> str | None:
    """Pin RTX 3060 arch on Linux CUDA builds; no-op on macOS."""
    if not is_linux():
        return None
    return os.environ.setdefault("TORCH_CUDA_ARCH_LIST", TARGET_CUDA_ARCH)


def _apple_clang() -> bool:
    if not is_macos():
        return False
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    if not cxx:
        return True
    try:
        out = subprocess.check_output([cxx, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return True
    return "Apple clang" in out or "Apple LLVM" in out


def _homebrew_libomp_paths() -> tuple[list[str], list[str]]:
    candidates = [
        Path("/opt/homebrew/opt/libomp"),
        Path("/usr/local/opt/libomp"),
    ]
    brew = shutil.which("brew")
    if brew:
        try:
            prefix = subprocess.check_output(
                [brew, "--prefix", "libomp"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            if prefix:
                candidates.insert(0, Path(prefix))
        except (OSError, subprocess.CalledProcessError):
            pass
    for root in candidates:
        inc, lib = root / "include", root / "lib"
        if inc.is_dir() and lib.is_dir():
            return [str(inc)], [str(lib)]
    return [], []


def macos_openmp_supported() -> bool:
    """True if Homebrew ``libomp`` is present (Apple Clang needs it for -fopenmp)."""
    if not is_macos():
        return False
    inc, lib = _homebrew_libomp_paths()
    return bool(inc and lib)


# ---------------------------------------------------------------------------
# Per-platform flag sets
# ---------------------------------------------------------------------------


def _macos_cxx_flags(*, openmp: bool) -> list[str]:
    """CPU-only Apple Silicon local-dev flags."""
    # Rely on at::parallel_for for threading; OpenMP is optional.
    flags = ["-O3", "-std=c++17"]
    if openmp:
        # Apple Clang: -Xpreprocessor -fopenmp (+ Homebrew headers at compile).
        flags.extend(["-Xpreprocessor", "-fopenmp"])
    return flags


def _linux_cxx_flags(*, openmp: bool) -> list[str]:
    """Linux x86_64 (RTX 3060 host) GCC/Clang flags."""
    flags = [
        "-O3",
        "-std=c++17",
        "-march=native",
        "-mtune=native",
        "-funroll-loops",
        "-fPIC",
    ]
    if openmp:
        flags.append("-fopenmp")
    return flags


def _windows_cxx_flags(*, openmp: bool) -> list[str]:
    flags = ["/O2", "/std:c++17", "/arch:AVX2"]
    if openmp:
        flags.append("/openmp")
    return flags


def _linux_nvcc_flags() -> list[str]:
    return [
        "-O3",
        "--use_fast_math",
        f"-gencode={TARGET_CUDA_GENCODE}",
        "-std=c++17",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fff_hard_extra_compile_args(*, prefer_openmp: bool | None = None) -> dict[str, list[str]]:
    """Return ``{"cxx": [...], "nvcc": [...]}`` for CppExtension / CUDAExtension.

    * macOS: ``-O3 -std=c++17``; OpenMP only if Homebrew libomp is installed.
    * Linux: ``-O3 -march=native -fopenmp -funroll-loops -std=c++17`` + nvcc sm_86.
    """
    apply_cuda_arch_env()

    if prefer_openmp is None:
        if is_macos():
            prefer_openmp = macos_openmp_supported()
        elif is_linux():
            prefer_openmp = True
        else:
            prefer_openmp = True

    if is_macos():
        out: dict[str, list[str]] = {"cxx": _macos_cxx_flags(openmp=prefer_openmp)}
        # Never attach nvcc flags on Darwin — CUDA extensions are skipped.
        return out

    if is_windows():
        return {
            "cxx": _windows_cxx_flags(openmp=bool(prefer_openmp)),
            "nvcc": _linux_nvcc_flags(),
        }

    # Linux (default / production)
    return {
        "cxx": _linux_cxx_flags(openmp=bool(prefer_openmp)),
        "nvcc": _linux_nvcc_flags(),
    }


def fff_hard_extra_link_args(*, prefer_openmp: bool | None = None) -> list[str]:
    if prefer_openmp is None:
        if is_macos():
            prefer_openmp = False  # avoid duplicate libomp vs PyTorch on Darwin
        else:
            prefer_openmp = True

    if is_linux() and prefer_openmp:
        return ["-fopenmp"]
    # macOS / Windows: leave OpenMP to the process (PyTorch) or MSVC runtime.
    return []


def fff_hard_include_dirs() -> list[str]:
    if is_macos() and macos_openmp_supported():
        inc, _lib = _homebrew_libomp_paths()
        return inc
    return []


def fff_hard_library_dirs() -> list[str]:
    # Intentionally empty on macOS (do not link -lomp → OMP Error #15 with torch).
    return []


def fff_hard_nvcc_flags() -> list[str]:
    apply_cuda_arch_env()
    return list(_linux_nvcc_flags())


def fff_hard_jit_cflags() -> list[str]:
    args = list(fff_hard_extra_compile_args().get("cxx", []))
    for inc in fff_hard_include_dirs():
        args.append(f"-I{inc}")
    return args


def fff_hard_jit_ldflags() -> list[str]:
    return list(fff_hard_extra_link_args())


def fff_hard_jit_cflags_no_openmp() -> list[str]:
    args = list(fff_hard_extra_compile_args(prefer_openmp=False).get("cxx", []))
    return args


def discover_cuda_sources(csrc_dir: Path) -> list[str]:
    """Return ``*.cu`` paths under ``csrc/`` (empty → skip CUDAExtension)."""
    if not csrc_dir.is_dir():
        return []
    return sorted(str(p) for p in csrc_dir.glob("*.cu"))


def describe_build_config() -> str:
    cuda = should_build_cuda()
    compile_args = fff_hard_extra_compile_args()
    return (
        f"host={host_system()} ({platform.machine()}) | "
        f"cuda_build={'yes' if cuda else 'no'} | "
        f"torch.cuda.is_available()={cuda_is_available()} | "
        f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '(unset)')} | "
        f"cxx={compile_args.get('cxx')} | "
        f"nvcc={compile_args.get('nvcc', [])} | "
        f"link={fff_hard_extra_link_args()}"
    )


def extension_build_plan(csrc_dir: Path) -> dict[str, Any]:
    """Summary dict used by ``setup.py`` to decide Cpp vs CUDA extensions."""
    cuda_sources = discover_cuda_sources(csrc_dir)
    return {
        "system": host_system(),
        "machine": platform.machine(),
        "build_cuda": should_build_cuda() and bool(cuda_sources),
        "cuda_available": cuda_is_available(),
        "cuda_sources": cuda_sources,
        "compile_args": fff_hard_extra_compile_args(),
        "link_args": fff_hard_extra_link_args(),
        "include_dirs": fff_hard_include_dirs(),
        "library_dirs": fff_hard_library_dirs(),
    }
