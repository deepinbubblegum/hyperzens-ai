#!/usr/bin/env python3
"""Cross-platform build for Hyperzens FFF + ``fff_hard`` native extension(s).

Hosts
-----
* **macOS (Apple Silicon)** — CPU-only ``fff_hard_cpp`` (``-O3 -std=c++17``).
* **Linux x86_64 + RTX 3060** — CPU + optional CUDA ``sm_86`` if ``csrc/*.cu`` exists.

Important — pip build isolation
-------------------------------
``pip install -e .`` runs ``setup.py`` in an **isolated** env that does **not**
see your venv's ``torch``. Always install torch first, then use
``--no-build-isolation`` so the extension links against the venv PyTorch::

    pip install torch  # or your CUDA wheel
    export TORCH_CUDA_ARCH_LIST=8.6   # Linux RTX 3060
    pip install -e . --no-build-isolation

If torch is missing, this setup still installs the pure-Python package; the
C++ kernel can JIT-compile later via ``models.fff_layer`` or you can re-run
the command above to build the extension in-place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
if str(_CSRC) not in sys.path:
    sys.path.insert(0, str(_CSRC))

# compiler_flags is pure-Python (torch import is deferred inside helpers).
from compiler_flags import (  # noqa: E402
    TARGET_CUDA_ARCH,
    apply_cuda_arch_env,
    describe_build_config,
    extension_build_plan,
    fff_hard_extra_compile_args,
    fff_hard_extra_link_args,
    fff_hard_include_dirs,
    fff_hard_library_dirs,
    fff_hard_nvcc_flags,
    is_linux,
    is_macos,
    should_build_cuda,
)

_TORCH_CPP_EXT_AVAILABLE = False
BuildExtension = None  # type: ignore[misc, assignment]
CppExtension = None  # type: ignore[misc, assignment]
CUDAExtension = None  # type: ignore[misc, assignment]

try:
    from torch.utils.cpp_extension import BuildExtension as _BuildExtension
    from torch.utils.cpp_extension import CppExtension as _CppExtension

    BuildExtension = _BuildExtension
    CppExtension = _CppExtension
    _TORCH_CPP_EXT_AVAILABLE = True
    try:
        from torch.utils.cpp_extension import CUDAExtension as _CUDAExtension

        CUDAExtension = _CUDAExtension
    except ImportError:
        CUDAExtension = None
except ImportError:
    print(
        "=" * 72 + "\n"
        "WARNING: PyTorch not importable in this build environment.\n"
        "  → Installing pure-Python package only (no fff_hard C++ extension).\n"
        "  → To compile the native extension against your venv torch:\n"
        "       pip install torch\n"
        "       pip install -e . --no-build-isolation\n"
        "  (pip's default build isolation hides venv packages like torch.)\n"
        + "=" * 72,
        file=sys.stderr,
    )


def _print_banner(plan: dict) -> None:
    print("=" * 72)
    print("fff_hard cross-platform native build")
    print(describe_build_config())
    if is_macos():
        print("Platform: macOS — CPU extension only (CUDA skipped by design).")
    elif is_linux():
        if plan["build_cuda"]:
            print(
                f"Platform: Linux — CPU + CUDA (sm_86 / "
                f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', TARGET_CUDA_ARCH)})"
            )
            print(f"  CUDA sources: {plan['cuda_sources']}")
            print(f"  NVCC flags:   {fff_hard_nvcc_flags()}")
        elif should_build_cuda() and not plan["cuda_sources"]:
            print(
                "Platform: Linux — CUDA available, but no csrc/*.cu sources; "
                "building CPU extension only."
            )
        else:
            print("Platform: Linux — CPU extension only (CUDA unavailable).")
    print("=" * 72)


def _cpu_extension():
    """CPU hard-routing kernel (macOS + Linux)."""
    assert CppExtension is not None
    compile_args = fff_hard_extra_compile_args()
    return CppExtension(
        name="fff_hard_cpp",
        sources=[str(_CSRC / "fff_hard.cpp")],
        include_dirs=fff_hard_include_dirs(),
        library_dirs=fff_hard_library_dirs(),
        extra_compile_args={"cxx": list(compile_args.get("cxx", []))},
        extra_link_args=fff_hard_extra_link_args(),
        language="c++",
    )


def _cuda_extension_or_none(cuda_sources: list[str]):
    """Optional CUDA sibling targeting RTX 3060 (sm_86). Never built on macOS."""
    if not cuda_sources or not should_build_cuda():
        return None
    if CUDAExtension is None:
        print("CUDAExtension unavailable in this torch build — skipping CUDA.")
        return None

    apply_cuda_arch_env()
    compile_args = fff_hard_extra_compile_args()
    return CUDAExtension(
        name="fff_hard_cuda",
        sources=[str(_CSRC / "fff_hard.cpp"), *cuda_sources],
        include_dirs=fff_hard_include_dirs(),
        library_dirs=fff_hard_library_dirs(),
        extra_compile_args={
            "cxx": list(compile_args.get("cxx", [])),
            "nvcc": list(compile_args.get("nvcc", fff_hard_nvcc_flags())),
        },
        extra_link_args=fff_hard_extra_link_args(),
    )


def build_ext_modules() -> list:
    if not _TORCH_CPP_EXT_AVAILABLE:
        return []

    plan = extension_build_plan(_CSRC)
    _print_banner(plan)

    modules: list = [_cpu_extension()]
    if is_macos():
        return modules

    cuda_ext = _cuda_extension_or_none(plan["cuda_sources"])
    if cuda_ext is not None:
        modules.append(cuda_ext)
    return modules


ext_modules = build_ext_modules()
cmdclass: dict = {}
if _TORCH_CPP_EXT_AVAILABLE and ext_modules and BuildExtension is not None:
    cmdclass = {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}

setup(
    name="hyperzens-ai",
    version="0.1.0",
    description=(
        "Fast Feedforward (FFF) LM — macOS CPU dev + Linux x86_64/RTX 3060 (sm_86)"
    ),
    packages=find_packages(exclude=("old", "data.cache", "csrc.build")),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "tiktoken>=0.7.0",
    ],
    zip_safe=False,
)
