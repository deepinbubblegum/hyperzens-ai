#!/usr/bin/env python3
"""Cross-platform build for Hyperzens FFF + ``fff_hard`` native extension(s).

Hosts
-----
* **macOS (Apple Silicon)** — CPU-only. Builds ``fff_hard_cpp`` with
  ``-O3 -std=c++17``. OpenMP is optional (Homebrew libomp); threading still
  works via PyTorch ``at::parallel_for``. CUDA is **skipped** (no errors).
* **Linux x86_64 + RTX 3060** — full build. C++ flags include
  ``-O3 -march=native -fopenmp -funroll-loops``. If CUDA is available **and**
  ``csrc/*.cu`` sources exist, also builds a ``CUDAExtension`` for ``sm_86``.

Examples
--------
    # Mac M4 local CPU benchmarks
    pip install -e .

    # Linux RTX 3060
    export TORCH_CUDA_ARCH_LIST=8.6
    pip install -e .
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

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to build fff_hard. Install torch first, then retry."
    ) from exc

# CUDAExtension is optional — missing on CPU-only torch wheels is fine.
try:
    from torch.utils.cpp_extension import CUDAExtension
except ImportError:  # pragma: no cover
    CUDAExtension = None  # type: ignore[misc, assignment]


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


def _cpu_extension() -> CppExtension:
    """Always-built CPU hard-routing kernel (macOS + Linux)."""
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
    if not cuda_sources:
        return None
    if not should_build_cuda():
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
    plan = extension_build_plan(_CSRC)
    _print_banner(plan)

    modules: list = [_cpu_extension()]

    # macOS: never attempt CUDA (even if a .cu file exists).
    if is_macos():
        return modules

    cuda_ext = _cuda_extension_or_none(plan["cuda_sources"])
    if cuda_ext is not None:
        modules.append(cuda_ext)
    return modules


ext_modules = build_ext_modules()

setup(
    name="hyperzens-ai",
    version="0.1.0",
    description=(
        "Fast Feedforward (FFF) LM — macOS CPU dev + Linux x86_64/RTX 3060 (sm_86)"
    ),
    packages=find_packages(exclude=("old", "data.cache", "csrc.build")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "tiktoken>=0.7.0",
    ],
    zip_safe=False,
)
