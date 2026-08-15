"""Build the packed-ternary C++/Metal extension with torch.utils.cpp_extension.

Generates ``csrc/metal_ternary_kernel.h`` from ``csrc/ternary_add.metal`` so the
Metal shader source stays a single-source-of-truth deliverable, then compiles
the CPU (NEON) and Metal (MPS) kernels. The extension module is cached by
torch keyed on source contents + flags; a metal-source hash flag forces a
rebuild whenever the shader changes.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent


def _ensure_ninja_on_path() -> None:
    """Prepend the interpreter's ``bin`` dir to PATH so ninja is found.

    torch's cpp_extension runs the build subprocess with ``cwd=build_directory``
    (a cache dir), so relative PATH entries and the caller's working dir are not
    reliable. The venv that spawned this interpreter always contains ninja.
    """
    venv_bin = pathlib.Path(sys.executable).resolve().parent
    if str(venv_bin) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")


def _build() -> "module":
    _ensure_ninja_on_path()
    metal_src = (_ROOT / "ternary_add.metal").read_text()
    metal_hash = int(hashlib.sha256(metal_src.encode()).hexdigest()[:12], 16)

    header = (
        "#pragma once\n"
        "static const char TERNARY_ADD_METAL_SOURCE[] = R\"METALSRC(\n"
        + metal_src
        + "\n)METALSRC\";\n"
    )
    (_ROOT / "metal_ternary_kernel.h").write_text(header)

    from torch.utils.cpp_extension import load

    return load(
        name="bitnet_fff_ternary",
        sources=[
            str(_ROOT / "ternary_packed.cpp"),
            str(_ROOT / "ternary_packed_mps.mm"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-fobjc-arc",
            f"-DMETAL_SRC_HASH={metal_hash}",
        ],
        extra_ldflags=[
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "MetalPerformanceShaders",
        ],
        is_python_module=True,
        verbose=False,
    )


_ext = None


def get_extension():
    global _ext
    if _ext is None:
        _ext = _build()
    return _ext
