"""Build the packed-ternary C++/Metal extension with torch.utils.cpp_extension.

Generates ``csrc/metal_ternary_kernel.h`` from ``csrc/ternary_add.metal`` so the
Metal shader source stays a single-source-of-truth deliverable, then compiles
the CPU (NEON) and Metal (MPS) kernels. The extension module is cached by
torch keyed on source contents + flags; a metal-source hash flag forces a
rebuild whenever the shader changes.
"""

from __future__ import annotations

import hashlib
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent


def _build() -> "module":
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
