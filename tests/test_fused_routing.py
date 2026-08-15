"""Numerical equivalence of the fused single-pass routing kernel vs the FP32 reference.

Validates that :meth:`FastFeedForwardBitNet.fast_forward` (the fused
Metal/NEON single-pass kernel: routing + AbsMax activation quantization +
packed-ternary leaf GEMM in one native call) reproduces
:meth:`FastFeedForwardBitNet.forward` (the FP32 reference) across tree depths
3, 5, 7 and 8, on CPU and MPS, with and without activation quantization, for
batched 2D and 3D inputs. Also checks that the in-kernel routing selects the
same leaves as the torch ``_routing_forward`` reference.
"""

from __future__ import annotations

import pytest
import torch

from bitnet_fff import FastFeedForwardBitNet
from bitnet_fff.fast_inference import (
    FusedPackedFFFEvaluator,
    PackedFFFEvaluator,
    extension_available,
)

requires_ext = pytest.mark.skipif(
    not extension_available(),
    reason="packed-ternary native extension not built",
)

DEPTHS = [3, 5, 7, 8]

D_IN = 32
D_OUT = 16
BATCH = 64
ATOL = 1e-4


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _model(depth: int, device: torch.device, bias: bool = True, **kw) -> FastFeedForwardBitNet:
    model = FastFeedForwardBitNet(
        d_in=D_IN, d_out=D_OUT, depth=depth, bias=bias, **kw
    ).to(device)
    model.eval()
    return model


@pytest.mark.parametrize("depth", DEPTHS)
@requires_ext
def test_fused_matches_fp32_reference_cpu(depth):
    torch.manual_seed(0)
    model = _model(depth, torch.device("cpu"))
    x = torch.randn(BATCH, D_IN)
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    assert fused.shape == ref.shape
    assert torch.allclose(fused, ref, atol=ATOL), (
        f"depth={depth} fused vs fp32 ref, max diff {_max_diff(fused, ref):.3e}"
    )


@pytest.mark.parametrize("depth", DEPTHS)
@requires_ext
def test_fused_matches_fp32_reference_mps(depth):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    model = _model(depth, torch.device("mps"))
    x = torch.randn(BATCH, D_IN, device="mps")
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    torch.mps.synchronize()
    assert fused.shape == ref.shape
    assert torch.allclose(fused, ref, atol=ATOL), (
        f"depth={depth} MPS fused vs fp32 ref, max diff {_max_diff(fused, ref):.3e}"
    )


@pytest.mark.parametrize("depth", DEPTHS)
@requires_ext
def test_fused_matches_fp32_reference_no_quant(depth):
    torch.manual_seed(0)
    model = _model(depth, torch.device("cpu"), bias=False, activation_bits=32)
    x = torch.randn(24, D_IN)
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=ATOL), (
        f"depth={depth} no-quant fused vs fp32 ref, max diff {_max_diff(fused, ref):.3e}"
    )


@pytest.mark.parametrize("depth", DEPTHS)
@requires_ext
def test_fused_matches_fp32_reference_3d(depth):
    torch.manual_seed(0)
    model = _model(depth, torch.device("cpu"))
    x = torch.randn(2, 5, D_IN)
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    assert fused.shape == ref.shape
    assert torch.allclose(fused, ref, atol=ATOL), (
        f"depth={depth} 3D fused vs fp32 ref, max diff {_max_diff(fused, ref):.3e}"
    )


@pytest.mark.parametrize("depth", DEPTHS)
@requires_ext
def test_fused_routing_agrees_with_torch_routing(depth):
    torch.manual_seed(0)
    model = _model(depth, torch.device("cpu"))
    x = torch.randn(96, D_IN)
    leaf, _ = model._routing_forward(x)
    fused = FusedPackedFFFEvaluator(model)(x)
    classic = PackedFFFEvaluator(model)(x, leaf)
    assert torch.allclose(fused, classic, atol=ATOL), (
        f"depth={depth} fused vs classic routing, max diff {_max_diff(fused, classic):.3e}"
    )


@requires_ext
def test_all_depths_route_multiple_leaves():
    torch.manual_seed(0)
    for depth in DEPTHS:
        model = _model(depth, torch.device("cpu"))
        x = torch.randn(BATCH, D_IN)
        leaf, _ = model._routing_forward(x)
        routed = set(leaf.tolist())
        assert len(routed) > 1, (
            f"depth={depth} expected a spread of leaves, got {len(routed)}"
        )
        assert max(routed) < model.num_leaves
