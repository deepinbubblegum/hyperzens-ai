"""Tests for the packed-ternary fast-inference path (native CPU + Metal kernels)."""

from __future__ import annotations

import math

import pytest
import torch

from bitnet_fff import FastFeedForwardBitNet
from bitnet_fff.bitlinear import absmax_quantize, absmean_ternarize
from bitnet_fff.fast_inference import (
    FusedPackedFFFEvaluator,
    PackedFFFEvaluator,
    PackedTernaryMM,
    extension_available,
    pack_ternary_weights,
    ternary_mm,
    unpack_ternary_weights,
)

requires_ext = pytest.mark.skipif(
    not extension_available(),
    reason="packed-ternary native extension not built",
)


def _packed(w: torch.Tensor) -> torch.Tensor:
    packed, _ = pack_ternary_weights(w)
    return packed


def _fp_ref(x: torch.Tensor, w: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
    t = absmean_ternarize(w)
    return torch.bmm(x.unsqueeze(1), t[leaf].transpose(-1, -2)).squeeze(1)


@requires_ext
def test_pack_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(4, 5, 8)
    t = absmean_ternarize(w)
    packed, d_in_pad = pack_ternary_weights(w)
    assert d_in_pad == 8
    assert packed.dtype == torch.uint8
    assert packed.shape == (4, 5, 2)
    decoded = unpack_ternary_weights(packed)
    assert torch.equal(decoded, t)


@requires_ext
def test_pack_pads_non_multiple_of_four():
    torch.manual_seed(0)
    w = torch.randn(2, 3, 6)
    packed, d_in_pad = pack_ternary_weights(w)
    assert d_in_pad == 8
    assert packed.shape == (2, 3, 2)
    decoded = unpack_ternary_weights(packed)[..., :6]
    assert torch.equal(decoded, absmean_ternarize(w))


@pytest.mark.parametrize("leaf_dtype", [torch.int64, torch.int32])
@requires_ext
def test_cpu_matches_fp32_reference(leaf_dtype):
    torch.manual_seed(0)
    B, d_in, d_out, L = 17, 8, 5, 4
    x = torch.randn(B, d_in)
    w = torch.randn(L, d_out, d_in)
    leaf = torch.randint(0, L, (B,)).to(leaf_dtype)
    ref = _fp_ref(x, w, leaf.to(torch.long))
    out = ternary_mm(x, _packed(w), leaf)
    assert out.dtype == torch.float32
    assert out.shape == (B, d_out)
    assert torch.allclose(out, ref, atol=1e-5)


@requires_ext
def test_cpu_large_matches_reference():
    torch.manual_seed(0)
    B, d_in, d_out, L = 512, 256, 256, 8
    x = torch.randn(B, d_in)
    w = torch.randn(L, d_out, d_in)
    leaf = torch.randint(0, L, (B,))
    ref = _fp_ref(x, w, leaf)
    out = ternary_mm(x, _packed(w), leaf)
    assert torch.allclose(out, ref, atol=1e-4)


@requires_ext
def test_mps_matches_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    B, d_in, d_out, L = 64, 32, 16, 4
    x = torch.randn(B, d_in)
    w = torch.randn(L, d_out, d_in)
    leaf = torch.randint(0, L, (B,))
    cpu_out = ternary_mm(x, _packed(w), leaf)
    mps_out = ternary_mm(
        x.to("mps"), _packed(w).to("mps"), leaf.to(torch.int32).to("mps")
    )
    assert torch.allclose(mps_out.cpu(), cpu_out, atol=1e-5)


@requires_ext
def test_mps_matches_fp32_reference_large():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    B, d_in, d_out, L = 512, 256, 256, 8
    x = torch.randn(B, d_in)
    w = torch.randn(L, d_out, d_in)
    leaf = torch.randint(0, L, (B,))
    ref = _fp_ref(x, w, leaf)
    out = ternary_mm(x.to("mps"), _packed(w).to("mps"), leaf.to(torch.int32).to("mps"))
    assert torch.allclose(out.cpu(), ref, atol=1e-4)


@requires_ext
def test_mps_stress_with_interleaved_ops():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    _, d_in, d_out, L = 1, 256, 256, 8
    w = torch.randn(L, d_out, d_in)
    packed_m = _packed(w).to("mps")
    t = absmean_ternarize(w)
    for i in range(15):
        B = 1 + i * 7
        x = torch.randn(B, d_in)
        leaf = torch.randint(0, L, (B,))
        _ = torch.mm(x.to("mps"), torch.ones(d_in, 16, device="mps"))  # interleave
        ref = _fp_ref(x, t, leaf)
        out = ternary_mm(
            x.to("mps"), packed_m, leaf.to(torch.int32).to("mps")
        )
        assert torch.allclose(out.cpu(), ref, atol=1e-4), f"iter {i}"


@requires_ext
def test_evaluator_matches_module_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2)
    model.eval()
    x = torch.randn(23, 32)
    with torch.no_grad():
        ref = model._forward(x)
    evalr = PackedFFFEvaluator(model)
    leaf, _ = model._routing_forward(x)
    out = evalr(x, leaf)
    assert torch.allclose(out, ref, atol=1e-5)


@requires_ext
def test_evaluator_padded_din_matches_module_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=6, d_out=8, depth=2)
    model.eval()
    x = torch.randn(11, 6)
    with torch.no_grad():
        ref = model._forward(x)
    evalr = PackedFFFEvaluator(model)
    leaf, _ = model._routing_forward(x)
    out = evalr(x, leaf)
    assert torch.allclose(out, ref, atol=1e-5)
    assert evalr.d_in_padded == 8


@requires_ext
def test_evaluator_chunked_matches_full():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=64, d_out=16, depth=3)
    model.eval()
    x = torch.randn(50, 64)
    leaf, _ = model._routing_forward(x)
    full = PackedFFFEvaluator(model, chunk_size=None)(x, leaf)
    chunked = PackedFFFEvaluator(model, chunk_size=8)(x, leaf)
    assert torch.allclose(full, chunked, atol=1e-5)


@requires_ext
def test_fast_forward_matches_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=16, depth=2)
    model.eval()
    x = torch.randn(19, 32)
    with torch.no_grad():
        ref = model(x)
        fast = model.fast_forward(x)
    assert torch.allclose(fast, ref, atol=1e-5)
    x3 = torch.randn(4, 5, 32)
    with torch.no_grad():
        ref3 = model(x3)
        fast3 = model.fast_forward(x3)
    assert torch.allclose(fast3, ref3, atol=1e-5)


@requires_ext
def test_packed_autograd_backward():
    torch.manual_seed(0)
    B, d_in, d_out, L = 11, 8, 5, 4
    x = torch.randn(B, d_in, requires_grad=True)
    w = torch.randn(L, d_out, d_in, requires_grad=True)
    leaf = torch.randint(0, L, (B,))
    out = PackedTernaryMM.apply(x, w, leaf, 1e-8)
    out.sum().backward()
    assert x.grad is not None and w.grad is not None
    assert x.grad.shape == (B, d_in)
    assert w.grad.shape == (L, d_out, d_in)
    packed, _ = pack_ternary_weights(w.detach())
    # numeric: grad flows to routed leaves only
    assert torch.allclose(x.grad, unpack_ternary_weights(packed)[leaf].sum(dim=1), atol=1e-5)
    # grad_w is zero for leaves that were never routed
    routed = set(leaf.unique().tolist())
    for l in range(L):
        if l not in routed:
            assert torch.allclose(w.grad[l], torch.zeros_like(w.grad[l]), atol=0.0)


@requires_ext
def test_activations_quantized_path_matches_reference():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2, activation_bits=8)
    model.eval()
    x = torch.randn(13, 16)
    with torch.no_grad():
        ref = model._forward(x)
    leaf, _ = model._routing_forward(x)
    evalr = PackedFFFEvaluator(model)
    assert torch.allclose(evalr(x, leaf), ref, atol=1e-5)


@requires_ext
def test_fused_matches_module_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2, bias=True)
    model.eval()
    x = torch.randn(23, 32)
    with torch.no_grad():
        ref = model._forward(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-5)


@requires_ext
def test_fused_padded_din_matches_module_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=6, d_out=8, depth=2)
    model.eval()
    x = torch.randn(11, 6)
    with torch.no_grad():
        ref = model._forward(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-5)


@requires_ext
def test_fused_no_bias_no_quant_matches_reference():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2, bias=False, activation_bits=32)
    model.eval()
    x = torch.randn(9, 16)
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-5)


@requires_ext
def test_fused_activations_quantized_path_matches_reference():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2, activation_bits=8)
    model.eval()
    x = torch.randn(13, 16)
    with torch.no_grad():
        ref = model._forward(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-5)


@requires_ext
def test_fused_3d_matches_module_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2)
    model.eval()
    x = torch.randn(4, 5, 16)
    with torch.no_grad():
        ref = model(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-5)


@requires_ext
def test_fused_chunked_matches_full():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=64, d_out=16, depth=3)
    model.eval()
    x = torch.randn(50, 64)
    full = model.fast_forward(x)
    model._packed_eval.chunk_size = 8
    chunked = model.fast_forward(x)
    assert torch.allclose(full, chunked, atol=1e-5)


@requires_ext
def test_fused_mps_matches_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    cpu = FastFeedForwardBitNet(d_in=64, d_out=16, depth=2, bias=True, activation_bits=8)
    cpu.eval()
    mps = FastFeedForwardBitNet(d_in=64, d_out=16, depth=2, bias=True, activation_bits=8).to("mps")
    mps.eval()
    mps.load_state_dict(cpu.state_dict())
    x = torch.randn(21, 64)
    with torch.no_grad():
        fc = cpu.fast_forward(x)
        fm = mps.fast_forward(x.to("mps"))
    assert torch.allclose(fm.cpu(), fc, atol=1e-4)


@requires_ext
def test_fused_mps_matches_reference_large():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=256, d_out=256, depth=3, bias=True)
    model = model.to("mps")
    model.eval()
    x = torch.randn(64, 256, device="mps")
    with torch.no_grad():
        ref = model._forward(x)
        fused = model.fast_forward(x)
    assert torch.allclose(fused, ref, atol=1e-4)


@requires_ext
def test_fused_r1_fallback_requires_leaf_idx():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2, router_rank="r1")
    model.eval()
    evalr = PackedFFFEvaluator(model)
    x = torch.randn(7, 32)
    leaf, _ = model._routing_forward(x)
    with torch.no_grad():
        ref = model._forward(x)
        out = evalr(x, leaf)
    assert torch.allclose(out, ref, atol=1e-5)
    with torch.no_grad():
        try:
            evalr(x)
            raise AssertionError("expected ValueError for r1 fused routing")
        except ValueError:
            pass


@requires_ext
def test_fused_r1_fast_forward_matches_forward():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=16, depth=2, router_rank="r1")
    model.eval()
    x = torch.randn(19, 32)
    with torch.no_grad():
        ref = model(x)
        fast = model.fast_forward(x)
    assert torch.allclose(fast, ref, atol=1e-5)


@requires_ext
def test_fused_evaluator_router_buffers_flattened():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=6, d_out=8, depth=2, bias=True)
    model.eval()
    evalr = FusedPackedFFFEvaluator(model)
    assert evalr.router_w.dtype == torch.float32
    assert evalr.router_w.is_contiguous()
    assert evalr.router_b.dtype == torch.float32
    assert evalr.router_b.is_contiguous()
    assert evalr.router_w.shape == (model.num_leaves - 1, evalr.d_in_padded)
    assert evalr.router_b.shape == (model.num_leaves - 1,)
    assert evalr.packed.dtype == torch.uint8
    x = torch.randn(11, 6)
    with torch.no_grad():
        ref = model._forward(x)
        out = evalr(x)
    assert torch.allclose(out, ref, atol=1e-5)


@requires_ext
def test_fused_evaluator_3d_and_chunked():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2)
    model.eval()
    evalr = FusedPackedFFFEvaluator(model)
    x = torch.randn(3, 4, 32)
    with torch.no_grad():
        ref = model(x)
        out = evalr(x)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5)
    evalr.chunk_size = 5
    x2 = torch.randn(13, 32)
    with torch.no_grad():
        chunked = evalr(x2)
        full = FusedPackedFFFEvaluator(model)(x2)
    assert torch.allclose(chunked, full, atol=1e-5)


@requires_ext
def test_fused_evaluator_matches_classic_evaluator():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2, bias=True)
    model.eval()
    x = torch.randn(21, 16)
    leaf, _ = model._routing_forward(x)
    fused = FusedPackedFFFEvaluator(model)(x)
    classic = PackedFFFEvaluator(model)(x, leaf)
    assert torch.allclose(fused, classic, atol=1e-5)


@requires_ext
def test_fused_evaluator_requires_full_rank():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2, router_rank="r1")
    try:
        FusedPackedFFFEvaluator(model)
        raise AssertionError("expected ValueError for r1 router")
    except ValueError:
        pass


@requires_ext
def test_fast_forward_uses_fused_evaluator_in_eval():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2)
    model.eval()
    x = torch.randn(9, 16)
    with torch.no_grad():
        model.fast_forward(x)
    assert isinstance(model._packed_eval, FusedPackedFFFEvaluator)


@requires_ext
def test_fast_forward_training_falls_back_to_reference():
    torch.manual_seed(0)
    model = FastFeedForwardBitNet(d_in=16, d_out=8, depth=2)
    model.train()
    x = torch.randn(7, 16, requires_grad=True)
    with torch.no_grad():
        ref = model(x)
        fast = model.fast_forward(x)
    assert torch.allclose(fast, ref, atol=1e-5)
    assert not hasattr(model, "_packed_eval")
    out = model.fast_forward(x)
    assert out.requires_grad


@requires_ext
def test_fused_evaluator_mps_matches_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    torch.manual_seed(0)
    cpu = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2, bias=True)
    cpu.eval()
    mps = FastFeedForwardBitNet(d_in=32, d_out=8, depth=2, bias=True).to("mps")
    mps.eval()
    mps.load_state_dict(cpu.state_dict())
    x = torch.randn(17, 32)
    fused_cpu = FusedPackedFFFEvaluator(cpu)(x)
    fused_mps = FusedPackedFFFEvaluator(mps)(x.to("mps"))
    assert torch.allclose(fused_mps.cpu(), fused_cpu, atol=1e-4)
