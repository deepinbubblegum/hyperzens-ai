"""Packed 2-bit BitNet ternary GEMM: packing, kernel accuracy and gradients.

CPU-side tests are device-agnostic and exercise the torch reference path of
:func:`bitnet_fff.triton_ternary_mm.packed_ternary_matmul` (used everywhere
without CUDA) plus the packing/unpacking round trip and the STE backward.
CUDA tests (skipped when no GPU/triton) compare the fused Triton kernel
against the FP32 reference and against ``torch.matmul`` for the exact ternary
matmul case (``quantize=False, apply_scale=False``).
"""

from __future__ import annotations

import pytest
import torch

from bitnet_fff.triton_ternary_mm import (
    PackedBitLinear,
    PackedTernaryBitLinear,
    has_triton,
    pack_ternary,
    pack_ternary_int8,
    packed_ternary_matmul,
    packed_ternary_mm_ref,
    unpack_ternary,
)

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_triton),
    reason="requires CUDA and triton",
)

# kernel is FP16; the reference is FP32. Random N(0,1) inputs with ternary
# weights give |Z| ~ sqrt(K), so a modest absolute tolerance covers FP16 noise.
KERNEL_ATOL = 5e-2


def _ternary_w(n: int, k: int, device=torch.device("cpu"), p_zero: float = 0.2):
    """Random weight in {-1, 0, +1} of shape (N, K)."""
    s = torch.randint(0, 3, (n, k), device=device)
    w = torch.where(s == 0, torch.zeros((), device=device),
                    torch.where(s == 1, torch.ones((), device=device),
                                -torch.ones((), device=device)))
    if p_zero > 0:
        drop = torch.rand(n, k, device=device) < p_zero
        w = w.masked_fill(drop, 0.0)
    return w


# --------------------------------------------------------------------------
# packing / unpacking
# --------------------------------------------------------------------------


def test_pack_unpack_roundtrip_exact():
    torch.manual_seed(0)
    for (n, k) in [(1, 1), (16, 16), (48, 64), (100, 33), (31, 128)]:
        w = _ternary_w(n, k)
        # Test int32 (16 weights/word)
        packed_i32 = pack_ternary(w, dtype=torch.int32)
        s_i32 = unpack_ternary(packed_i32, n=n, k=k)
        assert s_i32.shape == (n, k)
        assert torch.equal(s_i32, w), f"int32 roundtrip mismatch for ({n}, {k})"

        # Test int8 (4 weights/byte)
        packed_i8 = pack_ternary_int8(w)
        s_i8 = unpack_ternary(packed_i8, n=n, k=k)
        assert s_i8.shape == (n, k)
        assert torch.equal(s_i8, w), f"int8 roundtrip mismatch for ({n}, {k})"


def test_pack_word_layout():
    # N=16, K=1: single word packs 16 consecutive output channels along N (shape (16, 1)),
    # 2 bits each: 1 -> +1, 2 -> -1, 0 -> 0.
    w = torch.tensor([[1.0], [-1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0],
                       [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [-1.0]])
    packed = pack_ternary(w, dtype=torch.int32)
    assert packed.shape == (1, 1)
    assert packed.dtype == torch.int32
    word = int(packed[0, 0])
    assert (word & 0x03) == 1        # channel 0  -> +1
    assert ((word >> 2) & 0x03) == 2  # channel 1  -> -1
    assert ((word >> 4) & 0x03) == 0  # channel 2  -> 0
    assert ((word >> 6) & 0x03) == 1  # channel 3  -> +1
    assert ((word >> 30) & 0x03) == 2  # channel 15 -> -1


def test_pack_int8_layout():
    # N=4, K=1: single int8 byte packs 4 consecutive output channels along N (shape (4, 1))
    w = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])
    packed = pack_ternary_int8(w)
    assert packed.shape == (1, 1)
    assert packed.dtype == torch.int8
    byte_val = int(packed[0, 0]) & 0xFF
    assert (byte_val & 0x03) == 1       # channel 0 -> +1
    assert ((byte_val >> 2) & 0x03) == 2 # channel 1 -> -1
    assert ((byte_val >> 4) & 0x03) == 0 # channel 2 -> 0
    assert ((byte_val >> 6) & 0x03) == 1 # channel 3 -> +1


def test_pack_pads_n_to_word_boundary():
    w = torch.randn(3, 8)
    packed = pack_ternary(w, dtype=torch.int32)
    assert packed.shape == (8, 1)  # 3 -> padded to 16 channels -> 1 word
    s = unpack_ternary(packed, n=3, k=8)
    assert torch.equal(s, torch.sign(w))


# --------------------------------------------------------------------------
# reference forward accuracy
# --------------------------------------------------------------------------


def test_ref_matches_bruteforce_fp64():
    torch.manual_seed(1)
    n, k = 16, 32
    x = torch.randn(7, k, dtype=torch.float64)
    w = torch.randn(n, k, dtype=torch.float64) * 0.5
    s = torch.sign(w)
    beta = w.abs().mean(dim=-1)
    gamma = x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    xq = torch.floor(x / gamma + 0.5).clamp(-127.0, 127.0)
    y_manual = (xq @ s.T) * (gamma * beta)
    y_ref = packed_ternary_mm_ref(x, w, apply_scale=True, quantize=True)
    assert torch.allclose(y_ref.double(), y_manual, atol=1e-12)


def test_ref_unquantized_matches_torch_matmul():
    torch.manual_seed(2)
    n, k = 64, 128
    x = torch.randn(5, k)
    w = _ternary_w(n, k)
    y_ref = packed_ternary_mm_ref(x, w, apply_scale=False, quantize=False)
    assert torch.allclose(y_ref, x @ w.T, atol=1e-5)


# --------------------------------------------------------------------------
# autograd Function (forward and backward with BitNet STE)
# --------------------------------------------------------------------------


def test_forward_matches_reference_all_modes():
    torch.manual_seed(3)
    n, k = 24, 40
    x = torch.randn(6, k)
    w = torch.randn(n, k) * 0.5
    for apply_scale in (True, False):
        for quantize in (True, False):
            out = PackedTernaryBitLinear.apply(x, w, apply_scale, quantize)
            ref = packed_ternary_mm_ref(x, w, apply_scale, quantize)
            assert torch.allclose(out, ref, atol=1e-5), (
                f"mode apply_scale={apply_scale} quantize={quantize}"
            )


def test_backward_matches_formula_all_modes():
    torch.manual_seed(4)
    n, k = 24, 40
    for apply_scale in (True, False):
        for quantize in (True, False):
            x = torch.randn(6, k, requires_grad=True)
            w = (torch.randn(n, k) * 0.5).requires_grad_(True)
            out = PackedTernaryBitLinear.apply(x, w, apply_scale, quantize)
            loss = out.pow(2).sum()
            loss.backward()

            # Analytical formula: gZ = 2*out*scale, gX = gZ @ S, gW = gZ^T @ Xq
            gz = 2.0 * out.detach().float()
            s = torch.sign(w.detach()).float()
            xf = x.detach().float()
            if quantize:
                gamma = xf.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
                xq = torch.where(xf / gamma >= 0,
                                 torch.floor(xf / gamma + 0.5),
                                 torch.ceil(xf / gamma - 0.5)).clamp(-127, 127)
            else:
                xq = xf
                gamma = 1.0

            if apply_scale:
                beta = w.detach().abs().mean(dim=-1).float()
                scale = (gamma * beta) if quantize else beta[None, :]
                gz = gz * scale
            elif quantize:
                gz = gz * gamma

            gx_ref = gz @ s
            gw_ref = gz.T @ xq
            assert torch.allclose(x.grad, gx_ref, atol=1e-4), "gx mismatch"
            assert torch.allclose(w.grad, gw_ref, atol=1e-4), "gw mismatch"


def test_backward_matches_torch_matmul_unquantized():
    torch.manual_seed(5)
    n, k = 32, 64
    x = torch.randn(8, k, dtype=torch.float64, requires_grad=True)
    w = _ternary_w(n, k, p_zero=0.0).double().requires_grad_(True)

    out_trit = PackedTernaryBitLinear.apply(
        x, w, False, False
    )
    loss_trit = out_trit.pow(2).sum()
    gx_trit, gw_trit = torch.autograd.grad(loss_trit, (x, w))

    ref = torch.nn.functional.linear(x, w, None)
    loss_ref = ref.pow(2).sum()
    gx_ref, gw_ref = torch.autograd.grad(loss_ref, (x, w))

    assert torch.allclose(gx_trit, gx_ref, atol=1e-10), "gx vs torch.matmul"
    assert torch.allclose(gw_trit, gw_ref, atol=1e-10), "gw vs torch.matmul"


def test_finite_difference_unquantized_x_gradient():
    # With quantization and scales off, and W already ternary, forward is the
    # exact linear map X @ S^T, so autograd grads must match central
    # differences in X.
    torch.manual_seed(6)
    n, k = 16, 32
    x = torch.randn(5, k, dtype=torch.float64, requires_grad=True)
    w = _ternary_w(n, k, p_zero=0.0).double()

    def loss(v):
        out = PackedTernaryBitLinear.apply(v, w, False, False)
        return out.pow(2).sum()

    grad = torch.autograd.grad(loss(x), x)[0]
    eps = 1e-6
    v = torch.randn_like(x)
    v = v / v.norm()
    fd = (loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)
    assert torch.allclose(fd, (grad * v).sum(), atol=1e-4), (
        f"finite difference {fd.item()} vs autograd {(grad * v).sum().item()}"
    )


# --------------------------------------------------------------------------
# nn.Module integration
# --------------------------------------------------------------------------


def test_module_matches_nn_linear_unquantized():
    torch.manual_seed(7)
    n, k = 24, 48
    w = _ternary_w(n, k, p_zero=0.0)
    module = PackedBitLinear(k, n, apply_scale=False, quantize=False)
    module.weight.data.copy_(w)

    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data.copy_(w)

    x = torch.randn(4, k)
    assert torch.allclose(module(x), linear(x), atol=1e-6)

    # packed_weight() gives the exact ternary codes back
    assert torch.equal(unpack_ternary(module.packed_weight(), n=n, k=k),
                       torch.sign(module.weight.detach()))


def test_module_3d_input_and_shapes():
    torch.manual_seed(8)
    module = PackedBitLinear(16, 32)
    x = torch.randn(3, 7, 16)
    y = module(x)
    assert y.shape == (3, 7, 32)
    y.sum().backward()
    assert module.weight.grad is not None and module.weight.grad.shape == (32, 16)
    # 2D input stays 2D
    assert module(torch.randn(5, 16)).shape == (5, 32)


def test_module_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PackedBitLinear(0, 8)
    with pytest.raises(ValueError):
        PackedBitLinear(8, -1)


# --------------------------------------------------------------------------
# CUDA: fused Triton kernel (RTX 3060 / Ampere Tensor Cores)
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("apply_scale", [True, False])
@pytest.mark.parametrize("quantize", [True, False])
def test_kernel_matches_reference(apply_scale, quantize):
    torch.manual_seed(9)
    # Non-multiples of 16 and 64 exercise tile boundary masking/padding
    n, k = 48, 128
    x = torch.randn(100, k, device="cuda")
    w = torch.randn(n, k, device="cuda") * 0.5
    out = packed_ternary_matmul(x, w, apply_scale, quantize)
    ref = packed_ternary_mm_ref(x.cpu(), w.cpu(), apply_scale, quantize)
    assert out.shape == (100, n)
    assert torch.allclose(out.float().cpu(), ref, atol=KERNEL_ATOL), (
        f"apply_scale={apply_scale} quantize={quantize} "
        f"max diff {(out.float().cpu() - ref).abs().max().item():.3e}"
    )


@requires_cuda
def test_kernel_matches_torch_matmul_unquantized():
    torch.manual_seed(10)
    n, k = 64, 256
    x = torch.randn(70, k, device="cuda")
    w = _ternary_w(n, k, p_zero=0.0, device=torch.device("cuda"))
    out = packed_ternary_matmul(x, w, apply_scale=False, quantize=False)
    ref = x.float() @ w.float().T
    assert torch.allclose(out.float().cpu(), ref.cpu(), atol=KERNEL_ATOL), (
        f"kernel vs torch.matmul max diff {(out.float().cpu() - ref.cpu()).abs().max().item():.3e}"
    )


@requires_cuda
def test_kernel_backward_matches_formula():
    torch.manual_seed(11)
    n, k = 32, 128
    x = torch.randn(64, k, device="cuda", requires_grad=True)
    w = (torch.randn(n, k, device="cuda") * 0.5).requires_grad_(True)
    out = PackedTernaryBitLinear.apply(x, w, True, True)
    loss = out.pow(2).sum()
    gx, gw = torch.autograd.grad(loss, (x, w))

    s = torch.sign(w.detach())
    xf = x.detach().float()
    gamma = xf.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    xq = torch.where(xf / gamma >= 0,
                     torch.floor(xf / gamma + 0.5),
                     torch.ceil(xf / gamma - 0.5)).clamp(-127, 127)
    beta = w.detach().abs().mean(dim=-1)
    gz = 2.0 * out.detach().float() * (gamma * beta)
    gx_ref = gz @ s
    gw_ref = gz.T @ xq
    assert torch.allclose(gx.float().cpu(), gx_ref.cpu(), atol=KERNEL_ATOL)
    assert torch.allclose(gw.float().cpu(), gw_ref.cpu(), atol=KERNEL_ATOL)


@requires_cuda
@pytest.mark.parametrize("m,n,k", [
    (1, 1024, 1024),     # Token generation (single token decode)
    (32, 2048, 2048),   # Small batch inference
    (128, 4096, 4096),  # Prefill / training batch on RTX 3060
])
def test_kernel_shapes_rtx3060(m, n, k):
    torch.manual_seed(12)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.5
    out = packed_ternary_matmul(x, w, apply_scale=True, quantize=True)
    assert out.shape == (m, n)
    assert torch.isfinite(out).all()

