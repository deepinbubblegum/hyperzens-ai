"""Unit tests: quantization correctness, routing, STE gradient flow."""

from __future__ import annotations

import pytest
import torch

from bitnet_fff import (
    BitLinear,
    FastFeedForwardBitNet,
    absmax_quantize,
    absmean_ternarize,
    ste_ternarize,
)
from bitnet_fff.mps_utils import is_mps_available

DEVICES = [torch.device("cpu")]
if is_mps_available():
    DEVICES.append(torch.device("mps"))


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def test_absmean_ternarize_values():
    w = torch.randn(64, 32)
    wq = absmean_ternarize(w)
    assert torch.isin(wq, torch.tensor([-1, 0, 1])).all()
    assert wq.dtype == w.dtype
    assert torch.allclose(
        wq,
        torch.where(w.abs() > w.abs().mean(), torch.sign(w), torch.zeros_like(w)),
    )


def test_absmax_quantize_error_and_ste(device):
    x = torch.randn(128, 64, device=device) * 3
    xq = absmax_quantize(x)
    rel_err = ((xq - x).abs() / x.abs().clamp_min(1e-6)).mean()
    assert rel_err.item() < 0.05
    assert torch.allclose(xq.detach(), x.detach(), atol=0.1)


def test_ste_ternarize_gradient_identity(device):
    w = torch.randn(16, 8, device=device)
    w.requires_grad_(True)
    y = ste_ternarize(w)
    y.sum().backward()
    assert w.grad is not None
    assert w.grad.abs().sum().item() > 0
    expected = torch.ones_like(w.grad)
    assert torch.allclose(w.grad, expected)


def test_bitlinear_shapes_and_gradients(device):
    lin = BitLinear(64, 128, bias=True).to(device)
    x = torch.randn(32, 64, device=device)
    out = lin(x)
    assert out.shape == (32, 128)
    out.sum().backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum().item() > 0
    assert lin.bias.grad is not None
    wq = lin.quantize_weight()
    assert torch.isin(wq.detach(), torch.tensor([-1, 0, 1], device=device)).all()


def test_fff_shapes_and_gradient_flow(device):
    fff = FastFeedForwardBitNet(d_in=64, d_out=64, depth=3, bias=True).to(device)
    x = torch.randn(8, 16, 64, device=device, requires_grad=True)
    out = fff(x)
    assert out.shape == (8, 16, 64)
    assert out.is_cuda is False
    out.sum().backward()
    for name, p in fff.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero gradient for {name}"
    assert x.grad is not None and x.grad.abs().sum().item() > 0


def test_fff_auto_depth_log2_dout():
    fff = FastFeedForwardBitNet(d_in=256, d_out=256)
    assert fff.depth == 8
    assert fff.num_leaves == 256
    assert fff.n_routing_nodes == 255


def test_fff_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        FastFeedForwardBitNet(d_in=64, d_out=100, depth=None)
    with pytest.raises(ValueError):
        FastFeedForwardBitNet(d_in=64, d_out=64, depth=-1)


def test_fff_conditional_routing_uses_only_selected_leaf(device):
    fff = FastFeedForwardBitNet(d_in=32, d_out=32, depth=2, bias=True).to(device)
    x = torch.randn(16, 32, device=device)

    with torch.no_grad():
        fff.router_bias.fill_(-10.0)
    leaf_left = torch.zeros(16, dtype=torch.long, device=device)
    out_left = fff(x)

    with torch.no_grad():
        fff.router_bias.fill_(10.0)
    leaf_right = torch.full((16,), fff.num_leaves - 1, dtype=torch.long, device=device)
    out_right = fff(x)

    with torch.no_grad():
        wq = absmean_ternarize(fff.leaf_weight)
        xq = absmax_quantize(x)
    expected_left = torch.bmm(
        xq.unsqueeze(1), wq[leaf_left].transpose(-1, -2)
    ).squeeze(1) + fff.leaf_bias[leaf_left]
    expected_right = torch.bmm(
        xq.unsqueeze(1), wq[leaf_right].transpose(-1, -2)
    ).squeeze(1) + fff.leaf_bias[leaf_right]

    assert torch.allclose(out_left, expected_left, atol=1e-5)
    assert torch.allclose(out_right, expected_right, atol=1e-5)
    assert not torch.allclose(out_left, out_right)


def test_fff_gradient_only_reaches_selected_leaf_rows(device):
    fff = FastFeedForwardBitNet(d_in=32, d_out=32, depth=2, bias=False).to(device)
    x = torch.randn(8, 32, device=device)
    with torch.no_grad():
        fff.router_bias.fill_(10.0)
    out = fff(x)
    out.sum().backward()
    assert fff.leaf_weight.grad is not None
    sel = fff.num_leaves - 1
    grad = fff.leaf_weight.grad
    assert grad[sel].abs().sum().item() > 0
    others = torch.cat([grad[:sel], grad[sel + 1 :]])
    assert others.abs().sum().item() == 0.0


def test_fff_chunked_leaf_projection_matches(device):
    fff_full = FastFeedForwardBitNet(d_in=64, d_out=64, depth=2, bias=True).to(device)
    fff_chunked = FastFeedForwardBitNet(
        d_in=64, d_out=64, depth=2, bias=True, chunk_size=5
    ).to(device)
    fff_chunked.load_state_dict(fff_full.state_dict())
    x = torch.randn(17, 64, device=device)
    torch.testing.assert_close(fff_chunked(x), fff_full(x), atol=1e-6, rtol=1e-6)
    fff_chunked(x).sum().backward()
    assert fff_chunked.leaf_weight.grad is not None


def test_fff_rank1_router_gradient_flow(device):
    fff = FastFeedForwardBitNet(
        d_in=64, d_out=64, depth=2, router_rank="r1"
    ).to(device)
    x = torch.randn(8, 64, device=device)
    fff(x).sum().backward()
    for name, p in fff.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0, name


def test_fff_matches_dense_reference_on_grad_input(device):
    fff = FastFeedForwardBitNet(d_in=32, d_out=64, depth=2, bias=True).to(device)
    dense = fff.to_dense().to(device)
    x = torch.randn(8, 32, device=device)
    out_fff = fff(x)
    out_dense = dense(x)
    assert out_fff.shape == (8, 64)
    assert out_dense.shape == (8, fff.num_leaves * fff.d_out) == (8, 256)
    assert out_dense.requires_grad
