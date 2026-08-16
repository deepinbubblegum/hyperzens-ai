"""Unit tests for BitNetUFFLayer (Ultra-Fast Feedforward, flat top-k indexing)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bitnet_fff import (
    BitNetUFFLayer,
    FastFeedForwardBitNet,
    absmax_quantize,
    absmean_ternarize,
    ste_ternarize,
)
from bitnet_fff.models import BitNetFFTConfig, BitNetFFTMLP, BitNetFFTTransformer
from bitnet_fff.mps_utils import is_mps_available

DEVICES = [torch.device("cpu")]
if is_mps_available():
    DEVICES.append(torch.device("mps"))


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _manual_uff_forward(fff: BitNetUFFLayer, x: torch.Tensor) -> torch.Tensor:
    """Reference forward: per-leaf path loop + top-k gather (no flat indexing).

    Mirrors the layer's exact math but derives every leaf path by a plain
    loop over ``num_leaves * depth`` heap ids, so a match proves the
    vectorized flat-gather implementation is correct.
    """
    B, _ = x.shape
    depth, leaves, k = fff.depth, fff.num_leaves, fff.k
    wq_r = absmean_ternarize(fff.router_weight)
    wq_l = absmean_ternarize(fff.leaf_weight)
    xq = absmax_quantize(x, bits=fff.activation_bits) if fff.activation_bits < 32 else x

    logits = x @ wq_r.T + fff.router_bias

    leaf_logp = torch.zeros(B, leaves, device=x.device, dtype=x.dtype)
    for l in range(leaves):
        for level in range(depth):
            n = (1 << level) - 1 + (l >> (depth - level))
            bit = (l >> (depth - 1 - level)) & 1
            sign = 1.0 if bit else -1.0
            leaf_logp[:, l] += -F.softplus(-sign * logits[:, n])

    top_logp, top_idx = torch.topk(leaf_logp, k, dim=-1)
    w = torch.softmax(top_logp, dim=-1)
    w_sel = wq_l[top_idx]
    proj = torch.matmul(
        w_sel.transpose(-1, -2), xq[:, None, :, None]
    ).squeeze(-1)
    proj = proj + fff.leaf_bias[top_idx]
    return (proj * w.unsqueeze(-1)).sum(dim=1)


def test_uff_defaults():
    fff = BitNetUFFLayer(d_in=64, depth=10)
    assert fff.depth == 10
    assert fff.num_leaves == 1024
    assert fff.k == 8
    assert fff.n_routing_nodes == 1023
    assert fff.active_fraction == pytest.approx(8 / 1024)
    assert fff.router_weight.shape == (1023, 64)
    assert fff.leaf_weight.shape == (1024, 64, 64)
    assert fff._path_nodes.shape == (1024 * 10,)


def test_uff_validation():
    with pytest.raises(ValueError):
        BitNetUFFLayer(d_in=64, depth=-1)
    with pytest.raises(ValueError):
        BitNetUFFLayer(d_in=64, k=0)
    fff = BitNetUFFLayer(d_in=8, depth=2, k=99)
    assert fff.k == 4  # clamped to num_leaves


def test_uff_ternary_node_and_leaf_projections(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, depth=3, k=4).to(device)
    wq_r = ste_ternarize(fff.router_weight).detach()
    wq_l = ste_ternarize(fff.leaf_weight).detach()
    assert torch.isin(wq_r, torch.tensor([-1, 0, 1], device=device)).all()
    assert torch.isin(wq_l, torch.tensor([-1, 0, 1], device=device)).all()


def test_uff_shapes_and_gradient_flow(device):
    fff = BitNetUFFLayer(d_in=64, d_out=64, depth=3, k=4, bias=True).to(device)
    x = torch.randn(8, 16, 64, device=device, requires_grad=True)
    out = fff(x)
    assert out.shape == (8, 16, 64)
    out.sum().backward()
    for name, p in fff.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert p.grad.abs().sum().item() > 0, f"zero gradient for {name}"
    assert x.grad is not None and x.grad.abs().sum().item() > 0


def test_uff_flat_indexing_matches_per_leaf_reference(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, depth=4, k=8, bias=True).to(device)
    x = torch.randn(3, 16, device=device)
    with torch.no_grad():
        got = fff(x)
        ref = _manual_uff_forward(fff, x)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_uff_top1_matches_hard_routing_for_confident_router(device):
    """With an overwhelmingly confident router, top-1 == greedy hard routing.

    With every router bias strongly positive, only the all-right leaf has no
    near-zero-probability step, so both the greedy walk and the flat
    path-probability argmax must land on the last leaf.
    """
    torch.manual_seed(0)
    for depth in (2, 4, 8):
        fff = BitNetUFFLayer(d_in=16, depth=depth, k=1, bias=False).to(device)
        with torch.no_grad():
            fff.router_bias.fill_(30.0)
        x = torch.randn(8, 16, device=device)
        with torch.no_grad():
            wq_r = absmean_ternarize(fff.router_weight)
            node = torch.zeros(8, dtype=torch.long, device=device)
            for _ in range(depth):
                logit = (x * wq_r[node]).sum(-1) + fff.router_bias[node]
                node = 2 * node + 1 + (logit >= 0).to(torch.long)
            hard_leaf = node - (fff.num_leaves - 1)
            top_idx, w = fff._routing(x)
        assert torch.equal(
            top_idx.squeeze(-1), hard_leaf
        ), f"depth={depth}: top-1 leaf mismatch"
        assert torch.equal(
            top_idx.squeeze(-1), torch.full((8,), fff.num_leaves - 1, device=device)
        )
        assert torch.allclose(
            w.sum(-1), torch.ones(8, device=device)
        )  # single-leaf softmax weight is 1


def test_uff_topk_gradient_only_reaches_selected_leaf_rows(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, depth=2, k=2, bias=False).to(device)
    x = torch.randn(1, 16, device=device)
    out = fff(x)
    out.sum().backward()
    with torch.no_grad():
        top_idx, _ = fff._routing(x)
    selected = set(top_idx[0].tolist())
    assert len(selected) == fff.k
    grad = fff.leaf_weight.grad
    assert grad is not None
    for i in selected:
        assert grad[i].abs().sum().item() > 0
    others = torch.cat([grad[i] for i in range(fff.num_leaves) if i not in selected])
    assert others.abs().sum().item() == 0.0
    # router receives a differentiable signal through the softmax weights
    assert fff.router_weight.grad is not None
    assert fff.router_weight.grad.abs().sum().item() > 0


def test_uff_deep_depth_12_ultra_sparse(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, d_out=16, depth=12, k=8, bias=True).to(device)
    assert fff.num_leaves == 4096
    assert fff.active_fraction == pytest.approx(8 / 4096)
    assert fff.active_fraction < 0.002  # <0.2% of leaf capacity active
    x = torch.randn(2, 16, device=device)
    out = fff(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    out.sum().backward()
    # only the k selected rows per token receive gradients
    with torch.no_grad():
        top_idx, _ = fff._routing(x)
    grad = fff.leaf_weight.grad
    rows = grad.norm(dim=(1, 2))
    touched = rows.nonzero().squeeze(-1).tolist()
    selected = top_idx.flatten().unique().tolist()
    assert set(touched) <= set(selected)
    assert len(touched) <= 2 * fff.k


def test_uff_deep_depth_10_matches_reference(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, d_out=16, depth=10, k=8).to(device)
    x = torch.randn(2, 16, device=device)
    with torch.no_grad():
        got = fff(x)
        ref = _manual_uff_forward(fff, x)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_uff_chunked_projection_matches(device):
    torch.manual_seed(0)
    full = BitNetUFFLayer(d_in=16, depth=3, k=4, bias=True).to(device)
    chunked = BitNetUFFLayer(d_in=16, depth=3, k=4, bias=True, chunk_size=5).to(
        device
    )
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(17, 16, device=device)
    torch.testing.assert_close(chunked(x), full(x), atol=1e-6, rtol=1e-6)


def test_uff_fast_forward_matches_forward(device):
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, depth=4, k=8, bias=True).to(device).eval()
    x = torch.randn(4, 16, device=device)
    with torch.no_grad():
        torch.testing.assert_close(
            fff.fast_forward(x), fff(x), atol=1e-6, rtol=1e-6
        )


def test_uff_rank1_unsupported_shape_error():
    # UFF always uses the flat full-rank router; no router_rank argument.
    fff = BitNetUFFLayer(d_in=16, depth=2)
    assert not hasattr(fff, "router_u")


def test_uff_fp16_and_autocast(device):
    if device.type == "mps" and not is_mps_available():
        pytest.skip("MPS required")
    torch.manual_seed(0)
    fff = BitNetUFFLayer(d_in=16, depth=4, k=8, bias=True).to(device)
    x = torch.randn(4, 16, device=device)
    with torch.no_grad():
        ref = fff(x)

    fff16 = BitNetUFFLayer(d_in=16, depth=4, k=8, bias=True).to(device, torch.float16)
    fff16.load_state_dict(fff.state_dict())
    with torch.no_grad():
        out16 = fff16(x.to(device, torch.float16))
        assert torch.isfinite(out16).all()
        torch.testing.assert_close(
            out16.float(), ref, atol=5e-2, rtol=5e-2
        )

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out_ac = fff(x)
        assert torch.isfinite(out_ac).all()


def test_transformer_uses_uff_when_fff_k_set():
    torch.manual_seed(0)
    uff_cfg = BitNetFFTConfig(
        vocab_size=64, d_model=32, n_heads=2, n_layers=2,
        fff_depth=10, fff_k=8, max_seq_len=16,
    )
    m = BitNetFFTTransformer(uff_cfg)
    for layer in m.layers:
        assert isinstance(layer.fff, BitNetUFFLayer)
        assert layer.fff.k == 8
        assert layer.fff.num_leaves == 1024
    tokens = torch.randint(0, 64, (2, 8))
    logits = m(tokens)
    assert logits.shape == (2, 8, 64)
    logits.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # default (fff_k=None) still uses the classic single-leaf FFF
    classic_cfg = BitNetFFTConfig(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1, fff_depth=3,
        max_seq_len=16,
    )
    mc = BitNetFFTTransformer(classic_cfg)
    assert isinstance(mc.layers[0].fff, FastFeedForwardBitNet)


def test_transformer_uff_eval_fast_path_matches_train(device):
    if device.type == "mps" and not is_mps_available():
        pytest.skip("MPS required")
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1,
        fff_depth=8, fff_k=8, use_fast_inference=True, max_seq_len=16,
    )
    m = BitNetFFTTransformer(cfg).to(device)
    tokens = torch.randint(0, 64, (2, 8), device=device)
    m.eval()
    with torch.no_grad():
        out_fast = m(tokens)
        cfg.use_fast_inference = False
        out_plain = m(tokens)
    torch.testing.assert_close(out_fast, out_plain, atol=1e-6, rtol=1e-6)


def test_mlp_stack_uses_uff():
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(
        vocab_size=0, d_model=16, n_heads=2, n_layers=2,
        fff_depth=8, fff_k=4,
    )
    mlp = BitNetFFTMLP(cfg)
    x = torch.randn(3, 16)
    out = mlp(x)
    assert out.shape == (3, 16)
    out.sum().backward()
    for block in mlp.layers:
        assert isinstance(block.fff, BitNetUFFLayer)


def test_uff_extra_repr():
    fff = BitNetUFFLayer(d_in=32, depth=12, k=8)
    r = repr(fff)
    assert "depth=12" in r and "num_leaves=4096" in r and "k=8" in r


def test_uff_generate_deep_tree(device):
    if device.type == "mps" and not is_mps_available():
        pytest.skip("MPS required")
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1,
        fff_depth=12, fff_k=8, max_seq_len=16, use_fast_inference=True,
    )
    m = BitNetFFTTransformer(cfg).to(device)
    assert m.layers[0].fff.num_leaves == 4096
    prompt = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()
