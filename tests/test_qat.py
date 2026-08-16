"""Tests for QAT: dynamic activation scaling, FP16 master weights, training loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bitnet_fff.bitlinear import absmean_ternarize
from bitnet_fff.qat import ActivationQuantizer, BitNetQAT, FP16MasterAdamW

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.mark.parametrize("mode", ["absmax", "per_channel", "ema", "learned"])
def test_activation_quantizer_modes(mode):
    torch.manual_seed(0)
    aq = ActivationQuantizer(bits=8, mode=mode).to(DEVICE)
    x = torch.randn(4, 16, requires_grad=True, device=DEVICE)
    y = aq(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert aq.last_scale is not None and float(aq.last_scale.detach().cpu()) > 0


def test_activation_quantizer_disable():
    aq = ActivationQuantizer(bits=32)
    x = torch.randn(4, 8)
    out = aq(x)
    assert torch.equal(out, x)


def test_quantized_values_in_range():
    aq = ActivationQuantizer(bits=4).to(DEVICE)
    x = torch.randn(100, 32, device=DEVICE)
    y = aq(x)
    max_q = 2 ** 3 - 1
    assert float(y.abs().max()) <= float(x.abs().max())


def test_fp16_master_step_keeps_dtype_and_trains():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, device=DEVICE))
    opt = FP16MasterAdamW([p], lr=1e-2)
    assert p.dtype == torch.float16
    for _ in range(10):
        opt.zero_grad()
        (p.float().square().sum()).backward()
        assert p.grad.dtype == torch.float16
        opt.step()
    assert p.dtype == torch.float16
    assert torch.isfinite(p.float()).all()


def test_fp16_master_state_dict_roundtrip():
    p = torch.nn.Parameter(torch.randn(8, device=DEVICE))
    opt = FP16MasterAdamW([p], lr=1e-3)
    opt.zero_grad()
    p.grad = torch.ones_like(p)
    opt.step()
    sd = opt.state_dict()
    p2 = torch.nn.Parameter(torch.randn(8, device=DEVICE))
    opt2 = FP16MasterAdamW([p2], lr=1e-3)
    opt2.load_state_dict(sd)
    assert opt2.steps == sd["steps"]


def test_fp16_master_weight_decay_added_in_place():
    torch.manual_seed(0)
    wd = 0.1
    p = torch.nn.Parameter(torch.ones(4, device=DEVICE))
    opt = FP16MasterAdamW([p], lr=1e-3, weight_decay=wd)
    assert p.dtype == torch.float16
    opt.zero_grad()
    p.grad = torch.ones_like(p)
    opt.step()

    # first-step exp_avg = (1 - beta1) * decayed_grad, decayed_grad = 1 + wd * 1
    st = opt.state[p]
    expected = (1 - 0.9) * (1.0 + wd * 1.0)
    torch.testing.assert_close(
        st["exp_avg"], torch.full_like(st["exp_avg"], expected),
        atol=1e-3, rtol=1e-3,
    )
    # the fp16 grad is untouched (decay mutated only the promoted FP32 copy)
    torch.testing.assert_close(p.grad, torch.ones_like(p))
    assert p.dtype == torch.float16 and torch.isfinite(p.float()).all()


def test_fp16_master_zero_weight_decay_skips_add():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, device=DEVICE))
    opt = FP16MasterAdamW([p], lr=1e-3, weight_decay=0.0)
    opt.zero_grad()
    p.grad = torch.randn_like(p)
    opt.step()
    g = p.grad.detach().to(torch.float32)
    torch.testing.assert_close(opt.state[p]["exp_avg"], (1 - 0.9) * g, atol=1e-4, rtol=1e-4)


def test_qat_linear_ternarizes_weights():
    lin = torch.nn.Linear(16, 32).to(DEVICE)
    wq = BitNetQAT(lin, activation_bits=16)
    x = torch.randn(4, 16, device=DEVICE)
    out = wq(x)
    assert out.shape == (4, 32)
    w_tern = wq.quantized_weight()
    assert set(torch.unique(w_tern).tolist()) <= {-1.0, 0.0, 1.0}
    out.sum().backward()
    assert lin.weight.grad is not None


def test_qat_transformer_trains_and_lower_loss():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer

    torch.manual_seed(1)
    m = BitNetFFTTransformer(
        BitNetFFTConfig(vocab_size=256, d_model=32, n_heads=2, n_layers=1, fff_depth=2)
    ).to("mps")
    w = BitNetQAT(m, activation_bits=8)
    w.enable_fp16_master()
    assert sum(p.is_floating_point() and p.dtype == torch.float16 for p in m.parameters())
    opt = w.optimizer(lr=1e-2)
    tokens = torch.randint(0, 256, (4, 16), device="mps")
    target = torch.zeros(4, 16, 256, device="mps")
    target.scatter_(-1, tokens.unsqueeze(-1), 1.0)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss = (-w(tokens).float().log_softmax(-1) * target).sum(-1).mean()
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


def test_absmean_threshold_scale_changes_density():
    w = torch.tensor([[1.0, 0.5, 0.1, 0.05]])
    dense = absmean_ternarize(w, threshold_scale=0.5)
    sparse = absmean_ternarize(w, threshold_scale=2.0)
    assert int(dense.count_nonzero()) >= int(sparse.count_nonzero())
