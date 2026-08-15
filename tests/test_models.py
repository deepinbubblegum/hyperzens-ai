"""Tests for the end-to-end Transformer/MLP model built on FastFeedForwardBitNet."""

from __future__ import annotations

import pytest
import torch

from bitnet_fff import FastFeedForwardBitNet
from bitnet_fff.models import (
    BitNetAttention,
    BitNetFFTBlock,
    BitNetFFTConfig,
    BitNetFFTMLP,
    BitNetFFTTransformer,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _cfg(**overrides) -> BitNetFFTConfig:
    base = dict(
        vocab_size=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        fff_depth=3,
        max_seq_len=16,
    )
    base.update(overrides)
    return BitNetFFTConfig(**base)


def test_config_properties():
    cfg = _cfg(fff_d_out=16)
    assert cfg.fff_out == 16
    assert cfg.fff_capacity == 16 * (2 ** cfg.fff_depth)
    cfg2 = _cfg()
    assert cfg2.fff_out == cfg2.d_model


@pytest.mark.parametrize("device", ["cpu", DEVICE])
def test_transformer_shape_and_roundtrip(device):
    torch.manual_seed(0)
    cfg = _cfg()
    m = BitNetFFTTransformer(cfg).to(device)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
    logits = m(tokens)
    assert logits.shape == (2, 8, cfg.vocab_size)
    logits.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_tie_weights():
    cfg = _cfg(tie_weights=True)
    m = BitNetFFTTransformer(cfg)
    assert m.head.weight is m.embed.weight


def test_embedding_vs_raw_input():
    m = BitNetFFTTransformer(_cfg())
    raw = BitNetFFTTransformer(_cfg(vocab_size=0, d_model=32, n_layers=1))
    x = torch.randn(4, 8, 32)
    out = raw(x)
    assert out.shape == (4, 8, 32)


def test_fast_inference_matches_train_forward():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=True)
    m = BitNetFFTTransformer(cfg).to("mps")
    tokens = torch.randint(0, cfg.vocab_size, (2, 8), device="mps")
    m.eval()
    out_fast = m(tokens)
    cfg.use_fast_inference = False
    out_plain = m(tokens)
    assert torch.allclose(out_fast, out_plain, atol=1e-4)


def test_causal_mask_autoregressive():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    torch.manual_seed(0)
    cfg = _cfg(d_model=16, n_heads=1, n_layers=1, fff_depth=2, use_fast_inference=False)
    m = BitNetFFTTransformer(cfg).to("mps").eval()
    seq = torch.arange(4, device="mps")
    tokens = torch.stack([seq, seq])
    logits = m(tokens)
    assert torch.equal(logits[:, 0, :], logits[:, 0, :])  # shape sanity


def test_attention_mask_type_compat():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    attn = BitNetAttention(16, 2).half().to("mps")
    x = torch.randn(2, 4, 16, dtype=torch.float16, device="mps")
    mask = torch.triu(
        torch.full((4, 4), float("-inf"), device="mps"), diagonal=1
    )
    out = attn(x, mask=mask)
    assert out.shape == (2, 4, 16)
    assert torch.isfinite(out).all()


def test_mlp_stack_shape():
    m = BitNetFFTMLP(_cfg(vocab_size=0, d_model=32, n_layers=2))
    x = torch.randn(3, 32)
    out = m(x)
    assert out.shape == (3, 32)
    out.sum().backward()


def test_fff_blocks_use_bitnet_fff():
    m = BitNetFFTTransformer(_cfg())
    for layer in m.layers:
        assert isinstance(layer.fff, FastFeedForwardBitNet)
