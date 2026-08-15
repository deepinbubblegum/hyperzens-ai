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
    KVCache,
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


# --- KV-cache / autoregressive generation ----------------------------------


def _cache_for(model: BitNetFFTTransformer) -> list[KVCache]:
    head_dim = model.cfg.d_model // model.cfg.n_heads
    return [
        KVCache.preallocate(
            1, model.cfg.n_heads, model.cfg.max_seq_len, head_dim, dtype=torch.float32
        )
        for _ in range(model.cfg.n_layers)
    ]


def test_kvcache_preallocate_and_append():
    cache = KVCache.preallocate(2, 4, 16, 8, dtype=torch.float32)
    assert cache.size == 0
    assert cache.key_cache.shape == (2, 4, 16, 8)
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    cache.append(k, v)
    assert cache.size == 3
    assert torch.equal(cache.key_cache[:, :, :3], k)
    assert torch.equal(cache.value_cache[:, :, :3], v)
    k2 = torch.randn(2, 4, 1, 8)
    v2 = torch.randn(2, 4, 1, 8)
    cache.append(k2, v2, past_length=3)
    assert cache.size == 4
    assert torch.equal(cache.key_cache[:, :, 3:4], k2)


def test_kvcache_dynamic_growth():
    cache = KVCache(torch.empty(2, 2, 2, 4), torch.empty(2, 2, 2, 4))
    k = torch.randn(2, 2, 1, 4)
    v = torch.randn(2, 2, 1, 4)
    for _ in range(5):
        cache.append(k, v, past_length=cache.size)
    assert cache.size == 5
    assert cache.key_cache.shape[-2] >= 5
    assert torch.allclose(cache.key_cache[:, :, :5], k.expand(2, 2, 5, 4))


def test_kvcache_mismatch_raises():
    cache = KVCache.preallocate(1, 2, 8, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        cache.append(torch.randn(1, 2, 1, 3), torch.randn(1, 2, 1, 3))
    with pytest.raises(ValueError):
        cache.append(torch.randn(1, 2, 1, 4).half(), torch.randn(1, 2, 1, 4).half())


def test_attention_kv_cache_matches_full():
    torch.manual_seed(0)
    attn = BitNetAttention(16, 2, activation_bits=8)
    x0 = torch.randn(2, 4, 16)
    x1 = torch.randn(2, 6, 16)
    x_all = torch.cat([x0, x1], dim=1)
    with torch.no_grad():
        ref = attn(x_all, mask=torch.triu(torch.full((10, 10), float("-inf")), diagonal=1))
        cache = KVCache.preallocate(2, 2, 16, 8)
        out0 = attn(x0, kv_cache=cache)
        out1 = attn(x1, kv_cache=cache, past_length=4)
        got = torch.cat([out0, out1], dim=1)
    assert torch.allclose(got, ref, atol=1e-5)


def test_attention_kv_cache_no_recompute_quantization():
    torch.manual_seed(0)
    attn = BitNetAttention(16, 2, activation_bits=8)
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        ref = attn(x, mask=torch.triu(torch.full((8, 8), float("-inf")), diagonal=1))
        cache = KVCache.preallocate(2, 2, 8, 8)
        outs = []
        for t in range(8):
            outs.append(attn(x[:, t : t + 1], kv_cache=cache, past_length=t))
        got = torch.cat(outs, dim=1)
    assert torch.allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_attention_kv_cache_dtype(dtype):
    if dtype == torch.float16 and not torch.backends.mps.is_available():
        pytest.skip("MPS required for fp16")
    torch.manual_seed(0)
    dev = "mps" if dtype == torch.float16 else "cpu"
    attn = BitNetAttention(16, 2, activation_bits=8, eps=1e-5).to(dev).to(dtype)
    x = torch.randn(1, 8, 16, dtype=dtype, device=dev)
    with torch.no_grad():
        ref = attn(x, mask=torch.triu(torch.full((8, 8), float("-inf"), device=dev), diagonal=1))
        cache = KVCache.preallocate(1, 2, 8, 8, dtype=dtype, device=dev)
        outs = []
        for t in range(8):
            outs.append(attn(x[:, t : t + 1], kv_cache=cache, past_length=t))
        got = torch.cat(outs, dim=1)
    assert torch.allclose(got, ref, atol=5e-3)


def test_transformer_kv_cache_generation_matches_full():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = m(tokens)
        caches = _cache_for(m)
        outs = [
            m(tokens[:, t : t + 1], kv_cache=caches, past_length=t)
            for t in range(8)
        ]
        got = torch.cat(outs, dim=1)
    assert torch.allclose(got, full, atol=1e-5)


def test_transformer_kv_cache_single_layer_broadcast():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False, n_layers=1)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = m(tokens)
        cache = _cache_for(m)[0]
        outs = [m(tokens[:, t : t + 1], kv_cache=cache, past_length=t) for t in range(8)]
        got = torch.cat(outs, dim=1)
    assert torch.allclose(got, full, atol=1e-5)


def test_transformer_kv_cache_requires_per_layer():
    cfg = _cfg(use_fast_inference=False, n_layers=2)
    m = BitNetFFTTransformer(cfg)
    cache = _cache_for(m)[0]
    with pytest.raises(ValueError):
        m(torch.randint(0, cfg.vocab_size, (1, 4)), kv_cache=cache)
    with pytest.raises(ValueError):
        m(torch.randint(0, cfg.vocab_size, (1, 4)), kv_cache=_cache_for(m)[:1])


def test_block_kv_cache_shares_storage():
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    cache = KVCache.preallocate(1, cfg.n_heads, cfg.max_seq_len, cfg.d_model // cfg.n_heads)
    m.layers[0](torch.randn(1, 3, cfg.d_model), kv_cache=cache)
    assert cache.size == 3
    ref = cache.key_cache[..., :3, :].clone()
    m.layers[0](torch.randn(1, 2, cfg.d_model), kv_cache=cache, past_length=3)
    assert cache.size == 5
    assert torch.equal(cache.key_cache[..., :3, :], ref)
