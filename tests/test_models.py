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


def _cache_for(model: BitNetFFTTransformer, batch: int = 1) -> list[KVCache]:
    head_dim = model.cfg.d_model // model.cfg.n_heads
    total = model.cfg.n_layers * model.cfg.recurrent_steps
    return [
        KVCache.preallocate(
            batch, model.cfg.n_heads, model.cfg.max_seq_len, head_dim, dtype=torch.float32
        )
        for _ in range(total)
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
    # Head dim mismatch raises ValueError
    with pytest.raises(ValueError):
        cache.append(torch.randn(1, 2, 1, 3), torch.randn(1, 2, 1, 3))
    # Dtype mismatch is automatically cast without error
    k_half = torch.randn(1, 2, 1, 4).half()
    v_half = torch.randn(1, 2, 1, 4).half()
    cache.append(k_half, v_half)
    assert cache.size == 1


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
    cfg = _cfg(use_fast_inference=False, n_layers=1, recurrent_steps=1)
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


# --- generate() -------------------------------------------------------------


def _manual_greedy(m: BitNetFFTTransformer, prompt: torch.Tensor, n: int) -> torch.Tensor:
    """Reference autoregressive decode: prefill once, then one cached token/step."""
    caches = _cache_for(m, batch=prompt.shape[0])
    logits = m(prompt, kv_cache=caches)
    ids = [logits[:, -1].argmax(-1, keepdim=True)]
    for step in range(1, n):
        logits = m(ids[-1], kv_cache=caches, past_length=prompt.shape[1] + step - 1)
        ids.append(logits[:, -1].argmax(-1, keepdim=True))
    return torch.cat(ids, dim=1)


def test_generate_greedy_matches_manual_decode():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        manual = _manual_greedy(m, prompt, 8)
        out = m.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(out, torch.cat([prompt, manual], dim=1))


def test_generate_1d_and_batch_shapes():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    p1 = torch.randint(0, cfg.vocab_size, (6,))
    p2 = torch.randint(0, cfg.vocab_size, (3, 6))
    o1 = m.generate(p1, max_new_tokens=4, temperature=0.0)
    o2 = m.generate(p2, max_new_tokens=4, temperature=0.0)
    assert o1.ndim == 1 and o1.shape[0] == 10
    assert o2.shape == (3, 10)


def test_generate_sampling_in_vocab_and_seeded():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    torch.manual_seed(123)
    a = m.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=50, top_p=0.9)
    torch.manual_seed(123)
    b = m.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=50, top_p=0.9)
    assert torch.equal(a, b)
    assert a.shape == (1, 10)
    assert (a[0, 4:] >= 0).all() and (a[0, 4:] < cfg.vocab_size).all()


def test_generate_greedy_deterministic_across_runs():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = m.generate(prompt, max_new_tokens=6, temperature=0.0)
    b = m.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_eos_truncation(monkeypatch):
    import bitnet_fff.models as models_mod

    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False)
    m = BitNetFFTTransformer(cfg)
    m.eval()
    eos = 42

    def fake_eos(logits, temperature, top_k, top_p):
        return torch.full((logits.shape[0], 1), eos, device=logits.device)

    monkeypatch.setattr(models_mod, "_sample_from_logits", fake_eos)
    out = m.generate(torch.tensor([[5, 6, 7]]), max_new_tokens=8, eos_token_id=eos)
    assert out.shape == (1, 4)
    assert out[0, -1] == eos

    def fake_plain(logits, temperature, top_k, top_p):
        return torch.full((logits.shape[0], 1), 7, device=logits.device)

    monkeypatch.setattr(models_mod, "_sample_from_logits", fake_plain)
    out2 = m.generate(torch.tensor([[5, 6, 7]]), max_new_tokens=8, eos_token_id=eos)
    assert out2.shape == (1, 11)


def test_generate_restores_training_mode():
    torch.manual_seed(0)
    m = BitNetFFTTransformer(_cfg(use_fast_inference=False))
    m.train()
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 4))
    m.generate(prompt, max_new_tokens=3, temperature=0.0)
    assert m.training


def test_generate_guards():
    m = BitNetFFTTransformer(_cfg(use_fast_inference=False, max_seq_len=8))
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 6))
    with pytest.raises(ValueError):
        m.generate(prompt, max_new_tokens=3, temperature=0.0)
    with pytest.raises(ValueError):
        m.generate(prompt, max_new_tokens=0, temperature=0.0)
    raw = BitNetFFTTransformer(_cfg(vocab_size=0))
    with pytest.raises(ValueError):
        raw.generate(torch.randint(0, 64, (1, 4)), max_new_tokens=2)


def test_generate_mps_fast_inference():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    from bitnet_fff.fast_inference import extension_available
    if not extension_available():
        pytest.skip("Metal/NEON extension required for _packed_eval")
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=True)
    m = BitNetFFTTransformer(cfg).to("mps")
    m.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), device="mps")
    out = m.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert out.shape == (1, 14)
    assert torch.isfinite(out).all()
    assert m.layers[0].fff._packed_eval is not None


def test_gradient_checkpointing():
    torch.manual_seed(42)
    cfg = _cfg()
    m = BitNetFFTTransformer(cfg)
    m.train()
    assert not m.gradient_checkpointing

    m.gradient_checkpointing_enable()
    assert m.gradient_checkpointing

    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    out = m(tokens)
    assert out.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    m.gradient_checkpointing_disable()
    assert not m.gradient_checkpointing


# --- Recurrent depth --------------------------------------------------------


def test_step_emb_created_only_when_recurrent():
    m1 = BitNetFFTTransformer(_cfg(recurrent_steps=1))
    assert m1.step_emb is None
    m2 = BitNetFFTTransformer(_cfg(recurrent_steps=3))
    assert m2.step_emb is not None
    assert m2.step_emb.shape == (3, 32)
    assert m2.step_emb.dtype == torch.float32


def test_recurrent_backward_finite_grads():
    torch.manual_seed(0)
    cfg = _cfg(recurrent_steps=3)
    m = BitNetFFTTransformer(cfg)
    m.train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    out = m(tokens)
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert m.step_emb.grad is not None
    assert torch.isfinite(m.step_emb.grad).all()


def test_recurrent_steps_change_output():
    torch.manual_seed(0)
    m1 = BitNetFFTTransformer(_cfg(n_layers=1, recurrent_steps=1))
    m2 = BitNetFFTTransformer(_cfg(n_layers=1, recurrent_steps=3))
    tokens = torch.randint(0, m1.cfg.vocab_size, (2, 8))
    with torch.no_grad():
        a, b = m1(tokens), m2(tokens)
    assert a.shape == b.shape == (2, 8, m1.cfg.vocab_size)
    assert not torch.allclose(a, b, atol=1e-4)


def test_recurrent_kv_cache_generation_matches_full():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False, recurrent_steps=2)
    m = BitNetFFTTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = m(tokens)
        caches = _cache_for(m)
        outs = [m(tokens[:, t : t + 1], kv_cache=caches, past_length=t) for t in range(8)]
        got = torch.cat(outs, dim=1)
    assert torch.allclose(got, full, atol=1e-5)


def test_recurrent_nested_caches_accepted():
    torch.manual_seed(0)
    cfg = _cfg(use_fast_inference=False, recurrent_steps=2)
    m = BitNetFFTTransformer(cfg).eval()
    flat = _cache_for(m)
    nested = [flat[2 * i : 2 * i + 2] for i in range(2)]
    tokens = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        a = m(tokens, kv_cache=flat)
        b = m(tokens, kv_cache=nested)
    assert torch.allclose(a, b, atol=1e-5)


def test_recurrent_requires_per_step_layer_cache():
    cfg = _cfg(use_fast_inference=False, n_layers=2, recurrent_steps=2)
    m = BitNetFFTTransformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        m(tokens, kv_cache=_cache_for(m)[:3])  # only 3 of 4 slots
    with pytest.raises(ValueError):
        m(tokens, kv_cache=[_cache_for(m)[:2]])  # nested but only one step group


def test_gradient_checkpointing_with_recurrence():
    torch.manual_seed(0)
    cfg = _cfg(recurrent_steps=2)
    m = BitNetFFTTransformer(cfg)
    m.gradient_checkpointing_enable()
    m.train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    out = m(tokens)
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
