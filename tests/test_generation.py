"""KV-cache equivalence, generation stopping conditions, and CLI smoke tests.

Covers the two contracts the generation stack must guarantee:

* **KV-cache equivalence** — logits computed one token at a time through the
  cache (both the prefill pass and the stepwise decode pass) match the logits
  of a single full un-cached forward pass to fp32 tolerance.
* **Stopping conditions** — generation truncates at the first ``eos_token_id``
  and never exceeds ``max_new_tokens``.

Plus a subprocess smoke test of ``scripts/generate.py`` and the byte-level
tokenizer fallback.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from bitnet_fff.models import (
    BitNetFFTConfig,
    BitNetFFTTransformer,
    KVCache,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _caches(model: BitNetFFTTransformer, batch: int = 1) -> list[KVCache]:
    head_dim = model.cfg.d_model // model.cfg.n_heads
    total = model.cfg.n_layers * model.cfg.recurrent_steps
    return [
        KVCache.preallocate(batch, model.cfg.n_heads, model.cfg.max_seq_len, head_dim)
        for _ in range(total)
    ]


def _model(**cfg_overrides) -> BitNetFFTTransformer:
    torch.manual_seed(0)
    m = BitNetFFTTransformer(_cfg(use_fast_inference=False, **cfg_overrides))
    m.eval()
    return m


# --- KV-cache equivalence ----------------------------------------------------


def test_prefill_logits_match_full_forward():
    m = _model()
    ctx = torch.randint(0, m.cfg.vocab_size, (2, 8))
    with torch.no_grad():
        full = m(ctx)
        caches = _caches(m, batch=2)
        prefill = m(ctx, kv_cache=caches)
    assert torch.allclose(prefill, full, atol=1e-5)
    assert [c.size for c in caches] == [8] * (m.cfg.n_layers * m.cfg.recurrent_steps)


def test_stepwise_decode_logits_match_full_forward():
    m = _model()
    ctx = torch.randint(0, m.cfg.vocab_size, (2, 8))
    with torch.no_grad():
        full = m(ctx)
        caches = _caches(m, batch=2)
        rows = [
            m(ctx[:, t : t + 1], kv_cache=caches, past_length=t)
            for t in range(8)
        ]
        got = torch.cat(rows, dim=1)
    assert torch.allclose(got, full, atol=1e-5)
    assert [c.size for c in caches] == [8] * (m.cfg.n_layers * m.cfg.recurrent_steps)


def test_generate_matches_stepwise_cached_decode():
    """generate()'s prefill+decode reproduces a manual cached decode exactly."""
    m = _model()
    prompt = torch.randint(0, m.cfg.vocab_size, (2, 5))
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=0.0)
        # manual reference
        caches = _caches(m, batch=2)
        logits = m(prompt, kv_cache=caches)
        ids = [logits[:, -1].argmax(-1, keepdim=True)]
        for step in range(1, 8):
            logits = m(ids[-1], kv_cache=caches, past_length=5 + step - 1)
            ids.append(logits[:, -1].argmax(-1, keepdim=True))
        manual = torch.cat([prompt, torch.cat(ids, dim=1)], dim=1)
    assert torch.equal(out, manual)


def test_recurrent_generate_matches_stepwise_cached_decode():
    """Recurrent generate() reproduces a manual cached decode over the full
    R*n_layers cache layout."""
    m = _model(recurrent_steps=2)
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=0.0)
        caches = _caches(m, batch=1)
        logits = m(prompt, kv_cache=caches)
        ids = [logits[:, -1].argmax(-1, keepdim=True)]
        for step in range(1, 8):
            logits = m(ids[-1], kv_cache=caches, past_length=5 + step - 1)
            ids.append(logits[:, -1].argmax(-1, keepdim=True))
        manual = torch.cat([prompt, torch.cat(ids, dim=1)], dim=1)
    assert torch.equal(out, manual)


def test_recurrent_stream_generate_matches_greedy():
    m = _model(recurrent_steps=2)
    prompt = torch.randint(0, m.cfg.vocab_size, (5,))
    with torch.no_grad():
        full = m.generate(prompt, max_new_tokens=8, temperature=0.0)
        ids = list(m.stream_generate(prompt, max_new_tokens=8, temperature=0.0))
    assert ids == full[5:].tolist()


# --- stopping conditions -----------------------------------------------------


def _patch_sampler(monkeypatch, sequence: list[int]):
    import bitnet_fff.models as models_mod

    state = {"i": 0}

    def sampler(logits, temperature, top_k, top_p):
        val = sequence[min(state["i"], len(sequence) - 1)]
        state["i"] += 1
        return torch.full((logits.shape[0], 1), val, device=logits.device)

    monkeypatch.setattr(models_mod, "_sample_from_logits", sampler)


def test_generation_stops_at_eos(monkeypatch):
    m = _model()
    eos = 42
    # first two generated tokens are plain (7), third is eos
    _patch_sampler(monkeypatch, [7, 7, eos, 7, 7, 7, 7, 7])
    prompt = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=1.0, eos_token_id=eos)
    assert out.shape == (1, 3 + 3)
    assert out[0, -1] == eos
    assert (out[0, 3:6] == torch.tensor([7, 7, eos])).all()


def test_generation_truncates_at_immediate_eos(monkeypatch):
    m = _model()
    _patch_sampler(monkeypatch, [42])
    prompt = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=1.0, eos_token_id=42)
    assert out.shape == (1, 4)
    assert out[0, -1] == 42


def test_generation_respects_max_new_tokens_no_eos(monkeypatch):
    m = _model()
    # never emit eos -> must run the full budget
    _patch_sampler(monkeypatch, [7])
    prompt = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=1.0, eos_token_id=42)
    assert out.shape == (1, 3 + 8)
    assert (out[0, 3:] == 7).all()


def test_generation_respects_max_new_tokens_no_eos_arg():
    m = _model()
    prompt = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert out.shape == (1, 3 + 8)


def test_generation_shorter_than_max_new_tokens_with_eos(monkeypatch):
    """Batch where one row hits eos early and another runs to the budget."""
    m = _model()
    eos = 42

    def sampler(logits, temperature, top_k, top_p):
        b = logits.shape[0]
        vals = torch.full((b, 1), 7, device=logits.device)
        vals[0] = eos
        return vals

    import bitnet_fff.models as models_mod

    monkeypatch.setattr(models_mod, "_sample_from_logits", sampler)
    prompt = torch.tensor([[5, 6, 7], [8, 9, 10]])
    with torch.no_grad():
        out = m.generate(prompt, max_new_tokens=5, temperature=1.0, eos_token_id=eos)
    assert out.shape[0] == 2
    assert out[0, -1] == eos
    assert out[1].shape[0] == 3 + 5


# --- stream_generate (typewriter streaming) ----------------------------------


def test_stream_generate_matches_generate_greedy():
    """stream_generate yields exactly the tokens generate() produces."""
    m = _model()
    prompt = torch.randint(0, m.cfg.vocab_size, (5,))
    with torch.no_grad():
        full = m.generate(prompt, max_new_tokens=8, temperature=0.0)
        ids = list(m.stream_generate(prompt, max_new_tokens=8, temperature=0.0))
    assert ids == full[5:].tolist()


def test_stream_generate_stops_at_eos(monkeypatch):
    m = _model()
    _patch_sampler(monkeypatch, [7, 7, 42, 7, 7, 7, 7, 7])
    prompt = torch.tensor([5, 6, 7])
    with torch.no_grad():
        ids = list(m.stream_generate(
            prompt, max_new_tokens=8, temperature=1.0, eos_token_id=42
        ))
    assert ids == [7, 7, 42]


def test_stream_generate_truncates_at_immediate_eos(monkeypatch):
    m = _model()
    _patch_sampler(monkeypatch, [42])
    prompt = torch.tensor([5, 6, 7])
    with torch.no_grad():
        ids = list(m.stream_generate(
            prompt, max_new_tokens=8, temperature=1.0, eos_token_id=42
        ))
    assert ids == [42]


def test_stream_generate_respects_max_new_tokens_no_eos(monkeypatch):
    m = _model()
    _patch_sampler(monkeypatch, [7])
    prompt = torch.tensor([5, 6, 7])
    with torch.no_grad():
        ids = list(m.stream_generate(
            prompt, max_new_tokens=5, temperature=1.0, eos_token_id=42
        ))
    assert ids == [7] * 5


def test_stream_generate_decodes_text_pieces(monkeypatch):
    m = _model()
    _patch_sampler(monkeypatch, [40, 41, 33])
    prompt = torch.tensor([5, 6])
    with torch.no_grad():
        pieces = list(m.stream_generate(
            prompt, max_new_tokens=3, temperature=1.0, decode_token=chr
        ))
    assert pieces == ["(", ")", "!"]


def test_stream_generate_batch_returns_per_row_ids(monkeypatch):
    m = _model()

    def sampler(logits, temperature, top_k, top_p):
        b = logits.shape[0]
        return torch.full((b, 1), 7, device=logits.device)

    import bitnet_fff.models as models_mod

    monkeypatch.setattr(models_mod, "_sample_from_logits", sampler)
    prompt = torch.tensor([[5, 6, 7], [8, 9, 10]])
    with torch.no_grad():
        steps = list(m.stream_generate(
            prompt, max_new_tokens=4, temperature=1.0
        ))
    assert len(steps) == 4
    assert all(s.shape == (2,) and (s == 7).all() for s in steps)


def test_stream_generate_close_restores_training_state():
    m = _model()
    m.train()
    assert m.training
    prompt = torch.tensor([5, 6, 7])
    with torch.no_grad():
        stream = m.stream_generate(prompt, max_new_tokens=8, temperature=0.0)
        next(stream)
        assert not m.training  # eval() active while streaming
        stream.close()
    assert m.training  # finally restores the original mode


def test_stream_generate_requires_token_model():
    m = BitNetFFTTransformer(_cfg(vocab_size=0))
    m.eval()
    prompt = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError):
        next(m.stream_generate(prompt, max_new_tokens=2))


# --- repetition penalty ------------------------------------------------------


def test_apply_repetition_penalty_biases_away_from_sampled():
    from bitnet_fff.models import _apply_repetition_penalty

    logits = torch.tensor([[5.0, 3.0, 1.0, -2.0]])
    penalized = _apply_repetition_penalty(
        logits.clone(), torch.tensor([[1, 3]]), 2.0
    )
    # id 1 (logit > 0) is halved; id 3 (logit < 0) is doubled in magnitude.
    assert torch.equal(penalized, torch.tensor([[5.0, 1.5, 1.0, -4.0]]))
    assert torch.equal(logits, torch.tensor([[5.0, 3.0, 1.0, -2.0]]))  # no in-place


def test_apply_repetition_penalty_noop_when_disabled():
    from bitnet_fff.models import _apply_repetition_penalty

    logits = torch.tensor([[5.0, 3.0]])
    assert torch.equal(
        _apply_repetition_penalty(logits.clone(), torch.tensor([[0, 1]]), 1.0),
        logits,
    )
    assert torch.equal(
        _apply_repetition_penalty(logits.clone(), torch.empty(0, 0, dtype=torch.long), 2.0),
        logits,
    )


def test_stream_generate_applies_repetition_penalty(monkeypatch):
    import bitnet_fff.models as models_mod

    m = _model()
    calls = []

    def fake_penalty(logits, prev_ids, penalty):
        calls.append((prev_ids.shape[1], penalty))
        return logits

    monkeypatch.setattr(models_mod, "_apply_repetition_penalty", fake_penalty)
    prompt = torch.tensor([1, 2, 3])
    with torch.no_grad():
        steps = list(m.stream_generate(
            prompt, max_new_tokens=3, temperature=1.0, repetition_penalty=1.5
        ))
    assert len(steps) == 3
    # one penalty application per decode step, history grows each time
    assert calls == [(1, 1.5), (2, 1.5)]


def test_stream_generate_skips_penalty_when_disabled(monkeypatch):
    import bitnet_fff.models as models_mod

    m = _model()
    calls = []

    def fake_penalty(logits, prev_ids, penalty):
        calls.append(penalty)
        return logits

    monkeypatch.setattr(models_mod, "_apply_repetition_penalty", fake_penalty)
    prompt = torch.tensor([1, 2, 3])
    with torch.no_grad():
        steps = list(m.stream_generate(
            prompt, max_new_tokens=3, temperature=1.0, repetition_penalty=1.0
        ))
    assert len(steps) == 3
    assert calls == []


# --- sampling robustness -----------------------------------------------------


def test_sample_from_logits_survives_nonfinite_logits():
    from bitnet_fff.models import _sample_from_logits

    logits = torch.tensor([[float("nan"), 1.0, float("inf"), -float("inf"), 2.0]])
    ids = _sample_from_logits(logits, temperature=1.0)
    assert ids.ndim == 2 and ids.shape[1] == 1
    assert 0 <= ids.min().item() <= ids.max().item() < logits.shape[-1]


def test_sample_from_logits_tiny_temperature_no_overflow():
    from bitnet_fff.models import _sample_from_logits

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    ids = _sample_from_logits(logits, temperature=1e-300)
    assert 0 <= ids.min().item() <= ids.max().item() < logits.shape[-1]


def test_sample_from_logits_nan_temperature_is_greedy():
    from bitnet_fff.models import _sample_from_logits

    logits = torch.tensor([[1.0, 5.0, 2.0]])
    ids = _sample_from_logits(logits, temperature=float("nan"))
    assert ids[0, 0].item() == 1  # argmax of [1, 5, 2]


def test_sample_from_logits_clamps_ids_to_vocab(monkeypatch):
    from bitnet_fff.models import _sample_from_logits

    logits = torch.tensor([[1.0, 2.0, 3.0]])

    def fake_multinomial(probs, num_samples):
        return torch.tensor([[logits.shape[-1] + 5]])  # deliberately OOB

    monkeypatch.setattr("torch.multinomial", fake_multinomial)
    ids = _sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None)
    assert ids.max().item() == logits.shape[-1] - 1


def test_apply_repetition_penalty_ignores_oob_ids():
    from bitnet_fff.models import _apply_repetition_penalty

    logits = torch.tensor([[5.0, 3.0, 1.0]])
    penalized = _apply_repetition_penalty(
        logits.clone(), torch.tensor([[1, 99, -3]]), 2.0
    )
    assert torch.equal(penalized, torch.tensor([[5.0, 1.5, 1.0]]))


def test_stream_generate_rejects_oob_prompt_ids():
    m = _model(vocab_size=64)
    with pytest.raises(ValueError, match="must be in"):
        list(m.stream_generate(torch.tensor([5, 999]), max_new_tokens=2))


def test_generate_rejects_oob_prompt_ids():
    m = _model(vocab_size=64)
    with pytest.raises(ValueError, match="must be in"):
        m.generate(torch.tensor([5, -1]), max_new_tokens=2)


# --- CLI smoke tests ---------------------------------------------------------


def _load_script_module():
    path = os.path.join(REPO, "scripts", "generate.py")
    spec = importlib.util.spec_from_file_location("generate_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_byte_tokenizer_fallback_roundtrip():
    mod = _load_script_module()
    tok = mod.load_tokenizer("bytes", 256)
    assert isinstance(tok, mod.ByteTokenizer)
    ids = tok.encode("héllo wörld")
    assert tok.decode(ids) == "héllo wörld"
    assert all(0 <= i < 256 for i in ids)


def test_byte_tokenizer_vocab_too_small_raises():
    mod = _load_script_module()
    tok = mod.load_tokenizer("bytes", 64)
    with pytest.raises(ValueError):
        tok.encode("hello world")  # 'w' == byte 119 >= 64


def test_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/generate.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
    assert "--max-new-tokens" in proc.stdout


def test_cli_generates_and_reports_decode_benchmark():
    proc = subprocess.run(
        [
            sys.executable, "scripts/generate.py",
            "--device", "cpu",
            "--tokenizer", "bytes",
            "--d-model", "32", "--n-heads", "2", "--n-layers", "1",
            "--fff-depth", "2", "--vocab-size", "256",
            "--max-seq-len", "48",
            "--max-new-tokens", "6",
            "--bench-iters", "10",
            "--temperature", "0.0",
            "--prompt", "hi there",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "latency" in proc.stdout and "ms/token" in proc.stdout
    assert "speed" in proc.stdout and "tokens/sec" in proc.stdout
    assert "generated 6/6 tokens" in proc.stdout


def test_kv_cache_dtype_mismatch_auto_adoption():
    cache = KVCache.preallocate(batch_size=2, n_heads=2, seq_len=16, head_dim=8, dtype=torch.float32)
    assert cache.dtype == torch.float32
    assert cache.size == 0

    # Append FP16 keys/values to empty float32 cache -> cache dynamically adopts float16
    k_fp16 = torch.randn(2, 2, 4, 8, dtype=torch.float16)
    v_fp16 = torch.randn(2, 2, 4, 8, dtype=torch.float16)
    cache.append(k_fp16, v_fp16, past_length=0)

    assert cache.dtype == torch.float16
    assert cache.size == 4
    assert torch.allclose(cache.key_cache[:, :, :4], k_fp16)
    assert torch.allclose(cache.value_cache[:, :, :4], v_fp16)

    # Append float32 keys/values to float16 cache -> auto cast to float16
    k_fp32 = torch.randn(2, 2, 2, 8, dtype=torch.float32)
    v_fp32 = torch.randn(2, 2, 2, 8, dtype=torch.float32)
    cache.append(k_fp32, v_fp32, past_length=4)

    assert cache.dtype == torch.float16
    assert cache.size == 6
    assert torch.allclose(cache.key_cache[:, :, 4:6], k_fp32.to(torch.float16))
    assert torch.allclose(cache.value_cache[:, :, 4:6], v_fp32.to(torch.float16))


def test_generate_and_stream_with_fp16_autocast():
    torch.manual_seed(0)
    m = _model(vocab_size=64, d_model=32, n_heads=2, n_layers=2)
    prompt = torch.randint(0, 64, (1, 4))

    # Test generate() with autocast
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps":
        m = m.to("mps")
        prompt = prompt.to("mps")
        with torch.amp.autocast(device_type="mps", dtype=torch.float16):
            out = m.generate(prompt, max_new_tokens=4, temperature=0.0)
            assert out.shape == (1, 8)
            tokens = list(m.stream_generate(prompt, max_new_tokens=4, temperature=0.0))
            assert len(tokens) == 4
    else:
        out = m.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert out.shape == (1, 8)
        tokens = list(m.stream_generate(prompt, max_new_tokens=4, temperature=0.0))
        assert len(tokens) == 4
