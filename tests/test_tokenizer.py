"""BPE tokenizer tests: roundtrip, streaming decode, config binding, scaling.

Covers the production BPE tokenizer contract:

* **Roundtrip** — ``encode`` -> ``decode`` reproduces arbitrary text exactly,
  including multi-byte UTF-8 and split pieces.
* **Streaming decode** — :meth:`BPETokenizer.decode_step` assembles generated
  token ids token-by-token without corrupting multi-byte characters that the
  BPE splits across tokens.
* **Config sync** — :meth:`BitNetFFTConfig.from_tokenizer` /
  ``bind_tokenizer`` bind ``vocab_size`` and special ids from the tokenizer, and
  ``generate()`` honours the bound eos id.
* **Scaling** — the 50,257-vocab GPT-2 embedding/head train under the MPS
  FP16-master-weight QAT pipeline (CPU here, MPS when available).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.mps_utils import is_mps_available
from bitnet_fff.qat import BitNetQAT
from bitnet_fff.tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer

GPT2_VOCAB = 50257


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer("gpt2")


# --- vocabulary / special ids -------------------------------------------------


def test_gpt2_vocab_and_special_ids(tok):
    assert tok.vocab_size == GPT2_VOCAB
    assert tok.bos_token_id == 50256
    assert tok.eos_token_id == 50256
    assert tok.pad_token_id == 50256  # GPT-2 has no pad -> eos fallback


def test_load_tokenizer_prefers_bpe():
    assert isinstance(load_tokenizer(None), BPETokenizer)
    assert isinstance(load_tokenizer("bytes", 256), ByteTokenizer)
    assert load_tokenizer(None).vocab_size == GPT2_VOCAB


# --- roundtrip ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "héllo wörld 😀",
        "日本語のテストです。",
        "  leading and trailing  ",
        "line1\nline2\ttab",
        "3🙏🎉 mixed emoji",
        "a" * 120,
    ],
)
def test_encode_decode_roundtrip(tok, text):
    ids = tok.encode(text)
    assert isinstance(ids, list) and ids
    assert all(0 <= i < GPT2_VOCAB for i in ids)
    assert tok.decode(ids) == text


def test_byte_tokenizer_fallback_roundtrip():
    b = ByteTokenizer(256)
    assert b.bos_token_id is None and b.pad_token_id is None
    text = "héllo wörld 😀"
    assert b.encode(text) == list(text.encode("utf-8"))
    assert b.decode(b.encode(text)) == text
    with pytest.raises(ValueError):
        b = ByteTokenizer(64)
        b.encode("hello world")


# --- streaming decode ------------------------------------------------------------


def test_decode_step_streaming_exact(tok):
    text = "héllo wörld 😀 café 日本語 3🙏🎉  "
    pieces = [tok.decode_step(i) for i in tok.encode(text)]
    assert "".join(pieces) == text
    tok.reset_stream()


def test_decode_step_holds_partial_multibyte(tok):
    # 日 (E6 97 A5) is split across two GPT-2 tokens; the first piece alone is
    # an incomplete UTF-8 sequence and must be buffered, not emitted.
    ids = tok.encode("日本語")
    pieces = [tok.decode_step(i) for i in ids]
    assert pieces[0] == ""
    assert "".join(pieces) == "日本語"
    tok.reset_stream()


def test_reset_stream_flushes_pending_bytes(tok):
    first = tok.encode("日")[0]  # leading piece of a 3-byte char
    tok.decode_step(first)
    assert len(tok._buf) > 0
    tok.reset_stream()
    assert tok._buf == b""


def test_decode_step_matches_full_decode_over_generation(tok):
    cfg = BitNetFFTConfig.from_tokenizer(
        tok, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=32, use_fast_inference=False,
    )
    model = BitNetFFTTransformer(cfg)
    model.eval()
    prompt = torch.tensor([tok.encode("Once upon a time")])
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    pieces = [tok.decode_step(int(i)) for i in out[0].tolist()]
    assert "".join(pieces) == tok.decode(out[0].tolist())
    tok.reset_stream()


# --- model config sync ------------------------------------------------------------


def test_config_from_tokenizer_binds_vocab(tok):
    cfg = BitNetFFTConfig.from_tokenizer(
        tok, d_model=64, n_heads=2, n_layers=1, fff_depth=2, max_seq_len=32
    )
    assert cfg.vocab_size == GPT2_VOCAB
    assert cfg.bos_token_id == 50256
    assert cfg.eos_token_id == 50256
    assert cfg.pad_token_id == 50256


def test_config_bind_tokenizer_mutates(tok):
    cfg = BitNetFFTConfig(vocab_size=256, d_model=64, n_heads=2, n_layers=1,
                          fff_depth=2, max_seq_len=32)
    assert cfg.bind_tokenizer(tok) is cfg
    assert cfg.vocab_size == GPT2_VOCAB


def test_config_bind_byte_tokenizer():
    cfg = BitNetFFTConfig(vocab_size=512, d_model=64, n_heads=2, n_layers=1,
                          fff_depth=2, max_seq_len=32)
    cfg.bind_tokenizer(ByteTokenizer(256))
    assert cfg.vocab_size == 256


def test_model_embed_head_scale_with_large_vocab(tok):
    cfg = BitNetFFTConfig.from_tokenizer(
        tok, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=32, use_fast_inference=False,
    )
    model = BitNetFFTTransformer(cfg)
    assert model.embed.weight.shape == (GPT2_VOCAB, 32)
    assert model.head.weight.shape == (GPT2_VOCAB, 32)
    assert model.embed.weight.dtype == torch.float32


def test_generate_uses_config_bound_eos(monkeypatch, tok):
    cfg = BitNetFFTConfig.from_tokenizer(
        tok, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=32, use_fast_inference=False,
    )
    model = BitNetFFTTransformer(cfg)
    model.eval()
    eos = cfg.eos_token_id

    import bitnet_fff.models as models_mod

    monkeypatch.setattr(
        models_mod, "_sample_from_logits",
        lambda logits, temperature, top_k, top_p: torch.full(
            (logits.shape[0], 1), eos, device=logits.device
        ),
    )
    prompt = torch.tensor([tok.encode("hi there")])
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=8, temperature=1.0)
    assert out.shape == (1, len(prompt[0]) + 1)
    assert out[0, -1] == eos


# --- large-vocab FP16-master QAT ---------------------------------------------------


def _qat_model(tok, device: torch.device) -> tuple[BitNetQAT, object]:
    cfg = BitNetFFTConfig.from_tokenizer(
        tok, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=16, use_fast_inference=False,
    )
    qat = BitNetQAT(BitNetFFTTransformer(cfg).to(device))
    qat.enable_fp16_master()
    return qat, qat.optimizer(lr=1e-3)


def test_large_vocab_qat_fp16_trains(tok):
    torch.manual_seed(0)
    qat, opt = _qat_model(tok, torch.device("cpu"))
    assert qat.is_fp16_master
    assert all(p.dtype == torch.float16 for p in qat.module.parameters())
    ids = torch.randint(0, GPT2_VOCAB, (2, 8))  # fixed batch -> learnable

    def step():
        opt.zero_grad()
        logits = qat(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, GPT2_VOCAB).float(), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        opt.step()
        return float(loss.item())

    losses = [step() for _ in range(12)]
    assert losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.skipif(not is_mps_available(), reason="MPS not available")
def test_large_vocab_qat_fp16_mps(tok):
    torch.manual_seed(0)
    qat, opt = _qat_model(tok, torch.device("mps"))
    assert all(p.dtype == torch.float16 for p in qat.module.parameters())
    ids = torch.randint(0, GPT2_VOCAB, (2, 8), device="mps")

    def step():
        opt.zero_grad()
        logits = qat(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, GPT2_VOCAB).float(), ids[:, 1:].reshape(-1)
        )
        loss.backward()
        opt.step()
        return float(loss.item())

    losses = [step() for _ in range(8)]
    assert losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}"
