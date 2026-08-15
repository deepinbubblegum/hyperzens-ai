"""Interactive chat CLI tests: history windowing and streaming typewriter loop.

Covers the two pieces ``scripts/chat.py`` must guarantee: the multi-turn
dialogue history stays inside the ``max_seq_len`` window (oldest turns evicted,
latest turn always kept), and a turn streams text through
``BitNetFFTTransformer.stream_generate`` with end-of-response performance
stats. Plus a subprocess smoke test of the interactive loop driven from stdin.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.tokenizer import load_tokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_chat_module():
    path = os.path.join(REPO, "scripts", "chat.py")
    spec = importlib.util.spec_from_file_location("chat_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chat_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def _byte_tok():
    return load_tokenizer("bytes", 256)


def _tiny_model() -> BitNetFFTTransformer:
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=64,
    )
    return BitNetFFTTransformer(cfg).eval()


# --- dialogue history windowing ----------------------------------------------


def test_build_history_prompt_evicts_oldest_turns():
    mod = _load_chat_module()
    tok = _byte_tok()
    long_first = "x" * 50
    messages = [
        ("User", long_first),
        ("Assistant", "ok"),
        ("User", "how are you"),
    ]
    ids = mod.build_history_prompt(messages, tok, max_tokens=40, vocab_size=256)
    text = tok.decode(ids)
    assert len(ids) <= 40
    assert "ok" in text and "how are you" in text
    assert long_first not in text  # oldest turn evicted


def test_build_history_prompt_clips_oversized_single_turn():
    mod = _load_chat_module()
    tok = _byte_tok()
    messages = [("User", "y" * 100)]
    ids = mod.build_history_prompt(messages, tok, max_tokens=16, vocab_size=256)
    assert len(ids) == 16  # truncated to the window
    assert tok.decode(ids) == "y" * 15 + "\n"  # last 16 bytes kept


def test_build_history_prompt_clamps_vocab():
    mod = _load_chat_module()
    tok = _byte_tok()
    messages = [("User", "abc")]
    ids = mod.build_history_prompt(messages, tok, max_tokens=64, vocab_size=100)
    assert all(i < 100 for i in ids)


def test_build_history_prompt_keeps_small_transcript_intact():
    mod = _load_chat_module()
    tok = _byte_tok()
    messages = [("User", "hi"), ("Assistant", "hello")]
    ids = mod.build_history_prompt(messages, tok, max_tokens=256, vocab_size=256)
    text = tok.decode(ids)
    assert "User > hi" in text and "Assistant > hello" in text


# --- streaming turn + stats --------------------------------------------------


def test_run_turn_streams_text_and_reports_stats(capsys):
    mod = _load_chat_module()
    model = _tiny_model()
    tok = _byte_tok()
    ids = [ord(c) for c in "hello"]
    text, stats = mod.run_turn(
        model, tok, ids, torch.device("cpu"),
        max_new_tokens=4, temperature=0.0, top_k=50, top_p=0.9,
        eos_token_id=None,
    )
    out = capsys.readouterr().out
    assert stats["tokens"] == 4
    assert stats["ms_per_token"] > 0
    assert len(text) == 4  # byte tokenizer: one char per generated token
    assert out  # streamed pieces were written to stdout


def test_format_stats():
    mod = _load_chat_module()
    s = mod.format_stats(
        {"tokens": 12, "seconds": 0.5, "tokens_per_s": 24.0, "ms_per_token": 41.6}
    )
    assert "12 tokens" in s
    assert "tok/s" in s and "ms/token" in s and "0.50 s" in s


# --- CLI smoke test ----------------------------------------------------------


def test_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/chat.py", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
    assert "--max-new-tokens" in proc.stdout


def test_cli_chat_streams_and_reports_stats():
    proc = subprocess.run(
        [
            sys.executable, "scripts/chat.py",
            "--device", "cpu", "--tokenizer", "bytes",
            "--d-model", "32", "--n-heads", "2", "--n-layers", "1",
            "--fff-depth", "2", "--vocab-size", "256", "--max-seq-len", "64",
            "--max-new-tokens", "4", "--temperature", "0.0",
        ],
        input="hi\n/quit\n", cwd=REPO, capture_output=True, text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "User > " in proc.stdout
    assert "Assistant > " in proc.stdout
    assert "[stats]" in proc.stdout
    assert "tok/s" in proc.stdout and "ms/token" in proc.stdout


def test_cli_rejects_max_new_tokens_ge_seq_len():
    proc = subprocess.run(
        [
            sys.executable, "scripts/chat.py",
            "--device", "cpu", "--tokenizer", "bytes",
            "--max-seq-len", "16", "--max-new-tokens", "16",
        ],
        input="hi\n", cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "must be <" in proc.stderr
