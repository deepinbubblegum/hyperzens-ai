"""Interactive chat CLI tests: history windowing and streaming typewriter loop.

Covers the two pieces ``scripts/chat.py`` must guarantee: the multi-turn
dialogue history stays inside the ``max_seq_len`` window (oldest turns evicted,
latest turn always kept), and a turn streams text through
``BitNetFFTTransformer.stream_generate`` with end-of-response performance
stats. Plus a subprocess smoke test of the interactive loop driven from stdin.
"""

from __future__ import annotations

import dataclasses
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


def test_parse_args_repetition_penalty_default_and_override():
    mod = _load_chat_module()
    assert mod.parse_args([]).repetition_penalty == 1.2
    assert mod.parse_args(["--repetition-penalty", "1.0"]).repetition_penalty == 1.0


def test_run_turn_forwards_repetition_penalty(capsys):
    mod = _load_chat_module()
    model = _tiny_model()
    seen = {}

    def fake_stream(prompt, **kwargs):
        seen.update(kwargs)
        return iter(["a", "b"])

    model.stream_generate = fake_stream
    tok = _byte_tok()
    text, stats = mod.run_turn(
        model, tok, [97, 98], torch.device("cpu"),
        max_new_tokens=2, temperature=0.0, top_k=50, top_p=0.9,
        eos_token_id=None, repetition_penalty=1.7,
    )
    assert seen["repetition_penalty"] == 1.7
    assert text == "ab"
    assert stats["tokens"] == 2


def test_run_turn_enters_eval_and_no_grad(capsys, monkeypatch):
    mod = _load_chat_module()
    model = _tiny_model()
    model.train()

    def fake_stream(prompt, **kwargs):
        assert not model.training
        assert not torch.is_grad_enabled()
        return iter(["x", "y"])

    model.stream_generate = fake_stream
    tok = _byte_tok()
    mod.run_turn(
        model, tok, [97], torch.device("cpu"),
        max_new_tokens=2, temperature=0.0, top_k=50, top_p=0.9,
        eos_token_id=None,
    )
    assert not model.training


# --- checkpoint architecture config ------------------------------------------


def _save_ckpt(tmp_path, name, cfg, with_config=True):
    tok = _byte_tok()
    bound = cfg.bind_tokenizer(tok)
    model = BitNetFFTTransformer(bound)
    path = tmp_path / name
    payload = {"model": model.state_dict()}
    if with_config:
        payload["config"] = dataclasses.asdict(bound)
    torch.save(payload, path)
    return str(path), bound


def test_checkpoint_config_reconstructs_architecture():
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, fff_depth=1,
        max_seq_len=32, tie_weights=True,
    ).bind_tokenizer(_byte_tok())
    saved = dataclasses.asdict(cfg)
    assert mod._checkpoint_config({"config": saved}).d_model == 16
    assert mod._checkpoint_config({"config": saved}).n_layers == 1
    assert mod._checkpoint_config({"config": saved}).fff_depth == 1


def test_checkpoint_config_drops_unknown_keys():
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, max_seq_len=32,
    ).bind_tokenizer(_byte_tok())
    saved = dataclasses.asdict(cfg)
    saved["some_future_field"] = 999
    assert mod._checkpoint_config({"config": saved}).d_model == 16


def test_checkpoint_config_none_without_config_key():
    mod = _load_chat_module()
    assert mod._checkpoint_config({"model": {}}) is None
    assert mod._checkpoint_config({}) is None


def test_main_loads_architecture_from_checkpoint(monkeypatch, tmp_path, capsys):
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, fff_depth=1,
        max_seq_len=32,
    )
    path, _ = _save_ckpt(tmp_path, "arch.pt", cfg)
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
        "--max-new-tokens", "8", "--checkpoint", path,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "architecture loaded from checkpoint" in out
    assert "d_model=16" in out and "n_layers=1" in out and "fff_depth=1" in out
    assert "max_seq_len=32" in out


def test_main_cli_flags_override_checkpoint_metadata(monkeypatch, tmp_path, capsys):
    mod = _load_chat_module()
    tok = _byte_tok()
    # Embedded config metadata claims a different architecture than the actual
    # checkpoint weights, so only a CLI override makes the model loadable.
    meta_cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, fff_depth=3,
        max_seq_len=32,
    ).bind_tokenizer(tok)
    real_cfg = BitNetFFTConfig(
        vocab_size=256, d_model=32, n_heads=2, n_layers=3, fff_depth=2,
        max_seq_len=64,
    ).bind_tokenizer(tok)
    path = tmp_path / "meta.pt"
    torch.save(
        {"config": dataclasses.asdict(meta_cfg),
         "model": BitNetFFTTransformer(real_cfg).state_dict()},
        path,
    )
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
        "--max-new-tokens", "8", "--checkpoint", str(path),
        "--d-model", "32", "--n-heads", "2", "--n-layers", "3", "--fff-depth", "2",
        "--max-seq-len", "64",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "architecture loaded from checkpoint" in out
    assert "CLI overrides applied" in out
    assert "d-model" in out and "n-layers" in out and "max-seq-len" in out
    assert "d_model=32" in out and "n_layers=3" in out and "fff_depth=2" in out
    assert "max_seq_len=64" in out


def test_main_checkpoint_defaults_keep_checkpoint_arch(monkeypatch, tmp_path, capsys):
    # No explicit arch flags on the command line -> checkpoint metadata wins,
    # even though the CLI defaults differ from it.
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, fff_depth=1,
        max_seq_len=32,
    )
    path, _ = _save_ckpt(tmp_path, "arch.pt", cfg)
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
        "--max-new-tokens", "8", "--checkpoint", path,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "architecture loaded from checkpoint" in out
    assert "CLI overrides applied" not in out
    assert "d_model=16" in out and "n_layers=1" in out and "fff_depth=1" in out
    assert "max_seq_len=32" in out


def test_main_plain_state_dict_uses_cli_arch(monkeypatch, tmp_path, capsys):
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(vocab_size=256, d_model=32, n_heads=2, n_layers=1,
                          max_seq_len=64)
    path, _ = _save_ckpt(tmp_path, "plain.pt", cfg, with_config=False)
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
        "--d-model", "32", "--n-heads", "2", "--n-layers", "1",
        "--max-seq-len", "64", "--max-new-tokens", "8", "--checkpoint", path,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "architecture loaded from checkpoint" not in out
    assert "d_model=32" in out


def test_main_rejects_max_new_tokens_vs_checkpoint_seq_len(tmp_path):
    mod = _load_chat_module()
    cfg = BitNetFFTConfig(
        vocab_size=256, d_model=16, n_heads=2, n_layers=1, max_seq_len=16,
    )
    path, _ = _save_ckpt(tmp_path, "small.pt", cfg)
    with pytest.raises(SystemExit):
        mod.main([
            "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
            "--max-new-tokens", "32", "--checkpoint", path,
        ])


def test_build_model_resizes_pos_embedding_for_seq_len_override(tmp_path, capsys):
    mod = _load_chat_module()
    tok = _byte_tok()
    old_cfg = BitNetFFTConfig(
        vocab_size=256, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=32,
    ).bind_tokenizer(tok)
    old = BitNetFFTTransformer(old_cfg)
    path = tmp_path / "ctx32.pt"
    torch.save({"model": old.state_dict()}, path)

    new_cfg = BitNetFFTConfig(
        vocab_size=256, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=512,
    ).bind_tokenizer(tok)
    model = mod.build_model(new_cfg, torch.device("cpu"), checkpoint=str(path))
    assert model.pos.weight.shape[0] == 512
    assert torch.equal(model.pos.weight[:32], old.pos.weight)  # prefix kept
    assert "resized pos embedding 512 rows" in capsys.readouterr().out
    assert not model.training  # eval() mode


def test_main_max_seq_len_override_extends_checkpoint_context(
    monkeypatch, tmp_path, capsys,
):
    mod = _load_chat_module()
    tok = _byte_tok()
    old_cfg = BitNetFFTConfig(
        vocab_size=256, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=32,
    ).bind_tokenizer(tok)
    path = tmp_path / "ctx32.pt"
    torch.save(
        {"config": dataclasses.asdict(old_cfg),
         "model": BitNetFFTTransformer(old_cfg).state_dict()},
        path,
    )
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--vocab-size", "256",
        "--max-new-tokens", "8", "--checkpoint", str(path),
        "--max-seq-len", "512",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "resized pos embedding 512 rows" in out
    assert "CLI overrides applied" in out and "max-seq-len" in out
    assert "max_seq_len=512" in out


def test_cli_arch_overrides_detects_user_flags():
    mod = _load_chat_module()
    defaults = mod.parse_args([])
    args = mod.parse_args(["--d-model", "64", "--max-seq-len", "256", "--tie-weights"])
    kwargs, names = mod._cli_arch_overrides(args, defaults)
    assert kwargs["d_model"] == 64
    assert kwargs["max_seq_len"] == 256
    assert kwargs["tie_weights"] is True
    assert names == ["d-model", "max-seq-len", "tie-weights"]
    kwargs2, names2 = mod._cli_arch_overrides(mod.parse_args([]), defaults)
    assert kwargs2 == {} and names2 == []


def test_cli_arch_overrides_maps_no_fff_bias():
    mod = _load_chat_module()
    defaults = mod.parse_args([])
    kwargs, names = mod._cli_arch_overrides(mod.parse_args(["--no-fff-bias"]), defaults)
    assert kwargs["fff_bias"] is False
    assert names == ["no-fff-bias"]


def test_main_o200k_base_sets_vocab_and_runs(monkeypatch, capsys):
    pytest.importorskip("tiktoken")
    mod = _load_chat_module()
    captured = {}
    real_build = mod.build_model

    def fake_build(cfg, device, checkpoint=None, state=None):
        captured["vocab"] = cfg.vocab_size
        return real_build(cfg, device, checkpoint, state)

    monkeypatch.setattr(mod, "build_model", fake_build)
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "o200k_base",
        "--d-model", "16", "--n-heads", "2", "--n-layers", "1", "--fff-depth", "1",
        "--max-seq-len", "64", "--max-new-tokens", "8",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tiktoken o200k_base vocab=200019" in out
    assert captured["vocab"] == 200019


# --- device selection / compile ----------------------------------------------


def test_resolve_device_priority_cuda_mps_cpu(monkeypatch):
    mod = _load_chat_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod, "is_mps_available", lambda: True)
    assert mod._resolve_device(None) == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert mod._resolve_device(None) == torch.device("mps")
    monkeypatch.setattr(mod, "is_mps_available", lambda: False)
    assert mod._resolve_device(None) == torch.device("cpu")


def test_resolve_device_explicit_choice():
    mod = _load_chat_module()
    assert mod._resolve_device("cpu") == torch.device("cpu")
    assert mod._resolve_device("cuda") == torch.device("cuda")


def test_compile_flag_default_off_and_accepted():
    mod = _load_chat_module()
    assert mod.parse_args([]).compile is False
    assert mod.parse_args(["--compile"]).compile is True


def test_main_compile_wraps_model(monkeypatch, capsys):
    mod = _load_chat_module()
    compiled = {}
    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: compiled.setdefault("model", model)
    )
    monkeypatch.setattr("builtins.input", lambda _: "/quit")
    rc = mod.main([
        "--device", "cpu", "--tokenizer", "bytes", "--compile",
        "--d-model", "16", "--n-heads", "2", "--n-layers", "1", "--fff-depth", "1",
        "--max-seq-len", "64", "--max-new-tokens", "8",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert compiled["model"] is not None
    assert "torch.compile enabled" in out


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
    assert "exceeds max_seq_len" in proc.stderr


def test_amp_autocast_devices():
    mod = _load_chat_module()
    import contextlib
    assert isinstance(mod._amp_autocast(torch.device("cpu")), contextlib.nullcontext)
    assert not isinstance(mod._amp_autocast(torch.device("mps")), contextlib.nullcontext)
    with pytest.warns(UserWarning, match="CUDA is not available"):
        assert not isinstance(mod._amp_autocast(torch.device("cuda")), contextlib.nullcontext)

