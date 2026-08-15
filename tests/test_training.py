"""QAT + distillation training tests.

Verifies that the QAT training step (BitNetQAT + FP16MasterAdamW with dynamic
activation scaling and cross-entropy) reduces loss over a small number of
iterations on a dummy text batch, that the teacher-student distillation
pipeline (KL + CE blend) also reduces loss, and that checkpointing and the
byte-level streaming data loader behave correctly.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")


def _load_script(name: str):
    path = os.path.join(SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_mod"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


train_qat = _load_script("train_qat.py")
distill = _load_script("distill.py")

VOCAB = 32


def _train_args(**overrides) -> dict:
    base = dict(
        vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=1, fff_depth=2,
        max_seq_len=16, seq_len=8, batch_size=4, activation_bits=8,
        attention_activation_bits=None, router_rank="full", no_fff_bias=False,
    )
    base.update(overrides)
    return base


def _dummy_batch(batch: int = 4, seq: int = 8) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, VOCAB, (batch, seq))


# --- QAT training loop ------------------------------------------------------


def test_qat_train_step_reduces_loss():
    args = train_qat.parse_args(["--data", "x", "--lr", "1e-3", "--no-fp16"])
    cfg = train_qat.build_cfg(args)
    qat, opt = train_qat.make_qat_model(cfg, torch.device("cpu"), fp16=False, lr=1e-3)
    ids = _dummy_batch()
    losses = [train_qat.train_step(qat, opt, ids) for _ in range(15)]
    assert losses[-1] < losses[0], f"loss must decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_qat_fp16_master_adamw_pipeline():
    """Exercise the exact deliverable path: BitNetQAT + FP16MasterAdamW + CE."""
    args = train_qat.parse_args(["--data", "x", "--lr", "1e-3"])
    cfg = train_qat.build_cfg(args)
    qat, opt = train_qat.make_qat_model(cfg, torch.device("cpu"), fp16=True, lr=1e-3)
    assert qat.is_fp16_master
    assert all(p.dtype == torch.float16 for p in qat.module.parameters())
    ids = _dummy_batch()
    first = train_qat.train_step(qat, opt, ids)
    for _ in range(14):
        last = train_qat.train_step(qat, opt, ids)
    assert last < first


def test_train_step_uses_dynamic_activation_scaling():
    """BitNetQAT applies per-token AbsMax dynamic scaling to float activations."""
    from bitnet_fff import FastFeedForwardBitNet
    from bitnet_fff.qat import BitNetQAT

    fff = FastFeedForwardBitNet(d_in=32, d_out=32, depth=2, bias=True)
    qat = BitNetQAT(fff, activation_bits=8, quant_mode="absmax")
    torch.manual_seed(1)
    x = torch.randn(8, 32)
    out = qat(x)
    assert out.shape == (8, 32)
    assert qat.act_quant.last_scale.item() > 0
    assert qat.act_quant.running_scale.item() > 0


# --- streaming data loader ---------------------------------------------------


def test_iter_batches_chunks_bytes():
    tok = train_qat.ByteTokenizer(256)
    text = "the quick brown fox jumps over the lazy dog. " * 8
    stream = train_qat.iter_batches([tok.encode(text)], seq_len=8, batch_size=4, vocab_size=256)
    batches = list(stream)
    assert batches, "expected at least one batch"
    for b in batches:
        assert b.shape == (4, 8)
        assert b.dtype == torch.long
        assert (b >= 0).all() and (b < 256).all()


def test_tokenize_stream_from_file(tmp_path):
    data = tmp_path / "corpus.txt"
    data.write_text("hello world " * 20)
    tok = train_qat.ByteTokenizer(256)
    chunks = list(train_qat.tokenize_stream(str(data), tok, 256))
    assert chunks
    assert sum(len(c) for c in chunks) == len("hello world " * 20)


def test_tokenize_stream_hf_requires_datasets():
    with pytest.raises(SystemExit):
        list(train_qat.tokenize_stream("hf://roneneldan/TinyStories",
                                       train_qat.ByteTokenizer(256), 256, max_examples=1))


# --- checkpointing -----------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    args = train_qat.parse_args(["--data", "x"])
    cfg = train_qat.build_cfg(args)
    qat, opt = train_qat.make_qat_model(cfg, torch.device("cpu"), fp16=False, lr=1e-3)
    ids = _dummy_batch()
    loss = train_qat.train_step(qat, opt, ids)
    path = str(tmp_path / "ckpt.pt")
    train_qat.save_checkpoint(path, cfg, qat.module.state_dict(), opt.state_dict(),
                              step=1, loss=loss)
    ckpt = train_qat.load_checkpoint(path)
    assert set(ckpt) >= {"config", "model", "optimizer", "step", "loss"}
    assert ckpt["config"]["d_model"] == cfg.d_model
    assert ckpt["step"] == 1
    assert "layers.0.attn.q_proj.weight" in ckpt["model"]


# --- distillation ------------------------------------------------------------


def _distill_setup(alpha: float = 0.7, temperature: float = 2.0):
    args = distill.parse_args(
        ["--data", "x", "--teacher", "local", "--teacher-n-layers", "2",
         "--teacher-d-model", "64", "--d-model", "32", "--n-layers", "1",
         "--vocab-size", str(VOCAB), "--max-seq-len", "16", "--seq-len", "8",
         "--batch-size", "4", "--alpha", str(alpha), "--temperature", str(temperature)]
    )
    teacher, meta = distill.load_teacher("local", torch.device("cpu"), VOCAB, args)
    cfg = train_qat.build_cfg(args)
    student, opt = train_qat.make_qat_model(cfg, torch.device("cpu"), fp16=False, lr=1e-3)
    return student, opt, teacher, args, cfg


def test_distill_reduces_loss_with_local_teacher():
    student, opt, teacher, args, cfg = _distill_setup()
    ids = _dummy_batch()
    losses, kds, ces = [], [], []
    for _ in range(15):
        l, kd, ce = distill.distill_step(
            student, opt, teacher, ids, args.alpha, args.temperature, cfg.vocab_size
        )
        losses.append(l); kds.append(kd); ces.append(ce)
    assert losses[-1] < losses[0], f"distill loss must decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(torch.isfinite(torch.tensor(x)) for x in losses + kds + ces)


def test_distill_loss_alpha_extremes():
    args = train_qat.parse_args(["--data", "x"])
    cfg = train_qat.build_cfg(args)
    vocab = cfg.vocab_size
    s = torch.randn(2, 6, vocab)
    t = torch.randn(2, 6, vocab)
    labels = torch.randint(0, vocab, (2, 6))
    # alpha=0 -> pure CE
    loss0, kd0, ce0 = distill.distillation_loss(s, t, labels, 0.0, 2.0, vocab)
    assert loss0 == pytest.approx(ce0.item())
    # alpha=1 -> pure KL (batchmean * T^2)
    loss1, kd1, ce1 = distill.distillation_loss(s, t, labels, 1.0, 2.0, vocab)
    assert loss1 == pytest.approx(kd1.item())
    # manual formula check for alpha=0.7
    alpha = 0.7
    loss, kd, ce = distill.distillation_loss(s, t, labels, alpha, 2.0, vocab)
    assert loss.item() == pytest.approx(alpha * kd.item() + (1 - alpha) * ce.item())


def test_distill_saves_checkpoint(tmp_path):
    student, opt, teacher, args, cfg = _distill_setup()
    ids = _dummy_batch()
    loss, _, _ = distill.distill_step(student, opt, teacher, ids, args.alpha,
                                      args.temperature, cfg.vocab_size)
    path = str(tmp_path / "student.pt")
    train_qat.save_checkpoint(
        path, cfg, student.module.state_dict(), opt.state_dict(), step=1, loss=loss,
        extra={"teacher": {"type": "local"}, "alpha": args.alpha,
               "temperature": args.temperature},
    )
    ckpt = torch.load(path, weights_only=False)
    assert set(ckpt) >= {"config", "model", "optimizer", "step", "loss", "teacher"}
    assert ckpt["alpha"] == args.alpha


def test_teacher_is_frozen():
    args = distill.parse_args(
        ["--data", "x", "--teacher", "local", "--vocab-size", str(VOCAB),
         "--max-seq-len", "16"]
    )
    teacher, meta = distill.load_teacher("local", torch.device("cpu"), VOCAB, args)
    assert all(not p.requires_grad for p in teacher.parameters())


# --- CLI smoke ---------------------------------------------------------------


@pytest.mark.parametrize("script", ["train_qat.py", "distill.py"])
def test_cli_help(script):
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", script), "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout.lower()
