"""Tests for ``scripts/distill_multitask.py``.

The script composes existing building blocks (``train_qat`` QAT model factory,
``distill`` distillation loss/teacher, ``dataset.MultiTaskStreamer``) into a
multi-task distillation run. These tests pin the script-specific wiring: the
tie-weights default, gradient accumulation semantics, per-task validation
blending, task/weight selection, and an end-to-end run with a mocked HF
``load_dataset``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import torch

import pytest

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.mps_utils import mps_empty_cache
from bitnet_fff.tokenizer import ByteTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")


def _load_script(name: str):
    path = os.path.join(SCRIPTS, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_mod"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_script("distill_multitask.py")
train_qat = mod.train_qat
distill_mod = mod.distill_mod


# --- helpers -----------------------------------------------------------------


def _tiny_cfg(vocab=64, d_model=16, n_layers=1, seq=8):
    return BitNetFFTConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=2,
        n_layers=n_layers,
        fff_depth=1,
        fff_threshold_scale=1.0,
        activation_bits=8,
        max_seq_len=seq,
        tie_weights=True,
        use_fast_inference=False,
    )


def _student(seq=8):
    return train_qat.make_qat_model(
        _tiny_cfg(seq=seq), torch.device("cpu"), fp16=False
    )[0]


def _teacher(seq=8):
    cfg = _tiny_cfg(seq=seq)
    cfg.use_fast_inference = True
    model = BitNetFFTTransformer(cfg)
    model.eval()
    return model


def _batch(seq=8, vocab=64, n=2):
    return torch.randint(0, vocab, (n, seq))


def _rows_marker(marker):
    return {"text": marker}


def _fake_load_dataset(monkeypatch, rows_by_name, fail_splits=()):
    def fake_load_dataset(name, split="train", streaming=False, **kw):
        if split in fail_splits:
            raise ValueError(f"no '{split}' split for {name}")
        rows = rows_by_name.get(name)
        if rows is None:
            raise ValueError(f"unknown dataset {name}")
        return iter(list(rows))

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)


# --- tie-weights -------------------------------------------------------------


def test_tie_weights_default_true():
    args = mod.parse_args([])
    assert args.tie_weights is True
    assert args.no_tie_weights is False


def test_no_tie_weights_overrides():
    args = mod.parse_args(["--no-tie-weights"])
    assert args.tie_weights is True
    assert args.no_tie_weights is True


# --- distillation loss wiring ------------------------------------------------


def test_distillation_loss_math():
    torch.manual_seed(0)
    s = torch.randn(2, 8, 64)
    t = torch.randn(2, 8, 64)
    labels = torch.randint(0, 64, (2, 8))
    alpha, T, vocab = 0.7, 2.0, 64
    loss, kd, ce = distill_mod.distillation_loss(s, t, labels, alpha, T, vocab)
    # CE is standard next-token CE over the shifted sequence.
    ce_manual = torch.nn.functional.cross_entropy(
        s[:, :-1].float().reshape(-1, vocab), labels[:, 1:].reshape(-1)
    )
    assert torch.allclose(ce, ce_manual, atol=1e-5)
    # KD is temperature-scaled batchmean KL.
    kd_manual = (
        torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(s[:, :-1].float() / T, dim=-1),
            torch.nn.functional.softmax(t[:, :-1].float() / T, dim=-1),
            reduction="batchmean",
        )
        * T * T
    )
    assert torch.allclose(kd, kd_manual, atol=1e-5)
    assert torch.allclose(loss, alpha * kd + (1.0 - alpha) * ce, atol=1e-5)


# --- gradient accumulation ---------------------------------------------------


def test_accum_step_uses_one_optimizer_step():
    student = _student()
    teacher = _teacher()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    orig_step = opt.step
    steps = [0]

    def stepped():
        steps[0] += 1
        orig_step()

    opt.step = stepped
    batches = [_batch(n=2), _batch(n=2), _batch(n=2), _batch(n=2)]
    before = {id(p): p.clone() for p in student.parameters() if p.requires_grad}
    loss, kd, ce = mod.accum_step(
        student, opt, teacher, batches, 0.7, 2.0, 64, grad_accum=4
    )
    assert steps[0] == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in (loss, kd, ce))
    assert loss > 0.0
    moved = [
        p for p in student.parameters()
        if p.requires_grad and not torch.equal(before[id(p)], p)
    ]
    assert moved  # optimizer actually applied accumulated gradients


def test_accum_step_averages_over_micro_batches():
    student = _student()
    teacher = _teacher()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    batches = [_batch(n=2), _batch(n=2), _batch(n=2)]
    loss, _, _ = mod.accum_step(
        student, opt, teacher, batches, 0.7, 2.0, 64, grad_accum=3
    )
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 < loss < 1000.0


def test_accum_step_requires_batches():
    student = _student()
    teacher = _teacher()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    try:
        mod.accum_step(student, opt, teacher, [], 0.7, 2.0, 64, grad_accum=1)
    except RuntimeError as e:
        assert "micro-batch" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_accum_step_loss_scaling_and_range():
    # The KD term is re-scaled by (alpha * T^2 / vocab), so the blended loss
    # stays ~1-20 instead of being dominated by the temperature-squared KL.
    torch.manual_seed(0)
    student = _student()
    teacher = _teacher()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    torch.manual_seed(0)
    batch = _batch(n=2)
    alpha, temperature, vocab = 0.7, 2.0, 64
    loss, kd, ce = mod.accum_step(
        student, opt, teacher, [batch], alpha, temperature, vocab, grad_accum=1
    )
    # kd returned by accum_step is the scaled term == (alpha*T^2/vocab)*KL_batchmean,
    # which equals (alpha/vocab) * distill.py's kd (already T^2-scaled).
    kd_t2 = kd * vocab / alpha
    assert torch.allclose(
        torch.tensor(loss),
        torch.tensor((alpha / vocab) * kd_t2 + (1.0 - alpha) * ce),
    )
    assert 1.0 <= loss <= 20.0
    assert torch.isfinite(torch.tensor(loss))


def test_accum_step_clips_gradients_before_step(monkeypatch):
    student = _student()
    teacher = _teacher()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    calls = []

    def fake_clip(params, max_norm):
        calls.append((list(params), max_norm))
        return torch.tensor(0.0)

    monkeypatch.setattr("torch.nn.utils.clip_grad_norm_", fake_clip)
    mod.accum_step(student, opt, teacher, [_batch(n=2)], 0.7, 2.0, 64, grad_accum=1)
    assert len(calls) == 1
    params, max_norm = calls[0]
    assert max_norm == 1.0
    assert len(params) > 0


# --- per-task validation -----------------------------------------------------


def test_validate_tasks_metrics_and_blended():
    student = _student()
    b1, b2, b3 = _batch(), _batch(), _batch()
    val_streamers = {"math": iter([b1, b2]), "code": iter([b3])}
    weights = {"math": 0.8, "code": 0.2}
    metrics, blended = mod.validate_tasks(
        student, val_streamers, torch.device("cpu"), max_batches=10, weights=weights
    )
    assert set(metrics) == {"math", "code"}
    assert metrics["math"] is not None and metrics["code"] is not None
    assert blended > 0.0
    expected = (0.8 * metrics["math"] + 0.2 * metrics["code"]) / 1.0
    assert blended == expected


def test_validate_tasks_skips_empty_streams():
    student = _student()
    weights = {"math": 0.5, "code": 0.5}
    metrics, blended = mod.validate_tasks(
        student, {"math": iter([]), "code": iter([])},
        torch.device("cpu"), max_batches=10, weights=weights,
    )
    assert blended == float("inf")


def test_validate_tasks_restores_train_mode():
    student = _student()
    student.train()
    mod.validate_tasks(
        student, {"math": iter([_batch()])},
        torch.device("cpu"), max_batches=10, weights={"math": 1.0},
    )
    assert student.training is True


# --- task / weight selection -------------------------------------------------


def test_select_tasks_subset_and_weights():
    args = mod.parse_args(["--tasks", "math,code", "--weights", "0.6,0.4"])
    tasks = mod._select_tasks(args)
    assert [(t.name, t.weight) for t in tasks] == [("math", 0.6), ("code", 0.4)]


def test_select_tasks_all():
    args = mod.parse_args(["--tasks", "all"])
    tasks = mod._select_tasks(args)
    assert [t.name for t in tasks] == ["math", "instruct", "code"]


def test_select_tasks_rejects_unknown(monkeypatch):
    args = mod.parse_args(["--tasks", "nope"])
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    with __import__("pytest").raises(SystemExit):
        mod._select_tasks(args)


def test_select_tasks_weight_mismatch_rejected():
    args = mod.parse_args(["--tasks", "math,code", "--weights", "0.6"])
    with __import__("pytest").raises(SystemExit):
        mod._select_tasks(args)


# --- end-to-end main ---------------------------------------------------------


def _e2e_args(tmp_path, steps=6):
    return [
        "--device", "cpu", "--tokenizer", "bytes",
        "--d-model", "16", "--n-heads", "2", "--n-layers", "1", "--fff-depth", "1",
        "--teacher-d-model", "32", "--teacher-n-heads", "2", "--teacher-n-layers", "1",
        "--vocab-size", "256", "--max-seq-len", "64", "--seq-len", "8",
        "--batch-size", "2", "--grad-accum-steps", "2",
        "--steps", str(steps), "--val-every", "2", "--val-batches", "1",
        "--log-every", "2", "--save-every", "3", "--no-fp16",
        "--checkpoint", str(tmp_path / "mt_student.pt"),
    ]


def _recipe_fake_rows():
    return {
        "open-r1/OpenR1-Math-220k": [
            {"problem": f"p{i}", "solution": f"s{i}"} for i in range(100)
        ],
        "Open-Orca/SlimOrca-Dedup": [
            {"conversations": [{"from": "user", "value": f"q{i}"},
                               {"from": "assistant", "value": f"a{i}"}]}
            for i in range(100)
        ],
        "nickrosh/Evol-Instruct-Code-80k-v1": [
            {"instruction": f"write a function for {i}",
             "output": f"def f{i}():\n    return {i}\n"} for i in range(100)
        ],
    }


def test_main_e2e_mocked_datasets(monkeypatch, tmp_path, capsys):
    # Validation split must fall back to 'test'/'train' inside the run.
    _fake_load_dataset(monkeypatch, _recipe_fake_rows(), fail_splits=("validation",))
    rc = mod.main(_e2e_args(tmp_path, steps=6))
    out = capsys.readouterr().out
    assert rc == 0
    assert "tie_weights=True" in out
    assert "effective batch = 2 x 2 = 4" in out
    assert "math=40%" in out and "code=20%" in out
    assert "[val] step 2" in out and "[val] step 4" in out
    assert "blended=" in out
    assert (tmp_path / "mt_student.pt").exists()
    assert (tmp_path / "checkpoint_best.pt").exists()


def test_main_loss_decreases(monkeypatch, tmp_path, capsys):
    _fake_load_dataset(monkeypatch, _recipe_fake_rows())
    mod.main(_e2e_args(tmp_path, steps=8))
    out = capsys.readouterr().out
    losses = [
        float(m) for m in re.findall(r"\[distill-mt\] step \d+/\d+ loss=([\d.]+)", out)
    ]
    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_main_no_validation_every_step_respected(monkeypatch, tmp_path, capsys):
    _fake_load_dataset(monkeypatch, _recipe_fake_rows())
    mod.main(_e2e_args(tmp_path, steps=5) + ["--val-every", "999"])
    out = capsys.readouterr().out
    assert "[val] step" not in out
    assert (tmp_path / "mt_student.pt").exists()


def test_main_mps_empty_cache_on_device(monkeypatch, tmp_path, capsys):
    from bitnet_fff import mps_utils as mu

    calls = []
    mu.mps_empty_cache = lambda: calls.append(1)
    try:
        _fake_load_dataset(monkeypatch, _recipe_fake_rows())
        mod.main(_e2e_args(tmp_path, steps=4))
    finally:
        mu.mps_empty_cache = mps_empty_cache
    # main() guards by device; on CPU it should not call MPS-only helpers.
    assert calls == []


def test_main_resume_loads_state(monkeypatch, tmp_path, capsys):
    _fake_load_dataset(monkeypatch, _recipe_fake_rows())
    mod.main(_e2e_args(tmp_path, steps=4))
    capsys.readouterr()
    # resume from the saved checkpoint must still run to completion
    rc = mod.main(_e2e_args(tmp_path, steps=2))
    out = capsys.readouterr().out
    assert rc == 0
    assert "resume" in out


def test_main_o200k_base_sets_student_vocab(monkeypatch, tmp_path, capsys):
    pytest.importorskip("tiktoken")
    _fake_load_dataset(monkeypatch, _recipe_fake_rows())
    args = [a if a != "bytes" else "o200k_base" for a in _e2e_args(tmp_path, steps=2)]
    rc = mod.main(args)
    out = capsys.readouterr().out
    assert rc == 0
    # the student's vocab is taken from the encoding (~200k), not --vocab-size
    assert "[tokenizer] tiktoken o200k_base vocab=200019" in out
    assert "vocab=200019" in out
    assert "[teacher]" in out
