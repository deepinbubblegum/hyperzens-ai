"""Multi-task data-mixture streaming tests.

Covers the two contracts :class:`bitnet_fff.dataset.MultiTaskStreamer` must
guarantee: (1) examples are drawn at the configured sampling ratios (weighted
rejection sampling) and packed into fixed ``seq_len`` buffers with dynamic
BOS/EOS insertion, never padding; (2) streaming HuggingFace sources fall back
``validation -> test -> train`` and are re-iterable. All tests substitute a
fake ``datasets.load_dataset`` so no network is touched.
"""

from __future__ import annotations

import pytest
import torch

from bitnet_fff.dataset import (
    CODEFEEDBACK,
    OPENR1_MATH,
    SLIMORCA,
    MultiTaskStreamer,
    TaskSpec,
    _slimorca_format,
    multi_task_recipe,
)
from bitnet_fff.tokenizer import ByteTokenizer

VOCAB = 256


class _FakeRows:
    """Re-iterable fake streaming dataset."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(list(self.rows))


def _math_rows(n: int = 200):
    return [{"problem": f"problem {i}", "solution": f"solution {i}"} for i in range(n)]


def _instruct_rows(n: int = 200):
    return [
        {
            "conversations": [
                {"from": "user", "value": f"question {i}"},
                {"from": "assistant", "value": f"answer {i}"},
            ]
        }
        for i in range(n)
    ]


def _code_rows(n: int = 200):
    return [{"content": f"def f{i}():\n    return {i}\n"} for i in range(n)]


def _fake_datasets(monkeypatch, rows_by_name, fail_splits=()):
    calls = []

    def fake_load_dataset(name, split="train", streaming=False, **kw):
        calls.append((name, split))
        if split in fail_splits:
            raise ValueError(f"no '{split}' split for {name}")
        rows = rows_by_name.get(name)
        if rows is None:
            raise ValueError(f"unknown dataset {name}")
        return _FakeRows(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


def _default_fake(monkeypatch, n=200, fail_splits=()):
    return _fake_datasets(
        monkeypatch,
        {
            OPENR1_MATH: _math_rows(n),
            SLIMORCA: _instruct_rows(n),
            CODEFEEDBACK: _code_rows(n),
        },
        fail_splits=fail_splits,
    )


def _tok():
    return ByteTokenizer(VOCAB)


def _flatten(streamer):
    ids = []
    for batch in streamer:
        for row in batch:
            ids.extend(int(x) for x in row)
    return ids


# --- packing -----------------------------------------------------------------


def test_streamer_packs_fixed_seq_len_batches(monkeypatch):
    _default_fake(monkeypatch)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=16, batch_size=4, seed=1,
    )
    batches = list(streamer)
    assert batches
    for b in batches:
        assert b.shape == (4, 16)
        assert b.dtype == torch.long
        assert (b >= 0).all() and (b < VOCAB).all()


def test_bos_eos_insertion_and_trailing_drop(monkeypatch):
    _default_fake(monkeypatch)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=16, batch_size=4,
        bos_token_id=1, eos_token_id=2, seed=3,
    )
    ids = _flatten(streamer)
    emitted = sum(s.emitted for s in streamer._sources)
    assert ids.count(1) == emitted  # every example's BOS survives packing
    assert ids.count(2) in (emitted - 1, emitted)  # trailing EOS may be dropped


def test_dynamic_bos_eos_inside_single_buffer(monkeypatch):
    # Short examples: a single 16-token window must contain both the EOS of one
    # example and the BOS of the next (seamless cross-task packing).
    _default_fake(monkeypatch)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=16, batch_size=4,
        bos_token_id=1, eos_token_id=2, seed=3,
    )
    any_inner = False
    for batch in streamer:
        for row in batch:
            if 1 in row.tolist() and 2 in row.tolist():
                any_inner = True
                break
    assert any_inner


# --- weighted sampling -------------------------------------------------------


def test_weighted_sampling_ratios(monkeypatch):
    # Marker text must live in the fields the recipe's formatters read.
    fake = {
        OPENR1_MATH: [{"problem": "MATH", "solution": "x"} for _ in range(200)],
        SLIMORCA: [
            {"conversations": [{"from": "user", "value": "INST"}]} for _ in range(200)
        ],
        CODEFEEDBACK: [{"content": "CODE"} for _ in range(200)],
    }
    _fake_datasets(monkeypatch, fake)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=8, batch_size=1, seed=7,
    )
    counts = {"MATH": 0, "INST": 0, "CODE": 0}
    for batch in streamer:
        text = _tok().decode([int(x) for x in batch[0]])
        for marker in counts:
            if marker in text:
                counts[marker] += 1
    total = sum(counts.values())
    shares = {k: v / total for k, v in counts.items()}
    assert 0.30 <= shares["MATH"] <= 0.50
    assert 0.30 <= shares["INST"] <= 0.50
    assert 0.10 <= shares["CODE"] <= 0.30


def test_custom_weights_respected(monkeypatch):
    tasks = [
        TaskSpec(name="a", dataset=SLIMORCA, weight=0.8, field="text"),
        TaskSpec(name="b", dataset=CODEFEEDBACK, weight=0.2, field="content"),
    ]
    # fake rows must satisfy the field fallback; use marker text
    fake = {
        SLIMORCA: [{"text": "AAAA"} for _ in range(300)],
        CODEFEEDBACK: [{"content": "BBBB"} for _ in range(300)],
    }
    _fake_datasets(monkeypatch, fake)
    streamer = MultiTaskStreamer(tasks, _tok(), VOCAB, seq_len=8, batch_size=1, seed=5)
    counts = {"AAAA": 0, "BBBB": 0}
    for batch in streamer:
        text = _tok().decode([int(x) for x in batch[0]])
        for marker in counts:
            if marker in text:
                counts[marker] += 1
    total = sum(counts.values())
    assert counts["AAAA"] / total >= 0.65
    assert counts["BBBB"] / total <= 0.35


def test_max_examples_caps_each_task(monkeypatch):
    _default_fake(monkeypatch)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=16, batch_size=4,
        max_examples=5, seed=2,
    )
    list(streamer)
    for src in streamer._sources:
        assert src.emitted <= 5
    assert sum(s.emitted for s in streamer._sources) <= 15


# --- split fallback / sources ------------------------------------------------


def test_validation_split_falls_back_test_then_train(monkeypatch):
    calls = _fake_datasets(
        monkeypatch,
        {SLIMORCA: _instruct_rows(20)},
        fail_splits=("validation",),
    )
    streamer = MultiTaskStreamer(
        [TaskSpec(name="instruct", dataset=SLIMORCA, formatter=_slimorca_format)],
        _tok(), VOCAB, seq_len=8, batch_size=2, split="validation", seed=0,
    )
    batches = list(streamer)
    assert batches  # streamed from the 'test' fallback
    assert (SLIMORCA, "validation") in calls
    assert any(s == "test" for _, s in calls)


def test_missing_all_splits_raises(monkeypatch):
    _fake_datasets(
        monkeypatch,
        {SLIMORCA: _instruct_rows(5)},
        fail_splits=("validation", "test", "train"),
    )
    streamer = MultiTaskStreamer(
        [TaskSpec(name="instruct", dataset=SLIMORCA, formatter=_slimorca_format)],
        _tok(), VOCAB, seq_len=8, batch_size=2, split="validation",
    )
    with pytest.raises(SystemExit):
        list(streamer)


def test_streamer_is_reiterable(monkeypatch):
    _default_fake(monkeypatch)
    streamer = MultiTaskStreamer(
        multi_task_recipe(), _tok(), VOCAB, seq_len=16, batch_size=4, seed=11,
    )
    first = [b.tolist() for b in streamer]
    second = [b.tolist() for b in streamer]
    assert first == second and first


def test_field_and_formatter_extraction(monkeypatch):
    _fake_datasets(monkeypatch, {"ds": [{"content": "raw-code"} for _ in range(10)]})
    by_field = MultiTaskStreamer(
        [TaskSpec(name="t", dataset="ds", field="content")],
        _tok(), VOCAB, seq_len=8, batch_size=2, seed=0,
    )
    text = _tok().decode(_flatten(by_field))
    assert "raw-code" in text

    _fake_datasets(monkeypatch, {"ds": [{"a": "ignored", "b": "used"} for _ in range(10)]})
    by_formatter = MultiTaskStreamer(
        [TaskSpec(name="t", dataset="ds",
                  formatter=lambda ex: f"formatted:{ex['b']}")],
        _tok(), VOCAB, seq_len=8, batch_size=2, seed=0,
    )
    text = _tok().decode(_flatten(by_formatter))
    assert "formatted:used" in text


# --- validation / guards -----------------------------------------------------


def test_empty_tasks_raises():
    with pytest.raises(ValueError):
        MultiTaskStreamer([], _tok(), VOCAB, seq_len=8, batch_size=2)


def test_special_ids_out_of_range_raise():
    with pytest.raises(ValueError):
        MultiTaskStreamer(
            multi_task_recipe(), _tok(), 64, seq_len=8, batch_size=2,
            bos_token_id=100,
        )
    with pytest.raises(ValueError):
        MultiTaskStreamer(
            multi_task_recipe(), _tok(), 64, seq_len=8, batch_size=2,
            eos_token_id=100,
        )


def test_nonpositive_weight_raises():
    with pytest.raises(ValueError):
        MultiTaskStreamer(
            [TaskSpec(name="t", dataset="ds", weight=0.0)],
            _tok(), VOCAB, seq_len=8, batch_size=2,
        )
