"""Tests for bitnet_fff.dataset_mixture.

Covers the robust heterogeneous formatter, token-length filtering, delimiter
packing, domain weight ratios, and the MixtureStreamer end-to-end.  All tests
monkeypatch ``datasets.load_dataset`` so no network access is required.
"""

from __future__ import annotations

import pytest
import torch

from bitnet_fff.dataset import TaskSpec
from bitnet_fff.dataset_mixture import (
    DOMAINS,
    MASTER_DATASETS,
    MixtureStreamer,
    mixture_recipe,
    robust_parse,
)
from bitnet_fff.tokenizer import ByteTokenizer

VOCAB = 256


# ---------------------------------------------------------------------------
# Fake dataset helpers (mirrors tests/test_dataset.py)
# ---------------------------------------------------------------------------


class _FakeRows:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(list(self.rows))


def _fake_datasets(monkeypatch, rows_by_name):
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(name, split="train", streaming=False, **kw):
        calls.append((name, split))
        rows = rows_by_name.get(name)
        if rows is None:
            raise ValueError(f"unknown dataset {name}")
        return _FakeRows(rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    return calls


def _tok():
    return ByteTokenizer(VOCAB)


# ---------------------------------------------------------------------------
# robust_parse — schema coverage
# ---------------------------------------------------------------------------


class TestRobustParse:
    """Verify every documented schema is handled by the universal formatter."""

    def test_text_field(self):
        assert robust_parse({"text": "hello"}) == "hello"

    def test_content_field(self):
        assert robust_parse({"content": "code snippet"}) == "code snippet"

    def test_code_field(self):
        assert robust_parse({"code": "x = 1"}) == "x = 1"

    def test_strips_whitespace(self):
        assert robust_parse({"text": "  padded  "}) == "padded"

    def test_empty_text_skipped(self):
        assert robust_parse({"text": ""}) is None
        assert robust_parse({"text": "   "}) is None

    def test_conversations_from_value(self):
        ex = {"conversations": [
            {"from": "user", "value": "hi"},
            {"from": "assistant", "value": "hello"},
        ]}
        result = robust_parse(ex)
        assert "user: hi" in result
        assert "assistant: hello" in result

    def test_messages_role_content(self):
        ex = {"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]}
        result = robust_parse(ex)
        assert "system: You are helpful." in result
        assert "user: What is 2+2?" in result

    def test_conversations_missing_role_defaults_user(self):
        ex = {"conversations": [{"value": "solo turn"}]}
        result = robust_parse(ex)
        assert "user: solo turn" in result

    def test_conversations_skips_non_dict_turns(self):
        ex = {"conversations": [42, {"from": "user", "value": "ok"}]}
        result = robust_parse(ex)
        assert "user: ok" in result

    def test_instruction_output(self):
        ex = {"instruction": "translate", "output": "done"}
        result = robust_parse(ex)
        assert result == "Q: translate\nA: done"

    def test_question_answer(self):
        ex = {"question": "why?", "answer": "because"}
        result = robust_parse(ex)
        assert result == "Q: why?\nA: because"

    def test_problem_solution(self):
        ex = {"problem": "1+1", "solution": "2"}
        result = robust_parse(ex)
        assert result == "Q: 1+1\nA: 2"

    def test_prompt_response(self):
        ex = {"prompt": "say hi", "response": "hi!"}
        result = robust_parse(ex)
        assert result == "Q: say hi\nA: hi!"

    def test_qa_partial_empty(self):
        """Q or A alone (other empty) still produces output."""
        ex = {"question": "only q", "answer": ""}
        result = robust_parse(ex)
        assert result == "Q: only q\nA: "

    def test_qa_both_non_strings_skipped(self):
        ex = {"question": 123, "answer": 456}
        assert robust_parse(ex) is None

    def test_no_matching_schema_returns_none(self):
        assert robust_parse({"unrelated": 1}) is None
        assert robust_parse({}) is None

    def test_empty_conversations_returns_none(self):
        assert robust_parse({"conversations": []}) is None

    def test_empty_messages_returns_none(self):
        assert robust_parse({"messages": []}) is None

    def test_text_preferred_over_qa(self):
        """When both 'text' and QA keys exist, plain text wins (first match)."""
        ex = {"text": "plain", "question": "q", "answer": "a"}
        assert robust_parse(ex) == "plain"


# ---------------------------------------------------------------------------
# Token-length filtering
# ---------------------------------------------------------------------------


class TestTokenFiltering:
    """MixtureStreamer drops examples outside [min_tokens, max_tokens]."""

    def _make_rows(self, texts):
        """Each row has a 'text' field so robust_parse picks it up."""
        return [{"text": t} for t in texts]

    def test_short_examples_dropped(self, monkeypatch):
        # ByteTokenizer: each char = 1 token.  "hi" = 2 tokens < 32.
        short_rows = self._make_rows(["hi", "hey", "hello"])
        _fake_datasets(monkeypatch, {"ds_a": short_rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=2,
            min_tokens=32, max_tokens=512, seed=0,
        )
        batches = list(streamer)
        # All examples are < 32 tokens → streamer yields nothing
        assert batches == []

    def test_long_examples_dropped(self, monkeypatch):
        # 600-char string → 600 tokens with ByteTokenizer > 512.
        long_rows = self._make_rows(["x" * 600])
        _fake_datasets(monkeypatch, {"ds_a": long_rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=64, batch_size=2,
            min_tokens=32, max_tokens=512, seed=0,
        )
        batches = list(streamer)
        assert batches == []

    def test_in_range_examples_kept(self, monkeypatch):
        # 40-char string → 40 tokens, within [32, 512].
        rows = self._make_rows(["a" * 40 for _ in range(20)])
        _fake_datasets(monkeypatch, {"ds_a": rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=2,
            min_tokens=32, max_tokens=512, seed=0,
        )
        batches = list(streamer)
        assert len(batches) > 0
        for b in batches:
            assert b.shape == (2, 16)

    def test_boundary_at_exactly_min_tokens(self, monkeypatch):
        rows = self._make_rows(["b" * 32])
        _fake_datasets(monkeypatch, {"ds_a": rows * 10})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=2,
            min_tokens=32, max_tokens=512, seed=0,
        )
        batches = list(streamer)
        assert len(batches) > 0

    def test_boundary_at_exactly_max_tokens(self, monkeypatch):
        # 510 chars + delimiter "\n\n" (2 tokens) = 512 tokens = max_tokens.
        rows = self._make_rows(["c" * 510])
        _fake_datasets(monkeypatch, {"ds_a": rows * 10})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=64, batch_size=2,
            min_tokens=32, max_tokens=512, seed=0,
        )
        batches = list(streamer)
        assert len(batches) > 0

    def test_custom_filter_range(self, monkeypatch):
        rows = self._make_rows(["d" * 10 for _ in range(20)])
        _fake_datasets(monkeypatch, {"ds_a": rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        # min=5 max=15 — 10-char string is in range
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=8, batch_size=2,
            min_tokens=5, max_tokens=15, seed=0,
        )
        batches = list(streamer)
        assert len(batches) > 0


# ---------------------------------------------------------------------------
# Delimiter packing
# ---------------------------------------------------------------------------


class TestDelimiterPacking:
    """Verify the \\n\\n delimiter is appended before encoding."""

    def test_delimiter_in_packed_tokens(self, monkeypatch):
        rows = [{"text": "HELLO"} for _ in range(20)]
        _fake_datasets(monkeypatch, {"ds_a": rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=32, batch_size=1,
            min_tokens=5, max_tokens=512, seed=0,
            delimiter="\n\n",
        )
        all_ids: list[int] = []
        for batch in streamer:
            all_ids.extend(int(x) for x in batch[0])
        decoded = _tok().decode(all_ids)
        # Consecutive "HELLO" blocks separated by \n\n
        assert "HELLO\n\nHELLO" in decoded

    def test_custom_delimiter(self, monkeypatch):
        rows = [{"text": "AAA"} for _ in range(20)]
        _fake_datasets(monkeypatch, {"ds_a": rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=32, batch_size=1,
            min_tokens=5, max_tokens=512, seed=0,
            delimiter=" | ",
        )
        all_ids: list[int] = []
        for batch in streamer:
            all_ids.extend(int(x) for x in batch[0])
        decoded = _tok().decode(all_ids)
        assert "AAA | AAA" in decoded

    def test_empty_delimiter(self, monkeypatch):
        rows = [{"text": "BBB"} for _ in range(20)]
        _fake_datasets(monkeypatch, {"ds_a": rows})
        tasks = [TaskSpec(name="a", dataset="ds_a", weight=1.0)]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=32, batch_size=1,
            min_tokens=5, max_tokens=512, seed=0,
            delimiter="",
        )
        all_ids: list[int] = []
        for batch in streamer:
            all_ids.extend(int(x) for x in batch[0])
        decoded = _tok().decode(all_ids)
        # No separator between consecutive examples
        assert "BBB\n\nBBB" not in decoded


# ---------------------------------------------------------------------------
# Domain definitions & recipe
# ---------------------------------------------------------------------------


class TestDomainDefinitions:
    """Verify the master domain list structure and weight consistency."""

    def test_six_domains(self):
        assert len(DOMAINS) == 6

    def test_domain_names(self):
        names = [d["name"] for d in DOMAINS]
        assert names == ["reasoning", "general", "stem",
                         "specialties", "coding", "thai"]

    def test_weights_sum_to_one(self):
        total = sum(d["weight"] for d in DOMAINS)
        assert total == pytest.approx(1.0)

    def test_each_domain_has_datasets(self):
        for domain in DOMAINS:
            assert len(domain["datasets"]) >= 2

    def test_master_datasets_count(self):
        total_ds = sum(len(d["datasets"]) for d in DOMAINS)
        assert len(MASTER_DATASETS) == total_ds

    def test_master_datasets_weights_match_domain(self):
        for domain in DOMAINS:
            n = len(domain["datasets"])
            expected_per_ds = domain["weight"] / n
            for ds_spec in MASTER_DATASETS:
                if ds_spec.name.startswith(domain["name"] + "_"):
                    assert ds_spec.weight == pytest.approx(expected_per_ds)

    def test_recipe_returns_copy(self):
        r1 = mixture_recipe()
        r2 = mixture_recipe()
        assert r1 is not r2
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# MixtureStreamer — weighted sampling with filtering
# ---------------------------------------------------------------------------


class TestMixtureStreamerIntegration:
    """End-to-end tests for the filtered + delimited streamer."""

    def _build_fake_mixture(self, monkeypatch):
        """Create fake data for all 15 datasets with unique markers."""
        fake: dict[str, list[dict]] = {}
        for domain in DOMAINS:
            for i, ds in enumerate(domain["datasets"]):
                marker = f"{domain['name'][0].upper()}{i}"
                # Single-char marker * 40 = 40 tokens; with delimiter "\n\n"
                # that's 42 tokens total — below the default min_tokens=32.
                fake[ds["name"]] = [{"text": marker[0] * 40}] * 100
        _fake_datasets(monkeypatch, fake)
        return mixture_recipe()

    def test_streamer_produces_batches(self, monkeypatch):
        tasks = self._build_fake_mixture(monkeypatch)
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=4, seed=42,
        )
        batches = list(streamer)
        assert len(batches) > 0
        for b in batches:
            assert b.shape == (4, 16)
            assert b.dtype == torch.long

    def test_streamer_respects_filter(self, monkeypatch):
        """Only in-range examples survive the filter."""
        tasks = self._build_fake_mixture(monkeypatch)
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=4,
            min_tokens=50, max_tokens=100, seed=42,
        )
        batches = list(streamer)
        # Markers are 40 chars each → 40 tokens < min_tokens=50 → all dropped
        assert batches == []

    def test_streamer_is_reiterable(self, monkeypatch):
        tasks = self._build_fake_mixture(monkeypatch)
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=16, batch_size=4, seed=99,
        )
        first = [b.tolist() for b in streamer]
        second = [b.tolist() for b in streamer]
        assert first == second and first

    def test_streamer_handles_heterogeneous_schemas(self, monkeypatch):
        """Mix different schema types in different datasets."""
        fake = {
            "ds_text": [{"text": "PLAIN TEXT CONTENT AAAAA"}] * 50,
            "ds_conv": [{"conversations": [
                {"from": "user", "value": "QUESTION BBBBB"},
                {"from": "assistant", "value": "ANSWER BBBBB"},
            ]}] * 50,
            "ds_qa": [{"question": "CCCC question",
                       "answer": "CCCC answer"}] * 50,
            "ds_inst": [{"instruction": "DDDD task",
                         "output": "DDDD result"}] * 50,
        }
        _fake_datasets(monkeypatch, fake)
        tasks = [
            TaskSpec(name="text", dataset="ds_text", weight=0.25),
            TaskSpec(name="conv", dataset="ds_conv", weight=0.25),
            TaskSpec(name="qa", dataset="ds_qa", weight=0.25,
                     formatter=robust_parse),
            TaskSpec(name="inst", dataset="ds_inst", weight=0.25),
        ]
        streamer = MixtureStreamer(
            tasks, _tok(), VOCAB, seq_len=32, batch_size=2, seed=7,
            min_tokens=5, max_tokens=512,
        )
        all_text = ""
        for batch in streamer:
            for row in batch:
                all_text += _tok().decode([int(x) for x in row])
        # At least some of each schema's text should appear
        assert "PLAIN" in all_text
        assert "QUESTION" in all_text or "ANSWER" in all_text
        assert "CCCC" in all_text
        assert "DDDD" in all_text
