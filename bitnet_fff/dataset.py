"""Multi-task data mixture streaming for all-rounder distillation.

:class:`MultiTaskStreamer` combines several HuggingFace datasets (each wrapped
by a :class:`TaskSpec` with a configurable sampling weight and a text
formatter) into a single token stream. Examples are drawn with weighted
rejection sampling so each task contributes at the requested ratio, and the
resulting token ids are **packed** into fixed ``(batch_size, seq_len)``
buffers with dynamic BOS/EOS insertion: every example starts with
``bos_token_id`` and ends with ``eos_token_id`` wherever those markers fall
inside a buffer — buffers never contain padding, and tokens from different
tasks flow seamlessly across batch boundaries.

``multi_task_recipe()`` returns the default all-rounder mix:

* Reasoning/Math (40%): ``open-r1/OpenR1-Math-220k``
* General instruction (40%): ``Open-Orca/SlimOrca-Dedup``
* Code/Logic (20%): ``nickrosh/Evol-Instruct-Code-80k-v1``

Only streaming HuggingFace sources are supported; ``load_dataset`` is imported
lazily per iteration so tests (and offline runs) can substitute a fake source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch

__all__ = [
    "TaskSpec",
    "MultiTaskStreamer",
    "multi_task_recipe",
    "OPENR1_MATH",
    "SLIMORCA",
    "CODEFEEDBACK",
]

OPENR1_MATH = "open-r1/OpenR1-Math-220k"
SLIMORCA = "Open-Orca/SlimOrca-Dedup"
CODEFEEDBACK = "nickrosh/Evol-Instruct-Code-80k-v1"


@dataclass(frozen=True)
class TaskSpec:
    """One task in the mixture: an HF dataset plus how to read its text.

    Attributes:
        name: short task id used in logs / validation metrics.
        dataset: HuggingFace dataset name (``load_dataset(..., streaming=True)``).
        split: dataset split to stream (training).
        weight: relative sampling weight (``0.4`` = 40% of examples).
        field: direct ``example[field]`` extraction (mutually exclusive with
            ``formatter``; falls back to ``text``/``content``/``code`` keys).
        formatter: ``example -> text`` callable for structured rows (e.g. a
            QA pair or a conversation); returns ``None`` to skip the row.
    """

    name: str
    dataset: str
    split: str = "train"
    weight: float = 1.0
    field: str | None = None
    formatter: Callable[[dict], str | None] | None = None


# -- default formatters -------------------------------------------------------


def _openr1_format(ex: dict) -> str | None:
    problem = (ex.get("problem") or "").strip()
    solution = (ex.get("solution") or ex.get("answer") or "").strip()
    if not problem and not solution:
        return None
    return f"Q: {problem}\nA: {solution}\n"


def _slimorca_format(ex: dict) -> str | None:
    conv = ex.get("conversations") or ex.get("messages") or []
    parts = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        who = turn.get("from") or turn.get("role") or "user"
        value = (turn.get("value") or turn.get("content") or "").strip()
        if value:
            parts.append(f"{who}: {value}")
    if not parts:
        return None
    return "\n".join(parts) + "\n"


def _codefeedback_format(ex: dict) -> str | None:
    for key in ("content", "code", "text"):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value + "\n"
    instruction = ex.get("instruction")
    output = ex.get("output")
    if isinstance(instruction, str) and isinstance(output, str):
        instruction = instruction.strip()
        output = output.strip()
        if instruction or output:
            return f"Q: {instruction}\nA: {output}\n"
    return None


def multi_task_recipe() -> list[TaskSpec]:
    """Default all-rounder mixture (math 40% / instruct 40% / code 20%)."""
    return [
        TaskSpec(
            name="math",
            dataset=OPENR1_MATH,
            weight=0.4,
            formatter=_openr1_format,
        ),
        TaskSpec(
            name="instruct",
            dataset=SLIMORCA,
            weight=0.4,
            formatter=_slimorca_format,
        ),
        TaskSpec(
            name="code",
            dataset=CODEFEEDBACK,
            weight=0.2,
            formatter=_codefeedback_format,
        ),
    ]


# -- internals ----------------------------------------------------------------


class _TaskSource:
    """Stateful wrapper around one task's streaming example iterator."""

    __slots__ = (
        "spec", "dataset_iter", "tokenizer", "vocab_size", "bos", "eos",
        "max_examples", "emitted", "iterator",
    )

    def __init__(self, spec, dataset_iter, tokenizer, vocab_size, bos, eos, max_examples):
        self.spec = spec
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.bos = bos
        self.eos = eos
        self.max_examples = max_examples
        self.emitted = 0
        self.iterator = None

    def format(self, example: dict) -> str | None:
        spec = self.spec
        if spec.formatter is not None:
            return spec.formatter(example)
        for key in (spec.field, "text", "content", "code"):
            if key and isinstance(example.get(key), str) and example.get(key).strip():
                return example[key]
        return None

    def encode(self, text: str) -> list[int]:
        ids = [min(i, self.vocab_size - 1) for i in self.tokenizer.encode(text)]
        if self.bos is not None:
            ids = [self.bos] + ids
        if self.eos is not None:
            ids = ids + [self.eos]
        return ids


class MultiTaskStreamer:
    """Weighted multi-task streaming dataloader yielding packed token batches.

    Args:
        tasks: mixture (:class:`TaskSpec` list). Weights are normalized so they
            need not sum to 1.
        tokenizer: object with ``encode(str) -> list[int]``.
        vocab_size: clamp token ids below this bound.
        seq_len: fixed sequence-window length per row (packed, no padding).
        batch_size: rows per yielded batch.
        bos_token_id / eos_token_id: markers inserted at the start/end of every
            example wherever they fall inside the packed buffers.
        max_examples: per-task cap on streamed examples.
        split: optional split override for every task (used for validation);
            ``None`` uses each task's own ``split``. A missing split falls back
            ``validation -> test -> train`` so the same datasets can serve as
            validation.
        seed: reserved for reproducibility (streams are consumed in order).

    ``__iter__`` is re-iterable: each pass re-opens the streaming datasets, so
    the same instance can back repeated validation runs.
    """

    def __init__(
        self,
        tasks: list[TaskSpec],
        tokenizer,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_examples: int | None = None,
        split: str | None = None,
        seed: int = 0,
    ) -> None:
        if not tasks:
            raise ValueError("MultiTaskStreamer requires at least one task")
        if seq_len < 1 or batch_size < 1:
            raise ValueError("seq_len and batch_size must be >= 1")
        for spec in tasks:
            if spec.weight <= 0:
                raise ValueError(f"task {spec.name!r} weight must be > 0")
        if bos_token_id is not None and bos_token_id >= vocab_size:
            raise ValueError(
                f"bos_token_id {bos_token_id} out of range for vocab_size {vocab_size}"
            )
        if eos_token_id is not None and eos_token_id >= vocab_size:
            raise ValueError(
                f"eos_token_id {eos_token_id} out of range for vocab_size {vocab_size}"
            )
        self.tasks = list(tasks)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_examples = max_examples
        self.split = split
        self.seed = seed
        self._sources: list[_TaskSource] = []

    @property
    def task_names(self) -> list[str]:
        return [t.name for t in self.tasks]

    def _open_dataset(self, spec: TaskSpec, split: str):
        from datasets import load_dataset

        attempts = [split]
        if split != "train":
            attempts += ["test", "train"]
        last_err: Exception | None = None
        for s in attempts:
            try:
                return load_dataset(spec.dataset, split=s, streaming=True)
            except Exception as e:  # missing split, bad shard, ...
                last_err = e
        raise SystemExit(
            f"dataset {spec.dataset!r}: no streamable split in {attempts} "
            f"({last_err})"
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        self._sources = [
            _TaskSource(
                spec,
                self._open_dataset(spec, self.split or spec.split),
                self.tokenizer,
                self.vocab_size,
                self.bos_token_id,
                self.eos_token_id,
                self.max_examples,
            )
            for spec in self.tasks
        ]
        buffer: list[int] = []
        rows: list[list[int]] = []
        for ids in self._interleave():
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                rows.append(buffer[: self.seq_len])
                del buffer[: self.seq_len]
                if len(rows) == self.batch_size:
                    yield torch.tensor(rows, dtype=torch.long)
                    rows = []

    def _interleave(self) -> Iterator[list[int]]:
        """Yield per-example token ids at the configured sampling ratios.

        Weighted rejection sampling: a task is picked uniformly, its next
        example is fetched, and the example is *accepted* with probability
        ``weight / max_weight``. Each task's expected contribution is therefore
        proportional to its weight, so the mixture converges to the requested
        ratios regardless of the underlying dataset sizes. Exhausted or capped
        tasks drop out; when no active task remains the stream ends.
        """
        import random

        rng = random.Random(self.seed)
        active = list(self._sources)
        max_weight = max(s.spec.weight for s in active)
        while active:
            src = rng.choice(active)
            ids = self._fetch(src)
            if ids is None:
                active.remove(src)
                if active:
                    max_weight = max(s.spec.weight for s in active)
                continue
            if src.max_examples is not None and src.emitted >= src.max_examples:
                active.remove(src)
                continue
            if rng.random() >= src.spec.weight / max_weight:
                continue  # reject: keep the mixture at the configured ratios
            src.emitted += 1
            yield ids

    def _fetch(self, src: _TaskSource) -> list[int] | None:
        """Return the next example's token ids without accepting it."""
        while True:
            if src.iterator is None:
                src.iterator = iter(src.dataset_iter)
            try:
                example = next(src.iterator)
            except StopIteration:
                src.iterator = None
                return None
            text = src.format(example)
            if not text:
                continue
            ids = src.encode(text)
            if ids:
                return ids
