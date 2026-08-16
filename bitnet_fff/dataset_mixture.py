"""Dynamic multi-domain streaming from the master dataset list.

:class:`MixtureStreamer` draws from 15 HuggingFace datasets across 6 domains
(Reasoning, General, STEM, Specialties, Coding, Thai) at configurable sampling
ratios.  A **robust universal formatter** handles heterogeneous schemas
(``text``, ``conversations``, ``messages``, ``instruction``/``output``,
``question``/``answer``) and a **token-length filter** drops examples outside
a configurable range (default 32–512 tokens).  Continuous packing joins
consecutive examples with a configurable delimiter (default ``\\n\\n``).

The module reuses :class:`bitnet_fff.dataset.MultiTaskStreamer` for weighted
rejection sampling and ``seq_len``-locked batch packing, and extends its
internal ``_TaskSource`` with the filtering / delimiter logic so that no
out-of-range or empty example ever reaches a batch.

Usage::

    from bitnet_fff.dataset_mixture import MixtureStreamer, mixture_recipe
    from bitnet_fff.tokenizer import load_tokenizer

    tok = load_tokenizer("o200k_base")
    streamer = MixtureStreamer(
        mixture_recipe(), tok, tok.vocab_size,
        seq_len=512, batch_size=8,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    for batch in streamer:           # batch.shape == (8, 512)
        ...
"""

from __future__ import annotations

from typing import Iterator

import torch

from .dataset import (
    MultiTaskStreamer,
    TaskSpec,
    _TaskSource,
)

__all__ = [
    "DOMAINS",
    "MASTER_DATASETS",
    "robust_parse",
    "mixture_recipe",
    "MixtureStreamer",
]

# ---------------------------------------------------------------------------
# Robust heterogeneous formatter
# ---------------------------------------------------------------------------

# Conversation / message role keys -> label mapping
_ROLE_KEYS = ("from", "role")
_CONTENT_KEYS = ("value", "content", "text")


def robust_parse(example: dict) -> str | None:
    """Extract readable text from *any* known row schema.

    Tried in order (first non-empty match wins):

    1. **Plain text** -- ``text``, ``content``, ``code`` string fields.
    2. **Conversations / messages** -- list-of-dicts with role + content keys.
    3. **QA pairs** -- ``instruction``/``output``, ``question``/``,
       ``problem``/``solution``, ``prompt``/``response``.

    Returns ``None`` when no usable text can be extracted.
    """
    # --- 1. plain text -------------------------------------------------------
    for key in ("text", "content", "code"):
        value = example.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped

    # --- 2. conversations / messages -----------------------------------------
    for key in ("conversations", "messages"):
        conv = example.get(key)
        if isinstance(conv, list) and conv:
            parts: list[str] = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                role = "user"
                for rk in _ROLE_KEYS:
                    rv = turn.get(rk)
                    if isinstance(rv, str) and rv.strip():
                        role = rv.strip()
                        break
                content = ""
                for ck in _CONTENT_KEYS:
                    cv = turn.get(ck)
                    if isinstance(cv, str) and cv.strip():
                        content = cv.strip()
                        break
                if content:
                    parts.append(f"{role}: {content}")
            if parts:
                return "\n".join(parts)

    # --- 3. QA pairs ---------------------------------------------------------
    for q_key, a_key in (
        ("instruction", "output"),
        ("question", "answer"),
        ("problem", "solution"),
        ("prompt", "response"),
    ):
        q = example.get(q_key)
        a = example.get(a_key)
        if isinstance(q, str) and isinstance(a, str):
            q = q.strip()
            a = a.strip()
            if q or a:
                return f"Q: {q}\nA: {a}"

    return None


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS: list[dict] = [
    {
        "name": "reasoning",
        "weight": 0.25,
        "datasets": [
            {"name": "open-r1/OpenR1-Math-220k", "split": "train"},
            {"name": "BespokeLabs/Bespoke-Stratos-17k", "split": "train"},
        ],
    },
    {
        "name": "general",
        "weight": 0.20,
        "datasets": [
            {"name": "teknium/OpenHermes-2.5", "split": "train"},
            {"name": "Open-Orca/SlimOrca-Dedup", "split": "train"},
            {"name": "argilla/magpie-ultra-v0.1", "split": "train"},
        ],
    },
    {
        "name": "stem",
        "weight": 0.15,
        "datasets": [
            {"name": "camel-ai/physics", "split": "train"},
            {"name": "camel-ai/chemistry", "split": "train"},
            {"name": "camel-ai/biology", "split": "train"},
            {"name": "project-baize/arxiv-qa", "split": "train"},
        ],
    },
    {
        "name": "specialties",
        "weight": 0.15,
        "datasets": [
            {"name": "medalpaca/medical_meadow_medqa", "split": "train"},
            {"name": "nguien/legal-qa-v1", "split": "train"},
            {"name": "gbader/FinQA", "split": "train"},
        ],
    },
    {
        "name": "coding",
        "weight": 0.15,
        "datasets": [
            {"name": "glaiveai/glaive-code-assistant-v2", "split": "train"},
            {"name": "nickrosh/Evol-Instruct-Code-80k-v1", "split": "train"},
        ],
    },
    {
        "name": "thai",
        "weight": 0.10,
        "datasets": [
            {"name": "openthaigpt/thai_instruct", "split": "train"},
            {"name": "pythainlp/thai-wiki-dataset", "split": "train"},
        ],
    },
]

# Flatten domains -> TaskSpec list (domain weight split equally across its
# datasets).  robust_parse must be defined above this line.

MASTER_DATASETS: list[TaskSpec] = [
    TaskSpec(
        name=f"{domain['name']}_{i}",
        dataset=ds["name"],
        split=ds.get("split", "train"),
        weight=domain["weight"] / len(domain["datasets"]),
        formatter=robust_parse,
    )
    for domain in DOMAINS
    for i, ds in enumerate(domain["datasets"])
]


# ---------------------------------------------------------------------------
# Token-length filtered TaskSource
# ---------------------------------------------------------------------------


class _FilteredTaskSource(_TaskSource):
    """Extends ``_TaskSource`` with token-length gating and delimiter packing.

    ``encode`` returns an empty list (causing the caller to skip and retry)
    when the encoded token count falls outside
    ``[min_tokens, max_tokens]`` (measured *before* BOS/EOS insertion).
    A configurable ``delimiter`` string is appended to every example's text
    before encoding so that the continuous packed stream contains visible
    separators between documents.
    """

    __slots__ = ("min_tokens", "max_tokens", "delimiter")

    def __init__(
        self,
        spec: TaskSpec,
        dataset_iter,
        tokenizer,
        vocab_size: int,
        bos: int | None,
        eos: int | None,
        max_examples: int | None,
        *,
        min_tokens: int = 32,
        max_tokens: int = 512,
        delimiter: str = "\n\n",
    ) -> None:
        super().__init__(
            spec, dataset_iter, tokenizer, vocab_size, bos, eos, max_examples,
        )
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.delimiter = delimiter

    def encode(self, text: str) -> list[int]:
        # Append delimiter so consecutive packed examples are separated.
        text_with_delim = text + self.delimiter
        ids = [
            min(i, self.vocab_size - 1)
            for i in self.tokenizer.encode(text_with_delim)
        ]
        # Gate on raw-token count *before* BOS/EOS (those are fixed-size).
        if len(ids) < self.min_tokens or len(ids) > self.max_tokens:
            return []
        if self.bos is not None:
            ids = [self.bos] + ids
        if self.eos is not None:
            ids = ids + [self.eos]
        return ids


# ---------------------------------------------------------------------------
# Mixture streamer
# ---------------------------------------------------------------------------


class MixtureStreamer(MultiTaskStreamer):
    """Weighted multi-domain streaming dataloader with filtering.

    Extends :class:`MultiTaskStreamer` with:

    * **Token-length gating** -- examples outside ``[min_tokens, max_tokens]``
      are silently skipped (the iterator retries the next example from the
      same source).
    * **Delimiter packing** -- ``delimiter`` (default ``\\n\\n``) is appended to
      every example's text before encoding so that the packed token stream
      contains visible document separators.
    * **Robust formatting** -- when no per-task ``formatter`` is supplied the
      universal :func:`robust_parse` is used automatically.

    All constructor arguments from :class:`MultiTaskStreamer` are preserved;
    the new keyword-only parameters are:

    Args:
        min_tokens: minimum token count per example (before BOS/EOS).
            Examples with fewer tokens are silently dropped.
        max_tokens: maximum token count per example (before BOS/EOS).
        delimiter: string appended to every example before encoding.
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
        *,
        min_tokens: int = 32,
        max_tokens: int = 512,
        delimiter: str = "\n\n",
    ) -> None:
        super().__init__(
            tasks,
            tokenizer,
            vocab_size,
            seq_len,
            batch_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_examples=max_examples,
            split=split,
            seed=seed,
        )
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.delimiter = delimiter

    # Override __iter__ to swap _TaskSource -> _FilteredTaskSource and
    # auto-inject robust_parse when a task has no explicit formatter.

    def __iter__(self) -> Iterator[torch.Tensor]:
        resolved: list[TaskSpec] = []
        for spec in self.tasks:
            if spec.formatter is None:
                resolved.append(
                    TaskSpec(
                        name=spec.name,
                        dataset=spec.dataset,
                        split=spec.split,
                        weight=spec.weight,
                        field=spec.field,
                        formatter=robust_parse,
                    )
                )
            else:
                resolved.append(spec)
        self._sources = [
            _FilteredTaskSource(
                spec,
                self._open_dataset(spec, self.split or spec.split),
                self.tokenizer,
                self.vocab_size,
                self.bos_token_id,
                self.eos_token_id,
                self.max_examples,
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                delimiter=self.delimiter,
            )
            for spec in resolved
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


# ---------------------------------------------------------------------------
# Public recipe builder
# ---------------------------------------------------------------------------


def mixture_recipe() -> list[TaskSpec]:
    """Return the master dataset list as a flat :class:`TaskSpec` mixture.

    Each domain's weight is split equally among its constituent datasets.
    All tasks use :func:`robust_parse` as their formatter.
    """
    return list(MASTER_DATASETS)
