"""Dynamic multi-domain streaming from the master dataset list.

:class:`MixtureStreamer` draws from 20 HuggingFace datasets across 9 domains
(Reasoning, General, STEM, Specialties, Coding, Thai, Function Calling, Hard
Logic, Graduate STEM) at configurable sampling ratios.

A **robust universal formatter** handles heterogeneous schemas (``text``,
``conversations``, ``messages``, ``instruction``/``output``,
``question``/``answer``).  **Specialized formatters** produce structured
outputs for function-calling (ChatML ``<tool_call>`` blocks) and
multiple-choice CoT prompts with ``<thought>`` tags (BBH / GPQA).

A **token-length filter** drops examples outside a configurable range
(default 32–512 tokens).  Continuous packing joins consecutive examples with
a configurable delimiter (default ``\\n\\n``).

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

import json
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
    "function_calling_format",
    "bbh_format",
    "gpqa_format",
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
# Specialized formatters
# ---------------------------------------------------------------------------


def _parse_json_field(value: str | list | None) -> list | dict | None:
    """Best-effort JSON parse; returns the parsed object or ``None``."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def function_calling_format(ex: dict) -> str | None:
    """Format function-calling examples as structured ChatML tool calls.

    Handles schemas from ``Salesforce/xlam-function-calling-60k`` and similar
    datasets with ``query``, ``tools`` / ``functions``, and ``answers`` /
    ``function`` columns.

    Output format::

        User: <query>

        Available tools:
        ```json
        [{"name": "...", "description": "...", ...}]
        ```

        <tool_call>
        [{"name": "...", "arguments": {...}}]
        </tool_call>
    """
    # --- extract query --------------------------------------------------------
    query = None
    for key in ("query", "question", "instruction", "prompt", "text"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            query = val.strip()
            break
    if not query:
        return None

    # --- extract tools --------------------------------------------------------
    raw_tools = ex.get("tools") or ex.get("functions") or ex.get("tool")
    tools = _parse_json_field(raw_tools)
    tools_str = "[]"
    if isinstance(tools, list) and tools:
        try:
            tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            tools_str = str(tools)
    elif isinstance(tools, dict):
        try:
            tools_str = json.dumps([tools], indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            tools_str = str(tools)

    # --- extract answer / expected call --------------------------------------
    raw_answers = (
        ex.get("answers") or ex.get("answer") or ex.get("function")
        or ex.get("ground_truth") or ex.get("output")
    )
    answers = _parse_json_field(raw_answers)
    answers_str = "[]"
    if isinstance(answers, list) and answers:
        try:
            answers_str = json.dumps(answers, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            answers_str = str(answers)
    elif isinstance(answers, dict):
        try:
            answers_str = json.dumps([answers], indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            answers_str = str(answers)
    elif isinstance(raw_answers, str) and raw_answers.strip():
        answers_str = raw_answers.strip()

    return (
        f"User: {query}\n\n"
        f"Available tools:\n```json\n{tools_str}\n```\n\n"
        f"<tool_call>\n{answers_str}\n</tool_call>"
    )


def bbh_format(ex: dict) -> str | None:
    """Format BIG-Bench Hard multiple-choice questions as step-by-step CoT.

    Handles schemas from ``maira-res/BIG-bench-hard`` with ``question``,
    optional ``choices`` (struct with ``label`` / ``text`` lists), and
    ``target`` columns.  Produces structured ``<thought>`` prompts::

        Question: <question>

        Choices:
        (A) <text>
        (B) <text>
        ...

        <thought>
        Let me think step by step...
        </thought>

        Answer: (<label>) <text>
    """
    question = None
    for key in ("question", "input", "problem", "prompt", "text"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            question = val.strip()
            break
    if not question:
        return None

    # --- extract choices (optional) ------------------------------------------
    choices_obj = ex.get("choices")
    choice_labels = None
    choice_texts = None

    if isinstance(choices_obj, dict):
        choice_labels = choices_obj.get("label") or choices_obj.get("choices")
        choice_texts = choices_obj.get("text") or choices_obj.get("options")
    elif isinstance(choices_obj, list):
        # flat list of strings → treat as texts with auto labels
        if all(isinstance(c, str) for c in choices_obj):
            choice_texts = choices_obj
            choice_labels = [chr(65 + i) for i in range(len(choices_obj))]

    # Format choices block
    choices_block = ""
    if choice_texts and isinstance(choice_texts, list):
        labels = choice_labels if choice_labels and len(choice_labels) == len(choice_texts) else [chr(65 + i) for i in range(len(choice_texts))]
        lines = []
        for label, text in zip(labels, choice_texts):
            lines.append(f"({label}) {text}")
        choices_block = "\n".join(lines)

    # --- extract target -------------------------------------------------------
    target = None
    for key in ("target", "answer", "output", "solution", "correct"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            target = val.strip()
            break
        elif isinstance(val, list) and val:
            # BBH targets are often lists like ["A"] or ["(A) answer text"]
            first = val[0]
            if isinstance(first, str):
                target = first.strip()
                break

    # --- build output ---------------------------------------------------------
    parts = [f"Question: {question}"]
    if choices_block:
        parts.append(f"\nChoices:\n{choices_block}")
    parts.append(
        "\n\n<thought>\n"
        "Let me think step by step.\n"
        "</thought>"
    )
    if target:
        parts.append(f"\n\nAnswer: {target}")
    return "\n".join(parts)


def gpqa_format(ex: dict) -> str | None:
    """Format GPQA graduate-level multiple-choice as step-by-step CoT.

    Handles schemas from ``Idavidrein/gpqa`` with ``Question``,
    ``Correct Answer``, ``Incorrect Answer 1/2/3``, and ``Subdomain``
    columns.  Produces structured ``<thought>`` prompts::

        Question: <Question>
        Domain: <Subdomain>

        Choices:
        (A) <Correct Answer>
        (B) <Incorrect Answer 1>
        (C) <Incorrect Answer 2>
        (D) <Incorrect Answer 3>

        <thought>
        This is a graduate-level question in <Subdomain>...
        </thought>

        Answer: (A) <Correct Answer>
    """
    question = None
    for key in ("Question", "question", "input", "problem", "prompt"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            question = val.strip()
            break
    if not question:
        return None

    # --- extract domain metadata ---------------------------------------------
    domain = None
    for key in ("Subdomain", "subdomain", "High-level domain", "domain", "category"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            domain = val.strip()
            break

    # --- extract answers (shuffled into choices) -----------------------------
    correct = None
    incorrects: list[str] = []
    for key in ("Correct Answer", "correct_answer", "answer", "solution"):
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            correct = val.strip()
            break
    for i in range(1, 5):
        for key in (f"Incorrect Answer {i}", f"incorrect_answer_{i}", f"distractor_{i}"):
            val = ex.get(key)
            if isinstance(val, str) and val.strip():
                incorrects.append(val.strip())
                break

    if not correct:
        return None

    # Build choices list: correct at position A, incorrects after
    all_choices = [correct] + incorrects[:3]
    labels = [chr(65 + i) for i in range(len(all_choices))]

    # --- build output ---------------------------------------------------------
    parts = [f"Question: {question}"]
    if domain:
        parts.append(f"Domain: {domain}")
    if len(all_choices) > 1:
        choice_lines = [f"({label}) {text}" for label, text in zip(labels, all_choices)]
        parts.append("\nChoices:\n" + "\n".join(choice_lines))
    parts.append(
        "\n\n<thought>\n"
        "Let me think step by step.\n"
        "</thought>"
    )
    parts.append(f"\n\nAnswer: (A) {correct}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS: list[dict] = [
    {
        "name": "reasoning",
        "weight": 0.20,
        "datasets": [
            {"name": "open-r1/OpenR1-Math-220k", "split": "train"},
            {"name": "BespokeLabs/Bespoke-Stratos-17k", "split": "train"},
        ],
    },
    {
        "name": "general",
        "weight": 0.15,
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
        "weight": 0.10,
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
    {
        "name": "function_calling",
        "weight": 0.05,
        "datasets": [
            {
                "name": "Salesforce/xlam-function-calling-60k",
                "split": "train",
                "formatter": function_calling_format,
            },
            {
                "name": "gorilla-llm/berkeley-function-calling-leaderboard",
                "split": "train",
                "formatter": function_calling_format,
            },
        ],
    },
    {
        "name": "logic",
        "weight": 0.05,
        "datasets": [
            {
                "name": "maira-res/BIG-bench-hard",
                "split": "train",
                "formatter": bbh_format,
            },
        ],
    },
    {
        "name": "graduate_stem",
        "weight": 0.05,
        "datasets": [
            {
                "name": "Idavidrein/gpqa",
                "split": "train",
                "formatter": gpqa_format,
            },
        ],
    },
]

# Flatten domains -> TaskSpec list (domain weight split equally across its
# datasets).  Each dataset may specify its own ``formatter``; otherwise
# ``robust_parse`` is used as the universal fallback.

MASTER_DATASETS: list[TaskSpec] = [
    TaskSpec(
        name=f"{domain['name']}_{i}",
        dataset=ds["name"],
        split=ds.get("split", "train"),
        weight=domain["weight"] / len(domain["datasets"]),
        formatter=ds.get("formatter") or robust_parse,
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
