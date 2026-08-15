"""BPE tokenization for BitNet-FFF.

:class:`BPETokenizer` wraps a HuggingFace ``AutoTokenizer`` (default ``gpt2``,
50,257-token byte-level BPE; ``Qwen/Qwen2.5-0.5B`` also works) and adds a
byte-accurate **streaming decoder** (:meth:`BPETokenizer.decode_step`) that
assembles generated token ids into text token-by-token without corrupting
multi-byte UTF-8 sequences that a BPE splits across several tokens.

:class:`TikTokenizer` wraps a ``tiktoken`` encoding (``o200k_base``, the
GPT-4o vocabulary) with the same interface, including the byte-accurate
streaming decoder built on :meth:`tiktoken.Encoding.decode_single_token_bytes`.

A :class:`ByteTokenizer` remains as an explicit offline fallback (``bytes`` /
no ``transformers``); :func:`load_tokenizer` always prefers BPE and only drops
to bytes when BPE is impossible. Both expose the ``encode`` / ``decode`` /
``vocab_size`` / ``bos_token_id`` / ``eos_token_id`` / ``pad_token_id``
interface so the training / generation scripts and
:meth:`bitnet_fff.models.BitNetFFTConfig.bind_tokenizer` are backend-agnostic.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_BPE_TOKENIZER = "gpt2"
_BYTE_FALLBACK_NAME = "bytes"
TIKTOKEN_NAME = "o200k_base"


def _bytes_to_unicode() -> dict[int, int]:
    """byte -> unicode-char mapping used by byte-level BPE tokenizers.

    Bytes outside the printable Latin-1 ranges are re-coded into a private
    codepoint block (256+), which is how GPT-2 / Llama / Qwen vocabularies
    store arbitrary bytes. See OpenAI's ``gpt-2`` ``encoder.py``.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, cs))


@lru_cache(maxsize=1)
def _char_to_byte() -> dict[int, int]:
    return {c: b for b, c in _bytes_to_unicode().items()}


def _token_str_to_bytes(token: str) -> bytes | None:
    """Raw bytes a byte-level vocab token represents; ``None`` if not byte-level.

    Every character of a byte-level vocab entry is either a plain byte
    (codepoint < 256, recovered as-is) or a re-coded special byte (codepoint in
    the 256+ private block, recovered through the reverse table).
    """
    out = bytearray()
    for ch in token:
        o = ord(ch)
        mapped = _char_to_byte().get(o)
        if mapped is not None:
            out.append(mapped)
        elif o < 256:
            out.append(o)
        else:
            return None
    return bytes(out)


class BPETokenizer:
    """Production BPE tokenizer wrapping a HuggingFace ``AutoTokenizer``.

    Instances are process-wide cached by model name, so the tokenizer is
    downloaded / loaded at most once per process. Special-token ids and the
    vocabulary size are exposed as read-only properties so the model config can
    bind them dynamically (:meth:`BitNetFFTConfig.bind_tokenizer`).

    Attributes:
        name: HuggingFace tokenizer id (``gpt2``, ``Qwen/Qwen2.5-0.5B``, ...).
        bos_token_id / eos_token_id: the tokenizer's special ids (may be None).
        pad_token_id: the tokenizer's pad id, or its eos id when the underlying
            tokenizer defines no pad token (GPT-2 has none).
    """

    def __init__(self, name: str = DEFAULT_BPE_TOKENIZER, **from_kwargs) -> None:
        self.name = name
        self._tok = _load_auto_tokenizer(name, **from_kwargs)
        self._byte_level: bool | None = None
        self._buf = b""

    # -- vocabulary / special ids ---------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size)

    @property
    def bos_token_id(self) -> int | None:
        return self._tok.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        pad = self._tok.pad_token_id
        if pad is not None:
            return int(pad)
        return self.eos_token_id

    # -- encode / decode -------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self._tok.decode(list(int(i) for i in ids),
                                clean_up_tokenization_spaces=False)

    # -- streaming decode -------------------------------------------------------

    def decode_step(self, token_id: int) -> str:
        """Convert one generated token id to text, streaming-safe.

        Byte-level BPEs split multi-byte UTF-8 characters across several
        tokens; decoding each token in isolation would emit replacement
        characters. Instead the raw bytes of each token are buffered and only
        complete UTF-8 sequences are emitted, so successive calls concatenate
        into the exact decoded text. Call :meth:`reset_stream` between
        independent generations.
        """
        if self._byte_level is None:
            self._byte_level = self._probe_byte_level()
        if self._byte_level:
            piece = _token_str_to_bytes(self._tok.convert_ids_to_tokens([token_id])[0])
            if piece is None:  # pragma: no cover - non byte-level safety net
                piece = self._tok.decode([token_id]).encode("utf-8", errors="replace")
            self._buf += piece
            return self._emit_utf8()
        return self._tok.decode([token_id])  # pragma: no cover - fallback

    def reset_stream(self) -> str:
        """Flush pending bytes (incomplete UTF-8 tail) and reset the stream."""
        out = self._buf
        self._buf = b""
        return out.decode("utf-8", errors="replace") if out else ""

    # -- helpers ----------------------------------------------------------------

    def _probe_byte_level(self) -> bool:
        sample = "h\u00e9llo w\u00f6rld \U0001f600 \u65e5\u672c\u8a9e"
        ids = self.encode(sample)
        pieces = [_token_str_to_bytes(self._tok.convert_ids_to_tokens([i])[0])
                  for i in ids]
        if any(p is None for p in pieces):
            return False
        return b"".join(pieces) == sample.encode("utf-8")

    def _emit_utf8(self) -> str:
        buf = self._buf
        cut = len(buf)
        while cut > 0:
            try:
                text = buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                cut -= 1
                continue
            self._buf = buf[cut:]
            return text
        return ""

    def __repr__(self) -> str:
        return f"BPETokenizer(name={self.name!r}, vocab_size={self.vocab_size})"


@lru_cache(maxsize=8)
def _load_auto_tokenizer(name: str, **kwargs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, **kwargs)


@lru_cache(maxsize=8)
def _load_tiktoken_encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


class TikTokenizer:
    """Tiktoken BPE tokenizer wrapping a ``tiktoken.Encoding`` (o200k_base).

    ``o200k_base`` is the GPT-4o encoding: its vocabulary is intrinsic to the
    encoding at ``n_vocab`` (~200,019 tokens), so ``vocab_size`` is read from
    the encoding rather than user-supplied. Exposes the same ``encode`` /
    ``decode`` / ``vocab_size`` / ``bos_token_id`` / ``eos_token_id`` /
    ``pad_token_id`` / ``decode_step`` / ``reset_stream`` interface as
    :class:`BPETokenizer`, so the model config binds it identically.

    Attributes:
        name: tiktoken encoding name (``o200k_base``).
        eos_token_id / pad_token_id: the ``<|endoftext|>`` special token id
            when the encoding defines it, else ``None``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._enc = _load_tiktoken_encoding(name)
        self._buf = b""

    # -- vocabulary / special ids ---------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)

    @property
    def bos_token_id(self) -> int | None:
        return None  # tiktoken encodings do not define a BOS token

    @property
    def eos_token_id(self) -> int | None:
        for tok in ("<|endoftext|>",):
            try:
                return int(self._enc.encode_single_token(tok))
            except KeyError:
                continue
        return None

    @property
    def pad_token_id(self) -> int | None:
        return self.eos_token_id  # no pad token defined

    # -- encode / decode -------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids) -> str:
        return self._enc.decode(list(int(i) for i in ids))

    # -- streaming decode -------------------------------------------------------

    def decode_step(self, token_id: int) -> str:
        """Convert one generated token id to text, streaming-safe.

        Tiktoken is byte-level, so each token id maps to a raw byte string
        (:meth:`tiktoken.Encoding.decode_single_token_bytes`) that may end in
        the middle of a multi-byte UTF-8 character; bytes are buffered and only
        complete UTF-8 sequences are emitted, exactly like
        :meth:`BPETokenizer.decode_step`. Call :meth:`reset_stream` between
        independent generations.
        """
        self._buf += self._enc.decode_single_token_bytes(token_id)
        return self._emit_utf8()

    def reset_stream(self) -> str:
        """Flush pending bytes (incomplete UTF-8 tail) and reset the stream."""
        out = self._buf
        self._buf = b""
        return out.decode("utf-8", errors="replace") if out else ""

    # -- helpers ----------------------------------------------------------------

    def _emit_utf8(self) -> str:
        buf = self._buf
        cut = len(buf)
        while cut > 0:
            try:
                text = buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                cut -= 1
                continue
            self._buf = buf[cut:]
            return text
        return ""

    def __repr__(self) -> str:
        return f"TikTokenizer(name={self.name!r}, vocab_size={self.vocab_size})"


class ByteTokenizer:
    """Byte-level fallback tokenizer (id <-> raw UTF-8 byte).

    Used only when ``transformers`` is unavailable or ``load_tokenizer`` is
    asked for the explicit ``bytes`` fallback. Each id is a byte (0..255), so
    ``vocab_size`` must be at least 256 for arbitrary UTF-8 text.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if any(i >= self.vocab_size for i in ids):
            raise ValueError(
                f"text byte {max(ids)} >= vocab_size {self.vocab_size}; "
                "raise vocab_size (>= 256) or use a BPE tokenizer"
            )
        return ids

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def decode_step(self, token_id: int) -> str:
        return bytes([token_id]).decode("utf-8", errors="replace")

    def reset_stream(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={self.vocab_size})"


def load_tokenizer(
    name: str | None, vocab_size: int = 256, **from_kwargs
) -> BPETokenizer | TikTokenizer | ByteTokenizer:
    """Return a production tokenizer (BPE preferred, byte-level fallback).

    ``name`` of ``"bytes"`` forces the byte-level fallback; ``"o200k_base"``
    loads the fixed-vocabulary tiktoken encoding (:class:`TikTokenizer`).
    Otherwise a HuggingFace ``AutoTokenizer`` is used (``gpt2`` when ``name``
    is ``None``); if ``transformers`` is not installed the byte-level fallback
    is returned.
    """
    if name == _BYTE_FALLBACK_NAME:
        return ByteTokenizer(vocab_size)
    if name == TIKTOKEN_NAME:
        return TikTokenizer(name)
    try:
        import transformers  # noqa: F401
    except ImportError:
        return ByteTokenizer(vocab_size)
    return BPETokenizer(name or DEFAULT_BPE_TOKENIZER, **from_kwargs)


__all__ = [
    "BPETokenizer",
    "ByteTokenizer",
    "TikTokenizer",
    "load_tokenizer",
    "DEFAULT_BPE_TOKENIZER",
    "TIKTOKEN_NAME",
    "_BYTE_FALLBACK_NAME",
]
