"""BPE (GPT-2) data loading utilities for the FFF SLM.

Provides:
* ``tiktoken`` GPT-2 BPE encode/decode helpers
* ``get_wikitext_data()`` — download WikiText-2 / WikiText-103, tokenize,
  and cache as a memory-mapped ``uint16`` ``.bin`` file under ``data/``
* ``BPEDataset`` — **non-overlapping** windows of ``block_size`` tokens
  (``stride = block_size`` by default; never ``stride = 1``)

Example
-------
>>> from data.dataset_loader import get_wikitext_data, BPEDataset
>>> paths = get_wikitext_data(variant="wikitext-2", block_size=256)
>>> train_ds = BPEDataset(paths["train"], block_size=256)  # ~9.3k samples on WT-2
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent
CACHE_DIR = DATA_ROOT / "cache"

WikiVariant = Literal["wikitext-2", "wikitext-103"]
SplitName = Literal["train", "validation", "test"]

# GPT-2 BPE vocabulary size (tiktoken ``gpt2``); fits in uint16.
GPT2_VOCAB_SIZE = 50_257

# HuggingFace ``datasets`` configs (namespace/name required by recent hub clients)
_HF_DATASET: dict[WikiVariant, tuple[str, str]] = {
    "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
}
# Legacy short id (kept as secondary attempt)
_HF_LEGACY: dict[WikiVariant, tuple[str, str]] = {
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
}

# Direct-link fallbacks (try in order; some hosts 301 to HTTPS mirrors).
_WIKITEXT_ZIP_URLS: dict[WikiVariant, tuple[str, ...]] = {
    "wikitext-2": (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/data/wikitext-2-raw-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
        "http://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    ),
    "wikitext-103": (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/data/wikitext-103-raw-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
        "http://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    ),
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def get_gpt2_encoding():  # -> tiktoken.Encoding
    """Return the GPT-2 BPE encoding (``tiktoken.get_encoding('gpt2')``)."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tiktoken is required for BPE tokenization. "
            "Install with: pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding("gpt2")


def encode_text(text: str, *, encoding=None) -> list[int]:
    """Encode a Unicode string to GPT-2 BPE token ids."""
    enc = encoding or get_gpt2_encoding()
    return enc.encode_ordinary(text)


def decode_tokens(token_ids: list[int] | Tensor | np.ndarray, *, encoding=None) -> str:
    """Decode GPT-2 BPE token ids back to a Unicode string."""
    enc = encoding or get_gpt2_encoding()
    if isinstance(token_ids, Tensor):
        token_ids = token_ids.tolist()
    elif isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
    return enc.decode(token_ids)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


class _DownloadProgressBar(tqdm):
    """``tqdm`` progress bar wired to ``urllib`` reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download_file(url: str, dest: Path, desc: str | None = None) -> Path:
    """Download ``url`` to ``dest`` with a progress bar (skips if present).

    Follows HTTP redirects and sets a User-Agent (needed for some S3/HF mirrors).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"Downloading {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "hyperzens-fff-slm/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as out, _DownloadProgressBar(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc=desc or dest.name,
                total=total if total > 0 else None,
            ) as bar:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _load_via_huggingface(variant: WikiVariant) -> dict[SplitName, str]:
    """Load WikiText splits as plain text via HuggingFace ``datasets``."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace `datasets` is required (or use the zip fallback). "
            "Install with: pip install datasets"
        ) from exc

    last_err: Exception | None = None
    ds = None
    for repo_id, config_name in (_HF_DATASET[variant], _HF_LEGACY[variant]):
        try:
            print(f"Loading HuggingFace dataset '{repo_id}' / '{config_name}' ...")
            ds = load_dataset(repo_id, config_name)
            break
        except Exception as exc:
            last_err = exc
            print(f"  attempt failed: {exc.__class__.__name__}: {exc}")
    if ds is None:
        raise RuntimeError(f"HuggingFace load failed for {variant}") from last_err

    texts: dict[SplitName, str] = {}
    for split in ("train", "validation", "test"):
        lines = [
            row["text"]
            for row in tqdm(ds[split], desc=f"HF {split}", leave=False)
            if row["text"]
        ]
        texts[split] = "\n".join(lines)  # type: ignore[assignment]
        print(f"  {split}: {len(lines):,} rows, {len(texts[split]):,} characters")
    return texts


def _load_via_zip_fallback(variant: WikiVariant, cache_dir: Path) -> dict[SplitName, str]:
    """Download the MetaMind/Salesforce WikiText zip and read raw splits."""
    zip_path = cache_dir / f"{variant}-raw.zip"
    last_err: Exception | None = None
    for url in _WIKITEXT_ZIP_URLS[variant]:
        try:
            if zip_path.exists():
                zip_path.unlink()
            _download_file(url, zip_path, desc=f"{variant}.zip")
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            print(f"  zip URL failed: {exc}")
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
    if last_err is not None and not zip_path.exists():
        raise RuntimeError(f"All zip URLs failed for {variant}") from last_err

    folder = "wikitext-2-raw" if variant == "wikitext-2" else "wikitext-103-raw"
    split_files = {
        "train": f"{folder}/wiki.train.raw",
        "validation": f"{folder}/wiki.valid.raw",
        "test": f"{folder}/wiki.test.raw",
    }
    texts: dict[SplitName, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for split, member in split_files.items():
            if member not in names:
                matches = [n for n in names if n.endswith(Path(member).name)]
                if not matches:
                    raise FileNotFoundError(
                        f"Could not find {member} inside {zip_path}. "
                        f"Archive members (sample): {list(names)[:8]}"
                    )
                member = matches[0]
            with zf.open(member) as fh:
                raw = fh.read().decode("utf-8")
            texts[split] = raw  # type: ignore[index]
            print(f"  {split}: {len(raw):,} characters (from zip)")
    return texts


def _load_via_raw_mirrors(variant: WikiVariant, cache_dir: Path) -> dict[SplitName, str]:
    """Last-resort: fetch raw splits from mirrors, else TinyShakespeare demo corpus."""
    texts: dict[SplitName, str] = {}
    local_fallback = DATA_ROOT / "tinyshakespeare.txt"

    if variant == "wikitext-2":
        # PyTorch examples historically mirrored wikitext-2 files.
        for split, fname in (
            ("train", "wiki.train.raw"),
            ("validation", "wiki.valid.raw"),
            ("test", "wiki.test.raw"),
        ):
            dest = cache_dir / f"{variant}-{fname}"
            url = (
                "https://raw.githubusercontent.com/pytorch/examples/main/"
                f"word_language_model/data/wikitext-2/{fname}"
            )
            try:
                _download_file(url, dest, desc=f"{variant}-{split}")
                texts[split] = dest.read_text(encoding="utf-8")  # type: ignore[index]
                print(f"  {split}: {len(texts[split]):,} characters (mirror)")
            except Exception as exc:
                print(f"  mirror miss for {split}: {exc}")
                texts.clear()
                break
        if len(texts) == 3:
            return texts

    if local_fallback.exists():
        print(
            f"WARNING: WikiText download failed for '{variant}'. "
            f"Falling back to {local_fallback.name} so the pipeline stays runnable."
        )
        full = local_fallback.read_text(encoding="utf-8")
        n = len(full)
        return {
            "train": full[: int(0.9 * n)],
            "validation": full[int(0.9 * n) : int(0.95 * n)],
            "test": full[int(0.95 * n) :],
        }

    raise RuntimeError(
        f"Unable to download WikiText ({variant}). "
        "Install `datasets` with network access, or place raw files under data/cache/."
    )


def load_wikitext_texts(
    variant: WikiVariant = "wikitext-2",
    cache_dir: Path | None = None,
) -> dict[SplitName, str]:
    """Load WikiText plain-text splits with HF → zip → mirror fallbacks."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) HuggingFace datasets (preferred)
    try:
        return _load_via_huggingface(variant)
    except Exception as exc:
        print(f"HuggingFace load failed ({exc.__class__.__name__}: {exc})")
        print("Falling back to direct WikiText zip download...")

    # 2) Official zip
    try:
        return _load_via_zip_fallback(variant, cache_dir)
    except Exception as exc:
        print(f"Zip fallback failed ({exc.__class__.__name__}: {exc})")
        print("Falling back to raw mirrors / local demo corpus...")

    # 3) Mirrors / tinyshakespeare
    return _load_via_raw_mirrors(variant, cache_dir)


# ---------------------------------------------------------------------------
# Tokenize + cache
# ---------------------------------------------------------------------------


def _bin_path(variant: WikiVariant, split: SplitName, cache_dir: Path) -> Path:
    return cache_dir / f"{variant}.{split}.gpt2.bin"


def _meta_path(variant: WikiVariant, split: SplitName, cache_dir: Path) -> Path:
    return cache_dir / f"{variant}.{split}.gpt2.meta"


def tokenize_and_cache(
    text: str,
    dest_bin: Path,
    *,
    encoding=None,
    chunk_chars: int = 1_000_000,
) -> Path:
    """Encode ``text`` with GPT-2 BPE and write a ``uint16`` memory-mapped ``.bin``.

    Also writes a tiny ``.meta`` sidecar with ``num_tokens`` (uint64).
    """
    enc = encoding or get_gpt2_encoding()
    dest_bin.parent.mkdir(parents=True, exist_ok=True)
    meta = dest_bin.with_suffix(".meta")

    if dest_bin.exists() and meta.exists() and dest_bin.stat().st_size > 0:
        return dest_bin

    # Encode in character chunks to bound peak memory on large corpora (WT-103).
    tmp = dest_bin.with_suffix(dest_bin.suffix + ".partial")
    total_tokens = 0
    n_chars = len(text)
    print(f"Tokenizing → {dest_bin.name} ({n_chars:,} chars)...")

    with open(tmp, "wb") as out, tqdm(total=n_chars, unit="char", unit_scale=True, desc="BPE encode") as bar:
        for start in range(0, n_chars, chunk_chars):
            piece = text[start : start + chunk_chars]
            ids = enc.encode_ordinary(piece)
            # Safety: GPT-2 ids are in [0, 50256]
            arr = np.asarray(ids, dtype=np.uint16)
            if arr.size and int(arr.max()) >= GPT2_VOCAB_SIZE:
                raise ValueError("Token id exceeds uint16 GPT-2 vocab range")
            out.write(arr.tobytes())
            total_tokens += arr.size
            bar.update(len(piece))

    meta.write_text(str(total_tokens), encoding="utf-8")
    tmp.replace(dest_bin)
    print(f"  wrote {total_tokens:,} tokens → {dest_bin} ({dest_bin.stat().st_size / 1e6:.2f} MB)")
    return dest_bin


def load_token_array(bin_path: Path) -> np.ndarray:
    """Memory-map a ``uint16`` token file as a read-only NumPy array."""
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


def get_wikitext_data(
    variant: WikiVariant = "wikitext-2",
    block_size: int = 256,
    cache_dir: str | Path | None = None,
    force_retokenize: bool = False,
) -> dict[str, Path]:
    """Download WikiText, BPE-encode, and return paths to cached ``.bin`` files.

    Parameters
    ----------
    variant:
        ``"wikitext-2"`` (default, ~2M tokens) or ``"wikitext-103"`` (large).
    block_size:
        Sequence length used by training (stored only as metadata hint for callers).
    cache_dir:
        Cache directory (default: ``data/cache/``).
    force_retokenize:
        If True, delete existing caches and rebuild.

    Returns
    -------
    dict
        Keys ``train``, ``validation``, ``test``, ``vocab_size``, ``block_size``
        (path values for splits; ints for the rest).
    """
    _ = block_size  # reserved for future shard layout / metadata.json
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    need_download = force_retokenize
    for split in ("train", "validation", "test"):
        bin_p = _bin_path(variant, split, cache)  # type: ignore[arg-type]
        if force_retokenize and bin_p.exists():
            bin_p.unlink()
            meta = bin_p.with_suffix(".meta")
            if meta.exists():
                meta.unlink()
        if not bin_p.exists():
            need_download = True
        paths[split] = bin_p

    if need_download:
        texts = load_wikitext_texts(variant, cache_dir=cache)
        enc = get_gpt2_encoding()
        for split in ("train", "validation", "test"):
            tokenize_and_cache(texts[split], paths[split], encoding=enc)  # type: ignore[index]
    else:
        print(f"Using cached WikiText BPE bins under {cache}/")
        for split, p in paths.items():
            n = len(load_token_array(p))
            print(f"  {split}: {n:,} tokens ← {p.name}")

    return {
        "train": paths["train"],
        "validation": paths["validation"],
        "test": paths["test"],
        "vocab_size": GPT2_VOCAB_SIZE,  # type: ignore[dict-item]
        "block_size": block_size,  # type: ignore[dict-item]
        "encoding": "gpt2",  # type: ignore[dict-item]
        "variant": variant,  # type: ignore[dict-item]
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class BPEDataset(Dataset):
    """Causal LM windows over a GPT-2 BPE ``uint16`` memory-mapped token file.

    Uses **non-overlapping** chunks by default (``stride = block_size``).
    For WikiText-2 (~2.39M tokens) with ``block_size=256`` this yields ~9.3k
    samples instead of ~2.39M overlapping windows (``stride=1``).

    Each item:
        ``x``: ``(block_size,)`` LongTensor — input token ids
        ``y``: ``(block_size,)`` LongTensor — next-token targets (``x`` shifted +1)

    Parameters
    ----------
    bin_path:
        Path to a ``.bin`` produced by :func:`tokenize_and_cache` /
        :func:`get_wikitext_data`.
    block_size:
        Context length ``T`` (default ``256``).
    stride:
        Step between consecutive window starts. Defaults to ``block_size``
        (non-overlapping). Must be ``>= 1``.
    """

    def __init__(
        self,
        bin_path: str | Path,
        block_size: int = 256,
        stride: int | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        # Non-overlapping chunks: stride == block_size (never stride=1 by default).
        self.stride = int(block_size if stride is None else stride)
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        self.data = load_token_array(self.bin_path)
        # Need block_size + 1 tokens per sample (inputs + next-token target).
        if len(self.data) <= block_size:
            raise ValueError(
                f"token file {self.bin_path} has {len(self.data)} tokens; "
                f"need > block_size={block_size}"
            )

    def __len__(self) -> int:
        # Starts at 0, stride, 2*stride, ... while start + block_size < n
        # (need one extra token for the shifted target).
        max_start = len(self.data) - self.block_size - 1
        if max_start < 0:
            return 0
        return max_start // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for length {len(self)}")
        start = idx * self.stride
        # Copy out of the memmap so the tensor owns contiguous storage.
        chunk = np.array(
            self.data[start : start + self.block_size + 1],
            dtype=np.int64,
        )
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    @property
    def num_tokens(self) -> int:
        return int(len(self.data))


def build_wikitext_datasets(
    variant: WikiVariant = "wikitext-2",
    block_size: int = 256,
    cache_dir: str | Path | None = None,
    stride: int | None = None,
) -> tuple[BPEDataset, BPEDataset, BPEDataset, dict]:
    """Convenience: cache WikiText + return ``(train, val, test, meta)`` datasets."""
    meta = get_wikitext_data(variant=variant, block_size=block_size, cache_dir=cache_dir)
    train_ds = BPEDataset(meta["train"], block_size=block_size, stride=stride)
    val_ds = BPEDataset(meta["validation"], block_size=block_size, stride=stride)
    test_ds = BPEDataset(meta["test"], block_size=block_size, stride=stride)
    return train_ds, val_ds, test_ds, meta


# ---------------------------------------------------------------------------
# CLI smoke entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Download/tokenize WikiText-2 and print dataset stats."""
    import argparse

    p = argparse.ArgumentParser(description="Prepare WikiText GPT-2 BPE caches")
    p.add_argument(
        "--variant",
        choices=["wikitext-2", "wikitext-103"],
        default="wikitext-2",
    )
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    meta = get_wikitext_data(
        variant=args.variant,
        block_size=args.block_size,
        force_retokenize=args.force,
    )
    train_ds = BPEDataset(meta["train"], block_size=args.block_size)
    x, y = train_ds[0]
    print(
        f"OK | vocab={meta['vocab_size']} | tokens={train_ds.num_tokens:,} | "
        f"train_chunks={len(train_ds):,} (stride={train_ds.stride}) | "
        f"x.shape={tuple(x.shape)} y.shape={tuple(y.shape)} | "
        f"sample decode: {decode_tokens(x[:32].tolist())!r}"
    )


if __name__ == "__main__":
    main()
