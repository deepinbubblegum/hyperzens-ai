#!/usr/bin/env python3
"""Knowledge distillation: GPT-2 Small (Dense MLP) → FFF Soft-Routing Student.

Teacher
-------
HuggingFace ``gpt2`` (124M), frozen ``.eval()``.

Student
-------
Same GPT-2 stack (token embed, attention, LayerNorm, LM head) with each
block's ``mlp`` replaced by :class:`FFFBlock` (``FastFeedforwardLinear`` soft
routing + temperature ``τ``). Attention / LayerNorm / embeddings are copied
from the teacher. FFF **leaves are smart-initialized** from each Teacher MLP
(``c_fc`` / ``c_proj`` intermediate slices); routers use small orthogonal init.

Loss
----
``L = KL(student ‖ teacher) · T² + 0.5 · MSE(ffn_out) − 0.01 · H(leaf)``

Targets RTX 3060 12GB: short context, micro-batch, FP16 autocast, optional
grad accumulation. Default ``τ: 1.0 → 0.10`` over ``max_steps=2000`` on WikiText-2.

Example
-------
    pip install transformers
    python fff_distill.py --device cuda --max-steps 2000
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from device_utils import (
    amp_autocast,
    apply_hardware_optimizations,
    make_grad_scaler,
    pin_memory_for,
    print_device_info,
    resolve_device,
)
from models.fff_layer import FastFeedforwardLinear

CHECKPOINT_NAME = "fff_distill_checkpoint.pt"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _require_transformers() -> tuple[Any, Any, Any]:
    """Import HuggingFace GPT-2 APIs or exit with an install hint."""
    try:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "fff_distill.py requires HuggingFace transformers:\n"
            "  pip install transformers\n"
            f"(import error: {exc})"
        ) from exc
    return GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


# Soft type aliases filled at runtime when transformers is available.
GPT2LMHeadModel = Any  # type: ignore[misc, assignment]
GPT2Config = Any  # type: ignore[misc, assignment]
GPT2TokenizerFast = Any  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# FFF block (drop-in for GPT-2 MLP)
# ---------------------------------------------------------------------------


class FFFBlock(nn.Module):
    """GPT-2 MLP replacement: soft-routing FFF ``D → D`` with dropout.

    Matches the HuggingFace GPT-2 MLP calling convention:
    ``hidden_states (B, T, D) → (B, T, D)`` (no residual add here; the
    Transformer block adds the residual outside).

    Parameters
    ----------
    d_model:
        Residual width ``D`` (GPT-2 Small = 768).
    fff_depth:
        Tree depth; leaves = ``2**fff_depth``.
    init_temp:
        Initial soft-routing temperature ``τ``.
    dropout:
        Residual dropout after the FFF (mirrors GPT-2 ``resid_pdrop``).
    """

    def __init__(
        self,
        d_model: int,
        fff_depth: int = 4,
        init_temp: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fff = FastFeedforwardLinear(
            d_model,
            d_model,
            depth=fff_depth,
            init_temp=init_temp,
        )
        self.dropout = nn.Dropout(dropout)
        # Inference switch: "soft" | "hard" | "triton" (see set_routing_mode).
        self.routing_mode: str = "soft"

    def forward(self, hidden_states: Tensor) -> Tensor:
        """FFF forward using ``self.routing_mode`` (soft / hard / triton)."""
        mode = self.routing_mode
        if mode not in ("soft", "hard", "hard_cpp", "triton", "triton_int8", "triton_int4"):
            mode = "soft"
        y = self.fff(hidden_states, mode=mode)  # type: ignore[arg-type]
        if mode == "soft":
            return self.dropout(y)
        # Eval / generation: no dropout on hard / triton paths.
        return y

    def set_temperature(self, tau: float) -> None:
        """Update soft-routing temperature ``τ`` (annealing)."""
        self.fff.set_temperature(tau)

    def set_routing_mode(self, mode: str) -> None:
        """Set FFF routing path: ``soft``, ``hard``, or ``triton``."""
        self.routing_mode = str(mode)

    def leaf_probs(self) -> Tensor | None:
        """Last soft leaf mixture ``(N, L)`` or ``None`` if no soft forward yet."""
        return self.fff._last_leaf_probs


def _init_fff_gaussian(block: FFFBlock, std: float = 0.02) -> None:
    """Fallback small Gaussian init (used only if dense MLP init is unavailable)."""
    nn.init.normal_(block.fff.router_weights, mean=0.0, std=std)
    nn.init.zeros_(block.fff.router_biases)
    nn.init.normal_(block.fff.leaf_weights, mean=0.0, std=std)
    nn.init.zeros_(block.fff.leaf_biases)


@torch.no_grad()
def init_fff_from_dense_mlp(
    fff_block: FFFBlock,
    dense_mlp: nn.Module,
    *,
    noise_std: float = 0.001,
    router_gain: float = 0.1,
) -> None:
    """Smart-init FFF leaves from a pretrained GPT-2 MLP (``c_fc`` / ``c_proj``).

    GPT-2 MLP (HuggingFace ``Conv1D``)
    ---------------------------------
    * ``c_fc.weight``: ``(D, I)`` with ``I = 4D`` (e.g. 768×3072 on GPT-2 Small;
      or 512×2048 on a D=512 teacher).
    * ``c_proj.weight``: ``(I, D)``.

    For ``L = 2^k`` leaves, split the intermediate axis into ``L`` equal slices
    of width ``s = I / L``. Leaf ``i`` is the rank-``s`` factorization of that
    slice, folded into a single ``(D, D)`` map matching
    :class:`~models.fff_layer.FastFeedforwardLinear` leaf layout:

        ``W_leaf[i] = c_fc[:, i·s:(i+1)·s] @ c_proj[i·s:(i+1)·s, :]``
        ``b_leaf[i] = b_fc[i·s:(i+1)·s] @ c_proj[i·s:(i+1)·s, :] + b_proj``

    A tiny Gaussian ``N(0, noise_std²)`` breaks exact leaf symmetry. Router
    weights are small **orthogonal** matrices (QR / ``nn.init.orthogonal_``).

    Parameters
    ----------
    fff_block:
        Student :class:`FFFBlock` to overwrite in-place.
    dense_mlp:
        Teacher layer MLP with ``c_fc`` and ``c_proj`` (GPT-2 style).
    noise_std:
        Symmetry-breaking noise on leaf weights / biases (default ``1e-3``).
    router_gain:
        Gain for orthogonal router initialization.
    """
    if not hasattr(dense_mlp, "c_fc") or not hasattr(dense_mlp, "c_proj"):
        raise TypeError(
            "dense_mlp must expose GPT-2-style c_fc / c_proj (Conv1D) modules"
        )

    fff = fff_block.fff
    n_leaves = int(fff.num_leaves)
    d_model = int(fff.in_features)
    if fff.out_features != d_model:
        raise ValueError(
            "init_fff_from_dense_mlp expects square FFF (D→D); "
            f"got in={fff.in_features}, out={fff.out_features}"
        )

    w_fc = dense_mlp.c_fc.weight.detach()
    w_proj = dense_mlp.c_proj.weight.detach()
    # HF Conv1D: c_fc (D, I), c_proj (I, D)
    if w_fc.ndim != 2 or w_proj.ndim != 2:
        raise ValueError("c_fc / c_proj weights must be 2-D")
    if w_fc.shape[0] != d_model or w_proj.shape[1] != d_model:
        raise ValueError(
            f"MLP width mismatch: FFF D={d_model}, "
            f"c_fc={tuple(w_fc.shape)}, c_proj={tuple(w_proj.shape)}"
        )
    if w_fc.shape[1] != w_proj.shape[0]:
        raise ValueError(
            f"intermediate dim mismatch: c_fc[..., {w_fc.shape[1]}] vs "
            f"c_proj[{w_proj.shape[0]}, ...]"
        )

    intermediate = int(w_fc.shape[1])
    if intermediate % n_leaves != 0:
        raise ValueError(
            f"intermediate={intermediate} not divisible by n_leaves={n_leaves} "
            f"(fff_depth={fff.depth}); choose depth so 2^depth | I"
        )
    slice_size = intermediate // n_leaves

    b_fc = (
        dense_mlp.c_fc.bias.detach()
        if dense_mlp.c_fc.bias is not None
        else torch.zeros(intermediate, device=w_fc.device, dtype=w_fc.dtype)
    )
    b_proj = (
        dense_mlp.c_proj.bias.detach()
        if dense_mlp.c_proj.bias is not None
        else torch.zeros(d_model, device=w_proj.device, dtype=w_proj.dtype)
    )

    device = fff.leaf_weights.device
    dtype = fff.leaf_weights.dtype
    w_fc = w_fc.to(device=device, dtype=torch.float32)
    w_proj = w_proj.to(device=device, dtype=torch.float32)
    b_fc = b_fc.to(device=device, dtype=torch.float32)
    b_proj = b_proj.to(device=device, dtype=torch.float32)

    leaf_w = torch.empty(n_leaves, d_model, d_model, device=device, dtype=torch.float32)
    leaf_b = torch.empty(n_leaves, d_model, device=device, dtype=torch.float32)
    for i in range(n_leaves):
        start = i * slice_size
        end = start + slice_size
        w_in = w_fc[:, start:end]  # (D, s)
        w_out = w_proj[start:end, :]  # (s, D)
        leaf_w[i] = w_in @ w_out
        leaf_b[i] = b_fc[start:end] @ w_out + b_proj

    if noise_std > 0.0:
        leaf_w = leaf_w + noise_std * torch.randn_like(leaf_w)
        leaf_b = leaf_b + noise_std * torch.randn_like(leaf_b)

    fff.leaf_weights.copy_(leaf_w.to(dtype=dtype))
    fff.leaf_biases.copy_(leaf_b.to(dtype=dtype))

    # Routers: small orthogonal rows / matrix (symmetry-breaking random splits).
    nn.init.orthogonal_(fff.router_weights, gain=float(router_gain))
    nn.init.zeros_(fff.router_biases)


# ---------------------------------------------------------------------------
# Teacher / Student construction
# ---------------------------------------------------------------------------


def load_gpt2_teacher(model_name: str, device: torch.device) -> Any:
    """Load pretrained GPT-2, freeze all parameters, set eval mode."""
    _, GPT2LMHeadModelCls, _ = _require_transformers()
    teacher = GPT2LMHeadModelCls.from_pretrained(model_name)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_fff_student_from_teacher(
    teacher: Any,
    *,
    fff_depth: int = 4,
    init_temp: float = 1.0,
    fff_init_std: float = 0.02,
    leaf_noise_std: float = 0.001,
    router_gain: float = 0.1,
) -> Any:
    """Clone GPT-2 structure; replace each ``mlp`` with :class:`FFFBlock`.

    Copies embeddings, attention, LayerNorm, and LM head from ``teacher``.
    Each FFF block is **smart-initialized** from that layer's dense MLP via
    :func:`init_fff_from_dense_mlp` (intermediate axis sliced across leaves).
    ``fff_init_std`` is retained for API compatibility (router gain fallback
    only if dense init fails).
    """
    _, GPT2LMHeadModelCls, _ = _require_transformers()
    config = teacher.config
    student = GPT2LMHeadModelCls(config)
    # Full structural clone of weights (mlp will be overwritten next).
    student.load_state_dict(teacher.state_dict(), strict=True)

    d_model = int(config.n_embd)
    dropout = float(config.resid_pdrop)
    n_leaves = 1 << int(fff_depth)
    print(
        f"Smart FFF init from Teacher MLP: depth={fff_depth} "
        f"leaves={n_leaves} leaf_noise={leaf_noise_std} router_gain={router_gain}"
    )

    for layer_idx, (s_block, t_block) in enumerate(
        zip(student.transformer.h, teacher.transformer.h)
    ):
        fff_block = FFFBlock(
            d_model=d_model,
            fff_depth=fff_depth,
            init_temp=init_temp,
            dropout=dropout,
        )
        try:
            init_fff_from_dense_mlp(
                fff_block,
                t_block.mlp,
                noise_std=leaf_noise_std,
                router_gain=router_gain,
            )
        except (TypeError, ValueError) as exc:
            print(
                f"  layer {layer_idx}: dense MLP init failed ({exc}); "
                f"falling back to Gaussian std={fff_init_std}"
            )
            _init_fff_gaussian(fff_block, std=fff_init_std)
        s_block.mlp = fff_block

    student.train()
    return student


def set_student_temperature(student: Any, tau: float) -> None:
    """Broadcast soft-routing ``τ`` to every FFFBlock."""
    for block in student.transformer.h:
        mlp = block.mlp
        if isinstance(mlp, FFFBlock):
            mlp.set_temperature(tau)


def set_student_routing_mode(student: Any, mode: str) -> None:
    """Broadcast FFF routing mode (``soft`` / ``hard`` / ``triton``) to all blocks."""
    for block in iter_fff_blocks(student):
        block.set_routing_mode(mode)


def iter_fff_blocks(student: Any) -> Iterator[FFFBlock]:
    for block in student.transformer.h:
        if isinstance(block.mlp, FFFBlock):
            yield block.mlp


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def compute_leaf_entropy_loss(
    routing_probs: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Mean-leaf occupancy entropy ``H(p) = -Σ_ℓ p̄_ℓ log p̄_ℓ`` (nats).

    Higher entropy ⇒ more uniform leaf usage. Used as ``−λ H`` in the total
    loss so the optimizer *maximizes* uniformity.

    Parameters
    ----------
    routing_probs:
        Soft leaf mixture ``(N, L)`` from one or more FFF layers (concat OK).
    """
    if routing_probs.ndim != 2:
        raise ValueError(f"routing_probs must be (N, L), got {tuple(routing_probs.shape)}")
    p_bar = routing_probs.mean(dim=0).clamp(min=eps)
    p_bar = p_bar / p_bar.sum().clamp(min=eps)
    return -(p_bar * p_bar.log()).sum()


def gather_student_leaf_probs(student: Any) -> Tensor:
    """Concatenate cached leaf probs from all FFF blocks → ``(Σ N_ℓ, L)``."""
    chunks: list[Tensor] = []
    for fff_block in iter_fff_blocks(student):
        p = fff_block.leaf_probs()
        if p is not None:
            chunks.append(p)
    if not chunks:
        raise RuntimeError("no leaf probs cached — run a soft student forward first")
    return torch.cat(chunks, dim=0)


@dataclass
class DistillLossWeights:
    """Coefficients for the combined distillation objective."""

    kl_temperature: float = 2.0  # softmax temperature ``T`` (not FFF ``τ``)
    mse_coef: float = 0.5
    entropy_coef: float = 0.01


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_ffn_hidden: Tensor,
    teacher_ffn_hidden: Tensor,
    routing_probs: Tensor,
    weights: DistillLossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combined KD loss (KL + MSE − entropy).

    ``loss_kl = KLDiv(log_softmax(s/T), softmax(t/T)) * T²`` (batchmean)
    ``loss_mse = MSE(student_ffn, teacher_ffn)``
    ``loss_entropy = H(mean leaf occupancy)``
    ``total = loss_kl + 0.5 * loss_mse - 0.01 * loss_entropy``
    """
    t = float(weights.kl_temperature)
    t = max(t, 1e-6)

    loss_kl = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
    ) * (t**2)

    loss_mse = F.mse_loss(student_ffn_hidden, teacher_ffn_hidden)
    loss_entropy = compute_leaf_entropy_loss(routing_probs)

    total = (
        loss_kl
        + weights.mse_coef * loss_mse
        - weights.entropy_coef * loss_entropy
    )
    parts = {
        "kl": loss_kl,
        "mse": loss_mse,
        "entropy": loss_entropy,
        "total": total,
    }
    return total, parts


# ---------------------------------------------------------------------------
# Forward with FFN-output capture (for MSE)
# ---------------------------------------------------------------------------


class _FFNCapture:
    """Register forward hooks on each block ``mlp`` to record outputs."""

    def __init__(self, model: Any) -> None:
        self.outputs: list[Tensor] = []
        self._hooks: list[Any] = []
        for block in model.transformer.h:
            self._hooks.append(
                block.mlp.register_forward_hook(self._make_hook())
            )

    def _make_hook(self):  # type: ignore[no-untyped-def]
        def hook(_module: nn.Module, _inp: tuple[Tensor, ...], out: Tensor) -> None:
            self.outputs.append(out)

        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def stacked(self) -> Tensor:
        """Stack layer FFN outs → ``(n_layer, B, T, D)`` then flatten for MSE."""
        if not self.outputs:
            raise RuntimeError("no FFN outputs captured")
        return torch.stack(self.outputs, dim=0)


@torch.no_grad()
def teacher_forward_capture(
    teacher: Any,
    input_ids: Tensor,
    capture: _FFNCapture,
) -> tuple[Tensor, Tensor]:
    """Teacher forward; returns ``(logits, stacked_ffn_hidden)``."""
    capture.clear()
    out = teacher(input_ids=input_ids)
    logits = out.logits
    ffn = capture.stacked()
    return logits, ffn


def student_forward_capture(
    student: Any,
    input_ids: Tensor,
    capture: _FFNCapture,
) -> tuple[Tensor, Tensor]:
    """Student forward (grad-enabled); returns ``(logits, stacked_ffn_hidden)``."""
    capture.clear()
    out = student(input_ids=input_ids)
    logits = out.logits
    ffn = capture.stacked()
    return logits, ffn


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TokenChunkDataset(Dataset[Tensor]):
    """Non-overlapping fixed-length chunks from a 1-D token tensor."""

    def __init__(self, tokens: Tensor, block_size: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1-D")
        self.block_size = block_size
        n = (tokens.numel() // block_size) * block_size
        self.data = tokens[:n].contiguous()
        self.n_chunks = max(n // block_size, 0)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tensor:
        start = idx * self.block_size
        return self.data[start : start + self.block_size]


def load_wikitext_tokens(
    tokenizer: Any,
    *,
    split: str = "train",
    max_chars: int | None = 2_000_000,
) -> Tensor:
    """Tokenize WikiText-2 with the GPT-2 tokenizer → ``int64`` 1-D tensor.

    Uses ``Salesforce/wikitext`` (required by recent ``huggingface_hub``).
    Falls back to the legacy ``wikitext`` id, then to the MetaMind/HF zip.
    """
    text = _load_wikitext2_text(split=split)
    if max_chars is not None:
        text = text[:max_chars]
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def _load_wikitext2_text(*, split: str = "train") -> str:
    """Fetch WikiText-2 raw text for ``split`` (train/validation/test)."""
    last_err: Exception | None = None

    # 1) HuggingFace datasets — namespace/name first (fixes HfUriError on 'wikitext')
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None  # type: ignore[assignment]
        last_err = ImportError("datasets not installed")

    if load_dataset is not None:
        for repo_id, config_name in (
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("wikitext", "wikitext-2-raw-v1"),
        ):
            try:
                print(f"Loading HuggingFace '{repo_id}' / '{config_name}' ({split}) ...")
                ds = load_dataset(repo_id, config_name, split=split)
                text = "\n".join(t for t in ds["text"] if t and str(t).strip())
                if text.strip():
                    print(f"  loaded {len(text):,} characters")
                    return text
            except Exception as exc:
                last_err = exc
                print(f"  attempt failed: {exc.__class__.__name__}: {exc}")

    # 2) Zip fallback (same mirrors as data/dataset_loader.py)
    print("Falling back to WikiText-2 zip download ...")
    try:
        return _load_wikitext2_text_from_zip(split=split)
    except Exception as exc:
        last_err = exc
        print(f"  zip fallback failed: {exc}")

    raise SystemExit(
        "Could not load WikiText-2.\n"
        "  pip install -U datasets huggingface_hub\n"
        "  or check network access to huggingface.co\n"
        f"Last error: {last_err}"
    )


def _load_wikitext2_text_from_zip(*, split: str = "train") -> str:
    """Download WikiText-2 raw zip and return the requested split text."""
    import urllib.request
    import zipfile

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "wikitext-2-raw-v1.zip"
    urls = (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/data/wikitext-2-raw-v1.zip",
        "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    )
    if not zip_path.exists():
        last_err: Exception | None = None
        for url in urls:
            try:
                print(f"  downloading {url} ...")
                req = urllib.request.Request(
                    url, headers={"User-Agent": "hyperzens-ai/fff_distill"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp, open(
                    zip_path, "wb"
                ) as out:
                    out.write(resp.read())
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                zip_path.unlink(missing_ok=True)
        if last_err is not None and not zip_path.exists():
            raise RuntimeError("all WikiText-2 zip URLs failed") from last_err

    split_map = {
        "train": "wikitext-2-raw/wiki.train.raw",
        "validation": "wikitext-2-raw/wiki.valid.raw",
        "valid": "wikitext-2-raw/wiki.valid.raw",
        "test": "wikitext-2-raw/wiki.test.raw",
    }
    member = split_map.get(split)
    if member is None:
        raise ValueError(f"unknown split {split!r}; use train|validation|test")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Some archives use slightly different inner paths — search by suffix.
        names = zf.namelist()
        path = next((n for n in names if n.endswith(member.split("/")[-1])), None)
        if path is None:
            path = member if member in names else None
        if path is None:
            raise FileNotFoundError(
                f"split file for {split!r} not in zip; members sample={names[:8]}"
            )
        text = zf.read(path).decode("utf-8", errors="replace")
    print(f"  zip loaded {len(text):,} characters ({split})")
    return text


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def annealed_tau(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.1,
) -> float:
    """Exponential ``τ: tau_start → tau_end`` over optimizer steps."""
    if total_steps <= 1:
        return tau_end
    t = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(tau_start * (tau_end / tau_start) ** t)


def cosine_lr(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return lr_max * float(step + 1) / float(max(warmup_steps, 1))
    if total_steps <= warmup_steps:
        return lr_min
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cos


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


@dataclass
class DistillConfig:
    model_name: str = "gpt2"
    fff_depth: int = 4
    init_tau: float = 1.0
    min_tau: float = 0.1
    fff_init_std: float = 0.02
    kl_temperature: float = 2.0
    mse_coef: float = 0.5
    entropy_coef: float = 0.01
    block_size: int = 128
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    log_every: int = 20
    save_every: int = 500
    seed: int = 42
    max_train_chars: int = 2_000_000
    device: str = "auto"
    checkpoint: str = CHECKPOINT_NAME


def build_argparser() -> argparse.ArgumentParser:
    d = DistillConfig()
    p = argparse.ArgumentParser(description="GPT-2 → FFF knowledge distillation")
    p.add_argument("--model-name", type=str, default=d.model_name)
    p.add_argument("--fff-depth", type=int, default=d.fff_depth)
    p.add_argument("--init-tau", type=float, default=d.init_tau)
    p.add_argument("--min-tau", type=float, default=d.min_tau)
    p.add_argument("--fff-init-std", type=float, default=d.fff_init_std)
    p.add_argument("--kl-temperature", type=float, default=d.kl_temperature)
    p.add_argument("--mse-coef", type=float, default=d.mse_coef)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--block-size", type=int, default=d.block_size)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=d.grad_accum_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--min-lr", type=float, default=d.min_lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max-train-chars", type=int, default=d.max_train_chars)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    return p


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_argparser().parse_args()
    cfg = DistillConfig(
        model_name=args.model_name,
        fff_depth=args.fff_depth,
        init_tau=args.init_tau,
        min_tau=args.min_tau,
        fff_init_std=args.fff_init_std,
        kl_temperature=args.kl_temperature,
        mse_coef=args.mse_coef,
        entropy_coef=args.entropy_coef,
        block_size=args.block_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        max_train_chars=args.max_train_chars,
        device=args.device,
        checkpoint=args.checkpoint,
    )

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    print("=" * 72)
    print("FFF Knowledge Distillation — GPT-2 Teacher → FFF Student")
    print("=" * 72)
    print_device_info(device)
    print(f"config: {asdict(cfg)}")

    print("\nLoading GPT-2 tokenizer / WikiText-2 ...")
    _, _, GPT2TokenizerFastCls = _require_transformers()
    tokenizer = GPT2TokenizerFastCls.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = load_wikitext_tokens(
        tokenizer, split="train", max_chars=cfg.max_train_chars
    )
    dataset = TokenChunkDataset(tokens, cfg.block_size)
    if len(dataset) == 0:
        raise SystemExit("empty dataset — increase --max-train-chars")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory_for(device),
    )
    print(f"train chunks={len(dataset)} block_size={cfg.block_size}")

    print(f"\nLoading teacher `{cfg.model_name}` (frozen) ...")
    teacher = load_gpt2_teacher(cfg.model_name, device)
    print("Building FFF student (attn/LN copied, FFF Gaussian init) ...")
    student = build_fff_student_from_teacher(
        teacher,
        fff_depth=cfg.fff_depth,
        init_temp=cfg.init_tau,
        fff_init_std=cfg.fff_init_std,
    ).to(device)

    n_fff = sum(1 for _ in iter_fff_blocks(student))
    n_student = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"FFF blocks={n_fff} | student trainable params={n_student:,} | teacher={n_teacher:,}")

    loss_w = DistillLossWeights(
        kl_temperature=cfg.kl_temperature,
        mse_coef=cfg.mse_coef,
        entropy_coef=cfg.entropy_coef,
    )
    teacher_cap = _FFNCapture(teacher)
    student_cap = _FFNCapture(student)

    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = make_grad_scaler(device)

    ckpt_path = Path(cfg.checkpoint)
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    running: dict[str, float] = {"kl": 0.0, "mse": 0.0, "entropy": 0.0, "total": 0.0}
    log_count = 0

    pbar = tqdm(range(cfg.max_steps), desc="distill", dynamic_ncols=True)
    for step in pbar:
        tau = annealed_tau(step, cfg.max_steps, cfg.init_tau, cfg.min_tau)
        set_student_temperature(student, tau)
        lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.min_lr, cfg.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        input_ids = batch.to(device, non_blocking=True)

        with amp_autocast(device):
            teacher_logits, teacher_ffn = teacher_forward_capture(
                teacher, input_ids, teacher_cap
            )
            student_logits, student_ffn = student_forward_capture(
                student, input_ids, student_cap
            )
            # Align shapes: (n_layer, B, T, D)
            routing = gather_student_leaf_probs(student)
            loss, parts = distillation_loss(
                student_logits,
                teacher_logits.detach(),
                student_ffn,
                teacher_ffn.detach(),
                routing,
                loss_w,
            )
            loss = loss / float(cfg.grad_accum_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            running[k] += float(parts[k].detach().item())
        log_count += 1

        if (step + 1) % cfg.log_every == 0 or step == 0:
            inv = 1.0 / max(log_count, 1)
            msg = (
                f"step={step + 1}/{cfg.max_steps} "
                f"loss={running['total'] * inv:.4f} "
                f"kl={running['kl'] * inv:.4f} "
                f"mse={running['mse'] * inv:.4f} "
                f"H={running['entropy'] * inv:.4f} "
                f"τ={tau:.3f} lr={lr:.2e}"
            )
            pbar.set_postfix_str(
                f"L={running['total'] * inv:.3f} τ={tau:.2f}", refresh=False
            )
            tqdm.write(msg)
            running = {k: 0.0 for k in running}
            log_count = 0

        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.max_steps:
            payload = {
                "step": step + 1,
                "config": asdict(cfg),
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fff_depth": cfg.fff_depth,
                "model_name": cfg.model_name,
            }
            torch.save(payload, ckpt_path)
            tqdm.write(f"saved checkpoint → {ckpt_path}")

    teacher_cap.close()
    student_cap.close()
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60.0:.1f} min. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
