#!/usr/bin/env python3
"""Adapt HyperZens FFF to modern gated-GLU LLMs (Qwen / SmolLM / Gemma).

Replaces each Transformer ``MLP`` (``gate_proj`` / ``up_proj`` / ``down_proj``)
with a gated-GLU Fast Feedforward tree:

* **SwiGLU** (Qwen/SmolLM) — ``SiLU(gate) ⊙ up`` then ``down``.
* **GeGLU** (Gemma) — ``GELU_tanh(gate) ⊙ up`` then ``down`` via
  :class:`FFFGemmaBlock`.
* **Soft / STE training** — differentiable path mixture; STE forward activates
  only the winning leaf (Hard / Triton semantics).
* **Hard inference** — discrete tree walk + one gated-GLU leaf.
* **Triton mode** — same hard leaf selection on CUDA (router policy matches
  FFF Triton Hard); leaf GEMMs run as efficient PyTorch CUDA matmuls.

Example
-------
    python fff_swiglu.py --model Qwen/Qwen2.5-0.5B --device cuda
    python fff_swiglu.py --model HuggingFaceTB/SmolLM2-360M --device cuda
    python fff_swiglu.py --smoke-synthetic   # no HF download
    python fff_swiglu.py --smoke-synthetic-geglu
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Iterator, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from device_utils import (
    amp_autocast,
    apply_hardware_optimizations,
    print_device_info,
    resolve_device,
)
from models.fff_hard_triton import is_triton_available
from models.fff_layer import FastFeedforwardLinear

RoutingMode = Literal["soft", "hard", "triton"]

DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "HuggingFaceTB/SmolLM2-360M",
)


def _init_orthogonal_fp32_(weight: Tensor, gain: float = 0.1) -> None:
    """``nn.init.orthogonal_`` in FP32 (CPU LAPACK), then copy into ``weight``.

    Avoids ``RuntimeError: "geqrf_cpu" not implemented for 'Half'`` when the
    parameter lives in FP16/BF16.
    """
    w_fp32 = torch.empty(weight.shape, dtype=torch.float32, device="cpu")
    nn.init.orthogonal_(w_fp32, gain=float(gain))
    weight.data.copy_(w_fp32.to(device=weight.device, dtype=weight.dtype))


def _init_uniform_fp32_(weight: Tensor, a: float, b: float | None = None) -> None:
    """Uniform init in FP32, then cast/copy into ``weight``."""
    if b is None:
        b = a
        a = -a
    w_fp32 = torch.empty(weight.shape, dtype=torch.float32, device="cpu")
    nn.init.uniform_(w_fp32, a=a, b=b)
    weight.data.copy_(w_fp32.to(device=weight.device, dtype=weight.dtype))


# ---------------------------------------------------------------------------
# SwiGLU FFF tree
# ---------------------------------------------------------------------------


class FastFeedforwardSwiGLU(nn.Module):
    """FFF tree whose leaves are SwiGLU experts (gate / up / down).

    Routers match :class:`~models.fff_layer.FastFeedforwardLinear` (soft path
    products, hard sign thresholds). Each leaf ``ℓ`` stores:

        ``W_gate[ℓ] ∈ R^{D×S}``, ``W_up[ℓ] ∈ R^{D×S}``, ``W_down[ℓ] ∈ R^{S×D}``

    with slice width ``S = intermediate_size / num_leaves``. Leaf output:

        SwiGLU:  ``y_ℓ = (SiLU(x W_gate) ⊙ (x W_up)) W_down``
        GeGLU:   ``y_ℓ = (GELU_tanh(x W_gate) ⊙ (x W_up)) W_down``

    Soft mixture: ``y = Σ_ℓ P(ℓ|x) · y_ℓ``. Hard / Triton: single winning leaf.

    Shapes
    ------
    x: ``(..., D)``
    returns: ``(..., D)``
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        depth: int = 4,
        init_temp: float = 1.0,
        gate_activation: Literal["silu", "gelu_tanh"] = "silu",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if d_model < 1 or intermediate_size < 1:
            raise ValueError("d_model and intermediate_size must be >= 1")
        if init_temp <= 0.0:
            raise ValueError(f"init_temp must be > 0, got {init_temp}")
        if gate_activation not in ("silu", "gelu_tanh"):
            raise ValueError(
                f"gate_activation must be 'silu' or 'gelu_tanh', got {gate_activation!r}"
            )

        self.d_model: int = int(d_model)
        self.intermediate_size: int = int(intermediate_size)
        self.depth: int = int(depth)
        self.num_leaves: int = 1 << self.depth
        self.num_routers: int = self.num_leaves - 1
        self.gate_activation: str = gate_activation
        if intermediate_size % self.num_leaves != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} not divisible by "
                f"num_leaves={self.num_leaves} (depth={depth})"
            )
        self.slice_size: int = intermediate_size // self.num_leaves

        self.router_weights = nn.Parameter(
            torch.empty(self.num_routers, self.d_model)
        )
        self.router_biases = nn.Parameter(torch.empty(self.num_routers))

        # Leaf SwiGLU projections (stored as batchable 3-D tensors).
        self.w_gate_leaf = nn.Parameter(
            torch.empty(self.num_leaves, self.d_model, self.slice_size)
        )
        self.w_up_leaf = nn.Parameter(
            torch.empty(self.num_leaves, self.d_model, self.slice_size)
        )
        self.w_down_leaf = nn.Parameter(
            torch.empty(self.num_leaves, self.slice_size, self.d_model)
        )

        self.register_buffer(
            "temperature",
            torch.tensor(float(init_temp)),
            persistent=True,
        )

        path_router_idx, path_go_right, leaf_router_inc = (
            FastFeedforwardLinear._build_path_tables(
                self.depth, self.num_leaves, self.num_routers
            )
        )
        self.register_buffer("path_router_idx", path_router_idx, persistent=False)
        self.register_buffer("path_go_right", path_go_right, persistent=False)
        self.register_buffer("leaf_router_inc", leaf_router_inc, persistent=False)

        self._last_node_decisions: Tensor | None = None
        self._last_reach_probs: Tensor | None = None
        self._last_leaf_probs: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small orthogonal routers; Xavier-style leaf slices (always via FP32)."""
        _init_orthogonal_fp32_(self.router_weights, gain=0.1)
        nn.init.zeros_(self.router_biases)
        a = math.sqrt(1.0 / self.d_model)
        _init_uniform_fp32_(self.w_gate_leaf, a)
        _init_uniform_fp32_(self.w_up_leaf, a)
        _init_uniform_fp32_(self.w_down_leaf, a)

    def set_temperature(self, temp: float) -> None:
        """Set soft-routing temperature ``τ`` (must be ``> 0``)."""
        if temp <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temp}")
        self.temperature.fill_(float(temp))

    def _flatten_input(self, x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        """Flatten leading dims to ``(N, D)``; return flat + leading shape."""
        if x.ndim < 2:
            raise ValueError(f"expected x with ndim >= 2, got shape {tuple(x.shape)}")
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"last dim {x.size(-1)} != d_model={self.d_model}"
            )
        leading = tuple(x.shape[:-1])
        return x.reshape(-1, self.d_model), leading

    def _apply_gate(self, gate: Tensor) -> Tensor:
        """Gate nonlinearity: SiLU (LLaMA/Qwen) or GELU-tanh (Gemma GeGLU)."""
        if self.gate_activation == "gelu_tanh":
            return F.gelu(gate, approximate="tanh")
        return F.silu(gate)

    def _swiglu_all_leaves(self, flat: Tensor) -> Tensor:
        """Evaluate every gated-GLU leaf.

        Parameters
        ----------
        flat:
            ``(N, D)``

        Returns
        -------
        Tensor
            ``(N, L, D)`` leaf outputs.
        """
        # gate/up: (N, L, S); down: (N, L, D)
        gate = torch.einsum("nd,lds->nls", flat, self.w_gate_leaf)
        up = torch.einsum("nd,lds->nls", flat, self.w_up_leaf)
        hidden = self._apply_gate(gate) * up
        return torch.einsum("nls,lsd->nld", hidden, self.w_down_leaf)

    def _swiglu_selected(self, flat: Tensor, leaf_ids: Tensor) -> Tensor:
        """Evaluate one gated-GLU leaf per token.

        Parameters
        ----------
        flat:
            ``(N, D)``
        leaf_ids:
            ``(N,)`` contiguous leaf indices in ``[0, L)``.

        Returns
        -------
        Tensor
            ``(N, D)``
        """
        w_g = self.w_gate_leaf[leaf_ids]  # (N, D, S)
        w_u = self.w_up_leaf[leaf_ids]
        w_d = self.w_down_leaf[leaf_ids]  # (N, S, D)
        gate = torch.einsum("nd,nds->ns", flat, w_g)
        up = torch.einsum("nd,nds->ns", flat, w_u)
        hidden = self._apply_gate(gate) * up
        return torch.einsum("ns,nsd->nd", hidden, w_d)

    def _soft_leaf_logits(self, flat: Tensor) -> tuple[Tensor, Tensor]:
        """Log-path scores ``(N, L)`` and router decisions ``(N, R)``."""
        eps = 1e-7
        router_logits = (
            F.linear(flat, self.router_weights, self.router_biases)
            / self.temperature
        )
        node_decisions = torch.sigmoid(router_logits).clamp(min=eps, max=1.0 - eps)
        log_c = F.logsigmoid(router_logits)
        log_not_c = F.logsigmoid(-router_logits)
        idx = self.path_router_idx
        log_edges = torch.where(
            self.path_go_right.unsqueeze(0),
            log_c[:, idx],
            log_not_c[:, idx],
        )
        log_leaf = log_edges.sum(dim=-1)
        return log_leaf, node_decisions

    def _cache_soft_stats(
        self, leaf_probs: Tensor, node_decisions: Tensor
    ) -> None:
        self._last_leaf_probs = leaf_probs
        self._last_node_decisions = node_decisions
        self._last_reach_probs = leaf_probs @ self.leaf_router_inc

    def forward_soft(self, x: Tensor) -> Tensor:
        """Soft mixture over all SwiGLU leaves (training / soft eval)."""
        flat, leading = self._flatten_input(x)
        log_leaf, node_decisions = self._soft_leaf_logits(flat)
        leaf_probs = F.softmax(log_leaf, dim=-1)
        self._cache_soft_stats(leaf_probs, node_decisions)
        leaf_out = self._swiglu_all_leaves(flat)  # (N, L, D)
        y = torch.einsum("nl,nld->nd", leaf_probs, leaf_out)
        return y.view(*leading, self.d_model)

    def forward_soft_ste(self, x: Tensor) -> Tensor:
        """STE hard-aware forward: one leaf in forward, soft grads in backward."""
        flat, leading = self._flatten_input(x)
        log_leaf, node_decisions = self._soft_leaf_logits(flat)
        prob_soft = F.softmax(log_leaf, dim=-1)
        self._cache_soft_stats(prob_soft, node_decisions)

        mask_hard = F.one_hot(
            torch.argmax(log_leaf, dim=-1),
            num_classes=self.num_leaves,
        ).to(dtype=prob_soft.dtype)
        mask = (mask_hard - prob_soft).detach() + prob_soft

        # Mask tokens so forward ≡ winning leaf; grads flow through soft mask.
        masked_in = flat.unsqueeze(1) * mask.unsqueeze(-1)  # (N, L, D)
        gate = torch.einsum("nld,lds->nls", masked_in, self.w_gate_leaf)
        up = torch.einsum("nld,lds->nls", masked_in, self.w_up_leaf)
        hidden = self._apply_gate(gate) * up
        leaf_out = torch.einsum("nls,lsd->nld", hidden, self.w_down_leaf)
        leaf_out = leaf_out * mask.unsqueeze(-1)
        y = leaf_out.sum(dim=1)
        return y.view(*leading, self.d_model)

    def _hard_leaf_ids(self, flat: Tensor) -> Tensor:
        """Discrete tree walk → contiguous leaf ids ``(N,)``."""
        n_batch = flat.shape[0]
        node_ids = torch.zeros(n_batch, dtype=torch.long, device=flat.device)
        for _ in range(self.depth):
            w = self.router_weights[node_ids]
            b = self.router_biases[node_ids]
            logits = torch.einsum("ni,ni->n", flat, w) + b
            go_right = logits > 0
            node_ids = (node_ids << 1) + 1 + go_right.to(torch.long)
        return node_ids - (self.num_leaves - 1)

    def forward_hard(self, x: Tensor) -> Tensor:
        """Hard tree traversal — evaluate only the selected SwiGLU leaf."""
        flat, leading = self._flatten_input(x)
        leaf_ids = self._hard_leaf_ids(flat)
        y = self._swiglu_selected(flat, leaf_ids)
        return y.view(*leading, self.d_model)

    def forward_hard_triton(self, x: Tensor) -> Tensor:
        """CUDA hard routing with Triton-aligned leaf selection.

        Router decisions use the same ``(wᵀx+b) > 0`` policy as FFF Triton Hard.
        SwiGLU leaf compute runs as gathered CUDA GEMMs (correctness-first;
        matches hard inference semantics used by the Triton engine).
        """
        if x.device.type != "cuda":
            return self.forward_hard(x)
        return self.forward_hard(x)

    def forward(self, x: Tensor, mode: RoutingMode = "soft") -> Tensor:
        """Dispatch soft / hard / triton SwiGLU FFF."""
        if mode == "soft":
            return self.forward_soft(x)
        if mode == "hard":
            return self.forward_hard(x)
        if mode == "triton":
            return self.forward_hard_triton(x)
        raise ValueError(f"mode must be soft|hard|triton, got {mode!r}")


class FFFSwiGLUBlock(nn.Module):
    """Drop-in replacement for HF Llama/Qwen SwiGLU ``MLP``.

    Calling convention: ``hidden_states (B, T, D) → (B, T, D)``.
    Training soft path uses STE; eval soft uses full mixture; ``hard`` /
    ``triton`` use discrete one-leaf SwiGLU.

    Output scale
    ------------
    Single-leaf STE / Hard / Triton activates one of ``L`` intermediate slices.
    A learnable scalar ``output_scale`` (initialized to ``√L``) restores
    activation magnitude toward the teacher's full-width SwiGLU / RMSNorm range.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        fff_depth: int = 4,
        init_temp: float = 1.0,
        *,
        leaf_out_scale: float | None = None,
        learnable_scale: bool = True,
        gate_activation: Literal["silu", "gelu_tanh"] = "silu",
    ) -> None:
        super().__init__()
        self.fff = FastFeedforwardSwiGLU(
            d_model=d_model,
            intermediate_size=intermediate_size,
            depth=fff_depth,
            init_temp=init_temp,
            gate_activation=gate_activation,
        )
        self.routing_mode: str = "soft"
        scale0 = (
            float(leaf_out_scale)
            if leaf_out_scale is not None
            else math.sqrt(float(self.fff.num_leaves))
        )
        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(scale0, dtype=torch.float32))
        else:
            self.register_buffer(
                "output_scale",
                torch.tensor(scale0, dtype=torch.float32),
                persistent=True,
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """FFF gated-GLU forward using ``self.routing_mode``, then ``× output_scale``."""
        mode = self.routing_mode
        if mode not in ("soft", "hard", "triton", "sum"):
            mode = "soft"
        if mode == "sum":
            # Sum every leaf slice → reconstruct the original dense SwiGLU
            # (diagnostic / undertrained checkpoints). No √L output_scale.
            flat, leading = self.fff._flatten_input(hidden_states)
            y = self.fff._swiglu_all_leaves(flat).sum(dim=1)
            return y.view(*leading, self.fff.d_model)
        if mode == "soft":
            if self.training:
                y = self.fff.forward_soft_ste(hidden_states)
            else:
                y = self.fff.forward_soft(hidden_states)
        elif mode == "triton":
            y = self.fff.forward_hard_triton(hidden_states)
        else:
            y = self.fff.forward_hard(hidden_states)
        # Broadcast scalar scale to activation dtype/device.
        return y * self.output_scale.to(dtype=y.dtype, device=y.device)

    def set_temperature(self, tau: float) -> None:
        self.fff.set_temperature(tau)

    def set_routing_mode(self, mode: str) -> None:
        allowed = ("soft", "hard", "triton", "sum")
        if mode not in allowed:
            raise ValueError(f"routing_mode must be one of {allowed}, got {mode!r}")
        self.routing_mode = str(mode)

    def leaf_probs(self) -> Tensor | None:
        return self.fff._last_leaf_probs


class FFFGemmaBlock(FFFSwiGLUBlock):
    """Drop-in replacement for HF Gemma GeGLU ``MLP``.

    Leaf math (Gemma ``gelu_pytorch_tanh``)::

        ``y = (GELU_tanh(x W_gate) ⊙ (x W_up)) W_down``

    Same STE / hard / Triton routing contract as :class:`FFFSwiGLUBlock`.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        fff_depth: int = 4,
        init_temp: float = 1.0,
        *,
        leaf_out_scale: float | None = None,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__(
            d_model=d_model,
            intermediate_size=intermediate_size,
            fff_depth=fff_depth,
            init_temp=init_temp,
            leaf_out_scale=leaf_out_scale,
            learnable_scale=learnable_scale,
            gate_activation="gelu_tanh",
        )


# ---------------------------------------------------------------------------
# Smart init from pretrained SwiGLU MLP
# ---------------------------------------------------------------------------


def _get_swiglu_projections(
    mlp: nn.Module,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Resolve ``gate_proj`` / ``up_proj`` / ``down_proj`` on an HF MLP."""
    missing = [
        name
        for name in ("gate_proj", "up_proj", "down_proj")
        if not hasattr(mlp, name)
    ]
    if missing:
        raise TypeError(
            f"MLP missing SwiGLU projections {missing}; "
            "expected Llama/Qwen-style gate_proj, up_proj, down_proj"
        )
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj  # type: ignore[return-value]


@torch.no_grad()
def init_fff_from_swiglu(
    fff_block: FFFSwiGLUBlock,
    swiglu_mlp: nn.Module,
    *,
    noise_std: float = 1e-3,
    router_gain: float = 0.1,
) -> None:
    """Smart-init FFF SwiGLU leaves from a pretrained HF MLP.

    HuggingFace SwiGLU MLP
    ----------------------
    * ``gate_proj.weight``: ``(I, D)``
    * ``up_proj.weight``: ``(I, D)``
    * ``down_proj.weight``: ``(D, I)``

    For ``L = 2^k`` leaves, split the intermediate axis into ``L`` slices of
    width ``S = I / L`` and assign:

        ``W_gate[i] = gate[i·S:(i+1)·S, :].T``   → ``(D, S)``
        ``W_up[i]   = up[i·S:(i+1)·S, :].T``     → ``(D, S)``
        ``W_down[i] = down[:, i·S:(i+1)·S].T``   → ``(S, D)``

    Tiny Gaussian ``N(0, noise_std²)`` breaks leaf symmetry. Routers use small
    orthogonal init. **All init math runs in float32** (CPU ``geqrf`` / noise),
    then results are copied into the parameter dtype (FP16-safe).
    """
    gate_proj, up_proj, down_proj = _get_swiglu_projections(swiglu_mlp)
    fff = fff_block.fff
    n_leaves = int(fff.num_leaves)
    d_model = int(fff.d_model)
    slice_size = int(fff.slice_size)

    w_gate = gate_proj.weight.detach()
    w_up = up_proj.weight.detach()
    w_down = down_proj.weight.detach()
    if w_gate.ndim != 2 or w_up.ndim != 2 or w_down.ndim != 2:
        raise ValueError("gate/up/down weights must be 2-D")
    if w_gate.shape != (fff.intermediate_size, d_model):
        raise ValueError(
            f"gate_proj shape {tuple(w_gate.shape)} != "
            f"(I={fff.intermediate_size}, D={d_model})"
        )
    if w_up.shape != w_gate.shape:
        raise ValueError(
            f"up_proj shape {tuple(w_up.shape)} != gate {tuple(w_gate.shape)}"
        )
    if w_down.shape != (d_model, fff.intermediate_size):
        raise ValueError(
            f"down_proj shape {tuple(w_down.shape)} != "
            f"(D={d_model}, I={fff.intermediate_size})"
        )

    # Slice + noise entirely in FP32 on CPU (LAPACK / RNG safe), then cast.
    w_gate = w_gate.detach().to(device="cpu", dtype=torch.float32)
    w_up = w_up.detach().to(device="cpu", dtype=torch.float32)
    w_down = w_down.detach().to(device="cpu", dtype=torch.float32)

    gate_leaves = torch.empty(
        n_leaves, d_model, slice_size, device="cpu", dtype=torch.float32
    )
    up_leaves = torch.empty_like(gate_leaves)
    down_leaves = torch.empty(
        n_leaves, slice_size, d_model, device="cpu", dtype=torch.float32
    )
    for i in range(n_leaves):
        start = i * slice_size
        end = start + slice_size
        gate_leaves[i] = w_gate[start:end, :].T.contiguous()
        up_leaves[i] = w_up[start:end, :].T.contiguous()
        down_leaves[i] = w_down[:, start:end].T.contiguous()

    if noise_std > 0.0:
        gate_leaves = gate_leaves + noise_std * torch.randn_like(gate_leaves)
        up_leaves = up_leaves + noise_std * torch.randn_like(up_leaves)
        down_leaves = down_leaves + noise_std * torch.randn_like(down_leaves)

    tgt_device = fff.w_gate_leaf.device
    tgt_dtype = fff.w_gate_leaf.dtype
    fff.w_gate_leaf.copy_(gate_leaves.to(device=tgt_device, dtype=tgt_dtype))
    fff.w_up_leaf.copy_(up_leaves.to(device=tgt_device, dtype=tgt_dtype))
    fff.w_down_leaf.copy_(down_leaves.to(device=tgt_device, dtype=tgt_dtype))

    # orthogonal_ requires FP32 (geqrf); never call it on Half parameters.
    _init_orthogonal_fp32_(fff.router_weights, gain=float(router_gain))
    nn.init.zeros_(fff.router_biases)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------


def get_text_config(model: Any) -> Any:
    """Return the text transformer config (handles Qwen3.5 multimodal wrappers)."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    return text_cfg if text_cfg is not None else cfg


def iter_decoder_layers(model: Any) -> Iterator[Any]:
    """Yield decoder layers from HF CausalLM / Qwen3.5 text trunks.

    Supported layouts
    ----------------
    * ``model.model.layers`` — Llama / Qwen2 / Qwen3.5 text CausalLM
    * ``model.model.language_model.layers`` — Qwen3.5 multimodal wrapper
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise TypeError("expected AutoModelForCausalLM with `.model` trunk")

    layers = getattr(inner, "layers", None)
    if layers is None:
        lang = getattr(inner, "language_model", None)
        layers = getattr(lang, "layers", None) if lang is not None else None
    if layers is None:
        raise TypeError(
            "could not find decoder layers at `.model.layers` or "
            "`.model.language_model.layers` (unexpected HF layout)"
        )
    yield from layers


def iter_fff_swiglu_blocks(model: Any) -> Iterator[FFFSwiGLUBlock]:
    """Yield injected :class:`FFFSwiGLUBlock` modules."""
    for layer in iter_decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FFFSwiGLUBlock):
            yield mlp


def set_fff_routing_mode(model: Any, mode: str) -> None:
    """Set soft / hard / triton / sum on every injected FFF SwiGLU block."""
    for block in iter_fff_swiglu_blocks(model):
        block.set_routing_mode(mode)


def set_fff_temperature(model: Any, tau: float) -> None:
    """Set soft-routing ``τ`` on every injected FFF SwiGLU block."""
    for block in iter_fff_swiglu_blocks(model):
        block.set_temperature(tau)


def patch_model_with_fff_swiglu(
    model: Any,
    *,
    fff_depth: int = 4,
    init_temp: float = 1.0,
    noise_std: float = 1e-3,
    router_gain: float = 0.1,
    block_cls: type[FFFSwiGLUBlock] = FFFSwiGLUBlock,
) -> int:
    """Replace each ``layer.mlp`` with a gated-GLU FFF block (in-place).

    Blocks are constructed and smart-initialized in **float32**. Callers that
    want BF16/FP16 inference should cast the full model **after** this returns.

    Parameters
    ----------
    block_cls:
        ``FFFSwiGLUBlock`` (SiLU) or ``FFFGemmaBlock`` (GELU-tanh GeGLU).

    Returns the number of patched layers.
    """
    text_cfg = get_text_config(model)
    d_model = int(text_cfg.hidden_size)
    intermediate = int(text_cfg.intermediate_size)
    n_leaves = 1 << int(fff_depth)
    if intermediate % n_leaves != 0:
        raise ValueError(
            f"intermediate_size={intermediate} not divisible by "
            f"2^{fff_depth}={n_leaves}"
        )

    # Keep patching on the model's current device, but force FP32 params for init.
    ref = next(model.parameters())
    patch_device = ref.device
    act = "GeGLU" if block_cls is FFFGemmaBlock else "SwiGLU"

    n_patched = 0
    for layer_idx, layer in enumerate(iter_decoder_layers(model)):
        dense_mlp = getattr(layer, "mlp", None)
        if dense_mlp is None:
            continue
        block = block_cls(
            d_model=d_model,
            intermediate_size=intermediate,
            fff_depth=fff_depth,
            init_temp=init_temp,
        )
        block = block.to(device=patch_device, dtype=torch.float32)
        init_fff_from_swiglu(
            block,
            dense_mlp,
            noise_std=noise_std,
            router_gain=router_gain,
        )
        layer.mlp = block
        n_patched += 1
        if n_patched == 1:
            print(
                f"  layer {layer_idx}: FFF {act} depth={fff_depth} leaves={n_leaves} "
                f"slice={intermediate // n_leaves} D={d_model} I={intermediate} "
                f"(init dtype=float32)"
            )
    if n_patched == 0:
        raise RuntimeError("no decoder layers with `.mlp` found to patch")
    return n_patched


def patch_model_with_fff_gemma(
    model: Any,
    *,
    fff_depth: int = 4,
    init_temp: float = 1.0,
    noise_std: float = 1e-3,
    router_gain: float = 0.1,
) -> int:
    """Replace each ``layer.mlp`` with :class:`FFFGemmaBlock` (GeGLU, FP32 init)."""
    return patch_model_with_fff_swiglu(
        model,
        fff_depth=fff_depth,
        init_temp=init_temp,
        noise_std=noise_std,
        router_gain=router_gain,
        block_cls=FFFGemmaBlock,
    )


# Alias — Gemma uses the same gate/up/down slice projection as SwiGLU HF MLPs.
init_fff_from_geglu = init_fff_from_swiglu


def load_and_patch_modern_llm(
    model_name: str,
    device: torch.device,
    *,
    fff_depth: int = 4,
    dtype: torch.dtype | None = None,
    routing_mode: str = "triton",
    block_cls: type[FFFSwiGLUBlock] = FFFSwiGLUBlock,
) -> tuple[Any, Any]:
    """Load HF CausalLM in FP32, inject gated-GLU FFF, then cast to ``dtype``.

    Order (required to avoid Half ``geqrf`` failures)
    -------------------------------------------------
    1. Load pretrained weights in **float32**.
    2. Inject / smart-init all FFF blocks in **float32**.
    3. Cast patched model to BF16/FP16 and ``.to(device)``.

    Returns ``(model, tokenizer)``.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc

    if dtype is None:
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

    act = "GeGLU" if block_cls is FFFGemmaBlock else "SwiGLU"
    print(f"Loading `{model_name}` in float32 for safe FFF init ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    print(f"Injecting FFF {act} blocks (smart-init FP32) ...")
    n = patch_model_with_fff_swiglu(
        model, fff_depth=fff_depth, block_cls=block_cls
    )

    if dtype != torch.float32:
        print(f"Casting patched model → {dtype} and moving to {device} ...")
        model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    if routing_mode == "triton" and (
        device.type != "cuda" or not is_triton_available()
    ):
        print(
            "  Triton unavailable or non-CUDA — falling back to PyTorch hard routing"
        )
        routing_mode = "hard"
    set_fff_routing_mode(model, routing_mode)
    print(f"Patched {n} MLP layers | routing_mode={routing_mode} | dtype={dtype}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Synthetic smoke (no HF download)
# ---------------------------------------------------------------------------


class _ToySwiGLUMLP(nn.Module):
    """Minimal HF-like SwiGLU MLP for offline tests."""

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyGeGLUMLP(nn.Module):
    """Minimal HF-like Gemma GeGLU MLP for offline tests."""

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


def run_synthetic_smoke(
    device: torch.device,
    *,
    fff_depth: int = 4,
    gate_activation: Literal["silu", "gelu_tanh"] = "silu",
) -> None:
    """Verify gated-GLU FFF soft / STE / hard shapes and STE≈argmax leaf."""
    is_geglu = gate_activation == "gelu_tanh"
    label = "GeGLU" if is_geglu else "SwiGLU"
    print("=" * 72)
    print(f"Synthetic {label} FFF smoke (no HF weights)")
    print("=" * 72)
    d_model, intermediate = 64, 256
    assert intermediate % (1 << fff_depth) == 0
    mlp: nn.Module
    block: FFFSwiGLUBlock
    if is_geglu:
        mlp = _ToyGeGLUMLP(d_model, intermediate).to(device)
        block = FFFGemmaBlock(
            d_model, intermediate, fff_depth=fff_depth, init_temp=0.5
        ).to(device)
        init_fff_from_geglu(block, mlp, noise_std=0.0)
    else:
        mlp = _ToySwiGLUMLP(d_model, intermediate).to(device)
        block = FFFSwiGLUBlock(
            d_model, intermediate, fff_depth=fff_depth, init_temp=0.5
        ).to(device)
        init_fff_from_swiglu(block, mlp, noise_std=0.0)

    assert block.fff.gate_activation == gate_activation
    x = torch.randn(2, 8, d_model, device=device)
    block.train()
    block.set_routing_mode("soft")
    y_ste = block(x)
    block.eval()
    y_soft = block(x)
    block.set_routing_mode("hard")
    y_hard = block(x)
    block.set_routing_mode("triton")
    y_triton = block(x)

    assert y_ste.shape == x.shape
    assert y_soft.shape == x.shape
    assert y_hard.shape == x.shape
    assert torch.allclose(y_hard, y_triton, atol=1e-5)

    # Direct GeGLU leaf math check (tanh-approx GELU).
    flat, _ = block.fff._flatten_input(x)
    leaf_ids = torch.zeros(flat.shape[0], dtype=torch.long, device=device)
    y_leaf = block.fff._swiglu_selected(flat, leaf_ids)
    w_g = block.fff.w_gate_leaf[0]
    w_u = block.fff.w_up_leaf[0]
    w_d = block.fff.w_down_leaf[0]
    if is_geglu:
        y_ref = (
            F.gelu(flat @ w_g, approximate="tanh") * (flat @ w_u)
        ) @ w_d
    else:
        y_ref = (F.silu(flat @ w_g) * (flat @ w_u)) @ w_d
    leaf_delta = (y_leaf - y_ref).abs().max().item()
    print(f"  leaf math vs ref max|Δ|={leaf_delta:.3e}")
    assert leaf_delta < 1e-5

    log_leaf, _ = block.fff._soft_leaf_logits(flat)
    leaf_ids = log_leaf.argmax(dim=-1)
    y_argmax = block.fff._swiglu_selected(flat, leaf_ids).view_as(x)
    y_argmax = y_argmax * block.output_scale.to(dtype=y_argmax.dtype)
    block.train()
    block.set_routing_mode("soft")
    y_ste2 = block(x)
    max_delta = (y_ste2 - y_argmax).abs().max().item()
    print(f"  STE vs soft-argmax leaf max|Δ|={max_delta:.3e}")
    assert max_delta < 1e-5

    block.fff.router_weights.grad = None
    y_ste2.sum().backward()
    assert block.fff.router_weights.grad is not None
    print(f"  router grad norm={block.fff.router_weights.grad.norm().item():.4f}")
    print(f"Synthetic {label} smoke OK")


# ---------------------------------------------------------------------------
# Generation throughput probe
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_generation_throughput(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 32,
) -> tuple[str, float, int]:
    """Greedy generate ``max_new_tokens``; return text, tok/s, n_new."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    n_ctx = int(
        getattr(
            get_text_config(model),
            "max_position_embeddings",
            getattr(model.config, "max_position_embeddings", 262_144),
        )
    )

    use_cuda = device.type == "cuda"
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
    else:
        t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        cond = generated if generated.size(1) <= n_ctx else generated[:, -n_ctx:]
        with amp_autocast(device):
            logits = model(input_ids=cond).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        eos = tokenizer.eos_token_id
        if eos is not None and int(next_id.item()) == int(eos):
            break

    if use_cuda:
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(end))
        elapsed = max(elapsed_ms / 1000.0, 1e-12)
    else:
        elapsed = max(time.perf_counter() - t0, 1e-12)

    n_new = int(generated.size(1) - input_ids.size(1))
    tok_s = n_new / elapsed
    text = tokenizer.decode(
        generated[0, input_ids.size(1) :].tolist(),
        skip_special_tokens=True,
    ).strip()
    return text, tok_s, n_new


def warmup_fff_swiglu(model: Any, device: torch.device, *, seq_len: int = 16) -> None:
    """Dummy forward to warm CUDA / Triton-related paths."""
    text_cfg = get_text_config(model)
    vocab = int(getattr(text_cfg, "vocab_size", getattr(model.config, "vocab_size", 1)))
    dummy = torch.randint(0, max(vocab, 1), (1, seq_len), device=device)
    with amp_autocast(device):
        _ = model(input_ids=dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FFF SwiGLU adapter for modern HF CausalLMs"
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODELS[0],
        help=f"HF model id (default: {DEFAULT_MODELS[0]})",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (cuda>mps>cpu) | cuda | mps | cpu",
    )
    p.add_argument("--fff-depth", type=int, default=4)
    p.add_argument(
        "--routing",
        type=str,
        default="triton",
        choices=("soft", "hard", "triton"),
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The future of efficient language models is",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--smoke-synthetic",
        action="store_true",
        help="Run offline SwiGLU FFF unit smoke (skip HF download)",
    )
    p.add_argument(
        "--smoke-synthetic-geglu",
        action="store_true",
        help="Run offline Gemma GeGLU FFF unit smoke (skip HF download)",
    )
    p.add_argument(
        "--gemma",
        action="store_true",
        help="Inject FFFGemmaBlock (GeGLU) instead of SwiGLU",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    apply_hardware_optimizations(device)
    print_device_info(device)
    print(f"Triton available: {is_triton_available()}")

    if args.smoke_synthetic or args.smoke_synthetic_geglu:
        act: Literal["silu", "gelu_tanh"] = (
            "gelu_tanh" if args.smoke_synthetic_geglu else "silu"
        )
        run_synthetic_smoke(
            device, fff_depth=args.fff_depth, gate_activation=act
        )
        return

    block_cls: type[FFFSwiGLUBlock] = (
        FFFGemmaBlock if args.gemma else FFFSwiGLUBlock
    )
    try:
        model, tokenizer = load_and_patch_modern_llm(
            args.model,
            device,
            fff_depth=args.fff_depth,
            routing_mode=args.routing,
            block_cls=block_cls,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load/patch `{args.model}`: {exc}", file=sys.stderr)
        print(
            "Tip: try --smoke-synthetic / --smoke-synthetic-geglu, "
            "HF login for gated Gemma, or check network / disk."
        )
        raise SystemExit(1) from exc

    print("Warmup forward ...")
    warmup_fff_swiglu(model, device)
    print("  warmup done")

    print("\n" + "=" * 72)
    print(f"Dummy generation — routing={args.routing}")
    print("=" * 72)
    print(f"Prompt: {args.prompt}")
    text, tok_s, n_new = measure_generation_throughput(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Generated Text: {text}")
    print(f"Tokens generated: {n_new}")
    print(f"Speed: {tok_s:.2f} tok/s")
    if n_new > 0:
        print(f"Latency: {1000.0 / tok_s:.2f} ms/token")
    print("=" * 72)


if __name__ == "__main__":
    main()
