"""Decoder-only causal LM with Fast Feedforward (FFF) blocks.

Llama / NanoGPT style stack:
    TokenEmbedding → N × TransformerBlock(Attn + FFF) → Norm → LM Head

The standard MLP (Linear → GELU → Linear) is replaced by
:class:`~models.fff_layer.FastFeedforwardLinear` (``d_model → d_model``).

Positional information uses **RoPE** (no learned position table) for memory
efficiency and length flexibility up to ``block_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fff_layer import FastFeedforwardLinear, RoutingMode


@dataclass
class FFFConfig:
    """Configuration for :class:`FFFTransformer`.

    Attributes
    ----------
    vocab_size:
        Tokenizer vocabulary size ``V``.
    n_layer:
        Number of transformer blocks ``N``.
    n_head:
        Number of attention heads ``H``. Must divide ``n_embd``.
    n_embd:
        Model / residual stream width ``D``.
    block_size:
        Maximum sequence length ``T_max`` (RoPE cache / KV length limit).
    dropout:
        Dropout on attention probs and residual paths.
    bias:
        If ``True``, use biases in attention projections and norms that support it.
    fff_depth:
        Tree depth ``d`` for each FFF block (leaves = ``2^d``).
    init_temp:
        Initial soft-routing temperature ``τ`` for all FFF layers.
    rope_theta:
        RoPE base frequency ``θ`` (LLaMA default ``10000``).
    tie_weights:
        Share token embedding weights with the LM head (saves ``V·D`` params).
    """

    vocab_size: int = 50_257  # GPT-2 BPE (tiktoken ``gpt2``)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    fff_depth: int = 6  # 2^6 = 64 leaves per FFF block
    init_temp: float = 1.0
    rope_theta: float = 10_000.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.fff_depth < 1:
            raise ValueError(f"fff_depth must be >= 1, got {self.fff_depth}")
        if self.init_temp <= 0.0:
            raise ValueError(f"init_temp must be > 0, got {self.init_temp}")


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (LLaMA-style, no mean centering).

    ``y = x / rms(x) * weight``, with ``rms = sqrt(mean(x²) + eps)``.

    Shapes: ``x (..., D) → (..., D)``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Compute in float32 for stability under mixed / low precision.
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.float()).to(orig_dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """RoPE helper: ``(..., D) → (-x_{D/2:}, x_{:D/2})`` interleaved pairs."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query/key.

    Parameters
    ----------
    q, k:
        ``(B, H, T, Dh)``
    cos, sin:
        ``(T, Dh)`` broadcast over batch/heads.

    Returns
    -------
    q_rot, k_rot:
        Same shapes as ``q``, ``k``.
    """
    # Broadcast: (1, 1, T, Dh)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Uses ``F.scaled_dot_product_attention`` (Flash / mem-efficient kernels when
    available) with ``is_causal=True`` so a full ``(T, T)`` mask is not stored.
    """

    def __init__(self, config: FFFConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE, "
                f"got n_embd={config.n_embd}, n_head={config.n_head}"
            )

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE cache: (block_size, head_dim) cos/sin — small vs learned PE table.
        cos, sin = self._build_rope_cache(
            seq_len=config.block_size,
            head_dim=self.head_dim,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _build_rope_cache(
        seq_len: int, head_dim: int, theta: float
    ) -> tuple[Tensor, Tensor]:
        """Precompute RoPE cos/sin of shape ``(T_max, Dh)``."""
        half = head_dim // 2
        freq_seq = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freq_seq / half))
        positions = torch.arange(seq_len, dtype=torch.float32)
        # (T, half)
        freqs = torch.outer(positions, inv_freq)
        # Duplicate to full head_dim for rotate-half formulation: (T, Dh)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            ``(B, T, D)``

        Returns
        -------
        Tensor
            ``(B, T, D)``
        """
        bsz, seq_len, _ = x.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.block_size}"
            )

        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # (B, H, T, Dh)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:seq_len].to(dtype=q.dtype)
        sin = self.rope_sin[:seq_len].to(dtype=q.dtype)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # SDPA: causal, no explicit (T,T) mask allocation when backend allows.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, Dh)

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    """Pre-norm block: Causal MHA + FFF feedforward (replaces MLP).

    Residual form:
        ``x ← x + Attn(RMSNorm(x))``
        ``x ← x + FFF(RMSNorm(x))``
    """

    def __init__(self, config: FFFConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        # Drop-in replacement for Linear→GELU→Linear: single FFF map D→D.
        self.fff = FastFeedforwardLinear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            depth=config.fff_depth,
            init_temp=config.init_temp,
        )
        self.fff_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mode: RoutingMode = "soft") -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x:
            ``(B, T, D)``
        mode:
            FFF routing mode forwarded to :class:`FastFeedforwardLinear`.

        Returns
        -------
        x_out:
            ``(B, T, D)``
        balance_loss:
            Scalar tree load-balancing loss from this block's FFF
            (``0`` in hard mode).
        """
        x = x + self.attn(self.ln_1(x))

        h = self.ln_2(x)
        # FFF sees flat tokens as ``(B*T, D)`` internally; keep ``(B, T, D)`` here.
        fff_out = self.fff(h, mode=mode)
        x = x + self.fff_dropout(fff_out)

        if mode == "soft":
            balance_loss = self.fff.compute_balance_loss()
        else:
            balance_loss = x.new_zeros(())

        return x, balance_loss


class FFFTransformer(nn.Module):
    """Decoder-only causal language model with FFF feedforward layers.

    Forward
    -------
    ``logits, total_balance_loss = model(input_ids, mode="soft"|"hard")``

    - ``logits``: ``(B, T, V)``
    - ``total_balance_loss``: scalar sum of per-block FFF balance losses
      (``0`` under hard routing)

    Soft mode is for training (differentiable mixture + balance term).
    Hard mode is for inference (``O(depth)`` path per token in each FFF).
    """

    def __init__(self, config: FFFConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # Share embedding matrix — LM head reads ``wte.weight`` (saves V·D).
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Residual / output projections: GPT-2 style scaled init.
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # FastFeedforwardLinear keeps its own reset_parameters().

    def forward(
        self,
        input_ids: Tensor,
        mode: RoutingMode = "soft",
    ) -> tuple[Tensor, Tensor]:
        """Run the causal LM.

        Parameters
        ----------
        input_ids:
            Token ids ``(B, T)`` with ``T <= block_size``.
        mode:
            ``"soft"`` (train) or ``"hard"`` (infer) FFF routing.

        Returns
        -------
        logits:
            ``(B, T, vocab_size)``
        total_balance_loss:
            Scalar ``Σ_layers L_balance`` (detached from hard mode zeros).
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}"
            )
        _bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}"
            )

        x = self.drop(self.wte(input_ids))  # (B, T, D)

        total_balance_loss = x.new_zeros(())
        for block in self.blocks:
            x, block_balance = block(x, mode=mode)
            total_balance_loss = total_balance_loss + block_balance

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)
        return logits, total_balance_loss

    def set_temperature(self, temp: float) -> None:
        """Broadcast temperature ``τ`` to every FFF layer (annealing schedule)."""
        for layer in self.fff_layers():
            layer.set_temperature(temp)

    def get_routing_diagnostics(self) -> dict[str, float]:
        """Aggregate :meth:`FastFeedforwardLinear.get_routing_diagnostics` over blocks.

        Averages leaf utilization, router split ratio, and grad norms across
        all FFF layers. Requires a prior soft forward (+ backward for grads).
        """
        diagnostics = [block.fff.get_routing_diagnostics() for block in self.blocks]
        keys = (
            "leaf_utilization_pct",
            "router_split_ratio_mean",
            "router_split_ratio_std",
            "router_grad_norm",
            "leaf_grad_norm",
            "leaf_entropy_norm",
            "balance_loss",
            "num_active_leaves",
            "temperature",
        )
        out: dict[str, float] = {}
        for key in keys:
            out[key] = sum(float(d[key]) for d in diagnostics) / len(diagnostics)
        out["n_fff_layers"] = float(len(diagnostics))
        # Worst-layer utilization (collapse detector)
        out["leaf_utilization_pct_min"] = min(
            float(d["leaf_utilization_pct"]) for d in diagnostics
        )
        return out


    def fff_layers(self) -> Iterable[FastFeedforwardLinear]:
        """Yield all :class:`FastFeedforwardLinear` modules in block order."""
        for block in self.blocks:
            yield block.fff

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters; optionally exclude position-unrelated embed table.

        With weight tying, ``lm_head`` shares ``wte`` so embeddings are counted once.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Rough model FLOPs utilization vs A100 bfloat16 peak (312 TFLOPS).

        Uses a dense-transformer-style estimate for attention + treating FFF soft
        forward as denser than hard (approximate; for logging only).
        """
        cfg = self.config
        n_layer, n_head, n_embd, T = cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size
        # Attention ~ 12 N D² + 2 N D T (causal SDPA proxy) per token; FFF soft ≈
        # leaf mixture cost ~ N · L · D² proxy. Keep GPT-2-like formula for parity.
        flops_per_token = 6 * self.get_num_params() + 12 * n_layer * n_head * T * (n_embd // n_head)
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        mode: RoutingMode = "hard",
    ) -> Tensor:
        """Autoregressive greedy/sample generation (defaults to hard FFF routing).

        Parameters
        ----------
        input_ids:
            Conditioning tokens ``(B, T)``.
        max_new_tokens:
            Number of tokens to append.
        temperature:
            Softmax temperature for sampling (``1.0`` = unmodified logits).
        top_k:
            If set, restrict sampling to top-``k`` logits.
        mode:
            FFF routing mode (``"hard"`` recommended for inference speed).

        Returns
        -------
        Tensor
            ``(B, T + max_new_tokens)``
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size.
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.block_size
                else input_ids[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond, mode=mode)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# Dense MLP baseline (same outer architecture, standard FFN)
# ---------------------------------------------------------------------------


class StandardDenseBlock(nn.Module):
    """Pre-norm block identical to :class:`TransformerBlock` but with dense MLP.

    MLP (NanoGPT / Vaswani style expansion ``4×``):
        ``nn.Sequential(Linear(D, 4D), GELU(), Linear(4D, D))``
    """

    def __init__(self, config: FFFConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        d = config.n_embd
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * d, d, bias=config.bias),
        )
        self.mlp_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``x: (B, T, D) → (B, T, D)``."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp_dropout(self.mlp(self.ln_2(x)))
        return x


class StandardTransformer(nn.Module):
    """Decoder-only causal LM with standard dense MLP/FFN (benchmark baseline).

    Shares :class:`FFFConfig` fields ``vocab_size, n_embd, n_layer, n_head,
    block_size`` with :class:`FFFTransformer`. The sole architectural difference
    is the feedforward sublayer (dense ``D→4D→D`` vs FFF tree).

    Forward returns ``logits`` only (no balance loss): ``(B, T, V)``.
    """

    def __init__(self, config: FFFConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            StandardDenseBlock(config) for _ in range(config.n_layer)
        )
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input_ids:
            ``(B, T)``

        Returns
        -------
        Tensor
            Logits ``(B, T, vocab_size)``.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be (B, T), got shape {tuple(input_ids.shape)}"
            )
        _bsz, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}"
            )

        x = self.drop(self.wte(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters; optionally exclude the token embedding table."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressive sampling (dense baseline)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.block_size
                else input_ids[:, -self.config.block_size :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
