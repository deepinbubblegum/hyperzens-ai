"""End-to-end Transformer/MLP architectures built on FastFeedForwardBitNet.

``FastFeedForwardBitNet`` replaces the default feed-forward network: a balanced
binary decision tree routes each token to exactly one of ``2**depth`` leaves,
so the effective FFN capacity is ``num_leaves * fff_d_out`` while only
``depth`` decision nodes and a single leaf projection are executed per token.

Both architectures support a packed-ternary fast-inference path: in ``eval()``
mode (and with the native extension available) each FFF routes and then runs
the Metal/NEON kernel on the routed leaves only, skipping all other branches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import absmax_quantize
from .fast_fff import FastFeedForwardBitNet

__all__ = [
    "BitNetFFTConfig",
    "KVCache",
    "BitNetAttention",
    "BitNetFFTBlock",
    "BitNetFFTTransformer",
    "BitNetFFTMLP",
]


@dataclass
class BitNetFFTConfig:
    """Hyper-parameters for the BitNet FFF transformer / MLP.

    Attributes:
        vocab_size: embedding table size (``0`` disables the embedding + head
            and keeps the raw d_model -> d_model core, handy for probing).
        d_model: residual-stream / per-leaf projection width.
        n_heads: attention heads (transformer only).
        n_layers: number of stacked blocks.
        fff_d_out: per-leaf output width; defaults to ``d_model``.
        fff_depth: decision-tree depth (``log2(num_leaves)``). The tuner
            sweeps this against ``fff_threshold_scale``.
        fff_threshold_scale: multiplier on the AbsMean ternary threshold
            (see :func:`absmean_ternarize`); 1.0 is the classic BitNet b1.58.
        activation_bits: BitNet activation quantization width applied to the
            inputs of attention and each FFF.
        router_rank: ``"full"`` or ``"r1"`` decision nodes.
        fff_bias: use leaf biases.
        max_seq_len: learned positional embedding width (transformer only).
        dropout: dropout probability on residual branches.
        tie_weights: tie the output head to the input embedding.
        use_fast_inference: in ``eval()`` use the packed-ternary Metal/NEON
            path when available (falls back to the reference path otherwise).
        eps: numerical stability for quantizers.
        attention_activation_bits: ``None`` to reuse ``activation_bits``.
    """

    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    fff_d_out: int | None = None
    fff_depth: int = 3
    fff_threshold_scale: float = 1.0
    activation_bits: int = 8
    router_rank: str = "full"
    fff_bias: bool = True
    max_seq_len: int = 128
    dropout: float = 0.0
    tie_weights: bool = False
    use_fast_inference: bool = True
    eps: float = 1e-8
    attention_activation_bits: int | None = None
    chunk_size: int | None = None

    @property
    def fff_out(self) -> int:
        return self.fff_d_out if self.fff_d_out is not None else self.d_model

    @property
    def fff_capacity(self) -> int:
        return (1 << self.fff_depth) * self.fff_out


@dataclass
class KVCache:
    """Append-only key/value cache for autoregressive token generation.

    ``key_cache`` / ``value_cache`` are ``(batch_size, n_heads, seq_len,
    head_dim)`` buffers. The sequence dimension is dynamic: :meth:`append`
    writes new keys/values at ``[past_length:past_length + seq]`` and grows
    the buffers (doubling, preserving the filled prefix) whenever the capacity
    is exceeded, so past tokens are never recomputed. Use :meth:`preallocate`
    to size the buffers up front and avoid re-allocations during long
    generation runs. Works with float16/float32 on CPU or MPS.

    Attributes:
        key_cache: ``(B, n_heads, seq_len, head_dim)`` key states.
        value_cache: ``(B, n_heads, seq_len, head_dim)`` value states.
        size: number of filled positions along the sequence dimension.
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    size: int = 0

    @classmethod
    def preallocate(
        cls,
        batch_size: int,
        n_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "KVCache":
        """Build a cache with fixed-capacity ``(B, n_heads, seq_len, head_dim)`` buffers.

        ``dtype`` defaults to ``torch.get_default_dtype()`` (float32); pass
        ``torch.float16`` to halve the memory on MPS.
        """
        shape = (batch_size, n_heads, seq_len, head_dim)
        return cls(
            key_cache=torch.empty(shape, dtype=dtype, device=device),
            value_cache=torch.empty(shape, dtype=dtype, device=device),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.key_cache.dtype

    @property
    def device(self) -> torch.device:
        return self.key_cache.device

    def _grow(self, min_len: int) -> None:
        cap = max(min_len, 2 * self.key_cache.shape[-2])
        shape = self.key_cache.shape[:-2] + (cap, self.key_cache.shape[-1])
        new_k = torch.empty(shape, dtype=self.dtype, device=self.device)
        new_v = torch.empty(shape, dtype=self.dtype, device=self.device)
        new_k[..., : self.size, :] = self.key_cache[..., : self.size, :]
        new_v[..., : self.size, :] = self.value_cache[..., : self.size, :]
        self.key_cache, self.value_cache = new_k, new_v

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        past_length: int = 0,
    ) -> None:
        """Write ``key``/``value`` at ``[past_length:past_length + seq]``.

        Keys/values are written in place when capacity allows (pre-allocated
        mode); otherwise the buffers are grown first. ``size`` advances to the
        last written position so a subsequent call passes ``past_length =
        cache.size``.
        """
        if key.shape[-1] != self.key_cache.shape[-1]:
            raise ValueError(
                f"head_dim {key.shape[-1]} does not match cache "
                f"{self.key_cache.shape[-1]}"
            )
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError(
                f"cache dtype {self.dtype} vs incoming "
                f"key/value {key.dtype}/{value.dtype}"
            )
        start, end = past_length, past_length + key.shape[-2]
        if self.key_cache.shape[-2] < end:
            self._grow(end)
        self.key_cache[:, :, start:end] = key
        self.value_cache[:, :, start:end] = value
        self.size = max(self.size, end)


class BitNetAttention(nn.Module):
    """Multi-head attention with optional BitNet activation quantization.

    Queries/keys/values are passed through :func:`absmax_quantize` (per-token
    dynamic scale, STE) when ``activation_bits`` is set, mirroring BitNet's
    low-precision attention. The compatibility mask is a right-aligned
    lower-triangular ``(seq, seq)`` additive bias, so both full attention and
    causal (generative) attention are supported.

    Autoregressive generation: pass a :class:`KVCache` (plus ``past_length``)
    to append the new keys/values along the sequence dimension and attend only
    to them, without recomputing the projections of previous tokens. The
    per-token AbsMax activation quantization is preserved exactly, because
    each token is scaled by its own row only.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        activation_bits: int | None = 8,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.activation_bits = activation_bits
        self.eps = eps
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
        kv_cache: KVCache | None = None,
        past_length: int = 0,
    ) -> torch.Tensor:
        """Run attention on ``x`` (shape ``(*batch, seq, d_model)``).

        When ``kv_cache`` is given, ``x`` holds only the new tokens; their
        keys/values are appended at ``[past_length:past_length + seq]`` and
        attention attends over the full cached sequence ``(B, n_heads,
        past_length + seq, head_dim)``. A ``None`` mask becomes a causal
        ``(seq, past_length + seq)`` mask automatically; an explicit
        ``(seq, seq)`` mask is left-padded with zeros so cached keys stay
        visible.
        """
        *batch, seq, _ = x.shape
        if self.activation_bits is not None and self.activation_bits < 32:
            x = absmax_quantize(x, bits=self.activation_bits, eps=self.eps)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B = math.prod(batch) if batch else 1
        q = q.reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.append(k, v, past_length=past_length)
            k = kv_cache.key_cache[..., : kv_cache.size, :]
            v = kv_cache.value_cache[..., : kv_cache.size, :]

        full_len = kv_cache.size if kv_cache is not None else seq

        if kv_cache is not None:
            if mask is None:
                if full_len > 1:
                    mask = _causal_mask(full_len, x.device)[-seq:]
            elif mask.shape[-1] < full_len:
                pad = full_len - mask.shape[-1]
                mask = torch.cat(
                    [
                        torch.zeros(
                            *mask.shape[:-1], pad, device=mask.device, dtype=mask.dtype
                        ),
                        mask,
                    ],
                    dim=-1,
                )

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        attn = torch.softmax(scores, dim=-1)
        if self.dropout and self.training:
            attn = F.dropout(attn, p=self.dropout)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).reshape(*batch, seq, self.d_model)
        out = self.out_proj(ctx)
        if need_weights:
            return out, attn.mean(dim=1)
        return out


def _causal_mask(seq: int, device: torch.device) -> torch.Tensor:
    m = torch.triu(torch.full((seq, seq), float("-inf"), device=device), diagonal=1)
    return m


def _per_layer_caches(
    kv_cache: KVCache | list[KVCache] | None, n_layers: int
) -> list[KVCache | None]:
    """Expand a per-model cache into one cache per transformer layer.

    A bare :class:`KVCache` is only allowed for single-layer models (each
    layer needs its own key/value storage); otherwise a list of ``n_layers``
    caches is required so layer N never overwrites layer M's states.
    """
    if kv_cache is None:
        return [None] * n_layers
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) != n_layers:
            raise ValueError(
                f"expected {n_layers} KVCache entries, got {len(kv_cache)}"
            )
        return list(kv_cache)
    if n_layers != 1:
        raise ValueError(
            "provide one KVCache per layer (a list) for n_layers > 1; "
            f"got a single cache for {n_layers} layers"
        )
    return [kv_cache]


class BitNetFFTBlock(nn.Module):
    """Pre-norm transformer block: attention + FFF with residual connections."""

    def __init__(
        self,
        cfg: BitNetFFTConfig,
        layer_norm_fn: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = layer_norm_fn(cfg.d_model)
        self.attn = BitNetAttention(
            cfg.d_model,
            cfg.n_heads,
            activation_bits=(
                cfg.attention_activation_bits
                if cfg.attention_activation_bits is not None
                else cfg.activation_bits
            ),
            eps=cfg.eps,
            dropout=cfg.dropout,
        )
        self.norm2 = layer_norm_fn(cfg.d_model)
        self.fff = FastFeedForwardBitNet(
            d_in=cfg.d_model,
            d_out=cfg.fff_out,
            depth=cfg.fff_depth,
            bias=cfg.fff_bias,
            activation_bits=cfg.activation_bits,
            router_rank=cfg.router_rank,
            chunk_size=cfg.chunk_size,
            eps=cfg.eps,
            ternarize_threshold_scale=cfg.fff_threshold_scale,
        )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout else nn.Identity()

    def _fff_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_fast_inference and not self.training:
            return self.fff.fast_forward(x, chunk_size=self.cfg.chunk_size)
        return self.fff(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        past_length: int = 0,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, mask=mask, kv_cache=kv_cache, past_length=past_length)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self._fff_forward(h)
        if h.dtype != x.dtype:
            h = h.to(x.dtype)
        return x + self.dropout(h)


class BitNetFFTTransformer(nn.Module):
    """Causal decoder-only transformer whose FFN is a FastFeedForwardBitNet.

    Input tokens are embedded, summed with a learned positional embedding, run
    through ``n_layers`` :class:`BitNetFFTBlock` (causal attention + FFF), then
    layer-normalized and projected to logits. With ``vocab_size == 0`` the
    embedding/head are omitted and the model is a pure ``d_model`` sequence
    block (used by the profiler/tuner to isolate FFF cost).
    """

    def __init__(
        self,
        cfg: BitNetFFTConfig,
        layer_norm_fn: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = None
        self.pos = None
        if cfg.vocab_size > 0:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([BitNetFFTBlock(cfg, layer_norm_fn) for _ in range(cfg.n_layers)])
        self.norm = layer_norm_fn(cfg.d_model)
        self.head = None
        if cfg.vocab_size > 0:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_weights and self.embed is not None:
                self.head.weight = self.embed.weight

    def _apply_embeddings(
        self, token_ids: torch.Tensor, pos_start: int = 0
    ) -> torch.Tensor:
        if self.embed is None:
            raise ValueError("config.vocab_size must be > 0 to embed tokens")
        pos = torch.arange(
            pos_start,
            pos_start + token_ids.shape[-1],
            dtype=torch.long,
            device=token_ids.device,
        )
        return self.embed(token_ids) + self.pos(pos)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: KVCache | list[KVCache] | None = None,
        past_length: int = 0,
        **_: object,
    ) -> torch.Tensor:
        """Run the transformer on ``x`` (token ids or ``d_model`` rows).

        ``kv_cache`` may be a single :class:`KVCache` (broadcast to every
        layer) or a list/tuple with one cache per layer; ``past_length`` is the
        number of already-generated tokens so each step only embeds/projects
        the new token.
        """
        if self.embed is not None:
            x = self._apply_embeddings(x, pos_start=past_length)
        if mask is None and x.shape[-2] > 1:
            mask = _causal_mask(x.shape[-2], x.device)
        layer_caches = _per_layer_caches(kv_cache, len(self.layers))
        for layer, layer_cache in zip(self.layers, layer_caches):
            x = layer(x, mask=mask, kv_cache=layer_cache, past_length=past_length)
        x = self.norm(x)
        if self.head is not None:
            x = self.head(x)
        return x

    @property
    def fff_params(self) -> int:
        return sum(
            p.numel()
            for layer in self.layers
            for name, p in layer.fff.named_parameters()
        )


class BitNetFFTMLP(nn.Module):
    """Stacked FFF MLP: ``[FFF -> GELU -> LayerNorm] * n_layers -> head``.

    Used to profile pure FFF-conditioned computation without attention.
    """

    def __init__(
        self,
        cfg: BitNetFFTConfig,
        layer_norm_fn: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            block = nn.Module()
            block.norm = layer_norm_fn(cfg.d_model)
            block.fff = FastFeedForwardBitNet(
                d_in=cfg.d_model,
                d_out=cfg.fff_out,
                depth=cfg.fff_depth,
                bias=cfg.fff_bias,
                activation_bits=cfg.activation_bits,
                router_rank=cfg.router_rank,
                chunk_size=cfg.chunk_size,
                eps=cfg.eps,
                ternarize_threshold_scale=cfg.fff_threshold_scale,
            )
            self.layers.append(block)
        self.norm = layer_norm_fn(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        for block in self.layers:
            h = block.norm(x)
            h = (
                block.fff.fast_forward(h, chunk_size=self.cfg.chunk_size)
                if self.cfg.use_fast_inference and not self.training
                else block.fff(h)
            )
            h = F.gelu(h)
            x = x + h
        return self.head(self.norm(x))
