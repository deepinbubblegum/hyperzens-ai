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
from typing import Callable, Iterator

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
        bos_token_id / eos_token_id / pad_token_id: special token ids bound
            from the tokenizer via :meth:`bind_tokenizer` (used by
            :meth:`BitNetFFTTransformer.generate` as the default eos).
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
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    @property
    def fff_out(self) -> int:
        return self.fff_d_out if self.fff_d_out is not None else self.d_model

    @property
    def fff_capacity(self) -> int:
        return (1 << self.fff_depth) * self.fff_out

    def bind_tokenizer(self, tokenizer) -> "BitNetFFTConfig":
        """Bind ``vocab_size`` (+ special token ids) from a tokenizer.

        Works with any object exposing ``vocab_size`` / ``bos_token_id`` /
        ``eos_token_id`` / ``pad_token_id`` (e.g.
        :class:`bitnet_fff.tokenizer.BPETokenizer` or the byte fallback), so a
        model can be built directly against the tokenizer that will feed it.
        """
        vocab = int(getattr(tokenizer, "vocab_size", self.vocab_size))
        if vocab <= 0:
            raise ValueError(f"tokenizer.vocab_size must be > 0, got {vocab}")
        self.vocab_size = vocab
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        return self

    @classmethod
    def from_tokenizer(
        cls, tokenizer, **overrides
    ) -> "BitNetFFTConfig":
        """Build a config with ``vocab_size`` bound from ``tokenizer``."""
        cfg = cls(**overrides)
        return cfg.bind_tokenizer(tokenizer)


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


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
) -> torch.Tensor:
    """Sample one token per row from raw ``(..., vocab)`` logits, fully on-device.

    ``temperature <= 0`` selects greedily via ``argmax``. Otherwise the logits
    are temperature-scaled, softmaxed, optionally restricted to the ``top_k``
    largest probabilities and to the nucleus of cumulative probability mass
    ``top_p``, renormalized and drawn with ``torch.multinomial``. Every
    operation stays on the input tensor's device, so sampling never round-trips
    through the host.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), probs.shape[-1])
        kth = torch.topk(probs, k, dim=-1).values[..., -1:]
        probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs = sorted_probs.masked_fill(
            cumulative - sorted_probs > top_p, 0.0
        )
        probs = torch.zeros_like(probs).scatter_(-1, order, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probs.dtype).tiny
    )
    return torch.multinomial(probs, num_samples=1)


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

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate a continuation using a KV cache.

        **Prefill** — the full prompt is run through the model once to populate
        fresh per-layer :class:`KVCache` buffers (capacity ``max_seq_len``); the
        last prompt row's logits already predict the first new token.

        **Decode** — one token per step is embedded/projected using only the
        cached key/value states (nothing before the current token is
        recomputed), and each FFN block runs through the fused single-pass
        kernel (:meth:`FastFeedForwardBitNet.fast_forward`) when
        ``use_fast_inference`` is set; ``eval()`` mode is entered and restored
        around the call so the fast path is always eligible.

        **Sampling** — ``temperature <= 0`` gives greedy decoding (``argmax``);
        otherwise temperature scaling, top-``k`` filtering and top-``p``
        (nucleus) filtering are applied on-device followed by
        ``torch.multinomial``.

        **Stream discipline** — the whole loop runs under
        ``torch.inference_mode()`` and every tensor stays on the model's device;
        the only host-device synchronizations are a single scalar ``done`` check
        per step (required for early eos termination) and a final eos trim.

        Args:
            prompt_ids: token ids, shape ``(seq,)`` or ``(batch, seq)``.
            max_new_tokens: number of tokens to generate (>= 1).
            temperature: sampling temperature; ``<= 0`` disables sampling.
            top_k: keep only the ``top_k`` most probable tokens (``None``/0 off).
            top_p: nucleus mass (``None``/0/1 off).
            eos_token_id: stop generating for sequences that emit this token.

        Returns:
            Full sequence ``(batch, prompt_len + gen_len)`` (or ``(prompt_len +
            gen_len,)`` for a 1D prompt) with generation truncated at the first
            ``eos_token_id`` when one is supplied.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if self.embed is None or self.head is None:
            raise ValueError("generate() requires a token model (vocab_size > 0)")
        if eos_token_id is None:
            eos_token_id = self.cfg.eos_token_id

        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                squeeze = prompt_ids.ndim == 1
                tokens = prompt_ids if not squeeze else prompt_ids.unsqueeze(0)
                if tokens.ndim != 2:
                    raise ValueError(
                        f"prompt_ids must be 1D or 2D, got {tuple(prompt_ids.shape)}"
                    )
                batch, prompt_len = tokens.shape
                if prompt_len < 1:
                    raise ValueError("prompt_ids must contain at least one token")
                if prompt_len + max_new_tokens > self.cfg.max_seq_len:
                    raise ValueError(
                        f"prompt_len + max_new_tokens ({prompt_len} + "
                        f"{max_new_tokens}) exceeds max_seq_len "
                        f"{self.cfg.max_seq_len}"
                    )

                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                head_dim = self.cfg.d_model // self.cfg.n_heads
                caches = [
                    KVCache.preallocate(
                        batch,
                        self.cfg.n_heads,
                        self.cfg.max_seq_len,
                        head_dim,
                        dtype=dtype,
                        device=device,
                    )
                    for _ in self.layers
                ]

                # Prefill: full prompt once, cache all key/value states; the
                # last row's logits already predict the first new token.
                prefill = self(tokens, kv_cache=caches, past_length=0)
                next_ids = _sample_from_logits(
                    prefill[:, -1], temperature, top_k, top_p
                )

                pad = eos_token_id if eos_token_id is not None else 0
                generated = torch.full(
                    (batch, max_new_tokens), pad, dtype=torch.long, device=device
                )
                done = torch.zeros(batch, dtype=torch.bool, device=device)
                if eos_token_id is not None:
                    done = (next_ids == eos_token_id).squeeze(-1)
                    next_ids = next_ids.masked_fill(done.unsqueeze(-1), eos_token_id)
                generated[:, 0] = next_ids.squeeze(-1)

                # Decode: one cached token per step; the newly sampled token
                # occupies position prompt_len + step.
                for step in range(1, max_new_tokens):
                    if eos_token_id is not None and bool(done.all()):
                        break
                    logits = self(
                        next_ids, kv_cache=caches, past_length=prompt_len + step - 1
                    )
                    next_ids = _sample_from_logits(
                        logits[:, -1], temperature, top_k, top_p
                    )
                    if eos_token_id is not None:
                        done = done | (next_ids == eos_token_id).squeeze(-1)
                        next_ids = next_ids.masked_fill(
                            done.unsqueeze(-1), eos_token_id
                        )
                    generated[:, step] = next_ids.squeeze(-1)

                if eos_token_id is not None:
                    is_eos = generated == eos_token_id
                    has_eos = is_eos.any(dim=-1)
                    eos_at = is_eos.to(torch.long).argmax(dim=-1)
                    lengths = torch.where(
                        has_eos,
                        eos_at + 1,
                        torch.full_like(eos_at, generated.shape[1]),
                    )
                    generated = generated[:, : int(lengths.max())]

                out = torch.cat([tokens, generated], dim=1)
                return out.squeeze(0) if squeeze else out
        finally:
            self.train(was_training)

    def stream_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        decode_token: Callable[[int], str] | None = None,
    ) -> Iterator[str | int | tuple[str, ...] | torch.Tensor]:
        """Autoregressively stream a continuation, one token (or token text) at a time.

        A generator twin of :meth:`generate` for zero-latency typewriter output:
        the prompt is prefilled once into fresh per-layer :class:`KVCache`
        buffers and each decode step runs a single cached token through the
        model plus the fused single-pass FFF kernel
        (:meth:`FastFeedForwardBitNet.fast_forward`), exactly like ``generate``,
        but control returns to the caller after **every** sampled token instead
        of only at the end.

        Yielded item per step:
            * ``decode_token`` is ``None`` — the raw token id (``int`` for a
              single-row prompt, else a ``torch.Tensor`` of shape ``(batch,)``).
            * ``decode_token`` is a callable ``(int) -> str`` — the decoded text
              piece (``str`` for a single-row prompt, else a tuple of strings).
              Pass a streaming decoder (e.g. ``BPETokenizer.decode_step``) so
              multi-byte UTF-8 split across BPE tokens reassembles correctly.

        The whole loop runs under ``torch.inference_mode()``; ``eval()`` mode is
        entered when iteration starts and restored when the generator is
        exhausted or closed early (so it is safe to ``break`` out of the stream).
        Early ``eos_token_id`` termination is honoured (a per-step host-device
        sync, as in :meth:`generate`). Sampling parameters match :meth:`generate`.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if self.embed is None or self.head is None:
            raise ValueError("stream_generate() requires a token model (vocab_size > 0)")
        if eos_token_id is None:
            eos_token_id = self.cfg.eos_token_id

        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                squeeze = prompt_ids.ndim == 1
                tokens = prompt_ids if not squeeze else prompt_ids.unsqueeze(0)
                if tokens.ndim != 2:
                    raise ValueError(
                        f"prompt_ids must be 1D or 2D, got {tuple(prompt_ids.shape)}"
                    )
                batch, prompt_len = tokens.shape
                if prompt_len < 1:
                    raise ValueError("prompt_ids must contain at least one token")
                if prompt_len + max_new_tokens > self.cfg.max_seq_len:
                    raise ValueError(
                        f"prompt_len + max_new_tokens ({prompt_len} + "
                        f"{max_new_tokens}) exceeds max_seq_len "
                        f"{self.cfg.max_seq_len}"
                    )

                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                head_dim = self.cfg.d_model // self.cfg.n_heads
                caches = [
                    KVCache.preallocate(
                        batch,
                        self.cfg.n_heads,
                        self.cfg.max_seq_len,
                        head_dim,
                        dtype=dtype,
                        device=device,
                    )
                    for _ in self.layers
                ]

                def _emit(ids: torch.Tensor):
                    if decode_token is not None:
                        if batch == 1:
                            return decode_token(int(ids[0, 0]))
                        return tuple(decode_token(int(i)) for i in ids[:, 0])
                    if batch == 1:
                        return int(ids[0, 0])
                    return ids[:, 0]

                # Prefill: full prompt once; the last row's logits already
                # predict the first new token.
                prefill = self(tokens, kv_cache=caches, past_length=0)
                next_ids = _sample_from_logits(
                    prefill[:, -1], temperature, top_k, top_p
                )
                done = torch.zeros(batch, dtype=torch.bool, device=device)
                if eos_token_id is not None:
                    done = (next_ids == eos_token_id).squeeze(-1)
                    next_ids = next_ids.masked_fill(done.unsqueeze(-1), eos_token_id)
                yield _emit(next_ids)

                for step in range(1, max_new_tokens):
                    if eos_token_id is not None and bool(done.all()):
                        break
                    logits = self(
                        next_ids, kv_cache=caches, past_length=prompt_len + step - 1
                    )
                    next_ids = _sample_from_logits(
                        logits[:, -1], temperature, top_k, top_p
                    )
                    if eos_token_id is not None:
                        done = done | (next_ids == eos_token_id).squeeze(-1)
                        next_ids = next_ids.masked_fill(
                            done.unsqueeze(-1), eos_token_id
                        )
                    yield _emit(next_ids)
        finally:
            self.train(was_training)

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
