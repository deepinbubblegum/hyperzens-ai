"""Quantization-Aware Training (QAT) helpers for the BitNet FFF stack.

Two mechanisms:

* **Dynamic activation scaling factors.** :class:`ActivationQuantizer` computes
  a per-token (or per-channel / learned) scale from the live batch on every
  forward and quantizes activations with a straight-through estimator, exactly
  like BitNet's AbsMax scheme. Scales are exposed as ``last_scale`` /
  ``running_scale`` so downstream tooling can inspect calibration statistics.

* **FP16 master weights.** :class:`BitNetQAT` can pin every wrapped parameter
  to ``float16`` (the master copy). Forward passes run quantization on the
  fp32 view of that master, backward accumulates gradients into the fp16
  parameters, and :class:`FP16MasterAdamW` performs the Adam update in fp32
  before writing the result back to the fp16 master - so the master never
  stores quantized values and the update math is never done in low precision.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import ste_ternarize

__all__ = [
    "ActivationQuantizer",
    "FP16MasterAdamW",
    "BitNetQAT",
]

QUANT_MODES = ("absmax", "per_channel", "ema", "learned")


class ActivationQuantizer(nn.Module):
    """BitNet-style activation quantizer with dynamic scaling factors.

    Args:
        bits: activation width (``>= 32`` disables quantization).
        mode: scale selection
            - ``absmax``: per-token scale ``max(|x|)`` (dynamic, computed live).
            - ``per_channel``: one scale per feature column.
            - ``ema``: per-token scale smoothed with ``ema_decay`` across calls.
            - ``learned``: a learnable per-channel scale parameter.
        scale_init: initial value for the ``learned`` scale.
        ema_decay: decay for ``running_scale`` / the ``ema`` mode.
        eps: numerical floor for the scale.
        track_running: maintain ``running_scale`` (EMA of the observed scales).

    The quantization floor (resolution) is driven entirely by the dynamic scale
    each call; ``last_scale`` exposes the scale that was actually applied.
    """

    def __init__(
        self,
        bits: int = 8,
        mode: str = "absmax",
        scale_init: float = 1.0,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        track_running: bool = True,
    ) -> None:
        super().__init__()
        if mode not in QUANT_MODES:
            raise ValueError(f"mode must be one of {QUANT_MODES}, got {mode!r}")
        self.bits = bits
        self.mode = mode
        self.eps = eps
        self.ema_decay = ema_decay
        self.track_running = track_running
        if mode == "learned":
            self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        else:
            self.register_parameter("scale", None)
        self.register_buffer("running_scale", torch.tensor(0.0))
        self.register_buffer("last_scale", torch.tensor(1.0))

    def _sync_device(self, x: torch.Tensor) -> None:
        for name in ("running_scale", "last_scale"):
            buf = getattr(self, name)
            if buf.device != x.device:
                setattr(self, name, buf.to(x.device))

    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        self._sync_device(x)
        if self.mode == "absmax":
            return x.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        if self.mode == "per_channel":
            return x.abs().amax(dim=-2, keepdim=True).clamp_min(self.eps)
        if self.mode == "ema":
            scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
            with torch.no_grad():
                mean = scale.mean()
                if self.track_running:
                    self.running_scale.mul_(self.ema_decay).add_(mean, alpha=1 - self.ema_decay)
                self.last_scale.copy_(mean.detach())
            return torch.full_like(scale, self.running_scale.item()).clamp_min(self.eps)
        scale = torch.full_like(x.abs().amax(dim=-1, keepdim=True), self.scale.detach().abs())
        return scale.clamp_min(self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None or self.bits >= 32:
            return x
        max_q = 2 ** (self.bits - 1) - 1
        scale = self._compute_scale(x)
        q = (x / scale) * max_q
        q = q.round().clamp(-(max_q + 1), max_q)
        xq = (q / max_q) * scale
        if self.track_running and self.mode != "ema":
            with torch.no_grad():
                mean = scale.mean()
                self.running_scale.mul_(self.ema_decay).add_(mean, alpha=1 - self.ema_decay)
                self.last_scale.copy_(mean.detach())
        return xq + (x - x.detach())


class FP16MasterAdamW:
    """AdamW that keeps master weights in FP16 but does all update math in FP32.

    Intended to pair with :class:`BitNetQAT`: parameters are stored as FP16
    master weights (never quantized), gradients (FP16 or FP32, whichever
    autograd accumulated) are promoted to FP32 for the exponential moving
    averages and the per-step update, then the result is written back to the
    FP16 master. This keeps gradient updates stable before any quantization
    (ternary weights / 8-bit activations) is applied at forward time. Decoupled
    weight decay is added **in place** to the promoted FP32 gradient
    (``g.add_(p.detach(), alpha=weight_decay)``, exact for FP16 masters since
    FP16 -> FP32 is lossless), so no per-parameter FP32 temporary is
    allocated during :meth:`step`.

    Args:
        params: iterable of trainable parameters (forced to ``master_dtype``).
        lr: learning rate.
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        weight_decay: decoupled weight decay.
        master_dtype: storage dtype of the master weights (default FP16).
        clip_grad_norm: optional global norm clipping applied to FP32 grads.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        master_dtype: torch.dtype = torch.float16,
        clip_grad_norm: float | None = None,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("FP16MasterAdamW requires at least one parameter")
        self.master_dtype = master_dtype
        for p in self.params:
            if p.dtype != master_dtype:
                p.data = p.data.to(master_dtype)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.param_groups = [{"params": self.params, "lr": lr, "weight_decay": weight_decay}]
        self.state: dict = {}
        self.steps = 0

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def _grads(self) -> list[torch.Tensor]:
        return [p.grad.detach().to(torch.float32) for p in self.params if p.grad is not None]

    def step(self) -> None:
        grad_pairs = [
            (p, p.grad.detach().to(torch.float32))
            for p in self.params
            if p.grad is not None
        ]
        if not grad_pairs:
            return
        self.steps += 1
        if self.clip_grad_norm is not None:
            grads = [g for _, g in grad_pairs]
            total_norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
            clip = self.clip_grad_norm
            if total_norm > clip:
                factor = clip / total_norm
                grad_pairs = [(p, g * factor) for p, g in grad_pairs]
        for p, g in grad_pairs:
            st = self.state.setdefault(p, {"exp_avg": None, "exp_avg_sq": None})
            if st["exp_avg"] is None:
                st["exp_avg"] = torch.zeros_like(g)
                st["exp_avg_sq"] = torch.zeros_like(g)
            if self.weight_decay != 0:
                g.add_(p.detach(), alpha=self.weight_decay)
            st["exp_avg"].mul_(self.betas[0]).add_(g, alpha=1 - self.betas[0])
            st["exp_avg_sq"].mul_(self.betas[1]).addcmul_(g, g, value=1 - self.betas[1])
            bc1 = 1 - self.betas[0] ** self.steps
            bc2 = 1 - self.betas[1] ** self.steps
            step = (st["exp_avg"] / bc1) / (torch.sqrt(st["exp_avg_sq"] / bc2) + self.eps)
            p.data = (p.data.float() - self.lr * step).to(self.master_dtype)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "steps": self.steps,
            "master_dtype": str(self.master_dtype),
            "state": {
                str(id(p)): {
                    "exp_avg": st["exp_avg"].clone(),
                    "exp_avg_sq": st["exp_avg_sq"].clone(),
                }
                if st["exp_avg"] is not None
                else None
                for p, st in self.state.items()
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.lr = sd["lr"]
        self.steps = sd["steps"]
        for p, st in self.state.items():
            entry = sd["state"].get(str(id(p)))
            if entry is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                st[key] = entry[key].to(p.device)


class BitNetQAT(nn.Module):
    """QAT wrapper around any module (Linear or FastFeedForwardBitNet).

    Applies :class:`ActivationQuantizer` (dynamic scaling factors) to the input
    and optionally ternarizes a wrapped ``nn.Linear``'s weights with a tunable
    threshold via STE. ``enable_fp16_master()`` pins all parameters to FP16 so
    training runs on FP16 master weights with :class:`FP16MasterAdamW`.

    For :class:`FastFeedForwardBitNet` targets the wrapper only handles the
    activation scaling + FP16 master weights; leaf ternary quantization is
    performed inside the FFF itself (STE).
    """

    def __init__(
        self,
        module: nn.Module,
        activation_bits: int = 8,
        quant_mode: str = "absmax",
        threshold_scale: float = 1.0,
        master_dtype: torch.dtype = torch.float16,
        eps: float = 1e-8,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.module = module
        self.threshold_scale = threshold_scale
        self.master_dtype = master_dtype
        self.eps = eps
        self.act_quant = ActivationQuantizer(
            bits=activation_bits, mode=quant_mode, eps=eps, ema_decay=ema_decay
        )
        self._quantize_linear = isinstance(module, nn.Linear)
        self._fp16_enabled = False

    def enable_fp16_master(self) -> None:
        for p in self.module.parameters():
            p.data = p.data.to(self.master_dtype)
        self._fp16_enabled = True

    @property
    def is_fp16_master(self) -> bool:
        return self._fp16_enabled

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if hasattr(self.module, "gradient_checkpointing_enable"):
            self.module.gradient_checkpointing_enable(**kwargs)
        else:
            self.module.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.module, "gradient_checkpointing_disable"):
            self.module.gradient_checkpointing_disable()
        else:
            self.module.gradient_checkpointing = False

    def optimizer(self, lr: float = 1e-3, **kwargs) -> FP16MasterAdamW:
        return FP16MasterAdamW(
            self.module.parameters(), lr=lr, master_dtype=self.master_dtype, **kwargs
        )

    def quantized_weight(self) -> torch.Tensor | None:
        if not self._quantize_linear:
            return None
        return ste_ternarize(
            self.module.weight.float(), eps=self.eps, threshold_scale=self.threshold_scale
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        is_float = x.is_floating_point()
        if is_float:
            xq = self.act_quant(x)
        else:
            xq = x
        if self._quantize_linear:
            w = self.quantized_weight()
            return F.linear(xq, w, self.module.bias)
        if is_float:
            p0 = next(self.module.parameters(), None)
            if p0 is not None and xq.dtype != p0.dtype:
                xq = xq.to(p0.dtype)
        return self.module(xq, *args, **kwargs)
