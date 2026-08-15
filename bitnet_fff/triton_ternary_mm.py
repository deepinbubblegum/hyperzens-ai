"""2-bit packed BitNet ternary matrix multiplication with a fused Triton kernel.

BitNet b1.58-style GEMM where the weight is decomposed as ``W ~= S * beta``
with ``S in {-1, 0, +1}`` (ternary) and ``beta = mean(|W|)`` per output row.
The ternary weights are packed 2 bits each (4 per byte, 16 per int32 word) and
unpacked on the fly inside the Triton kernel with bit masks/shifts
(``(packed >> shift) & 0x03``), so the weight tensor in VRAM is 1/4 the FP32
size and the matmul runs on FP16 Ampere tensor cores via ``tl.dot``.

Forward math (per-token input scaling ``gamma = mean(|X|)``):
    Xq = clamp(round(X / gamma), -127, 127)   (round half away from zero)
    Z  = Xq @ S^T                              (FP16 tl.dot, FP32 accum)
    Y  = Z * (gamma * beta)                    (both scales fused at block output)

With quantization disabled (``quantize=False``) ``Xq = X`` and with
``apply_scale=False`` no scaling is applied, giving an exact (up to FP16
rounding) ternary matmul that matches ``torch.matmul`` against ternary weights.

Layout: ``pack_ternary`` stores the codes as ``packed[k, n_word]`` (16
consecutive output indices per int32 word), so the unpacked weight tile is
produced directly in ``(BLOCK_K, BLOCK_N)`` register layout and feeds
``tl.dot`` without any transpose. Each bit code is ``1`` -> +1, ``2`` -> -1,
``0`` -> 0 (3 unused).

Backward uses the standard BitNet straight-through estimator: quantization and
ternarization are treated as identity and the fused scales as constants, so
the packed kernel itself needs no backward path - gradients are computed in
torch from the unpacked ternary weights:
    gZ = gY * (gamma * beta)
    gX = gZ @ S
    gW = gZ^T @ Xq

Kernel is tuned for consumer Ampere (RTX 3060, CC 8.6): BLOCK_M=64,
BLOCK_N=64, BLOCK_K=32, num_warps=4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # triton is optional (CUDA only); the module still imports without it.
    import triton
    import triton.language as tl

    has_triton = True
except Exception:  # pragma: no cover - exercised only on non-CUDA machines
    triton = None
    tl = None
    has_triton = False


def _jit(fn):
    """Apply triton.jit when triton is installed, else leave the function as-is."""
    return triton.jit(fn) if has_triton else fn

__all__ = [
    "has_triton",
    "pack_ternary",
    "unpack_ternary",
    "packed_ternary_mm_ref",
    "packed_ternary_matmul",
    "PackedTernaryBitLinear",
    "PackedBitLinear",
]

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_NUM_WARPS = 4
_NUM_STAGES = 3
_EPS = 1e-8


# --------------------------------------------------------------------------
# packing / unpacking (torch, device agnostic)
# --------------------------------------------------------------------------


def pack_ternary(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pack ``sign(W)`` into 2-bit codes (16 per int32 word) along the output dim.

    Returns ``(K, ceil(N / 16))`` int32, where entry ``[k, nw]`` holds the
    2-bit codes for output channels ``[16*nw, 16*nw + 16)`` at input ``k``
    (code 0 -> 0, 1 -> +1, 2 -> -1). ``N`` is zero-padded to a multiple of 16.
    """
    del eps  # reserved for a future absmean threshold option
    w = w.detach()
    s = torch.sign(w)
    n, k = s.shape
    codes = torch.where(
        s == 1,
        torch.ones_like(s, dtype=torch.int32),
        torch.where(
            s == -1,
            torch.full_like(s, 2, dtype=torch.int32),
            torch.zeros_like(s, dtype=torch.int32),
        ),
    )
    n_pad = ((n + 15) // 16) * 16
    if n_pad > n:
        codes = F.pad(codes, (0, 0, 0, n_pad - n))
    ct = codes.T.reshape(k, n_pad // 16, 16)  # lane-minor so reshape == n index
    packed = torch.zeros(k, n_pad // 16, dtype=torch.int32, device=w.device)
    for lane in range(16):
        packed |= ct[..., lane] << (2 * lane)
    return packed


def unpack_ternary(
    packed: torch.Tensor, n: int | None = None, k: int | None = None
) -> torch.Tensor:
    """Unpack 2-bit codes back into FP32 ternary ``{-1, 0, +1}`` (``N, K``)."""
    k_dim, n_words = packed.shape
    n_pad = n_words * 16
    codes = torch.zeros(k_dim, n_words, 16, dtype=torch.int32, device=packed.device)
    for lane in range(16):
        codes[..., lane] = (packed >> (2 * lane)) & 0x03
    vals = torch.where(
        codes == 1,
        torch.ones_like(codes),
        torch.where(codes == 2, -torch.ones_like(codes), torch.zeros_like(codes)),
    )
    s = vals.reshape(k_dim, n_pad).T
    if n is not None:
        s = s[:n]
    if k is not None:
        s = s[:, :k]
    return s


# --------------------------------------------------------------------------
# reference forward (device agnostic, mirrors the kernel exactly)
# --------------------------------------------------------------------------


def _round_half_away(t: torch.Tensor) -> torch.Tensor:
    """Round half away from zero (matches the Triton floor/ceil code)."""
    return torch.where(t >= 0, torch.floor(t + 0.5), torch.ceil(t - 0.5))


def packed_ternary_mm_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    apply_scale: bool = True,
    quantize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PyTorch reference for the packed ternary GEMM (FP32 math)."""
    x = x.detach().float()
    s = unpack_ternary(pack_ternary(w), n=w.shape[0], k=w.shape[1]).float()
    if quantize:
        gamma = x.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        xq = _round_half_away(x / gamma).clamp(-127.0, 127.0)
        z = xq @ s.T
        if apply_scale:
            beta = w.detach().abs().mean(dim=-1)
            z = z * (gamma * beta)
    else:
        z = x @ s.T
        if apply_scale:
            beta = w.detach().abs().mean(dim=-1)
            z = z * beta
    return z


# --------------------------------------------------------------------------
# Triton kernel
# --------------------------------------------------------------------------


@_jit
def _packed_ternary_mm_kernel(
    A, PW, BETA, Y,
    M, N, K,
    stride_am, stride_ak,
    stride_pk,
    stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    QUANT: tl.constexpr,
    EPS: tl.constexpr,
):
    """Fused packed-ternary GEMM for one ``(BLOCK_M, BLOCK_N)`` output tile.

    ``PW`` is the packed weight ``(K, N // 16)`` int32; ``BETA`` is
    ``mean(|W|)`` per output row ``(N,)``. Per-row input scale ``gamma`` is
    computed inside the kernel, so the full ``Y = Z * (gamma * beta)`` scaling
    happens in one fused multiply at the block output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_nw = tl.arange(0, BLOCK_N // 16)
    offs_k = tl.arange(0, BLOCK_K)
    lane = tl.arange(0, 16)

    if QUANT:
        # pass 1: per-row mean(abs(X)) over the full K dimension
        acc_abs = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for kk in range(0, K, BLOCK_K):
            xk = tl.load(
                A + offs_m[:, None] * stride_am + (offs_k[None, :] + kk) * stride_ak,
                mask=(offs_m[:, None] < M) & ((offs_k[None, :] + kk) < K),
                other=0.0,
            ).to(tl.float32)
            acc_abs += tl.sum(tl.abs(xk), axis=1)
        gamma = tl.maximum(acc_abs / K, EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        a = tl.load(
            A + offs_m[:, None] * stride_am + (offs_k[None, :] + kk) * stride_ak,
            mask=(offs_m[:, None] < M) & ((offs_k[None, :] + kk) < K),
            other=0.0,
        ).to(tl.float32)
        if QUANT:
            aq = a / gamma[:, None]
            aq = tl.where(aq >= 0, tl.floor(aq + 0.5), tl.ceil(aq - 0.5))
            aq = tl.minimum(tl.maximum(aq, -127.0), 127.0).to(tl.float16)
        else:
            aq = a.to(tl.float16)

        words = tl.load(
            PW + (offs_k[:, None] + kk) * stride_pk + offs_nw[None, :],
            mask=(offs_k[:, None] + kk) < K,
            other=0,
        ).to(tl.int32)
        # unpack 16 lanes in one shot -> (BLOCK_K, N//16, 16)
        codes = (words[:, :, None] >> (2 * lane[None, None, :])) & 0x03
        s = (codes == 1).to(tl.float16) - (codes == 2).to(tl.float16)
        s_t = tl.reshape(s, (BLOCK_K, BLOCK_N))  # column = nw*16 + lane == n
        acc += tl.dot(aq, s_t)

    if APPLY_SCALE:
        beta = tl.load(BETA + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        if QUANT:
            acc = acc * (gamma[:, None] * beta[None, :])
        else:
            acc = acc * beta[None, :]

    out = acc.to(tl.float16)
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _triton_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    apply_scale: bool,
    quantize: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Run the fused kernel; ``x``/``w`` must be CUDA tensors."""
    if not has_triton:  # pragma: no cover - guarded by callers
        raise RuntimeError("triton is not installed")
    m, k_dim = x.shape
    n = w.shape[0]
    if block_n % 16 != 0:
        raise ValueError(f"block_n must be a multiple of 16, got {block_n}")

    packed = pack_ternary(w).contiguous()           # (K, Np//16) int32
    n_pad = packed.shape[1] * 16
    beta = w.detach().abs().mean(dim=-1).float().contiguous()  # (N,)
    x32 = x.detach().float().contiguous()
    y = torch.empty((m, n_pad), dtype=torch.float16, device=x.device)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n_pad, block_n))
    _packed_ternary_mm_kernel[grid](
        x32, packed, beta, y,
        m, n_pad, k_dim,
        x32.stride(0), x32.stride(1),
        packed.stride(0),
        beta.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        APPLY_SCALE=apply_scale, QUANT=quantize, EPS=_EPS,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y[:, :n]


def packed_ternary_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    apply_scale: bool = True,
    quantize: bool = True,
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> torch.Tensor:
    """Packed BitNet ternary GEMM: fused Triton kernel on CUDA, torch otherwise."""
    if x.is_cuda and has_triton:
        return _triton_forward(
            x, w, apply_scale, quantize,
            block_m, block_n, block_k, num_warps, num_stages,
        )
    return packed_ternary_mm_ref(x, w, apply_scale, quantize)


# --------------------------------------------------------------------------
# autograd integration
# --------------------------------------------------------------------------


class PackedTernaryBitLinear(torch.autograd.Function):
    """BitNet ternary GEMM with STE backward (see module docstring).

    Forward runs the packed Triton kernel on CUDA and the FP32 torch reference
    elsewhere; backward is always computed in torch from the unpacked ternary
    weights, so gradients are identical on every device.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        apply_scale: bool = True,
        quantize: bool = True,
        block_m: int = _BLOCK_M,
        block_n: int = _BLOCK_N,
        block_k: int = _BLOCK_K,
        num_warps: int = _NUM_WARPS,
        num_stages: int = _NUM_STAGES,
    ) -> torch.Tensor:
        ctx.apply_scale = bool(apply_scale)
        ctx.quantize = bool(quantize)
        ctx.save_for_backward(x, w)
        return packed_ternary_matmul(
            x, w, apply_scale, quantize,
            block_m, block_n, block_k, num_warps, num_stages,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        g = grad_out.float()
        xf = x.float()
        s = torch.sign(w)  # (N, K) ternary, STE identity for ternarization
        eps = _EPS
        if ctx.quantize:
            gamma = xf.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
            xq = _round_half_away(xf / gamma).clamp(-127.0, 127.0)
        else:
            xq = xf
            gamma = None
        if ctx.apply_scale:
            beta = w.abs().mean(dim=-1)
            scale = (gamma * beta) if ctx.quantize else beta[None, :]
            gz = g * scale
        else:
            gz = g
        gx = gz @ s
        gw = gz.T @ xq
        return (gx, gw, None, None, None, None, None, None, None)


class PackedBitLinear(nn.Module):
    """BitNet b1.58 ternary linear layer backed by the packed Triton kernel.

    ``weight`` is the real (continuous) parameter; it is packed to 2-bit
    ternary codes only when the kernel runs. Checkpoint-compatible with a plain
    ``nn.Linear`` of the same shape.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        apply_scale: bool = True,
        quantize: bool = True,
        block_m: int = _BLOCK_M,
        block_n: int = _BLOCK_N,
        block_k: int = _BLOCK_K,
        num_warps: int = _NUM_WARPS,
        num_stages: int = _NUM_STAGES,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive, got ({in_features}, {out_features})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.apply_scale = apply_scale
        self.quantize = quantize
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)

    def packed_weight(self) -> torch.Tensor:
        return pack_ternary(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x if x.dim() <= 2 else x.reshape(-1, x.shape[-1])
        out = PackedTernaryBitLinear.apply(
            flat, self.weight,
            self.apply_scale, self.quantize,
            self.block_m, self.block_n, self.block_k,
            self.num_warps, self.num_stages,
        )
        if x.dim() <= 2:
            return out
        return out.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"apply_scale={self.apply_scale}, quantize={self.quantize}"
        )
