"""2-bit packed BitNet ternary matrix multiplication with a fused Triton kernel.

BitNet b1.58-style GEMM where the weight is decomposed as ``W ~= S * beta``
with ``S in {-1, 0, +1}`` (ternary) and ``beta = mean(|W|)`` per output row.
The ternary weights are packed 2 bits each (4 per byte, 16 per int32 word) and
unpacked on the fly inside the Triton kernel with bit masks/shifts
(``(packed >> shift) & 0x03``), so the weight tensor in VRAM is 1/8 the FP16
size (1/16 the FP32 size) and the matmul runs on FP16 Ampere tensor cores via ``tl.dot``.

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

Backward uses the standard BitNet straight-through estimator (STE): quantization
and ternarization are treated as identity and the fused scales as constants, so
the packed kernel itself needs no backward path - gradients are computed in
torch from the unpacked ternary weights:
    gZ = gY * (gamma * beta)
    gX = gZ @ S
    gW = gZ^T @ Xq

Kernel is tuned for consumer Ampere (RTX 3060, CC 8.6, 28 SMs):
BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3.
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
    "pack_ternary_int8",
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


def pack_ternary(w: torch.Tensor, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Pack ternary weights {-1, 0, +1} into 2-bit codes along the output dimension.

    Encoding:
        0 -> 0 (00_2)
        1 -> +1 (01_2)
        -1 -> -1 (10_2, mapped to code 2)
        3 -> unused

    Args:
        w: Weight tensor of shape ``(N, K)`` where ``N=out_features``, ``K=in_features``.
        dtype: Output integer dtype (``torch.int32`` for 16 weights/word,
            ``torch.int8`` for 4 weights/byte).

    Returns:
        packed: Packed tensor of shape ``(K, ceil(N / pack_factor))`` where
        ``pack_factor = 16`` for int32 and ``4`` for int8.
    """
    w_det = w.detach()
    s = torch.sign(w_det).to(torch.int32)
    n, k = s.shape

    # Map -1 -> 2, 1 -> 1, 0 -> 0
    codes = torch.where(
        s == 1,
        torch.ones_like(s),
        torch.where(
            s == -1,
            torch.full_like(s, 2),
            torch.zeros_like(s),
        ),
    )

    if dtype == torch.int32:
        pack_factor = 16
    elif dtype == torch.int8:
        pack_factor = 4
    else:
        raise ValueError(f"Unsupported packing dtype {dtype}, expected torch.int32 or torch.int8")

    n_pad = ((n + pack_factor - 1) // pack_factor) * pack_factor
    if n_pad > n:
        codes = F.pad(codes, (0, 0, 0, n_pad - n))

    ct = codes.T.reshape(k, n_pad // pack_factor, pack_factor)
    packed = torch.zeros(k, n_pad // pack_factor, dtype=dtype, device=w.device)
    for lane in range(pack_factor):
        packed |= (ct[..., lane].to(dtype) << (2 * lane))
    return packed


def pack_ternary_int8(w: torch.Tensor) -> torch.Tensor:
    """Convenience helper to pack ternary weights into 2-bit values (4 per int8 byte)."""
    return pack_ternary(w, dtype=torch.int8)


def unpack_ternary(
    packed: torch.Tensor, n: int | None = None, k: int | None = None
) -> torch.Tensor:
    """Unpack 2-bit packed codes back into FP32 ternary weights {-1, 0, +1} of shape (N, K).

    Handles both int32 (16 weights/word) and int8 (4 weights/byte) packed tensors.
    """
    k_dim, n_words = packed.shape
    if packed.dtype == torch.int32:
        pack_factor = 16
        shift_mask = 0x03
    elif packed.dtype == torch.int8:
        pack_factor = 4
        shift_mask = 0x03
    else:
        raise ValueError(f"Unsupported packed dtype: {packed.dtype}")

    n_pad = n_words * pack_factor
    codes = torch.zeros(k_dim, n_words, pack_factor, dtype=torch.int32, device=packed.device)
    # Convert packed to int32 with positive bit masking to avoid signed shifts in int8
    packed_i32 = packed.to(torch.int32) & (0xFF if pack_factor == 4 else 0xFFFFFFFF)
    for lane in range(pack_factor):
        codes[..., lane] = (packed_i32 >> (2 * lane)) & shift_mask

    vals = torch.where(
        codes == 1,
        torch.ones_like(codes, dtype=torch.float32),
        torch.where(
            codes == 2,
            -torch.ones_like(codes, dtype=torch.float32),
            torch.zeros_like(codes, dtype=torch.float32),
        ),
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
    """Round half away from zero (matches the Triton floor/ceil implementation)."""
    return torch.where(t >= 0, torch.floor(t + 0.5), torch.ceil(t - 0.5))


def packed_ternary_mm_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    apply_scale: bool = True,
    quantize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PyTorch reference for the packed ternary GEMM."""
    calc_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    x_in = x.detach().to(calc_dtype)
    s = unpack_ternary(pack_ternary(w, dtype=torch.int32), n=w.shape[0], k=w.shape[1]).to(calc_dtype)
    if quantize:
        gamma = x_in.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        xq = _round_half_away(x_in / gamma).clamp(-127.0, 127.0)
        z = xq @ s.T
        if apply_scale:
            beta = w.detach().abs().mean(dim=-1).to(calc_dtype)
            z = z * (gamma * beta)
        else:
            z = z * gamma
    else:
        z = x_in @ s.T
        if apply_scale:
            beta = w.detach().abs().mean(dim=-1).to(calc_dtype)
            z = z * beta
    return z.to(x.dtype)


# --------------------------------------------------------------------------
# Triton kernel (Optimized for Ampere / RTX 3060)
# --------------------------------------------------------------------------


@_jit
def _packed_ternary_mm_kernel(
    A, PW, BETA, Y,
    M, N, K,
    stride_am, stride_ak,
    stride_pk, stride_pn,
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

    Optimized for NVIDIA Ampere (RTX 3060, CC 8.6, 28 SMs, 112 Tensor Cores):
    1. 2-bit packed weights loaded as 32-bit words (16 channels/word).
    2. Dynamic register-level on-the-fly unpacking via bit shifts and masks
       (``(words >> (2 * lane)) & 0x03``) producing FP16 values in 1 cycle.
    3. Per-token activation scaling factor ``gamma = mean(|X|)`` and 8-bit quantization.
    4. Tensor Core GEMM via ``tl.dot`` (FP16 inputs, FP32 accumulator).
    5. Fused block-output scale multiplication: ``Y = Z * (gamma * beta)``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Word offset along N dimension: 16 output channels per int32 word
    offs_nw = pid_n * (BLOCK_N // 16) + tl.arange(0, BLOCK_N // 16)
    offs_k = tl.arange(0, BLOCK_K)
    lane = tl.arange(0, 16)

    if QUANT:
        # Pass 1: Compute per-token activation scale gamma = mean(abs(X)) along K
        acc_abs = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for kk in range(0, K, BLOCK_K):
            k_indices = offs_k[None, :] + kk
            a_mask = (offs_m[:, None] < M) & (k_indices < K)
            xk = tl.load(
                A + offs_m[:, None] * stride_am + k_indices * stride_ak,
                mask=a_mask,
                other=0.0,
            ).to(tl.float32)
            acc_abs += tl.sum(tl.abs(xk), axis=1)
        gamma = tl.maximum(acc_abs / K, EPS)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pass 2: Main GEMM loop with on-the-fly unpacking and Tensor Core tl.dot
    for kk in range(0, K, BLOCK_K):
        k_indices = offs_k[None, :] + kk
        a_mask = (offs_m[:, None] < M) & (k_indices < K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + k_indices * stride_ak,
            mask=a_mask,
            other=0.0,
        ).to(tl.float32)

        if QUANT:
            # 8-bit dynamic quantization: clamp(round(X / gamma), -127, 127)
            aq = a / gamma[:, None]
            aq = tl.where(aq >= 0, tl.floor(aq + 0.5), tl.ceil(aq - 0.5))
            aq = tl.minimum(tl.maximum(aq, -127.0), 127.0).to(tl.float16)
        else:
            aq = a.to(tl.float16)

        # Load packed int32 words: shape (BLOCK_K, BLOCK_N // 16)
        w_mask = ((offs_k[:, None] + kk) < K) & (offs_nw[None, :] < (N + 15) // 16)
        words = tl.load(
            PW + (offs_k[:, None] + kk) * stride_pk + offs_nw[None, :] * stride_pn,
            mask=w_mask,
            other=0,
        ).to(tl.int32)

        # On-the-fly register unpacking:
        # words: (BLOCK_K, BLOCK_N // 16, 1)
        # lane:  (1, 1, 16)
        # codes: (BLOCK_K, BLOCK_N // 16, 16)
        codes = (words[:, :, None] >> (2 * lane[None, None, :])) & 0x03

        # Direct 1-cycle conversion: 1 -> +1.0, 2 -> -1.0, 0/3 -> 0.0
        s = (codes == 1).to(tl.float16) - (codes == 2).to(tl.float16)

        # Reshape to (BLOCK_K, BLOCK_N) matching GEMM layout (k, n)
        s_tile = tl.reshape(s, (BLOCK_K, BLOCK_N))

        # Ampere Tensor Core dot product (FP16 x FP16 -> FP32 accumulation)
        acc += tl.dot(aq, s_tile)

    # Fused scaling at block output
    if APPLY_SCALE:
        beta = tl.load(BETA + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        if QUANT:
            acc = acc * (gamma[:, None] * beta[None, :])
        else:
            acc = acc * beta[None, :]
    elif QUANT:
        acc = acc * gamma[:, None]

    out = acc.to(tl.float16)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=out_mask,
    )


def _triton_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    apply_scale: bool,
    quantize: bool,
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> torch.Tensor:
    """Launch the fused Triton kernel on CUDA."""
    if not has_triton:  # pragma: no cover - guarded by callers
        raise RuntimeError("triton is not installed")
    if not x.is_cuda:
        raise ValueError("Input tensor x must be on CUDA")

    m, k_dim = x.shape
    n = w.shape[0]
    if block_n % 16 != 0:
        raise ValueError(f"block_n must be a multiple of 16, got {block_n}")

    packed = pack_ternary(w, dtype=torch.int32).contiguous()  # (K, ceil(N / 16))
    n_pad = packed.shape[1] * 16
    beta = w.detach().abs().mean(dim=-1).float().contiguous()  # (N,)
    x32 = x.detach().float().contiguous()
    y = torch.empty((m, n_pad), dtype=torch.float16, device=x.device)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n_pad, block_n))
    _packed_ternary_mm_kernel[grid](
        x32, packed, beta, y,
        m, n, k_dim,
        x32.stride(0), x32.stride(1),
        packed.stride(0), packed.stride(1),
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
    """Packed BitNet ternary GEMM: fused Triton kernel on CUDA, torch reference fallback elsewhere."""
    if x.is_cuda and has_triton:
        return _triton_forward(
            x, w, apply_scale, quantize,
            block_m, block_n, block_k, num_warps, num_stages,
        )
    return packed_ternary_mm_ref(x, w, apply_scale, quantize)


# --------------------------------------------------------------------------
# PyTorch autograd integration (forward and backward with BitNet STE)
# --------------------------------------------------------------------------


class PackedTernaryBitLinear(torch.autograd.Function):
    """BitNet ternary GEMM autograd Function with Straight-Through Estimator (STE).

    Forward runs the packed Triton kernel on CUDA and the FP32 torch reference
    elsewhere; backward is computed using the BitNet straight-through estimator (STE),
    treating quantization and ternarization as identity, returning gradients matching
    input dtypes and devices.
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
        calc_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        g = grad_out.to(calc_dtype)
        xf = x.to(calc_dtype)
        s = torch.sign(w).to(calc_dtype)  # (N, K) ternary, STE identity for ternarization
        eps = _EPS
        if ctx.quantize:
            gamma = xf.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
            xq = _round_half_away(xf / gamma).clamp(-127.0, 127.0)
        else:
            xq = xf
            gamma = 1.0

        if ctx.apply_scale:
            beta = w.detach().abs().mean(dim=-1).to(calc_dtype)
            scale = (gamma * beta) if ctx.quantize else beta[None, :]
            gz = g * scale
        elif ctx.quantize:
            gz = g * gamma
        else:
            gz = g

        gx = (gz @ s).to(x.dtype)
        gw = (gz.T @ xq).to(w.dtype)
        return (gx, gw, None, None, None, None, None, None, None)


class PackedBitLinear(nn.Module):
    """BitNet b1.58 ternary linear layer backed by the packed Triton kernel.

    ``weight`` is the real (continuous) parameter ``(out_features, in_features)``;
    it is packed to 2-bit ternary codes on the fly when the kernel runs.
    Compatible with torch.nn.Linear and supports 2D ``(B, K)`` and 3D ``(B, T, K)``
    input tensors.
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

    def packed_weight(self, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        """Returns the weight packed into 2-bit format (int32 or int8)."""
        return pack_ternary(self.weight, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x if x.dim() <= 2 else x.reshape(-1, orig_shape[-1])
        out = PackedTernaryBitLinear.apply(
            flat, self.weight,
            self.apply_scale, self.quantize,
            self.block_m, self.block_n, self.block_k,
            self.num_warps, self.num_stages,
        )
        if x.dim() <= 2:
            return out
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"apply_scale={self.apply_scale}, quantize={self.quantize}"
        )

