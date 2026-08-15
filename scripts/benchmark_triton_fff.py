#!/usr/bin/env python3
"""Benchmark the fused Triton FFF kernel vs the PyTorch-native reference.

Runs on an NVIDIA CUDA GPU only (the Triton kernel is Ampere-or-newer
tensor-core code, CC >= 8.0; add/sub ternary routing has no CUDA arch guard so
it compiles anywhere triton does, but it is tuned for 8.6+). Exits with a clear
message on machines without CUDA/triton.

For every batch size in ``--batch-sizes`` it times, with ``torch.cuda.Event``
timers over ``--reps`` iterations (after ``--warmup``):

* ``native``  - :func:`bitnet_fff.triton_fff.fff_forward_ref` on pre-ternarized
  FP16 weights (the same math, PyTorch ops, FP32 accumulates).
* ``triton``  - :func:`bitnet_fff.triton_fff._triton_forward` (fused kernel,
  FP16 output).
* ``fwd+bwd`` - when ``--backward`` is set, a full ``FFFTritonLayer`` forward +
  backward (autograd) step, reported as training throughput.

The script also checks the Triton result against the reference once per batch
size and warns when the max abs diff is above ``--max-diff``.

Usage:
    python scripts/benchmark_triton_fff.py --batch-sizes 1,32,64,128 --reps 50
    python scripts/benchmark_triton_fff.py --backward --depth 4 --d-model 256
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_fff import triton_fff  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the fused Triton FFF kernel vs PyTorch native.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=5,
                   help="tree depth; leaves = 2**depth")
    p.add_argument("--batch-sizes", default="1,8,16,32,64,128",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--activation-bits", type=int, default=8)
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--block-n", type=int, default=64,
                   help="Triton BLOCK_N rows per program")
    p.add_argument("--num-warps", type=int, default=8)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--backward", action="store_true",
                   help="also time a full forward+backward training step")
    p.add_argument("--max-diff", type=float, default=2e-3,
                   help="max |triton-ref| abs diff before warning")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _ternarized_fp16(weight: torch.Tensor) -> torch.Tensor:
    """Ternary {-1,0,1} weights in fp16 (what the kernel expects pre-made)."""
    w = weight.detach()
    return torch.where(w.abs() > w.abs().mean().clamp_min(1e-8),
                       torch.sign(w), torch.zeros_like(w)).to(torch.float16)


def _bench_forward(
    x: torch.Tensor,
    wq_router: torch.Tensor,
    b_router: torch.Tensor,
    wq_leaf: torch.Tensor,
    b_leaf: torch.Tensor | None,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """Returns (native_ms, triton_ms, max_abs_diff)."""
    native = triton_fff.fff_forward_ref(
        x, wq_router, b_router, wq_leaf, b_leaf,
        depth=args.depth, activation_bits=args.activation_bits,
    )
    triton_out = triton_fff._triton_forward(
        x, wq_router, b_router, wq_leaf, b_leaf,
        depth=args.depth, activation_bits=args.activation_bits,
        block_n=args.block_n, num_warps=args.num_warps,
    )
    max_diff = (triton_out.float() - native.float()).abs().max().item()

    for _ in range(args.warmup):
        triton_fff.fff_forward_ref(
            x, wq_router, b_router, wq_leaf, b_leaf,
            depth=args.depth, activation_bits=args.activation_bits,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.reps):
        triton_fff.fff_forward_ref(
            x, wq_router, b_router, wq_leaf, b_leaf,
            depth=args.depth, activation_bits=args.activation_bits,
        )
    end.record()
    torch.cuda.synchronize()
    native_ms = start.elapsed_time(end) / args.reps

    for _ in range(args.warmup):
        triton_fff._triton_forward(
            x, wq_router, b_router, wq_leaf, b_leaf,
            depth=args.depth, activation_bits=args.activation_bits,
            block_n=args.block_n, num_warps=args.num_warps,
        )
    torch.cuda.synchronize()
    start.record()
    for _ in range(args.reps):
        triton_fff._triton_forward(
            x, wq_router, b_router, wq_leaf, b_leaf,
            depth=args.depth, activation_bits=args.activation_bits,
            block_n=args.block_n, num_warps=args.num_warps,
        )
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / args.reps
    return native_ms, triton_ms, max_diff


def _bench_training_step(
    x: torch.Tensor,
    layer: triton_fff.FFFTritonLayer,
    args: argparse.Namespace,
) -> float:
    """Full forward+backward training step time (ms) through autograd."""
    for _ in range(args.warmup):
        layer(x).sum().backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.reps):
        layer(x).sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / args.reps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("[bench] CUDA is required; found none. Exiting.")
        return 1
    if not triton_fff.has_triton:
        print("[bench] triton is not installed. Exiting.")
        return 1
    if args.d_model % 16 != 0:
        print(f"[bench] --d-model must be a multiple of 16 "
              f"(tensor-core tile), got {args.d_model}")
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float32
    d, depth = args.d_model, args.depth
    leaves = 1 << depth
    print(f"[bench] device={torch.cuda.get_device_name(0)} "
          f"(CC {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]})")
    print(f"[bench] d_model={d} depth={depth} leaves={leaves} "
          f"activation_bits={args.activation_bits} "
          f"bias={not args.no_bias} block_n={args.block_n} "
          f"num_warps={args.num_warps} reps={args.reps} warmup={args.warmup}")
    print(f"[bench] total params = "
          f"{(leaves - 1) * d + leaves * d * d + (leaves * d if not args.no_bias else 0):,}")

    w_router = torch.randn(leaves - 1, d, device=device, dtype=dtype)
    b_router = torch.zeros(leaves - 1, device=device, dtype=dtype)
    w_leaf = torch.randn(leaves, d, d, device=device, dtype=dtype)
    b_leaf = (
        torch.zeros(leaves, d, device=device, dtype=dtype)
        if not args.no_bias
        else None
    )
    wq_router = _ternarized_fp16(w_router)
    wq_leaf = _ternarized_fp16(w_leaf)

    layer = triton_fff.FFFTritonLayer(
        d_model=d, depth=depth, bias=not args.no_bias,
        activation_bits=args.activation_bits,
    ).to(device)
    with torch.no_grad():
        layer.router_weight.copy_(w_router)
        layer.router_bias.copy_(b_router)
        layer.leaf_weight.copy_(w_leaf)
        if b_leaf is not None:
            layer.leaf_bias.copy_(b_leaf)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    header = (f"{'batch':>8} {'native_ms':>10} {'triton_ms':>10} "
              f"{'speedup':>8} {'native_tok/s':>14} {'triton_tok/s':>14}"
              + (" {'fwd+bwd_ms':>12} {'train_tok/s':>14}" if args.backward else "")
              + "  diff")
    print(header)
    for b in batch_sizes:
        x = torch.randn(b, d, device=device, dtype=dtype)
        native_ms, triton_ms, max_diff = _bench_forward(
            x, wq_router, b_router, wq_leaf, b_leaf, args
        )
        speedup = native_ms / triton_ms if triton_ms > 0 else float("nan")
        native_tok = b * 1000.0 / native_ms
        triton_tok = b * 1000.0 / triton_ms
        row = (f"{b:>8} {native_ms:>10.3f} {triton_ms:>10.3f} "
               f"{speedup:>8.2f}x {native_tok:>14.1f} {triton_tok:>14.1f}")
        if args.backward:
            step_ms = _bench_training_step(x, layer, args)
            row += f" {step_ms:>12.3f} {b * 1000.0 / step_ms:>14.1f}"
        row += f"  {max_diff:.2e}"
        print(row)
        if max_diff > args.max_diff:
            print(f"[bench] WARNING: triton vs native max abs diff "
                  f"{max_diff:.2e} > {args.max_diff} (dtype/rounding drift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
