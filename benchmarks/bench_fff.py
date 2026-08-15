"""Benchmark the fused single-pass routing kernel vs the reference FFF.

Reports per-method latency (ms), peak unified-memory growth (MB) and
throughput (tokens/sec) for the MPS (Metal) and CPU (NEON) kernels:

* ``fff fast_forward (fused single-pass)`` - the fused kernel: routing +
  activation quantization + packed-ternary leaf GEMM in one native call
  (no host round-trip, no per-branch buffers).
* ``fff two-stage (packed)`` - the previous two-stage implementation: torch
  ``_routing_forward`` producing ``leaf_idx``, then the classic packed leaf
  matmul. The fused/two-stage speedup ratio is printed (for ``--depth 8`` this
  is the headline deliverable).
* ``fff fp32 forward`` - the reference PyTorch FFF (gather + bmm).
* ``dense linear`` - the capacity-equivalent dense linear layer.

Usage:
    python benchmarks/bench_fff.py [--device cpu|mps] [--batch 2048]
        [--d-in 256] [--d-out 256] [--depth 8] [--iters 30]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet_fff import FastFeedForwardBitNet
from bitnet_fff.fast_inference import PackedFFFEvaluator
from bitnet_fff.mps_utils import (
    is_mps_available,
    mps_current_allocated_bytes,
    mps_empty_cache,
    mps_synchronize,
    tensor_bytes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "mps"), default=None)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--d-in", type=int, default=256)
    p.add_argument("--d-out", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--chunk", type=int, default=None,
                   help="chunk batch for the packed path to cap memory")
    return p.parse_args()


def bench(fn, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    mps_synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        mps_synchronize()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.mean(times)


def main() -> None:
    args = parse_args()
    device = args.device or ("mps" if is_mps_available() else "cpu")
    dev = torch.device(device)

    torch.manual_seed(0)
    model = FastFeedForwardBitNet(
        d_in=args.d_in, d_out=args.d_out, depth=args.depth, bias=True
    ).to(dev)
    model.eval()
    dense = model.to_dense().to(dev)
    x = torch.randn(args.batch, args.d_in, device=dev)

    mps_empty_cache()
    mps_synchronize()
    base_current = mps_current_allocated_bytes()

    # The previous two-stage routing implementation: torch routing + classic
    # packed leaf matmul (what fast_forward did before the fused kernel).
    two_stage = PackedFFFEvaluator(model, device=dev)

    def two_stage_forward(t: torch.Tensor) -> torch.Tensor:
        leaf, _ = model._routing_forward(t)
        return two_stage(t, leaf)

    methods: dict[str, object] = {
        "fff fp32 forward": lambda: model(x),
        "fff fast_forward (fused single-pass)": lambda: model.fast_forward(x),
        "fff two-stage (packed)": lambda: two_stage_forward(x),
        "dense linear": lambda: dense(x),
    }
    if args.chunk is not None:
        methods["fff fast_forward chunked"] = lambda: model.fast_forward(
            x, chunk_size=args.chunk
        )

    # pack once outside the timing loop so first-call compile/build is excluded
    model.fast_forward(x[: min(64, args.batch)])
    two_stage_forward(x[: min(64, args.batch)])
    mps_synchronize()

    print(f"\n[{device}] d_in={args.d_in} d_out={args.d_out} depth={args.depth} "
          f"batch={args.batch} iters={args.iters}")
    print(f"{'method':<30}{'best(ms)':>12}{'mean(ms)':>12}"
          f"{'peakΔMB':>12}{'tok/s':>16}")
    results: dict[str, tuple[float, float]] = {}
    for name, fn in methods.items():
        best, mean = bench(fn, args.iters, args.warmup)
        mps_empty_cache()
        mps_synchronize()
        peak = mps_current_allocated_bytes() - base_current
        peak_mb = peak / 1e6
        tok_s = (args.batch * args.iters) / (mean * args.iters) if mean > 0 else float("nan")
        print(f"{name:<30}{best * 1e3:>12.2f}{mean * 1e3:>12.2f}"
              f"{peak_mb:>12.1f}{tok_s:>16,.0f}")
        results[name] = (best, mean)

    def speedup(against: str) -> str:
        fused = results.get("fff fast_forward (fused single-pass)")
        other = results.get(against)
        if fused and other and other[1] > 0:
            return f"{other[1] / fused[1]:.2f}x"
        return "n/a"

    print(f"\nfused single-pass speedup @depth={args.depth}:")
    print(f"  vs two-stage routing (packed): {speedup('fff two-stage (packed)')} "
          f"(mean latency)")
    print(f"  vs fp32 reference forward:     {speedup('fff fp32 forward')} "
          f"(mean latency)")

    # memory accounting for the packed path
    evalr = model._packed_eval
    packed_mb = tensor_bytes(evalr.packed) / 1e6 if evalr is not None else 0.0
    fused_extra_mb = (
        tensor_bytes(evalr.router_w) + tensor_bytes(evalr.router_b)
    ) / 1e6 if evalr is not None else 0.0
    print(f"\npacked weights: {packed_mb:.2f} MB "
          f"({tensor_bytes(model.leaf_weight.data) / 1e6:.2f} MB fp32 leaf_weight, "
          f"+{fused_extra_mb:.2f} MB router)")
    print(f"leaf-gather temp (fp32 path) vs packed path: "
          f"{model.leaf_gather_temp_bytes(args.batch) / 1e6:.1f} MB -> "
          f"{args.batch * (args.d_in + args.d_out) * 4 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
