"""CLI auto-tuner: find the best FFF depth vs. quantization threshold on M4.

Sweeps tree depth, the AbsMean ternary threshold scale and activation bits,
measuring synchronized latency and observed peak unified memory, rejecting any
config whose peak exceeds the 16 GB budget. The winner is written to JSON.

Usage:
    python benchmarks/tune_fff.py [--budget-gb 16] [--batch 2048]
        [--d-model 256] [--n-layers 1] [--depths 2 3 4 5]
        [--thresholds 0.5 1.0 1.5] [--bits 8 16] [--iters 10]
        [--out best_fff.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.tuning import DEFAULT_BITS, DEFAULT_DEPTHS, DEFAULT_THRESHOLDS, autotune


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget-gb", type=float, default=16.0)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seq", type=int, default=32,
                   help="sequence length; keep small so the FFF dominates the measurement")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=0, help="0 = raw d_model core")
    p.add_argument("--depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--bits", nargs="+", type=int, default=list(DEFAULT_BITS))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default="best_fff.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    cfg = BitNetFFTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    x = torch.randn(args.batch, args.seq, args.d_model, device=device)

    def builder(c: BitNetFFTConfig):
        return BitNetFFTTransformer(c)

    best, samples = autotune(
        builder,
        cfg,
        x,
        depths=tuple(args.depths),
        thresholds=tuple(args.thresholds),
        bits=tuple(args.bits),
        batches=(args.batch,),
        budget_gb=args.budget_gb,
        n_warmup=args.warmup,
        n_iters=args.iters,
        device=device,
    )

    print(f"\n[{device}] budget={args.budget_gb}GB batch={args.batch} "
          f"d_model={args.d_model} sweeps={len(samples)}")
    print(f"{'depth':>6}{'thr':>6}{'bits':>5}{'batch':>7}{'lat(ms)':>10}"
          f"{'tok/s':>14}{'peakMB':>10}{'drvGB':>8} ok")
    for s in samples:
        print(f"{s.depth:>6}{s.threshold_scale:>6.2f}{s.activation_bits:>5}"
              f"{s.batch:>7}{s.latency_ms:>10.3f}{s.throughput_tps:>14,.0f}"
              f"{s.peak_mb:>10.1f}{s.driver_gb:>8.2f} {'*' if s.within_budget else 'x'}")

    print(f"\nBEST: depth={best.depth} threshold_scale={best.threshold_scale:.2f} "
          f"activation_bits={best.activation_bits} batch={best.batch} "
          f"latency={best.latency_ms:.3f}ms throughput={best.throughput_tps:,.0f} tok/s")
    out = Path(args.out)
    out.write_text(json.dumps({"best": best.as_dict(), "samples": [s.as_dict() for s in samples]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
