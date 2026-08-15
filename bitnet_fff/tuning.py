"""Auto-tuning: discover the best FFF depth vs. BitNet quantization threshold.

Sweeps ``(depth, ternarize_threshold_scale, activation_bits, batch)`` over a
fresh model instance per candidate and measures synchronized MPS latency plus
observed peak unified-memory growth. Candidates whose observed peak exceeds the
``budget_gb`` are rejected; the winner maximizes throughput (tokens/sec) with
latency as the tie-break. The sweep is embarrassingly parallel-friendly but is
run serially to keep peak memory bounded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch

from .mps_utils import (
    is_mps_available,
    mps_current_allocated_bytes,
    mps_empty_cache,
    mps_synchronize,
)

__all__ = ["TuneSample", "candidate_configs", "measure_run", "autotune"]

DEFAULT_DEPTHS = (2, 3, 4, 5, 6)
DEFAULT_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_BITS = (8, 16)


@dataclass
class TuneSample:
    """One measured configuration."""

    depth: int
    threshold_scale: float
    activation_bits: int
    batch: int
    latency_ms: float
    throughput_tps: float
    peak_mb: float
    analytic_mb: float
    driver_gb: float
    within_budget: bool

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "threshold_scale": self.threshold_scale,
            "activation_bits": self.activation_bits,
            "batch": self.batch,
            "latency_ms": round(self.latency_ms, 3),
            "throughput_tps": round(self.throughput_tps, 1),
            "peak_mb": round(self.peak_mb, 2),
            "analytic_mb": round(self.analytic_mb, 2),
            "driver_gb": round(self.driver_gb, 3),
            "within_budget": self.within_budget,
        }


def candidate_configs(
    depths=DEFAULT_DEPTHS,
    thresholds=DEFAULT_THRESHOLDS,
    bits=DEFAULT_BITS,
    batches=(1,),
):
    """Yield ``(depth, threshold_scale, activation_bits, batch)`` tuples."""
    for batch in batches:
        for depth in depths:
            for threshold in thresholds:
                for b in bits:
                    yield depth, threshold, b, batch


def measure_run(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> tuple[float, float, float, float]:
    """Synchronized latency + peak unified-memory for one forward.

    Returns ``(best_ms, mean_ms, peak_mb, driver_gb)``. ``peak_mb`` is the
    largest ``current_allocated_memory`` sampled after each iteration relative
    to a pre-run baseline; ``driver_gb`` is the monotonic Metal driver
    allocation (cumulative high-water mark, useful as a global ceiling).
    """
    for _ in range(n_warmup):
        model(x)
    mps_synchronize()
    mps_empty_cache()
    mps_synchronize()
    baseline = mps_current_allocated_bytes()
    times = []
    peak = baseline
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(x)
        mps_synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        peak = max(peak, mps_current_allocated_bytes())
    driver = torch.mps.driver_allocated_memory() if is_mps_available() else 0
    return min(times), sum(times) / len(times), (peak - baseline) / 1e6, driver / 1e9


def autotune(
    builder,
    cfg,
    x: torch.Tensor,
    depths=DEFAULT_DEPTHS,
    thresholds=DEFAULT_THRESHOLDS,
    bits=DEFAULT_BITS,
    batches=(1,),
    budget_gb: float = 16.0,
    n_warmup: int = 3,
    n_iters: int = 10,
    device: torch.device | str | None = None,
) -> tuple[TuneSample, list[TuneSample]]:
    """Sweep ``builder(cfg)`` over depth/threshold/bits and pick the best.

    ``builder`` must return a fresh module for the given (mutated) config so
    the tree depth and quantization threshold actually change between runs.
    The memory footprint is estimated as the observed ``current_allocated``
    peak plus an analytic model-resident + leaf-gather estimate (the latter
    dominates for the plain FFN path); a candidate is rejected when that sum
    exceeds ``budget_gb``. The winner maximizes tokens/sec among the
    accepted configurations.
    """
    if not is_mps_available() and device is None:
        device = torch.device("cpu")
    dev = torch.device(device) if device is not None else x.device

    samples: list[TuneSample] = []
    for depth, threshold, b, batch in candidate_configs(
        depths=depths, thresholds=thresholds, bits=bits, batches=batches
    ):
        trial_cfg = replace(
            cfg,
            fff_depth=depth,
            fff_threshold_scale=threshold,
            activation_bits=b,
        )
        model = builder(trial_cfg).to(dev)
        model.eval()
        xb = x[:batch]
        best, mean, peak_mb, driver_gb = measure_run(
            model, xb, n_warmup=n_warmup, n_iters=n_iters
        )
        analytic_mb = _analytic_mb(model, batch)
        within = peak_mb + analytic_mb < budget_gb * 1024.0
        tokens = xb.numel() // xb.shape[-1]
        samples.append(
            TuneSample(
                depth=depth,
                threshold_scale=threshold,
                activation_bits=b,
                batch=batch,
                latency_ms=best,
                throughput_tps=tokens / (mean / 1e3),
                peak_mb=peak_mb,
                analytic_mb=analytic_mb,
                driver_gb=driver_gb,
                within_budget=within,
            )
        )
        mps_empty_cache()
        mps_synchronize()

    eligible = [s for s in samples if s.within_budget]
    pool = eligible or samples
    best = max(pool, key=lambda s: (s.throughput_tps, -s.latency_ms))
    return best, samples


def _analytic_mb(model: torch.nn.Module, batch: int) -> float:
    """Deterministic residency estimate: params + worst-case leaf gather."""
    mb = 0.0
    for p in model.parameters():
        mb += p.numel() * (2 if p.dtype == torch.float16 else 4)
    for _, mod in model.named_modules():
        gather = getattr(mod, "leaf_gather_temp_bytes", None)
        if callable(gather):
            mb += float(gather(batch))
    return mb / 1e6
