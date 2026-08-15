"""Tests for the auto-tuner (depth vs. quantization threshold sweep)."""

from __future__ import annotations

import pytest
import torch

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.tuning import TuneSample, autotune, candidate_configs

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _builder(cfg: BitNetFFTConfig) -> torch.nn.Module:
    return BitNetFFTTransformer(cfg).to(DEVICE).eval()


def test_candidate_configs_cartesian():
    configs = list(candidate_configs(depths=(2, 3), thresholds=(1.0,), bits=(8,), batches=(4,)))
    assert len(configs) == 2
    assert configs[0][0] == 2 and configs[1][0] == 3


def test_autotune_returns_best_within_budget():
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(vocab_size=0, d_model=16, n_heads=2, n_layers=1, fff_depth=2)
    x = torch.randn(16, 16, device=DEVICE)
    best, samples = autotune(
        _builder,
        cfg,
        x,
        depths=(2, 3),
        thresholds=(0.5, 1.0),
        bits=(8, 16),
        batches=(16,),
        budget_gb=16.0,
        n_warmup=1,
        n_iters=3,
        device=DEVICE,
    )
    assert isinstance(best, TuneSample)
    assert len(samples) == 2 * 2 * 2
    assert best.within_budget
    assert best in samples


def test_autotune_budget_rejection():
    torch.manual_seed(0)
    cfg = BitNetFFTConfig(vocab_size=0, d_model=16, n_heads=1, n_layers=1, fff_depth=2)
    x = torch.randn(8, 16, device=DEVICE)
    best, samples = autotune(
        _builder,
        cfg,
        x,
        depths=(2,),
        thresholds=(1.0,),
        bits=(8,),
        batches=(8,),
        budget_gb=1e-6,  # impossible budget -> nothing within budget
        n_warmup=1,
        n_iters=2,
        device=DEVICE,
    )
    assert all(not s.within_budget for s in samples)
    assert best in samples  # falls back to global best
