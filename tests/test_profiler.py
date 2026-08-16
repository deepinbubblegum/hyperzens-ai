"""Tests for the per-layer MPS profiler."""

from __future__ import annotations

import pytest
import torch

from bitnet_fff.models import BitNetFFTConfig, BitNetFFTTransformer
from bitnet_fff.profiler import LayerProfiler, layer_summary_table

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _model() -> torch.nn.Module:
    cfg = BitNetFFTConfig(
        vocab_size=0, d_model=32, n_heads=2, n_layers=2, fff_depth=2
    )
    return BitNetFFTTransformer(cfg).to(DEVICE).eval()


def test_layer_profiler_records_times():
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(8, 32, device=DEVICE)
    prof = LayerProfiler(m, sync=True).run(x, n_warmup=1, n_iters=3)
    summary = prof.summary()
    assert len(summary) >= 5
    for name, r in summary.items():
        assert r["best_ms"] <= r["mean_ms"]
        if name.startswith("layers."):
            expected = 4 * m.cfg.recurrent_steps  # layers run once per recurrent step
        else:
            expected = 4  # final norm / root run once per forward
        assert r["calls"] == expected
    assert summary["<root>"]["self_ms"] >= 0.0


def test_context_manager():
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(8, 32, device=DEVICE)
    with LayerProfiler(m) as prof:
        m(x)
    assert prof.summary()


def test_layer_summary_table_format():
    s = layer_summary_table({"a": {"best_ms": 1.0, "mean_ms": 2.0, "self_ms": 1.5, "calls": 3}})
    assert "a" in s and "mean(ms)" in s


def test_torch_profiler_table():
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(8, 32, device=DEVICE)
    prof = LayerProfiler(m)
    table = prof.torch_profiler_table(x, n_iters=2, row_limit=5)
    assert "Self CPU time total" in table or "Name" in table


def test_signpost_and_metal_capture_guards():
    torch.manual_seed(0)
    m = _model()
    x = torch.randn(8, 32, device=DEVICE)
    prof = LayerProfiler(m)
    prof.signpost(x, n_iters=1, mode="interval")
    prof.signpost(x, n_iters=1, mode="event")
    assert prof.metal_capture(x, path="/tmp/test_fff.gputrace") is False
