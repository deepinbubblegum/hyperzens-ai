"""Latency and memory-budget tests for the FFF + BitNet layer on MPS/CPU."""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from bitnet_fff import FastFeedForwardBitNet
from bitnet_fff.mps_utils import (
    is_mps_available,
    mps_current_allocated_bytes,
    mps_empty_cache,
    mps_synchronize,
)

DEVICE = torch.device("mps") if is_mps_available() else torch.device("cpu")


def bench(fn, iters: int = 10, warmup: int = 3):
    for _ in range(warmup):
        fn()
    mps_synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        mps_synchronize()
        times.append(time.perf_counter() - start)
    return min(times), statistics.mean(times)


@pytest.fixture()
def fff():
    model = FastFeedForwardBitNet(
        d_in=256, d_out=256, depth=3, bias=True
    ).to(DEVICE)
    x = torch.randn(2048, 256, device=DEVICE)
    return model, x


def test_forward_and_backward_latency_budget(fff):
    model, x = fff
    forward_best, forward_mean = bench(lambda: model(x))
    backward_best, backward_mean = bench(lambda: model(x).sum().backward())
    print(
        f"\n[{DEVICE}] forward best={forward_best * 1e3:.1f}ms mean={forward_mean * 1e3:.1f}ms | "
        f"backward best={backward_best * 1e3:.1f}ms mean={backward_mean * 1e3:.1f}ms"
    )
    if is_mps_available():
        assert forward_mean < 0.5
        assert backward_mean < 1.5
    else:
        assert forward_mean < 5.0
        assert backward_mean < 8.0


def test_conditional_execution_latency_is_flat_in_depth():
    x = torch.randn(512, 256, device=DEVICE)
    dense_times = {}
    fff_times = {}
    for depth in (2, 4):
        fff = FastFeedForwardBitNet(
            d_in=256, d_out=256, depth=depth, bias=True
        ).to(DEVICE)
        dense = fff.to_dense().to(DEVICE)
        fff_times[depth], _ = bench(lambda m=fff: m(x))
        dense_times[depth], _ = bench(lambda m=dense: m(x))

    fff_2, fff_4 = fff_times[2], fff_times[4]
    dense_2, dense_4 = dense_times[2], dense_times[4]
    print(
        f"\n[{DEVICE}] FFF depth2={fff_2 * 1e3:.2f}ms depth4={fff_4 * 1e3:.2f}ms | "
        f"dense depth2={dense_2 * 1e3:.2f}ms depth4={dense_4 * 1e3:.2f}ms"
    )
    assert fff_4 <= fff_2 * 1.5, "FFF leaf math is depth-independent; latency must stay flat"
    assert dense_4 >= dense_2 * 1.8, "dense latency must grow linearly with capacity"


def test_peak_memory_stays_within_unified_budget(fff):
    model, x = fff
    if not is_mps_available():
        pytest.skip("MPS only")
    temp_bytes = model.leaf_gather_temp_bytes(x.shape[0])
    temp_gb = temp_bytes / (1024 ** 3)
    print(f"\n[MPS] leaf gather temp = {temp_gb:.2f} GB")
    assert temp_gb < 1.0, "single forward intermediate must stay well under the 16 GB budget"
    mps_empty_cache()
    mps_synchronize()
    baseline = mps_current_allocated_bytes()
    peak = baseline
    for _ in range(5):
        model(x)
        mps_synchronize()
        peak = max(peak, mps_current_allocated_bytes())
    growth_gb = (peak - baseline) / (1024 ** 3)
    print(f"[MPS] observed allocated growth after fwd = {growth_gb:.2f} GB")


def test_chunked_gather_caps_peak_memory():
    if not is_mps_available():
        pytest.skip("MPS only")
    batch, chunk = 2048, 256
    full = FastFeedForwardBitNet(d_in=256, d_out=256, depth=3, bias=True)
    chunked = FastFeedForwardBitNet(
        d_in=256, d_out=256, depth=3, bias=True, chunk_size=chunk
    )
    full_temp = full.leaf_gather_temp_bytes(batch)
    chunked_temp = chunked.leaf_gather_temp_bytes(batch)
    print(
        f"\n[MPS] leaf gather temp: full={full_temp / 1e6:.0f}MB "
        f"chunked({chunk})={chunked_temp / 1e6:.0f}MB"
    )
    assert chunked_temp <= full_temp * (chunk / batch)
    assert chunked_temp < full_temp

    model = chunked.to(DEVICE)
    x = torch.randn(batch, 256, device=DEVICE)
    mps_empty_cache()
    mps_synchronize()
    baseline = mps_current_allocated_bytes()
    peak = baseline
    for _ in range(5):
        model(x)
        mps_synchronize()
        peak = max(peak, mps_current_allocated_bytes())
    growth_gb = (peak - baseline) / (1024 ** 3)
    print(f"[MPS] chunked observed allocated growth = {growth_gb:.2f} GB")
