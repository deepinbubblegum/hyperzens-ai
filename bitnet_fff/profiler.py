"""Per-layer latency profiling on Apple Silicon (MPS).

Combines three layers of measurement:

1. **Synchronized wall-clock hooks** (:class:`LayerProfiler`) - a forward hook
   is installed on every submodule; each layer's best/mean latency is recorded
   with ``torch.mps.synchronize()`` so numbers reflect real GPU time, and
   ``self`` time subtracts the children from the parent.
2. **OS Signpost tracing** (:meth:`LayerProfiler.signpost`) - wraps a run in
   ``torch.mps.profiler.profile()`` producing signposts readable from Xcode
   Instruments (Logging), including ``event`` / ``interval`` modes.
3. **Metal GPU trace capture** (:meth:`LayerProfiler.metal_capture`) - wraps a
   run in ``torch.mps.profiler.metal_capture(path)`` producing a ``.gputrace``
   for Xcode's GPU timeline (needs ``MTL_CAPTURE_ENABLED=1``).
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict

import torch

from .mps_utils import is_mps_available, mps_synchronize

__all__ = ["LayerProfiler", "layer_summary_table"]


def _module_name(root: torch.nn.Module, module: torch.nn.Module) -> str:
    for name, m in root.named_modules():
        if m is module:
            return name or "<root>"
    return "<unknown>"


class LayerProfiler:
    """Profiles every submodule of ``model`` with synchronized forward hooks.

    Args:
        model: the module to profile.
        sync: synchronize the MPS stream after every layer (accurate GPU time).
            Models with FP16 parameters skip per-module syncs - a Metal
            runtime assertion fires when MPSNDArray fp16 matmuls are
            interrupted between layers - and sync once at the end of the run
            instead.
        skip: module types (or names) to skip hooking (e.g. ``nn.Dropout``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sync: bool | None = None,
        skip: tuple = (torch.nn.Dropout, torch.nn.Identity),
    ) -> None:
        self.model = model
        self.sync = is_mps_available() if sync is None else sync
        self.skip = skip
        has_half = any(p.dtype == torch.float16 for p in model.parameters())
        self._module_sync = self.sync and not has_half
        self._hooks: list = []
        self._starts: dict[int, float] = {}
        self._times: dict[str, list[float]] = defaultdict(list)

    def _should_skip(self, module: torch.nn.Module) -> bool:
        return isinstance(module, self.skip)

    def _attach(self) -> None:
        for module in self.model.modules():
            if self._should_skip(module):
                continue
            pre = module.register_forward_pre_hook(self._pre_hook)
            post = module.register_forward_hook(self._post_hook)
            self._hooks.append(pre)
            self._hooks.append(post)

    def _detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _pre_hook(self, module, args):
        self._starts[id(module)] = time.perf_counter()

    def _post_hook(self, module, args, output):
        start = self._starts.pop(id(module), None)
        if start is None:
            return output
        if self._module_sync:
            mps_synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        self._times[_module_name(self.model, module)].append(elapsed)
        return output

    def run(
        self,
        x: torch.Tensor,
        n_warmup: int = 3,
        n_iters: int = 20,
    ) -> "LayerProfiler":
        self._attach()
        try:
            for _ in range(n_warmup):
                self.model(x)
            mps_synchronize()
            for _ in range(n_iters):
                self.model(x)
            mps_synchronize()
        finally:
            self._detach()
        return self

    def __enter__(self) -> "LayerProfiler":
        self._attach()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._detach()

    def summary(self) -> dict[str, dict]:
        """``{module_name: {best_ms, mean_ms, calls, self_ms}}`` (self excludes children)."""
        raw = {
            name: {"best_ms": min(v), "mean_ms": statistics.mean(v), "calls": len(v)}
            for name, v in self._times.items()
        }
        child_sum: dict[str, float] = defaultdict(float)
        names = list(raw)
        for name in names:
            prefix = name + "."
            for other in names:
                if other.startswith(prefix):
                    child_sum[name] += raw[other]["mean_ms"]
        out: dict[str, dict] = {}
        for name, r in raw.items():
            out[name] = dict(r)
            out[name]["self_ms"] = max(0.0, r["mean_ms"] - child_sum.get(name, 0.0))
        return out

    def table(self, top: int | None = None, sort_by: str = "mean_ms") -> str:
        return layer_summary_table(self.summary(), top=top, sort_by=sort_by)

    def torch_profiler_table(
        self, x: torch.Tensor, n_iters: int = 5, row_limit: int = 25
    ) -> str:
        """Per-op CPU-dispatch table from ``torch.profiler`` (works on MPS)."""
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(n_iters):
                self.model(x)
            mps_synchronize()
        return prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=row_limit)

    def signpost(
        self,
        x: torch.Tensor,
        n_iters: int = 5,
        mode: str = "interval",
        wait_until_completed: bool = False,
    ) -> None:
        """Wrap a run in ``torch.mps.profiler.profile`` (Xcode Instruments signposts)."""
        import torch.mps.profiler as mpsp

        with mpsp.profile(
            mode=mode, wait_until_completed=wait_until_completed
        ):
            for _ in range(n_iters):
                self.model(x)
            mps_synchronize()

    def metal_capture(self, x: torch.Tensor, path: str = "fff.gputrace") -> bool:
        """Capture a Metal GPU trace (requires ``MTL_CAPTURE_ENABLED=1``)."""
        import torch.mps.profiler as mpsp

        if not mpsp.is_metal_capture_enabled():
            return False
        with mpsp.metal_capture(path):
            self.model(x)
        return True


def layer_summary_table(
    summary: dict[str, dict], top: int | None = None, sort_by: str = "mean_ms"
) -> str:
    rows = sorted(summary.items(), key=lambda kv: kv[1][sort_by], reverse=True)
    if top:
        rows = rows[:top]
    header = f"{'layer':<42}{'best(ms)':>10}{'mean(ms)':>10}{'self(ms)':>10}{'calls':>7}"
    lines = [header, "-" * len(header)]
    for name, r in rows:
        lines.append(
            f"{name:<42}{r['best_ms']:>10.3f}{r['mean_ms']:>10.3f}"
            f"{r['self_ms']:>10.3f}{r['calls']:>7}"
        )
    return "\n".join(lines)
