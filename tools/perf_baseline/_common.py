"""Shared helpers for the PERF-BASELINE measurement drivers (2026-08-29, diagnostic only)."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch


def throwaway_config(src: str, dst: str, overrides: dict[str, Any]) -> str:
    """Write a THROWAWAY copy of a minted config with dotted-path overrides applied.

    Disclosed as a diagnostic artifact: it never enters `configs/`, and every override is
    echoed into the run record so a reading can be tied to the exact regime that produced it.
    """
    import yaml

    with open(src, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for dotted, value in overrides.items():
        node = raw
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node[key]
        if parts[-1] not in node:
            raise KeyError(f"override path {dotted!r} names a key the config lacks")
        node[parts[-1]] = value
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("# THROWAWAY diagnostic config (PERF-BASELINE 2026-08-29). Not minted, not shipped.\n")
        fh.write(f"# source: {src}\n")
        for dotted, value in overrides.items():
            fh.write(f"# override: {dotted} -> {value!r}\n")
        yaml.safe_dump(raw, fh, sort_keys=False)
    return dst


class GpuSampler(threading.Thread):
    """Sample utilization / memory / SM-clock through NVML at a fixed interval.

    `utilization.gpu` is NVIDIA's *fraction of sampled intervals in which at least one
    kernel was resident* — it is an occupancy-of-time reading, NOT an SM-occupancy one, so
    a 100 % sample is consistent with one tiny kernel per interval. Reported as such.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        super().__init__(daemon=True, name="perf-gpu-sampler")
        self.interval_s = interval_s
        self.samples: list[tuple[float, int, int, int]] = []
        self._stop_evt = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop_evt.is_set():
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.samples.append(
                    (time.monotonic(), int(rates.gpu), int(rates.memory), int(mem.used))
                )
                time.sleep(self.interval_s)
            pynvml.nvmlShutdown()
        except BaseException as exc:  # noqa: BLE001 — recorded, never silently lost
            self.error = exc

    def stop(self) -> None:
        self._stop_evt.set()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"n_samples": 0, "error": repr(self.error) if self.error else None}
        util = sorted(s[1] for s in self.samples)
        n = len(util)
        return {
            "n_samples": n,
            "interval_s": self.interval_s,
            "gpu_util_mean_pct": sum(util) / n,
            "gpu_util_median_pct": util[n // 2],
            "gpu_util_p90_pct": util[min(n - 1, int(n * 0.9))],
            "gpu_util_max_pct": util[-1],
            "idle_fraction_util0": sum(1 for u in util if u == 0) / n,
            "fraction_util_lt_25": sum(1 for u in util if u < 25) / n,
            "mem_used_max_bytes": max(s[3] for s in self.samples),
            "error": repr(self.error) if self.error else None,
            "note": (
                "nvml utilization.gpu is the fraction of sampled intervals with >=1 kernel "
                "resident, not SM occupancy"
            ),
        }


def cuda_ms(fn, n: int, warmup: int) -> list[float]:
    """Time `fn` `n` times with CUDA events after `warmup` untimed calls. Returns ms/call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out: list[float] = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def stats(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "p10": s[int(n * 0.10)],
        "p90": s[min(n - 1, int(n * 0.90))],
        "min": s[0],
        "max": s[-1],
    }


def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"wrote {path}")


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
