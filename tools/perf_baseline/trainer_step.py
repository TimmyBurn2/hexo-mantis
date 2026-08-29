"""PERF-BASELINE §2 D — one trainer step at run5 shape, decomposed.

Fills a real replay ring from a real self-play drive, then steps the PRODUCTION dispatch
(`mantis.train.coordinator.dispatch.train_one_batch`) against it. Stage attribution comes
from timing wrappers installed on the collaborators that dispatch imports AT CALL TIME —
the production code path runs unmodified; only the collaborators are observed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import git_sha, stats, write_json  # noqa: E402

from mantis.config.loader import load_config  # noqa: E402
from mantis.config.resolve.coordinator import resolve_coordinator_knobs  # noqa: E402
from mantis.config.resolve.pool_encoding import resolve_pool_encoding  # noqa: E402
from mantis.diagnostics.worker_sweep import build_sweep_pool  # noqa: E402
from mantis.model import arch_from_spec_and_config  # noqa: E402
from mantis.train.coordinator.dispatch import train_one_batch  # noqa: E402
from mantis.train.trainer.core import Trainer  # noqa: E402

TIMES: dict[str, list[float]] = defaultdict(list)
ACTIVE = {"on": False}


def timed(name: str, fn: Any) -> Any:
    def wrapper(*a: Any, **kw: Any) -> Any:
        if not ACTIVE["on"]:
            return fn(*a, **kw)
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        TIMES[name].append((time.perf_counter() - t0) * 1e3)
        return out
    return wrapper


def install_wrappers(model: Any) -> None:
    from mantis.selfplay import graph_collate, graph_wire_split

    graph_collate.graph_wire_from_rust = timed(
        "wire_copyout", graph_collate.graph_wire_from_rust)
    graph_collate.collate_graph_batch = timed(
        "collate_h2d", graph_collate.collate_graph_batch)
    graph_wire_split.plan_microbatches = timed(
        "plan_microbatches", graph_wire_split.plan_microbatches)
    graph_wire_split.slice_graph_wire = timed(
        "slice_wire", graph_wire_split.slice_graph_wire)
    graph_wire_split.slice_targets = timed("slice_targets", graph_wire_split.slice_targets)
    model.forward_batch = timed("forward", model.forward_batch)
    torch.Tensor.backward = timed("backward", torch.Tensor.backward)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--fill-sec", type=float, default=180.0)
    ap.add_argument("--fill-workers", type=int, default=12)
    ap.add_argument("--warm-steps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default="/workspace/perfbase/ckpt_throwaway")
    args = ap.parse_args()

    config = load_config(args.config)
    raw = config.model_dump()
    device = torch.device("cuda")
    resolved = resolve_pool_encoding(raw, arch=None)
    spec = resolved.registry_spec
    arch = arch_from_spec_and_config(spec, raw)
    knobs = resolve_coordinator_knobs(config.train)

    pool = build_sweep_pool(config, n_workers=args.fill_workers, device=device)
    buffer = pool.replay_buffer.buffer if hasattr(pool.replay_buffer, "buffer") else None
    if buffer is None:
        buffer = getattr(pool.replay_buffer, "_buffer", None)
    if buffer is None or not hasattr(buffer, "sample_graph_batch"):
        print(f"REFUSING: no sample_graph_batch on {type(pool.replay_buffer).__name__}; "
              f"attrs={[a for a in dir(pool.replay_buffer) if not a.startswith('__')]}")
        return 2

    record: dict[str, Any] = {
        "sha": git_sha(),
        "regime": {
            "batch_size": knobs.batch_size, "augment": knobs.augment,
            "recency_weight": knobs.recency_weight,
            "microbatch_caps": {"max_edges": config.train.microbatch_caps.max_edges,
                                "max_nodes": config.train.microbatch_caps.max_nodes},
            "amp_dtype": raw["train"]["amp_dtype"], "fill_workers": args.fill_workers,
            "fill_sec": args.fill_sec,
        },
    }
    try:
        pool.start()
        time.sleep(args.fill_sec)
        pool.check_producer_health()
        record["ring_len_after_fill"] = len(buffer)
    finally:
        pool.stop()
    print(f"ring holds {record['ring_len_after_fill']} samples after fill")

    trainer = Trainer(pool.model, raw, arch=arch, checkpoint_dir=args.ckpt_dir, device=device)
    install_wrappers(trainer.model)

    def caps_provider() -> Any:
        return config.train.microbatch_caps

    def one_step() -> dict[str, float]:
        return train_one_batch(
            trainer, buffer, spec, batch_size=knobs.batch_size, augment=knobs.augment,
            recency_weight=knobs.recency_weight, recent_buffer=None,
            caps_provider=caps_provider)

    for _ in range(args.warm_steps):
        one_step()
    torch.cuda.synchronize()

    ACTIVE["on"] = True
    TIMES.clear()
    step_ms: list[float] = []
    t_window = time.perf_counter()
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)
    window_wall = time.perf_counter() - t_window
    ACTIVE["on"] = False

    record["step_ms"] = stats(step_ms)
    record["stages_ms_per_step"] = {
        name: {"calls_per_step": len(v) / args.steps,
               "total_ms_per_step": sum(v) / args.steps,
               "mean_ms_per_call": sum(v) / len(v)}
        for name, v in sorted(TIMES.items())
    }
    attributed = sum(sum(v) for n, v in TIMES.items() if n != "slice_targets") / args.steps
    record["attributed_ms_per_step"] = attributed
    record["unattributed_ms_per_step"] = record["step_ms"]["mean"] - attributed

    # ── profiler pass: kernels + the GPU-idle fraction of the step ───────────────────
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            one_step()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    device_us = sum(getattr(e, "self_device_time_total", 0.0) for e in ev)
    record["profiler"] = {
        "iterations": 10,
        "self_device_us_total": device_us,
        "self_device_ms_per_step": device_us / 1e3 / 10,
        "table": ev.table(sort_by="self_device_time_total", row_limit=20,
                          max_name_column_width=70),
        "rows": [
            {"name": e.key, "count": e.count,
             "self_device_us": getattr(e, "self_device_time_total", 0.0),
             "self_cpu_us": e.self_cpu_time_total}
            for e in sorted(ev, key=lambda x: -getattr(x, "self_device_time_total", 0.0))[:25]
        ],
    }
    record["gpu_busy_fraction_of_step"] = (
        (device_us / 1e3 / 10) / record["step_ms"]["mean"] if record["step_ms"]["mean"] else None
    )
    record["window_wall_sec"] = window_wall
    write_json(args.out, record)
    print(f"step {record['step_ms']['mean']:.2f} ms  "
          f"gpu_busy_frac={record['gpu_busy_fraction_of_step']}")
    print(record["profiler"]["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
