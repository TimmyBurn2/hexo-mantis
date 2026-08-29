"""PERF-BASELINE §2 B/C/F — one self-play window, decomposed stage by stage.

DIAGNOSTIC ONLY (2026-08-29). Builds its own collaborators through
`mantis.diagnostics.worker_sweep`'s helpers, so no trainer exists and no production config
is touched. Every figure is stamped with the regime that produced it.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GpuSampler, git_sha, throwaway_config, write_json  # noqa: E402

from mantis._engine import selfplay_perf_reset, selfplay_perf_snapshot  # noqa: E402
from mantis.config.loader import load_config  # noqa: E402
from mantis.diagnostics.worker_sweep import build_sweep_pool  # noqa: E402
from mantis.selfplay.pool_hooks import runner_stats  # noqa: E402


def install_capture(out_dir: str, limit: int) -> list[int]:
    """Wrap `graph_wire_from_rust` so the first `limit` real wire payloads land on disk.

    Patched at MODULE scope before `pool.start()`: `_run_graph_loop` imports the name inside
    the thread body, so the loop picks up the wrapper. Production graphs, production radius,
    real halo sizes — §2 A refuses synthetic input.
    """
    from mantis.selfplay import graph_collate

    original = graph_collate.graph_wire_from_rust
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = [0]

    def wrapped(wire: Any) -> Any:
        payload = original(wire)
        if saved[0] < limit:
            idx = saved[0]
            saved[0] += 1
            with open(f"{out_dir}/wire_{idx:04d}.pkl", "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return payload

    graph_collate.graph_wire_from_rust = wrapped
    return saved


def diff_stage_block(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Difference two cumulative Python stage snapshots over the measured window.

    `max_ms` is run-cumulative and does NOT difference; it is carried through as the
    run's extreme and labelled so.
    """
    out: dict[str, Any] = {}
    for name, row in after["stages"].items():
        prev = before["stages"].get(name, {"count": 0, "total_ms": 0.0})
        count = row["count"] - prev["count"]
        total = row["total_ms"] - prev["total_ms"]
        out[name] = {
            "count": count,
            "total_ms": total,
            "mean_ms": (total / count) if count else None,
            "run_max_ms": row["max_ms"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--warmup-sec", type=float, default=20.0)
    ap.add_argument("--window-sec", type=float, default=60.0)
    ap.add_argument("--sync-cuda", action="store_true",
                    help="attribute GPU wait to `forward` instead of letting D2H absorb it; "
                         "serialises the stream, so throughput from a --sync-cuda run is NOT "
                         "the serving throughput")
    ap.add_argument("--capture-dir", default=None)
    ap.add_argument("--capture-limit", type=int, default=400)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/perf_baseline")
    args = ap.parse_args()

    if os.environ.get("MANTIS_PERF_STAGES") != "1":
        print("REFUSING: MANTIS_PERF_STAGES=1 must be set so the Rust timers are armed.")
        return 2

    overrides = {
        "selfplay.n_workers": args.n_workers,
        "inference.perf_timing": True,
        "inference.perf_sync_cuda": bool(args.sync_cuda),
    }
    cfg_path = throwaway_config(
        args.config, f"{args.scratch}/cfg_{args.label}.yaml", overrides)
    config = load_config(cfg_path)
    device = torch.device("cuda")

    saved_counter = None
    if args.capture_dir:
        saved_counter = install_capture(args.capture_dir, args.capture_limit)

    pool = build_sweep_pool(config, n_workers=args.n_workers, device=device)
    server = pool._inference_server  # noqa: SLF001 — diagnostic read of the one server
    sampler = GpuSampler()
    record: dict[str, Any] = {
        "label": args.label,
        "sha": git_sha(),
        "config_source": args.config,
        "overrides": overrides,
        "regime": {
            "n_workers": args.n_workers,
            "leaf_batch_size": config.selfplay.leaf_batch_size,
            "n_simulations": config.selfplay.mcts.n_simulations,
            "inference_batch_size": config.inference.inference_batch_size,
            "inference_max_wait_ms": config.inference.inference_max_wait_ms,
            "max_game_moves": config.selfplay.max_game_moves,
            "encoding": config.identity.encoding,
            "sync_cuda": bool(args.sync_cuda),
        },
        "warmup_sec": args.warmup_sec,
        "window_sec": args.window_sec,
    }

    try:
        pool.start()
        time.sleep(args.warmup_sec)
        pool.check_producer_health()

        selfplay_perf_reset()
        py_before = server.perf_stage_snapshot()
        bt_before = server.batch_timing_snapshot()
        rs_before = runner_stats(pool)
        sampler.start()
        t0 = time.monotonic()
        time.sleep(args.window_sec)
        wall = time.monotonic() - t0
        sampler.stop()
        rust = selfplay_perf_snapshot()
        py_after = server.perf_stage_snapshot()
        bt_after = server.batch_timing_snapshot()
        rs_after = runner_stats(pool)
        pool.check_producer_health()
        sampler.join(timeout=5.0)

        sims = int(rust["leaves"])
        moves = rs_after.positions_generated - rs_before.positions_generated
        record.update({
            "wall_sec": wall,
            "rust_stages": rust,
            "python_stages": diff_stage_block(py_before, py_after),
            "python_forward_count_delta":
                py_after["forward_count"] - py_before["forward_count"],
            "python_total_requests_delta":
                py_after["total_requests"] - py_before["total_requests"],
            "batch_timing_before": bt_before,
            "batch_timing_after": bt_after,
            "runner": {
                "games": rs_after.games_completed - rs_before.games_completed,
                "moves": moves,
            },
            "derived": {
                "sims_total": sims,
                "sims_per_sec_card": sims / wall if wall else None,
                "ms_per_sim_card": (wall * 1e3 / sims) if sims else None,
                "sims_per_move": (sims / moves) if moves else None,
                "moves_per_min": moves * 60.0 / wall if wall else None,
            },
            "gpu": sampler.summary(),
            "captured_wires": saved_counter[0] if saved_counter else 0,
        })
    finally:
        try:
            pool.stop()
        except Exception as exc:  # noqa: BLE001 — recorded on the record, never swallowed
            record["teardown_note"] = repr(exc)

    write_json(args.out, record)
    d = record.get("derived", {})
    print(f"[{args.label}] sims={d.get('sims_total')} "
          f"ms/sim(card)={d.get('ms_per_sim_card'):.4f} "
          f"moves/min={d.get('moves_per_min'):.1f}" if d.get("sims_total") else "[no sims]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
