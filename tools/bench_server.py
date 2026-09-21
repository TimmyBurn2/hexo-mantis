"""F-47's server bench (R360(b) step 2): the REAL batcher and `InferenceServer` per batch size, standalone, fed by threads replaying a run's own games."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from mantis._engine import Board, InferenceBatcher
from mantis.config.loader import load_config
from mantis.config.resolve.compile_trunk import resolve_compile_trunk
from mantis.config.resolve.edge_geometry_check import resolve_edge_geometry_check
from mantis.encoding import lookup
from mantis.model import build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.inference_server import InferenceServer
from mantis.train.checkpoints import load_checkpoint
from mantis.util.git import head_sha, is_dirty

Position = tuple[list[tuple[int, int, int]], int, int]
#: R360(b) step 3's curve: ms/batch for these batch sizes.
DEFAULT_BATCH_SIZES = (16, 32, 64, 128, 256)
_COLUMNS = ("batch_size", "pops_per_s", "b_mean", "full_share", "sat_share", "deadline_share",
            "leaves_per_s", "cycle_ms", "queue_wait_ms", "collate_ms", "launch_ms", "gpu_wait_ms",
            "cpu_ms_per_leaf", "gpu_ms_per_leaf", "edges_per_graph", "gpu_duty_estimate")


def positions_from_events(path: Path, encoding: str, *, limit: int | None) -> list[Position]:
    """Every position a search ran FROM in the stream's `game_complete` games — the LAST `limit` of them (the newest games' shape)."""
    out: list[Position] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "game_complete":
                continue
            board = Board.with_encoding_name(encoding)
            for move in row["moves_list"]:
                out.append((list(board.get_stones()), int(board.current_player),
                            int(board.moves_remaining)))
                q, r = (int(x) for x in move.strip("()").split(","))
                board.apply_move(q, r)
    return out if limit is None else out[-limit:]


def _delta(after: dict[str, Any], before: dict[str, Any], *keys: str, field: str = "total_ms") -> float:
    a, b = after, before
    for key in keys:
        a, b = a[key], b[key]
    return float(a[field]) - float(b[field])


def summarize(before: dict[str, Any], after: dict[str, Any], *, wall_s: float) -> dict[str, Any]:
    """PERF3_STEP1's columns from two `batch_timing_snapshot`s: differences of cumulative totals."""
    pops = int(after["occupancy"]["count"]) - int(before["occupancy"]["count"])
    leaves = int(after["occupancy"]["total"]) - int(before["occupancy"]["total"])
    if pops <= 0 or wall_s <= 0:
        raise ValueError(f"no pops in the window (pops={pops}, wall={wall_s:.3f} s)")
    ha, hb = after["occupancy"]["histogram"], before["occupancy"]["histogram"]
    hist = {int(k): int(ha.get(k, 0)) - int(hb.get(k, 0)) for k in set(ha) | set(hb)}
    batch = int(after["batch_size"])
    full = sum(v for k, v in hist.items() if k >= batch)
    sat = sum(v for k, v in hist.items() if batch // 2 <= k < batch)
    edges = _delta(after, before, "fusion", "fused_batch_edges", field="total")
    per_pop = {name: _delta(after, before, *keys) / pops for name, keys in (
        ("queue_wait_ms", ("queue_wait",)), ("collate_ms", ("collate",)),
        ("launch_ms", ("pipeline", "launch")), ("gpu_wait_ms", ("pipeline", "gpu_wait")))}
    cycle = wall_s * 1e3 / pops
    return {
        "pops": pops, "leaves": leaves, "wall_s": wall_s, "pops_per_s": pops / wall_s,
        "b_mean": leaves / pops, "leaves_per_s": leaves / wall_s, "cycle_ms": cycle,
        "full_share": full / pops, "sat_share": sat / pops,
        "deadline_share": (pops - full - sat) / pops, "histogram": dict(sorted(hist.items())),
        **per_pop, "cpu_ms_per_leaf": per_pop["launch_ms"] * pops / leaves,
        "gpu_ms_per_leaf": per_pop["gpu_wait_ms"] * pops / leaves,
        "edges_per_graph": edges / leaves, "gpu_duty_estimate": per_pop["gpu_wait_ms"] / cycle,
        "empty_polls": int(after["empty_polls"]) - int(before["empty_polls"]),
    }


def run_cell(model: torch.nn.Module, device: torch.device, config: dict[str, Any],
             positions: list[Position], *, batch_size: int, workers: int, leaf_batch: int,
             seconds: float, compile_trunk: bool, warmup_s: float = 0.0,
             probe: int = 4) -> dict[str, Any]:
    """One standalone cell: the server at `batch_size` fed by `workers` threads, `leaf_batch` leaves each."""
    cfg = copy.deepcopy(config)
    cfg["inference"]["inference_batch_size"] = int(batch_size)
    spec = lookup(cfg["identity"]["encoding"])
    batcher = InferenceBatcher(encoding_spec=spec)
    server = InferenceServer(model, device, cfg, batcher=batcher, encoding_spec=spec,
                             collate_check_period=1,
                             edge_geometry_check=resolve_edge_geometry_check(cfg),
                             compile_trunk=compile_trunk)
    server.start()
    stop = threading.Event()
    submitted = [0] * workers
    served = [0] * workers

    def worker(idx: int) -> None:
        cursor = (idx * leaf_batch) % len(positions)
        while not stop.is_set():
            chunk = [positions[(cursor + i) % len(positions)] for i in range(leaf_batch)]
            cursor = (cursor + leaf_batch) % len(positions)
            submitted[idx] += len(chunk)
            served[idx] += len(batcher.submit_graphs_and_wait(chunk, 1))

    try:
        # The probe: the same positions served alone, so a later B can be read against them.
        probe_values = [float(v) for _d, _o, v in
                        batcher.submit_graphs_and_wait(positions[:probe], 1)]
        threads = [threading.Thread(target=worker, args=(i,), daemon=True, name=f"bench-w{i}")
                   for i in range(workers)]
        for t in threads:
            t.start()
        if warmup_s > 0:
            time.sleep(warmup_s)
        # The window is `seconds` from the first snapshot; the probe and warm-up sit outside it.
        before = server.batch_timing_snapshot()
        t_before = time.perf_counter()
        time.sleep(seconds)
        after = server.batch_timing_snapshot()
        wall = time.perf_counter() - t_before
        stop.set()
        for t in threads:
            t.join()
    finally:
        server.stop()
        server.join(timeout=10.0)
    row = summarize(before, after, wall_s=wall)
    row.update({"batch_size": int(batch_size), "workers": workers, "leaf_batch": leaf_batch,
                "warmup_s": warmup_s, "device": device.type, "compile_trunk": compile_trunk,
                "edge_geometry_check": resolve_edge_geometry_check(cfg),
                "max_wait_ms": int(cfg["inference"]["inference_max_wait_ms"]),
                "submitted": sum(submitted), "served": sum(served), "probe_values": probe_values})
    return row


def _load_net(config: dict[str, Any], checkpoint: Path | None) -> tuple[torch.nn.Module, str]:
    if checkpoint is None:
        from mantis.model import arch_from_spec_and_config
        net = build_net(arch_from_spec_and_config(lookup(config["identity"]["encoding"]), config))
        return net, "random-init (no --checkpoint)"
    ck = load_checkpoint(checkpoint)
    if ck.metadata.arch is None:
        raise ValueError(f"{checkpoint}: the stamp resolves no arch, so the net cannot be rebuilt")
    net = build_net(ck.metadata.arch)
    net.load_state_dict(ck.model_state)  # the LEARNER's weights: the bench prices a shape, not a deploy net
    return net, net_param_hash(net)


def _table(rows: list[dict[str, Any]], reference: int) -> str:
    ref = next((r for r in rows if r["batch_size"] == reference), rows[0])
    lines = ["  ".join(f"{c:>16s}" for c in _COLUMNS)]
    for r in rows:
        lines.append("  ".join(f"{r[c]:16.3f}" if isinstance(r[c], float) else f"{r[c]:16d}"
                               for c in _COLUMNS))
    lines.append(f"ratios against B={ref['batch_size']} (collate and launch per GRAPH, the device stage per graph, leaves/s):")
    for r in rows:
        # No device stage on a CPU cell: the retirer's wait is microseconds and the ratio noise.
        gpu = (f"{r['gpu_ms_per_leaf'] / ref['gpu_ms_per_leaf']:.3f}" if r["device"] == "cuda"
               else "n/a (cpu)")
        lines.append(f"  B={r['batch_size']:4d}  collate/graph {r['collate_ms'] / r['b_mean'] / (ref['collate_ms'] / ref['b_mean']):.3f}"
                     f"  launch/graph {r['cpu_ms_per_leaf'] / ref['cpu_ms_per_leaf']:.3f}"
                     f"  gpu/graph {gpu}"
                     f"  leaves/s {r['leaves_per_s'] / ref['leaves_per_s']:.3f}")
    probe = [r["probe_values"] for r in rows]
    spread = max((abs(a - b) for row in probe[1:] for a, b in zip(probe[0], row, strict=True)), default=0.0)
    lines.append(f"probe: max |Δvalue| across batch sizes on the same {len(probe[0])} positions = {spread:.6f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """`python tools/bench_server.py --config C --events E [--checkpoint K] [--device cuda] --out R.json`."""
    parser = argparse.ArgumentParser(prog="python tools/bench_server.py", description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="the run's minted config")
    parser.add_argument("--events", required=True, type=Path,
                        help="an events jsonl whose game_complete rows supply the positions")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="the net (a v2 checkpoint); absent = random init, shapes only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES))
    parser.add_argument("--workers", type=int, default=None, help="default: selfplay.n_workers")
    parser.add_argument("--leaf-batch", type=int, default=None, help="default: selfplay.leaf_batch_size")
    parser.add_argument("--seconds", type=float, default=60.0, help="measured window per batch size")
    parser.add_argument("--warmup", type=float, default=10.0, help="seconds before the window opens")
    parser.add_argument("--limit-positions", type=int, default=20000)
    parser.add_argument("--no-compile", action="store_true",
                        help="eager trunk regardless of inference.compile_trunk")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config).model_dump()
    device = torch.device(args.device)
    net, net_id = _load_net(config, args.checkpoint)
    net.to(device)
    positions = positions_from_events(args.events, config["identity"]["encoding"],
                                      limit=args.limit_positions)
    workers = args.workers or int(config["selfplay"]["n_workers"])
    leaf_batch = args.leaf_batch or int(config["selfplay"]["leaf_batch_size"])
    compile_trunk = False if args.no_compile else resolve_compile_trunk(config)
    rows = []
    for batch in (int(b) for b in args.batch_sizes.split(",")):
        row = run_cell(net, device, config, positions, batch_size=batch, workers=workers,
                       leaf_batch=leaf_batch, seconds=args.seconds, compile_trunk=compile_trunk,
                       warmup_s=args.warmup)
        rows.append(row)
        print(f"B={batch}: {row['leaves_per_s']:.0f} leaves/s at B {row['b_mean']:.1f}, cycle "
              f"{row['cycle_ms']:.2f} ms (launch {row['launch_ms']:.2f}, gpu_wait "
              f"{row['gpu_wait_ms']:.2f}), served {row['served']} == submitted {row['submitted']}",
              flush=True)
    reference = int(config["inference"]["inference_batch_size"])
    print(_table(rows, reference))
    repo = Path(__file__).resolve().parents[1]
    record = {
        "tool": "tools/bench_server.py", "config": str(args.config), "events": str(args.events),
        "net": net_id, "device": args.device, "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "tree_sha": head_sha(repo), "tree_dirty": is_dirty(repo), "positions": len(positions),
        "workers": workers, "leaf_batch": leaf_batch, "seconds": args.seconds,
        "warmup": args.warmup, "reference_batch_size": reference, "rows": rows,
    }
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
