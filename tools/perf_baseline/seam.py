"""PERF-BASELINE §2 C — the Python<->Rust crossing, isolated from all GPU work.

A NULL-MODEL SERVER: the real `InferenceBatcher`, the real Rust queue, the real wire
copy-out and the real result submit — with zeros in place of the forward. What remains is
serialize + queue-cross + deserialize + the Rust-side legal-set assemble, per request, at
production payload sizes.

The Rust-side graph BUILD sits inside this round trip (the batcher builds each leaf's axis
graph from the submitted stone list, exactly as self-play does), so the small-board arm is
reported beside the production arm: the difference is the payload-size-dependent half and
the small-board arm is the fixed seam floor.
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import git_sha, stats, write_json  # noqa: E402

from mantis._engine import Board, InferenceBatcher  # noqa: E402
from mantis.config.loader import load_config  # noqa: E402
from mantis.selfplay.graph_collate import graph_wire_from_rust  # noqa: E402


def make_position(spec: Any, plies: int, rng: random.Random) -> tuple[list, int, int]:
    board = Board.with_encoding_name(spec.name)
    for _ in range(plies):
        moves = board.legal_moves()
        if not moves:
            break
        q, r = moves[rng.randrange(len(moves))]
        board.apply_move(q, r)
        if board.check_win():
            board = Board.with_encoding_name(spec.name)
    return (list(board.get_stones()), int(board.current_player), int(board.moves_remaining))


class EchoServer(threading.Thread):
    """The null model: pop a batch, read the wire, submit zeros. No torch, no device."""

    def __init__(self, batcher: InferenceBatcher, batch_size: int, max_wait_ms: int) -> None:
        super().__init__(daemon=True, name="perf-echo-server")
        self.batcher = batcher
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.stop_evt = threading.Event()
        self.copyout_ms: list[float] = []
        self.submit_ms: list[float] = []
        self.occupancy: list[int] = []
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self.stop_evt.is_set():
                ids, wire = self.batcher.next_graph_batch(self.batch_size, self.max_wait_ms)
                if not ids:
                    continue
                t0 = time.perf_counter()
                payload = graph_wire_from_rust(wire)
                t1 = time.perf_counter()
                legal_offsets = np.ascontiguousarray(
                    np.asarray(payload.legal_offsets), dtype=np.int64)
                n_legal = int(legal_offsets[-1])
                probs = np.zeros(n_legal, dtype=np.float32)
                values = np.zeros(payload.n_graphs, dtype=np.float32)
                t2 = time.perf_counter()
                self.batcher.submit_graph_inference_results(
                    ids, probs, legal_offsets, values)
                t3 = time.perf_counter()
                self.copyout_ms.append((t1 - t0) * 1e3)
                self.submit_ms.append((t3 - t2) * 1e3)
                self.occupancy.append(len(ids))
                _ = t2
        except BaseException as exc:  # noqa: BLE001 — recorded, never lost
            self.error = exc


def run_arm(spec: Any, *, plies: int, leaf_batch: int, n_submitters: int,
            batch_size: int, max_wait_ms: int, iters: int, rng: random.Random,
            ) -> dict[str, Any]:
    batcher = InferenceBatcher(encoding_spec=spec)
    server = EchoServer(batcher, batch_size, max_wait_ms)
    server.start()
    positions = [make_position(spec, plies, rng) for _ in range(leaf_batch)]
    n_stones = len(positions[0][0])
    round_trip: list[float] = []
    lock = threading.Lock()

    def submitter() -> None:
        local: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            batcher.submit_graphs_and_wait(positions)
            local.append((time.perf_counter() - t0) * 1e3)
        with lock:
            round_trip.extend(local)

    # Warm the path so the first-call allocations do not enter the sample.
    submitter_warm = [make_position(spec, plies, rng) for _ in range(leaf_batch)]
    batcher.submit_graphs_and_wait(submitter_warm)

    threads = [threading.Thread(target=submitter, daemon=True) for _ in range(n_submitters)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    server.stop_evt.set()
    batcher.close()
    server.join(timeout=5.0)

    total_reqs = iters * n_submitters * leaf_batch
    return {
        "plies": plies,
        "stones_on_board": n_stones,
        "leaf_batch": leaf_batch,
        "n_submitters": n_submitters,
        "iters_per_submitter": iters,
        "wall_sec": wall,
        "round_trip_ms_per_leaf_batch": stats(round_trip),
        "round_trip_ms_per_request": {
            k: (v / leaf_batch if k != "n" else v)
            for k, v in stats(round_trip).items()
        },
        "server_copyout_ms_per_pop": stats(server.copyout_ms) if server.copyout_ms else None,
        "server_submit_ms_per_pop": stats(server.submit_ms) if server.submit_ms else None,
        "server_occupancy": stats([float(o) for o in server.occupancy])
        if server.occupancy else None,
        "throughput_requests_per_sec": total_reqs / wall if wall else None,
        "server_error": repr(server.error) if server.error else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    config = load_config(args.config)
    raw = config.model_dump()
    from mantis.config.resolve.pool_encoding import resolve_pool_encoding

    spec = resolve_pool_encoding(raw, arch=None).registry_spec
    leaf_batch = config.selfplay.leaf_batch_size
    batch_size = config.inference.inference_batch_size
    max_wait_ms = config.inference.inference_max_wait_ms
    rng = random.Random(args.seed)

    record: dict[str, Any] = {
        "sha": git_sha(),
        "regime": {
            "encoding": spec.name, "leaf_batch_size": leaf_batch,
            "inference_batch_size": batch_size, "inference_max_wait_ms": max_wait_ms,
            "note": "null-model server: zeros in place of the forward, no torch, no device",
        },
        "arms": {},
    }
    for label, plies, submitters in (
        ("floor_2ply_w1", 2, 1),
        ("mid_60ply_w1", 60, 1),
        ("prod_128ply_w1", 128, 1),
        ("prod_128ply_w12", 128, 12),
    ):
        print(f"=== seam arm {label} ===")
        arm = run_arm(spec, plies=plies, leaf_batch=leaf_batch, n_submitters=submitters,
                      batch_size=batch_size, max_wait_ms=max_wait_ms,
                      iters=args.iters, rng=rng)
        record["arms"][label] = arm
        rt = arm["round_trip_ms_per_leaf_batch"]
        print(f"  stones={arm['stones_on_board']} median_round_trip="
              f"{rt['median']:.3f} ms/leaf-batch  ({rt['median']/leaf_batch:.4f} ms/request)")
    write_json(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
