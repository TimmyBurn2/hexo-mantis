"""Synthetic run records for the dash tests: event segments, game shards, cell sidecars, monitor saves, config and heartbeat."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SIX = {"commit": "c0ffee", "net_sha256": "beef", "generation": 30, "nodes": 16}
STRIX = {"commit": "5a77", "checkpoint_sha256": "351e", "sims": 256, "solver": "on"}
TACTICS = {"arm": "full", "block": {"kind": "strict_turn", "root_nodes": 20000}}


def trainer_rows(steps: range, *, entropy: bool = True) -> list[dict[str, Any]]:
    """`trainer_step` rows with smooth, distinct losses so a mixed-up series shows."""
    out = []
    for s in steps:
        row = {"event": "trainer_step", "step": s, "value_loss": 0.6 - s * 1e-5, "policy_loss": 2.4 - s * 2e-5,
               "loss": 3.0, "grad_norm": 4.0 + math.sin(s), "lr": 1e-3, "ts": 1000.0 + s}
        if entropy:
            row["policy_entropy"] = 2.2 + 0.01 * math.cos(s)
        out.append(row)
    return out


def iteration_rows(steps: range) -> list[dict[str, Any]]:
    return [{"event": "iteration_complete", "step": s, "games_per_hour": 800.0 + s % 7, "positions_per_hour": 3.0e4,
             "steps_per_hour": 700.0, "sims_per_sec": 6000.0, "avg_game_length": 40.0 + s % 3, "ts": 1000.5 + s}
            for s in steps]


def game_rows(n: int, *, first_wins_every: int = 2, cap_every: int = 0, ts: float = 1000.0) -> list[dict[str, Any]]:
    """`game_complete` rows: the first mover wins every `first_wins_every`-th game, every `cap_every`-th hits the cap."""
    out = []
    for i in range(n):
        cap = cap_every > 0 and i % cap_every == 0
        out.append({"event": "game_complete", "game_id": f"g{i}", "game_id_byte_hash": f"h{i}",
                    "winner": -1 if cap else 0 if i % first_wins_every == 0 else 1, "moves": 30 + i % 11,
                    "moves_list": "x" * 8, "terminal_reason": "ply_cap" if cap else "six_in_a_row", "ts": ts + i})
    return out


def segment_start(run_id: str, segment: int, pid: int = 4242, ts: float = 999.0) -> dict[str, Any]:
    return {"event": "run_segment_started", "run_id": run_id, "segment": segment, "pid": pid,
            "created_utc": "2026-10-05T10:00:00+00:00", "ts": ts}


def write_segment(logs: Path, run_id: str, number: int, rows: list[dict[str, Any]]) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"events_{run_id}_seg{number:04d}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def game(game_id: str, moves: list[list[int]], *, channel: str = "selfplay", result: str = "p1",
         termination: str = "six_in_a_row", stats: list[dict[str, Any]] | None = None, step: int = 7,
         **over: Any) -> dict[str, Any]:
    """One game record in contract #11's shape; `stats` is written only when given (un-sampled self-play has no key)."""
    row: dict[str, Any] = {"contract": "game-record-v1", "game_id": game_id, "run_id": "t", "channel": channel,
                           "step": step, "step_kind": "actor" if channel == "selfplay" else "round", "seed": 1,
                           "served_sims": 64, "plies": len(moves), "result": result, "termination": termination,
                           "moves": moves}
    if stats is not None:
        row["search_stats"] = stats
    row.update(over)
    return row


def write_shard(games_dir: Path, run_id: str, seg: int, hour: str, games: list[dict[str, Any]],
                *, closed: bool = True) -> Path:
    """A shard with its open marker; `closed` appends its `shard_closed` row to the run's index with the exact size."""
    games_dir.mkdir(parents=True, exist_ok=True)
    path = games_dir / f"games_{run_id}_seg{seg:04d}_{hour}.jsonl"
    rows = [{"record": "shard_opened", "run_id": run_id, "segment": seg, "hour": hour}] + games
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if closed:
        with (games_dir / f"games_{run_id}_index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"record": "shard_closed", "shard": path.name, "bytes": path.stat().st_size,
                                 "games": len(games)}) + "\n")
    return path


def six_in_a_row_for_p1() -> list[list[int]]:
    """p1 lays (0,0)…(5,0) along the E axis; p2 answers far away. p1's plies are 0, 3, 4, 7, 8, 11."""
    p1 = [[k, 0] for k in range(6)]
    p2 = [[2 * k, 5 + k % 2] for k in range(6)]
    return [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4], p2[4], p2[5], p1[5]]


def sidecar(directory: Path, run_id: str, step: int, wr: float, *, family: str = "six", n: int = 576,
            suffix: str | None = None, **over: Any) -> Path:
    """A follower sidecar beside a checkpoint name, in the six or strix shape; `over` replaces top-level fields."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{step:08d}_abcd1234"
    raw: dict[str, Any] = {"schema_version": 2, "run_id": run_id, "checkpoint": f"{stem}.ckpt", "step": step,
                           "unit": "six30_16" if family == "six" else "equal_work",
                           "ours": {"search_kind": "puct", "sims": 256}, "tactics": TACTICS, "regime": "IDLE",
                           "games": n, "eff_n": n, "pairs": n // 2, "wr": wr, "wr_ci_lower": wr - 0.04,
                           "wr_ci_upper": wr + 0.04}
    raw[family] = dict(SIX if family == "six" else STRIX)
    raw.update(over)
    path = directory / f"{stem}.ckpt.{suffix or ('six30_16.full' if family == 'six' else 'strix256')}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def write_config(run_dir: Path, run_id: str, parent: str | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    warm = f"  warm_start:\n    checkpoint: checkpoints/p/{parent}.ckpt\n" if parent else ""
    (run_dir / "resolved_config.yaml").write_text(
        f"schema_version: 1\nrun_id: {run_id}\nidentity:\n  encoding: gnn_axis_r8\n{warm}", encoding="utf-8")


def write_heartbeat(logs: Path, run_id: str, wall_ts: float) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"heartbeat_{run_id}.json"
    path.write_text(json.dumps({"seq": 1, "pid": 4242, "wall_ts": wall_ts}), encoding="utf-8")
    return path
