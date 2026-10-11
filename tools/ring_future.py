"""`python tools/ring_future.py`: an existing ring's rows given their future-occupancy field from the run's game records."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mantis._engine import HexgBuffer, game_future_stones
from mantis.diagnostics.ring_audit import future_disagreements
from mantis.diagnostics.ring_reader import Ring, load_ring
from mantis.util.hashing import sha256_file
from mantis.util.loadpkg import load_tools_package

load_tools_package("distill")
corpus = importlib.import_module("distill.corpus")

#: Beside every ring written here: its source, its records and what was settled.
PROVENANCE_SUFFIX = ".provenance.json"


def _groups(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Each distinct label with the row indices carrying it, in label order."""
    order = np.argsort(labels, kind="stable")
    values, first = np.unique(labels[order], return_index=True)
    return list(zip(values.tolist(), np.split(order, first[1:]), strict=True))


def settle_games(ring: Ring, games: Any) -> tuple[np.ndarray, dict[str, int]]:
    """Each row's record index (-1 unsettled), a ring game settled whole by the one record all its rows' positions fit."""
    keys, gidx, ks = corpus.game_position_keys(games)
    start = np.concatenate([[0], np.cumsum(games.n_moves + 1)[:-1]]).astype(np.int64)
    order = np.argsort(keys, kind="stable")
    rkeys = corpus.ring_row_keys(ring)
    lo = np.searchsorted(keys[order], rkeys, side="left")
    hi = np.searchsorted(keys[order], rkeys, side="right")
    row_game = np.full(ring.header.size, -1, np.int64)
    why = {"untagged": 0, "no_record": 0, "no_unique_row": 0, "records_disagree": 0}
    for gid, rows in _groups(ring.game_id):
        if gid < 0:
            why["untagged"] += rows.size
            continue
        unique = {int(gidx[order[lo[i]]]) for i in rows if hi[i] - lo[i] == 1}
        if any(hi[i] == lo[i] for i in rows):
            why["no_record"] += rows.size
        elif not unique:
            why["no_unique_row"] += rows.size
        elif len(unique) > 1:
            why["records_disagree"] += rows.size
        else:
            game = unique.pop()
            plies = ring.ply_index[rows]
            fits = (plies <= games.n_moves[game]) & (keys[start[game] + np.minimum(plies, games.n_moves[game])]
                                                       == rkeys[rows])
            if fits.all():
                row_game[rows] = game
            else:
                why["records_disagree"] += rows.size
    return row_game, why


def derive_fields(ring: Ring, games: Any, row_game: np.ndarray, encoding: str) -> list[list[tuple[int, int, int]] | None]:
    """Each settled row's field by the producer's derivation, `None` elsewhere; Raises: ValueError — a record that cannot replay."""
    fields: list[list[tuple[int, int, int]] | None] = [None] * ring.header.size
    for game, rows in _groups(row_game):
        if game < 0:
            continue
        moves = [(int(q), int(r)) for q, r in games.game_moves(int(game))]
        for i, field in zip(rows, game_future_stones(encoding, moves, [int(p) for p in ring.ply_index[rows]]),
                            strict=True):
            fields[i] = [(int(q), int(r), int(p)) for q, r, p in field]
    return fields


def write_ring(ring: Ring, rows: np.ndarray, fields: list[Any], out: Path, *, capacity: int) -> None:
    """Push `rows` of `ring` in order with their future fields into a ring at `out`; Raises: ValueError (a refused row), OSError."""
    buf = HexgBuffer(max(capacity, int(rows.size), 8), ring.header.encoding, ring.header.max_visits)
    for i in rows:
        buf.push_graph_position(
            [(int(s["q"]), int(s["r"]), int(s["p"])) for s in ring.row_stones(i)],
            [(int(v["q"]), int(v["r"]), float(v["prob"])) for v in ring.row_visits(i)],
            int(ring.current_player[i]), int(ring.moves_remaining[i]), int(ring.ply_index[i]),
            bool(ring.is_full_search[i]), float(ring.outcome[i]), bool(ring.value_valid[i]),
            int(ring.game_length[i]), game_id=int(ring.game_id[i]), tail_mass=float(ring.tail_mass[i]),
            root_value=float(ring.root_value[i]), root_value_valid=bool(ring.root_value_valid[i]),
            future_stones=fields[i])
    out.parent.mkdir(parents=True, exist_ok=True)
    buf.save_to_path(str(out))


def convert(args: argparse.Namespace) -> dict[str, Any]:
    """Settle, derive and write (held-out games apart when asked), auditing each ring; Raises: ValueError, OSError."""
    ring = load_ring(args.ring)
    games = corpus.load_games([Path(p) for p in args.games], run_id=args.run_id, step_lo=args.step_lo,
                              step_hi=args.step_hi)
    row_game, why = settle_games(ring, games)
    fields = derive_fields(ring, games, row_game, ring.header.encoding)
    held = np.zeros(ring.header.size, bool)
    if args.heldout_rows:
        held = np.isin(ring.game_id, corpus.heldout_games(ring.game_id, rows=args.heldout_rows, seed=args.heldout_seed))
    record: dict[str, Any] = {
        "source": str(args.ring), "source_sha256": sha256_file(args.ring), "run_id": args.run_id,
        "steps": [args.step_lo, args.step_hi], "records": len(games.uid), "rows": int(ring.header.size),
        "settled_rows": int((row_game >= 0).sum()), "settled_games": int(np.unique(row_game[row_game >= 0]).size),
        "unsettled_rows": why, "rings": {},
    }
    outs = [(args.out, np.nonzero(~held)[0], ring.header.capacity)]
    if args.heldout_rows:
        outs.append((args.heldout_out, np.nonzero(held)[0], 0))
    for path, rows, capacity in outs:
        write_ring(ring, rows, fields, path, capacity=capacity)
        written = load_ring(path)
        checked, bad = future_disagreements(written)
        record["rings"][str(path)] = {
            "sha256": sha256_file(path), "rows": int(rows.size), "field_rows": int(written.future_valid.sum()),
            "games": int(np.unique(written.game_id).size), "pairs_read": checked, "pairs_contradicting": len(bad),
        }
        path.with_name(path.name + PROVENANCE_SUFFIX).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    """CLI; exit 1 when a written ring's fields contradict each other, 2 on a usage error."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ring", type=Path, required=True, help="the source ring (any HEXG version)")
    ap.add_argument("--games", nargs="+", required=True, help="the run's self-play game record shards (.jsonl)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--step-lo", type=int, required=True, help="the records' actor steps, inclusive")
    ap.add_argument("--step-hi", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout-rows", type=int, default=0, help="whole ring games set apart until they reach this")
    ap.add_argument("--heldout-seed", type=int, default=0)
    ap.add_argument("--heldout-out", type=Path)
    args = ap.parse_args(argv)
    if args.heldout_rows and args.heldout_out is None:
        ap.error("--heldout-rows needs --heldout-out")
    record = convert(args)
    print(json.dumps(record, indent=2))
    return 1 if any(r["pairs_contradicting"] for r in record["rings"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
