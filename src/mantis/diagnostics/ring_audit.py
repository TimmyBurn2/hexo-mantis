# >300 justify (R8): the audit's rows, the two oracles they compose (tactics, the A-2 backup
# probe) and the band verdict are ONE authority — split, "what a miss means" lives in two places.
"""Ring audit (R357(b)): one ring proves its targets before START; exit 1 on a pre-stated band's miss."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mantis._engine import Board, MCTSTree
from mantis.diagnostics import tactics as T
from mantis.diagnostics.ring_reader import Ring, explicit_entropy, load_ring
from mantis.util.constants import is_alpha_full

#: A row with H(explicit) under this is a one-hot (R357(a)).
ONE_HOT_H = 1e-3
#: The census's `mass(F) < 0.1`: the one-hot went to a cell that loses to the completion next turn.
COUNTER_THREAT_MASS = 0.1
_OPS: dict[str, Callable[[float, float], bool]] = {
    "lt": lambda v, b: v < b, "le": lambda v, b: v <= b, "gt": lambda v, b: v > b, "ge": lambda v, b: v >= b,
}
_FENCE = re.compile(r"```toml[^\n]*\n(.*?)```", re.S)
_NAMED_ROWS = 20
ChildQ = Callable[[Board, tuple[int, int]], float | None]


@dataclass(frozen=True)
class Row:
    """One audit line: `value` None is NOT MEASURED and `note` says why; `producer` names the source."""

    key: str
    value: float | None
    n: int | None
    producer: str
    note: str = ""


@dataclass(frozen=True)
class BlockStats:
    """Forced-block rows (tactics oracle, the census's literal B(k)): explicit target mass on B and where it went."""

    n: int
    mass_median: float
    share_ge_half: float
    counter_threat_share: float | None
    counter_threat_rows: list[int]
    #: (row, the one-hot cell, a block cell) for k = 1 rows with mass(B) < 0.5 and the argmax outside B.
    residue_candidates: list[tuple[int, tuple[int, int], tuple[int, int]]]


@dataclass(frozen=True)
class Residue:
    """The A-2 falsifier's oracle over the candidates: counter-threat child >= block child through the backup."""

    n_through: int
    count: int
    rows: list[int]
    n_unreached: int


def _row_arrays(ring: Ring, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stones = ring.row_stones(i)
    return (stones["q"].astype(np.int64), stones["r"].astype(np.int64), stones["p"].astype(np.int64))


def forced_block_stats(ring: Ring) -> BlockStats:
    """Every row through `tactics.analyze`; block rows keep their mass on B and their argmax cell."""
    masses: list[float] = []
    counter: list[int] = []
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for i in range(ring.header.size):
        mover, k = int(ring.current_player[i]), int(ring.moves_remaining[i])
        tac = T.analyze(*_row_arrays(ring, i), mover, k)
        kind, cells = tac.forced(k)
        if kind != "block":
            continue
        visits = ring.row_visits(i)
        # The census's literal B(2): fours sharing a cell make EVERY first stone safe, mass = 1 − α.
        on_block = [tac.block_any or (int(v["q"]), int(v["r"])) in cells for v in visits]
        mass = float(visits["prob"][on_block].sum()) if visits.size else 0.0
        masses.append(mass)
        if mass < COUNTER_THREAT_MASS:
            counter.append(i)
        if visits.size and k == 1 and mass < 0.5:
            j = int(np.argmax(visits["prob"]))
            one_hot = (int(visits["q"][j]), int(visits["r"][j]))
            if one_hot not in cells:
                candidates.append((i, one_hot, sorted(cells)[0]))
    n = len(masses)
    arr = np.asarray(masses, dtype=np.float64)
    return BlockStats(
        n=n,
        mass_median=float(np.median(arr)) if n else float("nan"),
        share_ge_half=float((arr >= 0.5).mean()) if n else 0.0,
        counter_threat_share=len(counter) / n if n else None,
        counter_threat_rows=counter,
        residue_candidates=candidates,
    )


def _hexdist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[0] + a[1] - b[0] - b[1])) // 2


def reconstruct(ring: Ring, i: int) -> Board | None:
    """A `Board` holding row `i`'s stones with (to move, k) as stored, replayed in the legal cadence; None if none does."""
    found = reconstruct_moves(ring, i)
    return None if found is None else found[0]


def reconstruct_moves(ring: Ring, i: int) -> tuple[Board, list[tuple[int, int]]] | None:
    """`reconstruct` with the move list it replayed — the ONE cadence search shared by the board's and the sequence's readers."""
    stones = ring.row_stones(i)
    mover, k = int(ring.current_player[i]), int(ring.moves_remaining[i])
    ids = sorted({int(s["p"]) for s in stones})
    for p1 in ids:
        p2 = next((x for x in ids if x != p1), None)
        s1 = [(int(s["q"]), int(s["r"])) for s in stones if int(s["p"]) == p1]
        s2 = [(int(s["q"]), int(s["r"])) for s in stones if int(s["p"]) == p2]
        if not s1:
            continue
        origin = s1[0]
        s1.sort(key=lambda c: _hexdist(c, origin))
        s2.sort(key=lambda c: _hexdist(c, origin))
        seq, i1, i2, turn_p1 = [s1[0]], 1, 0, False
        while i1 < len(s1) or i2 < len(s2):
            src, idx = (s1, i1) if turn_p1 else (s2, i2)
            take = src[idx: idx + 2]
            seq.extend(take)
            if turn_p1:
                i1 += len(take)
            else:
                i2 += len(take)
            turn_p1 = not turn_p1
        board = Board.with_encoding_name(ring.header.encoding)
        try:
            for q, r in seq:
                board.apply_move(q, r)
        except ValueError:
            continue
        if board.check_win():
            continue
        ring_mover = p1 if board.current_player == 1 else p2
        if ring_mover == mover and board.moves_remaining == k:
            return board, seq
    return None


def _uniform_policy(board: Board) -> list[float]:
    n = board.size * board.size + 1
    legal = board.legal_moves()
    policy = [0.0] * n
    for q, r in legal:
        flat = board.to_flat(q, r)
        if flat < n:
            policy[flat] = 1.0 / len(legal)
    return policy


def child_q(board: Board, cell: tuple[int, int]) -> float | None:
    """The falsifier's probe: root expanded (uniform, NN = 0), one forced descent into `cell`, its Q in the root's view."""
    tree = MCTSTree()
    tree.configure_search("gumbel", 50.0, 1.0, True)
    tree.new_game(board)
    root = tree.select_leaves(1)
    tree.expand_and_backup([_uniform_policy(b) for b in root], [0.0] * len(root))
    row = next((r for r in tree.get_root_children_info() if tuple(r[0]) == cell), None)
    if row is None:
        return None
    leaves = tree.select_leaves_forced([row[1]])
    tree.expand_and_backup([_uniform_policy(b) for b in leaves], [0.0] * len(leaves))
    return float(next(r for r in tree.get_root_children_info() if tuple(r[0]) == cell)[4])


def quiescence_residue(ring: Ring, stats: BlockStats, child_q: ChildQ = child_q) -> Residue:
    """Over `stats.residue_candidates`: rows where the one-hot child reads >= the block child (A-2's 0/275)."""
    n_through = count = unreached = 0
    rows: list[int] = []
    for i, one_hot, block in stats.residue_candidates:
        board = reconstruct(ring, i)
        qc = child_q(board, one_hot) if board is not None else None
        qb = child_q(board, block) if board is not None else None
        if qc is None or qb is None:
            unreached += 1
            continue
        n_through += 1
        if qc >= qb:
            count += 1
            rows.append(i)
    return Residue(n_through, count, rows, unreached)


def _stat_rows(prefix: str, h: np.ndarray, producer: str) -> list[Row]:
    if h.size == 0:
        return [Row(f"h_{prefix}_median", None, 0, producer, "no rows in the arm"),
                Row(f"one_hot_share_{prefix}", None, 0, producer, "no rows in the arm")]
    n = int(h.size)
    return [
        Row(f"h_{prefix}_median", float(np.median(h)), n, producer),
        Row(f"h_{prefix}_mean", float(h.mean()), n, producer),
        Row(f"h_{prefix}_p75", float(np.quantile(h, 0.75)), n, producer),
        Row(f"h_{prefix}_p90", float(np.quantile(h, 0.9)), n, producer),
        Row(f"one_hot_share_{prefix}", float((h < ONE_HOT_H).mean()), n, producer, f"rows with H < {ONE_HOT_H:g}"),
    ]


def entropy_rows(ring: Ring) -> list[Row]:
    """H(explicit) in nats by arm (`is_full_search`) and pooled, with the one-hot share beside each arm, the full arm's share split by `moves_remaining` (R366(c): PROBE-1's decomposer read mr 1 at 30–37 % against mr 2 at 15–20 % on every run8 ring) and its tail-only (α = 1.0) row count."""
    h = explicit_entropy(ring)
    producer = "ring_reader.explicit_entropy"
    full_mask = ring.is_full_search != 0
    full, quick = h[full_mask], h[~full_mask]
    pooled = [Row("h_pooled_median", float(np.median(h)), int(h.size), producer),
              Row("h_pooled_mean", float(h.mean()), int(h.size), producer)] if h.size else []
    return (_stat_rows("full", full, producer) + _stat_rows("quick", quick, producer) + pooled
            + per_mr_rows(ring, h, full_mask, producer))


def per_mr_rows(ring: Ring, h: np.ndarray, full_mask: np.ndarray, producer: str) -> list[Row]:
    """`one_hot_share_full_mr<k>` over the full-arm rows at each stored `moves_remaining`, plus `tail_only_full` — the α = 1.0 rows the audited share counts as one-hots (H = 0), so a reader sees the conflation's size rather than trusting its absence."""
    rows: list[Row] = []
    for k in sorted({int(v) for v in ring.moves_remaining[full_mask]}):
        sel = full_mask & (ring.moves_remaining == k)
        rows.append(Row(f"one_hot_share_full_mr{k}", float((h[sel] < ONE_HOT_H).mean()), int(sel.sum()), producer,
                        f"full-arm rows at moves_remaining {k} with H < {ONE_HOT_H:g}"))
    tail_only = int(sum(1 for a in ring.tail_mass[full_mask] if is_alpha_full(float(a))))
    rows.append(Row("tail_only_full", float(tail_only), int(full_mask.sum()), "tail_mass at alpha = 1.0 (util.constants.is_alpha_full)",
                    "full-arm rows with no explicit mass, counted as one-hots by the share above"))
    return rows


def outcome_rows(ring: Ring) -> list[Row]:
    """Cap rate and draw share over DISTINCT games (`game_id`), mean |z| over value-supervised rows."""
    if ring.header.size == 0:
        return [Row("cap_rate", None, 0, "value_valid", "empty ring")]
    _, first = np.unique(ring.game_id, return_index=True)
    valid = ring.value_valid[first] != 0
    z = np.abs(ring.outcome[first])
    n_games = int(first.size)
    note = "" if (ring.game_id >= 0).all() else "untagged rows (game_id -1) counted as one game"
    rows = [Row("cap_rate", float((~valid).mean()), n_games,
                "value_valid == 0 (finalize: terminal_reason ply_cap)", note)]
    n_valid = int(valid.sum())
    rows.append(Row("draw_share", float((z[valid] < 1.0).mean()) if n_valid else None, n_valid,
                    "outcome not ±1 on a value-supervised game", "" if n_valid else "no supervised game"))
    sup = ring.value_valid != 0
    rows.append(Row("mean_abs_z", float(np.abs(ring.outcome[sup]).mean()) if sup.any() else None,
                    int(sup.sum()), "outcome over value_valid rows"))
    return rows


def audit(ring: Ring, child_q: ChildQ = child_q) -> list[Row]:
    """Every row the audit prints, each naming its producer; a band is checked against these keys."""
    blocks = forced_block_stats(ring)
    oracle = "tactics.analyze forced(k) == block; mass = explicit target mass on B(k)"
    named = blocks.counter_threat_rows[:_NAMED_ROWS]
    more = len(blocks.counter_threat_rows) - len(named)
    rows = [
        Row("forced_block_n", float(blocks.n), blocks.n, oracle),
        Row("block_mass_median", blocks.mass_median if blocks.n else None, blocks.n, oracle,
            "" if blocks.n else "no forced-block rows"),
        Row("block_mass_share_ge_half", blocks.share_ge_half if blocks.n else None, blocks.n, oracle,
            "" if blocks.n else "no forced-block rows"),
        Row("counter_threat_share", blocks.counter_threat_share, blocks.n, oracle,
            (f"mass(B) < {COUNTER_THREAT_MASS:g}; rows {named}" + (f" (+{more} more)" if more else ""))
            if blocks.n else "no forced-block rows"),
    ]
    residue = quiescence_residue(ring, blocks, child_q=child_q)
    rows.append(Row("quiescence_residue", float(residue.count), residue.n_through,
                    "A-2 falsifier oracle: child_q(one-hot) >= child_q(block) through the real backup",
                    f"rows {residue.rows[:_NAMED_ROWS]}; candidates unreached {residue.n_unreached}"))
    rows += entropy_rows(ring)
    rows += outcome_rows(ring)
    rows.append(Row("sample_age", None, None, "none",
                    "NOT MEASURED: the ring carries no step field (step written vs step read)"))
    return rows


_EVENT_ROWS = "iteration_complete rows"
_REPLAY_PAIR = ("samples_consumed_total", "positions_produced_total")


def _iteration_rows(events: Path) -> list[dict[str, Any]]:
    """Every `iteration_complete` row of an events file, in order; a partial last line is skipped."""
    out: list[dict[str, Any]] = []
    with events.open(encoding="utf-8") as fh:
        for line in fh:
            if "iteration_complete" not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("event") == "iteration_complete":
                out.append(row)
    return out


def _int_field(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def replay_ratio_row(rows: list[dict[str, Any]], ring_size: int) -> Row:
    """Δsamples_consumed_total ÷ Δpositions_produced_total from the first row within `ring_size` positions of the last paired row."""
    producer = f"{_REPLAY_PAIR[0]} ÷ {_REPLAY_PAIR[1]} on {_EVENT_ROWS} (R358(c))"
    paired = [r for r in rows if all(_int_field(r, k) is not None for k in _REPLAY_PAIR)]
    if not paired:
        missing = sorted({k for r in rows for k in _REPLAY_PAIR if _int_field(r, k) is None}) or list(_REPLAY_PAIR)
        return Row("replay_ratio", None, len(rows), producer, f"NOT MEASURED: no row carries {missing}")
    if len(paired) < 2:
        return Row("replay_ratio", None, 1, producer, "NOT MEASURED: one row is no span")
    end = paired[-1]
    end_pos = _int_field(end, _REPLAY_PAIR[1]) or 0
    span = [r for r in paired if (_int_field(r, _REPLAY_PAIR[1]) or 0) >= end_pos - ring_size]
    start = span[0]
    d_pos = end_pos - (_int_field(start, _REPLAY_PAIR[1]) or 0)
    d_samples = (_int_field(end, _REPLAY_PAIR[0]) or 0) - (_int_field(start, _REPLAY_PAIR[0]) or 0)
    ts0, ts1 = start.get("ts"), end.get("ts")
    hours = (float(ts1) - float(ts0)) / 3600.0 if isinstance(ts0, (int, float)) and isinstance(ts1, (int, float)) else None
    where = (f"span {hours:.2f} h" if hours is not None else "span (no ts)") + \
        f", positions {_int_field(start, _REPLAY_PAIR[1])} → {end_pos}, " \
        f"samples {_int_field(start, _REPLAY_PAIR[0])} → {_int_field(end, _REPLAY_PAIR[0])}"
    if d_pos <= 0:
        return Row("replay_ratio", None, len(span), producer, f"NOT MEASURED: no position produced over the span; {where}")
    return Row("replay_ratio", d_samples / d_pos, len(span), producer, where)


def sym_uniformity_row(rows: list[dict[str, Any]]) -> Row:
    """bin 0 ÷ the mean bin off the LAST row carrying `sym_draws`: 1.0 is uniform, 12 is a draw stuck on the identity."""
    producer = f"sym_draws.bins on the last of the {_EVENT_ROWS} (R358(b), LAW-18)"
    blocks = [r["sym_draws"] for r in rows if isinstance(r.get("sym_draws"), dict)]
    if not blocks:
        return Row("sym_bin0_over_mean", None, len(rows), producer, "NOT MEASURED: no row carries sym_draws")
    block = blocks[-1]
    bins = block.get("bins")
    if not isinstance(bins, list) or not bins or not all(isinstance(b, int) for b in bins):
        return Row("sym_bin0_over_mean", None, len(rows), producer, f"NOT MEASURED: malformed bins {bins!r}")
    total = sum(bins)
    note = f"bins {bins}; empty_skipped {block.get('empty_skipped')}"
    if total == 0:
        return Row("sym_bin0_over_mean", None, 0, producer, f"NOT MEASURED: no draw yet; {note}")
    return Row("sym_bin0_over_mean", bins[0] / (total / len(bins)), total, producer, note)


def event_rows(events: Path | None, *, ring_size: int) -> list[Row]:
    """The two rows read off the events stream (R358(b)/(c)); NOT MEASURED, naming `--events`, without one."""
    if events is None:
        absent = "NOT MEASURED: pass --events <events_<run>_seg*.jsonl>; the ring carries no counter"
        return [Row("replay_ratio", None, None, "none", absent), Row("sym_bin0_over_mean", None, None, "none", absent)]
    rows = _iteration_rows(events)
    return [replay_ratio_row(rows, ring_size), sym_uniformity_row(rows)]


def load_bands(path: Path) -> dict[str, tuple[str, float]]:
    """`[ring_audit.bands]` off a `.toml` or the first fenced toml block of a `.md` carrying it; Raises: ValueError — no table, a malformed file or an operator outside lt/le/gt/ge."""
    text = path.read_text(encoding="utf-8")
    bodies = [text] if path.suffix == ".toml" else _FENCE.findall(text)
    table: dict[str, object] | None = None
    for body in bodies:
        try:
            parsed = tomllib.loads(body)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: malformed toml: {exc}") from exc
        found = parsed.get("ring_audit", {}).get("bands")
        if isinstance(found, dict):
            table = found
            break
    if table is None:
        raise ValueError(f"{path}: no [ring_audit.bands] table")
    bands: dict[str, tuple[str, float]] = {}
    for key, spec in table.items():
        if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in _OPS:
            raise ValueError(f"{path}: band {key!r} must be one of {sorted(_OPS)} = <number>, got {spec!r}")
        op, bound = next(iter(spec.items()))
        bands[key] = (op, float(bound))
    return bands


def check_bands(rows: list[Row], bands: dict[str, tuple[str, float]]) -> tuple[list[str], list[str]]:
    """(misses, unknown keys): a banded row with no value is a miss, never a pass."""
    by_key = {row.key: row for row in rows}
    misses, unknown = [], []
    for key, (op, bound) in bands.items():
        row = by_key.get(key)
        if row is None:
            unknown.append(key)
        elif row.value is None:
            misses.append(f"{key}: NOT MEASURED ({row.note})")
        elif not _OPS[op](row.value, bound):
            misses.append(f"{key}: {row.value:.6g} not {op} {bound:g}; {row.note}".rstrip("; "))
    return misses, unknown


def _print_table(rows: list[Row], bands: dict[str, tuple[str, float]]) -> None:
    print(f"{'row':<26} {'value':>14} {'n':>7} {'band':<12} producer / note")
    for row in rows:
        value = "NOT MEASURED" if row.value is None else f"{row.value:.6g}"
        band = "{} {:g}".format(*bands[row.key]) if row.key in bands else "-"
        note = f" — {row.note}" if row.note else ""
        n = "-" if row.n is None else str(row.n)
        print(f"{row.key:<26} {value:>14} {n:>7} {band:<12} {row.producer}{note}")


def main(argv: list[str]) -> int:
    """CLI: `<ring> [--bands <prereg.md|bands.toml>] [--events <jsonl>]` → the table; rc 1 on a miss, 2 on a refusal."""
    parser = argparse.ArgumentParser(prog="mantis.diagnostics.ring_audit")
    parser.add_argument("ring", type=Path)
    parser.add_argument("--bands", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None,
                        help="the run's events file: replay_ratio and sym_bin0_over_mean read off its iteration_complete rows")
    args = parser.parse_args(argv)
    try:
        ring = load_ring(args.ring)
        bands = load_bands(args.bands) if args.bands else {}
        rows = audit(ring) + event_rows(args.events, ring_size=ring.header.size)
    except (OSError, ValueError) as exc:
        print(f"ring_audit: REFUSED: {exc}")
        return 2
    misses, unknown = check_bands(rows, bands)
    print(f"{args.ring}: encoding={ring.header.encoding} rows={ring.header.size}")
    _print_table(rows, bands)
    if unknown:
        print(f"ring_audit: REFUSED: bands name rows the audit does not produce: {unknown}")
        return 2
    if not bands:
        print("ring_audit: no bands given — reported, nothing gated")
        return 0
    if misses:
        print("ring_audit: MISS on " + "; ".join(misses))
        return 1
    print(f"ring_audit: PASS ({len(bands)} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
