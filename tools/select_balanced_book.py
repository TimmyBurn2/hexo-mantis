"""BOOK_V2 selector: keep the pool openings whose paired anchor-vs-itself result SPLITS by seat; `--procedure` prints the box run-book (`BOX_PROCEDURE`, a constant because gate 14's docstring ratchet sits at its floor). >300 justify (R8): the pair law, the verdict, the cut, the report and the run-book that orders them are one measurement's definition and must read as one unit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mantis.arena.books import Opening, book_openings
from mantis.monitor.game_record import read_shard
from mantis.util.hashing import sha256_file

SEAT_WINS = ("p1", "p2")
CHANNEL = "promotion"
DEFAULT_SIMS = 256
DEFAULT_N = 128
DEFAULT_MIN_REPLAYS = 4

BOX_PROCEDURE = """\
BOOK_V2 box procedure (R354(d)). Every command runs from the repo root of the box checkout;
$CONFIG is the anchor's minted config under configs/.
Cost, stated once: 4 replays x 2 games x 512 pool openings = 4096 games at PUCT-256, i.e. 1024
games per replay cell. At run7's measured gate cost beside a live run (RUN7_EVAL_COST: 0.49-0.84
s/ply, 55-77 plies/game, concurrency 8) that is roughly 27-65 s/game, so 31-74 box-hours serial;
the box measures it, this line does not.
Honesty about the replays: under `puct` the deploy head consumes NO seed (`_move_seed` feeds only
the Gumbel root pick; `MCTSTree()` is unseeded), so `seed_base` changes the opening ORDER and the
record's `seed` field, never the search. Four replays differ only through the engine's own
run-to-run nondeterminism (batched GPU inference, games in flight). The selector reports
`identical_replays` per opening and in its summary: if it is near the pool size, the four replays
were one measurement and the verdicts are one replay's, not four.

0. The pool (already minted and registered; re-mint to check its sha before spending a box-day):
   uv run python tools/mint_opening_book.py --seed 20260915 --plies 4 --n 512 \\
     --exclude-book src/mantis/arena/books/book_v1_s20260625_p4.json --out $WORK/pool_check.json
   sha256sum $WORK/pool_check.json src/mantis/arena/books/book_v2_pool_s20260915_p4.json  # equal

1. The anchor-vs-itself cells: `candidate` and `opponent` are the SAME checkpoint (the run's
   promoted anchor's .ckpt), through the GATE block, 4 cells with distinct `seed_base`, `games`
   = 2 x 512 so every replay's window covers the whole pool. Write $WORK/cells.json:
   [
    {"label": "anchor_self_r1", "candidate": "$ANCHOR", "opponent": "$ANCHOR", "search_kind": "puct",
     "sims": 256, "games": 1024, "concurrency": 8, "opening_book": "book_v2_pool_s20260915_p4",
     "seed_base": 1},
    {"label": "anchor_self_r2", ... "seed_base": 2},
    {"label": "anchor_self_r3", ... "seed_base": 3},
    {"label": "anchor_self_r4", ... "seed_base": 4}
   ]
   uv run python tools/strength_frontier.py --config $CONFIG --cells $WORK/cells.json \\
     --work-dir $WORK/book_v2 --parallel 1
   Each cell writes its games to $WORK/book_v2/<label>/games/games_frontier1_seg*_*.jsonl
   (channel `promotion`, `seed` = the cell's seed_base, `served_sims` 256).

2. The selector (fails loud below 128 balanced openings or below 4 complete replays per opening):
   uv run python tools/select_balanced_book.py --pool book_v2_pool_s20260915_p4 --sims 256 --n 128 \\
     --games $WORK/book_v2/anchor_self_r*/games/games_frontier1_seg*.jsonl \\
     --out src/mantis/arena/books/book_v2_p4.json --book-id book_v2_p4 \\
     --report $WORK/book_v2/book_v2_report.md
   It prints the book's sha256 and the manifest.toml lines.

3. Append the printed lines to src/mantis/arena/books/manifest.toml; commit the book file and the
   manifest together (the book does not exist until this measurement ran).

4. The bridge cell, recorded ONCE: the anchor vs strix 256/256 (the external scale; the sealbot
   cell went with the sealbot rung, R362(c)) on BOTH books, same games:
   [
    {"label": "bridge_v1", "candidate": "$ANCHOR", "opponent": "strix", "strix_sims": 256,
     "search_kind": "puct", "sims": 256, "games": 256, "concurrency": 8,
     "opening_book": "book_v1_s20260625_p4"},
    {"label": "bridge_v2", "candidate": "$ANCHOR", "opponent": "strix", "strix_sims": 256,
     "search_kind": "puct", "sims": 256, "games": 256, "concurrency": 8,
     "opening_book": "book_v2_p4"}
   ]
   uv run python tools/strength_frontier.py --config $CONFIG --cells $WORK/bridge.json \\
     --work-dir $WORK/bridge --parallel 1
   Readings on book_v2 are a NEW unit; the two bridge rows are the only comparison across books.
"""


class SelectionError(ValueError):
    """The selector's named failure: malformed or mismatched game rows."""


class ReplayCoverageError(SelectionError):
    """An opening has fewer complete colour-swapped pairs than `min_replays` distinct seeds."""


class BalancedCountError(SelectionError):
    """Fewer openings split than the book needs."""


@dataclass
class OpeningAssessment:
    """One pool opening's verdict over its replays; `kept` is set by `select_book`."""

    pool_id: int
    moves: list[list[int]]
    replays: int
    pairs_split: int
    pairs_seat_decided: int
    pairs_undecided: int
    distinct_trajectories: tuple[int, int]
    verdict: str
    kept: bool = False
    replay_seeds: list[int] = field(default_factory=list)
    incomplete_replays: int = 0

    @property
    def identical_replays(self) -> bool:
        """True when every replay produced the same two games: one measurement, not `replays`."""
        return self.distinct_trajectories == (1, 1)


def classify_pair(first: str, second: str) -> str:
    """`split` when the two seat-relative results differ and both are seat wins, `seat_decided` when they are the same seat win, else `undecided` (a `draw` or `unknown` leg is neither seat's win)."""
    if first not in SEAT_WINS or second not in SEAT_WINS:
        return "undecided"
    return "seat_decided" if first == second else "split"


def _opening_key(moves: Iterable[Iterable[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(q), int(r)) for q, r in moves)


def match_pool(pool: Sequence[Any], records: Iterable[Mapping[str, Any]], *, sims: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the `promotion` rows whose first plies are a pool opening, tagged `pool_index`; returns them and the skip tally. Raises: SelectionError when a matched row's `served_sims` is not `sims` or a row lacks a needed field."""
    plies = {len(o["moves"] if isinstance(o, Mapping) else o.moves) for o in pool}
    if len(plies) != 1:
        raise SelectionError(f"the pool mixes opening lengths {sorted(plies)}; one length is needed to key games")
    depth = plies.pop()
    index = {_opening_key(o["moves"] if isinstance(o, Mapping) else o.moves): i for i, o in enumerate(pool)}
    matched: list[dict[str, Any]] = []
    skipped = {"unmatched": 0, "other_channel": 0}
    for record in records:
        if record.get("channel") != CHANNEL:
            skipped["other_channel"] += 1
            continue
        try:
            key = _opening_key(record["moves"][:depth])
            seat, seed, served = int(record["colors"]["candidate"]), int(record["seed"]), int(record["served_sims"])
            result = str(record["result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SelectionError(f"game {record.get('game_id')!r} lacks a field the selector reads: {exc}") from exc
        if key not in index or len(record["moves"]) < depth:
            skipped["unmatched"] += 1
            continue
        if served != sims:
            raise SelectionError(f"game {record.get('game_id')!r} has served_sims={served}, not the cell's {sims}; mixed budgets are not one measurement")
        matched.append({**record, "pool_index": index[key], "seat": seat, "seed": seed, "result": result})
    return matched, skipped


def _trajectory(record: Mapping[str, Any]) -> str:
    return str(record.get("trajectory_hash") or json.dumps(record["moves"]))


def _assess_one(pool_id: int, moves: list[list[int]], by_seed: Mapping[int, list[dict[str, Any]]], *, min_replays: int, min_split: int) -> OpeningAssessment:
    counts = {"split": 0, "seat_decided": 0, "undecided": 0}
    trajectories: tuple[set[str], set[str]] = (set(), set())
    seeds: list[int] = []
    incomplete = 0
    for seed in sorted(by_seed):
        legs = by_seed[seed]
        seats = sorted(int(r["seat"]) for r in legs)
        if len(seats) == 1:
            incomplete += 1
            continue
        if seats != [-1, 1]:
            raise SelectionError(f"pool opening {pool_id} under seed {seed} has candidate seat(s) {seats}, not one game per colour")
        first = next(r for r in legs if r["seat"] == 1)
        second = next(r for r in legs if r["seat"] == -1)
        counts[classify_pair(first["result"], second["result"])] += 1
        trajectories[0].add(_trajectory(first))
        trajectories[1].add(_trajectory(second))
        seeds.append(seed)
    if len(seeds) < min_replays:
        raise ReplayCoverageError(f"pool opening {pool_id} is covered by {len(seeds)} replay(s) with a complete pair (seeds {seeds}, {incomplete} incomplete); {min_replays} are required")
    if counts["split"] >= min_split:
        verdict = "split"
    elif counts["undecided"] == 0:
        verdict = "seat_decided"
    else:
        verdict = "undecided"
    return OpeningAssessment(pool_id=pool_id, moves=moves, replays=len(seeds), pairs_split=counts["split"],
                             pairs_seat_decided=counts["seat_decided"], pairs_undecided=counts["undecided"],
                             distinct_trajectories=(len(trajectories[0]), len(trajectories[1])),
                             verdict=verdict, replay_seeds=seeds, incomplete_replays=incomplete)


def assess(pool: Sequence[Any], records: Iterable[Mapping[str, Any]], *, sims: int, min_replays: int = DEFAULT_MIN_REPLAYS, min_split: int = 1) -> list[OpeningAssessment]:
    """Every pool opening's verdict in pool order: `split` (kept) when at least `min_split` replays' colour-swapped pair splits by seat, `seat_decided` when every replay's pair was won by one seat or the other (fewer than `min_split` of them splitting), else `undecided` (a `draw`/`unknown` leg somewhere and no keep); a replay with one leg is counted `incomplete_replays` and covers nothing. Raises: ReplayCoverageError below `min_replays` complete pairs; SelectionError on a malformed row or a mixed `served_sims`."""
    matched, _skipped = match_pool(pool, records, sims=sims)
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in matched:
        grouped[record["pool_index"]][record["seed"]].append(record)
    out: list[OpeningAssessment] = []
    for i, opening in enumerate(pool):
        moves = [list(m) for m in (opening["moves"] if isinstance(opening, Mapping) else opening.moves)]
        pool_id = int(opening["id"]) if isinstance(opening, Mapping) else int(opening.opening_id)
        out.append(_assess_one(pool_id, moves, grouped.get(i, {}), min_replays=min_replays, min_split=min_split))
    return out


def select_book(assessed: Sequence[OpeningAssessment], *, n: int = DEFAULT_N) -> list[OpeningAssessment]:
    """Mark and return the first `n` `split` openings in pool order. Raises: BalancedCountError when fewer than `n` split."""
    kept = [a for a in assessed if a.verdict == "split"]
    if len(kept) < n:
        raise BalancedCountError(f"only {len(kept)} of {len(assessed)} pool openings split; the book needs {n}. Grow the pool or rerun the replays before minting")
    for a in kept[:n]:
        a.kept = True
    return kept[:n]


def summarize(assessed: Sequence[OpeningAssessment]) -> dict[str, Any]:
    """The per-run totals a reader checks first: verdict counts, kept count and how many openings replayed identically."""
    return {
        "openings": len(assessed), "kept": sum(a.kept for a in assessed),
        "split": sum(a.verdict == "split" for a in assessed),
        "seat_decided": sum(a.verdict == "seat_decided" for a in assessed),
        "undecided": sum(a.verdict == "undecided" for a in assessed),
        "identical_replays": sum(a.identical_replays for a in assessed),
        "replays_min": min((a.replays for a in assessed), default=0),
    }


def book_payload(kept: Sequence[OpeningAssessment], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The book in book_v1's shape (`openings[].id/.moves`, ids `0..n-1`) plus a `provenance` block naming the pool ids kept."""
    openings = [{"id": i, "moves": a.moves} for i, a in enumerate(kept)]
    return {"openings": openings, "provenance": {**provenance, "pool_ids": [a.pool_id for a in kept]}}


def manifest_lines(book_id: str, file_name: str, sha256: str) -> str:
    """The `manifest.toml` entry to append, verbatim."""
    return f'[books."{book_id}"]\nfile = "{file_name}"\nsha256 = "{sha256}"\n'


def _row(a: OpeningAssessment) -> dict[str, Any]:
    return {"row": "opening", **asdict(a), "identical_replays": a.identical_replays}


def write_report(path: Path, assessed: Sequence[OpeningAssessment], summary: Mapping[str, Any]) -> None:
    """Write the per-opening table: JSONL (`row` = `opening` | `summary`) for `.jsonl`, a markdown table for `.md`. Raises: SelectionError on any other suffix."""
    if path.suffix == ".jsonl":
        lines = [json.dumps(_row(a), sort_keys=True) for a in assessed]
        lines.append(json.dumps({"row": "summary", **summary}, sort_keys=True))
    elif path.suffix == ".md":
        lines = ["| pool_id | replays | split | seat_decided | undecided | distinct (p1 seat, p2 seat) | verdict | kept |",
                 "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {a.pool_id} | {a.replays} | {a.pairs_split} | {a.pairs_seat_decided} | {a.pairs_undecided} "
                  f"| {a.distinct_trajectories[0]}, {a.distinct_trajectories[1]} | {a.verdict} | {'yes' if a.kept else 'no'} |"
                  for a in assessed]
        lines += ["", "summary: " + ", ".join(f"{k} {v}" for k, v in summary.items())]
    else:
        raise SelectionError(f"--report {path}: the suffix must be .jsonl or .md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_games(paths: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    torn = 0
    for shard in paths:
        rows, skipped = read_shard(shard)
        records.extend(rows)
        torn += skipped
    return records, torn


def main(argv: Sequence[str] | None = None) -> int:
    """Select the balanced book from the pool and the replays' game shards; print the sha and the manifest lines. Raises: SelectionError and its subclasses as `assess`/`select_book`/`write_report` do; BookError when the pool does not resolve."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--procedure", action="store_true", help="print the box procedure and exit")
    parser.add_argument("--pool", help="the pool's manifest id (resolved through mantis.arena.books)")
    parser.add_argument("--books-dir", default=None, help="override the packaged books dir (tests)")
    parser.add_argument("--games", nargs="+", default=[], help="game-record shard files of the replay cells")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--min-replays", type=int, default=DEFAULT_MIN_REPLAYS)
    parser.add_argument("--min-split", type=int, default=1, help="replays whose pair must split")
    parser.add_argument("--out", help="the book JSON to write")
    parser.add_argument("--book-id", default="book_v2_p4")
    parser.add_argument("--report", default=None, help=".jsonl or .md per-opening table")
    args = parser.parse_args(argv)
    if args.procedure:
        print(BOX_PROCEDURE)
        return 0
    if not (args.pool and args.games and args.out):
        parser.error("--pool, --games and --out are required (or --procedure)")
    pool: list[Opening] = book_openings(args.pool, books_dir=args.books_dir)
    records, torn = _read_games(args.games)
    _matched, skipped = match_pool(pool, records, sims=args.sims)
    assessed = assess(pool, records, sims=args.sims, min_replays=args.min_replays, min_split=args.min_split)
    kept = select_book(assessed, n=args.n)
    summary = summarize(assessed)
    seeds = sorted({s for a in assessed for s in a.replay_seeds})
    provenance = {"selector": "tools/select_balanced_book.py", "pool": args.pool, "sims": args.sims,
                  "replay_seeds": seeds, "min_replays": args.min_replays, "min_split": args.min_split,
                  "rule": "kept when >= min_split replays' colour-swapped pair splits by seat"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(book_payload(kept, provenance=provenance), sort_keys=True, separators=(",", ":")), encoding="utf-8")
    sha = sha256_file(out)
    if args.report:
        write_report(Path(args.report), assessed, summary)
    print(f"games read {len(records)} (torn lines {torn}, unmatched {skipped['unmatched']}, other channel {skipped['other_channel']})")
    print("summary: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
    if summary["identical_replays"]:
        print(f"WARNING: {summary['identical_replays']} of {summary['openings']} openings replayed IDENTICALLY across every seed; those verdicts rest on one measurement (PUCT consumes no seed)")
    print(f"wrote {out} ({len(kept)} openings) sha256 {sha}")
    print("manifest.toml lines to append:\n" + manifest_lines(args.book_id, out.name, sha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
