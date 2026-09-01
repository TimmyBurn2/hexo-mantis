# >300 justify: the replay, the record stamp and the provenance the artifact is worthless
# without are ONE producer over ONE external contract. Splitting the encoder from the
# provenance would let an artifact be written whose sidecar was computed elsewhere, which is
# the class the corpus sidecar (`corpus_io`) already exists to close one layer up.
"""Encode an audited move-list bootstrap corpus into GRAPH-PATH training records.

WHY THIS EXISTS. `tools/audit_bootstrap_corpus.py` CERTIFIES a corpus of axial move lists
(`{game_hash, winner, elo, moves: [[q, r], ...]}`, contract v2), and R279 made that
certification prereg grounds. Nothing in the tree could then USE it: SITTING4-PREP-1 §3.5
measured the gap — *"the encoder between them does not exist in-tree"*, `corpus_io.save_corpus`
has zero non-test callers, and `mantis.data.generate` produces bot self-play games as JSON move
lists, not arrays. This module is that encoder for the GRAPH arch, which is the one run5 and
run6 actually train.

**IT IS CAPABILITY, NOT POSTURE.** Nothing here is selected by any config, armed by any
resolver or reached by any production path. `train.mixing.pretrained_buffer_path` resolves
through `resolve_corpus_path` to `data/gnn_corpus_v1.hexg` for `gnn_axis_v1`, and this module
is what can produce that file — but a run consumes it only if the operator mints the key.
Landing is not arming.

WHAT IT PRODUCES, and why in this shape. One row per PLY, pushed through the production
`HexgBuffer.push_graph_position` — the same ring the self-play worker writes and the same one
the trainer samples. No new artifact format is invented: the trainer's own `.hexg` is the
target, so the loader that reads it already exists and is already tested.

THE THREE THINGS A ROW NEEDS, each taken from an authority rather than restated:
  * the POSITION — replayed on a production `Board`, one `apply_move` per stone. The corpus
    is per-STONE (the audit's contract says so explicitly), and Hex Tac Toe's compound
    two-stone turn is the BOARD's business, tracked by `moves_remaining`.
  * the POLICY TARGET — a one-hot on the stone actually played. Behaviour cloning has no
    visit distribution to copy, and the engine refuses a `visits` row that is not a
    distribution, so a one-hot is both the honest target and the only admissible one.
  * the VALUE TARGET — `mantis._engine.graph_row_outcome`, which IS
    `mantis_selfplay::records::finalize_graph_outcome`. A Python transcription of that sign
    convention would agree today and drift the first time the §178 split moves.

WHAT IT REFUSES, loudly, and never coerces: an illegal move against the replayed board, a
winner outside `{+1, -1}`, an empty move list, a record missing a required field, a source
file whose sha256 disagrees with its manifest. A corpus is training data; a coerced record is
a wrong label that no downstream check can see.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.encoding import assert_not_heldout_sha

#: The runner's terminal-reason code for a normal decided/drawn end. `2` is the ply-cap
#: branch, which a completed human game never takes — a corpus game ended, it was not
#: truncated by our search budget.
_TERMINAL_DECIDED = 0

#: Values the decided branch never reads. Named rather than passed as bare zeros so a reader
#: can see they are inert here, and so a future draw-aware corpus has one place to change.
_PLY_CAP_VALUE = 0.0
_DRAW_REWARD = 0.0

_HASH_CHUNK = 1 << 20


class CorpusEncodeError(ValueError):
    """A corpus record cannot be encoded. Names the record and what was wrong."""


def sha256_of(path: Path) -> str:
    """Streaming sha256 of a file, for the manifest handshake."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_record(rec: Any, idx: int) -> tuple[str, int, list[tuple[int, int]]]:
    """The audit's contract v2, re-checked at the point of USE.

    Re-checked and not trusted: the audit runs over a dataset directory and this runs over
    whatever the caller hands it, and "it was audited once" is not a property of the object
    in front of us (LAW-01).

    Raises:
        CorpusEncodeError: any required field absent, of the wrong type, or out of range.
    """
    if not isinstance(rec, dict):
        raise CorpusEncodeError(f"record {idx}: not a JSON object ({type(rec).__name__})")
    game_hash = rec.get("game_hash")
    if not isinstance(game_hash, str) or not game_hash:
        raise CorpusEncodeError(f"record {idx}: `game_hash` absent or not a non-empty string")
    winner = rec.get("winner")
    # `isinstance(..., bool)` FIRST: `True == 1` in Python, so a bare membership test admits
    # a boolean winner and stamps every row of that game with a real sign. Caught by this
    # module's own oracle rather than in a corpus.
    if isinstance(winner, bool) or winner not in (1, -1):
        raise CorpusEncodeError(
            f"record {game_hash}: `winner` is {winner!r}; the contract declares 1 or -1. "
            "Refused rather than mapped — a guessed winner is a value target with the wrong "
            "sign on every row of the game."
        )
    moves = rec.get("moves")
    if not isinstance(moves, list) or not moves:
        raise CorpusEncodeError(f"record {game_hash}: `moves` absent or empty")
    out: list[tuple[int, int]] = []
    for j, mv in enumerate(moves):
        if (not isinstance(mv, (list, tuple)) or len(mv) != 2
                or not all(isinstance(c, int) and not isinstance(c, bool) for c in mv)):
            raise CorpusEncodeError(
                f"record {game_hash}: move {j} is {mv!r}; the contract declares a 2-element "
                "array of ints [q, r]"
            )
        out.append((int(mv[0]), int(mv[1])))
    return game_hash, int(winner), out


def encode_game(
    moves: Sequence[tuple[int, int]], winner: int, *, board_factory: Any,
    game_hash: str = "<unnamed>",
) -> Iterator[tuple[Any, ...]]:
    """Yield one `push_graph_position` row per ply of one completed game.

    The row tuple is in the ENGINE'S POSITIONAL ORDER, which is also the order
    `mantis.selfplay.pool_push.push_graph` forwards verbatim — so a row produced here and a
    row produced by self-play are the same object to the ring.

    Args:
        moves: axial `(q, r)` per STONE, in placement order.
        winner: `+1` or `-1`, the contract's winner field.
        board_factory: returns a fresh production `Board` bound to the target encoding.
        game_hash: the record's identity, used only in error messages.

    Yields:
        `(stones, visits, current_player, moves_remaining, ply_index, is_full_search,
        outcome, value_valid, game_length)`.

    Raises:
        CorpusEncodeError: a move is not legal on the replayed board, or the game ends
            before its move list does.
    """
    from mantis._engine import graph_row_outcome  # noqa: PLC0415 — extension, import-time cost

    board = board_factory()
    n = len(moves)
    for ply, (q, r) in enumerate(moves):
        if board.check_win():
            raise CorpusEncodeError(
                f"record {game_hash}: the board is already won at ply {ply} but the move "
                f"list has {n - ply} moves left. The replay and the corpus disagree about "
                "the rules; neither is silently preferred."
            )
        legal = board.legal_moves()
        if (q, r) not in legal:
            raise CorpusEncodeError(
                f"record {game_hash}: move {ply} = ({q}, {r}) is not legal on the replayed "
                f"board ({len(legal)} legal cells). Refused rather than skipped — a skipped "
                "move desynchronises every later position from its own label."
            )
        rec_player = int(board.current_player)
        outcome, value_valid = graph_row_outcome(
            rec_player, int(winner), _TERMINAL_DECIDED, _PLY_CAP_VALUE, _DRAW_REWARD,
        )
        yield (
            [(int(sq), int(sr), int(sp)) for sq, sr, sp in board.get_stones()],
            [(int(q), int(r), 1.0)],          # one-hot on the played stone
            rec_player,
            int(board.moves_remaining),
            ply,
            False,                             # is_full_search: BC has no search behind it
            float(outcome),
            bool(value_valid),
            n,
        )
        board.apply_move(q, r)


def _iter_records(path: Path) -> Iterator[Any]:
    """Records from a `.jsonl` (one object per non-blank line) or `.json` (array) file."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CorpusEncodeError(
            f"{path}: a `.json` record file must hold a JSON ARRAY of objects; got "
            f"{type(payload).__name__}"
        )
    yield from payload


def _manifest_pin(dataset_dir: Path) -> tuple[Path, str]:
    """The single record file and its declared sha256, from the audit's contract v2 manifest.

    Both declared manifest shapes are accepted and EXACTLY ONE must match, which is the
    audit's own rule rather than a widening invented here.

    Raises:
        CorpusEncodeError: the manifest is absent, matches neither shape or both, or names
            a file that is not present.
    """
    manifest_path = dataset_dir / "dataset_metadata.json"
    if not manifest_path.is_file():
        raise CorpusEncodeError(
            f"{manifest_path} is absent. The corpus's identity is its manifest sha; without "
            "it there is nothing to verify the records against and the artifact would carry "
            "a provenance nobody can check."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    has_a, has_b = "files" in manifest, "file" in manifest
    if has_a == has_b:
        raise CorpusEncodeError(
            f"{manifest_path}: exactly one of the declared shapes must be present; found "
            f"files={has_a} file={has_b}"
        )
    if has_b:
        entries = [(manifest["file"], manifest["sha256"])]
    else:
        entries = [(e["path"], e["sha256"]) for e in manifest["files"]]
    records = [(dataset_dir / rel, sha) for rel, sha in entries
               if Path(rel).suffix in (".json", ".jsonl")]
    if len(records) != 1:
        raise CorpusEncodeError(
            f"{manifest_path}: expected exactly one record file, found {len(records)}"
        )
    path, declared = records[0]
    if not path.is_file():
        raise CorpusEncodeError(f"{manifest_path} pins {path}, which is not present")
    return path, declared


def _ply_histogram(lengths: list[int], bucket: int = 64) -> dict[str, int]:
    """Game-length histogram in `bucket`-ply bins, as `{"<lo>-<hi>": count}`.

    Carried in the provenance because the truncation's SHAPE is what a reader needs and a
    single "88 games truncated" cannot show it: the loss is the late phase of the longest
    games, and only the distribution says how long those are.
    """
    out: dict[str, int] = {}
    width = max((len(str(n)) for n in lengths), default=1)
    for n in lengths:
        lo = (n // bucket) * bucket
        # ZERO-PADDED so lexicographic order IS numeric order. The provenance is written with
        # `sort_keys=True`, which re-sorts these keys and put "64-127" after "512-575" — a
        # histogram a reader has to re-sort by eye is one they will read wrong.
        key = f"{lo:0{width}d}-{lo + bucket - 1:0{width}d}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class CorpusSplit:
    """A seeded, GAME-level partition of the corpus into `train` and `heldout`.

    ONE object rather than three parameters, so the split cannot be half-specified: a seed
    without a fraction, or a fraction without a part, would each be a partition nobody
    declared. `None` means no split — the whole corpus, the behaviour that already existed.

    THE SPLIT IS BY GAME AND NOT BY PLY, and that is the whole point. The corpus encodes one
    row per PLY; a ply-level split puts positions from the SAME game on both sides, and a
    held-out loss measured over them is measuring memorisation of a game the model has already
    seen 60 positions of. The encoder is the last place that still knows which plies came from
    which game.

    ASSIGNMENT IS A KEYED HASH OF THE GAME'S OWN IDENTITY, not a shuffle of an order. Three
    consequences a shuffle would not give: the partition is INDEPENDENT of record order and of
    `max_games`, so a truncated smoke run draws the same side for the same game as a full run;
    it is reproducible from the seed alone, with no permutation to store; and it can never
    split a game, because the game is the unit being hashed.
    """

    seed: int
    heldout_frac: float
    part: str  # "train" | "heldout"

    def __post_init__(self) -> None:
        if self.part not in ("train", "heldout"):
            raise ValueError(f"CorpusSplit.part must be 'train' or 'heldout', got {self.part!r}")
        if not 0.0 < self.heldout_frac < 1.0:
            raise ValueError(
                f"CorpusSplit.heldout_frac must be strictly inside (0, 1), got "
                f"{self.heldout_frac!r}. A 0 or 1 fraction makes one side empty, which is a "
                "split nobody can evaluate against."
            )

    def selects(self, game_hash: str) -> bool:
        """True iff `game_hash` belongs to THIS part of the partition."""
        digest = hashlib.blake2b(
            game_hash.encode("utf-8"), key=str(self.seed).encode("utf-8"), digest_size=8
        ).digest()
        draw = int.from_bytes(digest, "big") / 2.0**64
        return (draw < self.heldout_frac) == (self.part == "heldout")


def encode_corpus(
    dataset_dir: Path, out_path: Path, *, encoding: str, capacity: int,
    visit_capacity: int, max_games: int | None = None, split: CorpusSplit | None = None,
) -> dict[str, Any]:
    """Encode an audited dataset directory into a `.hexg` ring, with provenance.

    Args:
        dataset_dir: the audited dataset (manifest + one record file).
        out_path: the `.hexg` artifact to write. Its provenance sidecar is written beside it.
        encoding: the registered graph encoding the ring is bound to.
        capacity: the ring's record capacity. Sized by the CALLER from the corpus, never
            guessed here: a ring smaller than the corpus silently drops the head.
        visit_capacity: the ring's per-row visit-slot capacity.
        max_games: stop after this many games. For smoke runs; recorded in the provenance so
            a truncated artifact can never read as a whole one.
        split: a `CorpusSplit` selecting one side of a seeded game-level partition, or None
            for the whole corpus. Recorded in the provenance, so a split artifact can never
            read as a whole one either.

    Returns:
        The provenance mapping that was written beside the artifact.

    Raises:
        CorpusEncodeError: the manifest, a record, or the source sha fails its check, or the
            selected side of a split holds no games at all.
    """
    from mantis._engine import Board, HexgBuffer  # noqa: PLC0415 — extension

    record_path, declared_sha = _manifest_pin(dataset_dir)
    actual_sha = sha256_of(record_path)
    if actual_sha != declared_sha:
        raise CorpusEncodeError(
            f"{record_path}: sha256 {actual_sha} != the manifest's {declared_sha}. This is "
            "R279's certification handshake; a corpus that does not match its pin is a "
            "different corpus."
        )
    # The handshake above proves the file is the file the manifest names. It does NOT prove the
    # file is outside the evaluation hold-out set — different properties, and a corpus that
    # passes the first and fails the second is contaminated training data (R327(e)).
    assert_not_heldout_sha(actual_sha, path=record_path)

    from mantis._engine import max_stones  # noqa: PLC0415 — extension

    ceiling = max_stones()
    buf = HexgBuffer(capacity, encoding, visit_capacity)
    games = plies = 0
    rows_over_ceiling = games_truncated = 0
    game_lengths: list[int] = []
    winners = {1: 0, -1: 0}
    hashes: list[str] = []
    for idx, raw in enumerate(_iter_records(record_path)):
        if max_games is not None and games >= max_games:
            break
        game_hash, winner, moves = _require_record(raw, idx)
        if split is not None and not split.selects(game_hash):
            continue
        lost_here = 0
        for row in encode_game(moves, winner, board_factory=(
                lambda: Board.with_encoding_name(encoding)), game_hash=game_hash):
            # THE RING'S STONE CEILING IS CHECKED BEFORE THE PUSH, not caught after it.
            # `push_graph_position` refuses a wider position by raising, and catching that
            # would make the count depend on an error STRING; `len(stones)` against the
            # engine's own `max_stones()` is the same fact read on the near side. Rows are
            # counted, never dropped silently: R328(c) rules the residue a finding.
            if len(row[0]) > ceiling:
                lost_here += 1
                continue
            buf.push_graph_position(*row, game_id=-1)
            plies += 1
        if lost_here:
            games_truncated += 1
            rows_over_ceiling += lost_here
        games += 1
        game_lengths.append(len(moves))
        winners[winner] += 1
        hashes.append(game_hash)

    if split is not None and games == 0:
        raise CorpusEncodeError(
            f"the {split.part!r} side of the seed-{split.seed} / frac-{split.heldout_frac} "
            "split selected ZERO games. An empty ring is not a small ring: a held-out loss "
            "over nothing is a number with no producer, and a training ring of nothing "
            "trains nothing while reporting steps."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf.save_to_path(str(out_path))

    from mantis._engine import registry_sha_hex  # noqa: PLC0415 — extension
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "artifact": out_path.name,
        "artifact_sha256": sha256_of(out_path),
        "encoding": encoding,
        "registry_sha": registry_sha_hex(),
        "source_record_file": record_path.name,
        "source_sha256": actual_sha,
        "source_sha256_declared": declared_sha,
        "games": games,
        "plies": plies,
        "winners": {"p1": winners[1], "p2": winners[-1]},
        # The corpus's own dedupe key (R247/LAW-04), hashed as a SET so the artifact carries
        # a checkable identity for its game population without carrying the population.
        "game_hash_set_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode("utf-8")
        ).hexdigest(),
        "truncated_at_max_games": max_games,
        # NULL when unsplit, so a whole-corpus ring and a split ring are distinguishable by a
        # reader that knows nothing about how either was produced.
        # THE ROW-LEVEL TRUTH, carried so that "8 698 / 8 698 games" can never be read alone.
        # A game whose late positions exceed the ring's fixed-width stone slot is ACCEPTED and
        # TRUNCATED, by ruling (R328 amendment): MAX_STONES stays 256 because the run's own
        # games never reach it, and the price is these rows. `plies` above is what LANDED;
        # `plies_offered` is what the corpus held.
        "max_stones_ceiling": ceiling,
        "plies_offered": plies + rows_over_ceiling,
        "rows_refused_over_max_stones": rows_over_ceiling,
        "games_truncated": games_truncated,
        "ply_histogram_64": _ply_histogram(game_lengths),
        "split_seed": split.seed if split is not None else None,
        "split_heldout_frac": split.heldout_frac if split is not None else None,
        "split_part": split.part if split is not None else None,
        "ring_capacity": capacity,
        "ring_visit_capacity": visit_capacity,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
        ).stdout.strip() or None,
        "written_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    sidecar = out_path.with_name(out_path.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=1, sort_keys=True), encoding="utf-8")
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m mantis.data.bootstrap_encode` — the `data/` producer.

    Raises:
        CorpusEncodeError: propagated from `encode_corpus`; a refusal is the point.
    """
    ap = argparse.ArgumentParser(
        prog="python -m mantis.data.bootstrap_encode",
        description="Encode an audited move-list bootstrap corpus into a graph-path .hexg "
                    "ring with provenance. Selected by nothing; armed by nothing.",
    )
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--capacity", required=True, type=int)
    ap.add_argument("--visit-capacity", required=True, type=int)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed for the game-level held-out partition (with the two below)")
    ap.add_argument("--split-heldout-frac", type=float, default=None,
                    help="held-out share of GAMES, strictly inside (0, 1)")
    ap.add_argument("--split-part", choices=("train", "heldout"), default=None,
                    help="which side of the partition to encode")
    args = ap.parse_args(argv)
    supplied = [args.split_seed, args.split_heldout_frac, args.split_part]
    if any(x is not None for x in supplied) and not all(x is not None for x in supplied):
        ap.error("--split-seed, --split-heldout-frac and --split-part are all-or-none: a "
                 "partially specified split is a partition nobody declared")
    split = (CorpusSplit(args.split_seed, args.split_heldout_frac, args.split_part)
             if args.split_seed is not None else None)
    prov = encode_corpus(
        args.dataset_dir, args.out, encoding=args.encoding, capacity=args.capacity,
        visit_capacity=args.visit_capacity, max_games=args.max_games, split=split,
    )
    print(json.dumps(prov, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
