# >300 justify: the replay, the record stamp and the provenance the artifact is worthless
# without are ONE producer over ONE external contract. Splitting the encoder from the
# provenance would let an artifact be written whose sidecar was computed elsewhere.
"""Encode an audited move-list bootstrap corpus into GRAPH-PATH training records.

The corpus audit CERTIFIES axial move lists and that certification is prereg grounds, but the
encoder between them did not exist in-tree. This is it, for the GRAPH arch, and it is
CAPABILITY, NOT POSTURE: no config selects it and no production path reaches it.

One row per PLY, pushed through the production `HexgBuffer.push_graph_position` — the same ring
self-play writes and the trainer samples. The POSITION is replayed on a production `Board`, one
`apply_move` per stone, because the corpus is per-STONE; the POLICY TARGET is a one-hot, since
BC has no visit distribution and the engine refuses a `visits` row that is not one; the VALUE
TARGET is `mantis._engine.graph_row_outcome` itself, because a transcription would drift.

It REFUSES loudly and never coerces: an illegal move, a winner outside `{+1, -1}`, an empty
move list, a missing field, or a sha256 disagreeing with the manifest.
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

#: The runner's terminal-reason code for a normal decided/drawn end. `2` is the ply-cap branch,
#: which a completed human game never takes.
_TERMINAL_DECIDED = 0

#: Values the decided branch never reads. Named rather than passed as bare zeros so a reader
#: can see they are inert, and so a future draw-aware corpus has one place to change.
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
    """The audit's contract v2, re-checked at the point of USE — the audit ran over a dataset
    directory, this runs over whatever the caller hands it.

    Raises:
        CorpusEncodeError: any required field absent, of the wrong type, or out of range.
    """
    if not isinstance(rec, dict):
        raise CorpusEncodeError(f"record {idx}: not a JSON object ({type(rec).__name__})")
    game_hash = rec.get("game_hash")
    if not isinstance(game_hash, str) or not game_hash:
        raise CorpusEncodeError(f"record {idx}: `game_hash` absent or not a non-empty string")
    winner = rec.get("winner")
    # `isinstance(..., bool)` FIRST: `True == 1` in Python, so a bare membership test admits a
    # boolean winner and stamps every row of that game with a real sign.
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
    """Yield one `push_graph_position` row per ply of one completed game, in the ENGINE'S
    POSITIONAL ORDER — what `pool_push.push_graph` forwards verbatim, so a row from here and one
    from self-play are the same object to the ring.

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
            # TRUE, decided by the flag's ROLE and not its NAME: its only semantic consumer
            # turns it into the row's POLICY WEIGHT, and `recency_buffer` defaults it to 1. A
            # BC row's one-hot IS a policy target, and FALSE zeroed the policy loss and its
            # denominator on EVERY row — measured 0.000000 held-out policy loss across 2 000
            # steps over 511 145 human positions.
            True,                              # is_full_search: the policy target is learnable

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
    """The single record file and its declared sha256, from the audit's manifest. Both declared
    shapes are accepted and EXACTLY ONE must match, which is the audit's own rule.

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
    """Game-length histogram in `bucket`-ply bins. Carried in the provenance because the
    truncation's SHAPE is what a reader needs: a single "88 games truncated" cannot show that
    the loss is the late phase of the longest games."""
    out: dict[str, int] = {}
    width = max((len(str(n)) for n in lengths), default=1)
    for n in lengths:
        lo = (n // bucket) * bucket
        # ZERO-PADDED so lexicographic order IS numeric order: the provenance is written with
        # `sort_keys=True`, which otherwise puts "64-127" after "512-575".
        key = f"{lo:0{width}d}-{lo + bucket - 1:0{width}d}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class CorpusSplit:
    """A seeded, GAME-level partition of the corpus into `train` and `heldout`.

    ONE object rather than three parameters, so the split cannot be half-specified. BY GAME AND
    NOT BY PLY: a ply-level split puts positions from the SAME game on both sides. Assignment is
    a KEYED HASH of the game's own identity, so it is independent of record order and of
    `max_games`, reproducible from the seed alone, and can never split a game.
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
        out_path: the `.hexg` artifact; its provenance sidecar is written beside it.
        encoding: the registered graph encoding the ring is bound to.
        capacity: the ring's record capacity, sized by the CALLER — a ring smaller than the
            corpus silently drops the head.
        visit_capacity: the ring's per-row visit-slot capacity.
        max_games: stop after this many games; recorded so a truncated artifact cannot read as
            a whole one. `split` is recorded likewise.
        split: one side of a seeded game-level partition, or None for the whole corpus.

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
    # The handshake proves the file is the file the manifest names. It does NOT prove the file
    # is outside the evaluation hold-out set — passing the first and failing the second is
    # contaminated training data.
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
            # THE STONE CEILING IS CHECKED BEFORE THE PUSH: catching `push_graph_position`'s
            # refusal would make the count depend on an error STRING, while `len(stones)`
            # against `max_stones()` is the same fact on the near side. Rows are counted,
            # never dropped silently.
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
        # The corpus's own dedupe key, hashed as a SET so the artifact carries a checkable
        # identity for its game population without carrying the population.
        "game_hash_set_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode("utf-8")
        ).hexdigest(),
        "truncated_at_max_games": max_games,
        # NULL when unsplit, so a whole-corpus ring and a split ring are distinguishable by a
        # reader who knows nothing about how either was produced.
        # THE ROW-LEVEL TRUTH, so "8 698 / 8 698 games" can never be read alone: a game whose
        # late positions exceed the fixed-width stone slot is ACCEPTED and TRUNCATED by ruling.
        # `plies` is what LANDED; `plies_offered` is what the corpus held.
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
