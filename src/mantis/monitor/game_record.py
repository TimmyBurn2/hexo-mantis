# >300 justify (R8): the format, its writer, its reader and the two builders that turn a played
# game into one are ONE unit. A record shape defined apart from its writer drifts from it, and a
# reader defined apart from both is a second authority over the same bytes.
"""The GAME RECORD store: every game a run plays, written where it can be read back.

A run that does not write its games cannot be viewed, replayed or mined, and the producer must
exist at step 0 — so this is a WRITER that ships with the run, not a later report. It is NOT a
second event stream: `game_complete` still carries move lists into the JSONL event channel, but
that stream is keyed by TIME, while this one is keyed by GAME with an index over its shards.

SHARDS ARE KEYED BY (run, SEGMENT, hour): keying on the hour alone would let a resume in the
same wall-clock hour append to the stopped process's shard. Construction RAISES, but after it a
write failure increments `persist_errors_total`, logs and disables the store, because losing a
record must not kill a healthy run. READING lives here too: `read_shard` SKIPS a partial
trailing record and COUNTS it.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mantis.monitor.sink import validate_run_id

_LOG = logging.getLogger(__name__)

#: Stamped into every shard's header line. Bumping it is a format change.
GAME_RECORD_CONTRACT = "game-record-v1"

#: `games_<run_id>_seg<NNNN>_<YYYYMMDDHH>.jsonl`
_SHARD_RE = re.compile(
    r"^games_(?P<run>.+)_seg(?P<seg>\d+)_(?P<hour>\d{10})\.jsonl$"
)

_MAX_SHARD_CLAIM_RETRIES = 64

#: The four channels a game can come from (R344(b) names all four).
CHANNELS = ("selfplay", "promotion", "external", "random_floor")


class GameRecordError(RuntimeError):
    """A shard could not be claimed at construction — a loud startup failure."""


def shard_filename(run_id: str, segment: int, hour: str) -> str:
    """The ONE filename convention: `games_<run_id>_seg<NNNN>_<YYYYMMDDHH>.jsonl`."""
    return f"games_{run_id}_seg{segment:04d}_{hour}.jsonl"


def index_filename(run_id: str) -> str:
    """The per-run shard index: one JSON line per CLOSED shard."""
    return f"games_{run_id}_index.jsonl"


def next_segment_index(record_dir: Path, run_id: str) -> int:
    """`max(segment index for run_id) + 1`, or 1; segments are per-`run_id`."""
    highest = 0
    if record_dir.is_dir():
        for entry in record_dir.iterdir():
            match = _SHARD_RE.match(entry.name)
            if match is not None and match.group("run") == run_id:
                highest = max(highest, int(match.group("seg")))
    return highest + 1


def _utc_hour(when: float | None = None) -> str:
    moment = datetime.now(tz=UTC) if when is None else datetime.fromtimestamp(when, tz=UTC)
    return moment.strftime("%Y%m%d%H")


class GameRecordWriter:
    """Append-only game records for ONE run segment, rotating on the UTC hour.

    Raises:
        RunIdError: `run_id` cannot safely become part of a filename.
        GameRecordError: no shard could be claimed at construction.
    """

    def __init__(self, *, record_dir: Path | str, run_id: str) -> None:
        self._run_id = validate_run_id(str(run_id))
        self._dir = Path(record_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / index_filename(self._run_id)
        self.persist_errors_total: int = 0
        self.games_written: int = 0
        self._disabled = False
        self._segment, self._hour, self._path, self._handle = self._claim(_utc_hour())
        self._shard_games = 0
        self._opened_at = datetime.now(tz=UTC).isoformat()

    def _claim(self, hour: str, *, segment: int | None = None) -> tuple[int, str, Path, Any]:
        """Atomically claim a shard for `hour` and return its open handle. `O_CREAT|O_EXCL`, so
        two processes starting at one instant cannot both own a file and a loser advances its
        SEGMENT. `segment` is scanned for only at construction, because a run writes from TWO
        processes into one directory and re-scanning each rotation would step over the last eval
        child's segment."""
        last_exc: OSError | None = None
        for _ in range(_MAX_SHARD_CLAIM_RETRIES):
            claim = next_segment_index(self._dir, self._run_id) if segment is None else segment
            path = self._dir / shard_filename(self._run_id, claim, hour)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError as exc:
                last_exc = exc
                continue
            handle = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
            handle.write(json.dumps({
                "record": "shard_opened", "contract": GAME_RECORD_CONTRACT,
                "run_id": self._run_id, "segment": claim, "hour": hour,
            }, sort_keys=True) + "\n")
            return claim, hour, path, handle
        raise GameRecordError(
            f"could not claim a game shard for run_id={self._run_id!r} under {self._dir} "
            f"after {_MAX_SHARD_CLAIM_RETRIES} attempts"
        ) from last_exc

    def _close_shard(self) -> None:
        """fsync, close and index this shard; the index may only name bytes already on disk."""
        if self._handle.closed:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError as exc:                       # pragma: no cover - platform-dependent
            self._note_failure(exc, "fsync")
        finally:
            self._handle.close()
        row = {
            "record": "shard_closed", "contract": GAME_RECORD_CONTRACT,
            "run_id": self._run_id, "segment": self._segment, "hour": self._hour,
            "shard": self._path.name, "games": self._shard_games,
            "bytes": self._path.stat().st_size if self._path.exists() else 0,
            "opened_utc": self._opened_at,
            "closed_utc": datetime.now(tz=UTC).isoformat(),
        }
        try:
            with self._index_path.open("a", encoding="utf-8") as index:
                index.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            self._note_failure(exc, "index")

    def _note_failure(self, exc: BaseException, where: str) -> None:
        self.persist_errors_total += 1
        _LOG.error("game_record_%s_failed path=%s: %r", where, self._path, exc)

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one game record, rotating the shard on the UTC hour. Never raises: a failure
        disables the store, counts and logs, so a lost record cannot end a healthy run."""
        if self._disabled:
            return
        try:
            hour = _utc_hour()
            if hour != self._hour:
                self._close_shard()
                self._segment, self._hour, self._path, self._handle = self._claim(
                    hour, segment=self._segment)
                self._shard_games = 0
                self._opened_at = datetime.now(tz=UTC).isoformat()
            self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        except (OSError, TypeError, ValueError, GameRecordError) as exc:
            self._disabled = True
            self._note_failure(exc, "write")
            return
        self._shard_games += 1
        self.games_written += 1

    def close(self) -> None:
        """Close the open shard and index it. Idempotent."""
        if self._disabled and self._handle.closed:
            return
        self._close_shard()

    @property
    def shard_path(self) -> Path:
        return self._path

    @property
    def index_path(self) -> Path:
        return self._index_path


def read_shard(path: Path | str) -> tuple[list[dict[str, Any]], int]:
    """Return `(records, skipped)` from one shard, tolerating a partial trailing line: a live
    shard has no closing guarantee on its last line and a killed process leaves one forever, so
    the malformed line is skipped and COUNTED. Header/footer rows are not games."""
    records: list[dict[str, Any]] = []
    skipped = 0
    # BINARY, decoded per line: text mode would raise `UnicodeDecodeError` on a line torn
    # mid-character and take the whole file with it, which is what this function prevents.
    with Path(path).open("rb") as handle:
        for raw in handle:
            try:
                stripped = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                skipped += 1
                continue
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(parsed, dict):
                skipped += 1
                continue
            if parsed.get("record") in ("shard_opened", "shard_closed"):
                continue
            records.append(parsed)
    return records, skipped


def iter_run_games(record_dir: Path | str, run_id: str) -> Iterator[dict[str, Any]]:
    """Every game of `run_id`, in shard order (segment, then hour, then file order)."""
    directory = Path(record_dir)
    shards: list[tuple[int, str, Path]] = []
    if directory.is_dir():
        for entry in directory.iterdir():
            match = _SHARD_RE.match(entry.name)
            if match is not None and match.group("run") == run_id:
                shards.append((int(match.group("seg")), match.group("hour"), entry))
    for _segment, _hour, path in sorted(shards):
        records, _skipped = read_shard(path)
        yield from records


#: Self-play's Rust terminal codes, already mapped by `pool_drain`. Repeated here as the record's
#: declared vocabulary, so a record on disk cannot change meaning when a producer's mapping does.
TERMINATIONS = ("six_in_a_row", "colony", "ply_cap", "other_draw", "unknown")


def _axial(moves: Any) -> list[list[int]]:
    """`[(q, r), ...]` -> `[[q, r], ...]`, one entry per PLY rather than per turn, which is the
    flat sequence the corpus pipeline's `moves` already uses."""
    return [[int(q), int(r)] for q, r in moves]


def selfplay_record(
    *,
    game_id: str,
    run_id: str,
    step: int,
    moves: Any,
    result: str,
    plies: int,
    termination: str,
    worker_id: int,
    seed: int,
    served_sims: int,
    game_id_byte_hash: str | None = None,
) -> dict[str, Any]:
    """Build one self-play game as a record. `step` is the ACTOR step — the weights that played
    this game, not the learner's live step — and `step_kind` says so. `colors` is ABSENT because
    both seats are the same net. `search_stats` is a GAP, not a nothing: the visit distribution
    reaches the replay ring but every row is pushed `game_id=-1`.
    """
    record: dict[str, Any] = {
        "contract": GAME_RECORD_CONTRACT,
        "game_id": game_id,
        "run_id": run_id,
        "channel": "selfplay",
        "step": int(step),
        "step_kind": "actor",
        "worker_id": int(worker_id),
        "seed": int(seed),
        "served_sims": int(served_sims),
        "plies": int(plies),
        "result": result,
        "termination": termination,
        "moves": _axial(moves),
    }
    if game_id_byte_hash is not None:
        # LAW-04's dedupe input, carried so effective-n is counted off the RECORD.
        record["game_id_byte_hash"] = game_id_byte_hash
    return record


def eval_record(
    *,
    game_id: str,
    run_id: str,
    step: int,
    channel: str,
    rung: str,
    phase: str,
    game_index: int,
    moves: Any,
    result: str,
    plies: int,
    termination: str,
    candidate_color: int,
    seed: int,
    served_sims: int,
    trajectory_hash: str | None = None,
    search_stats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one eval-channel game as a record. `step` is the ROUND's step, distinguished by
    `step_kind` from self-play's actor step; `result` is SEAT-relative and `colors` says which
    seat the candidate held, because a viewer needs the seat, not the role."""
    if channel not in CHANNELS:
        raise ValueError(
            f"eval_record: channel {channel!r} is not one of {CHANNELS} — an unnamed channel "
            f"would land in the store and be invisible to every filter that reads it"
        )
    record: dict[str, Any] = {
        "contract": GAME_RECORD_CONTRACT,
        "game_id": game_id,
        "run_id": run_id,
        "channel": channel,
        "rung": rung,
        "phase": phase,
        "step": int(step),
        "step_kind": "round",
        "game_index": int(game_index),
        "colors": {"candidate": int(candidate_color), "opponent": -int(candidate_color)},
        "seed": int(seed),
        "served_sims": int(served_sims),
        "plies": int(plies),
        "result": result,
        "termination": termination,
        "moves": _axial(moves),
    }
    if trajectory_hash is not None:
        record["trajectory_hash"] = trajectory_hash
    if search_stats:
        record["search_stats"] = search_stats
    return record


def seat_result(winner: str, candidate_color: int) -> str:
    """Convert an arena `winner` to a seat result; `candidate_color` is `1` when the candidate
    moved first.

    Raises:
        KeyError: `winner` is not one of the arena's three values.
    """
    if winner == "draw":
        return "draw"
    role_is_first = (candidate_color == 1) == (winner == "candidate")
    return "p1" if role_is_first else "p2"
