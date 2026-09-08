# >300 justify (R8): the format, its writer, its reader and the two builders that turn a
# played game into one are ONE unit. A record shape defined apart from the writer that stores
# it drifts from it, and a reader defined apart from both is a second authority over the same
# bytes — which is exactly the failure this store exists to make impossible for a run's games.
"""The GAME RECORD store: every game a run plays, written where it can be read back.

R344(b). A run that does not write its games cannot be viewed, replayed, or mined, and the
producer must exist at step 0 — so this is a WRITER that ships with the run, not a report
generated afterwards from something else.

WHAT IT IS NOT. It is not a second event stream. `game_complete` already carries a self-play
game's move list into the JSONL event channel and keeps doing so; that stream is keyed by
TIME and mixes forty event kinds, which is the wrong shape for "show me game 1 837". This
store is keyed by GAME, one record per line, with an index over its shards.

FORMAT: JSONL, by the ruling's own conditional — *"length-delimited msgpack (falls back to
JSONL if msgpack is not already a dependency — no new hard dependency for this)"*. `msgpack`
is not in `pyproject.toml`'s dependencies or its one extra, so the fallback fires and there is
nothing here to decide.

SHARDS ARE KEYED BY (run, SEGMENT, hour), NOT BY (run, hour), and the segment is not
decoration. `monitor/sink.py` claims its event segments with `O_CREAT|O_EXCL` so that no file
ever spans two run segments — a law it earned when a scan-then-append TOCTOU put two process
headers in one file. Keying a game shard on the hour alone would reintroduce exactly that: a
resume landing in the same wall-clock hour would append to the stopped process's shard. The
hour rotation the ruling asks for therefore happens WITHIN a segment.

FAILURE POSTURE, matching the two neighbours that already ruled it. Construction RAISES: a
store that cannot open at boot is a loud startup error, not a mid-run surprise (the
`JsonlEventSink` posture). After construction, a write failure increments
`persist_errors_total`, logs an ERROR and disables the store — the eval progress writer's
posture under R319(e)(ii), for its reason: losing the record of a game must not kill a run
that is otherwise healthy, and the counter is what makes the loss visible rather than silent.

READING IS PART OF THE FORMAT, so it lives here. A reader in another module is a second
authority over the same bytes, and the one thing this store must survive is being read while
it is being written: the last line of a live shard is routinely a partial line. `read_shard`
SKIPS a trailing partial record and counts it; it does not raise, and it does not silently
treat the file as ending cleanly.
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
    """`max(existing segment index for run_id) + 1`, 1 when the run has no shard yet.

    Segments are per-`run_id`, exactly as the event sink's are: one run's resumes never
    advance another run's counter.
    """
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

    # -- shard lifecycle ----------------------------------------------------------------
    def _claim(self, hour: str, *, segment: int | None = None) -> tuple[int, str, Path, Any]:
        """Atomically claim a shard for `hour`; return its open handle.

        The claim is `O_CREAT|O_EXCL` for the event sink's reason: two processes starting at
        the same instant must not both believe they own one file. A loser re-scans and
        advances its SEGMENT, so the two runs' games never interleave in one shard.

        `segment` is REUSED on an hour rotation and only scanned for at construction, and the
        difference is not cosmetic. A run writes from TWO processes into one directory — the
        trainer continuously, and each eval round's child for its own lifetime — so a trainer
        that re-scanned at every rotation would step over whatever segment the last round's
        child took, and its own shards would carry three segment numbers across three hours.
        The segment names the WRITER; the hour names the window. Re-scanning conflates them,
        and there is no collision to avoid: this writer only ever rotates FORWARD, so
        `(its own segment, a new hour)` is a filename nothing can already hold.
        """
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
        """fsync, close, and append this shard's row to the index. Order is load-bearing:
        the index may only name a shard whose bytes are already on the platter."""
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

    # -- the write path -----------------------------------------------------------------
    def write(self, record: Mapping[str, Any]) -> None:
        """Append one game record, rotating the shard when the UTC hour has turned.

        Never raises: a lost game record must not end a healthy run. A failure disables the
        store, counts, and logs — `persist_errors_total` is the observable that makes the
        loss loud (LAW-18), and a disabled store stays disabled rather than retrying into a
        full volume once per game.
        """
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


# --------------------------------------------------------------------------------------- #
# Reading — same module, because the format has one authority
# --------------------------------------------------------------------------------------- #
def read_shard(path: Path | str) -> tuple[list[dict[str, Any]], int]:
    """Return `(records, skipped)` from one shard, tolerating a partial trailing line.

    A shard being written has no closing guarantee on its last line, and a shard whose
    process was killed mid-write keeps that partial line forever. Both are the SAME shape to
    a reader and neither is fatal: the malformed line is skipped and COUNTED, so a caller can
    tell "one torn tail" from "this file is garbage" — the distinction a bare `try: continue`
    throws away.

    Header/footer rows (`record` in {shard_opened, shard_closed}) are not games and are not
    returned.
    """
    records: list[dict[str, Any]] = []
    skipped = 0
    # BINARY, decoded per line. Text mode would raise `UnicodeDecodeError` on a line torn
    # mid-character and take the whole file with it — the one outcome this function exists to
    # prevent. `json.dumps` defaults to `ensure_ascii=True` so a torn line is ASCII today and
    # the case is unreachable; relying on that would make the guarantee depend on a default in
    # another module. A line that will not decode is skipped and counted like any other.
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


# --------------------------------------------------------------------------------------- #
# The record shape — ONE authority, two producers
# --------------------------------------------------------------------------------------- #
#: Self-play's Rust terminal codes, already mapped to these names by `pool_drain`.
#: Repeated here as the record's declared vocabulary rather than imported, because a record
#: written to disk must not change meaning when a producer's internal mapping does.
TERMINATIONS = ("six_in_a_row", "colony", "ply_cap", "other_draw", "unknown")


def _axial(moves: Any) -> list[list[int]]:
    """`[(q, r), ...]` -> `[[q, r], ...]`, one entry per PLY.

    PLIES, not turns: a turn places two stones (the first turn one), so the move list is the
    flat ply sequence and a viewer derives the turn grouping from its own index. Storing the
    grouping would be storing a derivation, and the corpus pipeline's `GameRecord.moves` is
    already the flat sequence — this is the field the two formats share.
    """
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
    """One self-play game as a record.

    `step` is the ACTOR step — the training step whose weights played this game, forwarded by
    `ActorSync` through `WorkerPool.update_checkpoint_step`. It is NOT the learner's live
    step, and the record says which it is in `step_kind` rather than leaving a reader to
    assume (LAW-03: verify the measurement unit before framing anything on it).

    `colors` is ABSENT here, deliberately. On the eval channels it says which seat the
    candidate held; in self-play both seats are the same net, so there is nothing for it to
    say and a `{"p1": 1, "p2": -1}` tautology would read like information (R4's "absent is
    not zero", applied to a field).

    `search_stats` is likewise absent on this channel and it is a GAP, not a nothing: the
    per-position visit distribution exists in the engine and reaches the replay ring, but
    every row is pushed `game_id=-1` by construction, so no position can be attributed to a
    game without an engine change on the hot drain path. `CARD-GAME-RECORD-SELFPLAY-STATS`
    names the producer that would fill it.
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
        # LAW-04's dedupe input, carried so effective-n can be counted off the RECORD rather
        # than recomputed from the moves by every later consumer.
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
    """One eval-channel game as a record.

    `step` is the ROUND's step (`RoundSpec.step`) — the learner step the candidate snapshot
    was cut at — so `step_kind` distinguishes it from self-play's actor step.

    `result` is SEAT-relative (`p1`/`p2`/`draw`, p1 being the side that moved first) and
    `colors` says which seat the candidate held. The arena's own `winner` is
    candidate-relative; converting here rather than storing both keeps one fact in one field,
    and a viewer that renders a board needs the seat, not the role.

    `search_stats` is present when the candidate's search exposed its root — the visit
    distribution and root value the deploy head computes and, before R344(b), discarded one
    line before returning its move.
    """
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
    """Arena `winner` (`candidate`/`opponent`/`draw`) -> seat result (`p1`/`p2`/`draw`).

    `candidate_color` is `1` when the candidate moved first. A viewer draws seats, not roles,
    so the record stores the seat and `colors` recovers the role.

    Raises:
        KeyError: `winner` is not one of the arena's three values.
    """
    if winner == "draw":
        return "draw"
    role_is_first = (candidate_color == 1) == (winner == "candidate")
    return "p1" if role_is_first else "p2"
