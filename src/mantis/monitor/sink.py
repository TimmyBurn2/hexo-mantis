"""The REAL event sink: a thread-safe line-buffered JSONL writer (WP13-A §c.1).

Ports `hexo_rl/monitoring/events.py::JSONLSink` with three structural changes:

* **rotation-on-resume** (§11 log identity): the segment file is CLAIMED atomically at
  construction with ``O_CREAT|O_EXCL`` over ``max(existing segments for run_id) + 1``,
  retrying on collision, so no process start can ever append to a prior segment and no
  JSONL file ever spans two run segments — not even when N processes start at the same
  instant (a scan-then-``open("a")`` was a TOCTOU: two racers claimed one file and wrote
  two ``run_segment_started`` headers into it). ``run_id`` is validated AT THIS BOUNDARY
  (path separators / ``..`` / control chars / empty), because the schema pattern that
  would otherwise be the only defence sits behind a caller that does not exist yet;
* **LAW-14**: the old triple ``except: pass`` dies. A serialize/IO failure increments
  ``persist_errors_total`` and logs an ERROR; the emit site never raises (emits run on
  daemon threads where a raise would only kill the feeder) — run-fatality is delivered by
  the watchdog observing the counter from ANY thread;
* the global renderer registry (``register_renderer``/``emit_event`` fan-out) dies with
  the injected-single-sink seam: this class implements `mantis.train.emit.EventSink`
  structurally (one ``emit``) and is INJECTED, never imported by producers.

``json`` is imported at MODULE scope on purpose: the persist-fatal oracle patches
``mantis.monitor.sink.json`` to inject a serialization failure.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# The seam-7 contract version stamped into every segment header line.
EVENT_CONTRACT = "event-manifest-v1"

_SEGMENT_RE = re.compile(r"^events_(?P<run>.+)_seg(?P<seg>\d+)\.jsonl$")

# How many segment indices a racing construction may walk before giving up. A collision
# means another process claimed that index between our scan and our claim; the loop is
# bounded so a pathological directory can never spin forever.
_MAX_SEGMENT_CLAIM_RETRIES = 64


class RunIdError(ValueError):
    """A `run_id` that cannot safely become part of a segment FILENAME."""


def validate_run_id(run_id: str) -> str:
    """Reject a `run_id` that would break the segment-filename law, LOUD.

    The filename is `events_<run_id>_seg<NNNN>.jsonl`, so the id must be a single safe path
    COMPONENT. Rejected: empty (the segment regex cannot match an empty run token, so the
    index never advances and every start appends to one file), anything containing a path
    separator or `..` (the file escapes `log_dir` and the sibling scan never sees it, so
    again the index never advances), NUL/control characters, and leading/trailing
    whitespace. `config/schema.py::run_id` carries a stricter pattern, but it is enforced by
    a caller that does not exist yet — an absolute law may not rest on that.
    """
    if not run_id:
        raise RunIdError("run_id must be a non-empty string (an empty run_id makes every "
                         "process start append to ONE segment file)")
    if run_id != run_id.strip():
        raise RunIdError(f"run_id {run_id!r} has leading/trailing whitespace")
    if "/" in run_id or "\\" in run_id or os.sep in run_id or (os.altsep and os.altsep in run_id):
        raise RunIdError(f"run_id {run_id!r} contains a path separator")
    if run_id == "." or run_id == ".." or ".." in run_id:
        raise RunIdError(f"run_id {run_id!r} contains a parent-directory reference")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in run_id):
        raise RunIdError(f"run_id {run_id!r} contains a control character")
    return run_id


def segment_filename(run_id: str, segment: int) -> str:
    """The ONE filename convention: ``events_<run_id>_seg<NNNN>.jsonl``."""
    return f"events_{run_id}_seg{segment:04d}.jsonl"


def next_segment_index(log_dir: Path, run_id: str) -> int:
    """``max(existing segment index for run_id) + 1`` (1 when the run has no segment yet).

    Segments are per-``run_id``: one run's resumes never bump another run's counter.
    """
    highest = 0
    if log_dir.is_dir():
        for entry in log_dir.iterdir():
            match = _SEGMENT_RE.match(entry.name)
            if match is not None and match.group("run") == run_id:
                highest = max(highest, int(match.group("seg")))
    return highest + 1


def _claim_segment(directory: Path, run_id: str) -> tuple[int, Path, Any]:
    """ATOMICALLY claim the next segment file; return ``(segment, path, handle)``.

    `O_CREAT|O_EXCL` makes the claim indivisible: exactly one racer can create a given
    segment file, and a loser re-scans and advances. The previous
    scan-then-``open(path, "a")`` was a TOCTOU — under 12 concurrent constructions three
    files ended up with TWO ``run_segment_started`` headers from two pids, i.e. a JSONL file
    spanning two run segments, which §11 forbids absolutely.

    The handle is opened write-only + line-buffered; because the file is brand new by
    construction, "append" and "write" are the same thing and the never-append law is
    structural rather than conventional.
    """
    last_exc: OSError | None = None
    for _ in range(_MAX_SEGMENT_CLAIM_RETRIES):
        segment = next_segment_index(directory, run_id)
        path = directory / segment_filename(run_id, segment)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:      # another process claimed it between scan+claim
            last_exc = exc
            continue
        return segment, path, os.fdopen(fd, "w", buffering=1, encoding="utf-8")
    raise OSError(
        f"could not claim an event segment for run_id={run_id!r} under {directory} after "
        f"{_MAX_SEGMENT_CLAIM_RETRIES} attempts"
    ) from last_exc


class JsonlEventSink:
    """Append-only JSONL event sink for ONE run segment.

    Construction claims the next segment file and writes its ``run_segment_started``
    header. A construction failure raises (an un-openable sink is a loud startup error,
    not a mid-run persistence failure); every failure AFTER construction is counted.
    """

    def __init__(self, *, log_dir: Path | str, run_id: str) -> None:
        self._run_id = validate_run_id(str(run_id))
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.persist_errors_total: int = 0
        # Re-entrant: the failure counter is bumped from inside the write critical section.
        self._lock = threading.RLock()
        self._closed = False
        # Line-buffered: a later `os._exit` (the watchdog fire path) cannot lose an
        # already-emitted line.
        self._segment, self._path, self._fh = _claim_segment(directory, self._run_id)
        self.emit(
            {
                "event": "run_segment_started",
                "run_id": self._run_id,
                "segment": self._segment,
                "pid": os.getpid(),
                "created_utc": datetime.now(UTC).isoformat(),
                "contract": EVENT_CONTRACT,
            }
        )

    @property
    def path(self) -> Path:
        """The segment file this sink writes."""
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def segment(self) -> int:
        return self._segment

    def emit(self, event: Mapping[str, Any]) -> None:
        """Serialize ``event`` and append it as ONE line.

        A missing ``"event"`` key is a producer BUG → loud ``ValueError`` (it is not a
        persistence failure, so the counter does not move). ``ts`` is stamped iff absent —
        a producer-supplied ``ts`` wins, which is parity with the old
        ``{"ts": time.time(), **payload}`` later-key-wins funnel.
        """
        if "event" not in event:
            # LOG BEFORE RAISING (RED-TEAM F13): emits happen on daemon feeder threads where
            # an uncaught exception kills that thread SILENTLY — the traceback goes nowhere
            # and the only remaining signal is the 1800 s staleness deadline. The raise is
            # still the right contract (a missing name is a producer bug, not a persistence
            # failure, so the counter must not move), but its cost must be visible.
            _LOG.error("event_sink_missing_event_key keys=%s thread=%s — the emitting thread "
                       "will die on this ValueError",
                       sorted(event), threading.current_thread().name)
            raise ValueError(f"event payload must carry an 'event' key, got keys {sorted(event)}")
        payload = dict(event)
        if "ts" not in payload:
            payload["ts"] = time.time()
        try:
            line = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001 — LAW-14: count, log, never raise here
            self._count_persist_error("serialize", payload.get("event"), exc)
            return
        with self._lock:
            try:
                self._fh.write(line)
            except Exception as exc:  # noqa: BLE001 — LAW-14: the watchdog makes it fatal
                self._count_persist_error("write", payload.get("event"), exc)

    def close(self) -> None:
        """Flush + close; a close failure is counted exactly like a write failure."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.close()
            except Exception as exc:  # noqa: BLE001 — LAW-14
                self._count_persist_error("close", None, exc)

    def _count_persist_error(self, stage: str, event_name: Any, exc: BaseException) -> None:
        """LAW-14: a persistence failure is COUNTED (the watchdog's `counters_fn` reads
        this) and logged LOUD — never swallowed, never re-raised at the emit site."""
        with self._lock:
            self.persist_errors_total += 1
        _LOG.error(
            "event_sink_persist_error stage=%s event=%r path=%s total=%d exc=%r",
            stage,
            event_name,
            self._path,
            self.persist_errors_total,
            exc,
        )
