"""The REAL event sink: a thread-safe, line-buffered JSONL writer.

The segment file is CLAIMED atomically with ``O_CREAT|O_EXCL``, retrying on collision, so no
process start can append to a prior segment. ``run_id`` is validated AT THIS BOUNDARY, because
the schema pattern that would otherwise be the only defence sits behind a caller that does not
exist yet. A serialize or IO failure counts and logs; the emit site never raises, because emits
run on daemon threads where a raise only kills the feeder, and the watchdog reads the counter.
``json`` is imported at MODULE scope so the persist-fatal oracle can patch it there.
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

# The contract version stamped into every segment header line.
EVENT_CONTRACT = "event-manifest-v1"

_SEGMENT_RE = re.compile(r"^events_(?P<run>.+)_seg(?P<seg>\d+)\.jsonl$")

# How many indices a racing construction may walk, so it can never spin forever.
_MAX_SEGMENT_CLAIM_RETRIES = 64


class RunIdError(ValueError):
    """A `run_id` that cannot safely become part of a segment FILENAME."""


def validate_run_id(run_id: str) -> str:
    """Reject, with a `RunIdError`, a `run_id` that would break the segment-filename law: it
    must be one safe path COMPONENT, or the segment index cannot advance."""
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
    """Return the ONE filename convention: ``events_<run_id>_seg<NNNN>.jsonl``."""
    return f"events_{run_id}_seg{segment:04d}.jsonl"


def next_segment_index(log_dir: Path, run_id: str) -> int:
    """Return the run's next segment index — segments are per-``run_id``, so one run's
    resumes never bump another's counter."""
    highest = 0
    if log_dir.is_dir():
        for entry in log_dir.iterdir():
            match = _SEGMENT_RE.match(entry.name)
            if match is not None and match.group("run") == run_id:
                highest = max(highest, int(match.group("seg")))
    return highest + 1


def _claim_segment(directory: Path, run_id: str) -> tuple[int, Path, Any]:
    """ATOMICALLY claim the next segment file; return ``(segment, path, handle)``.

    `O_CREAT|O_EXCL` makes the claim indivisible, so exactly one racer creates a file and a loser
    re-scans; a scan-then-``open(path, "a")`` is a TOCTOU that put two headers from two pids into
    one file under 12 concurrent starts. The file is brand new, so never-append is structural.
    """
    last_exc: OSError | None = None
    for _ in range(_MAX_SEGMENT_CLAIM_RETRIES):
        segment = next_segment_index(directory, run_id)
        path = directory / segment_filename(run_id, segment)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:      # claimed by another process between scan and claim
            last_exc = exc
            continue
        return segment, path, os.fdopen(fd, "w", buffering=1, encoding="utf-8")
    raise OSError(
        f"could not claim an event segment for run_id={run_id!r} under {directory} after "
        f"{_MAX_SEGMENT_CLAIM_RETRIES} attempts"
    ) from last_exc


class JsonlEventSink:
    """Append-only JSONL event sink for ONE run segment. Construction claims the segment and
    writes its header, and RAISES on failure; every failure after construction is counted."""

    def __init__(self, *, log_dir: Path | str, run_id: str) -> None:
        self._run_id = validate_run_id(str(run_id))
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.persist_errors_total: int = 0
        # Re-entrant: the failure counter is bumped inside the write critical section.
        self._lock = threading.RLock()
        self._closed = False
        # Line-buffered, so a later `os._exit` cannot lose an already-emitted line.
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
        """Return the segment file this sink writes."""
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def segment(self) -> int:
        return self._segment

    def emit(self, event: Mapping[str, Any]) -> None:
        """Serialize ``event`` and append it as ONE line, stamping ``ts`` only when absent.

        A payload with no ``"event"`` key raises `ValueError`: a producer bug, not a persistence
        failure, so the counter does not move.
        """
        if "event" not in event:
            # Log BEFORE raising: emits run on daemon feeder threads, where an uncaught
            # exception kills the thread silently and the traceback goes nowhere.
            _LOG.error("event_sink_missing_event_key keys=%s thread=%s — the emitting thread "
                       "will die on this ValueError",
                       sorted(event), threading.current_thread().name)
            raise ValueError(f"event payload must carry an 'event' key, got keys {sorted(event)}")
        payload = dict(event)
        if "ts" not in payload:
            payload["ts"] = time.time()
        try:
            line = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001 — count, log, never raise here
            self._count_persist_error("serialize", payload.get("event"), exc)
            return
        with self._lock:
            try:
                self._fh.write(line)
            except Exception as exc:  # noqa: BLE001 — the watchdog makes it fatal
                self._count_persist_error("write", payload.get("event"), exc)

    def close(self) -> None:
        """Flush and close; a close failure is counted exactly like a write failure."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.close()
            except Exception as exc:  # noqa: BLE001 — counted, not raised
                self._count_persist_error("close", None, exc)

    def _count_persist_error(self, stage: str, event_name: Any, exc: BaseException) -> None:
        """Count and LOUDLY log a persistence failure: the watchdog reads this counter."""
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
