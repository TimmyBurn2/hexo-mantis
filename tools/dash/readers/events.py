"""The event-stream tail: every segment in order from its byte offset, a torn tail held until its newline arrives."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .series import Reducers, Snapshot

SEGMENT_RE = re.compile(r"^events_(?P<run>.+)_seg(?P<seg>\d{4,})\.jsonl$")
#: Bytes read per call so a first read of a 166 MB record peaks near this, not near the file's size.
CHUNK_BYTES = 8 * 1024 * 1024


class EmptyRunRecord(RuntimeError):
    """No segment has yielded one parseable row — refuse, a page from nothing looks like a clean run."""


def record_run_ids(logs_dir: Path) -> list[str]:
    """The run ids that have event segments under `logs_dir`, sorted; OSError propagates."""
    if not logs_dir.is_dir():
        return []
    return sorted({m.group("run") for e in logs_dir.iterdir() if (m := SEGMENT_RE.match(e.name)) is not None})


class EventTail:
    """Tails `events_<run_id>_seg*.jsonl` under `logs_dir` into one `Reducers`; call `poll()` repeatedly."""

    def __init__(self, logs_dir: Path | str, run_id: str) -> None:
        self.logs_dir = Path(logs_dir)
        self.run_id = run_id
        self.segments: list[tuple[int, Path]] = []
        self.offsets: dict[Path, int] = {}
        self.held: dict[Path, bytes] = {}
        self.unparseable = 0
        self.rows_read = 0
        self._reducers = Reducers()

    def _discover(self) -> None:
        found: list[tuple[int, Path]] = []
        if self.logs_dir.is_dir():
            for entry in self.logs_dir.iterdir():
                match = SEGMENT_RE.match(entry.name)
                if match is not None and match.group("run") == self.run_id:
                    found.append((int(match.group("seg")), entry))
        self.segments = sorted(found)

    def poll(self) -> Snapshot:
        """Read every new byte of every segment, in segment order, and return the snapshot. Raises: EmptyRunRecord, OSError."""
        self._discover()
        for number, path in self.segments:
            self._reducers.enter_segment(number)
            self._read_new(path)
        if self.rows_read == 0:
            raise EmptyRunRecord(
                f"{self.logs_dir} holds no parseable event of run {self.run_id!r} "
                f"({len(self.segments)} segment(s), {self.unparseable} unparseable line(s))")
        return self._reducers.snapshot()

    def _read_new(self, path: Path) -> None:
        offset = self.offsets.get(path, 0)
        size = path.stat().st_size
        if size <= offset:
            return
        held = self.held.pop(path, b"")
        with path.open("rb") as handle:
            handle.seek(offset)
            while offset < size:
                chunk = handle.read(min(CHUNK_BYTES, size - offset))
                if not chunk:
                    break
                offset += len(chunk)
                lines = (held + chunk).split(b"\n")
                held = lines.pop()
                for raw in lines:
                    self._feed_line(raw)
                self.offsets[path], self.held[path] = offset, held
        if not self.held.get(path):
            self.held.pop(path, None)

    def _feed_line(self, raw: bytes) -> None:
        text = raw.strip()
        if not text:
            return
        try:
            row = json.loads(text)
        except (ValueError, UnicodeDecodeError):
            self.unparseable += 1
            return
        if not isinstance(row, dict):
            self.unparseable += 1
            return
        try:
            self._reducers.feed(row)
        except (TypeError, ValueError):
            self.unparseable += 1
            return
        self.rows_read += 1
