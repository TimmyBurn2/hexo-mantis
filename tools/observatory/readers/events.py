"""The event-stream tail: every segment from its byte offset, a torn tail held until its newline arrives."""
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


class EventTail:
    """Tails `events_<run_id>_seg*.jsonl` under `logs_dir` into one `Reducers`; call `poll()` repeatedly."""

    def __init__(self, logs_dir: Path | str, run_id: str) -> None:
        self.logs_dir = Path(logs_dir)
        self.run_id = run_id
        self.segments: list[Path] = []
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
        self.segments = [path for _seg, path in sorted(found)]

    def poll(self) -> Snapshot:
        """Read every new byte of every segment, in segment order, and return the snapshot.

        Raises:
            EmptyRunRecord: no parseable row exists in any segment yet.
            OSError: a segment could not be read.
        """
        self._discover()
        for path in self.segments:
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
        self.offsets[path] = offset
        if held:
            self.held[path] = held

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
        self._reducers.feed(row)
        self.rows_read += 1
