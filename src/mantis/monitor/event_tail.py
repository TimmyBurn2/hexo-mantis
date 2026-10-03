"""Tail a run's event stream: the NEW lines of every `events_<run_id>_seg*.jsonl`, filtered to the named events."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class EventTail:
    """Reads only the NEW lines of every `events_<run_id>_seg*.jsonl` in `logs/`, in segment order, keeping `events`."""

    def __init__(self, run_dir: Path, run_id: str, events: Iterable[str]) -> None:
        self.logs = Path(run_dir) / "logs"
        self.prefix = f"events_{run_id}_seg"
        self.events = frozenset(events)
        self.offsets: dict[Path, int] = {}

    def read_new(self) -> list[dict[str, Any]]:
        """The rows appended since the last call; a partial last line (the writer mid-append) waits. Raises: OSError."""
        rows: list[dict[str, Any]] = []
        names = [name.encode() for name in self.events]
        for path in sorted(self.logs.glob(f"{self.prefix}*.jsonl")):
            start = self.offsets.get(path, 0)
            with path.open("rb") as fh:
                fh.seek(start)
                data = fh.read()
            end = data.rfind(b"\n") + 1
            self.offsets[path] = start + end
            for line in data[:end].splitlines():
                if not any(name in line for name in names):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") in self.events:
                    rows.append(row)
        return rows


__all__ = ["EventTail"]
