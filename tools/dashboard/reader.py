"""Read an existing run record — the event stream and the ladder file, as the run writes them."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EmptyRunRecord(RuntimeError):
    """The record carried no parseable events — refuse rather than render an empty page."""


#: Per-game payloads the dashboard never reads; dropped at parse time so a 35k-game record does
#: not hold two million move strings in memory. The drop is counted and printed on the page.
HEAVY_FIELDS: dict[str, tuple[str, ...]] = {
    "game_complete": ("moves_list", "moves_detail", "value_trace"),
}


@dataclass
class Record:
    """A parsed run record. `events` keeps stream order; `by` indexes by event name."""

    events: list[dict[str, Any]]
    ladder: dict[str, Any] | None = None
    ladder_note: str = ""
    source: str = ""
    record_dir: Path | None = None
    dropped_fields: dict[str, int] = field(default_factory=dict)
    by: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index: dict[str, list[dict[str, Any]]] = {}
        for row in self.events:
            index.setdefault(str(row.get("event", "?")), []).append(row)
        self.by = index

    def rows(self, name: str) -> list[dict[str, Any]]:
        return self.by.get(name, [])

    def last(self, name: str) -> dict[str, Any] | None:
        rows = self.rows(name)
        return rows[-1] if rows else None

    def series(self, name: str, x_key: str, y_key: str) -> list[tuple[float, float]]:
        """`(x, y)` pairs from every row of `name` whose two fields are finite numbers."""
        out: list[tuple[float, float]] = []
        for row in self.rows(name):
            x, y = _finite(row.get(x_key)), _finite(row.get(y_key))
            if x is not None and y is not None:
                out.append((x, y))
        return out

    def rungs(self) -> list[tuple[str, list[dict[str, Any]]]]:
        """`(rung name, history rows)` per rung, in file order; empty without a ladder."""
        if not self.ladder:
            return []
        return [(name, list((state or {}).get("history") or []))
                for name, state in self.ladder.items() if isinstance(state, dict)]

    def wall_hours(self) -> float | None:
        stamps = [t for t in (_finite(r.get("ts")) for r in self.events) if t is not None]
        return (stamps[-1] - stamps[0]) / 3600.0 if len(stamps) >= 2 else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def load_record(events_path: Path, ladder_path: Path | None = None,
                record_dir: Path | None = None) -> Record:
    """Parse the event stream line by line, and the ladder state file when one is given.

    Raises:
        EmptyRunRecord: the stream parsed to zero events.
        OSError: the stream could not be read.
    """
    events: list[dict[str, Any]] = []
    unparseable = 0
    dropped: dict[str, int] = {}
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                unparseable += 1
                continue
            if not isinstance(row, dict):
                continue
            heavy = HEAVY_FIELDS.get(str(row.get("event")))
            if heavy:
                for key in heavy:
                    if key in row:
                        del row[key]
                        dropped[str(row["event"])] = dropped.get(str(row["event"]), 0) + 1
            events.append(row)
    if not events:
        raise EmptyRunRecord(
            f"{events_path} carried no parseable events ({unparseable} unparseable line(s)). "
            "Refusing to render: a page built from nothing looks exactly like a page built "
            "from a clean run."
        )
    ladder, note = _load_ladder(ladder_path)
    return Record(events=events, ladder=ladder, ladder_note=note, source=str(events_path),
                  record_dir=record_dir, dropped_fields=dropped)


def _load_ladder(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if path is None:
        return None, "no eval_ladder_state.json was given"
    if not path.exists():
        return None, f"{path.name} does not exist"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None, f"{path.name} did not parse as JSON"
    if not isinstance(raw, dict):
        return None, f"{path.name} did not parse as a rung mapping"
    return raw, "loaded"
