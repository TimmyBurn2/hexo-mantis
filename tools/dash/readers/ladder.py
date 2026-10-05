"""The ruler ladder's state file: the comparison ruler now, its streak, the cells it read and the rung changes; absent is a stated gap."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Change:
    """The comparison ruler moved from one unit to another at a step."""

    step: int
    frm: str
    to: str


@dataclass(frozen=True)
class Ladder:
    """`current` and `streak` as the file states them, and its rung changes in step order; `note` says why a field is missing."""

    current: str | None
    streak: int | None
    changes: tuple[Change, ...]
    cells: int
    note: str


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def read(path: Path | None) -> Ladder | None:
    """The ladder at `path`; None when no ladder was given, a Ladder whose note names the gap when the file is unreadable."""
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Ladder(None, None, (), 0, "the ladder file is not there yet")
    except (OSError, ValueError) as exc:
        return Ladder(None, None, (), 0, f"the ladder file did not read ({type(exc).__name__})")
    if not isinstance(raw, dict):
        return Ladder(None, None, (), 0, "the ladder file is not an object")
    changes = []
    for row in raw.get("changes") or []:
        if isinstance(row, dict) and _int(row.get("step")) is not None and row.get("from") and row.get("to"):
            changes.append(Change(int(row["step"]), str(row["from"]), str(row["to"])))
    history = [h for h in raw.get("history") or [] if isinstance(h, dict) and isinstance(h.get("wr"), (int, float))
               and math.isfinite(float(h["wr"]))]
    current = raw.get("current_unit")
    return Ladder(current=str(current) if current else None, streak=_int(raw.get("streak")),
                  changes=tuple(sorted(changes, key=lambda c: c.step)), cells=len(history), note="read")
