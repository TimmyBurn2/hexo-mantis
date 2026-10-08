"""The records' units file: the rule and its switches, the second ruler, the legacy tags and the rung state; absent is a stated gap."""
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
    """The file's rule (its unit now and each switch), second ruler, legacy units and rung state; `note` names what did not read."""

    current: str | None
    streak: int | None
    changes: tuple[Change, ...]
    cells: int
    note: str
    rule: str | None = None
    switches: tuple[Change, ...] = ()
    second: str | None = None
    legacy: frozenset[str] = frozenset()


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def read(path: Path | None) -> Ladder | None:
    """The units file at `path`; None when none was given, a Ladder whose note names the gap when it is unreadable."""
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Ladder(None, None, (), 0, "the units file is not there yet")
    except (OSError, ValueError) as exc:
        return Ladder(None, None, (), 0, f"the units file did not read ({type(exc).__name__})")
    if not isinstance(raw, dict):
        return Ladder(None, None, (), 0, "the units file is not an object")
    rows, hist = raw.get("changes"), raw.get("history")
    bad = [k for k, v in (("changes", rows), ("history", hist)) if v is not None and not isinstance(v, list)]
    changes, skipped = [], 0
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and _int(row.get("step")) is not None and row.get("from") and row.get("to"):
            changes.append(Change(int(row["step"]), str(row["from"]), str(row["to"])))
        else:
            skipped += 1
    history = [h for h in hist if isinstance(h, dict) and _num(h.get("wr")) is not None] if isinstance(hist, list) else []
    legacy = raw.get("legacy")
    bad += ["legacy"] if legacy is not None and not isinstance(legacy, list) else []
    notes = [f"{k} is not a list" for k in bad] + ([f"{skipped} change row(s) lack step, from or to"] if skipped else [])
    notes += [f"{k} is not a string" for k in ("rule", "second") if raw.get(k) is not None and not isinstance(raw[k], str)]
    rule, switches = None, ()
    if isinstance(raw.get("rule"), str):
        try:
            rule, switches = parse_rule(raw["rule"])
        except ValueError as exc:
            notes.append(str(exc))
    current, second = raw.get("current_unit"), raw.get("second")
    return Ladder(current=str(current) if isinstance(current, str) and current else None, streak=_int(raw.get("streak")),
                  changes=tuple(sorted(changes, key=lambda c: c.step)), cells=len(history),
                  note="read" if not notes else "read, but " + "; ".join(notes), rule=rule, switches=switches,
                  second=second if isinstance(second, str) and second else None,
                  legacy=frozenset(str(u) for u in legacy if isinstance(u, str)) if isinstance(legacy, list) else frozenset())


def parse_rule(text: str) -> tuple[str, tuple[Change, ...]]:
    """`UNIT[,UNIT@STEP…]`: the unit the rule reads now, and each switch from the unit before it. Raises: ValueError, naming the rule."""
    units: list[str] = []
    switches: list[Change] = []
    for part in text.split(","):
        unit, at, step = part.partition("@")
        if not unit or bool(at) != bool(units) or (at and not (step.isascii() and step.isdigit())):
            raise ValueError(f"the rule {text!r} wants UNIT[,UNIT@STEP…], its first unit without a step")
        if units:
            if int(step) <= (switches[-1].step if switches else 0) or unit == units[-1]:
                raise ValueError(f"the rule {text!r} must switch at rising steps above 0, each time to another unit")
            switches.append(Change(int(step), units[-1], unit))
        units.append(unit)
    return units[-1], tuple(switches)
