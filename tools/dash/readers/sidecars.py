"""Strength from the follower's cell sidecars: one series per unit, the parent joined by its checkpoint stem, the going-forward read."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The sidecars the follower writes beside a checkpoint: `<ckpt>.six30_16[.arm].json`, `<ckpt>.strix256[_r6].json`.
GLOBS = ("*.six*.json", "*.strix*.json")
#: The going-forward read: the mean logit of the last `CELLS` cells against the parent's logit plus `LINE_LOGIT`.
CELLS = 4
LINE_LOGIT = 0.17


@dataclass(frozen=True)
class Cell:
    """One sidecar: whose checkpoint, at which step, the opponent family, the unit it was read in, and its reading."""

    run_id: str
    stem: str
    step: int
    family: str
    unit: tuple[str, ...]
    label: str
    wr: float
    lo: float | None
    hi: float | None
    n: int
    regime: str
    path: str

    @property
    def logit(self) -> float:
        p = min(max(self.wr, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _tactics(raw: dict[str, Any]) -> str:
    """The tactics arm and its block's hash: two blocks under one arm name are two instruments."""
    t = raw.get("tactics")
    if not isinstance(t, dict):
        return "no tactics"
    block = t.get("block")
    digest = "" if block is None else hashlib.sha256(json.dumps(block, sort_keys=True).encode()).hexdigest()[:8]
    return f"{t.get('arm') or 'config'} {digest}".strip()


def parse(path: Path, raw: Any) -> Cell | None:
    """A cell from one sidecar's JSON, or None when a field the series needs is absent."""
    if not isinstance(raw, dict):
        return None
    step, wr, ours = _int(raw.get("step")), _num(raw.get("wr")), raw.get("ours") or {}
    six, strix = raw.get("six"), raw.get("strix")
    if step is None or wr is None or not isinstance(ours, dict):
        return None
    tactics = _tactics(raw)
    common = (str(raw.get("unit")), str(ours.get("search_kind")), str(ours.get("sims")), tactics)
    if isinstance(six, dict):
        family = "six"
        unit = ("six", *common, str(six.get("commit")), str(six.get("net_sha256")), str(six.get("nodes")))
        label = f"ours {ours.get('search_kind')}-{ours.get('sims')} vs Six gen {six.get('generation')} @ {six.get('nodes')} nodes, tactics {tactics}"
    elif isinstance(strix, dict):
        family = "strix"
        unit = ("strix", *common, str(strix.get("commit")), str(strix.get("checkpoint_sha256")), str(strix.get("sims")),
                str(strix.get("solver", "on")), str(strix.get("radius")))
        label = (f"ours {ours.get('search_kind')}-{ours.get('sims')} vs Strix {strix.get('sims')} sims, solver "
                 f"{strix.get('solver', 'on')}" + ("" if strix.get("radius") is None else f", r{strix.get('radius')}"))
    else:
        return None
    n = _int(raw.get("eff_n")) or _int(raw.get("games")) or 0
    stem = str(raw.get("checkpoint") or path.name.split(".ckpt")[0]).split(".ckpt")[0]
    return Cell(run_id=str(raw.get("run_id")), stem=stem, step=step, family=family, unit=unit, label=label, wr=wr,
                lo=_num(raw.get("wr_ci_lower")), hi=_num(raw.get("wr_ci_upper")), n=n,
                regime=str(raw.get("regime", "?")), path=str(path))


def load(dirs: Iterable[Path]) -> tuple[list[Cell], list[str]]:
    """Every sidecar under the directories (failed cells skipped and named), and the skip notes."""
    cells: list[Cell] = []
    skipped: list[str] = []
    seen: set[Path] = set()
    for root in dirs:
        if not root.is_dir():
            skipped.append(f"{root} is not a directory")
            continue
        for path in sorted({p for g in GLOBS for p in root.rglob(g)}):
            if path in seen:
                continue
            seen.add(path)
            if path.name.endswith(".failed.json"):
                skipped.append(f"{path.name}: a failed cell")
                continue
            try:
                cell = parse(path, json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                skipped.append(f"{path.name}: {type(exc).__name__}")
                continue
            if cell is None:
                skipped.append(f"{path.name}: no step, win rate or opponent")
            else:
                cells.append(cell)
    unique = {(c.stem, c.unit): c for c in cells}
    return sorted(unique.values(), key=lambda c: (c.run_id, c.step, c.label)), skipped


@dataclass(frozen=True)
class Strength:
    """One family's panel for one run: its line (one unit), the other-unit cells, the parent cell in the line's unit."""

    family: str
    line: tuple[Cell, ...]
    other: tuple[Cell, ...]
    parent: Cell | None
    parent_other: Cell | None

    @property
    def going_forward(self) -> tuple[float, int] | None:
        """`(mean logit of the last cells − the parent's logit, cells used)`, or None without a parent or a cell."""
        if self.parent is None or not self.line:
            return None
        last = self.line[-CELLS:]
        return sum(c.logit for c in last) / len(last) - self.parent.logit, len(last)


def strength(cells: list[Cell], family: str, run_id: str, parent_stem: str | None) -> Strength:
    """The run's cells of one family: the unit with the most cells (the newest on a tie) is the line, the rest are other units."""
    own = [c for c in cells if c.family == family and c.run_id == run_id]
    by_unit: dict[tuple[str, ...], list[Cell]] = {}
    for c in own:
        by_unit.setdefault(c.unit, []).append(c)
    parents = [c for c in cells if c.family == family and parent_stem is not None and c.stem == parent_stem]
    if not by_unit:
        return Strength(family, (), (), None, parents[0] if parents else None)
    unit = max(by_unit, key=lambda u: (len(by_unit[u]), max(c.step for c in by_unit[u])))
    line = tuple(sorted(by_unit[unit], key=lambda c: c.step))
    other = tuple(sorted((c for c in own if c.unit != unit), key=lambda c: c.step))
    parent = next((c for c in parents if c.unit == unit), None)
    return Strength(family, line, other, parent, None if parent else (parents[0] if parents else None))
