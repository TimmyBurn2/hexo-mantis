"""The strength read: cell sidecars to logits, two-save panels against a reference, the controls' validity and the outcome rule."""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis.util.loadpkg import load_tools_package

#: The sidecar suffix each ruler's cells carry; the read takes the full-arm receipts only.
RULERS = {"rung16": "ladder455_n16", "s": "strix256_arena"}
LINE = 0.17
#: The control is near the reference iff its panel is no further below than this, on every ruler.
CONTROL_NEAR = 0.5
#: The known-bad is valid iff its rung-16 logit sits at least this far under the control's panel.
KNOWNBAD_GAP = 1.0


def _sidecars() -> Any:
    """The dash's sidecar reader, the one reading of a cell (forfeits and the instrument tuple included)."""
    load_tools_package("dash")
    return importlib.import_module("dash.readers.sidecars")


@dataclass(frozen=True)
class Cell:
    """One cell as the dash reads it (a Six forfeit is no game of ours; its interval then Wilson over the real games), with its receipt and instrument."""

    path: str
    receipt: tuple[str, str, str]
    unit: tuple[str, ...]
    wr: float
    lo: float
    hi: float
    games: int
    forfeits: int

    @property
    def interval(self) -> str:
        """Which interval the se comes from: the cell's pair bootstrap, or Wilson over its real games after forfeits."""
        return "wilson_real_games" if self.forfeits else "pair_bootstrap"

    @property
    def logit(self) -> float:
        p = min(max(self.wr, 0.5 / self.games), 1 - 0.5 / self.games)
        return math.log(p / (1 - p))

    @property
    def se(self) -> float:
        p = min(max(self.wr, 0.5 / self.games), 1 - 0.5 / self.games)
        return (self.hi - self.lo) / 3.92 / (p * (1 - p))


def read_cell(path: Path) -> Cell:
    """A sidecar as a Cell through the dash's `parse`; Raises: ValueError — a failed cell, a sidecar the dash cannot read, no interval, or a receipt field absent (CellRefused among them)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("rc", 0) != 0:
        raise ValueError(f"{path.name}: not a finished cell")
    cell = _sidecars().parse(path, raw)
    if cell is None or cell.lo is None or cell.hi is None:
        raise ValueError(f"{path.name}: the sidecar lacks a field the reading needs")
    fields = [raw.get(k) for k in ("checkpoint_sha256", "unit", "started_utc")]
    receipt = tuple(v for v in fields if isinstance(v, str) and v)
    if len(receipt) != 3:
        raise ValueError(f"{path.name}: its receipt (checkpoint sha, unit, start time) is incomplete")
    return Cell(path=str(path), receipt=(receipt[0], receipt[1], receipt[2]), unit=tuple(cell.unit), wr=cell.wr,
                lo=cell.lo, hi=cell.hi, games=cell.n, forfeits=cell.forfeits)


def pool(cells: list[Cell]) -> tuple[float, float]:
    """Mean logit and its se (independent cells); Raises: ValueError — no cell."""
    if not cells:
        raise ValueError("no cells to pool")
    return (sum(c.logit for c in cells) / len(cells), math.sqrt(sum(c.se ** 2 for c in cells)) / len(cells))


def find_cells(roots: list[Path], checkpoint: str, ruler: str) -> list[Cell]:
    """Every distinct finished receipt of `checkpoint` on `ruler` under `roots`: a copied sidecar is one cell, not two; Raises: ValueError — a sidecar `read_cell` refuses."""
    pattern = f"{checkpoint}.{RULERS[ruler]}.full.json"
    seen: dict[tuple[str, str, str], Cell] = {}
    for root in roots:
        for p in sorted(root.rglob(pattern)):
            cell = read_cell(p)
            seen.setdefault(cell.receipt, cell)
    return list(seen.values())


def one_per_save(roots: list[Path], saves: list[str], ruler: str, name: str) -> list[Cell]:
    """Exactly one receipt per save on `ruler`; Raises: ValueError — a save with none or with two."""
    cells = []
    for ck in saves:
        found = find_cells(roots, ck, ruler)
        if len(found) != 1:
            raise ValueError(f"{name}: {ck} has {len(found)} {ruler} receipts, not one")
        cells += found
    return cells


def one_unit(cells: list[Cell], ruler: str) -> tuple[str, ...]:
    """The one instrument every cell on a ruler was read in; Raises: ValueError — cells from two instruments."""
    units = {c.unit for c in cells}
    if len(units) != 1:
        raise ValueError(f"{ruler}: cells span {len(units)} instruments: {sorted(units)}")
    return units.pop()


def delta(panel: tuple[float, float], ref: tuple[float, float]) -> dict[str, float]:
    """Panel − reference with its se and 95 % interval."""
    d, se = panel[0] - ref[0], math.hypot(panel[1], ref[1])
    return {"delta": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


def verdict(spec: dict[str, Any], roots: list[Path], *, line: float = LINE, near: float = CONTROL_NEAR,
            kb_gap: float = KNOWNBAD_GAP) -> dict[str, Any]:
    """The pre-stated read over `spec` (reference saves, control/known-bad/arm saves with each arm's class); Raises: ValueError — a save without exactly one receipt, a ruler over two instruments, a refused sidecar."""
    ref_cells = {r: [c for ck in spec["reference"] for c in find_cells(roots, ck, r)] for r in RULERS}
    ref = {r: pool(cells) for r, cells in ref_cells.items()}
    out: dict[str, Any] = {"reference": {r: {"logit": v[0], "se": v[1], "cells": [c.path for c in ref_cells[r]]}
                                         for r, v in ref.items()}, "arms": {}}
    every: dict[str, list[Cell]] = {r: list(ref_cells[r]) for r in RULERS}
    panels: dict[str, dict[str, tuple[float, float]]] = {}
    for name, arm in spec["arms"].items():
        row: dict[str, Any] = {"class": arm["class"]}
        panels[name] = {}
        for r in RULERS:
            cells = one_per_save(roots, arm["saves"], r, name)
            every[r] += cells
            panels[name][r] = pool(cells)
            row[r] = {"cells": [{"path": c.path, "wr": c.wr, "logit": c.logit, "forfeits": c.forfeits,
                                 "interval": c.interval} for c in cells],
                      "panel": panels[name][r][0], "se": panels[name][r][1], **delta(panels[name][r], ref[r])}
        row["reaches"] = all(row[r]["delta"] >= line for r in RULERS)
        row["mean_delta"] = sum(row[r]["delta"] for r in RULERS) / len(RULERS)
        out["arms"][name] = row
    control = out["arms"].pop(spec["control"])
    kb = one_per_save(roots, [spec["knownbad"]], "rung16", "known-bad")[0]
    every["rung16"].append(kb)
    out["units"] = {r: list(one_unit(cells, r)) for r, cells in every.items()}
    out["control"] = control
    out["control_near"] = all(control[r]["delta"] >= -near for r in RULERS)
    out["control_above_line"] = all(control[r]["delta"] >= line for r in RULERS)
    if out["control_above_line"]:
        for name, row in out["arms"].items():
            row["v_control"] = {r: delta(panels[name][r], panels[spec["control"]][r]) for r in RULERS}
    out["knownbad"] = {"path": kb.path, "wr": kb.wr, "logit": kb.logit, "gap": control["rung16"]["panel"] - kb.logit}
    out["knownbad_valid"] = out["knownbad"]["gap"] >= kb_gap
    out["halt"] = not (out["control_near"] and out["knownbad_valid"])
    out["outcome"] = outcome(out["arms"]) if not out["halt"] else "HALT"
    return out


def outcome(arms: dict[str, dict[str, Any]]) -> str:
    """The pre-stated outcome: a narrow row reaching wins; else a wide one; else none (the larger mean Δ breaks a tie)."""
    for cls in ("narrow", "wide"):
        reaching = {n: a for n, a in arms.items() if a["class"] == cls and a["reaches"]}
        if reaching:
            best = max(reaching, key=lambda n: reaching[n]["mean_delta"])
            return f"{cls}: {best}"
    return "none"
