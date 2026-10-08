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
    load_tools_package("dash")
    return importlib.import_module("dash.readers.sidecars")


@dataclass(frozen=True)
class Cell:
    """One cell: win rate over the real games (a Six forfeit is no game of ours), its interval, and its logit with the delta-method se."""

    path: str
    receipt: tuple[str, str, str]
    wr: float
    lo: float
    hi: float
    games: int
    forfeits: int

    @property
    def logit(self) -> float:
        p = min(max(self.wr, 0.5 / self.games), 1 - 0.5 / self.games)
        return math.log(p / (1 - p))

    @property
    def se(self) -> float:
        p = min(max(self.wr, 0.5 / self.games), 1 - 0.5 / self.games)
        return (self.hi - self.lo) / 3.92 / (p * (1 - p))


def read_cell(path: Path) -> Cell:
    """A sidecar as a Cell, forfeits out through the dash's one reading; Raises: ValueError — a failed or short cell; CellRefused — forfeit counts that do not line up."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("rc", 0) != 0 or "wr" not in raw:
        raise ValueError(f"{path.name}: not a finished cell")
    games = int(raw.get("eff_n", raw["games"]))
    forfeits = int((raw.get("six_findings") or {}).get("count") or 0)
    wr, lo, hi = float(raw["wr"]), float(raw["wr_ci_lower"]), float(raw["wr_ci_upper"])
    if forfeits:
        wr, lo, hi, games = _sidecars()._without_forfeits(raw, wr, games, forfeits)
    receipt = (str(raw.get("checkpoint_sha256")), str(raw.get("unit")), str(raw.get("started_utc")))
    return Cell(path=str(path), receipt=receipt, wr=wr, lo=lo, hi=hi, games=games, forfeits=forfeits)


def pool(cells: list[Cell]) -> tuple[float, float]:
    """Mean logit and its se (independent cells); Raises: ValueError — no cell."""
    if not cells:
        raise ValueError("no cells to pool")
    return (sum(c.logit for c in cells) / len(cells), math.sqrt(sum(c.se ** 2 for c in cells)) / len(cells))


def find_cells(roots: list[Path], checkpoint: str, ruler: str) -> list[Cell]:
    """Every distinct finished receipt of `checkpoint` on `ruler` under `roots`: a copied sidecar is one cell, not two."""
    pattern = f"{checkpoint}.{RULERS[ruler]}.full.json"
    seen: dict[tuple[str, str, str], Cell] = {}
    for root in roots:
        for p in sorted(root.rglob(pattern)):
            cell = read_cell(p)
            seen.setdefault(cell.receipt, cell)
    return list(seen.values())


def delta(panel: tuple[float, float], ref: tuple[float, float]) -> dict[str, float]:
    """Panel − reference with its se and 95 % interval."""
    d, se = panel[0] - ref[0], math.hypot(panel[1], ref[1])
    return {"delta": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


def verdict(spec: dict[str, Any], roots: list[Path], *, line: float = LINE, near: float = CONTROL_NEAR,
            kb_gap: float = KNOWNBAD_GAP) -> dict[str, Any]:
    """The pre-stated read over `spec` (reference saves, control/known-bad/arm saves with each arm's class); Raises: ValueError — a missing or extra cell."""
    ref_cells = {r: [c for ck in spec["reference"] for c in find_cells(roots, ck, r)] for r in RULERS}
    ref = {r: pool(cells) for r, cells in ref_cells.items()}
    out: dict[str, Any] = {"reference": {r: {"logit": v[0], "se": v[1], "cells": [c.path for c in ref_cells[r]]}
                                         for r, v in ref.items()}, "arms": {}}
    panels: dict[str, dict[str, tuple[float, float]]] = {}
    for name, arm in spec["arms"].items():
        row: dict[str, Any] = {"class": arm["class"]}
        panels[name] = {}
        for r in RULERS:
            cells = [c for ck in arm["saves"] for c in find_cells(roots, ck, r)]
            if len(cells) != len(arm["saves"]):
                raise ValueError(f"{name}: {len(cells)} {r} cells for {len(arm['saves'])} saves")
            panels[name][r] = pool(cells)
            row[r] = {"cells": [{"path": c.path, "wr": c.wr, "logit": c.logit, "forfeits": c.forfeits} for c in cells],
                      "panel": panels[name][r][0], "se": panels[name][r][1], **delta(panels[name][r], ref[r])}
        row["reaches"] = all(row[r]["delta"] >= line for r in RULERS)
        row["mean_delta"] = sum(row[r]["delta"] for r in RULERS) / len(RULERS)
        out["arms"][name] = row
    control = out["arms"].pop(spec["control"])
    kb_cells = find_cells(roots, spec["knownbad"], "rung16")
    if len(kb_cells) != 1:
        raise ValueError(f"the known-bad has {len(kb_cells)} rung-16 cells, not one")
    kb = kb_cells[0]
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
