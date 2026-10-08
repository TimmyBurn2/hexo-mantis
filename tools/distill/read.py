"""The strength read: cell sidecars to logits, two-save panels against a reference, the controls' validity and the outcome rule."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The sidecar suffix each ruler's cells carry.
RULERS = {"rung16": "ladder455_n16", "s": "strix256_arena"}
LINE = 0.17
#: The control is near the reference iff its panel is no further below than this, on every ruler.
CONTROL_NEAR = 0.5
#: The known-bad is valid iff its rung-16 logit sits at least this far under the control's panel.
KNOWNBAD_GAP = 1.0


@dataclass(frozen=True)
class Cell:
    """One cell: win rate with Six's forfeits out of our wins, its pair-bootstrap interval, and its logit with the delta-method se."""

    path: str
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
    """A sidecar as a Cell; Raises: ValueError — a failed or short cell."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("rc", 0) != 0 or "wr" not in raw:
        raise ValueError(f"{path.name}: not a finished cell")
    games = int(raw["games"])
    forfeits = int((raw.get("six_findings") or {}).get("count") or 0)
    wr = float(raw["wr"])
    if forfeits:
        wr = max(0.0, raw["wins"] - forfeits + raw.get("draws", 0) / 2) / (games - forfeits)
    return Cell(path=str(path), wr=wr, lo=float(raw["wr_ci_lower"]), hi=float(raw["wr_ci_upper"]), games=games,
                forfeits=forfeits)


def pool(cells: list[Cell]) -> tuple[float, float]:
    """Mean logit and its se (independent cells)."""
    if not cells:
        raise ValueError("no cells to pool")
    return (sum(c.logit for c in cells) / len(cells), math.sqrt(sum(c.se ** 2 for c in cells)) / len(cells))


def find_cells(roots: list[Path], checkpoint: str, ruler: str) -> list[Cell]:
    """Every finished sidecar of `checkpoint` on `ruler` under `roots`."""
    pattern = f"{checkpoint}.{RULERS[ruler]}.full.json"
    return [read_cell(p) for root in roots for p in sorted(root.rglob(pattern))]


def delta(panel: tuple[float, float], ref: tuple[float, float]) -> dict[str, float]:
    """Panel − reference with its se and 95 % interval."""
    d, se = panel[0] - ref[0], math.hypot(panel[1], ref[1])
    return {"delta": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


def verdict(spec: dict[str, Any], roots: list[Path], *, line: float = LINE, near: float = CONTROL_NEAR,
            kb_gap: float = KNOWNBAD_GAP) -> dict[str, Any]:
    """The pre-stated read over `spec` (reference saves, control/known-bad/arm saves with each arm's class); Raises: ValueError — a missing cell."""
    ref = {r: pool([c for ck in spec["reference"] for c in find_cells(roots, ck, r)]) for r in RULERS}
    out: dict[str, Any] = {"reference": {r: {"logit": v[0], "se": v[1]} for r, v in ref.items()}, "arms": {}}
    for name, arm in spec["arms"].items():
        row: dict[str, Any] = {"class": arm["class"]}
        for r in RULERS:
            cells = [c for ck in arm["saves"] for c in find_cells(roots, ck, r)]
            if len(cells) != len(arm["saves"]):
                raise ValueError(f"{name}: {len(cells)} {r} cells for {len(arm['saves'])} saves")
            panel = pool(cells)
            row[r] = {"cells": [(c.wr, c.logit) for c in cells], "panel": panel[0], "se": panel[1],
                      **delta(panel, ref[r])}
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
    out["knownbad"] = {"wr": kb.wr, "logit": kb.logit, "gap": control["rung16"]["panel"] - kb.logit}
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
