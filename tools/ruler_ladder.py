"""The ruler ladder's reading: one save at one rung, and its N50 where logit(win rate), linear in log2(nodes) between two
adjacent rungs, crosses zero. usage: ruler_ladder.py <checkpoint> --rung N | --pair LO HI --arm ARM"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_TOOLS = Path(__file__).resolve().parent
_DRAWS = 20_000
#: Past this share of inverted draws the N50 interval is unbounded, never the interval of the draws that were not.
_INVERTED_OPEN = 0.025
#: log2(nodes) is held inside this band, so a near-flat pair extrapolates to a finite N50 instead of overflowing.
_X_BOUND = 60.0


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / rel)
    if spec is None or spec.loader is None:
        raise ImportError(f"no module at {_TOOLS / rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_follower = _load("strix_follower_for_ladder", "strix_follower.py")
_sidecars = _load("dash_sidecars_for_ladder", "dash/readers/sidecars.py")
#: rung (Six's nodes per turn) -> its follower unit; the follower's table is the one list.
UNIT_OF = {nodes: unit for unit, nodes in _follower.LADDER_UNITS.items()}
RUNGS = tuple(sorted(UNIT_OF))


@dataclass(frozen=True)
class N50:
    """`nodes` is None when the pair is inverted or flat; `extrapolated` when both rungs sit on one side of 50 %."""

    nodes: float | None
    extrapolated: bool
    inverted: bool
    flat: bool = False


def _x(lo_nodes: int, l0: float, hi_nodes: int, l1: float) -> float:
    x0, x1 = math.log2(lo_nodes), math.log2(hi_nodes)
    return max(-_X_BOUND, min(_X_BOUND, x0 + l0 * (x1 - x0) / (l0 - l1)))


def n50(lo_nodes: int, wr_lo: float, hi_nodes: int, wr_hi: float, *, games: int | None = None) -> N50:
    """N50 off two rungs, `lo_nodes` < `hi_nodes`, our win rates there (a 0 or 1 moved half a game in, given `games`)."""
    l0, l1 = _sidecars.logit(wr_lo, games), _sidecars.logit(wr_hi, games)
    if l0 == l1:
        return N50(nodes=None, extrapolated=False, inverted=False, flat=True)
    if l0 < l1:
        return N50(nodes=None, extrapolated=False, inverted=True)
    return N50(nodes=2 ** _x(lo_nodes, l0, hi_nodes, l1), extrapolated=not l0 >= 0 >= l1, inverted=False)


def pick_pair(screen: dict[int, float]) -> tuple[int, int]:
    """The highest rung reading >= 0.5 and the one above. Raises: ValueError naming the rung to read next."""
    read = sorted(screen)
    at_or_above = [n for n in read if screen[n] >= 0.5]
    if not at_or_above:
        below = [n for n in RUNGS if n < read[0]]
        if below:
            raise ValueError(f"rung {read[0]} already reads {screen[read[0]]:.3f} < 0.5: read rung {below[-1]} next")
        return RUNGS[0], RUNGS[1]
    lo = at_or_above[-1]
    above = [n for n in RUNGS if n > lo]
    if not above:
        return RUNGS[-2], RUNGS[-1]
    if above[0] not in screen:
        raise ValueError(f"rung {lo} still reads {screen[lo]:.3f} >= 0.5: read rung {above[0]} next")
    return lo, above[0]


def read_rung(checkpoint: Path, nodes: int, *, arm: str | None) -> dict[str, Any]:
    """One save's whole-book cell at one rung. Raises: FileNotFoundError, ValueError (a screen, no CI), JSONDecodeError."""
    unit = UNIT_OF[nodes]
    path = _follower.sidecar_path(checkpoint, unit, arm)
    if not path.is_file():
        raise FileNotFoundError(f"{unit}: no sidecar at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    whole, games = _follower.whole_book(unit), int(raw.get("games") or 0)
    if games != whole:
        raise ValueError(f"{path.name} holds {games} games, not the whole book's {whole}: a screen is never a save's cell")
    if raw.get("wr_ci_lower") is None or raw.get("wr_ci_upper") is None:
        raise ValueError(f"{path.name} carries no CI")
    lo, hi = _sidecars.logit(float(raw["wr_ci_lower"]), games), _sidecars.logit(float(raw["wr_ci_upper"]), games)
    return {"unit": unit, "nodes": nodes, "wr": float(raw["wr"]), "ci": [raw["wr_ci_lower"], raw["wr_ci_upper"]],
            "logit": _sidecars.logit(float(raw["wr"]), games), "se_logit": (hi - lo) / (2 * _sidecars.Z95),
            "games": games, "eff_n": raw.get("eff_n"), "regime": raw.get("regime"),
            "opening_book": raw.get("opening_book")}


def read_save(checkpoint: Path, pair: tuple[int, int], *, arm: str | None) -> dict[str, Any]:
    """Both rungs, N50 and its seeded 95 % interval (unbounded past 2.5 % inverted draws). Raises: as `read_rung`."""
    lo, hi = pair
    if (lo, hi) not in zip(RUNGS, RUNGS[1:], strict=False):
        raise ValueError(f"{pair} is not two adjacent rungs of {RUNGS}")
    rows = {nodes: read_rung(checkpoint, nodes, arm=arm) for nodes in pair}
    point = n50(lo, rows[lo]["wr"], hi, rows[hi]["wr"], games=rows[lo]["games"])
    rng = random.Random(0)
    draws, inverted = [], 0
    for _ in range(_DRAWS):
        l0 = rng.gauss(rows[lo]["logit"], rows[lo]["se_logit"])
        l1 = rng.gauss(rows[hi]["logit"], rows[hi]["se_logit"])
        if l0 <= l1:
            inverted += 1
        else:
            draws.append(_x(lo, l0, hi, l1))
    draws.sort()
    share = inverted / _DRAWS
    bounded = share <= _INVERTED_OPEN and draws

    def bound(q: float) -> float | None:
        return 2 ** draws[min(len(draws) - 1, int(q * len(draws)))] if bounded else None

    return {"checkpoint": checkpoint.name, "pair": list(pair), "rates": {str(n): rows[n]["wr"] for n in pair},
            "ci": {str(n): rows[n]["ci"] for n in pair}, "games": {str(n): rows[n]["games"] for n in pair},
            "regime": {str(n): rows[n]["regime"] for n in pair}, "n50": point.nodes, "n50_lo": bound(0.025),
            "n50_hi": bound(0.975), "extrapolated": point.extrapolated, "inverted": point.inverted, "flat": point.flat,
            "inverted_draw_share": share}


def main(argv: list[str] | None = None) -> int:
    """Print one save's ladder row. Raises: FileNotFoundError, ValueError as `read_rung`."""
    ap = argparse.ArgumentParser(description="Read one save off the ruler ladder.")
    ap.add_argument("checkpoint", type=Path)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--rung", type=int, choices=RUNGS)
    what.add_argument("--pair", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--arm", required=True, help="the A/B arm the sidecars name ('' for none)")
    args = ap.parse_args(argv)
    arm = args.arm or None
    row = (read_rung(args.checkpoint, args.rung, arm=arm) if args.rung is not None
           else read_save(args.checkpoint, (args.pair[0], args.pair[1]), arm=arm))
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
