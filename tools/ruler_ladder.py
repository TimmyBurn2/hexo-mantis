"""The ruler ladder's reading: Six gen455 at N nodes per turn against ours 128 per stone, a save read at the two rungs
bracketing 50 %, and its N50 — the node count where the logit of our win rate, linear in log2(N) between the two
rungs, crosses zero.

usage: ruler_ladder.py <checkpoint> --pair LO HI [--arm full]  (prints the save's row as JSON)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The rungs, Six's `go nodes` per turn; ours plays 128 per stone at every rung.
RUNGS = (16, 32, 64, 128, 256, 512)
UNIT_PREFIX = "ladder455_n"
_Z95 = 1.959964
_DRAWS = 20_000


@dataclass(frozen=True)
class N50:
    """`nodes` is None when the pair is inverted; `extrapolated` when both rungs sit on one side of 50 %."""

    nodes: float | None
    extrapolated: bool
    inverted: bool


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def n50(lo_nodes: int, wr_lo: float, hi_nodes: int, wr_hi: float) -> N50:
    """N50 off two rungs, `lo_nodes` < `hi_nodes`, our win rates there."""
    l0, l1 = _logit(wr_lo), _logit(wr_hi)
    if l0 <= l1:
        return N50(nodes=None, extrapolated=False, inverted=True)
    x0, x1 = math.log2(lo_nodes), math.log2(hi_nodes)
    x = x0 + l0 * (x1 - x0) / (l0 - l1)
    return N50(nodes=2 ** x, extrapolated=not l0 >= 0 >= l1, inverted=False)


def pick_pair(screen: dict[int, float]) -> tuple[int, int]:
    """The highest screened rung reading >= 0.5 and the rung above it; (16, 32) when even 16 reads below.

    Raises: ValueError naming the rung to read next when the screen's top rung still reads >= 0.5.
    """
    read = sorted(screen)
    at_or_above = [n for n in read if screen[n] >= 0.5]
    if not at_or_above:
        return RUNGS[0], RUNGS[1]
    lo = at_or_above[-1]
    above = [n for n in RUNGS if n > lo]
    if not above:
        return RUNGS[-2], RUNGS[-1]
    if above[0] not in screen:
        raise ValueError(f"rung {lo} still reads {screen[lo]:.3f} >= 0.5: read rung {above[0]} next")
    return lo, above[0]


def _se(row: dict[str, Any]) -> float:
    return (_logit(float(row["wr_ci_upper"])) - _logit(float(row["wr_ci_lower"]))) / (2 * _Z95)


def read_save(checkpoint: Path, pair: tuple[int, int], *, arm: str | None) -> dict[str, Any]:
    """The save's raw rates at `pair`, its N50 and a 95 % interval drawn from each cell's logit CI (seeded).

    Raises: FileNotFoundError naming a rung's missing sidecar; json.JSONDecodeError on a corrupt one.
    """
    rows = {}
    for nodes in pair:
        unit = f"{UNIT_PREFIX}{nodes}"
        tag = "" if arm is None else f".{arm}"
        path = checkpoint.with_name(f"{checkpoint.name}.{unit}{tag}.json")
        if not path.is_file():
            raise FileNotFoundError(f"{unit}: no sidecar at {path}")
        rows[nodes] = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = pair
    point = n50(lo, float(rows[lo]["wr"]), hi, float(rows[hi]["wr"]))
    rng = random.Random(0)
    draws, inverted = [], 0
    for _ in range(_DRAWS):
        l0 = rng.gauss(_logit(float(rows[lo]["wr"])), _se(rows[lo]))
        l1 = rng.gauss(_logit(float(rows[hi]["wr"])), _se(rows[hi]))
        if l0 <= l1:
            inverted += 1
            continue
        draws.append(math.log2(lo) + l0 * (math.log2(hi) - math.log2(lo)) / (l0 - l1))
    draws.sort()
    bound = (lambda q: 2 ** draws[min(len(draws) - 1, int(q * len(draws)))]) if draws else (lambda _q: None)
    return {"checkpoint": checkpoint.name, "pair": list(pair), "rates": {str(n): float(rows[n]["wr"]) for n in pair},
            "ci": {str(n): [rows[n]["wr_ci_lower"], rows[n]["wr_ci_upper"]] for n in pair},
            "n50": point.nodes, "n50_lo": bound(0.025), "n50_hi": bound(0.975), "extrapolated": point.extrapolated,
            "inverted": point.inverted, "inverted_draw_share": inverted / _DRAWS}


def main(argv: list[str] | None = None) -> int:
    """Print one save's ladder row. Raises: FileNotFoundError on a missing rung sidecar."""
    ap = argparse.ArgumentParser(description="Read one save's N50 off its two ladder rungs.")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--pair", type=int, nargs=2, required=True, metavar=("LO", "HI"))
    ap.add_argument("--arm", default="full", help="the A/B arm named in the sidecars ('' for none)")
    args = ap.parse_args(argv)
    if tuple(args.pair) not in zip(RUNGS, RUNGS[1:], strict=False):
        ap.error(f"--pair must be two adjacent rungs of {RUNGS}")
    print(json.dumps(read_save(args.checkpoint, (args.pair[0], args.pair[1]), arm=args.arm or None)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
