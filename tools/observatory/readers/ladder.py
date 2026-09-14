"""The ladder file and the round join over a Snapshot: points per rung, the promotion reading, the Elo slope."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .series import Snapshot
from .stats import SlopeFit, clamp_wr, elo_of_wr, ols_slope, wilson_interval

_ROUND_ID = re.compile(r"^r(\d+)_(\d+)$")


def load_ladder(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    """`(rung mapping, note)`; the mapping is `None` with the reason when the file is absent or unparseable."""
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


def rungs(ladder: dict[str, Any] | None) -> list[tuple[str, list[dict[str, Any]]]]:
    """`(rung name, history rows)` per rung in file order; empty without a ladder."""
    if not ladder:
        return []
    return [(name, list((state or {}).get("history") or []))
            for name, state in ladder.items() if isinstance(state, dict)]


@dataclass(frozen=True)
class RoundPoint:
    """One rung's reading for one round; `wr` is `None` and `broken` is set on a killed round."""

    step: int
    round_idx: int | None
    round_id: str
    wr: float | None
    games: int | None
    ci: tuple[float, float] | None
    wilson: tuple[float, float] | None
    promoted: bool | None
    broken: bool


@dataclass(frozen=True)
class PromotionReading:
    """`state` is one of "promoted" / "not promoted" / "no decision taken"."""

    state: str
    step: int
    round_id: str


def _round_idx(round_id: Any) -> int | None:
    m = _ROUND_ID.match(str(round_id))
    return int(m.group(1)) if m else None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _ci(row: dict[str, Any]) -> tuple[float, float] | None:
    lo, hi = _num(row.get("wr_sealbot_ci_lower")), _num(row.get("wr_sealbot_ci_upper"))
    return (lo, hi) if lo is not None and hi is not None else None


def primary_rung(snap: Snapshot, ladder: dict[str, Any] | None) -> str:
    """The hero's rung: the channel-health rung, else the ladder's first, else `wr_sealbot`."""
    health = snap.last("eval_channel_health")
    if health and isinstance(health.get("rung"), str):
        return health["rung"]
    named = rungs(ladder)
    return named[0][0] if named else "sealbot (wr_sealbot)"


def rung_series(snap: Snapshot, ladder: dict[str, Any] | None) -> dict[str, list[RoundPoint]]:
    """Per rung, one point per round: the ladder row's (games, wr) joined by round index to the
    round event's step, promotion, broken flag and CI; without a ladder, the events alone."""
    rounds = [r for r in snap.rows("eval_round_complete") if isinstance(r.get("step"), int)]
    by_idx = {_round_idx(r.get("round_id")): r for r in rounds}
    out: dict[str, list[RoundPoint]] = {}
    primary = primary_rung(snap, ladder)
    for name, history in rungs(ladder):
        points = []
        for entry in history:
            idx = entry.get("round_idx")
            row = by_idx.get(idx if isinstance(idx, int) else None)
            if row is None:
                continue
            games = entry.get("games") if isinstance(entry.get("games"), int) else None
            wr = _num(entry.get("wr"))
            points.append(_point(row, wr=wr, games=games,
                                 ci=_ci(row) if name == primary else None))
        out[name] = sorted(points, key=lambda p: p.step)
    covered = {p.step for p in out.get(primary, [])}
    extra = [_point(r, wr=_num(r.get("wr_sealbot")), games=None, ci=_ci(r))
             for r in rounds if r["step"] not in covered]
    if extra:
        out[primary] = sorted(out.get(primary, []) + extra, key=lambda p: p.step)
    return out


def _point(row: dict[str, Any], *, wr: float | None, games: int | None,
           ci: tuple[float, float] | None) -> RoundPoint:
    broken = row.get("games_total") is None
    wilson = wilson_interval(wr, games) if (wr is not None and games) else None
    promoted = row.get("promoted") if isinstance(row.get("promoted"), bool) else None
    return RoundPoint(step=int(row["step"]), round_idx=_round_idx(row.get("round_id")),
                      round_id=str(row.get("round_id")), wr=None if broken else wr,
                      games=None if broken else games, ci=None if broken else ci,
                      wilson=None if broken else wilson,
                      promoted=None if broken else promoted, broken=broken)


def promotion_reading(snap: Snapshot) -> tuple[PromotionReading | None, PromotionReading | None]:
    """`(last decided round, latest completed round)`; each `None` when no such round exists."""
    rounds = [r for r in snap.rows("eval_round_complete") if isinstance(r.get("step"), int)]
    if not rounds:
        return None, None
    decided = None
    for row in rounds:
        if row.get("games_total") is None or not isinstance(row.get("promoted"), bool):
            continue
        decided = PromotionReading("promoted" if row["promoted"] else "not promoted",
                                   int(row["step"]), str(row.get("round_id")))
    last = rounds[-1]
    if last.get("games_total") is None:
        state = "broken round — no decision taken"
    elif isinstance(last.get("promoted"), bool):
        state = "promoted" if last["promoted"] else "not promoted"
    else:
        state = "no decision taken"
    return decided, PromotionReading(state, int(last["step"]), str(last.get("round_id")))


def trend(points: list[RoundPoint]) -> SlopeFit | None:
    """OLS slope of Elo(WR) on step, in Elo per 1 000 steps, over completed rounds.

    Elo of a 0 % or 100 % round is not finite, so the rate is clamped by half a game
    (`clamp_wr`) where the games count is known and skipped where it is not.
    """
    xs, ys = [], []
    for p in points:
        if p.broken or p.wr is None:
            continue
        wr = clamp_wr(p.wr, p.games) if p.games else p.wr
        elo = elo_of_wr(wr)
        if not math.isfinite(elo):
            continue
        xs.append(p.step / 1000.0)
        ys.append(elo)
    return ols_slope(xs, ys)
