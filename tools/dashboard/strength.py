"""Strength readings: round points per rung, the promotion reading, the Elo-slope witness."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .reader import Record
from .stats import SlopeFit, clamp_wr, elo_of_wr, ols_slope, wilson_interval

_ROUND_ID = re.compile(r"^r(\d+)_(\d+)$")


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


#: R362(c): the sealbot rung, its ladder file and `wr_sealbot` have no producer at HEAD; a record
#: without a single reading is drawn as this stated gap, never as a zero.
RUNG_RETIRED_NOTE = ("the sealbot rung was deleted by R362(c) — no round row carries "
                     "<code>wr_sealbot</code> and no ladder file exists for a run minted after it; "
                     "the run's external scale is the strix points panel")


def sealbot_readings_present(rec: Record) -> bool:
    """True iff some round row or ladder history carries a sealbot reading (a pre-R362 record)."""
    if rec.rungs():
        return True
    return any(_num(r.get("wr_sealbot")) is not None for r in rec.rows("eval_round_complete"))


def primary_rung(rec: Record) -> str:
    """The hero's rung: the channel-health rung, else the ladder's first, else `wr_sealbot`."""
    health = rec.last("eval_channel_health")
    if health and isinstance(health.get("rung"), str):
        return health["rung"]
    rungs = rec.rungs()
    return rungs[0][0] if rungs else "sealbot (wr_sealbot)"


def rung_series(rec: Record) -> dict[str, list[RoundPoint]]:
    """Per rung, one point per round: the ladder row's (games, wr) joined by round index to the
    round event's step, promotion, broken flag and CI; without a ladder, the events alone."""
    rounds = [r for r in rec.rows("eval_round_complete") if isinstance(r.get("step"), int)]
    by_idx = {_round_idx(r.get("round_id")): r for r in rounds}
    out: dict[str, list[RoundPoint]] = {}
    primary = primary_rung(rec)
    for name, history in rec.rungs():
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


def promotion_reading(rec: Record) -> tuple[PromotionReading | None, PromotionReading | None]:
    """`(last decided round, latest completed round)`; each `None` when no such round exists."""
    rounds = [r for r in rec.rows("eval_round_complete") if isinstance(r.get("step"), int)]
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
