"""The knowledge horizon: how often the sign of the search's root value at a turn's first stone matches the result, by turns to the end."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any

from .hexlogic import first_of_turn, owner, turn_of

#: Turns to the end the chart reaches; a later position is counted at this cap.
K_MAX = 30
#: Agreement the reach is read at, and the positions a turn needs before it is drawn.
REACH_SHARE = 0.90
MIN_POSITIONS = 20
#: The early and late windows are this share of the sampled games, first and last in record order.
WINDOW = 0.2


@dataclass(frozen=True)
class Curve:
    """One window and outcome: `share[k]` and `n[k]` for k = 0..K_MAX turns to the end (None below `MIN_POSITIONS`)."""

    share: tuple[float | None, ...]
    n: tuple[int, ...]


@dataclass(frozen=True)
class Horizon:
    """The four curves, each window's reach and game count, and the steps the windows span."""

    early_won: Curve
    early_lost: Curve
    late_won: Curve
    late_lost: Curve
    early_reach: int | None
    late_reach: int | None
    games: int
    window_games: int
    early_steps: tuple[int, int] | None
    late_steps: tuple[int, int] | None


class HorizonReducer:
    """Observes game records; keeps, per sampled decided self-play game, `(k, mover won, agrees)` triples and its step."""

    def __init__(self) -> None:
        self._games: list[array[int]] = []
        self._steps: list[int] = []

    def __call__(self, game: dict[str, Any]) -> None:
        stats, result = game.get("search_stats"), game.get("result")
        if game.get("channel") != "selfplay" or not isinstance(stats, list) or result not in ("p1", "p2"):
            return
        moves = game.get("moves") or []
        if not moves:
            return
        last_turn, winner = turn_of(len(moves) - 1), 0 if result == "p1" else 1
        rows: array[int] = array("h")
        for entry in stats:
            ply, v = entry.get("ply") if isinstance(entry, dict) else None, entry.get("root_value") if isinstance(entry, dict) else None
            if not isinstance(ply, int) or isinstance(v, bool) or not isinstance(v, (int, float)) or not first_of_turn(ply):
                continue
            won = owner(ply) == winner
            agrees = (v > 0) if won else (v < 0)
            rows.extend((min(K_MAX, last_turn - turn_of(ply)), int(won), int(agrees)))
        if rows:
            self._games.append(rows)
            step = game.get("step")
            self._steps.append(step if isinstance(step, int) and not isinstance(step, bool) else -1)

    def read(self) -> Horizon | None:
        """The horizon over the games seen so far, or None before any sampled decided self-play game."""
        total = len(self._games)
        if total == 0:
            return None
        size = max(1, int(total * WINDOW))
        early, late = self._games[:size], self._games[total - size:]
        ew, el, er = _curves(early)
        lw, ll, lr = _curves(late)
        return Horizon(early_won=ew, early_lost=el, late_won=lw, late_lost=ll, early_reach=er, late_reach=lr,
                       games=total, window_games=size, early_steps=_span(self._steps[:size]),
                       late_steps=_span(self._steps[total - size:]))


def _span(steps: list[int]) -> tuple[int, int] | None:
    known = [s for s in steps if s >= 0]
    return (min(known), max(known)) if known else None


def _curves(games: list[array[int]]) -> tuple[Curve, Curve, int | None]:
    hits = [[0] * (K_MAX + 1) for _ in range(4)]
    for rows in games:
        for i in range(0, len(rows), 3):
            k, won, agrees = rows[i], rows[i + 1], rows[i + 2]
            hits[won][k] += 1
            hits[2 + won][k] += agrees
    def curve(won: int) -> Curve:
        n = tuple(hits[won])
        return Curve(share=tuple(hits[2 + won][k] / n[k] if n[k] >= MIN_POSITIONS else None for k in range(K_MAX + 1)),
                     n=n)
    reach: int | None = None
    for k in range(K_MAX + 1):
        n = hits[0][k] + hits[1][k]
        if n < MIN_POSITIONS or (hits[2][k] + hits[3][k]) / n < REACH_SHARE:
            break
        reach = k
    return curve(1), curve(0), reach
