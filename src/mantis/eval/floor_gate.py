"""The strength-floor gate — the cheap probe that decides whether the EXPENSIVE ladder runs.

MEASURED GROUNDS: a terminal eval round spent its whole hard-cap budget and completed ZERO
spec'd games on a healthy worker, because the gate block ran first and is the round's most
expensive phase. Two pure functions over already-played records plus their verdict type — it
plays nothing, spawns nothing and reads no config, so the rule is testable without a GPU.

DECISIVENESS AND NOT ONLY WIN RATE, because a win-rate bar reads an all-draw set (that burn
measured `draw_rate` 1.0 at the arena's move cap) as a healthy 0.5. `decisive_rate` comes from
the arena's recorded `terminal` field, which `(winner, plies)` cannot reconstruct. The verdict
IS armed in two committed configs, and changing its three values is a mint event.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mantis.arena.adjudicate import TERMINAL_PLY_CAP

#: The floor probe's own regime label, deliberately NOT `"random"` even though the probe plays
#: the random opponent: the two sets are scored by different rules and must never pool.
FLOOR_PROBE_VARIANT = "floor_probe"


@dataclass(frozen=True)
class StrengthFloorVerdict:
    """One floor decision plus every number that produced it.

    `passed` is the only field the round branches on; the rest exist so the emitted event can
    show HOW the bar was met or missed — a bare `False` cannot distinguish a starved probe from
    a failing candidate.
    """

    passed: bool
    games: int
    decisive_games: int
    decisive_rate: float
    wins: float
    draws: int
    winrate: float
    min_decisive_rate: float
    min_winrate: float
    failed_bars: tuple[str, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "games": self.games,
            "decisive_games": self.decisive_games,
            "decisive_rate": self.decisive_rate,
            "wins": self.wins,
            "draws": self.draws,
            "winrate": self.winrate,
            "min_decisive_rate": self.min_decisive_rate,
            "min_winrate": self.min_winrate,
            "failed_bars": list(self.failed_bars),
        }


def probe_measurements(records: Sequence[Any]) -> tuple[int, int, float, int]:
    """`(games, decisive_games, draw_aware_wins, draws)` over arena `GameRecord`s.

    A draw counts as half a win, the convention every other win rate in this package uses, so
    the floor's number is comparable with the ones beside it.
    """
    games = len(records)
    decisive_games = sum(1 for rec in records if rec.terminal != TERMINAL_PLY_CAP)
    draws = sum(1 for rec in records if rec.winner == "draw")
    wins = sum(1.0 for rec in records if rec.winner == "candidate") + 0.5 * draws
    return games, decisive_games, wins, draws


def evaluate_strength_floor(records: Sequence[Any], spec: Any) -> StrengthFloorVerdict:
    """Decide the floor from the probe's records and the resolved `StrengthFloorSpec`.

    Both bars must hold and BOTH are reported either way, so an operator re-tuning the floor
    sees the axis they are not currently failing on. An EMPTY probe fails rather than dividing
    by zero: zero games is zero evidence.
    """
    games, decisive_games, wins, draws = probe_measurements(records)
    decisive_rate = (decisive_games / games) if games else 0.0
    winrate = (wins / games) if games else 0.0

    failed: list[str] = []
    if games <= 0:
        failed.append("no_probe_games")
    if decisive_rate < spec.min_decisive_rate:
        failed.append("decisive_rate")
    if winrate < spec.min_winrate:
        failed.append("winrate")

    return StrengthFloorVerdict(
        passed=not failed,
        games=games,
        decisive_games=decisive_games,
        decisive_rate=decisive_rate,
        wins=wins,
        draws=draws,
        winrate=winrate,
        min_decisive_rate=float(spec.min_decisive_rate),
        min_winrate=float(spec.min_winrate),
        failed_bars=tuple(failed),
    )


__all__ = [
    "FLOOR_PROBE_VARIANT",
    "StrengthFloorVerdict",
    "evaluate_strength_floor",
    "probe_measurements",
]
