"""The strength-floor gate — the cheap probe that decides whether the EXPENSIVE ladder runs.

MEASURED GROUNDS (F-R-P2B-5). A terminal eval round at training step 33 spent its entire
`monitor.drain.terminal_eval_hard_cap_sec` budget and completed ZERO spec'd games. The worker
was healthy the whole time — single process, ~73-74% CPU, GPU active, 2:54:53 of CPU time at
the cap — so nothing wedged; the round was simply asked for more games than a near-random
candidate can finish at the configured search width. `run_round`'s phase order is what turned
that into a total loss: the gate block runs FIRST and is the round's most expensive phase, so
the budget was gone before the cheapest opponent in the spec was ever reached.

WHAT THIS MODULE IS. Two pure functions over already-played game records, plus the verdict
type that carries their arithmetic. It plays nothing, spawns nothing, and reads no config: the
worker owns the probe games and `mantis.config.resolve.eval_posture` owns the terms. That
split is deliberate — the decision rule is the part worth testing without a GPU, a book, or a
subprocess.

WHY DECISIVENESS AND NOT ONLY WIN RATE. The same burn measured `draw_rate` 1.0 with
`avg_game_length` at the arena's 128-move cap: every game was a ply-cap non-result. A win-rate
bar alone reads such a round as a perfectly healthy 0.5 — the draw-aware win rate of an
all-draw set is exactly 0.5 — and would let the ladder run on a candidate that has never
finished a game. `decisive_rate` is the axis that separates "the two sides are evenly matched"
from "neither side can finish", and it is measured from the arena's recorded `terminal` field
rather than re-derived from `(winner, plies)`, which cannot tell a win found ON the cap ply
from the cap itself.

THE VERDICT IS ADVISORY UNTIL ARMED. Every committed config mints `eval.strength_floor: null`,
so `evaluate_strength_floor` has no caller in a shipped run and the round's phase order is
untouched. Arming it is a mint event whose three values are operator-owned prereg rows.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mantis.arena.adjudicate import TERMINAL_PLY_CAP

#: The floor probe's own regime label. It is NOT `"random"` even though the probe plays the
#: random opponent: a probe game and a `random_floor_games` game are scored by different rules
#: (the probe's outcome gates a round; the floor's outcome is reported as `wr_random`), and
#: `aggregate_rung`'s MixedRegimeError exists precisely so two differently-purposed sets never
#: pool. Keeping the label distinct is what stops a future change from pooling them silently.
FLOOR_PROBE_VARIANT = "floor_probe"


@dataclass(frozen=True)
class StrengthFloorVerdict:
    """One floor decision plus every number that produced it.

    `passed` is the only field the round branches on; the rest exist so the emitted event can
    show HOW the bar was met or missed. LAW-18's complaint about a bare flag is exactly this:
    a `False` with no measurement beside it cannot distinguish a starved probe from a failing
    candidate.
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

    `draw_aware_wins` counts a draw as half a win — the same convention
    `worker.py::_draw_aware_wr` and `aggregate.py` already use for every other win rate in
    this package, so the floor's number is comparable with the ones beside it rather than a
    second definition of "win rate" in the same result payload.
    """
    games = len(records)
    decisive_games = sum(1 for rec in records if rec.terminal != TERMINAL_PLY_CAP)
    draws = sum(1 for rec in records if rec.winner == "draw")
    wins = sum(1.0 for rec in records if rec.winner == "candidate") + 0.5 * draws
    return games, decisive_games, wins, draws


def evaluate_strength_floor(records: Sequence[Any], spec: Any) -> StrengthFloorVerdict:
    """Decide the floor from the probe's records and the resolved `StrengthFloorSpec`.

    Both bars must hold, and BOTH are reported whether or not either fails — a verdict that
    stopped at the first failing bar would make the other one invisible, and an operator
    re-tuning the floor needs to see the axis they are not currently failing on.

    An EMPTY probe fails, and fails loudly rather than dividing by zero: zero games is zero
    evidence, and a floor that passes on no evidence is the phantom-gate class LAW-07 exists
    to prevent. `probe_games >= 1` makes an empty probe unreachable through a validated
    config, so this arm is defence in depth, not the expected path.
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
