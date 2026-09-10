"""External-channel health: the SATURATION rule and the DEGRADATION flag.

Both are READINGS over the channel's recent history, so they consume `aggregate.py`'s output
rather than re-deriving a win rate — and the pooled interval comes from the same function the
per-rung gate reads. SATURATION: pooled external WR >= 0.85 over the last four rounds means the
opponent has stopped resolving (at 32 games a CI cannot separate 0.85 from 0.95), so strength
claims answer to the NEXT rung. DEGRADATION: a pooled WR more than `2 x CI` below its own
running maximum WHILE PROMOTIONS CONTINUE, a conjunction that is the whole discriminator — the
same drop with promotions stopped is a plateau. The module returns a verdict, never stops a run.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from mantis.eval.aggregate import pair_bootstrap_wr_ci

#: The saturation threshold, verbatim from the rule. Not a config key: it is a property of the
#: INSTRUMENT (a 32-game battery's resolving power), not of the run.
SATURATION_WR = 0.85

#: The window both rules read, in ROUNDS.
WINDOW_ROUNDS = 4

#: The degradation multiplier: the drop must exceed TWICE the pooled CI half-width.
DEGRADATION_CI_MULTIPLE = 2.0


@dataclasses.dataclass(frozen=True)
class RoundReading:
    """One external-channel round, plus whether the PROMOTION channel promoted in the same
    round — `promoted` is what makes degradation a signature rather than a plateau detector."""

    round_idx: int
    games: int
    wins: int
    promoted: bool
    #: WHICH rung produced this reading: a saturated rung draws 0 games off-cadence and the
    #: reported number silently becomes the NEXT rung's, so pooling across an identity change
    #: would compare two opponents.
    rung: str


@dataclasses.dataclass(frozen=True)
class ChannelHealth:
    """The verdict a dashboard draws and an exit screen prints."""

    pooled_wr: float | None
    pooled_games: int
    ci_lower: float | None
    ci_upper: float | None
    rounds_pooled: int
    saturated: bool
    running_max_wr: float | None
    degraded: bool
    consecutive_degradation_flags: int

    @property
    def label(self) -> str:
        """The one word the dashboard prints beside the win rate."""
        if self.pooled_wr is None:
            return "NO-DATA"
        if self.degraded:
            return "DEGRADED"
        return "SATURATED" if self.saturated else "MEASURING"


def _window(history: Sequence[RoundReading], width: int) -> list[RoundReading]:
    """The trailing window, TRUNCATED at the most recent rung-identity change: a pooled WR
    compared against a maximum taken over a DIFFERENT opponent is two instruments averaged, not
    a trajectory. A one-round window is the honest answer after a rung change."""
    if not history:
        return []
    newest = history[-1].rung
    out: list[RoundReading] = []
    for r in reversed(history):
        if r.rung != newest or len(out) >= width:
            break
        out.append(r)
    return list(reversed(out))


def _pooled_outcomes(window: Sequence[RoundReading]) -> np.ndarray:
    """The window's games as a flat 1/0 outcome vector, reconstructed from counts because the
    per-round outcome arrays do not survive the round-result seam."""
    total = sum(r.games for r in window)
    wins = sum(r.wins for r in window)
    out = np.zeros(total, dtype=np.float64)
    out[:wins] = 1.0
    return out


def assess(
    history: Sequence[RoundReading],
    *,
    window: int = WINDOW_ROUNDS,
    saturation_wr: float = SATURATION_WR,
    ci_multiple: float = DEGRADATION_CI_MULTIPLE,
    bootstrap_resamples: int = 1000,
    bootstrap_ci_level: float = 0.95,
    bootstrap_seed: int = 0,
    previous_consecutive_flags: int = 0,
) -> ChannelHealth:
    """Assess the external channel over the last `window` rounds of `history` (oldest-first).

    `previous_consecutive_flags` carries the streak across calls, so the caller owns the
    counter's lifetime and this stays pure; module state would be silently reset by a resume.
    The running maximum is taken over PRIOR windows: one including the reading being judged
    could never be exceeded by it.

    Raises:
        ValueError: `window` is not positive, or a reading has more wins than games.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    for r in history:
        if r.wins > r.games or r.games < 0 or r.wins < 0:
            raise ValueError(
                f"round {r.round_idx} reports {r.wins} wins in {r.games} games — a reading that "
                "cannot have happened, and pooling it would produce a win rate above 1"
            )
    if not history:
        return ChannelHealth(None, 0, None, None, 0, False, None, False,
                             previous_consecutive_flags)

    current = _window(history, window)
    pooled_games = sum(r.games for r in current)
    if pooled_games == 0:
        return ChannelHealth(None, 0, None, None, len(current), False, None, False,
                             previous_consecutive_flags)
    pooled_wr = sum(r.wins for r in current) / pooled_games
    ci_lo, ci_hi = pair_bootstrap_wr_ci(
        _pooled_outcomes(current), resamples=bootstrap_resamples,
        ci_level=bootstrap_ci_level, seed=bootstrap_seed,
    )

    saturated = pooled_wr >= saturation_wr

    # Every EARLIER window, so the maximum is a fact about the run's past rather than about this
    # reading; a partial early window is included, or the rule is blind for the first rounds.
    # SCOPED TO THE CURRENT RUNG: otherwise the rule compares a fresh reading against the
    # previous opponent's peak and reports the ladder ADVANCING as a degradation.
    current_rung = current[-1].rung
    running_max = None
    for end in range(1, len(history)):
        prior = _window(list(history)[:end], window)
        if not prior or prior[-1].rung != current_rung:
            continue
        g = sum(r.games for r in prior)
        if g:
            wr = sum(r.wins for r in prior) / g
            running_max = wr if running_max is None else max(running_max, wr)

    half_width = None
    if ci_lo is not None and ci_hi is not None:
        half_width = (ci_hi - ci_lo) / 2.0

    promotions_continue = any(r.promoted for r in current)
    degraded = bool(
        running_max is not None
        and half_width is not None
        and promotions_continue
        and pooled_wr < running_max - ci_multiple * half_width
    )
    flags = previous_consecutive_flags + 1 if degraded else 0
    return ChannelHealth(
        pooled_wr=pooled_wr, pooled_games=pooled_games, ci_lower=ci_lo, ci_upper=ci_hi,
        rounds_pooled=len(current), saturated=saturated, running_max_wr=running_max,
        degraded=degraded, consecutive_degradation_flags=flags,
    )


__all__ = [
    "DEGRADATION_CI_MULTIPLE",
    "SATURATION_WR",
    "WINDOW_ROUNDS",
    "ChannelHealth",
    "RoundReading",
    "assess",
]
