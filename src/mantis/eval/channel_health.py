"""External-channel health: the SATURATION rule and the DEGRADATION flag (R343(b)(iii)/(iv)).

Both are READINGS over the external channel's recent history, not new measurements — which is
why they live beside `aggregate.py` and consume its output rather than re-deriving a win rate.

SATURATION (iii). A fixed-depth opponent stops being an instrument once the candidate beats it
almost always: at 32 games a CI cannot separate 0.85 from 0.95, so every further round reports
"still winning" and none of them report PROGRESS. The rule names that state — pooled external
WR over the last four rounds `>= 0.85` — and its consequence is about what may be CLAIMED:
strength claims from a saturated rung answer to the next rung, not to this one.

DEGRADATION (iv). A pooled external WR more than `2 x CI` below its own running maximum WHILE
PROMOTIONS CONTINUE is the self-play-cycling signature: the net keeps beating its own anchor
(so the gate keeps promoting) while getting weaker against a FIXED opponent, which is what an
internal-only bar cannot see by construction. The conjunction is the whole discriminator — a
drop with promotions STOPPED is an ordinary plateau, and flagging it would be noise.

WARN-ONLY FOR RUN6 (G-3 stands): this module returns a verdict and never stops a run. Two
CONSECUTIVE flags are an architect read, which is a decision for a person; the counter is here
so that decision is made against a number rather than an impression (LAW-18).

ONE CI AUTHORITY. The pooled interval comes from `aggregate.pair_bootstrap_wr_ci`, the same
function the per-rung gate reads, over an outcome vector reconstructed from the window's own
win/game counts. A second interval estimator here would be a second answer to "is this
difference real", free to disagree with the gate's the first time it mattered.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from mantis.eval.aggregate import pair_bootstrap_wr_ci

#: R343(b)(iii)'s threshold, verbatim. Not a config key: it is a property of the INSTRUMENT
#: (a 32-game battery's resolving power), not of the run, and a run that could lower it could
#: keep claiming progress from a rung that has stopped measuring any.
SATURATION_WR = 0.85

#: The window both rules read, in ROUNDS. R343(b)(iii) says "the last four rounds".
WINDOW_ROUNDS = 4

#: R343(b)(iv)'s multiplier: the drop must exceed TWICE the pooled CI half-width.
DEGRADATION_CI_MULTIPLE = 2.0


@dataclasses.dataclass(frozen=True)
class RoundReading:
    """One external-channel round: how many games, how many the candidate won, and whether the
    PROMOTION channel promoted in the same round. `promoted` is what makes (iv) a signature
    rather than a plateau detector."""

    round_idx: int
    games: int
    wins: int
    promoted: bool
    #: WHICH rung produced this reading. Load-bearing, and AUDIT-1 F-14 is why: the sealbot WR
    #: alone is NOT a series — *"once `sealbot_d5` saturates it draws 0 games off-cadence and
    #: the reported number silently becomes `sealbot_d6`'s, so a trajectory rule testing
    #: `wr < peak * ratio` compares two opponents"* (`eval/rounds.py::_first_sealbot_wr`). Both
    #: rules here are trajectory rules over a POOLED WINDOW, so they are exposed to exactly
    #: that, and `_window` below is what refuses to pool across an identity change.
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
    """The trailing window, TRUNCATED at the most recent rung-identity change.

    AUDIT-1 F-14 applied to a pooled window: comparing a pooled WR against a running maximum
    taken over a DIFFERENT opponent is not a trajectory, it is two instruments averaged. So the
    window stops at the first reading (walking back) whose rung differs from the newest one. A
    window that shrinks to one round is the honest answer after a rung change, and both rules
    degrade gracefully — saturation still reads, and the running maximum has fewer prior
    windows to draw on until the new rung has history of its own.
    """
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
    """The window's games as a flat 1/0 outcome vector.

    Reconstructed from counts rather than carried as raw records: the per-round outcome arrays
    do not survive the round-result seam, and a pooled binomial over `sum(wins)` of
    `sum(games)` is exactly what "pooled WR over the last four rounds" denotes.
    """
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
    """Assess the external channel over the last `window` rounds of `history`.

    `history` is oldest-first. `previous_consecutive_flags` carries the streak across calls, so
    the caller owns the counter's lifetime and this function stays pure — the alternative is
    module state that a resumed run silently resets, which is the class R343(c) is about.

    THE RUNNING MAXIMUM IS TAKEN OVER PRIOR WINDOWS, NOT INCLUDING THE CURRENT ONE. A maximum
    that included the reading being judged can never be exceeded by it, so the drop would be
    measured against itself and (iv) could only fire on the first window that ever dropped.

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

    # Every EARLIER window, so the maximum is a fact about the run's past rather than about
    # this reading. Windows are taken at the same width; a partial early window is included
    # because excluding it would blind the rule for the run's first `window` rounds.
    #
    # SCOPED TO THE CURRENT RUNG, and this is the second half of AUDIT-1 F-14 — the half a
    # truncating window alone does NOT fix. Truncation makes the CURRENT reading single-opponent;
    # if the running maximum still ranged over every earlier window, the rule would compare a
    # fresh `sealbot_d6` reading against `sealbot_d5`'s peak and report the ladder ADVANCING as
    # a degradation. A maximum and the value judged against it must be the same instrument.
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
