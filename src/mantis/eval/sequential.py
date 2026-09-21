"""The sequential promotion gate (`eval.gate.sequential`): a GSPRT over opening-pair outcomes; grounds in `docs/design/eval_gate_memo_2026-09-15.md` §3."""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from mantis.eval.aggregate import pair_units

#: The floor on the pair-score standard deviation: a window of identical scores has sd 0 and an
#: undefined t̂; the memo's simulation ran with this floor and its error curve assumes it.
SD_FLOOR = 0.05

Stopped = Literal["accept", "reject", "max"]
Decision = Literal["accept", "reject", "continue"]
AtMaxPairs = Literal["sign", "promote"]


@dataclass(frozen=True)
class SequentialGateSpec:
    """The rule's eight minted values. Raises: ValueError on a band that cannot stop or an unknown cap rule."""

    mu0: float
    mu1: float
    alpha: float
    beta: float
    check_every_pairs: int
    min_pairs: int
    max_pairs: int
    at_max_pairs: AtMaxPairs

    def __post_init__(self) -> None:
        if self.at_max_pairs not in ("sign", "promote"):
            raise ValueError(f"sequential gate: at_max_pairs must be 'sign' or 'promote', got {self.at_max_pairs!r}")
        if not 0.0 < self.mu0 < self.mu1 < 1.0:
            raise ValueError(f"sequential gate: need 0 < mu0 < mu1 < 1, got mu0={self.mu0} mu1={self.mu1}")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError(f"sequential gate: alpha and beta must lie in (0, 1), got {self.alpha}, {self.beta}")
        if self.check_every_pairs < 1:
            raise ValueError(f"sequential gate: check_every_pairs must be >= 1, got {self.check_every_pairs}")
        if not 1 <= self.min_pairs <= self.max_pairs:
            raise ValueError(f"sequential gate: need 1 <= min_pairs <= max_pairs, got {self.min_pairs}, {self.max_pairs}")


@dataclass(frozen=True)
class SequentialVerdict:
    decision: Literal["promote", "reject"]
    stopped: Stopped
    llr: float
    pairs_played: int
    checks: int
    llr_lower: float
    llr_upper: float


def llr_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's `(lower, upper)`: reject at or below `log(β/(1−α))`, accept at or above `log((1−β)/α)`."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def gsprt_llr(pair_scores: Sequence[float], mu0: float, mu1: float) -> float:
    """Van den Bergh's normalized-t LLR (4.14) over pair scores, σ̂ floored at `SD_FLOOR`; raises ValueError below two scores."""
    n = len(pair_scores)
    if n < 2:
        raise ValueError(f"the sequential gate's LLR needs >= 2 pair scores, got {n}")
    mean = statistics.fmean(pair_scores)
    sd = max(statistics.stdev(pair_scores), SD_FLOOR)
    t, t0, t1 = (mean - 0.5) / sd, (mu0 - 0.5) / sd, (mu1 - 0.5) / sd
    return 0.5 * n * math.log((1.0 + (t - t0) ** 2) / (1.0 + (t - t1) ** 2))


def gsprt_decision(llr: float, lower: float, upper: float, *, at_max: bool, at_max_pairs: AtMaxPairs) -> Decision:
    """`accept` at/above the upper bound, `reject` at/below the lower, else `continue`; at the maximum `at_max_pairs` decides — the LLR's SIGN, or `promote` (accept)."""
    if llr >= upper:
        return "accept"
    if llr <= lower:
        return "reject"
    if at_max:
        if at_max_pairs == "promote":
            return "accept"
        return "accept" if llr > 0.0 else "reject"
    return "continue"


def run_sequential_gate(
    play_pairs: Callable[[int, int], Sequence[Mapping[str, Any]]],
    spec: SequentialGateSpec,
) -> tuple[list[Mapping[str, Any]], SequentialVerdict]:
    """Play `[0, min_pairs)`, then `check_every_pairs` at a time to `max_pairs`, reading the LLR after each batch."""
    lower, upper = llr_bounds(spec.alpha, spec.beta)
    records: list[Mapping[str, Any]] = []
    played = 0
    checks = 0
    next_end = spec.min_pairs
    while True:
        records.extend(play_pairs(played, next_end))
        played = next_end
        checks += 1
        llr = gsprt_llr(pair_units(records), spec.mu0, spec.mu1)
        at_max = played >= spec.max_pairs
        decision = gsprt_decision(llr, lower, upper, at_max=at_max, at_max_pairs=spec.at_max_pairs)
        if decision == "continue":
            next_end = min(played + spec.check_every_pairs, spec.max_pairs)
            continue
        stopped: Stopped = "max" if at_max and not (llr >= upper or llr <= lower) else decision
        return records, SequentialVerdict(
            decision="promote" if decision == "accept" else "reject", stopped=stopped, llr=llr,
            pairs_played=played, checks=checks, llr_lower=lower, llr_upper=upper,
        )


__all__ = [
    "SD_FLOOR",
    "AtMaxPairs",
    "SequentialGateSpec",
    "SequentialVerdict",
    "gsprt_decision",
    "gsprt_llr",
    "llr_bounds",
    "run_sequential_gate",
]
