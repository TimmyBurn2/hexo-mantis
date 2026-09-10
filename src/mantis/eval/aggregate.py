# >300 justify (R8): one arithmetic unit. Every function here reads the SAME record convention
# and feeds ONE decision, the promotion gate, in a chain where each step's output is the next
# step's input. Splitting it would put the dedupe key in one file and the estimator that depends
# on its semantics in another — and the defect this file most recently carried was exactly a
# dedupe key whose meaning had drifted from what the estimator assumed. The run3 parity
# citations are load-bearing for the same reason: they sit beside the arithmetic they certify.
"""Vectorized aggregation over game-record arrays (design §a.3).

`aggregate_rung` RAISES `MixedRegimeError` on >1 distinct `regime_key` in one call, and
trajectory-hash dedupe feeds `eff_n` (LAW-04). `pair_bootstrap_wr_ci` is vectorized numpy with
no per-game Python loop.

`aggregate_gate` reproduces run3's POOLED draw-aware gate arithmetic EXACTLY
(deploy_strength_eval.py:494,522-533,560-563): `wr_screen` is draw-aware over the screen games
ALONE, and on escalation the POOLED set feeds `wr_confirm`, the bootstrap Elo-CI-vs-best and
the low-power guard. `gate_promotion_decision` / `should_escalate` are the oracle-chosen pure
truth-table functions it calls, never reimplemented ad hoc.

Game records follow the hexo_rl `_play_pair` convention, or the arena-native
`{"regime_key", "trajectory_hash"}` shape; either satisfies the LAW-04 dedupe key.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mantis.arena.regime import MixedRegimeError

__all__ = [
    "GateAggregate",
    "MixedRegimeError",
    "RungAggregate",
    "aggregate_gate",
    "aggregate_rung",
    "gate_promotion_decision",
    "pair_bootstrap_wr_ci",
    "pair_units",
    "should_escalate",
]


def _outcome_value(record: Mapping[str, Any]) -> float:
    """1.0 (p1/candidate win), 0.0 (p2/opponent win), 0.5 (draw) — draw-aware."""
    winner = record["winner"]
    if winner == "p1":
        return 1.0
    if winner == "p2":
        return 0.0
    if winner == "draw":
        return 0.5
    raise ValueError(f"unrecognised winner label {winner!r} (expected p1/p2/draw)")


def _traj_key(record: Mapping[str, Any]) -> str:
    """The LAW-04 dedupe key: the trajectory, QUALIFIED BY WHO SAT WHERE.

    LAW-04 dedupes COPIES of one game, because a deterministic regime replays the same game and
    a CI over the raw count is over-confident by sqrt(copies). Two colour-swapped legs are not
    copies, but `trajectory_hash` is a sha256 over the MOVE LIST alone, so coinciding legs hash
    identically and the unqualified key threw one away — biasing the win rate toward whichever
    leg was seen first AND halving eff_n. `candidate_color` is absent on legacy records, which
    key as before, so this is additive.
    """
    seat = record.get("candidate_color")
    qualifier = f"{record.get('p1')}|{record.get('p2')}|{seat}|"
    if "trajectory_hash" in record:
        return qualifier + str(record["trajectory_hash"])
    moves = record.get("moves")
    if moves is None:
        raise KeyError(
            "game record carries neither 'trajectory_hash' nor 'moves' — no LAW-04 "
            "dedupe key available"
        )
    h = hashlib.sha256()
    for q, r in moves:
        h.update(f"{int(q)},{int(r)};".encode())
    return qualifier + h.hexdigest()


def _pair_key(record: Mapping[str, Any]) -> tuple[Any, Any]:
    return (record.get("p1"), record.get("p2"))


def _distinct_outcomes(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One outcome value per DISTINCT trajectory (LAW-04: copies of one game count once;
    the first-seen record for a trajectory supplies its outcome)."""
    seen: dict[str, float] = {}
    for record in records:
        key = _traj_key(record)
        if key not in seen:
            seen[key] = _outcome_value(record)
    return np.asarray(list(seen.values()), dtype=np.float64)


def _unit_key(record: Mapping[str, Any]) -> str:
    """The PAIR key: a matchup and an opening, with the SEAT deliberately absent — both legs of
    one opening are ONE observation, because they start from the same position and their
    outcomes are correlated. A record with NO `opening_id` falls back to its own trajectory key,
    since pairing legacy records on a missing field would collapse a round into one
    observation."""
    opening = record.get("opening_id")
    if opening is None:
        return _traj_key(record)
    return f"{record.get('p1')}|{record.get('p2')}|{opening}"


def pair_units(records: Sequence[Mapping[str, Any]]) -> list[float]:
    """One draw-aware outcome value per OPENING PAIR — the bootstrap's resampling unit. Both legs
    average into one value, and an opening with only one leg contributes that leg. Order is the
    first-seen order of the keys, which keeps a seeded bootstrap reproducible."""
    by_unit: dict[str, list[float]] = {}
    for record in records:
        by_unit.setdefault(_unit_key(record), []).append(_outcome_value(record))
    return [sum(values) / len(values) for values in by_unit.values()]


def _distinct_per_pair(records: Sequence[Mapping[str, Any]]) -> int:
    """Port of round_robin.py's `distinct_per_pair` semantics (:203-252): distinct-game
    dedup by `(p1, p2, tuple(moves))`, minimum count over the pairs present."""
    by_pair: dict[tuple[Any, Any], set[str]] = {}
    for record in records:
        pair = _pair_key(record)
        by_pair.setdefault(pair, set()).add(_traj_key(record))
    if not by_pair:
        return 0
    return min(len(trajs) for trajs in by_pair.values())


def pair_bootstrap_wr_ci(
    pair_outcomes: np.ndarray, *, resamples: int, ci_level: float, seed: int
) -> tuple[float | None, float | None]:
    """Vectorized bootstrap CI over `pair_outcomes` (one value per DISTINCT game, LAW-04).
    `n == 0` degenerates to `(None, None)` — never an exception."""
    arr = np.asarray(pair_outcomes, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(resamples), n))
    resample_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(resample_means, alpha))
    hi = float(np.quantile(resample_means, 1.0 - alpha))
    return lo, hi


@dataclass(frozen=True)
class RungAggregate:
    games: int
    wins: int
    losses: int
    draws: int
    wr: float | None
    wr_ci_lower: float | None
    wr_ci_upper: float | None
    eff_n: int
    regime_key: str


def aggregate_rung(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 1000,
    bootstrap_ci_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> RungAggregate:
    """Aggregate one rung's game records. Raises `MixedRegimeError` (A3) if the records
    carry more than one distinct `regime_key`."""
    if not records:
        return RungAggregate(
            games=0, wins=0, losses=0, draws=0, wr=None,
            wr_ci_lower=None, wr_ci_upper=None, eff_n=0, regime_key="",
        )
    regime_keys = {r["regime_key"] for r in records if r.get("regime_key") is not None}
    if len(regime_keys) > 1:
        raise MixedRegimeError(
            f"aggregate_rung: mixed regime_key set in one aggregation: {sorted(regime_keys)}"
        )
    regime_key = next(iter(regime_keys)) if regime_keys else ""

    games = len(records)
    wins = sum(1 for r in records if r["winner"] == "p1")
    losses = sum(1 for r in records if r["winner"] == "p2")
    draws = sum(1 for r in records if r["winner"] == "draw")
    wr = (wins + 0.5 * draws) / games if games > 0 else None

    distinct_outcomes = _distinct_outcomes(records)
    eff_n = int(distinct_outcomes.shape[0])
    wr_ci_lower, wr_ci_upper = pair_bootstrap_wr_ci(
        distinct_outcomes, resamples=bootstrap_resamples,
        ci_level=bootstrap_ci_level, seed=bootstrap_seed,
    )

    return RungAggregate(
        games=games, wins=wins, losses=losses, draws=draws, wr=wr,
        wr_ci_lower=wr_ci_lower, wr_ci_upper=wr_ci_upper, eff_n=eff_n, regime_key=regime_key,
    )


def should_escalate(wr_screen: float, screen_confirm_lo: float) -> bool:
    """The SINGLE lower-bound escalation test (deploy_strength_eval.py:504) — NO upper
    band (`screen_confirm_hi` was inert in run3 and is not ported, MUST-FIX 1)."""
    return wr_screen >= screen_confirm_lo


def gate_promotion_decision(
    wr_confirm: float, ci_lo_boot: float | None, low_power: bool, promotion_winrate: float
) -> bool:
    """The run3 promotion truth table (:560-563): `wr_ok AND ci_clean AND not low_power`."""
    wr_ok = wr_confirm >= promotion_winrate
    ci_clean = ci_lo_boot is not None and ci_lo_boot > 0.0
    return bool(wr_ok and ci_clean and not low_power)


@dataclass(frozen=True)
class GateAggregate:
    wr_screen: float | None
    wr_confirm: float | None
    n_screen: int
    n_confirm: int
    n_pooled: int
    escalated: bool
    elo_ci_lower_boot: float | None
    low_power: bool
    eff_n: int
    promoted: bool
    #: The pooled game counts, RETAINED beside the decision: a promotion recorded as a win rate
    #: and a boolean cannot be re-read afterwards, since 0.58 over 24 games and 0.58 over 240
    #: are the same field. Counted over GAMES while the CI is over PAIRS, deliberately — these
    #: are the raw tallies, not the estimator's unit.
    wins: int = 0
    losses: int = 0
    draws: int = 0


def aggregate_gate(
    screen_records: Sequence[Mapping[str, Any]],
    confirm_records: Sequence[Mapping[str, Any]],
    gate_cfg: Any,
) -> GateAggregate:
    """The run3 deploy-strength gate, pooled draw-aware arithmetic EXACTLY.

    UNIT NOTE on `elo_ci_lower_boot`: despite its name the value is NOT a per-resample BT/Elo
    bound — it is the pooled distinct-game WR bootstrap's lower bound RE-CENTERED to the Elo
    zero-point, so it lives in `[-0.5, 0.5]`. It is DECISION-EQUIVALENT for the
    `ci_lo_boot > 0.0` test, because any monotone transform commutes with taking a quantile.
    The field keeps its historical name for run3-parity continuity.
    """
    n_screen = len(screen_records)
    n_confirm = len(confirm_records)

    screen_wins = sum(1 for r in screen_records if r["winner"] == "p1")
    screen_draws = sum(1 for r in screen_records if r["winner"] == "draw")
    wr_screen = (
        (screen_wins + 0.5 * screen_draws) / n_screen if n_screen > 0 else None
    )

    # Escalation is normally the worker's own decision BEFORE it plays a confirm game.
    # `screen_confirm_lo` is read when present so the field matches the cited formula; a minimal
    # duck-typed `gate_cfg` that omits it falls back to confirm games having been played at all,
    # which any correctly-wired caller already guarantees.
    screen_confirm_lo = getattr(gate_cfg, "screen_confirm_lo", None)
    if screen_confirm_lo is not None and wr_screen is not None:
        escalated = should_escalate(wr_screen, screen_confirm_lo)
    else:
        escalated = n_confirm > 0

    pooled = list(screen_records) + list(confirm_records) if escalated else list(screen_records)
    n_pooled = len(pooled)
    pooled_wins = sum(1 for r in pooled if r["winner"] == "p1")
    pooled_draws = sum(1 for r in pooled if r["winner"] == "draw")
    wr_confirm = (pooled_wins + 0.5 * pooled_draws) / n_pooled if n_pooled > 0 else None

    # The resampling UNIT is the opening pair, not the game: resampling games treats two legs of
    # one opening as independent draws, understating the between-opening variance on the LOWER
    # bound `gate_promotion_decision` reads, so the bar cleared more often than its stated
    # confidence. `eff_n` moves with it, since an effective-n counted in games beside a CI over
    # pairs would be two answers to one question.
    unit_outcomes = np.asarray(pair_units(pooled), dtype=np.float64)
    eff_n = int(unit_outcomes.shape[0])
    # Distinct-game bootstrap Elo-CI-vs-best, seeded from `gate.seed_base`: the pooled
    # distinct-game WR bootstrap lower bound RE-CENTERED to the Elo zero-point, so it is
    # positive iff the candidate's bootstrap-lower WR clears 50% — exactly the sign
    # `gate_promotion_decision` reads. A literal BT-rating bootstrap is NOT reproduced
    # numerically, since no oracle pins its magnitude; recorded as an implementation choice.
    wr_lower_boot, _wr_upper_boot = pair_bootstrap_wr_ci(
        unit_outcomes, resamples=gate_cfg.bootstrap_resamples,
        ci_level=0.95, seed=gate_cfg.seed_base,
    )
    elo_ci_lower_boot = (wr_lower_boot - 0.5) if wr_lower_boot is not None else None

    distinct_per_pair = _distinct_per_pair(pooled) if pooled else 0
    low_power = distinct_per_pair < int(gate_cfg.min_distinct_per_pair)
    # `min_distinct_per_pair` is `Field(ge=1)`, so the low-power guard can be SATISFIED by ONE
    # distinct game — and a bootstrap over one sample collapses to the point estimate, so a
    # single distinct WIN re-centres to +0.5 > 0 and promotes. The smoke config deliberately
    # mints 1, so the floor cannot be raised in the schema; the refusal belongs to the
    # STATISTIC, which reports no interval below two distinct games.
    _MIN_DISTINCT_FOR_AN_INTERVAL = 2
    if distinct_per_pair < _MIN_DISTINCT_FOR_AN_INTERVAL:
        elo_ci_lower_boot = None

    promoted = (
        wr_confirm is not None
        and gate_promotion_decision(
            wr_confirm, elo_ci_lower_boot, low_power, gate_cfg.promotion_winrate
        )
    )

    return GateAggregate(
        wr_screen=wr_screen, wr_confirm=wr_confirm, n_screen=n_screen, n_confirm=n_confirm,
        n_pooled=n_pooled, escalated=escalated, elo_ci_lower_boot=elo_ci_lower_boot,
        low_power=low_power, eff_n=eff_n, promoted=promoted,
        wins=pooled_wins, losses=n_pooled - pooled_wins - pooled_draws, draws=pooled_draws,
    )
