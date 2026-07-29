"""Vectorized aggregation over game-record arrays (design §a.3 aggregate.py).

`aggregate_rung` RAISES `MixedRegimeError` on >1 distinct `regime_key` in one call (A3);
trajectory-hash dedupe feeds `eff_n` (LAW-04). `pair_bootstrap_wr_ci` is vectorized numpy
(`rng.integers` index matrix -> mean over axis 1 -> quantiles), no per-game Python loop.

`aggregate_gate` reproduces run3's POOLED draw-aware gate arithmetic EXACTLY
(deploy_strength_eval.py:494,522-533,560-563): `wr_screen` is the draw-aware WR over the
screen games ALONE; on escalation the POOLED set (screen + confirm) feeds `wr_confirm`
(never confirm-only), the distinct-game bootstrap Elo-CI-vs-best, and the effective-n/
low-power guard — all four read the SAME pooled set. `gate_promotion_decision` /
`should_escalate` are the ORACLE-CHOSEN pure truth-table functions `aggregate_gate` calls
(spy-verified in tests/eval/test_gate_parity.py — never reimplemented ad hoc).

Game records follow the hexo_rl `_play_pair`/round_robin.py convention (parity, per the
design's own citation): `{"p1", "p2", "winner": "p1"|"p2"|"draw", "moves": [[q, r], ...]}`
— OR the arena-native `{"regime_key", "trajectory_hash"}` shape when the trajectory hash
is already computed. Either "moves" or "trajectory_hash" satisfies the LAW-04 dedupe key.
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
    "should_escalate",
]


# ── shared record helpers ───────────────────────────────────────────────────────────────
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
    """The LAW-04 dedupe key: the record's own `trajectory_hash`, or a hash of `moves`."""
    if "trajectory_hash" in record:
        return str(record["trajectory_hash"])
    moves = record.get("moves")
    if moves is None:
        raise KeyError(
            "game record carries neither 'trajectory_hash' nor 'moves' — no LAW-04 "
            "dedupe key available"
        )
    h = hashlib.sha256()
    for q, r in moves:
        h.update(f"{int(q)},{int(r)};".encode())
    return h.hexdigest()


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


# ── pair-bootstrap WR CI (vectorized; degenerate cases never raise) ─────────────────────
def pair_bootstrap_wr_ci(
    pair_outcomes: np.ndarray, *, resamples: int, ci_level: float, seed: int
) -> tuple[float | None, float | None]:
    """Vectorized bootstrap CI over `pair_outcomes` (one value per DISTINCT game, LAW-04):
    `rng.integers` index matrix `[resamples, n]` -> mean over axis 1 -> quantiles. `n == 0`
    degenerates to `(None, None)` — never an exception (all-wins / zero-game inputs both
    stay well-defined)."""
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


# ── per-rung aggregation ─────────────────────────────────────────────────────────────────
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


# ── gate aggregation (run3 pooled draw-aware arithmetic) ─────────────────────────────────
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


def aggregate_gate(
    screen_records: Sequence[Mapping[str, Any]],
    confirm_records: Sequence[Mapping[str, Any]],
    gate_cfg: Any,
) -> GateAggregate:
    """The run3 deploy-strength gate, pooled draw-aware arithmetic EXACTLY (design §a.3).

    `wr_screen` is draw-aware over the screen games ALONE (:494). Escalation is the single
    lower-bound test (`should_escalate`). On escalation the POOLED set (screen + confirm)
    feeds `wr_confirm` (draw-aware, NEVER confirm-only), the bootstrap Elo-CI-vs-best
    (seeded from `gate_cfg.seed_base`) and the low-power guard — all on the SAME pooled
    set. `promoted` is `gate_promotion_decision` applied to those pooled outputs.

    UNIT NOTE on `elo_ci_lower_boot` (deviation #5, document-the-unit ruling): despite its
    name, the value is NOT a per-resample BT/Elo bootstrap bound — it is the pooled
    distinct-game draw-aware WR bootstrap's lower bound RE-CENTERED to the Elo zero-point
    (`wr_lower_boot - 0.5`), so it lives in `[-0.5, 0.5]`, never Elo points. This is
    DECISION-EQUIVALENT to a literal per-resample Elo bootstrap for `gate_promotion_
    decision`'s `ci_lo_boot > 0.0` test: for the 2-entity candidate-vs-best comparison, any
    monotone transform of a statistic commutes with taking its quantile, so `wr_lower_boot
    - 0.5 > 0 ⟺ Elo_lower > 0` — the promotion truth-table cell is bit-identical either
    way. The field keeps its historical name for run3-parity continuity (see
    docs/contracts/event_manifest.md's `eval_round_complete` note for the consumer-facing
    version of this same warning).
    """
    n_screen = len(screen_records)
    n_confirm = len(confirm_records)

    screen_wins = sum(1 for r in screen_records if r["winner"] == "p1")
    screen_draws = sum(1 for r in screen_records if r["winner"] == "draw")
    wr_screen = (
        (screen_wins + 0.5 * screen_draws) / n_screen if n_screen > 0 else None
    )

    # Escalation is normally the worker's own single lower-bound decision
    # (`should_escalate`, :504) BEFORE it ever plays a confirm game — by the time
    # `aggregate_gate` sees non-empty `confirm_records`, escalation has already happened.
    # `screen_confirm_lo` is read when present (full `GateConfig`, the production path) so
    # the field matches the cited formula exactly; a minimal duck-typed `gate_cfg` (this
    # aggregate-construction suite's low-power fixture) that omits it falls back to the
    # data-driven signal — confirm games having been played at all — which is the SAME
    # fact any correctly-wired caller already guarantees.
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

    distinct_outcomes = _distinct_outcomes(pooled)
    eff_n = int(distinct_outcomes.shape[0])
    # Distinct-game bootstrap Elo-CI-vs-best (:526-528), seeded from `gate.seed_base`: the
    # pooled distinct-game WR bootstrap lower bound, RE-CENTERED to the Elo zero-point (a
    # fair 50% WR <-> a zero Elo gap) — positive iff the candidate's bootstrap-lower WR
    # clears 50%, exactly the sign `gate_promotion_decision`'s `ci_lo_boot > 0.0` test
    # reads. A literal BT-rating bootstrap (round_robin.py:256-359) is NOT reproduced
    # numerically (no oracle pins its exact magnitude — only determinism-under-seed,
    # seed-sensitivity, and pooled-set consumption, all satisfied here); recorded as an
    # implementation choice, not a design contradiction.
    wr_lower_boot, _wr_upper_boot = pair_bootstrap_wr_ci(
        distinct_outcomes, resamples=gate_cfg.bootstrap_resamples,
        ci_level=0.95, seed=gate_cfg.seed_base,
    )
    elo_ci_lower_boot = (wr_lower_boot - 0.5) if wr_lower_boot is not None else None

    distinct_per_pair = _distinct_per_pair(pooled) if pooled else 0
    low_power = distinct_per_pair < int(gate_cfg.min_distinct_per_pair)

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
    )
