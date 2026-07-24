"""⊕ WP11-A — A3 regime_key discipline + LAW-04 dedupe + low-power guard (mantis.eval.aggregate).

RED-at-import until IMPL writes `mantis.eval.aggregate`. Game records are plain dicts
carrying `p1`, `p2`, `winner` ("p1"|"p2"|"draw"), `regime_key` (the canonical `str` form
`RegimeKey.canonical()` produces — this suite never imports `mantis.arena.regime` so it
stays decoupled from the arena package; it treats `regime_key` as an opaque string tag,
exactly what a JSON-serialized game record carries) and `trajectory_hash` (a sha256-shaped
opaque string standing in for the real move-list hash; LAW-04 dedupe keys ONLY on this
field, so a synthetic string is a faithful substitute for arena's real hash).

`aggregate_rung` RAISES `MixedRegimeError` the instant more than one distinct `regime_key`
appears in one call — A3's core invariant (every eval record is regime_key'd; an aggregator
that silently pooled two regimes would corrupt every rating built on top of it).
`test_low_power_guard_blocks_promotion` duck-types `gate_cfg` (a `SimpleNamespace` with the
`GateConfig` field names `aggregate_gate` reads) rather than constructing the full pydantic
schema — this suite's job is the AGGREGATE-CONSTRUCTION-level guard mechanics; the full
run3 promotion truth table (schema-config-driven, pooled screen+confirm arithmetic) is
`tests/eval/test_gate_parity.py`'s pin, not this file's.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mantis.eval.aggregate import MixedRegimeError, aggregate_gate, aggregate_rung, pair_bootstrap_wr_ci


def _record(*, p1: str, p2: str, winner: str, regime_key: str, traj: str) -> dict:
    return {"p1": p1, "p2": p2, "winner": winner, "regime_key": regime_key, "trajectory_hash": traj}


def test_mixed_regime_keys_in_one_aggregation_raises() -> None:
    records = [
        _record(p1="cand", p2="rung_a", winner="p1", regime_key="rk_A", traj="h1"),
        _record(p1="cand", p2="rung_a", winner="p2", regime_key="rk_B", traj="h2"),
    ]
    with pytest.raises(MixedRegimeError) as excinfo:
        aggregate_rung(records)
    message = str(excinfo.value)
    assert "rk_A" in message and "rk_B" in message


def test_trajectory_hash_dedupe_collapses_copies_to_eff_n() -> None:
    # 40 raw records, all the SAME move sequence (same trajectory_hash) — the deterministic
    # deploy regime's known failure mode (§D-ARGMAX heritage). LAW-04: the effective sample
    # size for a CI must count this as 1 distinct game, not 40.
    records = [
        _record(p1="cand", p2="rung_a", winner="p1", regime_key="rk", traj="same_traj")
        for _ in range(40)
    ]
    agg = aggregate_rung(records)
    assert agg.games == 40
    assert agg.eff_n == 1


def test_trajectory_hash_dedupe_preserves_distinct_games() -> None:
    records = [
        _record(p1="cand", p2="rung_a", winner="p1", regime_key="rk", traj=f"traj_{i}")
        for i in range(12)
    ]
    agg = aggregate_rung(records)
    assert agg.games == 12
    assert agg.eff_n == 12


def test_pair_bootstrap_lower_ci_all_wins_and_n_zero_degenerates() -> None:
    all_wins = np.ones(50, dtype=np.float64)
    lo, hi = pair_bootstrap_wr_ci(all_wins, resamples=1000, ci_level=0.95, seed=1234)
    assert lo is not None and lo > 0.9
    assert hi is not None and hi <= 1.0 + 1e-9

    lo_zero, hi_zero = pair_bootstrap_wr_ci(
        np.zeros(0, dtype=np.float64), resamples=1000, ci_level=0.95, seed=1234
    )
    assert lo_zero is None and hi_zero is None  # zero games -> None + guard, never an exception


def test_low_power_guard_blocks_promotion() -> None:
    # 20 screen games + 20 confirm games, all WINS for cand (a clear point-estimate pass),
    # but only 2 DISTINCT trajectories repeated 20x each in both halves — distinct-per-pair
    # (2) is far below min_distinct_per_pair (10). The low_power guard must trip and BLOCK
    # promotion even though the raw win rate alone would clear the bar (run3 :555-563 parity:
    # low_power blocks regardless of wr_ok/ci_clean).
    gate_cfg = SimpleNamespace(
        promotion_winrate=0.55,
        min_distinct_per_pair=10,
        bootstrap_resamples=200,
        seed_base=20260625,
    )
    screen = [
        _record(p1="cand", p2="best", winner="p1", regime_key="rk", traj=f"screen_traj_{i % 2}")
        for i in range(20)
    ]
    confirm = [
        _record(p1="cand", p2="best", winner="p1", regime_key="rk", traj=f"confirm_traj_{i % 2}")
        for i in range(20)
    ]
    agg = aggregate_gate(screen, confirm, gate_cfg)
    assert agg.low_power is True
    assert agg.promoted is False
