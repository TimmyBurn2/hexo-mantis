"""AUDIT-1 F-27 — one distinct game is not a confidence interval, and cannot promote.

THE DEFECT. `gate_promotion_decision` is `wr_ok and ci_lo_boot > 0 and not low_power`. The
low-power guard is `_distinct_per_pair(pooled) < gate_cfg.min_distinct_per_pair`, and the
schema declares `min_distinct_per_pair: int = Field(ge=1)`. At 1, ONE distinct game satisfies
the guard — and a bootstrap over a single sample is not an interval: every resample draws the
same value, the "CI" collapses onto the point estimate, and a single distinct WIN re-centres
to +0.5 > 0. A candidate could be promoted to `best` on one game.

WHY THE SCHEMA FLOOR IS NOT WHAT MOVED. `configs/run5.yaml` and `shakedown_20260807.yaml` mint
10; `configs/smoke_preflight_armed.yaml` deliberately mints 1 (`# delta:
eval.gate.min_distinct_per_pair: 10 -> 1`) so the preflight smoke boots fast. Raising `ge=1`
would refuse a config that exists to be cheap. The refusal belongs to the STATISTIC: below two
distinct games there is no interval, so none is reported, and the promotion rule's
`ci_lo_boot is not None` arm cannot clear. The smoke config still boots and simply cannot
promote off one game — which is the right outcome either way.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.eval.aggregate import aggregate_gate, gate_promotion_decision


class _Gate:
    """The gate knobs `aggregate_gate` reads, at the smoke config's own floor of 1."""

    promotion_winrate = 0.55
    screen_confirm_lo = 0.4
    bootstrap_resamples = 64
    min_distinct_per_pair = 1
    seed_base = 7


def _record(traj: str, *, won: bool = True) -> dict[str, Any]:
    """The shape `aggregate_gate` reads — the same one `test_aggregate_regime.py` builds:
    `trajectory_hash` is the LAW-04 dedupe key, and `winner` is `p1`/`p2`."""
    return {"p1": "cand", "p2": "best", "winner": "p1" if won else "p2",
            "regime_key": "rk", "trajectory_hash": traj}


def _agg(records: list[dict[str, Any]]) -> Any:
    return aggregate_gate(records, records, _Gate())


def test_a_single_distinct_WIN_does_not_promote() -> None:
    """THE PIN. Before this the round promoted: wr 1.0 clears 0.55, the one-sample bootstrap
    re-centres to +0.5, and `low_power` is False at a floor of 1."""
    result = _agg([_record("t1")])
    assert result.low_power is False, (
        "the premise: at min_distinct_per_pair 1 the low-power guard is SATISFIED by one game"
    )
    assert result.elo_ci_lower_boot is None, (
        "a bootstrap over one distinct game reported an interval"
    )
    assert result.promoted is False


def test_two_distinct_games_DO_produce_an_interval() -> None:
    """The control: the repair must not disable the gate, only refuse the degenerate n."""
    result = _agg([_record("t1"), _record("t2")])
    assert result.elo_ci_lower_boot is not None
    assert result.promoted is True


def test_the_promotion_rule_still_refuses_a_None_interval_by_its_own_arm() -> None:
    """The decision function is unchanged — the repair works THROUGH its existing
    `ci_lo_boot is not None` arm rather than adding a second refusal beside it."""
    assert gate_promotion_decision(1.0, None, False, 0.55) is False
    assert gate_promotion_decision(1.0, 0.5, False, 0.55) is True


@pytest.mark.parametrize("n_distinct", [1, 2, 3, 8])
def test_the_interval_appears_exactly_at_two_distinct_games(n_distinct: int) -> None:
    """The boundary, stated as a boundary rather than inferred from two rows."""
    records = [_record(f"t{i}") for i in range(n_distinct)]
    result = _agg(records)
    assert (result.elo_ci_lower_boot is None) == (n_distinct < 2), (
        f"n_distinct={n_distinct}: {result.elo_ci_lower_boot!r}"
    )


def test_a_repeated_trajectory_is_not_a_second_distinct_game() -> None:
    """LAW-04's own point, and the reason the count is DISTINCT games: two byte-identical
    games are one observation, and cannot manufacture an interval."""
    same = _record("t1")
    result = _agg([same, dict(same)])
    assert result.eff_n == 1
    assert result.elo_ci_lower_boot is None
    assert result.promoted is False
