"""One distinct game is not a confidence interval: the gate reports none below two distinct games.

A bootstrap over a single sample collapses onto its point estimate, so the aggregate reports no
interval there (`elo_ci_lower_boot is None`) rather than a degenerate one. Promotion itself is the
GSPRT's verdict, whose LLR reads at least two pair scores, so no single game can promote.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.eval.aggregate import aggregate_gate


class _Gate:
    """The gate knobs `aggregate_gate` reads, at the smoke config's own floor of 1."""

    bootstrap_resamples = 64
    min_distinct_per_pair = 1
    seed_base = 7


def _record(traj: str, *, won: bool = True) -> dict[str, Any]:
    """The shape `aggregate_gate` reads — the same one `test_aggregate_regime.py` builds:
    `trajectory_hash` is the dedupe key, and `winner` is `p1`/`p2`."""
    return {"p1": "cand", "p2": "best", "winner": "p1" if won else "p2",
            "regime_key": "rk", "trajectory_hash": traj}


def _agg(records: list[dict[str, Any]]) -> Any:
    return aggregate_gate(records, _Gate(), {"decision": "promote", "checks": 1})


@pytest.mark.parametrize("n_distinct", [1, 2, 3, 8])
def test_the_interval_appears_exactly_at_two_distinct_games(n_distinct: int) -> None:
    """The boundary, stated as a boundary rather than inferred from two rows."""
    records = [_record(f"t{i}") for i in range(n_distinct)]
    result = _agg(records)
    assert (result.elo_ci_lower_boot is None) == (n_distinct < 2), (
        f"n_distinct={n_distinct}: {result.elo_ci_lower_boot!r}"
    )


def test_a_repeated_trajectory_is_not_a_second_distinct_game() -> None:
    """The reason the count is DISTINCT games: two byte-identical
    games are one observation, and cannot manufacture an interval."""
    same = _record("t1")
    result = _agg([same, dict(same)])
    assert result.eff_n == 1
    assert result.elo_ci_lower_boot is None
