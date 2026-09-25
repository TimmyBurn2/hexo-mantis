"""`_steps_budget` carries its fractional remainder, so a fractional `training_steps_per_game` holds long-run."""
from __future__ import annotations

import pytest

from mantis.train.mixing import _steps_budget


def _bursts(games: list[int], ratio: float, burst: int) -> list[int]:
    carry = 0.0
    out = []
    for n in games:
        steps, carry = _steps_budget(n, ratio, burst, carry)
        assert 0.0 <= carry < 1.0
        out.append(steps)
    return out


def test_one_game_per_burst_at_two_and_a_half_realises_two_and_a_half_not_two() -> None:
    # HEAD before this fix read min(max(1, round(2.5)), burst) = 2 on every such burst (banker's rounding)
    assert _bursts([1] * 4, 2.5, 8) == [2, 3, 2, 3]
    assert _bursts([1] * 10, 2.4, 8) == [2, 2, 3, 2, 3, 2, 2, 3, 2, 3]
    assert sum(_bursts([1] * 100, 2.4, 8)) == 240


def test_the_carry_holds_the_ratio_under_run8s_arrival_mix() -> None:
    games = ([1] * 49 + [2]) * 20  # 2 % of bursts see two games, as run8's stream reads
    for ratio in (1.0, 2.0, 2.4, 2.5, 3.0):
        steps = sum(_bursts(games, ratio, 8))
        assert steps == pytest.approx(ratio * sum(games), abs=1.0), ratio


def test_an_integer_ratio_is_unchanged_from_the_round_rule() -> None:
    assert _bursts([1, 2, 3, 1], 1.0, 2) == [1, 2, 2, 1]
    assert _bursts([3], 2.0, 100) == [6]


def test_the_ceiling_drops_its_excess_and_the_floor_borrows_nothing() -> None:
    steps, carry = _steps_budget(3, 2.0, 4, 0.0)
    assert (steps, carry) == (4, 0.0), "a ceiling, not a debt: the two dropped steps are not carried"
    steps, carry = _steps_budget(1, 0.3, 8, 0.0)
    assert steps == 1 and carry == pytest.approx(0.3), "below one step the burst is still one, and the fraction stays"


def test_a_planted_round_instead_of_floor_reds_the_half_step() -> None:
    # the defect this file exists for: round(2.5) is 2 and the half step is lost forever
    assert min(max(1, round(1 * 2.5)), 8) == 2
    assert _bursts([1, 1], 2.5, 8) == [2, 3]
