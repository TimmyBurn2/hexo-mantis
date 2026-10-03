"""`train.training_steps_fill_ramp`: steps per game scale with the ring's fill, so every row is drawn as a steady row is."""
from __future__ import annotations

import numpy as np
import pytest

from mantis.train import mixing

_CAPACITY = 500_000
_MIN_BUF = 4096
_BATCH = 256
_BURST = 8
#: Rows per game: the production rings' ≈ 91 plies, an integer that divides nothing in sight.
_ROWS_PER_GAME = 91


def _draws_per_row(tspg: float, *, games_per_burst: list[int], ramp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Expected draws per row over its FIFO stay, by insertion fill: a step gives every resident row batch/size draws."""
    n_games = 3 * _CAPACITY // _ROWS_PER_GAME
    density = np.zeros(n_games + 1)
    size, carry, game, i = 0, 0.0, 0, 0
    while game < n_games:
        n = min(games_per_burst[i % len(games_per_burst)], n_games - game)
        i += 1
        steps = 0
        size_after = min(_CAPACITY, size + n * _ROWS_PER_GAME)
        if size_after >= _MIN_BUF:
            fill = size_after / _CAPACITY if ramp else None
            steps, carry = mixing._steps_budget(n, tspg, _BURST, carry, fill=fill)
        for g in range(n):
            density[game + g + 1] = density[game + g]
        density[game + n] += steps * _BATCH / size_after
        size, game = size_after, game + n
    games = np.arange(n_games)
    inserted_fill = np.minimum(1.0, (games + 1) * _ROWS_PER_GAME / _CAPACITY)
    stay = int(np.ceil(_CAPACITY / _ROWS_PER_GAME))
    evicted = games + stay
    alive = evicted <= n_games
    return inserted_fill[alive], density[evicted[alive]] - density[games[alive]]


def _assert_every_fill_draws_the_steady_rate(tspg: float, games_per_burst: list[int]) -> None:
    """Rows inserted at every fill level, the first warm-up rows included, are drawn tspg·batch/rows_per_game ± 2 %."""
    fill, draws = _draws_per_row(tspg, games_per_burst=games_per_burst, ramp=True)
    target = tspg * _BATCH / _ROWS_PER_GAME
    for lo, hi in ((0.0, 0.01), (0.01, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.99), (0.999, 1.0)):
        band = draws[(fill > lo) & (fill <= hi)]
        assert band.size, (lo, hi)
        worst = float(np.max(np.abs(band / target - 1.0)))
        assert worst <= 0.02, f"rows inserted at fill ({lo}, {hi}] read {worst:.3%} off {target:.3f} draws at {tspg}"


@pytest.mark.parametrize("tspg", [2.4, 1.2])
@pytest.mark.parametrize("games_per_burst", [[1], [1] * 49 + [2]])
def test_rows_inserted_at_any_fill_are_drawn_as_often_as_a_steady_row(tspg: float, games_per_burst: list[int]) -> None:
    _assert_every_fill_draws_the_steady_rate(tspg, games_per_burst)


def test_without_the_ramp_the_early_rows_are_over_drawn() -> None:
    """The control: the flat rate draws a row inserted at 1 % fill several times the steady rate."""
    fill, draws = _draws_per_row(2.4, games_per_burst=[1], ramp=False)
    target = 2.4 * _BATCH / _ROWS_PER_GAME
    assert float(draws[fill <= 0.01].min()) > 3 * target


def test_a_planted_one_step_floor_under_the_ramp_reds(monkeypatch: pytest.MonkeyPatch) -> None:
    """PLANTED BREAK: a burst that always takes one step (the flat path's floor) over-draws the low-fill rows."""
    real = mixing._steps_budget

    def floored(n: int, tspg: float, burst: int, carry: float, *, fill: float | None) -> tuple[int, float]:
        steps, carry = real(n, tspg, burst, carry, fill=fill)
        return max(1, steps), carry

    monkeypatch.setattr(mixing, "_steps_budget", floored)
    with pytest.raises(AssertionError, match="off"):
        _assert_every_fill_draws_the_steady_rate(1.2, [1])


def test_the_ramp_scales_the_ratio_by_the_fill_and_a_burst_may_take_no_step() -> None:
    _steps_budget = mixing._steps_budget
    assert _steps_budget(4, 2.0, 100, 0.0, fill=0.25) == (2, 0.0)
    assert _steps_budget(1, 2.4, 8, 0.0, fill=0.01) == (0, pytest.approx(0.024))
    assert _steps_budget(10, 2.0, 100, 0.0, fill=3.0) == (20, 0.0), "a fill past capacity is capped at 1"
    assert _steps_budget(1, 0.3, 8, 0.0, fill=None)[0] == 1, "without the ramp a burst still takes one step"
