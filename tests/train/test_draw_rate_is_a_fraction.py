"""Prove the emitted `draw_rate` is a fraction whose denominator the pool owns.

Measured on a real burn stream: 1.3333, 1.5, 1.125, 1.2222 and 1.0909 — exactly 4/3, 3/2, 9/8,
11/9 and 12/11, from dividing a LIVE `pool.draws` by a `games_played` snapshot frozen while the
feeder thread kept draining games in.

Killer: restore `pool.draws / games_played` in the event builder; the first row then emits 4/3.
"""
from __future__ import annotations

from typing import Any

from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.events import emit_iteration_complete_event


def _rstats() -> RunnerStats:
    """Build a real stats snapshot, so a rename in `pool_hooks` is not papered over."""
    return RunnerStats(
        games_completed=0, positions_generated=0, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1,
    )


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))


class _StraddlingPool:
    """Model a pool that DRAINS between the coordinator's snapshot and the emit: both counts
    advance to `now` while the caller passes the stale snapshot."""

    search_kind = "puct"
    avg_game_length = 128.0
    sims_per_sec = 100.0
    batch_fill_pct = 0.9
    inference_batch_timing = None

    def __init__(self, *, completed_now: int) -> None:
        self._completed = completed_now
        self._draws = completed_now  # every game a draw
        self.draws = completed_now  # raw count, present only so M-DR-1 is runnable
        self.recent_move_histories: list = []

    @property
    def x_winrate(self) -> float:
        return 0.0

    @property
    def o_winrate(self) -> float:
        return 0.0

    @property
    def draw_rate(self) -> float:
        # The pool's own lock-consistent read: one denominator, taken with the numerator.
        return self._draws / self._completed if self._completed else 0.0

    def runner_stats(self) -> Any:
        return _rstats()


class _Buffer:
    size = 1000
    capacity = 100_000


def _emit(*, completed_now: int, games_played_snapshot: int) -> dict:
    sink = _Sink()
    emit_iteration_complete_event(
        7, 0.0, games_played_snapshot, 0, _StraddlingPool(completed_now=completed_now),
        _Buffer(), {}, {}, 100_000, lambda: 10.0, lambda: 5.0, {}, _rstats(), sink,
    )
    events = [e for e in sink.events if e["event"] == "iteration_complete"]
    assert len(events) == 1, f"expected one iteration_complete, got {len(events)}"
    return events[0]


def test_the_emitted_draw_rate_cannot_exceed_one_when_the_pool_drains_mid_step() -> None:
    """Prove the emitted draw rate cannot exceed one when the pool drains mid-step: at a snapshot
    of 3 against a pool of 4, the old form emits the burn stream's own 1.3333."""
    payload = _emit(completed_now=4, games_played_snapshot=3)
    assert payload["draw_rate"] <= 1.0, (
        f"draw_rate emitted {payload['draw_rate']} — a fraction above 1. The share is being "
        "divided by the coordinator's stale game count instead of the pool's own "
        "denominator (F-816-2)"
    )
    assert payload["draw_rate"] == 1.0, (
        "a run where every completed game is a draw has a draw share of exactly 1.0; got "
        f"{payload['draw_rate']}"
    )


def test_the_denominator_is_the_pools_and_not_the_coordinators_snapshot() -> None:
    """Prove the stale snapshot cannot influence the value at all: one that moves with it is still
    reading the coordinator's number, whether or not it stays under 1."""
    values = {
        snap: _emit(completed_now=8, games_played_snapshot=snap)["draw_rate"]
        for snap in (1, 8, 10_000)
    }
    assert len(set(values.values())) == 1, (
        f"the emitted share moved with the coordinator's snapshot: {values} — the pool owns "
        "this denominator, and a payload that mixes the two is reporting a ratio of two "
        "different clocks"
    )


def test_the_three_outcome_shares_share_a_denominator() -> None:
    """Prove the three outcome shares share a denominator and sum to 1 — why the fix is the pool
    property and not a clamp, which would keep the payload plausible but incommensurable."""
    payload = _emit(completed_now=8, games_played_snapshot=3)
    total = payload["win_rate_p0"] + payload["win_rate_p1"] + payload["draw_rate"]
    assert abs(total - 1.0) < 1e-9, (
        f"the three outcome shares sum to {total}, not 1 — they are not shares of the same "
        "quantity"
    )
