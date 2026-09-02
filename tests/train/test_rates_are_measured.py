"""AUDIT-1 F-28 rows C06 and C07 — a rate nobody measured is `None`, not zero.

FOUR FABRICATIONS, ONE CLASS. `iteration_complete` published `games_per_hour`,
`positions_per_hour`, `avg_game_length` and `sims_per_sec` as hard `0.0`s before anything had
measured them: a rate over zero elapsed clock, a mean over zero completed games, a product of
those two, and a pool counter that starts at zero and stays there until the first drain
observes a positive `positions_generated` delta. Every one of them reads as "the run is doing
nothing" in the ONE channel — which is what a stalled run also looks like.

`steps_per_hour` was already correct (`None` = NOT MEASURED, the same doctrine as
`quiescence_fires_per_step`) and is the pattern the other four now follow.

C06 is the same class one field over: `StepOutcome.games_per_hour` was built as a hard `0.0`
by `_build_outcome`, had NO reader in `src/`, `tests/` or `tools/`, and sat beside a REAL
games-per-hour that `iteration_complete` publishes from the coordinator's own clock. Deleted
rather than wired — LAW-08 wants a live consumer, and an always-zero twin of a measured field
is worse than no field.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from mantis.train.coordinator.config import StepOutcome


# ── C06: the field is gone, and the real one is not ───────────────────────────────────

def test_StepOutcome_carries_no_unmeasured_games_per_hour() -> None:
    names = {f.name for f in dataclasses.fields(StepOutcome)}
    assert "games_per_hour" not in names, (
        "an always-0.0 field with no reader is back beside a measured one — the "
        "duplicate-authority class, in the ONE channel"
    )


def test_the_measured_games_per_hour_still_exists_on_the_coordinator() -> None:
    """The control: deleting the fabricated twin must not delete the real producer."""
    from mantis.train.coordinator.step import StepCoordinator

    assert callable(StepCoordinator._games_per_hour)


# ── C07: the pool starts UNMEASURED ───────────────────────────────────────────────────

class _FreshPool:
    """The two pool readings at construction, read off the REAL class rather than restated.

    Building a whole `SelfPlayPool` needs the engine and a runner; the two properties under
    test are pure reads of two attributes the constructor sets, so the class's own property
    objects are invoked against a stand-in carrying exactly those attributes. If either
    property grows a computation this stops being equivalent — and the row below that drives
    the production `iteration_complete` builder is what would catch it.
    """

    def __init__(self) -> None:
        from mantis.selfplay.pool import WorkerPool

        self._sims_per_sec: float | None = None
        self._avg_game_length: float | None = None
        self._cls = WorkerPool

    @property
    def sims_per_sec(self) -> Any:
        return self._cls.sims_per_sec.fget(self)

    @property
    def avg_game_length(self) -> Any:
        return self._cls.avg_game_length.fget(self)


def test_a_pool_that_has_drained_nothing_reports_no_rate_and_no_mean() -> None:
    """THE PIN. Both read `0.0` before the repair — indistinguishable from a stalled run."""
    pool = _FreshPool()
    assert pool.sims_per_sec is None
    assert pool.avg_game_length is None


def test_a_measured_rate_is_carried_through_unchanged() -> None:
    """The control: absence must not eat a real reading, including a genuine zero."""
    pool = _FreshPool()
    pool._sims_per_sec = 0.0        # a MEASURED zero: a drain interval that billed no sims
    pool._avg_game_length = 12.5
    assert pool.sims_per_sec == 0.0 and pool.sims_per_sec is not None
    assert pool.avg_game_length == 12.5


# ── C07: the coordinator's own clock ──────────────────────────────────────────────────

def test_a_rate_over_zero_elapsed_is_absent_not_zero() -> None:
    from mantis.train.coordinator.step import StepCoordinator

    class _Stub:
        _games_played = 7
        _train_step = 3
        _run_started = 100.0
        _clock = type("C", (), {"now": staticmethod(lambda: 100.0)})()

    stub = _Stub()
    assert StepCoordinator._games_per_hour(stub) is None
    assert StepCoordinator._steps_per_hour(stub) is None
    # and once the clock advances, both are real numbers again
    stub._clock = type("C", (), {"now": staticmethod(lambda: 100.0 + 3600.0)})()
    assert StepCoordinator._games_per_hour(stub) == pytest.approx(7.0)
    assert StepCoordinator._steps_per_hour(stub) == pytest.approx(3.0)


# ── the payload the rules and the perf floors actually read ───────────────────────────

def _iteration_complete(pool: Any, *, gph: Any, sph: Any) -> dict[str, Any]:
    """The PRODUCTION builder, driven with the collaborators it takes."""
    from mantis.train.events import emit_iteration_complete_event

    events: list[dict[str, Any]] = []

    class _Sink:
        def emit(self, event: Any) -> None:
            events.append(dict(event))

    class _Buffer:
        size = 0
        capacity = 1024

    emit_iteration_complete_event(
        train_step=0, w_pre=0.0, games_played=0, last_iter_games=0, pool=pool,
        buffer=_Buffer(), config={}, mcts_config={}, capacity=1024,
        games_per_hour_fn=lambda: gph, steps_per_hour_fn=(lambda: sph) if sph is not None else None,
        target_integrity={}, rstats=_Rstats(), sink=_Sink(),
    )
    assert len(events) == 1, events
    return events[0]


class _Rstats:
    """The `RunnerStats` fields the builder reads, named off its own call sites."""

    mcts_mean_depth = None
    mcts_mean_root_concentration = None
    cluster_value_std_mean = None
    cluster_policy_disagreement_mean = None
    cluster_variance_sample_count = 0
    k_cluster_histogram = None


class _StubPool:
    """A pool that has drained NOTHING, with the property values a fresh one carries."""

    sims_per_sec = None
    avg_game_length = None
    gumbel_mcts = False
    x_winrate = 0.0
    o_winrate = 0.0
    draw_rate = 0.0
    batch_fill_pct = 0.0
    inference_batch_timing = None


def test_the_iteration_complete_payload_reports_four_absences_not_four_zeros() -> None:
    """THE PIN, on the payload a perf floor and a sitting record both read."""
    payload = _iteration_complete(_StubPool(), gph=None, sph=None)
    for key in ("games_per_hour", "positions_per_hour", "avg_game_length", "sims_per_sec",
                "steps_per_hour"):
        assert payload[key] is None, f"{key} = {payload[key]!r}: a fabricated rate"


def test_the_payload_carries_real_rates_when_they_were_measured() -> None:
    """The control. A run that IS producing must publish its numbers unchanged."""
    class _Live(_StubPool):
        sims_per_sec = 480.0
        avg_game_length = 20.0

    payload = _iteration_complete(_Live(), gph=120.0, sph=45.0)
    assert payload["games_per_hour"] == pytest.approx(120.0)
    assert payload["avg_game_length"] == pytest.approx(20.0)
    assert payload["positions_per_hour"] == pytest.approx(2400.0)
    assert payload["sims_per_sec"] == pytest.approx(480.0)
    assert payload["steps_per_hour"] == pytest.approx(45.0)


def test_a_measured_rate_of_zero_survives_as_zero() -> None:
    """The other half: `None` must mean absence and only absence. `sims_per_sec` reached the
    payload through `pool.sims_per_sec or 0.0`, which is where a genuine zero and a
    not-yet-measured reading became the same value."""
    class _Idle(_StubPool):
        sims_per_sec = 0.0
        avg_game_length = 20.0

    payload = _iteration_complete(_Idle(), gph=0.0, sph=0.0)
    assert payload["sims_per_sec"] == 0.0 and payload["sims_per_sec"] is not None
    assert payload["games_per_hour"] == 0.0 and payload["games_per_hour"] is not None
