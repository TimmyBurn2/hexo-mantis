"""A dead self-play worker must surface at the next health check.

Without it the failure presents as a healthy pool draining nothing, and the run trains on an
increasingly stale ring until the self-play stall timeout notices half an hour later.

The operand is `worker_panics`, not `is_running()`: `running` goes false on a clean stop too,
so reading it would abort every orderly shutdown.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.selfplay.pool import WorkerPool


class _Runner:
    """Runner double exposing exactly the two members the health check may read."""

    def __init__(self, *, panics: int = 0, running: bool = True) -> None:
        self._panics = panics
        self._running = running

    def worker_panics(self) -> int:
        return self._panics

    def is_running(self) -> bool:
        return self._running


class _Pool:
    """Pool double carrying the method under test."""

    check_producer_health = WorkerPool.check_producer_health

    def __init__(self, runner: Any, producer_exc: BaseException | None = None) -> None:
        self._runner = runner
        self._producer_exc = producer_exc


def test_a_worker_panic_raises_at_the_next_health_check() -> None:
    pool = _Pool(_Runner(panics=1, running=False))
    with pytest.raises(RuntimeError) as excinfo:
        pool.check_producer_health()
    message = str(excinfo.value)
    assert "worker" in message.lower(), f"the abort does not name the cause: {message}"
    assert "1" in message, "the abort does not report HOW MANY workers died"


def test_more_than_one_panic_is_reported_as_the_count_it_is() -> None:
    """The abort reports the panic COUNT: four dead workers is a different fact from one."""
    with pytest.raises(RuntimeError, match="4"):
        _Pool(_Runner(panics=4, running=False)).check_producer_health()


def test_a_healthy_pool_passes() -> None:
    """Mutation half: a check that always raises stops every run at its first step."""
    _Pool(_Runner(panics=0, running=True)).check_producer_health()


def test_a_clean_stop_is_not_a_panic() -> None:
    """`running` goes false on an orderly shutdown, so it must not be the trigger."""
    _Pool(_Runner(panics=0, running=False)).check_producer_health()


def test_the_python_producer_death_still_raises_first() -> None:
    """A dead feeder takes precedence: it is the more specific fact and carries the original
    exception as its cause."""
    boom = ValueError("the feeder died")
    with pytest.raises(RuntimeError) as excinfo:
        _Pool(_Runner(panics=2, running=False), producer_exc=boom).check_producer_health()
    assert excinfo.value.__cause__ is boom, (
        "the feeder's own exception was dropped in favour of the panic count"
    )


def test_the_counter_is_read_whether_it_is_a_method_or_a_field() -> None:
    """`worker_panics` is a METHOD on the engine runner and a FIELD on `RunnerStats`, and the
    check must read both shapes or it passes vacuously against half its callers."""
    class _FieldShaped:
        worker_panics = 3

    with pytest.raises(RuntimeError, match="3"):
        _Pool(_FieldShaped()).check_producer_health()


def test_a_runner_without_the_counter_does_not_break_the_check() -> None:
    """A runner exposing no `worker_panics` reads as "no measurement", never as a failure."""
    class _Bare:
        pass

    _Pool(_Bare()).check_producer_health()
