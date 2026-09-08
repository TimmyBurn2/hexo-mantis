"""R345(b)(7) — a dead self-play worker must surface at the next health check.

THE STATE AT HEAD. `runner::spawn::guard_worker` already does the hard half: it catches the
panic, increments `worker_panics` and stores `running = false`, and the counter reaches Python
as `RunnerStats.worker_panics`. Every piece of the instrument exists.

Nothing reads it. `WorkerPool.check_producer_health` — the method whose own docstring calls
it *"the fail-fast hook the trainer calls every step"*, invoked from
`train/coordinator/step.py` once per training step — tests ONLY `self._producer_exc`, which is
the death of the PYTHON drain thread. A grep for `worker_panics` under `src/mantis/monitor/`
and `src/mantis/train/` finds no reader at all.

So the failure presents as a healthy pool draining nothing: the drain loop keeps running, the
buffer stops growing, and the run continues training on an increasingly stale ring until
`selfplay_stall_timeout_sec` (1800.0 in `configs/run6.yaml`) notices thirty minutes later. That
is thirty minutes of a promoting run training on data no worker is producing, and it is exactly
the shape `guard_worker`'s own docstring says it exists to prevent — *"the pool therefore kept
reporting healthy while producing nothing"* — solved on the Rust side and never connected.

WHY `worker_panics` AND NOT `is_running()`. `running` goes false on a CLEAN stop too, so
reading it here would abort every orderly shutdown. The panic counter is unambiguous: it is
zero in a healthy run and non-zero only because a worker died.
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.selfplay.pool import WorkerPool


class _Runner:
    """A runner double exposing exactly the two members the health check may read."""

    def __init__(self, *, panics: int = 0, running: bool = True) -> None:
        self._panics = panics
        self._running = running

    def worker_panics(self) -> int:
        return self._panics

    def is_running(self) -> bool:
        return self._running


class _Pool:
    """The health check on a pool double — the method is what is under test, not the pool."""

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
    """A run that lost four workers and a run that lost one are different facts."""
    with pytest.raises(RuntimeError, match="4"):
        _Pool(_Runner(panics=4, running=False)).check_producer_health()


def test_a_healthy_pool_passes() -> None:
    """Mutation half. A check that always raises stops every run at its first step."""
    _Pool(_Runner(panics=0, running=True)).check_producer_health()


def test_a_clean_stop_is_not_a_panic() -> None:
    """`running` goes false on an ORDERLY shutdown, so it must not be the trigger.

    Reading `is_running()` here would abort every clean stop — the exact inverse of the
    defect, and a far more visible one. `worker_panics` is zero in a healthy run and non-zero
    only because a worker died, which is why it is the operand.
    """
    _Pool(_Runner(panics=0, running=False)).check_producer_health()


def test_the_python_producer_death_still_raises_first() -> None:
    """The pre-existing arm is untouched, and it takes precedence.

    A dead feeder is the more specific fact — it carries the original exception as its cause —
    so a pool with both conditions reports that one.
    """
    boom = ValueError("the feeder died")
    with pytest.raises(RuntimeError) as excinfo:
        _Pool(_Runner(panics=2, running=False), producer_exc=boom).check_producer_health()
    assert excinfo.value.__cause__ is boom, (
        "the feeder's own exception was dropped in favour of the panic count"
    )


def test_the_counter_is_read_whether_it_is_a_method_or_a_field() -> None:
    """`worker_panics` is a METHOD on the engine runner and a FIELD on `RunnerStats`.

    Doubles in this tree use both shapes, and a check that understood only one would pass
    vacuously against every caller using the other — which is how a live gate becomes a
    phantom one without anybody editing it.
    """
    class _FieldShaped:
        worker_panics = 3

    with pytest.raises(RuntimeError, match="3"):
        _Pool(_FieldShaped()).check_producer_health()


def test_a_runner_without_the_counter_does_not_break_the_check() -> None:
    """A pool double or an older wheel exposes no `worker_panics`; that is not a panic.

    Absence must read as "no measurement", never as zero AND never as a failure — the second
    would make the health check itself the thing that stops runs.
    """
    class _Bare:
        pass

    _Pool(_Bare()).check_producer_health()
