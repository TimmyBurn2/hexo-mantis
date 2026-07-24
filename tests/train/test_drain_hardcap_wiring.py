"""⊕ WP11-A — R-DRAIN-HARDCAP disposition: WIRE all four fields (PREREG P-1).

RED-at-import until IMPL writes `mantis.eval.pipeline`. ORACLE-FIRST (⊕): the top-level
`import mantis.eval.pipeline` raises ModuleNotFoundError before any port code exists.

All FOUR drain-cap fields on `StepCoordinatorConfig` (coordinator/config.py:176-180, ALREADY
SHIPPED at HEAD — WP13-A) are consumed as subprocess-join bounds (isolation law 2): the
mid-run/teardown `drain_pending` budget = `min(final_eval_drain_timeout_sec *
eval_final_drain_safety_factor, eval_final_drain_hard_cap_sec)`; the terminal round join
bound = `terminal_eval_hard_cap_sec` directly. Overrun on either bound => terminate => kill,
each step bounded by `worker_kill_grace_sec`, and the round yields a named `eval_broken`.

This suite pins TWO small pure seams the pipeline's join-bound arithmetic must expose (the
design gives the formula/escalation in prose, not a named symbol — pinning them here as pure
functions makes the arithmetic unit-testable in microseconds, with no real subprocess/wall
sleep needed; full subprocess kill mechanics — SIGKILL, garbage JSON, hung child — are
`tests/eval/test_eval_broken.py`'s and `tests/eval/test_pipeline_isolation.py`'s territory,
not duplicated here):

  * `DrainCaps` — the frozen 4-tuple lifted from `StepCoordinatorConfig` (design §c.3).
  * `drain_budget_sec(caps: DrainCaps) -> float` — the P-1 formula.
  * `drain_or_kill(proc, *, budget_sec: float, worker_kill_grace_sec: float, clock)
    -> tuple[bool, str]` — join(budget) -> overrun? terminate -> join(grace) -> kill ->
    join(grace); returns (broken, reason). Used for BOTH the mid-run/teardown drain budget
    and the terminal round's `terminal_eval_hard_cap_sec` bound (same primitive, different
    budget input) — this IS `test_all_four_drain_cap_fields_have_live_consumers`'s pin: all
    four fields feed a live call to one of these two functions.
"""
from __future__ import annotations

import inspect

from mantis.eval.pipeline import DrainCaps, drain_budget_sec, drain_or_kill


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


class FakeHangingProcess:
    """Models a spawn-ctx child that never exits on its own: `join(timeout)` always times
    out (never sets `.exitcode`); `terminate()`/`kill()` record their call but do NOT flip
    `is_alive()` until `join` is called again post-signal (mirrors a slow-to-die child that
    still needs its OWN bounded join after terminate/kill — isolation law 2: 'every join is
    timeout-bounded')."""

    def __init__(self, *, dies_after_terminate: bool = True, dies_after_kill: bool = True) -> None:
        self.exitcode: int | None = None
        self.terminate_called = False
        self.kill_called = False
        self.join_calls: list[float] = []
        self._dies_after_terminate = dies_after_terminate
        self._dies_after_kill = dies_after_kill

    def is_alive(self) -> bool:
        return self.exitcode is None

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(float(timeout) if timeout is not None else -1.0)
        if self.terminate_called and self._dies_after_terminate and self.exitcode is None:
            self.exitcode = -15
        if self.kill_called and self._dies_after_kill:
            self.exitcode = -9

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True


def test_drain_pending_budget_formula() -> None:
    """budget == min(final_eval_drain_timeout_sec * eval_final_drain_safety_factor,
    eval_final_drain_hard_cap_sec) — both branches of the min()."""
    safety_bound = DrainCaps(
        final_eval_drain_timeout_sec=10.0, eval_final_drain_safety_factor=2.0,
        eval_final_drain_hard_cap_sec=100.0, terminal_eval_hard_cap_sec=50.0,
    )
    assert drain_budget_sec(safety_bound) == 20.0, "safety-factor branch: 10 * 2 = 20 < 100"

    hard_cap_bound = DrainCaps(
        final_eval_drain_timeout_sec=1000.0, eval_final_drain_safety_factor=100.0,
        eval_final_drain_hard_cap_sec=5.0, terminal_eval_hard_cap_sec=50.0,
    )
    assert drain_budget_sec(hard_cap_bound) == 5.0, "hard-cap branch: 1000*100=1e5 clamped to 5"

    # WP13-A shipped defaults (coordinator/config.py:176-180) — sanity-pin the real numbers.
    shipped = DrainCaps(
        final_eval_drain_timeout_sec=900.0, eval_final_drain_safety_factor=3.0,
        eval_final_drain_hard_cap_sec=14400.0, terminal_eval_hard_cap_sec=14400.0,
    )
    assert drain_budget_sec(shipped) == 2700.0, "900*3=2700 < 14400 hard cap"


def test_drain_overrun_kills_worker_and_yields_eval_broken() -> None:
    """Isolation law 2: overrun the drain budget -> terminate -> kill, each bounded by
    `worker_kill_grace_sec`, and the outcome is a named eval_broken (broken=True), never a
    silent hang."""
    caps = DrainCaps(
        final_eval_drain_timeout_sec=0.05, eval_final_drain_safety_factor=2.0,
        eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0,
    )
    budget = drain_budget_sec(caps)  # 0.1
    clock = FakeClock(0.0)
    proc = FakeHangingProcess()

    def _clock_that_overruns_immediately() -> float:
        clock.advance(budget + 0.01)
        return clock.t

    broken, reason = drain_or_kill(
        proc, budget_sec=budget, worker_kill_grace_sec=0.2,
        clock=_clock_that_overruns_immediately,
    )

    assert broken is True, "an overrun drain must be reported broken, never silently OK"
    assert reason == "join_timeout", f"drain overrun reason must name join_timeout, got {reason!r}"
    assert proc.terminate_called, "an overrun budget must terminate() the child"
    assert proc.kill_called, "a child that does not die on terminate must be kill()ed"
    assert proc.join_calls, "every escalation step must call a BOUNDED join()"
    assert all(t >= 0 for t in proc.join_calls), "no bare (unbounded) join anywhere"


def test_drain_within_budget_is_not_broken() -> None:
    """A process that exits cleanly WITHIN the budget must not be reported broken (contrast
    arm — proves the overrun test isn't vacuously always-broken)."""
    caps = DrainCaps(
        final_eval_drain_timeout_sec=10.0, eval_final_drain_safety_factor=2.0,
        eval_final_drain_hard_cap_sec=100.0, terminal_eval_hard_cap_sec=100.0,
    )
    budget = drain_budget_sec(caps)
    clock = FakeClock(0.0)
    proc = FakeHangingProcess()
    proc.exitcode = 0  # already exited cleanly before drain_or_kill is even called

    broken, reason = drain_or_kill(
        proc, budget_sec=budget, worker_kill_grace_sec=0.2, clock=clock,
    )

    assert broken is False, "a clean exit within budget must not be reported broken"
    assert not proc.terminate_called and not proc.kill_called, (
        "a clean exit must never be terminated/killed"
    )


def test_terminal_round_bounded_by_terminal_eval_hard_cap_sec() -> None:
    """The TERMINAL round's join bound is `terminal_eval_hard_cap_sec` directly (not the
    mid-run drain formula) — same escalation primitive, a different budget input."""
    caps = DrainCaps(
        final_eval_drain_timeout_sec=900.0, eval_final_drain_safety_factor=3.0,
        eval_final_drain_hard_cap_sec=14400.0, terminal_eval_hard_cap_sec=0.05,
    )
    clock = FakeClock(0.0)
    proc = FakeHangingProcess()

    def _clock_that_overruns_immediately() -> float:
        clock.advance(caps.terminal_eval_hard_cap_sec + 0.01)
        return clock.t

    broken, reason = drain_or_kill(
        proc, budget_sec=caps.terminal_eval_hard_cap_sec, worker_kill_grace_sec=0.2,
        clock=_clock_that_overruns_immediately,
    )

    assert broken is True and reason == "join_timeout"
    assert proc.terminate_called and proc.kill_called


def test_all_four_drain_cap_fields_have_live_consumers() -> None:
    """LAW-08 closure of R-DRAIN-HARDCAP-CONSUMERS: every one of the four fields feeds a
    live read inside `mantis.eval.pipeline`'s own source (not merely carried, unread)."""
    field_names = {f for f in DrainCaps.__dataclass_fields__} if hasattr(
        DrainCaps, "__dataclass_fields__"
    ) else set(DrainCaps._fields)  # tolerate either a frozen dataclass or a NamedTuple
    expected = {
        "final_eval_drain_timeout_sec", "eval_final_drain_safety_factor",
        "eval_final_drain_hard_cap_sec", "terminal_eval_hard_cap_sec",
    }
    assert field_names == expected, f"DrainCaps field set drifted: {field_names}"

    import mantis.eval.pipeline as _pipeline_mod

    src = inspect.getsource(_pipeline_mod)
    missing = [name for name in expected if name not in src]
    assert missing == [], (
        f"DrainCaps field(s) with no live read in mantis/eval/pipeline.py source: {missing}"
    )
