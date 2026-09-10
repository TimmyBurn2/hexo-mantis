"""All four drain-cap fields are wired as subprocess-join bounds.

The mid-run budget is `min(timeout * safety_factor, hard_cap)`; the terminal round's bound is
`terminal_eval_hard_cap_sec` directly. Overrun escalates terminate -> kill, each step bounded,
and names the failure. The seams are pinned as pure functions so the arithmetic is testable in
microseconds; the kill mechanics belong to the isolation suites and are not duplicated.
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
    """A child that never exits on its own: a signal flips `is_alive()` only on the NEXT join,
    so a slow-to-die child still needs its own bounded join after terminate/kill."""

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
    """The drain budget is the min of the safety-factor product and the hard cap — both
    branches driven."""
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

    # The shipped defaults, pinned as real numbers.
    shipped = DrainCaps(
        final_eval_drain_timeout_sec=900.0, eval_final_drain_safety_factor=3.0,
        eval_final_drain_hard_cap_sec=14400.0, terminal_eval_hard_cap_sec=14400.0,
    )
    assert drain_budget_sec(shipped) == 2700.0, "900*3=2700 < 14400 hard cap"


def test_drain_overrun_kills_worker_and_yields_eval_broken() -> None:
    """An overrun budget escalates terminate -> kill, each bounded, and names the failure
    rather than hanging silently."""
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

    reason = drain_or_kill(
        proc, budget_sec=budget, worker_kill_grace_sec=0.2,
        clock=_clock_that_overruns_immediately,
    )

    assert reason is not None, "an overrun drain must be reported broken, never silently OK"
    assert reason == "join_timeout", f"drain overrun reason must name join_timeout, got {reason!r}"
    assert proc.terminate_called, "an overrun budget must terminate() the child"
    assert proc.kill_called, "a child that does not die on terminate must be kill()ed"
    assert proc.join_calls, "every escalation step must call a BOUNDED join()"
    assert all(t >= 0 for t in proc.join_calls), "no bare (unbounded) join anywhere"


def test_drain_within_budget_is_not_broken() -> None:
    """A clean exit within budget is not broken — the contrast arm that stops the overrun row
    being vacuously always-broken."""
    caps = DrainCaps(
        final_eval_drain_timeout_sec=10.0, eval_final_drain_safety_factor=2.0,
        eval_final_drain_hard_cap_sec=100.0, terminal_eval_hard_cap_sec=100.0,
    )
    budget = drain_budget_sec(caps)
    clock = FakeClock(0.0)
    proc = FakeHangingProcess()
    proc.exitcode = 0  # already exited cleanly before drain_or_kill is even called

    reason = drain_or_kill(
        proc, budget_sec=budget, worker_kill_grace_sec=0.2, clock=clock,
    )

    assert reason is None, "a clean exit within budget must not be reported broken"
    assert not proc.terminate_called and not proc.kill_called, (
        "a clean exit must never be terminated/killed"
    )


def test_terminal_round_bounded_by_terminal_eval_hard_cap_sec() -> None:
    """The terminal round's bound is `terminal_eval_hard_cap_sec` directly, not the mid-run
    formula — same escalation primitive, different budget input."""
    caps = DrainCaps(
        final_eval_drain_timeout_sec=900.0, eval_final_drain_safety_factor=3.0,
        eval_final_drain_hard_cap_sec=14400.0, terminal_eval_hard_cap_sec=0.05,
    )
    clock = FakeClock(0.0)
    proc = FakeHangingProcess()

    def _clock_that_overruns_immediately() -> float:
        clock.advance(caps.terminal_eval_hard_cap_sec + 0.01)
        return clock.t

    reason = drain_or_kill(
        proc, budget_sec=caps.terminal_eval_hard_cap_sec, worker_kill_grace_sec=0.2,
        clock=_clock_that_overruns_immediately,
    )

    assert reason is not None and reason == "join_timeout"
    assert proc.terminate_called and proc.kill_called


def test_all_four_drain_cap_fields_have_live_consumers() -> None:
    """Every one of the four fields feeds a live read in the pipeline's own source, rather
    than being carried unread."""
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


# `drain_or_kill` returns ONE typed value: a `(bool, str)` pair is a boolean beside a value
# that can contradict it, and `None` as the clean state leaves no second field to disagree.
def test_drain_or_kill_returns_a_typed_reason_or_none() -> None:
    """Clean exit returns `None`; an overrun returns the typed `JOIN_TIMEOUT` member. Killer:
    restore the `(bool, str)` tuple, which nothing else in the suite would notice."""
    from mantis.eval.errors import EvalBrokenReason

    clean = FakeHangingProcess()
    clean.exitcode = 0
    assert drain_or_kill(clean, budget_sec=1.0, worker_kill_grace_sec=0.2,
                         clock=FakeClock(0.0)) is None, (
        "a child that exited inside its budget has NO reason — absence is the clean state, "
        "and a `(False, 'clean_exit')` pair is two authorities for one fact (R79)"
    )
    assert not clean.terminate_called and not clean.kill_called, (
        "premise: the clean arm really was the clean arm"
    )

    hung = FakeHangingProcess()
    overrun = drain_or_kill(hung, budget_sec=0.01, worker_kill_grace_sec=0.2,
                            clock=FakeClock(0.0))
    assert overrun is EvalBrokenReason.JOIN_TIMEOUT, (
        f"an overrun drain escalates to the JOIN_TIMEOUT member, not a bare string; got "
        f"{overrun!r}"
    )
    assert hung.terminate_called and hung.kill_called, (
        "premise: the overrun arm really did escalate"
    )


def test_the_clean_exit_literal_is_gone_from_the_package() -> None:
    """No `"clean_exit"` literal survives under `src/mantis`, and the annotation declares the
    typed reason — a function can declare the union and still return a tuple at one branch."""
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "src" / "mantis"
    offenders = [str(path.relative_to(package.parent)) for path in sorted(package.rglob("*.py"))
                 if "clean_exit" in path.read_text(encoding="utf-8")]
    assert offenders == [], (
        f"the `clean_exit` literal survives in {offenders} — a reason spelling with no enum "
        "member is the free-form string R152 deletes"
    )

    annotation = inspect.signature(drain_or_kill).return_annotation
    assert "tuple" not in str(annotation), (
        f"`drain_or_kill` still declares a tuple return ({annotation!r}) — the bool and the "
        "string are two authorities for one fact"
    )
    assert "EvalBrokenReason" in str(annotation), (
        f"…and it must declare the typed reason so pyright (gate 14, held at ZERO) refuses a "
        f"bare string at every call site; got {annotation!r}"
    )
