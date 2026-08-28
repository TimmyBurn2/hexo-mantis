"""The round-progress escalation is its own route, and its phase says so (R316(c)).

`_poll_loop` kills a round that exceeds `eval.round_timeout_sec` — a PROGRESS budget — and
reported it as `join_timeout` until R316(c). That named the kill sequence rather than the cause:
at the 2026-08-27 re-sit every in-run round ended this way and the operator was told the child
would not exit. `_drain_escalate`'s genuine join timeout keeps the name, and the two must stay
distinguishable.

The frozen `test_eval_broken_reason_routes.py` pins reason→phase for the routes it enumerates and
is OUT OF SCOPE of the grant, so the new member's phase is pinned here instead — an unpinned
phase is a second axis nobody reads, which is the defect that file was written against.
"""
from __future__ import annotations

from mantis.eval.errors import EvalBrokenReason


def test_the_two_escalating_routes_are_distinct_members() -> None:
    assert EvalBrokenReason.ROUND_TIMEOUT is not EvalBrokenReason.JOIN_TIMEOUT
    assert EvalBrokenReason.ROUND_TIMEOUT.value == "round_timeout"
    assert EvalBrokenReason.JOIN_TIMEOUT.value == "join_timeout"


def test_the_wire_spelling_round_trips_through_the_taxonomy() -> None:
    """An unregistered spelling raises, which is what makes a reason no member spells loud."""
    assert EvalBrokenReason("round_timeout") is EvalBrokenReason.ROUND_TIMEOUT


def test_the_round_budget_escalation_reports_the_round_timeout_phase() -> None:
    """Phase is a FUNCTION of the reason; a constant would send a supervisor to the drain budget."""
    import inspect

    from mantis.eval import pipeline

    src = inspect.getsource(pipeline.EvalPipeline._finalize_round)
    assert '"round_timeout" if escalated_reason is EvalBrokenReason.ROUND_TIMEOUT' in src, (
        "`_finalize_round`'s escalated branch must derive the phase from the reason — a literal "
        '"drain" for both escalating routes is the mislabelling this ruling corrected'
    )


def test_the_poll_loop_escalates_with_ROUND_TIMEOUT_and_the_drain_keeps_JOIN_TIMEOUT() -> None:
    """Read off the source: the two producers live in different functions and only their PAIRING
    is the claim — swapping them would leave every other row in this file green."""
    import inspect

    from mantis.eval import pipeline

    escalate = inspect.getsource(pipeline.EvalPipeline._escalate_and_finalize)
    assert "EvalBrokenReason.ROUND_TIMEOUT" in escalate
    assert "escalated_reason=EvalBrokenReason.JOIN_TIMEOUT" not in escalate

    drain = inspect.getsource(pipeline.drain_or_kill)
    assert "EvalBrokenReason.JOIN_TIMEOUT" in drain, (
        "the DRAIN's join timeout is a genuine one and keeps its name; only the round-budget "
        "escalation was mislabelled"
    )
    assert "EvalBrokenReason.ROUND_TIMEOUT" not in drain
