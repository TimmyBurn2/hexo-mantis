"""Every eval-failure route produces its OWN typed reason, and the stream says which.

The defect: every broken round used to route a bare `str` reason that nothing in `src/` read,
so the failures were indistinguishable to anything but a human reading a log line. (The
`ladder_persist_failed` route left with the sealbot rung.)

Per-oracle mutations: M-O2 swap two reasons; M-O3 collapse two members onto one value; M-O4
emit `phase="drain"` for `result_missing`; M-O30a downgrade `_LOG.exception` to `_LOG.error`;
M-O31 derive the event's `reason` from a second local. Each row names the one it is the only
witness to.

REAL: the shipped `EvalPipeline`, its `_finalize_round` / `_read_worker_result` /
`_broken_result` / `_success_result` chain, real emission and a real round-result mapping. FAKE: the worker SUBPROCESS (an injected fake `multiprocessing` context) and, on two routes, a monkeypatched raise — spawning real
subprocesses would trade determinism for nothing, since the subject is the reason assembly.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import pytest
from _pipeline_harness import (
    FakeCtx,
    _InjectedCompletionError,
    eval_config,
    pipeline_kwargs,
    tiny_model,
)

from mantis.eval.errors import EvalBrokenReason
from mantis.eval.pipeline import build_eval_pipeline

#: The routes, each with the member it must produce and the phase that member forces, stated
#: here rather than derived from the enum under test.
_ROUTE_REASON = {
    "join_timeout": "join_timeout",
    "killed": "killed",
    "exit_nonzero": "exit_nonzero",
    "result_missing": "result_missing",
    "result_invalid": "result_invalid",
    "round_completion_error": "round_completion_error",
    "abandoned": "abandoned",
}
_ROUTE_PHASE = {
    "join_timeout": "drain",
    "killed": "worker_exit",
    "exit_nonzero": "worker_exit",
    "result_missing": "worker_exit",
    "result_invalid": "worker_exit",
    "round_completion_error": "round_completion",
    "abandoned": "abandon",
}
_ROUTES = tuple(_ROUTE_REASON)

class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        # `event` is subscripted, not `.get`-ed: a payload without it is a producer defect
        # and must be loud here rather than silently filtered out of every assertion.
        return [e for e in self.events if e["event"] == name]


class _Drive:
    """One driven route: the routed result and the events the round actually emitted."""

    def __init__(self, result: dict[str, Any], sink: _SpySink) -> None:
        self.result = result
        self.sink = sink

    def broken_event(self) -> dict[str, Any]:
        events = self.sink.named("eval_broken")
        assert events, "the route emitted NO eval_broken event — a broken round is never silent"
        return events[-1]


def _quiesce_poller(pipeline: Any) -> None:
    """Stop the persistent poller BEFORE the round is driven — a RACE, not a weakening: the
    poller finalizes any round whose process stops looking alive, so leaving it running would
    make which path assembled the reason nondeterministic."""
    pipeline._stop_event.set()          # noqa: SLF001 -- deliberate, test-only quiescing
    pipeline._poller.join(5.0)          # noqa: SLF001
    assert not pipeline._poller.is_alive(), (  # noqa: SLF001
        "the poller thread did not stop inside 5 s; the drive below would race it"
    )


def _drive(route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Drive:
    """Drive ONE censused failure route through the real `EvalPipeline` and return what it routed
    and emitted. Every branch reproduces a real production condition; none reaches into the
    reason assembly itself."""
    assert route in _ROUTE_REASON, f"unknown route {route!r}"
    ctx = FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    sink = _SpySink()
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        _quiesce_poller(pipeline)
        ack = pipeline.run_evaluation(tiny_model(), 1000, None, full_config={},
                                      best_model_step=None)
        assert ack["kicked"] is True, "premise: the round was kicked"
        proc = ctx.last_process
        assert proc is not None, "premise: the pipeline spawned a (fake) worker"
        result_path = Path(pipeline._inflight["spec"].result_path)  # noqa: SLF001

        if route == "join_timeout":
            proc.alive = True                       # the child never exits on its own
        elif route == "killed":
            proc.alive, proc.exitcode = False, -9   # POSIX-signed: negative == signal death
        elif route == "exit_nonzero":
            proc.alive, proc.exitcode = False, 3
        elif route == "result_missing":
            proc.alive, proc.exitcode = False, 0    # clean exit, no sidecar written
        elif route == "result_invalid":
            proc.alive, proc.exitcode = False, 0
            result_path.write_text(json.dumps({"step": 1000}), encoding="utf-8")   # contract keys missing
        elif route == "abandoned":
            proc.alive = True                       # a live round a resumable stop abandons
        else:  # round_completion_error
            proc.alive, proc.exitcode = False, 0

            def _completion_boom(inflight: Any, *, exit_code: Any, wall_sec: Any) -> None:
                raise _InjectedCompletionError("simulated round-completion crash (F1 shape)")

            pipeline._read_worker_result = _completion_boom  # noqa: SLF001

        result = pipeline.abandon_pending() if route == "abandoned" else pipeline.drain_pending()
        assert isinstance(result, dict), (
            f"premise: route {route!r} must route ONE completed round mapping; got {result!r}"
        )
        return _Drive(result, sink)
    finally:
        pipeline.stop()


@pytest.mark.parametrize("route", _ROUTES)
def test_each_route_yields_its_own_typed_reason(route, tmp_path, monkeypatch) -> None:
    """O-02. The routed result carries a typed `eval_broken_reason`, the member THIS route owns.
    Read off the ROUTED RESULT and not the emitted event, since `promote.py` consumes the
    mapping and pairing this with the event would make M-O31 invisible to the whole file."""
    drive = _drive(route, tmp_path, monkeypatch)
    reason = drive.result["eval_broken_reason"]
    assert isinstance(reason, EvalBrokenReason), (
        f"route {route!r} routed a {type(reason).__name__} ({reason!r}) — a bare string is "
        "exactly the second authority R152 deletes; the builder must take the enum"
    )
    assert reason.value == _ROUTE_REASON[route], (
        f"route {route!r} must produce {_ROUTE_REASON[route]!r}; got {reason.value!r}"
    )
    assert drive.result["promoted"] is False, (
        "a broken round never promotes — `promoted` is DERIVED from the reason, and a "
        "broken round that promotes is the defect the derivation exists to make impossible"
    )


def test_the_emitted_reasons_are_pairwise_distinct(tmp_path, monkeypatch) -> None:
    """O-03. The "distinguishable from each other" leg, taken in the EVENT STREAM. The rc
    taxonomy is many-to-one by decision — every member maps to 48 — so if the emitted `reason`
    values ever collide, nothing separates a killed worker from a garbage result. Reads the
    EMITTED values, so a collision introduced on the emit side alone is still caught."""
    emitted = {}
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        emitted[route] = drive.broken_event()["reason"]

    values = list(emitted.values())
    collisions = [(a, b) for i, a in enumerate(_ROUTES) for b in _ROUTES[i + 1:]
                  if emitted[a] == emitted[b]]
    assert collisions == [], (
        f"routes sharing one emitted reason: {collisions} (full map: {emitted})"
    )
    assert len(set(values)) == len(_ROUTES), (
        f"{len(_ROUTES)} routes must emit {len(_ROUTES)} distinct reasons; got {sorted(set(values))}"
    )


def test_the_reason_to_phase_map_is_fixed(tmp_path, monkeypatch) -> None:
    """O-04. `phase` stays on the payload precisely because it is a FUNCTION of the reason, and
    nothing else in the tree reads it — so without this pin a mislabelled phase is invisible,
    and a supervisor triaging `result_missing` under `phase=drain` would go looking at the
    drain budget for a missing sidecar file."""
    observed = {}
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        event = drive.broken_event()
        observed[event["reason"]] = event["phase"]

    expected = {_ROUTE_REASON[route]: _ROUTE_PHASE[route] for route in _ROUTES}
    assert observed == expected, (
        "the reason→phase map moved; a phase that can contradict its reason is a second "
        f"axis nobody reads.\n  expected: {expected}\n  observed: {observed}"
    )


def test_the_emitted_event_reason_equals_the_routed_result_reason(tmp_path, monkeypatch) -> None:
    """O-31. Two consumers read two different objects: a supervisor reads the `eval_broken`
    EVENT, `promote.py` reads the round-result MAPPING. A second local feeding the event would
    leave O-02 and O-03 green while the stream told an operator the wrong story."""
    for route in _ROUTES:
        drive = _drive(route, tmp_path / route, monkeypatch)
        event_reason = drive.broken_event()["reason"]
        result_reason = drive.result["eval_broken_reason"]
        assert event_reason == result_reason, (
            f"route {route!r}: the emitted event says {event_reason!r} and the routed "
            f"result says {result_reason!r} — one emitter, one value (R152)"
        )


def test_the_round_completion_route_logs_a_traceback_and_the_detail(
    tmp_path, monkeypatch, caplog
) -> None:
    """O-30, arm 1 (round_completion_error). The route's contract is "never a swallowed
    exception, NEVER A BARE LOG LINE": `repr(exc)` says WHAT was raised, only the traceback says
    WHERE. The emitter it collapses into logs with `_LOG.error` and no traceback, so without
    this oracle the collapse silently deletes the stack.

    MUTATION (M-O30a): downgrade the raising site's `_LOG.exception`. Every payload assertion
    stays green — the mechanism is the LOG RECORD alone.
    """
    with caplog.at_level(logging.ERROR, logger="mantis.eval.pipeline"):
        drive = _drive("round_completion_error", tmp_path, monkeypatch)

    records = [r for r in caplog.records if r.name == "mantis.eval.pipeline"]
    with_traceback = [r for r in records if r.exc_info is not None]
    assert with_traceback, (
        "the round-completion catch-all must log WITH the live traceback (`_LOG.exception`); "
        f"captured {len(records)} record(s), none carrying exc_info: "
        f"{[r.getMessage() for r in records]}"
    )
    assert any("_InjectedCompletionError" in repr(r.exc_info) for r in with_traceback), (
        "the captured traceback must be THIS round's exception, not an unrelated live "
        f"context: {[repr(r.exc_info) for r in with_traceback]}"
    )
    assert any("_InjectedCompletionError" in r.getMessage() for r in records), (
        "…and the `repr(exc)` detail must still travel in the message text, so an operator "
        f"grepping the log without a traceback reader still sees the class: "
        f"{[r.getMessage() for r in records]}"
    )
    assert any("eval_broken" in r.getMessage() for r in records), (
        "…AND the emitter's own `eval_broken …` ERROR line is still there: the two records "
        f"are the raising site and the one emitter, not one replacing the other: "
        f"{[r.getMessage() for r in records]}"
    )
    assert drive.result["eval_broken_reason"] is EvalBrokenReason.ROUND_COMPLETION_ERROR
