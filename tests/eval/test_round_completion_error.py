"""The poller/round-completion thread must be exception-proof.

A timeout, crash or garbage result ALWAYS escalates to a named `eval_broken` event and a routed
broken result — never a hang, never a silent skip. An uncaught exception inside the completion
path used to kill the daemon poller thread, so `_finalize_round`'s shared tail never ran: the
in-flight round was never cleared and never delivered, and the poller's `eval_round` heartbeat —
its SOLE source — stopped beating until the watchdog's staleness deadline fired.

No real OS subprocess is spawned: a fake process whose exit looks clean plus a monkeypatched
`_read_worker_result` reaches the catch-all deterministically, with no torch-import-latency race.
"""
from __future__ import annotations

import time
from typing import Any

import pytest
from _pipeline_harness import (
    _InjectedCompletionError,
    bounded,
    fake_mp,
    pipeline_kwargs,
    tiny_model,
)

from mantis.eval.pipeline import build_eval_pipeline


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _inject_completion_crash(pipeline) -> None:
    def _boom(inflight, *, exit_code, wall_sec):
        raise _InjectedCompletionError("simulated round-completion crash (RED-TEAM F1 shape)")

    pipeline._read_worker_result = _boom  # noqa: SLF001 -- intentional, test-only injection


def test_poller_thread_survives_an_uncaught_exception_in_round_completion(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        _inject_completion_crash(pipeline)

        proc = fake_mp.last_process
        assert proc is not None
        proc.alive = False
        proc.exitcode = 0  # "clean exit" -> the poller's tick routes to _read_worker_result

        def _wait_for_result():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = pipeline.poll_completed()
                if result is not None:
                    return result
                time.sleep(0.01)
            return None

        result = bounded(_wait_for_result, timeout=6.0)

        # A routed result WAS delivered: never a silent hang or dropped round.
        assert result is not None, "poll_completed() must eventually deliver a result, never hang forever"
        assert result["eval_broken_reason"] == "round_completion_error"
        assert result.get("promoted") is False
        assert "_InjectedCompletionError" in (result["eval_broken_detail"] or "")

        # A named eval_broken event WAS emitted.
        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the injected completion crash"
        ev = broken[-1]
        assert ev.get("reason") == "round_completion_error"
        assert "_InjectedCompletionError" in ev.get("detail", "") or "simulated round-completion crash" in ev.get("detail", "")

        # The poller THREAD itself is still alive.
        assert pipeline._poller.is_alive(), "the eval-pipeline-poller thread must survive an uncaught exception"

        # The pipeline is still usable afterwards: in-flight cleared, not stuck forever.
        assert pipeline.poll_completed() is None  # mailbox drained, nothing left pending
    finally:
        pipeline.stop()


# `drain_pending()` is the synchronous teardown join point, a second route to the same crash.
def test_drain_pending_survives_an_uncaught_exception_in_round_completion(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        _inject_completion_crash(pipeline)

        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0

        result = bounded(lambda: pipeline.drain_pending(), timeout=5.0)

        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        broken = sink.named("eval_broken")
        assert broken and broken[-1].get("reason") == "round_completion_error"
        assert "exception_class" in broken[-1]
        assert broken[-1]["exception_class"] == "_InjectedCompletionError"
    finally:
        pipeline.stop()


def test_round_completion_error_never_silent_never_dropped(fake_mp, tmp_path) -> None:
    """A routed result WITH no event, or an event WITH no routed result, are each rejected."""
    sink = _SpySink()
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        _inject_completion_crash(pipeline)
        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0

        result = bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        got_routed_result = result is not None and result.get("promoted") is False
        got_event = bool(sink.named("eval_broken"))
        assert got_routed_result, "a round-completion crash must still route a result with promoted=False"
        assert got_event, "a round-completion crash must still emit eval_broken (never silent)"
    finally:
        pipeline.stop()
