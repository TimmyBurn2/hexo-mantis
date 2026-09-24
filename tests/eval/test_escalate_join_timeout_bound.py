"""Layer 2 of the non-finite `worker_kill_grace_sec` fix: `_bounded_join_timeout`.

A pre-fix `worker_kill_grace_sec=float("inf")` reached `_escalate_and_finalize` from the poller's
tick — outside `_finalize_round`'s catch-all — and called `Process.join(inf)`, which raises an
UNCAUGHT `OverflowError` inside `selectors.select()` and kills the poller thread silently. Layer
1 makes such a value unreachable through a config load; this suite proves that one arriving by a
bug that bypasses validation (simulated with `model_copy(update=...)`, which does not re-run
field validators) still ends in a delivered `eval_broken` result with the poller alive.
"""
from __future__ import annotations

import math
import time

import pytest
from _drivable import FakeClock
from _pipeline_harness import (
    bounded,
    eval_config,
    fake_mp,
    pipeline_kwargs,
    tiny_model,
)

from mantis.eval.pipeline import (
    _bounded_join_timeout,
    _JOIN_TIMEOUT_CEILING_SEC,
    build_eval_pipeline,
)


def _cfg_with_bypassed_worker_kill_grace_sec(value: float):
    """The one supported way to build a schema-shaped but schema-INVALID `EvalConfig` for
    injection testing: `model_copy(update=...)` does not re-run field validators. It simulates a
    future code path that mutates an `EvalConfig` without going through config-load validation,
    which is the residual risk layer 2 covers."""
    return eval_config(
        round_timeout_sec=0.05, worker_kill_grace_sec=0.05,
    ).model_copy(update={"worker_kill_grace_sec": value})


@pytest.mark.parametrize(
    "raw,expected",
    [
        (float("inf"), _JOIN_TIMEOUT_CEILING_SEC),
        (float("-inf"), _JOIN_TIMEOUT_CEILING_SEC),
        (float("nan"), _JOIN_TIMEOUT_CEILING_SEC),
        (-5.0, 0.0),
        (0.0, 0.0),
        (10.0, 10.0),
        (_JOIN_TIMEOUT_CEILING_SEC, _JOIN_TIMEOUT_CEILING_SEC),
        (_JOIN_TIMEOUT_CEILING_SEC * 10.0, _JOIN_TIMEOUT_CEILING_SEC),
    ],
    ids=["inf", "-inf", "nan", "negative", "zero", "normal", "at_ceiling", "far_above_ceiling"],
)
def test_bounded_join_timeout_never_raises_and_stays_finite(raw: float, expected: float) -> None:
    result = _bounded_join_timeout(raw)
    assert math.isfinite(result), f"_bounded_join_timeout({raw!r}) must always return finite, got {result!r}"
    assert 0.0 <= result <= _JOIN_TIMEOUT_CEILING_SEC
    assert result == expected


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


# Integration: the REAL poller's own tick invokes `_escalate_and_finalize` directly — the
# reproduction path, entirely outside `_finalize_round`'s catch-all.
def test_escalate_and_finalize_survives_non_finite_worker_kill_grace_sec(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    clock = FakeClock(0.0)
    bad_cfg = _cfg_with_bypassed_worker_kill_grace_sec(float("inf"))
    assert not math.isfinite(bad_cfg.worker_kill_grace_sec)  # confirm the injection landed
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, eval_cfg=bad_cfg, sink=sink, clock=clock), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None
        assert proc.alive is True  # started, never told to die -> a genuine hang

        # Push the fake clock past `round_timeout_sec`: the REAL poller's next tick sees the
        # elapsed time and calls `_escalate_and_finalize` ON ITS OWN, rather than by hand.
        clock.advance(1000.0)

        def _wait_for_result():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = pipeline.poll_completed()
                if result is not None:
                    return result
                time.sleep(0.01)
            return None

        result = bounded(_wait_for_result, timeout=6.0)

        # 1. escalation completed and delivered a result, never a dead thread.
        assert result is not None, (
            "escalation with a non-finite worker_kill_grace_sec must still deliver a "
            "result via the mailbox, never hang the poller forever"
        )
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        # 2. the escalation ran, and every timeout that reached `.join()` was bounded BEFORE it.
        assert proc.terminated is True
        assert proc.killed is True
        assert proc.join_calls, "escalate_and_finalize must join() the process at least once"
        for called_timeout in proc.join_calls:
            if called_timeout is not None:
                assert math.isfinite(called_timeout), (
                    f"a non-finite timeout ({called_timeout!r}) reached Process.join() -- "
                    "the layer-2 clamp must run BEFORE every join() call"
                )

        # 3. a named eval_broken event was emitted (never silent).
        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the non-finite-grace escalation"

        # The poller THREAD is still alive, the exact invariant the finding found broken.
        assert pipeline._poller.is_alive(), (  # noqa: SLF001 -- intentional internal check
            "the eval-pipeline-poller thread must survive a non-finite worker_kill_grace_sec"
        )

        # 5. the pipeline is still usable afterwards.
        assert pipeline.poll_completed() is None
    finally:
        pipeline.stop()


# The drain/teardown path shares the identical `proc.join(...)` call shape and is reachable by
# the same crash, so it is verified here too.
def test_drain_pending_survives_non_finite_worker_kill_grace_sec(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    bad_cfg = _cfg_with_bypassed_worker_kill_grace_sec(float("inf"))
    pipeline = build_eval_pipeline(**pipeline_kwargs(tmp_path, eval_cfg=bad_cfg, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        proc = fake_mp.last_process
        assert proc is not None
        assert proc.alive is True  # a genuine hang: drain_pending must terminate/kill it

        result = bounded(lambda: pipeline.drain_pending(), timeout=5.0)

        assert result is not None, "drain_pending() must never hang on a non-finite grace period"
        assert result["eval_broken_reason"] is not None
        assert proc.terminated is True
        assert proc.killed is True
        for called_timeout in proc.join_calls:
            if called_timeout is not None:
                assert math.isfinite(called_timeout)

        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the non-finite-grace drain"
    finally:
        pipeline.stop()
