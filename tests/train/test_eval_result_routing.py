"""The main-thread eval-result routing seam: `StepCoordinator.step()`'s eval poll.

A non-blocking `poll_completed()` at the TOP of every `step()` iteration, routed through
`drain._route_eval_result` to the promotion seam, all on the MAIN thread — `step()` never blocks
on eval and never consumes the kick ACK as a result.
"""
from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest

from _drivable import DrivablePoolStub
from _graph_drive import DEV_DRAIN_CAPS, DEV_GATE_INTERVAL, DEV_KNOBS, GRAPH_FULL_CONFIG, GraphSampleBuffer, mirrored
from _monitor_config import monitor_config
from mantis.run import _step_coordinator_config
from mantis.train.coordinator import drain
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from _spy import SpyEventSink


class FakeTrainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    # The double conforms to the DECLARED seam: typed entry points plus `device`.
    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3, "opp_reply_loss": 0.0,
                "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def save_checkpoint(self, loss_info) -> None:
        return None


class ThreadIdentSpyEvalPipeline:
    """An eval pipeline that records `poll_completed()`'s calling thread; the rest are spies."""

    def __init__(self, *, poll_result=None, ack: dict | None = None) -> None:
        self._poll_result = poll_result
        self._ack = ack if ack is not None else {"kicked": True, "round_id": "r0", "step": 0,
                                                  "reason": None}
        self.poll_calls_from_thread: list[int] = []
        self.run_calls = 0
        self.drain_calls = 0
        self.apply_gate_calls: list[dict] = []

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.run_calls += 1
        return dict(self._ack)

    def poll_completed(self):
        self.poll_calls_from_thread.append(threading.get_ident())
        return self._poll_result

    def drain_pending(self):
        self.drain_calls += 1
        return None

    def apply_gate_decision(self, result) -> int | None:
        # Single-signature applier: a drain that still threads a sync flag fails here loudly.
        self.apply_gate_calls.append({"result": dict(result)})
        return result.get("promoted_step") if result.get("promoted") else None


def _make_config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder, this file's deltas only; `None` is the EXPLICIT
    disarmed draw-rate posture, defaulted by neither the builder nor this factory."""
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
                                 drain_caps=DEV_DRAIN_CAPS, gate_interval=DEV_GATE_INTERVAL,
                                 knobs=DEV_KNOBS),
        **mirrored({"eval_interval": 1, "log_interval": 1, "min_buf_size": 10,
                    **overrides}),
    )


def _make_coordinator(*, eval_pipeline=None, config=None):
    pool = DrivablePoolStub()
    trainer = FakeTrainer()
    buffer = GraphSampleBuffer()
    shutdown = ShutdownState()
    sink = SpyEventSink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(),
        config=config or _make_config(), full_config=GRAPH_FULL_CONFIG,
        sink=sink, monitor_cfg=monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           shutdown=shutdown, sink=sink, eval_pipeline=eval_pipeline)


def test_step_polls_and_routes_completed_rounds_on_main_thread() -> None:
    """`step()` polls the pipeline and routes a completed round on the MAIN thread — to the
    promotion seam, the one consumer a routed round has."""
    pipe = ThreadIdentSpyEvalPipeline(poll_result={"step": 5, "promoted": True,
                                                    "eval_broken_reason": None})
    h = _make_coordinator(eval_pipeline=pipe)
    routed: list[dict] = []
    pipe.apply_gate_decision = lambda result: routed.append(dict(result))
    h.pool.games_completed = 5
    main_thread_id = threading.get_ident()

    h.coord.step()

    assert pipe.poll_calls_from_thread, "step() must call poll_completed() at least once"
    assert all(tid == main_thread_id for tid in pipe.poll_calls_from_thread), (
        "poll_completed() must be called from the SAME (main) thread that called step()"
    )
    assert routed and routed[0]["step"] == 5, (
        "a completed round returned by poll_completed() must reach the promotion seam"
    )


def test_step_never_blocks_on_eval() -> None:
    """`step()` makes ZERO `drain_pending()` calls: a blocking drain in step() is the wedge."""
    pipe = ThreadIdentSpyEvalPipeline(poll_result=None)
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5

    h.coord.step()

    assert pipe.drain_calls == 0, "step() must never call drain_pending() (blocking call)"


def test_kick_ack_busy_sets_eval_skipped_busy_outcome() -> None:
    """A busy kick ack must surface as `eval_skipped_busy`, never be silently dropped."""
    pipe = ThreadIdentSpyEvalPipeline(
        ack={"kicked": False, "reason": "busy", "round_id": "r-inflight", "step": 5},
    )
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5

    outcome = h.coord.step()

    assert pipe.run_calls >= 1, "the eval kick must still have fired at the boundary"
    assert outcome.eval_skipped_busy is True, (
        "a busy kick ack must set StepOutcome.eval_skipped_busy=True"
    )
    assert outcome.eval_kicked_off is False, (
        "a busy ack (kicked=False) must not also report eval_kicked_off=True"
    )


def test_promoted_result_advances_deploy_tag_midrun_without_touching_pool() -> None:
    """A promoted round routed mid-run reaches the ONE applier with ZERO pool sync calls."""
    result = {"step": 7, "promoted": True, "promoted_step": 7,
              "eval_broken_reason": None}
    pipe = ThreadIdentSpyEvalPipeline(poll_result=result)
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5

    h.coord.step()

    assert pipe.apply_gate_calls, "a promoted result must invoke apply_gate_decision"
    assert pipe.apply_gate_calls[0]["result"]["step"] == 7, (
        "the applier must receive the promoted round's step (the deploy-tag step)"
    )
    assert h.pool.sync_payloads == [] and h.pool.step_calls == [], (
        "a mid-run gate decision must never reach the pool's sync surface (R49)"
    )


def test_terminal_route_applies_identically_to_midrun() -> None:
    """The terminal route calls the SAME applier as the mid-run route, pool stopped."""
    result = {"step": 9, "promoted": True, "promoted_step": 9,
              "eval_broken_reason": None}
    pipe = ThreadIdentSpyEvalPipeline()
    pipe.run_evaluation = lambda *a, **k: dict(result)  # terminal eval RETURNS the result
    h = _make_coordinator(eval_pipeline=pipe)

    drain.run_terminal_eval(h.coord)  # pool never started, never touched

    assert pipe.apply_gate_calls, "the terminal route must invoke apply_gate_decision"
    assert pipe.apply_gate_calls[0]["result"]["step"] == 9, (
        "the terminal route must hand the applier the same single-signature call shape"
    )
    assert h.pool.sync_payloads == [] and h.pool.step_calls == [], (
        "a terminal gate decision must never reach the pool's sync surface (R49)"
    )


def test_flush_before_pool_stop_before_terminal_order() -> None:
    """close_out ordering: disarm -> flush_pending_eval -> on_drained -> run_terminal_eval, so
    the flush joins the in-flight round under its budget before the pool goes down."""
    order: list[str] = []
    watchdog = SimpleNamespace(disarm_staleness=lambda: order.append("disarm"))
    pipe = SimpleNamespace(
        drain_pending=lambda: (order.append("flush_pending_eval"), None)[1],
        # The terminal call answers a ROUND RESULT whose `eval_broken_reason` the seam reads.
        run_evaluation=lambda *a, **k: (order.append("run_terminal_eval"),
                                        {"eval_broken_reason": None})[1],
    )
    coord = SimpleNamespace(
        heartbeat_watchdog=watchdog, eval_pipeline=pipe,
        config=SimpleNamespace(terminal_eval_enabled=True),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        _train_step=1000, _sink=None, eval_model=object(), full_config=GRAPH_FULL_CONFIG,
        record_terminal_eval_reason=lambda reason: None,
    )
    drain.close_out(coord, on_drained=lambda: order.append("on_drained"))

    assert order == ["disarm", "flush_pending_eval", "on_drained", "run_terminal_eval"], (
        f"close_out ordering must be disarm -> flush -> on_drained -> terminal: {order}"
    )


def test_an_absent_terminal_eval_enabled_raises_instead_of_inheriting_true() -> None:
    """`run_terminal_eval` reads `cfg.terminal_eval_enabled` as a plain attribute: a `getattr`
    fallback was a SECOND default authority, and a call site omitting the field would silently
    inherit "run the terminal eval" with every field census still green."""
    coord = SimpleNamespace(
        eval_pipeline=SimpleNamespace(run_evaluation=lambda *a, **k: None),
        config=SimpleNamespace(),  # no terminal_eval_enabled — the shape a required field makes
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        _train_step=1000, _sink=None, eval_model=object(), full_config=GRAPH_FULL_CONFIG,
    )
    with pytest.raises(AttributeError, match="terminal_eval_enabled"):
        drain.run_terminal_eval(coord)
