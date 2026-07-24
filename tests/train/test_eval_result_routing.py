"""⊕ WP11-A — the main-thread eval-result routing seam (`StepCoordinator.step()` polling).

RED-at-import until IMPL writes `mantis.eval.pipeline` (the concrete `EvalPipelineLike`).
ORACLE-FIRST (⊕): the top-level `import mantis.eval.pipeline` raises ModuleNotFoundError
before any port code exists. `StepCoordinator`/`drain.py` ALREADY EXIST at HEAD (WP13-A) —
`step()` does not yet call a `_poll_eval_results()` at its top (that call lands with IMPL,
design §a.4); this suite pins the behavior the addition must produce.

Twin of the WP13-A `_pending_eval_result` drain (old `step_coordinator.py` L1120-1125): a
non-blocking `poll_completed()` at the TOP of every `step()` iteration, routed through
`drain._route_eval_result` -> `on_eval_round_complete`, all on the MAIN thread — `step()`
never blocks on eval (WP13-A P-06 twin) and never consumes the kick ACK for WR.

>300 justify: one seam (`step()`'s eval-poll integration) exercised against the real
`StepCoordinator` + `drain.py` with one shared fake-pool/fake-trainer/fake-pipeline harness,
mirroring `tests/train/test_coordinator_gates.py`'s harness shape so the two suites read as
one family; splitting the harness from its seven call sites would duplicate it for no gain.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import mantis.eval.pipeline  # noqa: F401 — RED-at-import anchor
from mantis.monitor.config import MonitorConfig
from mantis.train.coordinator import drain
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState


# ── fakes (mirrors tests/train/test_coordinator_gates.py's harness shape) ────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class FakePool:
    def __init__(self) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def check_producer_health(self) -> None:
        return None

    def per_worker_draw_rates(self) -> dict[int, float]:
        return {}

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


class FakeTrainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3, "opp_reply_loss": 0.0,
                "loss_total": 1.0}

    def save_checkpoint(self, loss_info) -> None:
        return None


class FakeBuffer:
    def __init__(self, size: int = 1000, capacity: int = 100_000) -> None:
        self.size = size
        self.capacity = capacity

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None


class SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


class ThreadIdentSpyEvalPipeline:
    """An eval pipeline whose `poll_completed()` records the thread it was called from
    (main-thread proof) and whose `run_evaluation`/`drain_pending`/`apply_gate_decision`
    are call-count spies (the never-blocks assertion)."""

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

    def apply_gate_decision(self, result, *, sync_inference: bool) -> int | None:
        self.apply_gate_calls.append({"result": dict(result), "sync_inference": sync_inference})
        return result.get("promoted_step") if result.get("promoted") else None


def _make_config(**overrides) -> StepCoordinatorConfig:
    base = dict(
        eval_interval=1, log_interval=1, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=10, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        instrumentation_enabled=False, stop_step=10**9, final_eval_drain_timeout_sec=900.0,
    )
    base.update(overrides)
    return StepCoordinatorConfig(**base)


def _make_coordinator(*, eval_pipeline=None, config=None):
    pool = FakePool()
    trainer = FakeTrainer()
    buffer = FakeBuffer()
    shutdown = ShutdownState()
    sink = SpySink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None,
        config=config or _make_config(), full_config={}, train_cfg={}, mixing_cfg={},
        sink=sink, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           shutdown=shutdown, sink=sink, eval_pipeline=eval_pipeline)


def test_step_polls_and_routes_completed_rounds_on_main_thread() -> None:
    """`step()` must poll the injected eval pipeline and route ANY completed round through
    `on_eval_round_complete`, on the MAIN thread that called `step()`."""
    pipe = ThreadIdentSpyEvalPipeline(poll_result={"step": 5, "wr_sealbot": 0.6,
                                                    "promoted": False, "eval_broken": False})
    h = _make_coordinator(eval_pipeline=pipe)
    routed: list[dict] = []
    h.coord.on_eval_round_complete = lambda result: routed.append(dict(result))
    h.pool.games_completed = 5
    main_thread_id = threading.get_ident()

    h.coord.step()

    assert pipe.poll_calls_from_thread, "step() must call poll_completed() at least once"
    assert all(tid == main_thread_id for tid in pipe.poll_calls_from_thread), (
        "poll_completed() must be called from the SAME (main) thread that called step()"
    )
    assert routed and routed[0]["wr_sealbot"] == 0.6, (
        "a completed round returned by poll_completed() must reach on_eval_round_complete"
    )


def test_step_never_blocks_on_eval() -> None:
    """Blocking-call spy: `step()` must make ZERO calls to `drain_pending()` (the WP13-A P-06
    twin — a blocking drain inside step() is the run3 wedge class)."""
    pipe = ThreadIdentSpyEvalPipeline(poll_result=None)
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5

    h.coord.step()

    assert pipe.drain_calls == 0, "step() must never call drain_pending() (blocking call)"


def test_kick_ack_busy_sets_eval_skipped_busy_outcome() -> None:
    """A busy kick ack (`{"kicked": False, "reason": "busy", ...}`) must surface as
    `eval_skipped_busy` on the returned `StepOutcome` — never silently dropped."""
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


def test_promoted_result_applies_gate_decision_midrun_with_sync() -> None:
    """A completed round with `promoted=True` routed mid-run (via `poll_completed()`) must
    apply the gate decision WITH `sync_inference=True` — the pool is still up mid-run."""
    result = {"step": 7, "promoted": True, "promoted_step": 7, "wr_sealbot": 0.9,
              "eval_broken": False}
    pipe = ThreadIdentSpyEvalPipeline(poll_result=result)
    h = _make_coordinator(eval_pipeline=pipe)

    def _on_complete(r):
        drain._apply_promotion(h.coord, r, sync_inference=True)

    h.coord.on_eval_round_complete = _on_complete
    h.pool.games_completed = 5

    h.coord.step()

    assert pipe.apply_gate_calls, "a promoted result must invoke apply_gate_decision"
    assert pipe.apply_gate_calls[0]["sync_inference"] is True, (
        "mid-run promotion must sync inference weights (pool still up)"
    )


def test_terminal_route_applies_without_sync() -> None:
    """The terminal route (`drain.run_terminal_eval` -> `_route_eval_result`) must apply a
    promoted gate decision with `sync_inference=False` (pool already stopped; run3 parity,
    step_coordinator.py:1705-1710)."""
    result = {"step": 9, "promoted": True, "promoted_step": 9, "wr_sealbot": 0.9,
              "eval_broken": False}
    pipe = ThreadIdentSpyEvalPipeline()
    pipe.run_evaluation = lambda *a, **k: dict(result)  # terminal eval RETURNS the result
    h = _make_coordinator(eval_pipeline=pipe)

    def _on_complete(r):
        drain._apply_promotion(h.coord, r, sync_inference=False)

    h.coord.on_eval_round_complete = _on_complete
    drain.run_terminal_eval(h.coord)

    assert pipe.apply_gate_calls, "the terminal route must invoke apply_gate_decision"
    assert pipe.apply_gate_calls[0]["sync_inference"] is False, (
        "terminal promotion must NOT sync inference weights (pool already stopped)"
    )


def test_flush_before_pool_stop_before_terminal_order() -> None:
    """close_out ordering (drain.py:105-134 contract; disarm-first, O-27 untouched): disarm
    -> flush_pending_eval -> on_drained -> run_terminal_eval. Exercises ONLY today's
    already-shipped `drain.close_out` (WP13-A) — a regression pin on ordering the new eval
    pipeline must not disturb, NOT an oracle for unbuilt IMPL behavior. It would PASS today
    in isolation (drain.py already implements this order), but this whole FILE's top-level
    `import mantis.eval.pipeline` fails first (RED-at-import, correct per the design's
    per-suite law), so this test's own correctness-today is not separately observable until
    that import exists — noted here so a reviewer doesn't mistake the file-level RED for
    this specific assertion being wrong."""
    order: list[str] = []
    watchdog = SimpleNamespace(disarm_staleness=lambda: order.append("disarm"))
    pipe = SimpleNamespace(
        drain_pending=lambda: (order.append("flush_pending_eval"), None)[1],
        run_evaluation=lambda *a, **k: (order.append("run_terminal_eval"), None)[1],
    )
    coord = SimpleNamespace(
        heartbeat_watchdog=watchdog, eval_pipeline=pipe,
        config=SimpleNamespace(terminal_eval_enabled=True),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        _train_step=1000, _sink=None, eval_model=object(), full_config={},
        on_eval_round_complete=lambda result: None,
    )
    drain.close_out(coord, on_drained=lambda: order.append("on_drained"))

    assert order == ["disarm", "flush_pending_eval", "on_drained", "run_terminal_eval"], (
        f"close_out ordering must be disarm -> flush -> on_drained -> terminal: {order}"
    )
