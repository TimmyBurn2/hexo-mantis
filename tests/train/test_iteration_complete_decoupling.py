"""`iteration_complete` is decoupled from `log_interval`, and costs ONE pool read per emit.

`games_total` is a per-iteration counter, not a training-logging event, so the emit rides every
coordinator step while the `training_step` alerting path stays `log_interval`-gated. The single
`runner_stats()` read is a semantic change, not a saving: the target-integrity block and the
mcts/cluster block become one atomic read instead of two that could straddle a game boundary.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from _graph_drive import dev_coordinator_config, GRAPH_FULL_CONFIG, GraphSampleBuffer
from _spy import SpyEventSink
from _monitor_config import monitor_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from _drivable import DrivablePoolStub, DrivableTrainerStub, RunnerStats


class _CountingPool(DrivablePoolStub):
    """A pool fake whose `runner_stats` call counter is the collapse oracle's instrument."""

    def __init__(self) -> None:
        super().__init__()
        self.runner_stats_calls = 0

    def runner_stats(self) -> Any:
        self.runner_stats_calls += 1
        return RunnerStats()


class _FakeEvalPipeline:
    def __init__(self) -> None:
        self.run_calls = 0
        self.drain_calls = 0
        self.poll_calls = 0

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.run_calls += 1
        return {"status": "skipped"}

    def drain_pending(self):
        self.drain_calls += 1
        return None

    def poll_completed(self):
        self.poll_calls += 1
        return None


def _make_coordinator(*, pool=None, config=None):
    pool = pool or _CountingPool()
    trainer = DrivableTrainerStub()
    buffer = GraphSampleBuffer()
    sink = SpyEventSink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer,
        pool=pool, eval_pipeline=_FakeEvalPipeline(),
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(),
        config=config or dev_coordinator_config(),
        full_config=GRAPH_FULL_CONFIG,
        sink=sink, monitor_cfg=monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer, sink=sink)


def test_on1_iteration_complete_emits_below_log_interval() -> None:
    """At `log_interval=1000` a step well below 1000 emits exactly ONE `iteration_complete`
    carrying `games_total`.

    Killer: re-introduce the `_train_step % cfg.log_interval != 0` early return on the emit
    path, which drops the count to zero.
    """
    cfg = dev_coordinator_config(log_interval=1000)
    h = _make_coordinator(config=cfg)
    h.pool.games_completed = 5  # above min_buf_size, new_games > 0 → O6 burst path

    h.coord.step()

    iter_events = h.sink.named("iteration_complete")
    assert len(iter_events) == 1, (
        f"O-N1: iteration_complete must emit on EVERY coordinator step (R210: per-iteration "
        f"counter, not a training-logging event). At log_interval=1000, _train_step=1, "
        f"expected 1 emit; got {len(iter_events)}. If 0, the emit is still gated by "
        f"_run_log_interval's `step % log_interval != 0` early return (the coupling defect "
        f"this chunk fixes)."
    )
    assert iter_events[0].get("games_total") == 5, (
        f"O-N1: iteration_complete.games_total must carry the coordinator's games-played "
        f"counter. Got {iter_events[0].get('games_total')!r}."
    )


def test_on1_training_step_alerting_stays_gated_below_log_interval() -> None:
    """The alerting path stays gated: below `log_interval` there is no `training_step`, no
    `training_alert` and no `monitor_gates`.

    `monitor_gates` rides `monitor.gate_interval`, which this drive MIRRORS onto `log_interval`
    as every committed config does; the two knobs stated apart are pinned elsewhere.
    """
    cfg = dev_coordinator_config(log_interval=1000)
    h = _make_coordinator(config=cfg)
    h.pool.games_completed = 5

    h.coord.step()

    assert h.sink.named("training_step") == [], (
        "O-N1 conjunct: the coordinator's training_step event must STAY log_interval-gated "
        "(R210). If this fires at step < log_interval, the decoupling over-reached."
    )
    assert h.sink.named("monitor_gates") == [], (
        "O-N1 conjunct: monitor_gates must stay GATED — on monitor.gate_interval since R242, "
        "which this drive mirrors onto log_interval exactly as every committed config does."
    )
    assert h.sink.named("training_alert") == [], (
        "O-N1 conjunct: the WARN rules must STAY log_interval-gated (R210)."
    )


def test_on1b_collapse_one_runner_stats_call_per_iteration_complete() -> None:
    """Exactly ONE `pool.runner_stats()` call fires per `iteration_complete` emit — the
    snapshot `_target_integrity_report` took, passed on rather than re-read.

    Killer: restore the local `rstats = pool.runner_stats()` inside the emit, which is two
    reads that can straddle a game boundary.
    """
    cfg = dev_coordinator_config(log_interval=1000)
    h = _make_coordinator(config=cfg)
    h.pool.games_completed = 5

    calls_before = h.pool.runner_stats_calls
    h.coord.step()
    calls_after = h.pool.runner_stats_calls
    calls_this_emit = calls_after - calls_before

    iter_events = h.sink.named("iteration_complete")
    assert len(iter_events) == 1, (
        "O-N1b precondition: iteration_complete must emit once (O-N1). If this is 0, "
        "O-N1 has not landed yet and O-N1b cannot be evaluated."
    )
    assert calls_this_emit == 1, (
        f"O-N1b (R218 rider 1): exactly ONE pool.runner_stats() call per "
        f"iteration_complete emit (the _target_integrity_report snapshot, passed into "
        f"emit_iteration_complete_event via the rstats kwarg). Got {calls_this_emit} "
        f"calls. If 2, the Q-O-TWO-POOL-READS collapse has not landed — events.py:297 "
        f"still makes its own pool.runner_stats() call alongside step.py:650's. The "
        f"collapse ELIMINATES the straddle (R218: a semantic change, not a no-op)."
    )
