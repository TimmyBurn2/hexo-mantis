"""⊕ WP12R Step 3 narration — oracle 1 (part i): `iteration_complete` decoupled from
`log_interval` + the R218 rider 1 `Q-O-TWO-POOL-READS` collapse oracle.

RED-at-IMPL until the narration chunk's part (i) lands. Two oracles, both falsifying:

  O-N1  — `iteration_complete` emits on EVERY coordinator step (every O6 burst return),
          INDEPENDENT of `log_interval`. At `log_interval=1000`, a `step()` at
          `_train_step < 1000` emits exactly ONE `iteration_complete` carrying `games_total`.
          The `training_step` alerting path does NOT fire at step < `log_interval`
          (R210: "training_step alerting stays gated").
          FALSIFYING MUTATION: re-introduce the
          `self._train_step % cfg.log_interval != 0` early return on the
          `iteration_complete` path (re-couple to `_run_log_interval`). This oracle MUST turn
          RED (zero `iteration_complete` emits at step < `log_interval`).

  O-N1b — R218 rider 1 collapse: `emit_iteration_complete_event` uses the `RunnerStats`
          snapshot passed from `_target_integrity_report` (ONE `pool.runner_stats()` call
          per emit), NOT its own `pool.runner_stats()` call at `events.py:297`.
          FALSIFYING MUTATION: re-introduce the second `pool.runner_stats()` call inside
          `emit_iteration_complete_event` (drop the `rstats` kwarg, restore the local
          `rstats = pool.runner_stats()`). This oracle MUST turn RED (two calls per emit).

R210: "games_total is a per-iteration counter, not a training-logging event."
R214: "the narration DESIGN's ORACLE-WRITE stage must assert, mutation-tested, that
iteration_complete emits on every coordinator step at run5's log_interval=1000."
R218 rider 1: "the Q-O-TWO-POOL-READS collapse is a SEMANTIC CHANGE — the target_integrity
snapshot and the mcts_mean_depth/cluster-stats snapshot become ONE atomic read instead of
two microseconds-apart reads that could straddle a game boundary. The oracle for part (i)
MUST assert that emit_iteration_complete_event uses the snapshot passed from
_target_integrity_report (not its own pool.runner_stats() call), and a falsifying mutation
that re-introduces the second runner_stats() call MUST turn RED."
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

# ── minted-config-derived constants (WPMINT Phase K-A/K-B precedent, same as
#    test_coordinator_gates.py — no hand-restated knobs) ─────────────────────────────────
_CONFIG = load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_CONFIG.train)


def _make_config(**overrides) -> StepCoordinatorConfig:
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        **{"eval_interval": 1, "log_interval": 1, "min_buf_size": 10, **overrides},
    )


# ── fakes (minimal, same surface as test_coordinator_gates.py) ──────────────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _CountingPool:
    """FakePool with a `runner_stats` call counter — the R218 rider 1 collapse oracle's
    spy. The counter is what O-N1b reads; every other surface matches FakePool."""

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
        self.runner_stats_calls = 0

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        self.runner_stats_calls += 1
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _FakeTrainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def save_checkpoint(self, loss_info) -> None:
        return None


class _FakeBuffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


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


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _make_coordinator(*, pool=None, config=None):
    pool = pool or _CountingPool()
    trainer = _FakeTrainer()
    buffer = _FakeBuffer()
    sink = _SpySink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=_FakeEvalPipeline(),
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None,
        config=config or _make_config(),
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer, sink=sink)


# ═══ O-N1 — iteration_complete emits every coordinator step, not just at log_interval ═══
def test_on1_iteration_complete_emits_below_log_interval() -> None:
    """O-N1 (R214 part (i) oracle) — at `log_interval=1000`, a `step()` at `_train_step`
    well below 1000 emits exactly ONE `iteration_complete` carrying `games_total`.

    RED until IMPL decouples `iteration_complete` from `_run_log_interval`. Currently
    `_run_log_interval` (`step.py:576`) early-returns on `self._train_step % cfg.log_interval
    != 0`, so at `_train_step=1, log_interval=1000` ZERO `iteration_complete` events emit.

    FALSIFYING MUTATION (drive both ways after IMPL): re-introduce the
    `self._train_step % cfg.log_interval != 0` early return on the `iteration_complete`
    path. This test MUST turn RED (zero emits).
    """
    cfg = _make_config(log_interval=1000)
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
    """O-N1 conjunct (R210: "training_step alerting stays gated") — at `log_interval=1000`,
    a `step()` at `_train_step < 1000` does NOT emit the coordinator's `training_step`
    event, does NOT run the WARN rules (no `training_alert`), and does NOT emit
    `monitor_gates`. These stay on their `log_interval` cadence.

    This conjunct is GREEN at HEAD (the gating is unchanged for the alerting path) and MUST
    stay GREEN after IMPL (the decoupling removes the gate for `iteration_complete` ONLY).
    """
    cfg = _make_config(log_interval=1000)
    h = _make_coordinator(config=cfg)
    h.pool.games_completed = 5

    h.coord.step()

    assert h.sink.named("training_step") == [], (
        "O-N1 conjunct: the coordinator's training_step event must STAY log_interval-gated "
        "(R210). If this fires at step < log_interval, the decoupling over-reached."
    )
    assert h.sink.named("monitor_gates") == [], (
        "O-N1 conjunct: monitor_gates must STAY log_interval-gated (R210)."
    )
    assert h.sink.named("training_alert") == [], (
        "O-N1 conjunct: the WARN rules must STAY log_interval-gated (R210)."
    )


# ═══ O-N1b — R218 rider 1: Q-O-TWO-POOL-READS collapse (ONE runner_stats() per emit) ═══
def test_on1b_collapse_one_runner_stats_call_per_iteration_complete() -> None:
    """O-N1b (R218 rider 1) — `emit_iteration_complete_event` uses the `RunnerStats`
    snapshot passed from `_target_integrity_report` (ONE `pool.runner_stats()` call per
    emit), NOT its own `pool.runner_stats()` call at `events.py:297`.

    RED until IMPL collapses the two reads into one (passes the `rstats` kwarg). Currently
    TWO `runner_stats()` calls fire per `iteration_complete` emit: one at `step.py:650`
    (`_target_integrity_report`) and one at `events.py:297` (inside
    `emit_training_events`/`emit_iteration_complete_event`).

    FALSIFYING MUTATION (drive both ways after IMPL): re-introduce the second
    `pool.runner_stats()` call inside `emit_iteration_complete_event` (drop the `rstats`
    kwarg, restore the local `rstats = pool.runner_stats()`). This test MUST turn RED (two
    calls per emit, not one).

    SEMANTIC CHANGE (R218): the collapse ELIMINATES the straddle — the target_integrity
    block and the mcts_mean_depth/cluster block become ONE atomic read instead of two
    microseconds-apart reads that could straddle a game boundary. This is more correct, not
    neutral.
    """
    cfg = _make_config(log_interval=1000)
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
