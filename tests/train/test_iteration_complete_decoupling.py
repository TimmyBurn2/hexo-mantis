# >300 justify (R8): ONE decoupling claim driven by two oracles that must see the SAME emit —
# one counts the `iteration_complete` events a burst produces, the other the
# `pool.runner_stats()` calls those same events cost. The `_CountingPool` spy is the instrument
# for both, so a split forks the fake that makes each side's numbers meaningful.
"""`iteration_complete` is decoupled from `log_interval`, and costs ONE pool read per emit.

`games_total` is a per-iteration counter, not a training-logging event, so the emit rides every
coordinator step while the `training_step` alerting path stays `log_interval`-gated. The single
`runner_stats()` read is a semantic change, not a saving: the target-integrity block and the
mcts/cluster block become one atomic read instead of two that could straddle a game boundary.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

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

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """Build a real graph ring the coordinator stubs sample through."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



#: What a `StepCoordinator` reads on the graph route: the dispatch identity plus the sections
#: its resolvers read. The caps are the NON-BINDING pair — nothing here exercises a split.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


# Constants derived from the minted config — no hand-restated knobs.
_CONFIG = load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_CONFIG.train)
#: The ARMING cadence, from the same minted config.
_GATE_INTERVAL = _CONFIG.monitor.gate_interval


def _make_config(**overrides) -> StepCoordinatorConfig:
    """Build a coordinator config whose `gate_interval` mirrors `log_interval` unless a drive
    names it — the shipped posture, since every committed config mints the two equal."""
    settings = {"eval_interval": 1, "log_interval": 1, "min_buf_size": 10, **overrides}
    settings.setdefault("gate_interval", settings["log_interval"])
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **settings,
    )


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _CountingPool:
    """A pool fake whose `runner_stats` call counter is the collapse oracle's instrument."""

    def __init__(self) -> None:
        self.games_completed = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # the third outcome share
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
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # Delegated to a real `HexgBuffer`: the dispatcher collates the wire for real, so a
        # hand-built payload would be a second wire format for the collate to disagree with.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


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
        full_config=_GRAPH_FULL_CONFIG,
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer, sink=sink)


def test_on1_iteration_complete_emits_below_log_interval() -> None:
    """At `log_interval=1000` a step well below 1000 emits exactly ONE `iteration_complete`
    carrying `games_total`.

    Killer: re-introduce the `_train_step % cfg.log_interval != 0` early return on the emit
    path, which drops the count to zero.
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
    """The alerting path stays gated: below `log_interval` there is no `training_step`, no
    `training_alert` and no `monitor_gates`.

    `monitor_gates` rides `monitor.gate_interval`, which this drive MIRRORS onto `log_interval`
    as every committed config does; the two knobs stated apart are pinned elsewhere.
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
