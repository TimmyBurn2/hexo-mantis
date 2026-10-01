"""A root that did not expand is counted by the engine and reaches `iteration_complete.target_integrity`."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _monitor_config import monitor_config
from mantis._engine import SelfPlayRunner, SelfPlayRunnerConfig
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import RunnerStats, runner_stats
from mantis.train.coordinator.step import _SEARCH_LEVER_COUNTERS, StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_REPO = Path(__file__).resolve().parents[2]


class _Pool:
    """The emission-side pool surface; `runner_stats` is the production read."""

    search_kind = "puct"
    avg_game_length = 4.0
    x_winrate = 0.5
    o_winrate = 0.5
    draw_rate = 0.0
    sims_per_sec = None
    batch_fill_pct = 0.0
    recent_move_histories: list = []

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def runner_stats(self) -> RunnerStats:
        return runner_stats(self)


def _stand_in(root_expansion_failed: int) -> SimpleNamespace:
    return SimpleNamespace(**dict.fromkeys(_SEARCH_LEVER_COUNTERS, 0), positions_dropped=0, positions_generated=8,
                           root_expansion_failed=root_expansion_failed, tactics_totals=dict)


def test_the_engine_runner_has_the_getter_and_it_reads_zero_before_a_search() -> None:
    cfg = SelfPlayRunnerConfig(n_workers=1, q_rescale=True, search_stats_every=0, encoding_name="gnn_axis_r8")
    assert SelfPlayRunner(cfg).root_expansion_failed == 0


def test_the_count_reaches_iteration_complete() -> None:
    """The coordinator's `target_integrity` block carries the failure total, delta and rate."""
    dev = load_config(_REPO / "configs" / "dev_example.yaml")
    config = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
        drain_caps=resolve_drain_caps(dev.monitor), gate_interval=dev.monitor.gate_interval,
        knobs=resolve_coordinator_knobs(dev.train))
    events: list[dict[str, Any]] = []
    coord = StepCoordinator(
        trainer=None, buffer=SimpleNamespace(size=0, capacity=1),
        pool=_Pool(_stand_in(3)), eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), config=config,
        full_config={}, sink=SimpleNamespace(emit=lambda e: events.append(dict(e))),
        monitor_cfg=monitor_config(),
    )
    coord._emit_iteration_complete(config)
    (payload,) = [e for e in events if e["event"] == "iteration_complete"]
    block = payload["target_integrity"]["root_expansion_failed"]
    assert block["total"] == 3 and block["delta"] == 3, block
    assert block["per_position"] == pytest.approx(3 / 8)


def test_a_runner_without_the_getter_is_refused_not_published_as_zero() -> None:
    runner = _stand_in(0)
    del runner.root_expansion_failed
    with pytest.raises(AttributeError, match="root_expansion_failed"):
        runner_stats(_Pool(runner))
