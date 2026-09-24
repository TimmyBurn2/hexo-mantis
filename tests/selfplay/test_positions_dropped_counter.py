"""The results-queue cap's drop count reaches `iteration_complete` from a REAL runner."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mantis import _engine
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from _monitor_config import monitor_config
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import RunnerStats, runner_stats
from mantis.train.coordinator.step import _SEARCH_LEVER_COUNTERS, StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_CAP = 1
_WANT_POSITIONS = 24
_TIMEOUT_S = 120.0
_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Drive:
    stats: RunnerStats
    runner: Any
    retained: int


class _Pool:
    """The emission-side pool surface over the REAL runner; `runner_stats` is the production read."""

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


def _answer_uniformly(batcher: Any, stop: threading.Event) -> None:
    """Serve every graph request a uniform prior over its legal cells and a zero value."""
    while not stop.is_set():
        ids, wire = batcher.next_graph_batch(8, 5)
        ids = list(ids)
        if not ids:
            continue
        offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
        probs = np.zeros((int(offsets[-1]),), dtype=np.float32)
        for i in range(len(ids)):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if end > start:
                probs[start:end] = 1.0 / (end - start)
        batcher.submit_graph_inference_results(
            ids, probs, offsets, np.zeros((len(ids),), dtype=np.float32))


@pytest.fixture(scope="module")
def drive() -> Iterator[_Drive]:
    """Self-play with a one-row results queue and NO drain, so every finished game overflows it."""
    cfg = _engine.SelfPlayRunnerConfig(
        n_workers=1, max_moves_per_game=4, n_simulations=4, leaf_batch_size=4,
        quiescence_enabled=False, q_rescale=True, search_stats_every=0, results_queue_cap=_CAP,
        random_opening_plies=0, encoding_name="gnn_axis_v1",
    )
    runner = _engine.SelfPlayRunner(cfg)
    stop = threading.Event()
    consumer = threading.Thread(target=_answer_uniformly, args=(runner.batcher, stop), daemon=True)
    consumer.start()
    runner.start()
    try:
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and runner.positions_generated < _WANT_POSITIONS:
            time.sleep(0.01)
    finally:
        runner.stop()
        stop.set()
        consumer.join(timeout=10)
    stats = _Pool(runner).runner_stats()
    yield _Drive(stats=stats, runner=runner, retained=len(runner.collect_graph_data()))


def test_the_cap_drops_undrained_rows_and_the_runner_counts_them(drive: _Drive) -> None:
    """Every finished row beyond the cap is dropped AND counted: none vanishes uncounted."""
    st = drive.stats
    assert st.positions_generated >= _WANT_POSITIONS, f"the drive stalled: {st}"
    assert drive.retained <= _CAP, f"{drive.retained} rows survived a cap of {_CAP}"
    assert st.positions_dropped > 0, "an undrained queue over its cap dropped nothing it counted"
    assert st.positions_dropped + drive.retained <= st.positions_generated, (
        f"{st.positions_dropped} dropped + {drive.retained} kept > {st.positions_generated} made"
    )


def test_the_count_reaches_iteration_complete(drive: _Drive) -> None:
    """The coordinator's `target_integrity` block carries the drop total, delta and rate."""
    dev = load_config(_REPO / "configs" / "dev_example.yaml")
    config = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
        drain_caps=resolve_drain_caps(dev.monitor), gate_interval=dev.monitor.gate_interval,
        knobs=resolve_coordinator_knobs(dev.train))
    events: list[dict[str, Any]] = []
    coord = StepCoordinator(
        trainer=None, buffer=SimpleNamespace(size=0, capacity=1),
        pool=_Pool(drive.runner), eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), config=config,
        full_config={}, sink=SimpleNamespace(emit=lambda e: events.append(dict(e))),
        monitor_cfg=monitor_config(),
    )
    coord._emit_iteration_complete(config)
    (payload,) = [e for e in events if e["event"] == "iteration_complete"]
    block = payload["target_integrity"]["positions_dropped"]
    want = drive.stats.positions_dropped
    assert block["total"] == want and block["delta"] == want, f"{block} against {want}"
    assert block["per_position"] == pytest.approx(want / drive.stats.positions_generated)


def test_a_runner_without_the_getter_is_refused_not_published_as_zero() -> None:
    """An engine lacking the drop getter raises in `runner_stats` instead of reporting 0."""
    runner = SimpleNamespace(**dict.fromkeys(_SEARCH_LEVER_COUNTERS, 1))
    with pytest.raises(AttributeError, match="positions_dropped"):
        runner_stats(_Pool(runner))
