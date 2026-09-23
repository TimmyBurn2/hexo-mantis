"""The playout-cap and Gumbel round counters reach `iteration_complete` from a REAL runner."""
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
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import RunnerStats, runner_stats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_ENCODING = "gnn_axis_v1"
_N_WORKERS = 1
_N_SIMS_QUICK = 8
_N_SIMS_FULL = 24
# 48 moves put a stuck fair-coin draw under 2^-47 (the Rust leg's reasoning, same lever).
_WANT_POSITIONS = 48
_TIMEOUT_S = 120.0
#: Transcribed, not read off the subject: a consistent rename must not satisfy the oracle.
_LEVERS = ("pcr_full_moves", "pcr_quick_moves", "gumbel_round_leaves", "gumbel_rounds")
_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Drive:
    stats: RunnerStats
    runner: Any


class _Pool:
    """The emission-side pool surface over the REAL runner; `runner_stats` is the production read."""

    search_kind = "gumbel"
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
    """One Gumbel self-play drive with the playout cap armed at a fair coin between two budgets."""
    cfg = _engine.SelfPlayRunnerConfig(
        n_workers=_N_WORKERS, max_moves_per_game=4, n_simulations=_N_SIMS_QUICK,
        leaf_batch_size=4, quiescence_enabled=False, q_rescale=True, search_stats_every=0,
        gumbel_m=4, full_search_prob=0.5, n_sims_quick=_N_SIMS_QUICK, n_sims_full=_N_SIMS_FULL,
        random_opening_plies=0, encoding_name=_ENCODING,
    )
    cfg.search_kind = "gumbel"
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
    # Read AFTER the join, so no counter can trail the positions it is compared against.
    yield _Drive(stats=_Pool(runner).runner_stats(), runner=runner)


def test_the_real_runner_publishes_both_playout_cap_arms(drive: _Drive) -> None:
    """Every searched move draws one arm; only a stop-interrupted move may lead, one per worker."""
    st = drive.stats
    assert st.positions_generated >= _WANT_POSITIONS, f"the drive stalled: {st}"
    assert st.pcr_full_moves > 0 and st.pcr_quick_moves > 0, (
        f"one arm never reached Python (full={st.pcr_full_moves}, quick={st.pcr_quick_moves}) "
        "at p=0.5 over this many moves: a getter that reads 0 is a lever nobody can see fire"
    )
    drawn = st.pcr_full_moves + st.pcr_quick_moves
    assert 0 <= drawn - st.positions_generated <= _N_WORKERS, (
        f"{drawn} arms drawn against {st.positions_generated} moves played: the two counters "
        "are not counting the same moves"
    )


def test_the_real_runner_publishes_the_gumbel_round_width(drive: _Drive) -> None:
    """At least one round per Gumbel move, one leaf per round, never past the full budget."""
    st = drive.stats
    drawn = st.pcr_full_moves + st.pcr_quick_moves
    assert st.gumbel_rounds >= st.positions_generated > 0, (
        f"{st.gumbel_rounds} rounds over {st.positions_generated} Gumbel moves"
    )
    assert st.gumbel_rounds <= st.gumbel_round_leaves <= drawn * _N_SIMS_FULL, (
        f"round width terms out of range: {st.gumbel_round_leaves} leaves / "
        f"{st.gumbel_rounds} rounds over {drawn} searches"
    )


def test_the_counters_reach_iteration_complete(drive: _Drive) -> None:
    """The coordinator's `search_levers` block carries the same totals, deltas and rates."""
    dev = load_config(_REPO / "configs" / "dev_example.yaml")
    config = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
        drain_caps=resolve_drain_caps(dev.monitor), gate_interval=dev.monitor.gate_interval,
        knobs=resolve_coordinator_knobs(dev.train))
    events: list[dict[str, Any]] = []
    coord = StepCoordinator(
        trainer=None, buffer=SimpleNamespace(size=0, capacity=1), pretrained_buffer=None,
        recent_buffer=None, pool=_Pool(drive.runner), eval_pipeline=None,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        full_config={}, sink=SimpleNamespace(emit=lambda e: events.append(dict(e))),
        monitor_cfg=MonitorConfig(),
    )
    coord._emit_iteration_complete(config)
    (payload,) = [e for e in events if e["event"] == "iteration_complete"]
    block = payload["search_levers"]
    st = drive.stats
    assert block["positions_delta"] == st.positions_generated
    assert sorted(set(_LEVERS) - set(block)) == [], f"levers missing from the stream: {block}"
    for name in _LEVERS:
        want = getattr(st, name)
        assert block[name]["total"] == want and block[name]["delta"] == want, (
            f"{name} did not thread 1:1 into the stream: {block[name]} against {want}"
        )
        assert block[name]["per_position"] == pytest.approx(want / st.positions_generated)
