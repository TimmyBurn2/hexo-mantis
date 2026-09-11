"""Suite H (lifecycle), plus the pool's representation dispatch.

>300 justify: one lifecycle, one set of collaborator stubs. The producer-death contract, the
start/stop protocol, the four forwarders and the dispatch arms all drive the SAME constructed
pool with the SAME stub runner and stub server, so splitting them would duplicate both stubs and
let the copies drift. The stubs replace the runner and inference server AFTER construction, so
every assertion runs against a real `WorkerPool` built by the real constructor.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.selfplay import pool as pool_mod
from mantis.selfplay.buffers import BufferKind
from mantis.selfplay.pool import WorkerPool

_INTEGRATION_TIMEOUT_S = 60.0


def _cfg(encoding: str, **over: Any) -> dict[str, Any]:
    # `selfplay`/`inference`/`train` are nested schema-shaped sections; `over` layers onto
    # `selfplay`.
    selfplay: dict[str, Any] = {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 8, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 8, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    selfplay.update(over)
    inference = {
        "inference_batch_size": 4, "inference_max_wait_ms": 10,
        "edge_geometry_check": "inline",
        # The graph arm resolves the fused-forward memory bound at construction; NON-BINDING
        # BY CONSTRUCTION here, since this fixture is about wiring and nothing asserts the M.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    train = {"draw_reward": -0.5, "ply_cap_value": -0.5}
    return {"encoding": encoding, "search": {"kind": "puct"}, "selfplay": selfplay,
            "inference": inference, "train": train}


def _graph_pool(**kw: Any) -> WorkerPool:
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=16, num_layers=1)
    return WorkerPool(
        build_net(arch), _cfg("gnn_axis_v1"), torch.device("cpu"),
        HexgBuffer(capacity=256, encoding="gnn_axis_v1", visit_capacity=128), arch=arch, **kw,
    )


class _StubRunner:
    """A runner that produces nothing: the loop spins and sleeps until it is stopped."""

    def __init__(self, *, graph_rows: list | None = None) -> None:
        self.started = 0
        self.stopped = 0
        self.running = False
        self.calls: list[str] = []
        self._graph_rows = graph_rows if graph_rows is not None else []
        self.games_completed = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        self.positions_generated = 0

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.started += 1
        self.running = True

    def stop(self) -> None:
        self.stopped += 1
        self.running = False

    def collect_data(self):
        self.calls.append("collect_data")
        empty = np.zeros(0, dtype=np.float32)
        return (empty,) * 10

    def collect_graph_data(self):
        self.calls.append("collect_graph_data")
        return list(self._graph_rows)

    def drain_game_results(self):
        self.calls.append("drain_game_results")
        return []

class _StubServer:
    def __init__(self, *, forward_count: int = 0, total_requests: int = 0,
                 batch_size: int = 8) -> None:
        self.started = 0
        self.stopped = 0
        self.joins: list[float | None] = []
        self.state_dicts: list[Any] = []
        self._forward_count = forward_count
        self._total_requests = total_requests
        self._batch_size = batch_size
        self.encoding_spec = None

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def load_state_dict_safe(self, state_dict: Any) -> None:
        self.state_dicts.append(state_dict)


class _StubRecorder:
    def __init__(self, path: Any = None) -> None:
        self.steps: list[int] = []
        self.stopped = 0
        self._path = path

    def set_step(self, step: int) -> None:
        self.steps.append(step)

    def maybe_record(self, **kwargs: Any) -> None:
        return None

    def latest_replay_path(self):
        return self._path

    def stop(self) -> None:
        self.stopped += 1


def _stub_collaborators(pool: WorkerPool, **runner_kw: Any) -> tuple[_StubRunner, _StubServer]:
    runner = _StubRunner(**runner_kw)
    server = _StubServer()
    pool._runner = runner
    pool._inference_server = server
    return runner, server


def test_producer_death_is_re_raised_with_its_cause(monkeypatch) -> None:
    """A drain-loop exception leaves the pool flagged and the next `check_producer_health()`
    raises with the original as its `__cause__`. The feeder is the only thing writing training
    data: if it dies quietly, loss and eval numbers stay plausible for hours."""
    pool = _graph_pool()
    boom = ZeroDivisionError("scripted drain failure")

    def _explode(_pool):
        raise boom

    monkeypatch.setattr(pool_mod, "run_stats_loop", _explode)
    pool._stats_loop()

    assert pool._producer_exc is boom
    with pytest.raises(RuntimeError) as exc:
        pool.check_producer_health()
    assert exc.value.__cause__ is boom
    assert "feeder died" in str(exc.value)
    assert "stale buffer" in str(exc.value)


def test_healthy_and_cleanly_stopped_pools_do_not_raise(monkeypatch) -> None:
    """`check_producer_health()` is silent on a fresh pool and after a clean `stop()`."""
    pool = _graph_pool()
    pool.check_producer_health()

    monkeypatch.setattr(pool_mod, "run_stats_loop", lambda _pool: None)
    pool._stats_loop()
    assert pool._producer_exc is None
    pool.check_producer_health()


def test_stats_loop_guard_does_not_let_the_thread_die_silently(monkeypatch, caplog) -> None:
    """The guard LOGS at error level and records the exception: two independent traces of a
    daemon thread that would otherwise unwind unobserved."""
    pool = _graph_pool()

    def _explode(_pool):
        raise RuntimeError("scripted")

    monkeypatch.setattr(pool_mod, "run_stats_loop", _explode)
    with caplog.at_level("ERROR"):
        pool._stats_loop()

    assert pool._producer_exc is not None
    assert any("selfplay_producer_died" in record.message for record in caplog.records), (
        "the death must be visible in the run's own log, not only in a flag"
    )


def test_start_is_idempotent_while_running() -> None:
    """A second `start()` on a running pool is a no-op; two feeder threads on one Rust queue
    would double-count pushes and interleave two `system_stats` cadences."""
    pool = _graph_pool()
    runner, server = _stub_collaborators(pool)

    pool.start()
    first_thread = pool._stats_thread
    assert runner.started == 1 and server.started == 1
    assert first_thread is not None and first_thread.is_alive()

    pool.start()
    assert runner.started == 1, "the runner must not be started twice"
    assert server.started == 1
    assert pool._stats_thread is first_thread, "a second feeder thread was spawned"

    pool.stop()


def test_stop_joins_both_threads_and_stops_the_recorder() -> None:
    """`stop()` sets the stop event, stops runner and server, joins the server with a BOUNDED
    timeout, joins and clears the feeder thread, and stops the recorder. An unbounded join turns
    a wedged inference thread into a hung shutdown with no final checkpoint."""
    recorder = _StubRecorder()
    pool = _graph_pool(recorder=recorder)
    runner, server = _stub_collaborators(pool)

    pool.start()
    pool.stop()

    assert pool._stop_event.is_set()
    assert runner.stopped == 1 and server.stopped == 1
    assert server.joins == [5.0], "the server join must be bounded"
    assert pool._stats_thread is None, "the feeder handle must be cleared"
    assert recorder.stopped == 1


def test_stopped_feeder_thread_actually_exits() -> None:
    """The feeder thread is no longer alive after `stop()`; asserting the `join` call alone
    would pass on a loop that ignores its stop event."""
    pool = _graph_pool()
    _stub_collaborators(pool)

    pool.start()
    thread = pool._stats_thread
    assert thread is not None
    pool.stop()
    assert not thread.is_alive()


def test_sync_inference_weights_forwards_to_the_server() -> None:
    """A promoted state_dict reaches the server's safe swap by identity: a pool that drops the
    call keeps serving OLD weights while promotion logs say otherwise."""
    pool = _graph_pool()
    _, server = _stub_collaborators(pool)
    state = {"layer.weight": torch.zeros(1)}
    pool.sync_inference_weights(state)
    assert len(server.state_dicts) == 1
    assert server.state_dicts[0] is state


def test_recorder_seam_forwards_and_defaults_to_inert() -> None:
    """An injected recorder receives `set_step`; the DEFAULT recorder is inert, because the
    concrete recorder does not exist in this tree."""
    recorder = _StubRecorder(path="replays/games_0001.jsonl")
    pool = _graph_pool(recorder=recorder)
    pool.update_checkpoint_step(42)
    assert recorder.steps == [42]
    assert pool.latest_replay_path() == "replays/games_0001.jsonl"

    default_pool = _graph_pool()
    default_pool.update_checkpoint_step(7)
    assert default_pool.latest_replay_path() is None


@pytest.mark.parametrize(
    "forward_count,total_requests,batch_size,expected",
    [
        (0, 0, 8, 0.0),        # no forwards yet — defined, not a division by zero
        (0, 99, 8, 0.0),       # requests without forwards still cannot divide
        (10, 40, 8, 50.0),     # 40 requests over 10 forwards of 8 = half full
        (10, 80, 8, 100.0),    # exactly full
        (10, 800, 8, 100.0),   # over-full (padding/duplication) clamps at 100
        (4, 4, 0, 100.0),      # a zero batch size must not divide by zero
    ],
)
def test_batch_fill_pct_math(forward_count, total_requests, batch_size, expected) -> None:
    """Batch occupancy reproduces the frozen arithmetic across its four edge cases."""
    pool = _graph_pool()
    pool._inference_server = _StubServer(forward_count=forward_count,
                                         total_requests=total_requests,
                                         batch_size=batch_size)
    assert pool.batch_fill_pct == pytest.approx(expected)


def test_graph_pool_takes_the_graph_arm() -> None:
    """A graph pool calls `collect_graph_data` and never `collect_data`: different formats."""
    pool = _graph_pool()
    runner, _ = _stub_collaborators(pool)

    assert pool._is_graph is True
    assert pool.replay_buffer.kind is BufferKind.GRAPH
    assert (pool._feat_len, pool._chain_len) == (0, 0)
    assert pool._pol_len > 0

    pool.start()
    time.sleep(0.25)
    pool.stop()

    assert "collect_graph_data" in runner.calls
    assert "collect_data" not in runner.calls, (
        "a graph pool must never reach the dense collect path"
    )


@pytest.mark.integration
def test_worker_pool_produces_positions_threaded_smoke() -> None:
    """A real pool with a real Rust runner produces positions and drains them into the buffer —
    the only row that proves the assembled thing runs."""
    pool = _graph_pool()
    pool.start()
    try:
        deadline = time.monotonic() + _INTEGRATION_TIMEOUT_S
        while time.monotonic() < deadline and pool.positions_pushed == 0:
            pool.check_producer_health()
            time.sleep(0.2)
    finally:
        pool.stop()

    assert pool.positions_pushed > 0, "no self-play positions reached the replay buffer"
    assert pool.self_play_positions_pushed == pool.positions_pushed
    assert pool.replay_buffer.size > 0
    pool.check_producer_health()


@pytest.mark.integration
def test_graph_pool_smoke_drains_without_producer_death() -> None:
    """The graph drain arm runs against the real runner without killing the feeder."""
    pool = _graph_pool()
    pool.start()
    try:
        deadline = time.monotonic() + _INTEGRATION_TIMEOUT_S / 4
        while time.monotonic() < deadline:
            pool.check_producer_health()
            time.sleep(0.2)
    finally:
        pool.stop()
    pool.check_producer_health()


def test_pool_threads_are_not_leaked_by_construction() -> None:
    """Constructing a pool starts no thread: the inference server is a `Thread` subclass."""
    before = threading.active_count()
    pool = _graph_pool()
    assert threading.active_count() == before, "construction must not start a thread"
    assert pool._stats_thread is None
