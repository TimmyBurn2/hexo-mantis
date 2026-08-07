"""Suite H (lifecycle) — H-02 … H-11 — plus D-16, the pool's representation dispatch.

>300 justify: one lifecycle, one set of collaborator stubs. The producer-death contract
(H-02/H-04), the start/stop protocol (H-03), the four forwarders (H-05..H-08) and the
dispatch arms (D-16) all drive the SAME constructed pool with the SAME stub runner and stub
server; splitting them would duplicate both stubs and let the copies drift apart.

IMPL-written (non-⊕) per DESIGN §b.

The stubs replace the runner and inference server AFTER construction, so every assertion
still runs against a real `WorkerPool` built by the real constructor — the thing under test
is the pool's own wiring, not a re-implementation of it. H-09/H-10/H-11 are the integration
tier and use the real Rust runner end to end.

The load-bearing row is H-02. The feeder thread is the SOLE producer of training data: if
it dies and nothing notices, training continues happily on a buffer that stops growing, and
every metric except throughput looks healthy. `check_producer_health` exists so that
failure is loud on the next step, and this file is what proves the wiring is intact.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer, ReplayBuffer
from mantis.encoding import lookup
from mantis.model import CnnArch, GnnArch, build_net
from mantis.selfplay import pool as pool_mod
from mantis.selfplay.buffers import BufferKind
from mantis.selfplay.pool import WorkerPool

_INTEGRATION_TIMEOUT_S = 60.0


def _cfg(encoding: str, **over: Any) -> dict[str, Any]:
    # WPSC Phase 2 SC-A2 reshape: `selfplay`/`inference`/`train` are nested schema-shaped
    # sections now (no top-level `mcts`/flat-namespace fallback). `over` still layers onto
    # `selfplay` (its historical target — no call site in this file uses it today).
    selfplay: dict[str, Any] = {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
        "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
        "instrumentation_enabled": False,
        "mcts": {"n_simulations": 8, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 8, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    selfplay.update(over)
    inference = {
        "inference_batch_size": 4, "trace_inference": False, "inference_max_wait_ms": 10,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
    }
    # WPSC Phase 3 SC-B3: InferenceServer (via WorkerPool) now hard-reads
    # config["train"]["amp_dtype"] unconditionally (R30b, no fallback).
    train = {"draw_reward": -0.5, "ply_cap_value": -0.5, "amp_dtype": "fp16"}
    return {"encoding": encoding, "selfplay": selfplay, "inference": inference, "train": train}


def _grid_pool(encoding: str = "v6", **kw: Any) -> WorkerPool:
    spec = lookup(encoding)
    arch = CnnArch(board_size=spec.trunk_size, in_channels=spec.n_planes,
                   filters=8, res_blocks=1)
    return WorkerPool(
        build_net(arch), _cfg(encoding), torch.device("cpu"),
        ReplayBuffer(capacity=256, encoding=encoding), arch=arch, **kw,
    )


def _graph_pool(**kw: Any) -> WorkerPool:
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=16, num_layers=1)
    return WorkerPool(
        build_net(arch), _cfg("gnn_axis_v1"), torch.device("cpu"),
        HexgBuffer(capacity=256, encoding="gnn_axis_v1", visit_capacity=128), arch=arch, **kw,
    )


# ── collaborator stubs, installed after construction ────────────────────────────
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


# ═══ H-02 — the sole-producer contract ═══════════════════════════════════════════
def test_producer_death_is_re_raised_with_its_cause(monkeypatch) -> None:
    """H-02 — PASS iff a drain-loop exception leaves the pool flagged, and the next
    `check_producer_health()` raises `RuntimeError` with the original exception attached as
    its `__cause__`.

    The feeder is the only thing writing training data. If it dies quietly, training runs
    on a buffer that has stopped growing while loss, throughput per step, and every eval
    number stay plausible for hours. FAIL = exactly that silence. The cause must survive
    because 'the feeder died' without the traceback is not actionable."""
    pool = _grid_pool()
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
    """H-02 (no-false-abort arm) — PASS iff `check_producer_health()` is silent on a fresh
    pool AND after a clean `stop()`: a normal shutdown sets the stop event and the loop
    returns without an exception, so nothing is flagged.

    FAIL = every orderly shutdown aborts the run, which is how a fail-fast guard gets
    disabled by whoever is on call that night."""
    pool = _grid_pool()
    pool.check_producer_health()

    monkeypatch.setattr(pool_mod, "run_stats_loop", lambda _pool: None)
    pool._stats_loop()
    assert pool._producer_exc is None
    pool.check_producer_health()


def test_stats_loop_guard_does_not_let_the_thread_die_silently(monkeypatch, caplog) -> None:
    """H-04 — PASS iff the guard LOGS at error level and records the exception, rather than
    letting the daemon thread unwind unobserved.

    A daemon thread that raises prints to stderr at best and vanishes at worst; the log
    line plus the flag are the two independent traces that make the death discoverable
    from a run's own artefacts."""
    pool = _grid_pool()

    def _explode(_pool):
        raise RuntimeError("scripted")

    monkeypatch.setattr(pool_mod, "run_stats_loop", _explode)
    with caplog.at_level("ERROR"):
        pool._stats_loop()

    assert pool._producer_exc is not None
    assert any("selfplay_producer_died" in record.message for record in caplog.records), (
        "the death must be visible in the run's own log, not only in a flag"
    )


# ═══ H-03 — start / stop protocol ════════════════════════════════════════════════
def test_start_is_idempotent_while_running() -> None:
    """H-03 — PASS iff a second `start()` on a running pool is a no-op: no second runner
    start, no second server start, no second feeder thread.

    FAIL = two feeder threads draining the same Rust queue, which double-counts pushes and
    interleaves two `system_stats` cadences."""
    pool = _grid_pool()
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
    """H-03 (teardown arm) — PASS iff `stop()` sets the stop event, stops the runner and
    the server, joins the server with a bounded timeout, joins and clears the feeder
    thread, and stops the recorder.

    The bounded join matters: an unbounded one turns a wedged inference thread into a
    hung shutdown, and the run never writes its final checkpoint."""
    recorder = _StubRecorder()
    pool = _grid_pool(recorder=recorder)
    runner, server = _stub_collaborators(pool)

    pool.start()
    pool.stop()

    assert pool._stop_event.is_set()
    assert runner.stopped == 1 and server.stopped == 1
    assert server.joins == [5.0], "the server join must be bounded"
    assert pool._stats_thread is None, "the feeder handle must be cleared"
    assert recorder.stopped == 1


def test_stopped_feeder_thread_actually_exits() -> None:
    """H-03 (liveness arm) — PASS iff the feeder thread is no longer alive after `stop()`.
    Asserting only that `join` was called would pass on a loop that ignores its stop
    event."""
    pool = _grid_pool()
    _stub_collaborators(pool)

    pool.start()
    thread = pool._stats_thread
    assert thread is not None
    pool.stop()
    assert not thread.is_alive()


# ═══ H-06 … H-08 — the forwarders ════════════════════════════════════════════════
def test_sync_inference_weights_forwards_to_the_server() -> None:
    """H-06 — PASS iff a promoted state_dict reaches the server's safe swap, by identity.

    This is the promotion path's landing point: a pool that accepts the call and drops it
    keeps serving the OLD weights while every promotion log line says the new ones are
    live — the run then evaluates a model it is not actually playing."""
    pool = _grid_pool()
    _, server = _stub_collaborators(pool)
    state = {"layer.weight": torch.zeros(1)}
    pool.sync_inference_weights(state)
    assert len(server.state_dicts) == 1
    assert server.state_dicts[0] is state


def test_recorder_seam_forwards_and_defaults_to_inert() -> None:
    """H-07 — PASS iff an injected recorder receives `set_step` and answers
    `latest_replay_path`, and the DEFAULT recorder is inert (`None`, no error).

    The concrete recorder is a display-surface concern that does not exist in this tree, so
    the default has to be a working no-op rather than a missing attribute."""
    recorder = _StubRecorder(path="replays/games_0001.jsonl")
    pool = _grid_pool(recorder=recorder)
    pool.update_checkpoint_step(42)
    assert recorder.steps == [42]
    assert pool.latest_replay_path() == "replays/games_0001.jsonl"

    default_pool = _grid_pool()
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
    """H-08 — PASS iff batch occupancy reproduces the frozen arithmetic across the edge
    cases: zero forwards, exact fill, over-fill clamped at 100, and a zero batch size.

    The metric drives a throughput panel; an unclamped value above 100 or a
    ZeroDivisionError on the first read both make the panel useless at exactly the moment
    someone is looking at it."""
    pool = _grid_pool()
    pool._inference_server = _StubServer(forward_count=forward_count,
                                         total_requests=total_requests,
                                         batch_size=batch_size)
    assert pool.batch_fill_pct == pytest.approx(expected)


# ═══ D-16 — the pool's representation dispatch ═══════════════════════════════════
def test_grid_pool_takes_the_dense_arm() -> None:
    """D-16 (grid arm) — PASS iff a grid pool resolves to the dense branch, wraps a GRID
    facade, derives non-zero dense dims, and — when the drain runs — calls `collect_data`
    and never `collect_graph_data`."""
    pool = _grid_pool()
    runner, _ = _stub_collaborators(pool)

    assert pool._is_graph is False
    assert pool.replay_buffer.kind is BufferKind.GRID
    assert pool._feat_len > 0 and pool._chain_len > 0

    pool.start()
    time.sleep(0.25)
    pool.stop()

    assert "collect_data" in runner.calls
    assert "collect_graph_data" not in runner.calls, (
        "a grid pool must never reach the graph collect path"
    )


def test_graph_pool_takes_the_graph_arm() -> None:
    """D-16 (graph arm) — PASS iff a graph pool resolves to the graph branch, wraps a
    GRAPH facade, has degenerate (zero) dense dims, and calls `collect_graph_data` and
    never `collect_data`.

    The two arms write DIFFERENT storage formats. Dispatching on anything other than the
    resolved representation is how a graph run ends up writing dense rows."""
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


# ═══ H-09 … H-11 — the real thing (integration tier) ════════════════════════════
@pytest.mark.integration
def test_worker_pool_produces_positions_threaded_smoke() -> None:
    """H-09 — PASS iff a real pool with a real Rust runner and a tiny net actually
    produces positions and drains them into the buffer within the timeout, then shuts down
    cleanly with the producer still healthy.

    Every other row in this file stubs one collaborator or the other. This is the only one
    that proves the assembled thing runs: inference server, Rust workers, feeder thread and
    replay buffer, all live at once."""
    pool = _grid_pool()
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
@pytest.mark.parametrize("encoding,trunk,feat,chain,policy",
                         [("v6", 19, 2888, 2166, 362), ("v6w25", 25, 5000, 3750, 626)])
def test_encoding_aware_pool_wires_trunk_derived_dims(
        encoding, trunk, feat, chain, policy) -> None:
    """H-10 — PASS iff constructing a pool for each grid encoding wires the dims that
    encoding implies — trunk size, feature length, chain length, policy length — and the
    pool then runs against the real runner without a producer failure.

    These dims are what the drain reshapes every row with. Wiring the default encoding's
    numbers for a wider board reshapes correctly-sized data into the wrong geometry, which
    trains fine and learns nothing."""
    pool = _grid_pool(encoding)
    assert (pool._trunk_size, pool._feat_len, pool._chain_len, pool._pol_len) == (
        trunk, feat, chain, policy)
    assert pool.encoding_spec.name == encoding

    pool.start()
    try:
        time.sleep(0.5)
        pool.check_producer_health()
    finally:
        pool.stop()
    pool.check_producer_health()


@pytest.mark.integration
def test_pool_encoding_wired_no_warn(recwarn) -> None:
    """H-11 — PASS iff constructing and running a pool for a non-default encoding emits NO
    warning at the pool layer.

    A warning here would mean some component fell back to a default it was not given — the
    exact silent-substitution class this package's construction path exists to prevent.
    Runner internals stay gated by their own crate tests; this row is pool-level only."""
    pool = _grid_pool("v6w25")
    pool.start()
    try:
        time.sleep(0.3)
    finally:
        pool.stop()

    unexpected = [str(w.message) for w in recwarn.list]
    assert not unexpected, f"pool construction/run warned: {unexpected}"


@pytest.mark.integration
def test_graph_pool_smoke_drains_without_producer_death() -> None:
    """H-09 (graph arm) — PASS iff the graph pool runs the graph drain arm against the real
    runner without killing the feeder. The dense smoke cannot cover it: the two arms share
    no code below the branch, and the production identity is a graph encoding."""
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
    """H-03 (hygiene arm) — PASS iff merely CONSTRUCTING a pool starts no thread. The
    inference server is a `Thread` subclass, so an accidental `start()` in the constructor
    would leave a live thread behind every time a pool is built and discarded — including
    once per test in this file."""
    before = threading.active_count()
    pool = _grid_pool()
    assert threading.active_count() == before, "construction must not start a thread"
    assert pool._stats_thread is None
