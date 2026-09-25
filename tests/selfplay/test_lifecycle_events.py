"""⊕ Step 3 narration — producer tests for the six lifecycle events.

Each lifecycle event (DESIGN §6) has a producer + a mutation test: kill the
producer (remove the emit) → its pin reds. These are PERMANENT instrumentation,
emitted in-run through the injected selfplay-local `EventSink`.

The six events (DESIGN §4.4):
  - runner_started        (pool.py:start())
  - workers_spawned        (pool.py:start())
  - game_loop_entered     (pool_drain.py:run_stats_loop)
  - first_inference_enqueued (inference_server.py, graph loop)
  - first_inference_served   (inference_server.py, graph retire)
  - first_record_drained   (pool_drain.py:run_stats_loop, first non-empty drain)

`game_complete` (part iii) is NOT tested here — it has its own goldens (C-03/J-05) and its
own delivery oracle (O-N2, test_game_complete_delivery.py).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import torch

import _fused_graph_harness as H
from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.selfplay import pool_drain
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.instrumentation import PoolInstrumentation


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


# runner_started + workers_spawned (pool.start)
def test_runner_started_emits_on_pool_start() -> None:
    """Producer test — `runner_started` emits once when `WorkerPool.start()` runs."""
    from mantis.selfplay.pool import WorkerPool
    from unittest.mock import MagicMock, patch

    sink = _RecordingSink()
    pool = WorkerPool.__new__(WorkerPool)
    pool._sink = sink
    pool._runner = MagicMock()
    pool._runner.is_running.return_value = False
    pool._stop_event = threading.Event()
    pool.model = MagicMock()
    pool._inference_server = MagicMock()
    pool.n_workers = 4
    pool.encoding_spec = MagicMock()
    pool.encoding_spec.name = "test_enc"
    pool._stats_thread = None

    with patch.object(WorkerPool, "_stats_loop", lambda self: None):
        pool.start()

    events = sink.named("runner_started")
    assert len(events) == 1, (
        f"runner_started must emit exactly once on start(); got {len(events)}"
    )
    assert events[0]["n_workers"] == 4
    assert events[0]["encoding"] == "test_enc"


def test_workers_spawned_emits_on_pool_start() -> None:
    """Producer test — `workers_spawned` emits once after `_runner.start()` returns."""
    from mantis.selfplay.pool import WorkerPool
    from unittest.mock import MagicMock, patch

    sink = _RecordingSink()
    pool = WorkerPool.__new__(WorkerPool)
    pool._sink = sink
    pool._runner = MagicMock()
    pool._runner.is_running.return_value = False
    pool._stop_event = threading.Event()
    pool.model = MagicMock()
    pool._inference_server = MagicMock()
    pool.n_workers = 4
    pool.encoding_spec = MagicMock()
    pool.encoding_spec.name = "test_enc"
    pool._stats_thread = None

    with patch.object(WorkerPool, "_stats_loop", lambda self: None):
        pool.start()

    events = sink.named("workers_spawned")
    assert len(events) == 1, (
        f"workers_spawned must emit exactly once after _runner.start(); got {len(events)}"
    )
    assert events[0]["n_workers"] == 4


# game_loop_entered + first_record_drained (pool_drain)
def _make_drain_pool(games, sink):
    pool = type("P", (), {})()
    pool._stop_event = type("S", (), {"_n": 0, "is_set": lambda self: self._n > 0 or (setattr(self, "_n", self._n + 1) or False)})()
    pool._runner = type("R", (), {
        "games_completed": len(games), "x_wins": 0, "o_wins": 0, "draws": 0,
        "positions_generated": 100,
        "collect_graph_data": lambda self: [],
        "drain_game_results": lambda self: list(games),
    })()
    pool.replay_buffer = type("B", (), {
        "size": 0, "capacity": 100_000,
        "push_graph_position": lambda self, *a, **k: None,
    })()
    pool._lock = threading.Lock()
    pool.positions_pushed = 0
    pool.self_play_positions_pushed = 0
    pool.graph_rows_pushed = 0
    pool.alpha_full_rows = 0
    pool.alpha_full_rows_emitted = 0
    pool._last_drain_time = 1000.0
    pool._last_pos_generated = 0
    pool._effective_sims_per_move = 50
    pool._total_sims = 0
    pool._sims_per_sec = 0.0
    pool._game_lengths = deque(maxlen=200)
    pool._avg_game_length = 0.0
    pool._sink = sink
    pool._heartbeat = None
    pool.games_completed = len(games)
    pool.x_wins = 0
    pool.o_wins = 0
    pool.draws = 0
    pool._instrumentation = PoolInstrumentation(log_investigation_metrics=False, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD)
    pool._recorder = type("NR", (), {
        "set_step": lambda self, s: None,
        "maybe_record": lambda self, **k: None,
        "stop": lambda self: None,
    })()
    return pool


def test_game_loop_entered_emits_once(monkeypatch) -> None:
    """Producer test — `game_loop_entered` emits once on drain-thread entry."""
    sink = _RecordingSink()
    pool = _make_drain_pool([(4, 1, [], 0, 0, 0, 0, 0, [], None)], sink)
    monkeypatch.setattr(pool_drain, "time", type("T", (), {
        "monotonic": lambda self: 1000.0, "sleep": lambda self, s: None,
    })())
    monkeypatch.setattr(pool_drain, "push_graph", lambda p, c: None)

    pool_drain.run_stats_loop(pool)

    assert len(sink.named("game_loop_entered")) == 1


def test_first_record_drained_emits_on_first_non_empty_drain(monkeypatch) -> None:
    """Producer test — `first_record_drained` emits once on the first NON-EMPTY drain
    (DESIGN §4.5 (β) — a record actually flowed)."""
    sink = _RecordingSink()
    pool = _make_drain_pool([(4, 1, [], 0, 0, 0, 0, 0, [], None)], sink)
    monkeypatch.setattr(pool_drain, "time", type("T", (), {
        "monotonic": lambda self: 1000.0, "sleep": lambda self, s: None,
    })())
    monkeypatch.setattr(pool_drain, "push_graph", lambda p, c: None)

    pool_drain.run_stats_loop(pool)

    events = sink.named("first_record_drained")
    assert len(events) == 1, (
        f"first_record_drained must emit once on the first non-empty drain; got {len(events)}"
    )
    assert events[0]["representation"] == "graph"


# first_inference_enqueued + first_inference_served (InferenceServer._run_graph_loop)
def _drive_graph_pops(monkeypatch, pops: list[list[int]]) -> tuple[_RecordingSink, Any]:
    """Run the REAL graph loop over scripted pops (legal counts per graph) with a sink injected."""
    import mantis.selfplay.graph_collate as collate_mod

    monkeypatch.setattr(collate_mod, "collate_graph_batch", H.collate_from_payload)
    sink = _RecordingSink()
    batcher = H.ScriptedGraphBatcher(
        [H.build_payload(counts, uid_base=1 + 1000 * i) for i, counts in enumerate(pops)]
    )
    server = InferenceServer(
        H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
        batcher=batcher, encoding_spec=H.GRAPH_SPEC, sink=sink,
    )
    batcher.server = server
    server.run()
    assert batcher.failures == [], batcher.failures
    assert len(batcher.results) == len(pops), "every scripted pop must be served"
    return sink, batcher


def test_first_inference_enqueued_emits_once(monkeypatch) -> None:
    """Producer test — the graph loop emits `first_inference_enqueued` on its first pop only."""
    sink, _ = _drive_graph_pops(monkeypatch, [[3, 4, 5], [2, 2]])

    assert sink.named("first_inference_enqueued") == [
        {"event": "first_inference_enqueued", "batch_size": 3, "representation": "graph"},
    ]


def test_first_inference_served_emits_once(monkeypatch) -> None:
    """Producer test — the retire stage emits `first_inference_served` on its first pop only."""
    sink, _ = _drive_graph_pops(monkeypatch, [[3, 4, 5], [2, 2]])

    assert sink.named("first_inference_served") == [
        {"event": "first_inference_served", "batch_size": 3, "representation": "graph"},
    ]
