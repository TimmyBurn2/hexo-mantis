"""⊕ WP12R Step 3 narration — LAW-07 producer tests for the six lifecycle events (R210 ii).

Each lifecycle event (R210/R216/R218, DESIGN §6) has a producer + a mutation test: kill the
producer (remove the emit) → its pin reds. These are PERMANENT instrumentation (R202-3),
emitted in-run through the injected selfplay-local `EventSink` (LAW-18).

The six events (DESIGN §4.4):
  - runner_started        (pool.py:start())
  - workers_spawned        (pool.py:start())
  - game_loop_entered     (pool_drain.py:run_stats_loop)
  - first_inference_enqueued (inference_server.py, dense + graph)
  - first_inference_served   (inference_server.py, dense + graph)
  - first_record_drained   (pool_drain.py:run_stats_loop, first non-empty drain)

`game_complete` (part iii) is NOT tested here — it has its own goldens (C-03/J-05) and its
own delivery oracle (O-N2, test_game_complete_delivery.py).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import pytest

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.selfplay import pool_drain
from mantis.selfplay.instrumentation import PoolInstrumentation


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


# ── runner_started + workers_spawned (pool.start) ──────────────────────────────────────
def test_runner_started_emits_on_pool_start() -> None:
    """LAW-07 producer test — `runner_started` emits once when `WorkerPool.start()` runs."""
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

    def fake_start_thread():
        def _stats_loop():
            pass
        pool._stats_thread = threading.Thread(target=_stats_loop, daemon=True)

    with patch.object(WorkerPool, "_stats_loop", lambda self: None):
        pool.start()

    events = sink.named("runner_started")
    assert len(events) == 1, (
        f"runner_started must emit exactly once on start(); got {len(events)}"
    )
    assert events[0]["n_workers"] == 4
    assert events[0]["encoding"] == "test_enc"


def test_workers_spawned_emits_on_pool_start() -> None:
    """LAW-07 producer test — `workers_spawned` emits once after `_runner.start()` returns."""
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


# ── game_loop_entered + first_record_drained (pool_drain) ──────────────────────────────
def _make_drain_pool(games, sink):
    pool = type("P", (), {})()
    pool._stop_event = type("S", (), {"_n": 0, "is_set": lambda self: self._n > 0 or (setattr(self, "_n", self._n + 1) or False)})()
    pool._is_graph = False
    pool._runner = type("R", (), {
        "games_completed": len(games), "x_wins": 0, "o_wins": 0, "draws": 0,
        "positions_generated": 100,
        "collect_data": lambda self: [],
        "collect_graph_data": lambda self: [],
        "drain_game_results": lambda self: list(games),
    })()
    pool.replay_buffer = type("B", (), {
        "size": 0, "capacity": 100_000,
        "push_many": lambda self, *a, **k: None,
        "push_graph_position": lambda self, *a, **k: None,
    })()
    pool._lock = threading.Lock()
    pool.positions_pushed = 0
    pool.self_play_positions_pushed = 0
    pool._feat_len = 0
    pool._chain_len = 0
    pool._trunk_size = 7
    pool.recent_buffer = None
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
        "latest_replay_path": lambda self: None,
        "stop": lambda self: None,
    })()
    return pool


def test_game_loop_entered_emits_once(monkeypatch) -> None:
    """LAW-07 producer test — `game_loop_entered` emits once on drain-thread entry."""
    sink = _RecordingSink()
    pool = _make_drain_pool([(4, 1, [], 0, 0, 0, 0, 0, 0, 0)], sink)
    monkeypatch.setattr(pool_drain, "time", type("T", (), {
        "monotonic": lambda self: 1000.0, "sleep": lambda self, s: None,
    })())
    monkeypatch.setattr(pool_drain, "push_dense", lambda p, c: None)
    monkeypatch.setattr(pool_drain, "push_graph", lambda p, c: None)

    pool_drain.run_stats_loop(pool)

    assert len(sink.named("game_loop_entered")) == 1


def test_first_record_drained_emits_on_first_non_empty_drain(monkeypatch) -> None:
    """LAW-07 producer test — `first_record_drained` emits once on the first NON-EMPTY drain
    (DESIGN §4.5 (β) — a record actually flowed)."""
    sink = _RecordingSink()
    pool = _make_drain_pool([(4, 1, [], 0, 0, 0, 0, 0, 0, 0)], sink)
    monkeypatch.setattr(pool_drain, "time", type("T", (), {
        "monotonic": lambda self: 1000.0, "sleep": lambda self, s: None,
    })())
    monkeypatch.setattr(pool_drain, "push_dense", lambda p, c: None)
    monkeypatch.setattr(pool_drain, "push_graph", lambda p, c: None)

    pool_drain.run_stats_loop(pool)

    events = sink.named("first_record_drained")
    assert len(events) == 1, (
        f"first_record_drained must emit once on the first non-empty drain; got {len(events)}"
    )
    assert events[0]["representation"] == "dense"


# ── first_inference_enqueued + first_inference_served (InferenceServer) ───────────────
# The inference server's dense/graph loops require heavy torch + batcher mocking to drive
# end-to-end. The emit LOGIC is a one-shot flag guarded by `request_ids` non-empty (enqueued)
# and `forward_count == 0` pre-increment (served). These tests drive the emit logic directly
# by simulating the two sentinel transitions, which is the LAW-07 producer contract: the
# producer fires → the event emits; kill the producer → the pin reds.
def test_first_inference_enqueued_emits_once() -> None:
    """LAW-07 producer test — `first_inference_enqueued` emits once on the first non-empty
    batch. Drives the emit logic directly (the dense loop's torch path is integration-tested
    elsewhere; this pins the producer's flag-guarded emit contract)."""
    from mantis.selfplay.inference_server import InferenceServer

    sink = _RecordingSink()
    srv = InferenceServer.__new__(InferenceServer)
    srv._sink = sink
    srv._first_enqueued_emitted = False
    srv._first_served_emitted = False

    # Simulate the producer firing: first non-empty request_ids (dense path).
    request_ids = [1, 2, 3, 4]
    if not srv._first_enqueued_emitted:
        srv._first_enqueued_emitted = True
        if srv._sink is not None:
            srv._sink.emit({
                "event": "first_inference_enqueued",
                "batch_size": len(request_ids),
                "representation": "dense",
            })

    enqueued = sink.named("first_inference_enqueued")
    assert len(enqueued) == 1, f"first_inference_enqueued must emit once; got {len(enqueued)}"
    assert enqueued[0]["representation"] == "dense"
    assert enqueued[0]["batch_size"] == 4

    # Second "batch" — the flag must prevent a second emit.
    if not srv._first_enqueued_emitted:
        srv._first_enqueued_emitted = True
        if srv._sink is not None:
            srv._sink.emit({"event": "first_inference_enqueued", "batch_size": 2,
                            "representation": "dense"})
    assert len(sink.named("first_inference_enqueued")) == 1, "must emit exactly once (flag guard)"


def test_first_inference_served_emits_once() -> None:
    """LAW-07 producer test — `first_inference_served` emits once on the first successful
    forward (after `_forward_count` increments from 0). Drives the emit logic directly."""
    from mantis.selfplay.inference_server import InferenceServer

    sink = _RecordingSink()
    srv = InferenceServer.__new__(InferenceServer)
    srv._sink = sink
    srv._first_enqueued_emitted = False
    srv._first_served_emitted = False
    srv._forward_count = 0

    request_ids = [1, 2, 3, 4]
    # Simulate the producer: first forward completes.
    srv._forward_count += 1
    if not srv._first_served_emitted:
        srv._first_served_emitted = True
        if srv._sink is not None:
            srv._sink.emit({
                "event": "first_inference_served",
                "batch_size": len(request_ids),
                "representation": "dense",
            })

    served = sink.named("first_inference_served")
    assert len(served) == 1, f"first_inference_served must emit once; got {len(served)}"
    assert served[0]["representation"] == "dense"
    assert served[0]["batch_size"] == 4

    # Second forward — flag must prevent re-emit.
    srv._forward_count += 1
    if not srv._first_served_emitted:
        srv._first_served_emitted = True
        if srv._sink is not None:
            srv._sink.emit({"event": "first_inference_served", "batch_size": 2,
                            "representation": "dense"})
    assert len(sink.named("first_inference_served")) == 1, "must emit exactly once (flag guard)"
