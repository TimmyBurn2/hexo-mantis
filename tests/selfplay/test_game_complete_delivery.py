"""`game_complete` is DELIVERED to the sink in production, not merely emitted.

The event was already emitted by the drain loop and golden-pinned, but the production
`WorkerPool` was constructed with `sink=None`, so every one of them was dropped.

Two arms. SOURCE: the `WorkerPool(...)` call in `run.py` does not pass `sink=None` — an AST
check, because that keyword is the exact defect. RUNTIME: a pool built with a non-None sink
delivers N `game_complete` events for N drained games, which is the delivery-count witness
beside the goldens that pin only the payload shape.

FALSIFYING MUTATION: revert the construction site to `sink=None`; both arms must red.
"""
from __future__ import annotations

import ast
import threading
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.selfplay import pool_drain
from mantis.selfplay.instrumentation import PoolInstrumentation

_RUN_PY = Path(__file__).resolve().parents[2] / "src" / "mantis" / "run.py"


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


class _OneShotStop:
    def __init__(self) -> None:
        self._n = 0

    def is_set(self) -> bool:
        if self._n == 0:
            self._n += 1
            return False
        return True


class _ScriptedTime:
    def __init__(self, sequence) -> None:
        self.sequence = list(sequence)
        self.i = 0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        value = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _ScriptedRunner:
    def __init__(self, games, positions_generated=100) -> None:
        self._games = games
        self.games_completed = len(games)
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        self.positions_generated = positions_generated

    def collect_data(self):
        return []

    def collect_graph_data(self):
        return []

    def drain_game_results(self):
        return list(self._games)


class _ScriptedBuffer:
    def __init__(self) -> None:
        self.size = 0
        self.capacity = 100_000

    def push_many(self, *args, **kwargs):
        return None

    def push_graph_position(self, *args, **kwargs):
        return None


def _make_scripted_pool(games, sink):
    """Minimal pool surface for `run_stats_loop` — just enough to drain N games and emit N
    `game_complete` events."""
    pool = type("ScriptedPool", (), {})()
    pool._stop_event = _OneShotStop()
    pool._is_graph = False
    pool._runner = _ScriptedRunner(games)
    pool.replay_buffer = _ScriptedBuffer()
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
    pool._recorder = type("NullRec", (), {
        "set_step": lambda self, step: None,
        "maybe_record": lambda self, **kw: None,
        "latest_replay_path": lambda self: None,
        "stop": lambda self: None,
    })()
    return pool


def _make_games(n: int) -> list[tuple]:
    """N scripted game-result 10-tuples (the `drain_game_results` shape): winner 1, 4 plies,
    empty move history, no solver fires."""
    return [(4, 1, [], 0, 0, 0, 0, 0, 0, 0) for _ in range(n)]


def test_on2a_production_pool_construction_does_not_pass_sink_none() -> None:
    """SOURCE arm: the `WorkerPool(...)` construction call in `run.py` does NOT pass
    `sink=None`, which is what dropped every `game_complete` and `system_stats` event the
    drain loop emits. Only `sink=` is inspected; the `heartbeat=` keyword is another oracle's
    subject."""
    source = _RUN_PY.read_text()
    tree = ast.parse(source)

    worker_pool_calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WorkerPool"):
            worker_pool_calls.append(node)

    assert worker_pool_calls, (
        "O-N2 (a): no `WorkerPool(...)` construction call found in run.py — the injection "
        "site the oracle inspects has moved or been renamed."
    )
    # There is exactly one production construction call in run.py.
    assert len(worker_pool_calls) == 1, (
        f"O-N2 (a): expected exactly one `WorkerPool(...)` call in run.py; found "
        f"{len(worker_pool_calls)}. The oracle inspects the ONE production construction "
        f"site; if a second call was added, this test needs updating."
    )

    call = worker_pool_calls[0]
    sink_kwarg = None
    for kw in call.keywords:
        if kw.arg == "sink":
            sink_kwarg = kw
            break

    assert sink_kwarg is not None, (
        "O-N2 (a): the `WorkerPool(...)` call in run.py has no `sink=` keyword at all. "
        "The production pool must inject a sink (R216: inject a sink at run.py:349)."
    )

    # At the defect it is `ast.Constant(None)`; once injected it is the adapter's Call/Name.
    is_none = (isinstance(sink_kwarg.value, ast.Constant)
               and sink_kwarg.value.value is None)
    assert not is_none, (
        "O-N2 (a): the production `WorkerPool(...)` call in run.py passes `sink=None` — "
        "the EXACT defect R215/R216 name. game_complete IS emitted (pool_drain.py:177, "
        "golden-pinned) but DROPPED because pool._sink is None. IMPL must inject a sink "
        "(the _DeferredSink adapter per DESIGN §4.2). FALSIFYING MUTATION: revert to "
        "`sink=None` and this assertion turns RED (the `not is_none` fails)."
    )


def test_on2b_n_games_yield_n_delivered_game_complete_events(monkeypatch) -> None:
    """RUNTIME arm: a pool built with a non-None sink delivers N `game_complete` events for N
    drained games. Green already — delivery was never the bug, only the production injection —
    so a red here means the injection broke the delivery contract rather than the oracle."""
    n = 7
    sink = _RecordingSink()
    pool = _make_scripted_pool(_make_games(n), sink)

    monkeypatch.setattr(pool_drain, "time", _ScriptedTime([1000.0, 1002.0, 1006.5]))
    # The test is about EVENT DELIVERY, not buffer pushes, so the push arms are no-ops.
    monkeypatch.setattr(pool_drain, "push_dense", lambda pool, collected: None)
    monkeypatch.setattr(pool_drain, "push_graph", lambda pool, collected: None)

    pool_drain.run_stats_loop(pool)

    delivered = sink.named("game_complete")
    assert len(delivered) == n, (
        f"O-N2 (b): {n} drained games must yield {n} delivered `game_complete` events at "
        f"the sink. Got {len(delivered)}. If 0, the sink was None (the production defect); "
        f"if <{n}, events were dropped mid-drain. C-03/J-05 pin the payload shape; this "
        f"pins the delivery count."
    )
