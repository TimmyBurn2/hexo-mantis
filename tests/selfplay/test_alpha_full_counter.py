"""R349(c): the alpha = 1.0 row count is a LAW-18 reading with a live producer.

A sparse Gumbel row whose explicit entries carry no target mass to f32 resolution ships them
with masses below the tail's ULP and a tail that rounds to 1.0 (a row with NO explicit cell is
refused at the ring as `EmptyTarget`, so the ring never holds one). The push arm counts those
rows against every graph row pushed, publishes the pair on `iteration_complete` as a per-1,000
rate, and emits the first rows whole so they can be reconstructed. Each half is pinned by the
mutation that would silence it.
"""
from __future__ import annotations

import threading
from typing import Any

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.selfplay.buffers import ReplayFacade
from mantis.selfplay.pool_push import ALPHA_FULL_ROW_EVENT_CAP, push_graph

_ENC = "gnn_axis_v1"


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


class _Pool:
    def __init__(self, sink: _Sink | None) -> None:
        self.replay_buffer = ReplayFacade(lookup(_ENC), HexgBuffer(512, _ENC, 128))
        self._lock = threading.Lock()
        self._sink = sink
        self.positions_pushed = 0
        self.self_play_positions_pushed = 0
        self.graph_rows_pushed = 0
        self.alpha_full_rows = 0
        self.alpha_full_rows_emitted = 0
        self.config: dict[str, Any] = {}


def _row(*, tail: float, ply: int = 3, game_id: int = 1) -> tuple[Any, ...]:
    """A full-vector row at `tail == 0.0`; at `tail == 1.0` the explicit cells carry the
    sub-ULP masses the recorder ships for such a row, so the ring's unity guard admits it."""
    visits = ([(2, 0, 1e-9), (1, 1, 1e-9)] if tail >= 1.0
              else [(2, 0, 0.6 * (1 - tail)), (1, 1, 0.4 * (1 - tail))])
    return ([(0, 0, 1), (1, 0, -1), (0, 1, 1)], visits, 1, 2, ply, True, 1.0, True, 10,
            tail, game_id)


def test_the_counter_moves_only_on_alpha_full_rows() -> None:
    """Killer: `if tail_mass >= ALPHA_FULL_THRESHOLD` deleted, or the threshold widened."""
    pool = _Pool(sink=None)
    push_graph(pool, [_row(tail=0.0), _row(tail=0.5), _row(tail=1.0), _row(tail=0.999)])
    assert pool.graph_rows_pushed == 4
    assert pool.alpha_full_rows == 1, (
        f"one of four rows carries alpha = 1.0; the counter read {pool.alpha_full_rows}"
    )


def test_an_alpha_full_row_is_emitted_whole_and_the_stream_is_capped() -> None:
    """Killer: the event dropped, or the cap removed so a pathological regime floods the stream."""
    sink = _Sink()
    pool = _Pool(sink)
    push_graph(pool, [_row(tail=1.0, ply=7, game_id=3), _row(tail=0.0)])
    rows = [e for e in sink.events if e["event"] == "alpha_full_row"]
    assert len(rows) == 1
    assert rows[0]["ply_index"] == 7 and rows[0]["moves_remaining"] == 2
    assert rows[0]["stones"] == [[0, 0, 1], [1, 0, -1], [0, 1, 1]]
    assert [c[:2] for c in rows[0]["explicit_cells"]] == [[2, 0], [1, 1]], (
        "the sampled cells travel with the row: the reconstruction needs to know WHICH "
        "candidates the target left empty"
    )
    push_graph(pool, [_row(tail=1.0, game_id=4 + i) for i in range(ALPHA_FULL_ROW_EVENT_CAP + 8)])
    emitted = sum(1 for e in sink.events if e["event"] == "alpha_full_row")
    assert emitted == ALPHA_FULL_ROW_EVENT_CAP, (
        f"the stream carried {emitted} rows against a cap of {ALPHA_FULL_ROW_EVENT_CAP}"
    )
    assert pool.alpha_full_rows == ALPHA_FULL_ROW_EVENT_CAP + 9, "the COUNT is not capped"


def test_the_pool_publishes_the_pair_as_a_rate() -> None:
    """The read `iteration_complete` carries: `None` before any graph row, a rate after."""
    from mantis.selfplay.pool import WorkerPool

    reading = WorkerPool.alpha_full.fget  # type: ignore[attr-defined]
    pool = _Pool(sink=None)
    assert reading(pool) == {"rows": 0, "graph_rows": 0, "per_1000": None}
    push_graph(pool, [_row(tail=1.0)] + [_row(tail=0.0, game_id=2)] * 3)
    assert reading(pool) == {"rows": 1, "graph_rows": 4, "per_1000": 250.0}
