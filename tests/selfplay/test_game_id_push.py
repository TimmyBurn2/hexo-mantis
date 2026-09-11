"""Self-play rows carry a real game id, so same-game dedupe becomes live.

`sample_indices` skips its uniqueness guard entirely on `game_id=-1`, which every self-play row
once carried: a sampled batch could be a dozen positions of one game counted as a dozen
independent samples, and the ring's guard had never fired on real data.

THE ID IS ALLOCATED BY THE BUFFER, NOT THE RUNNER: `next_game_id()` is re-based past the highest
id a persisted ring was loaded with, so the runner's own run-local sequence number would collide
with every resumed run's history on its first game.
"""
from __future__ import annotations

from typing import Any

from mantis._engine import HexgBuffer
from mantis.selfplay.pool_push import push_graph

_ENC = "gnn_axis_v1"


class _Pool:
    """The push arm on a pool double — the free function is what is under test."""

    def __init__(self, buffer: Any) -> None:
        import threading

        self.replay_buffer = buffer
        self._lock = threading.Lock()
        self.positions_pushed = 0
        self.self_play_positions_pushed = 0
        self.config: dict[str, Any] = {}


def _row(ply: int, game_id: int) -> tuple[Any, ...]:
    """One drained graph row: `push_graph_position`'s nine, the tail mass, then the game id."""
    return (
        [(0, 0, 1), (1, 0, -1), (0, 1, 1)],  # stones
        [(2, 0, 0.6), (1, 1, 0.4)],          # visits
        1, 30, ply, True, 1.0, True, 10,     # player, remaining, ply, full, outcome, valid, len
        0.0,                                 # tail mass (a full-vector row has none)
        game_id,
    )


def _buffer() -> tuple[Any, Any]:
    """`(facade, raw)` — the push goes through the FACADE, the reads through the raw handle.

    `ReplayFacade` has no `__getattr__` and forwards each member by hand, so a test pushing into
    a raw `HexgBuffer` would pass against a facade that never forwards `next_game_id`. The reads
    use the raw handle because `game_id_at` is a diagnostic no production caller uses, and the
    facade carries what production calls rather than what a test finds convenient.
    """
    from mantis.encoding import lookup
    from mantis.selfplay.buffers import ReplayFacade

    raw = HexgBuffer(64, _ENC, 128)
    return ReplayFacade(lookup(_ENC), raw), raw


def test_rows_from_one_game_share_one_buffer_id() -> None:
    facade, buf = _buffer()
    push_graph(_Pool(facade), [_row(0, 7), _row(1, 7), _row(2, 7)])
    ids = {buf.game_id_at(i) for i in range(buf.size)}
    assert len(ids) == 1, f"three rows of one game landed under {len(ids)} ids"
    assert -1 not in ids, "the untagged sentinel survived — the dedupe is still inert"


def test_rows_from_different_games_get_different_buffer_ids() -> None:
    facade, buf = _buffer()
    push_graph(_Pool(facade), [_row(0, 7), _row(0, 8), _row(1, 7), _row(0, 9)])
    ids = [buf.game_id_at(i) for i in range(buf.size)]
    assert len(set(ids)) == 3, f"three games collapsed to {len(set(ids))} ids: {ids}"
    assert ids[0] == ids[2], "two rows of game 7 were given different ids"


def test_the_id_comes_from_the_buffers_allocator_not_the_runner() -> None:
    """A resumed ring already holds ids, and `next_game_id` is re-based past the highest one read."""
    facade, buf = _buffer()
    for _ in range(50):
        facade.next_game_id()
    push_graph(_Pool(facade), [_row(0, 0)])
    assert buf.game_id_at(0) >= 50, (
        f"the pushed id is {buf.game_id_at(0)} — the runner's own low sequence number was "
        "forwarded verbatim, so it collides with every id a resumed ring already holds"
    )


def test_an_untagged_row_stays_untagged() -> None:
    """A corpus preload row genuinely has no game; inventing an id would thin a batch for no reason."""
    facade, buf = _buffer()
    push_graph(_Pool(facade), [_row(0, -1), _row(1, -1)])
    assert [buf.game_id_at(i) for i in range(buf.size)] == [-1, -1]


def test_the_push_counts_what_it_pushed() -> None:
    pool = _Pool(_buffer()[0])
    push_graph(pool, [_row(0, 1), _row(1, 1)])
    assert pool.positions_pushed == 2
    assert pool.self_play_positions_pushed == 2
