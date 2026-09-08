"""R345(b)(6) — self-play rows carry a real game id, so same-game dedupe becomes live.

WHAT `-1` COST. `push_graph` pushed EVERY self-play row with `game_id=-1`, and its docstring
called that correct on the ground that *"a whole-board graph position is one row with no
intra-position correlation to dedupe"*. That answers a question LAW-04 does not ask. The
dedupe is over copies of a GAME — `sample_indices` skips its uniqueness guard entirely on
`-1` — so a sampled batch of 256 could be a dozen positions from one game reported as a dozen
independent samples, and the guard the ring carries for exactly this had never once fired on
real data. Nothing said so, because a guard that never fires and a guard with nothing to catch
produce identical output.

THE ID IS ALLOCATED BY THE BUFFER, NOT THE RUNNER, AND THAT IS THE WHOLE DESIGN. The runner
stamps a run-local sequence number; the buffer's `next_game_id()` is the allocator that cannot
collide with ids already loaded from a persisted ring (`load_from_path_impl` re-bases it past
the highest it read). Pushing the runner's number directly would collide with every resumed
run's history on its first game.
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
    """One drained graph row: `push_graph_position`'s nine, then the runner's game id."""
    return (
        [(0, 0, 1), (1, 0, -1), (0, 1, 1)],  # stones
        [(2, 0, 0.6), (1, 1, 0.4)],          # visits
        1, 30, ply, True, 1.0, True, 10,     # player, remaining, ply, full, outcome, valid, len
        game_id,
    )


def _buffer() -> tuple[Any, Any]:
    """`(facade, raw)` — the push goes through the FACADE, the reads through the raw handle.

    THE FACADE, not the raw handle — because the push arm sees the facade.

    `pool.replay_buffer` is a `ReplayFacade` (`selfplay/pool.py:145`) and that class has NO
    `__getattr__`: every forwarded member is written out by hand *"so the forwarded surface is
    greppable"*. A test that used a raw `HexgBuffer` would pass against a facade that does not
    forward `next_game_id` — which is exactly what happened on the first cut of this leg, and
    the production write path would have raised `AttributeError` on its first drained game.

    The READS use the raw handle deliberately. `game_id_at` is a diagnostic accessor no
    production caller uses, and forwarding it would widen a surface whose whole value is
    being minimal and greppable — the facade should carry what production calls, not what a
    test finds convenient.
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
    """A resumed ring already holds ids; the runner's own sequence would collide with them.

    `next_game_id` is re-based past the highest id a load read, so allocating through it is
    what makes a resumed run's first game distinct from the loaded history rather than a
    duplicate of it.
    """
    facade, buf = _buffer()
    for _ in range(50):
        facade.next_game_id()
    push_graph(_Pool(facade), [_row(0, 0)])
    assert buf.game_id_at(0) >= 50, (
        f"the pushed id is {buf.game_id_at(0)} — the runner's own low sequence number was "
        "forwarded verbatim, so it collides with every id a resumed ring already holds"
    )


def test_an_untagged_row_stays_untagged() -> None:
    """Mutation half AND a real case: a corpus preload row genuinely has no game.

    Inventing an id for it would make unrelated corpus positions look like one game to the
    dedupe, which thins a batch for no reason.
    """
    facade, buf = _buffer()
    push_graph(_Pool(facade), [_row(0, -1), _row(1, -1)])
    assert [buf.game_id_at(i) for i in range(buf.size)] == [-1, -1]


def test_the_push_counts_what_it_pushed() -> None:
    pool = _Pool(_buffer()[0])
    push_graph(pool, [_row(0, 1), _row(1, 1)])
    assert pool.positions_pushed == 2
    assert pool.self_play_positions_pushed == 2
