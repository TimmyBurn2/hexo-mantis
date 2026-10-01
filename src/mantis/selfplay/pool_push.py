"""Feed the replay buffer: the graph push and the buffer-composition read.

Free functions taking the pool instance; `pool.py` imports this module and never the reverse.
"""
from __future__ import annotations

from typing import Any

from mantis.util.constants import is_alpha_full

#: At most this many `alpha_full_row` events per run: enough to reconstruct, never a second ring.
ALPHA_FULL_ROW_EVENT_CAP = 256


def _alpha_full_row_event(rec: tuple[Any, ...], tail_mass: float,
                          runner_game_id: int) -> dict[str, Any]:
    """The row as the reconstruction needs it; field order is `GraphRecord`'s drain tuple."""
    stones, visits, current_player, moves_remaining, ply_index, is_full_search = rec[:6]
    return {
        "event": "alpha_full_row",
        "tail_mass": tail_mass,
        "stones": [[int(q), int(r), int(c)] for q, r, c in stones],
        "explicit_cells": [[int(q), int(r), float(p)] for q, r, p in visits],
        "current_player": int(current_player),
        "moves_remaining": int(moves_remaining),
        "ply_index": int(ply_index),
        "is_full_search": bool(is_full_search),
        "runner_game_id": int(runner_game_id),
    }


def push_graph(pool: Any, rows: list[tuple[Any, ...]]) -> None:
    """Push one drained batch of graph records, one row per position, with its game id.

    The game id is TRANSLATED, not forwarded: the runner's sequence restarts at 0 every launch
    while a resumed ring already holds ids from before the stop, so it is allocated through the
    buffer's own `next_game_id()`. A sentinel id would skip the sampler's uniqueness guard
    entirely and let one game's positions count as independent samples.
    """
    allocated: dict[int, int] = {}
    alpha_full = 0
    for rec in rows:
        # `(…positional fields…, tail_mass, (root_value, root_value_valid), runner_game_id)`: the tail and the
        # root pair ride by keyword because the push signature carries `game_id` before them.
        tail_mass, (root_value, root_value_valid), runner_game_id = rec[-3:]
        tail_mass, runner_game_id = float(tail_mass), int(runner_game_id)
        if is_alpha_full(tail_mass):
            alpha_full += 1
            sink = getattr(pool, "_sink", None)
            if sink is not None and pool.alpha_full_rows_emitted < ALPHA_FULL_ROW_EVENT_CAP:
                pool.alpha_full_rows_emitted += 1
                sink.emit(_alpha_full_row_event(rec, tail_mass, runner_game_id))
        if runner_game_id < 0:
            # A genuinely untagged row: inventing an id would make unrelated positions look like
            # one game and thin a batch for no reason.
            buffer_game_id = -1
        else:
            buffer_game_id = allocated.get(runner_game_id, -1)
            if buffer_game_id < 0:
                buffer_game_id = int(pool.replay_buffer.next_game_id())
                allocated[runner_game_id] = buffer_game_id
        pool.replay_buffer.push_graph_position(*rec[:-3], game_id=buffer_game_id, tail_mass=tail_mass,
                                               root_value=root_value, root_value_valid=root_value_valid)
    n = len(rows)
    with pool._lock:
        pool.positions_pushed += n
        pool.self_play_positions_pushed += n
        pool.graph_rows_pushed += n
        pool.alpha_full_rows += alpha_full


def buffer_composition(pool: Any) -> dict[str, float]:
    """Return a composition snapshot of the live replay buffer and the games' terminal reasons."""
    size = max(1, int(pool.replay_buffer.size))
    sp_pushed = int(pool.self_play_positions_pushed)
    corpus_fraction = max(0.0, 1.0 - (sp_pushed / size))
    tr = pool.terminal_reason_counts()
    total_games = max(1, sum(tr.values()))
    return {
        "buffer_size": int(pool.replay_buffer.size),
        "buffer_capacity": int(pool.replay_buffer.capacity),
        "corpus_fraction":      round(corpus_fraction, 6),
        "six_terminal_fraction":    tr["six_in_a_row"] / total_games,
        "colony_terminal_fraction": tr["colony"]       / total_games,
        "cap_terminal_fraction":    tr["ply_cap"]      / total_games,
        "other_draw_fraction":      tr["other_draw"]   / total_games,
        "n_games_observed": sum(tr.values()),
    }


__all__ = ["ALPHA_FULL_ROW_EVENT_CAP", "buffer_composition", "push_graph"]
