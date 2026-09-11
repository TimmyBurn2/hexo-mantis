"""Feed the replay buffer: the two push arms and the buffer-composition read.

Free functions taking the pool instance; `pool.py` imports this module and never the reverse.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _draw_outcome_band(
    draw_value: float, ply_cap_value: float, eps: float = 0.05
) -> tuple[float, float]:
    """Return the outcome band capturing draw-like value targets, spanning both configured
    draw-like values so decisive games' ±1 are excluded."""
    lo = min(draw_value, ply_cap_value) - eps
    hi = max(draw_value, ply_cap_value) + eps
    return lo, hi


#: A sparse Gumbel row whose explicit entries carry NO target mass (R349(c)): alpha within one
#: f32 ULP of unity. Counted per row pushed and published on `iteration_complete`.
ALPHA_FULL_THRESHOLD = 1.0 - 1e-6
#: The `alpha_full_row` events a run publishes at most: enough rows to reconstruct, bounded so a
#: pathological regime cannot turn the event stream into a second ring.
ALPHA_FULL_ROW_EVENT_CAP = 256


def _alpha_full_row_event(rec: tuple[Any, ...], tail_mass: float,
                          runner_game_id: int) -> dict[str, Any]:
    """The row, as the reconstruction needs it: the position, whose stone it was, and the
    explicit cells the target left empty. Field order is `GraphRecord`'s drain tuple."""
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
        # `(…nine positional fields…, tail_mass, runner_game_id)`: the tail mass rides by keyword
        # because the push signature carries `game_id` before it.
        runner_game_id = int(rec[-1])
        tail_mass = float(rec[-2])
        if tail_mass >= ALPHA_FULL_THRESHOLD:
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
        pool.replay_buffer.push_graph_position(*rec[:-2], game_id=buffer_game_id,
                                               tail_mass=tail_mass)
    n = len(rows)
    with pool._lock:
        pool.positions_pushed += n
        pool.self_play_positions_pushed += n
        pool.graph_rows_pushed += n
        pool.alpha_full_rows += alpha_full


def push_dense(pool: Any, collected: tuple[np.ndarray, ...]) -> None:
    """Push one drained dense batch through one bulk buffer call, then the recency mirror.

    `collected` is the runner's 10-tuple in its own return order. The recent buffer still takes a
    per-row push because its lock semantics are Python-side, off the supply critical path.
    """
    _in_ch = pool._feat_len // (pool._trunk_size * pool._trunk_size)
    (
        feats_np, chain_np, pols_np, vals_np, plies_np,
        own_np, wl_np, ifs_np, pidx_np, vv_np,
    ) = collected
    n = len(vals_np)
    if n > 0:
        feats_f16 = feats_np.astype(np.float16).reshape(
            n, _in_ch, pool._trunk_size, pool._trunk_size,
        )
        chain_f16 = chain_np.astype(np.float16).reshape(
            n, 6, pool._trunk_size, pool._trunk_size,
        )
        # Per-row compound-move count; clamp into u16 range.
        game_lengths = np.minimum(
            (plies_np.astype(np.int64) + 1) // 2, 65535,
        ).astype(np.uint16)
        pool.replay_buffer.push_dense_many(
            feats_f16, chain_f16, pols_np, vals_np, own_np, wl_np,
            game_lengths, ifs_np, pidx_np,   # per-row 0-based ply index
            value_target_valid=vv_np,        # per-row value-supervision mask
        )

        if pool.recent_buffer is not None:
            for i in range(n):
                pool.recent_buffer.push(
                    feats_f16[i],
                    chain_planes=chain_f16[i],
                    policy=pols_np[i],
                    outcome=float(vals_np[i]),
                    ownership=own_np[i],
                    winning_line=wl_np[i],
                    is_full_search=bool(ifs_np[i]),
                    value_target_valid=bool(vv_np[i]),
                )

        with pool._lock:
            pool.positions_pushed += n
            pool.self_play_positions_pushed += n


def buffer_composition(pool: Any) -> dict[str, float]:
    """Return a composition snapshot of the live replay buffer.

    The draw/ply-cap values are re-resolved from the LIVE config with no fallback default, so this
    read adds no second authority. `draw_target_fraction` is NaN when the bound buffer has no
    `outcome_in_range_count`: the graph buffer does not expose it, so NaN is the true value there.
    """
    size = max(1, int(pool.replay_buffer.size))
    sp_pushed = int(pool.self_play_positions_pushed)
    corpus_fraction = max(0.0, 1.0 - (sp_pushed / size))
    try:
        _train = pool.config["train"]
        _draw = float(_train["draw_reward"])
        _ply = float(_train["ply_cap_value"])
        _lo, _hi = _draw_outcome_band(_draw, _ply)
        draws_in_buf = int(pool.replay_buffer.outcome_in_range_count(_lo, _hi))
        draw_target_fraction = draws_in_buf / size
    except (AttributeError, TypeError):
        draw_target_fraction = float("nan")
    tr = pool.terminal_reason_counts()
    total_games = max(1, sum(tr.values()))
    return {
        "buffer_size": int(pool.replay_buffer.size),
        "buffer_capacity": int(pool.replay_buffer.capacity),
        "corpus_fraction":      round(corpus_fraction, 6),
        "draw_target_fraction": (
            round(draw_target_fraction, 6)
            if draw_target_fraction == draw_target_fraction
            else float("nan")
        ),
        "six_terminal_fraction":    tr["six_in_a_row"] / total_games,
        "colony_terminal_fraction": tr["colony"]       / total_games,
        "cap_terminal_fraction":    tr["ply_cap"]      / total_games,
        "other_draw_fraction":      tr["other_draw"]   / total_games,
        "n_games_observed": sum(tr.values()),
    }


__all__ = ["ALPHA_FULL_ROW_EVENT_CAP", "ALPHA_FULL_THRESHOLD", "buffer_composition",
           "push_dense", "push_graph"]
