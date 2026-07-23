"""Replay-buffer feeding: the two push arms + the buffer-composition read.

The "buffer push" quarter of the pool split. Both arms are reached from
`pool_drain.run_stats_loop` and both write through `pool.replay_buffer`, which is the
`ReplayFacade` the pool wraps its raw engine buffer in — so a graph payload sent down the
dense arm (or the inverse) dies at the facade with a named error instead of becoming
corrupt training data.

The array work here is BEHAVIOUR, not plumbing: the f16 cast, the trunk-window reshape and
the u16 game-length clamp are the wire format the Rust replay buffer stores, and they are
ported unchanged. The facade itself performs no array operation at all.

Free functions taking the pool instance; `pool.py` imports this module and never the
reverse.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _draw_outcome_band(
    draw_value: float, ply_cap_value: float, eps: float = 0.05
) -> tuple[float, float]:
    """Outcome band capturing draw-like value targets for `draw_target_fraction`.

    Organic draws store `draw_value` and ply-cap truncations store `ply_cap_value`;
    decisive games store ±1. Span [min, max] of the two config values ± `eps` so both
    draw-like targets are counted and the ±1 wins excluded — instead of a stale hardcoded
    window that captured neither current value (so the metric read ~0 for every run).
    """
    lo = min(draw_value, ply_cap_value) - eps
    hi = max(draw_value, ply_cap_value) + eps
    return lo, hi


def push_graph(pool: Any, rows: list[tuple[Any, ...]]) -> None:
    """Push one drained batch of graph records, one row per position.

    Each record is forwarded VERBATIM and inspected by nothing on the way: the tuple order
    already matches the engine's positional signature. `game_id=-1` is the untagged
    sentinel — a whole-board graph position is one row with no intra-position correlation
    to dedupe, so the self-play write path never consumes a real game id and is
    resume-collision-free by construction.
    """
    for rec in rows:
        pool.replay_buffer.push_graph_position(*rec, game_id=-1)
    n = len(rows)
    with pool._lock:
        pool.positions_pushed += n
        pool.self_play_positions_pushed += n


def push_dense(pool: Any, collected: tuple[np.ndarray, ...]) -> None:
    """Push one drained dense batch: ONE bulk buffer call, then the recency mirror.

    `collected` is the runner's 10-tuple, in its own return order — features, chain
    planes, policies, values, plies, ownership, winning line, is-full-search, per-row ply
    index and the per-row value-supervision mask. The bulk push is a single FFI call
    instead of N per-row pushes, and the vectorised cast + reshape is much cheaper than
    the per-row buffer-assignment pattern it replaced.

    The recent buffer still takes a per-row push (its lock semantics are Python-side); it
    is off the supply critical path and is a separate lever from the bulk write.
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
    """Composition snapshot of the live replay buffer.

    Reads:
      - `corpus_fraction`: 1 − self_play_pushed / size (corpus == preload)
      - `draw_target_fraction`: outcomes inside the live draw/ply-cap band
        (`_draw_outcome_band`) over size
      - terminal-reason fractions over cumulative pushes since start

    The draw/ply-cap values are re-resolved from the LIVE config here rather than read off
    the frozen ctor-time hparams: this monitoring read is deliberately independent of the
    wire site that hands the same two values to the Rust runner.

    `draw_target_fraction` degrades to NaN when the bound buffer has no
    `outcome_in_range_count`. That is not a nicety — the graph buffer genuinely does not
    expose the getter, so the NaN is the true value on the graph path and the branch must
    stay reachable. Fabricating a number there would be an undeclared behaviour change.
    """
    size = max(1, int(pool.replay_buffer.size))
    sp_pushed = int(pool.self_play_positions_pushed)
    corpus_fraction = max(0.0, 1.0 - (sp_pushed / size))
    try:
        _tcfg = pool.config.get("training", pool.config)
        _draw = float(_tcfg.get("draw_value", -0.5))
        _ply = float(_tcfg.get("ply_cap_value", _tcfg.get("draw_value", -0.5)))
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


__all__ = ["buffer_composition", "push_dense", "push_graph"]
