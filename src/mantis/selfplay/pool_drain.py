"""The feeder-daemon loop body: drain Rust results, push them, emit the per-game events.

The "result drain" quarter of the pool split, expressed as a free function over the pool
instance (`pool.py` imports this module, never the reverse — the split is acyclic).

This loop is the SOLE producer of training data: it is the only thing that moves rows out
of the Rust runner into the replay buffer. Everything it does is drain-cadence work at
~10 Hz, not per-request hot-path work:

  1. one push arm (dense or graph, chosen once at construction from the encoding spec);
  2. mirror the runner's game counters onto the pool;
  3. bill sims/sec per MOVE — `delta(positions_generated) * effective_sims_per_move`.
     Billing per GAME instead undercounts by roughly the game length, because every move
     runs a search;
  4. one `game_complete` payload per drained game;
  5. a `system_stats` payload at ~5 s resolution so a monitor stays fresh between
     training-iteration events;
  6. one heartbeat per iteration.

`time` is imported as a module so the drain oracle can substitute a scripted clock — the
same seam the old-side capture harness used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from mantis.selfplay.instrumentation import _compute_stride5_metrics
from mantis.selfplay.pool_push import push_dense, push_graph

_LOG = logging.getLogger(__name__)

# Rust `winner_code` → log-facing name; codes at or above 3 are unknown to this map.
_WINNER_NAMES = ("draw", "x", "o")

# Emitted once per drain iteration when a heartbeat is injected. The consuming watchdog
# belongs to the monitoring work package and is NOT built here.
_HEARTBEAT_SOURCE = "selfplay_drain"

# Buffer-stats emission cadence, seconds.
_SYSTEM_STATS_PERIOD_S = 5.0


def _emit(pool: Any, payload: dict[str, Any]) -> None:
    """Hand one payload to the injected event sink. No sink ⇒ the event is dropped."""
    sink = pool._sink
    if sink is not None:
        sink.emit(payload)


def _beat(pool: Any) -> None:
    """Fire the drain heartbeat. No heartbeat injected ⇒ nothing happens at all."""
    hb = pool._heartbeat
    if hb is not None:
        hb(_HEARTBEAT_SOURCE)


def run_stats_loop(pool: Any) -> None:
    """Run the feeder loop until the pool's stop event is set."""
    _last_buf_emit = time.monotonic()
    # WP12R Step 3 narration (R210/R216/R218, LAW-18): lifecycle events through the injected
    # selfplay-local `EventSink`. `game_loop_entered` fires once on drain-thread entry.
    if pool._sink is not None:
        pool._sink.emit({"event": "game_loop_entered"})
    _first_record_drained = False
    while not pool._stop_event.is_set():
        # ONE hoisted branch. A graph spec drains via `collect_graph_data()` →
        # `push_graph_position` (no planes / chain / ownership / winning line — the graph
        # net has only policy and value heads); every grid encoding takes the dense
        # `collect_data()` → bulk-push arm unchanged.
        if pool._is_graph:
            collected_rows = pool._runner.collect_graph_data()
            push_graph(pool, collected_rows)
        else:
            collected_rows = pool._runner.collect_data()
            push_dense(pool, collected_rows)

        with pool._lock:
            pool.games_completed = int(pool._runner.games_completed)
            pool.x_wins = int(pool._runner.x_wins)
            pool.o_wins = int(pool._runner.o_wins)
            pool.draws = int(pool._runner.draws)

        # Fully consumed each iteration; no unbounded accumulation. The drain returns
        # metadata-only tuples — spatial aux targets flow per-row through the push arm.
        games_batch = pool._runner.drain_game_results()

        # WP12R Step 3 narration (R210/R216/R218, DESIGN §4.5 (β)): `first_record_drained`
        # fires once on the first NON-EMPTY drain (a record actually flowed — the event name
        # is honest, not a "drain loop first iteration" regardless of content). OR semantics:
        # positions flow before games complete, so the gate is (collected_rows OR games_batch)
        # — a position push counts as a record flowing even when no game has finished yet.
        if not _first_record_drained and (collected_rows or games_batch):
            _first_record_drained = True
            if pool._sink is not None:
                pool._sink.emit({
                    "event": "first_record_drained",
                    "representation": "graph" if pool._is_graph else "dense",
                })

        # Bill one search per MOVE, not per GAME. `positions_generated` counts
        # row-producing moves, so the delta tracks moves accrued in this interval and is
        # decoupled from game completions.
        now = time.monotonic()
        elapsed = now - pool._last_drain_time
        pool._last_drain_time = now
        pos_generated = int(getattr(pool._runner, "positions_generated", 0))
        # AUDIT-1 F-28/C07. `positions_generated` is monotone in the runner, so a NEGATIVE
        # delta means the counter was reset (a replaced runner) and not that no work happened.
        # `max(0, ...)` published the two as one observable and left the rate reading as a
        # measured stall. The clamp stays — the rate must not go negative — but the event says
        # it fired, so the two are distinguishable in the stream.
        raw_delta = pos_generated - pool._last_pos_generated
        pos_delta = max(0, raw_delta)
        if raw_delta < 0 and pool._sink is not None:
            pool._sink.emit({"event": "positions_counter_reset", "delta": int(raw_delta),
                             "last_seen": int(pool._last_pos_generated),
                             "now": int(pos_generated)})
        pool._last_pos_generated = pos_generated
        if pos_delta > 0:
            sims = pos_delta * pool._effective_sims_per_move
            pool._total_sims += sims
            if elapsed > 0:
                pool._sims_per_sec = sims / elapsed

        for entry in games_batch:
            (plies, winner_code, move_history, worker_id,
             terminal_reason, mv_min, mv_max, mv_distinct,
             seeded, solver_fires) = entry
            winner = _WINNER_NAMES[winner_code] if winner_code < 3 else "unknown"
            game_length = (plies + 1) // 2  # compound moves
            pool._game_lengths.append(game_length)
            pool._avg_game_length = sum(pool._game_lengths) / len(pool._game_lengths)
            # Stride-5 per-game detection: a pure function the drain computes, while the
            # instrumentation owns the rolling window and the P90.
            if move_history:
                _stride5_run, _row_max_density = _compute_stride5_metrics(move_history)
            else:
                _stride5_run = 0
                _row_max_density = 0

            (_ext_count, _ext_total, _ext_frac, _stride5_p90,
             _longest_line, _longest_line_frac, _n_components) = (
                pool._instrumentation.on_game_complete(
                    pool._lock, winner_code, move_history, worker_id,
                    terminal_reason, mv_min, mv_max, mv_distinct, _stride5_run,
                    # cluster_threshold defaults to the PINNED engine constant; there is
                    # no config plumbing for it.
                )
            )

            # winner_code → spec convention: 0=P0, 1=P1, -1=draw.
            # AUDIT-1 F-28/C04: an UNRECOGNISED code used to fall to `-1`, i.e. to be
            # reported as a measured DRAW. It is `None` — no outcome was decoded — and it is
            # logged, the way `terminal_reason` already maps an unknown code to a named
            # "unknown" rather than to one of its real values.
            winner_int = {0: -1, 1: 0, 2: 1}.get(winner_code)
            if winner_int is None:
                _LOG.error("game_complete_winner_undecodable: winner_code=%s", winner_code)

            moves_list = [f"({q},{r})" for q, r in move_history] if move_history else []

            # Rust terminal_reason u8 → the string convention the monitor reads. An
            # unrecognised code maps to "unknown" rather than raising.
            _TR_NAMES = {0: "six_in_a_row", 1: "colony", 2: "ply_cap", 3: "other_draw"}
            terminal_reason_name = _TR_NAMES.get(int(terminal_reason), "unknown")
            # Deterministic byte-hash over the move sequence so byte-identical games
            # dedupe — effective-n counts DISTINCT games, not game count (LAW-04). The
            # symmetry-canonical hash is a later, Rust-side instrument. `game_id` (uuid)
            # stays unique per emit for monitor keying.
            _game_id_byte_hash = hashlib.sha1(
                json.dumps([list(m) for m in move_history]).encode()
            ).hexdigest()
            game_complete_payload: dict[str, Any] = {
                "event": "game_complete",
                "game_id": uuid.uuid4().hex,
                "game_id_byte_hash": _game_id_byte_hash,
                "winner": winner_int,
                "moves": plies,
                "moves_list": moves_list,
                "worker_id": worker_id,
                # Per-move MCTS detail: None until the Rust game runner stores
                # top_visits / root_value per move in its drain.
                "moves_detail": None,
                "value_trace": None,
                # Colony extension: count / total / fraction of stones placed beyond the
                # pinned hex distance from any opponent stone.
                "colony_extension_stone_count": _ext_count,
                "colony_extension_stone_total": _ext_total,
                "colony_extension_fraction":    _ext_frac,
                # PER-PLAYER (winner) structural metrics. longest_line is capped at the
                # win length; fraction = longest_line / max(1, winner stones);
                # n_components under the pinned cluster threshold.
                "longest_line_fraction":        _longest_line_frac,
                "n_components":                 _n_components,
                # Always emitted (cheap), so post-hoc analysis can pick these up without
                # re-running with the investigation flag set.
                "terminal_reason":          terminal_reason_name,
                "model_version_min":        int(mv_min),
                "model_version_max":        int(mv_max),
                "model_version_distinct":   int(mv_distinct),
                "stride5_run_p90":   int(_stride5_p90),
                # Densest hex-row stone count over the three axes.
                "row_max_density":   int(_row_max_density),
                # Per-game seeding + solver-fire metadata.
                "seeded":            int(seeded),
                "solver_fires":      int(solver_fires),
            }
            _emit(pool, game_complete_payload)

            _LOG.info(
                "game_complete: plies=%s winner=%s game_length=%s sims_per_sec=%s "
                "colony_extension_stone_count=%s colony_extension_stone_total=%s "
                "colony_extension_fraction=%s",
                plies, winner, game_length, pool._sims_per_sec,
                _ext_count, _ext_total, _ext_frac,
            )
            pool._recorder.maybe_record(
                moves=move_history,
                winner_code=winner_code,
                game_length=plies,
            )

        # Buffer stats at ~5 s resolution. `_last_buf_emit` is a LOCAL of this loop: a
        # copy hoisted onto the pool and left un-updated would emit every tick.
        _now_buf = time.monotonic()
        if _now_buf - _last_buf_emit >= _SYSTEM_STATS_PERIOD_S:
            _last_buf_emit = _now_buf
            _emit(pool, {
                "event": "system_stats",
                "buffer_size": pool.replay_buffer.size,
                "buffer_capacity": pool.replay_buffer.capacity,
            })

        _beat(pool)
        time.sleep(0.1)


__all__ = ["run_stats_loop"]
