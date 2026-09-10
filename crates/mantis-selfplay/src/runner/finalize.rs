//! R8-justify: the finalize phase + its in-src §178 ply-cap-outcome unit test
//! exercise the VALUE branch (`outcome == ply_cap_value`) that the `drain_game_results`
//! tuple cannot observe (outcome lives on `WorkerResultRow`, absent from `GameResultRow`).
//!
//! Finalize phase (WP6 D1) — `finalize_game_graph` (frozen
//! `worker_loop/inner.rs:1624/1767`, dispatch branch `:571`).
//!
//! Ports the §178 ply-cap value branch VERBATIM: the `winner == None` arm pays
//! `ply_cap_value` when `terminal_reason == 2` else `draw_reward`; `value_valid`
//! is the DRAW-MASK (`terminal_reason != 2`). The per-game push loop holds the
//! results-queue lock ONCE across the whole game (frozen `:1689`) so every game's
//! rows are CONTIGUOUS in the shared queue (observable only multi-worker; ported
//! as a verbatim obligation). The terminal reason / outcome are read from
//! `board.winner()` + `terminal_reason` (never re-derived from ply parity,
//! LAW-03). Feeds the shared `VecDeque` result queues (the pyo3 `collect_data`
//! drain is WP7); drop-oldest backpressure bumps `positions_dropped`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use mantis_core::{Board, Player};

use crate::records;
use crate::replay::hexg::GraphRecord;

use super::GameResultRow;

/// Per-game terminal handler (frozen `inner.rs:1624`; warm path).
///
/// Classifies the outcome (winner / `terminal_reason` / `version_seen` range),
/// reprojects + rotates per-row aux targets, pushes all rows into the shared
/// results queue under ONE lock, bumps the win/draw counters, caps the queue at
/// `results_queue_cap`, and pushes a single `recent_game_results` metadata row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_game_graph(
    board: &Board,
    max_moves: usize,
    graph_records: Vec<GraphRecord>,
    move_history: Vec<(i32, i32)>,
    version_seen: &[u64],
    draw_reward: f32,
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,

    graph_results_queue: &Mutex<VecDeque<GraphRecord>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    games_completed: &AtomicUsize,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
    positions_dropped: &AtomicU64,
    graph_game_seq: &AtomicU64,
) {
    // ── Game End: determine outcome ──
    let winner = board.winner();
    let plies = board.ply.index() as usize;
    let winner_code: u8 = match winner {
        Some(Player::One) => 1,
        Some(_) => 2,
        None => 0,
    };
    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();
    let terminal_reason: u8 = match winner {
        Some(_) => u8::from(winning_cells.is_empty()),
        None => {
            if plies >= max_moves {
                2
            } else {
                3
            }
        }
    };
    let (mv_min, mv_max, mv_distinct) = version_range(version_seen);
    // Compound-move sampling weight — same `(plies+1)/2` (== `div_ceil(2)`)
    // convention the dense drain applies to `plies` before push.
    let game_length: u16 = plies.div_ceil(2).min(u16::MAX as usize) as u16;

    // R345(b)(6): ONE id for the whole game, taken before the loop. Taking it per record
    // would tag every position as its own game, which is the `-1` sentinel's behaviour wearing
    // real numbers — the dedupe would still never fire and nothing would say so.
    let game_id = graph_game_seq.fetch_add(1, Ordering::Relaxed) as i64;
    let mut gq = graph_results_queue
        .lock()
        .expect("graph_results_queue lock poisoned");
    for mut rec in graph_records {
        // §178 KEEP-verbatim split — reads rec.current_player / winner /
        // terminal_reason only, no cell geometry.
        let (outcome, value_valid_u8) = records::finalize_graph_outcome(
            rec.current_player,
            winner,
            terminal_reason,
            ply_cap_value,
            draw_reward,
        );
        rec.outcome = outcome;
        rec.value_valid = value_valid_u8 != 0;
        rec.game_length = game_length;
        rec.game_id = game_id;
        gq.push_back(rec);
    }
    games_completed.fetch_add(1, Ordering::Relaxed);
    bump_win_counters(winner, x_wins, o_wins, draws);

    push_recent_meta(
        recent_game_results,
        plies,
        winner_code,
        move_history,
        worker_id,
        terminal_reason,
        (mv_min, mv_max, mv_distinct),
    );

    // Cap the graph results queue (parity with the dense backpressure drop).
    if gq.len() > results_queue_cap {
        let to_drop = gq.len() - results_queue_cap;
        for _ in 0..to_drop {
            gq.pop_front();
        }
        positions_dropped.fetch_add(to_drop as u64, Ordering::Relaxed);
    }
}

/// Collapse `version_seen` into `(min, max, distinct)` (frozen `:1675`).
fn version_range(version_seen: &[u64]) -> (u64, u64, u32) {
    if version_seen.is_empty() {
        (0, 0, 0)
    } else {
        let mn = *version_seen.iter().min().unwrap();
        let mx = *version_seen.iter().max().unwrap();
        (mn, mx, version_seen.len() as u32)
    }
}

/// Increment the per-outcome win/draw counters (frozen `:1716`).
fn bump_win_counters(
    winner: Option<Player>,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
) {
    match winner {
        Some(Player::One) => {
            x_wins.fetch_add(1, Ordering::Relaxed);
        }
        Some(_) => {
            o_wins.fetch_add(1, Ordering::Relaxed);
        }
        None => {
            draws.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Push the single per-game `recent_game_results` metadata row (frozen `:1723`),
/// capped at 2000 entries. Shared by both finalize variants (representation-blind
/// drain).
#[allow(clippy::too_many_arguments)]
fn push_recent_meta(
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    plies: usize,
    winner_code: u8,
    move_history: Vec<(i32, i32)>,
    worker_id: usize,
    terminal_reason: u8,
    versions: (u64, u64, u32),
) {
    let (mv_min, mv_max, mv_distinct) = versions;
    let mut rg = recent_game_results
        .lock()
        .expect("recent_game_results lock poisoned");
    rg.push_back((
        plies,
        winner_code,
        move_history,
        worker_id,
        terminal_reason,
        mv_min,
        mv_max,
        mv_distinct,
    ));
    if rg.len() > 2000 {
        rg.pop_front();
    }
}
