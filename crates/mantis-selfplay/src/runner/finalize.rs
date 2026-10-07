//! Finalize phase — `finalize_game_graph`.
//!
//! A game with no winner trains no value: its rows carry 0 and `value_valid` false, at the cap
//! (reason 2) and short of it (reason 3) alike. The loop holds the results-queue lock ONCE per
//! game so its rows are CONTIGUOUS in the shared queue (observable only multi-worker). The terminal reason / outcome are read from
//! `board.winner()` + `terminal_reason`, never re-derived from ply parity. Drop-oldest past
//! `results_queue_cap` bumps `positions_dropped`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use mantis_core::{Board, Player};

use crate::poison::lock_or_recover;
use crate::records;
use crate::replay::hexg::GraphRecord;

use super::{GameResultRow, MoveArm, PositionStats};

/// Per-game terminal handler (warm path).
///
/// Classifies the outcome (winner / `terminal_reason` / `version_seen` range), pushes all rows
/// into the shared graph results queue under ONE lock, bumps the win/draw counters, caps the queue
/// at `results_queue_cap`, and pushes a single `recent_game_results` metadata row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_game_graph(
    board: &Board,
    max_moves: usize,
    graph_records: Vec<GraphRecord>,
    move_history: Vec<(i32, i32)>,
    move_arms: Vec<MoveArm>,
    search_stats: Option<Vec<PositionStats>>,
    version_seen: &[u64],
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
    // Compound-move game length: `(plies+1)/2` (== `div_ceil(2)`).
    let game_length: u16 = plies.div_ceil(2).min(u16::MAX as usize) as u16;

    // ONE id for the whole game, taken before the loop: a per-record id would silently defeat
    // the same-game dedupe.
    let game_id = graph_game_seq.fetch_add(1, Ordering::Relaxed) as i64;
    // Written through a poisoned lock: the poisoning panic already halted the run.
    let mut gq = lock_or_recover(graph_results_queue, None);
    for mut rec in graph_records {
        let (outcome, value_valid_u8) =
            records::finalize_graph_outcome(rec.current_player, winner, terminal_reason);
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
        move_arms,
        search_stats,
        worker_id,
        terminal_reason,
        (mv_min, mv_max, mv_distinct),
    );

    // Cap the graph results queue: drop-oldest backpressure.
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
    match (version_seen.iter().min(), version_seen.iter().max()) {
        (Some(&mn), Some(&mx)) => (mn, mx, version_seen.len() as u32),
        _ => (0, 0, 0),
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

/// Push the single per-game `recent_game_results` metadata row,
/// capped at 2000 entries. Shared by both finalize variants (representation-blind
/// drain).
#[allow(clippy::too_many_arguments)]
fn push_recent_meta(
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    plies: usize,
    winner_code: u8,
    move_history: Vec<(i32, i32)>,
    move_arms: Vec<MoveArm>,
    search_stats: Option<Vec<PositionStats>>,
    worker_id: usize,
    terminal_reason: u8,
    versions: (u64, u64, u32),
) {
    let (mv_min, mv_max, mv_distinct) = versions;
    let mut rg = lock_or_recover(recent_game_results, None);
    rg.push_back((
        plies,
        winner_code,
        move_history,
        worker_id,
        terminal_reason,
        mv_min,
        mv_max,
        mv_distinct,
        move_arms,
        search_stats,
    ));
    if rg.len() > 2000 {
        rg.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Mutex;

    use mantis_core::Board;

    use super::finalize_game_graph;
    use crate::replay::hexg::GraphRecord;

    /// PLANTED BREAK: mask the cap alone and this game's rows train a value with no result behind it.
    #[test]
    fn a_planted_game_without_a_winner_short_of_the_cap_trains_no_value() {
        let (queue, recent) = (Mutex::new(VecDeque::new()), Mutex::new(VecDeque::new()));
        let (games, seq) = (AtomicUsize::new(0), AtomicU64::new(0));
        let (x, o, d, dropped) = (
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        );
        let rows = [1, -1].map(|current_player| GraphRecord {
            current_player,
            ..GraphRecord::default()
        });
        // An empty board under a 200-ply cap: no winner, short of the cap.
        finalize_game_graph(
            &Board::new(),
            200,
            rows.to_vec(),
            vec![],
            vec![],
            None,
            &[0],
            1_000,
            0,
            &queue,
            &recent,
            &games,
            &x,
            &o,
            &d,
            &dropped,
            &seq,
        );
        let recent = recent.into_inner().expect("unpoisoned");
        assert_eq!(recent[0].4, 3, "the planted game ends as reason 3");
        let queue = queue.into_inner().expect("unpoisoned");
        assert_eq!(queue.len(), 2);
        for rec in &queue {
            assert!(!rec.value_valid, "a reason-3 row trained its value");
            assert_eq!(rec.outcome, 0.0);
        }
    }
}
