//! R8-justify: the finalize phase + its in-src §178 ply-cap-outcome unit test
//! (P-03) stay together — the test must reach the `pub(crate)` `finalize_game` to
//! exercise the VALUE branch (`outcome == ply_cap_value`) that the `drain_game_results`
//! tuple cannot observe (outcome lives on `WorkerResultRow`, absent from `GameResultRow`).
//!
//! Finalize phase (WP6 D1) — `finalize_game` / `finalize_game_graph` (frozen
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

use mantis_core::{Board, Cell, Player};

use crate::records;
use crate::replay::hexg::GraphRecord;
use crate::replay::sym::SymTables;

use super::record::RecordTuple;
use super::rotate::rotate_aux_inplace;
use super::{GameResultRow, WorkerResultRow};

/// Per-game terminal handler (frozen `inner.rs:1624`; warm path).
///
/// Classifies the outcome (winner / `terminal_reason` / `version_seen` range),
/// reprojects + rotates per-row aux targets, pushes all rows into the shared
/// results queue under ONE lock, bumps the win/draw counters, caps the queue at
/// `results_queue_cap`, and pushes a single `recent_game_results` metadata row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_game(
    board: &Board,
    max_moves: usize,
    records_vec: Vec<RecordTuple>,
    move_history: Vec<(i32, i32)>,
    version_seen: &[u64],
    sym_idx: usize,
    sym_tables: &'static SymTables,
    n_cells: usize,
    draw_reward: f32,
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,
    seeded: bool,
    solver_fires: u32,
    results_queue: &Mutex<VecDeque<WorkerResultRow>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    games_completed: &AtomicUsize,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
    positions_dropped: &AtomicU64,
) {
    // ── Game End: determine outcome ──
    let winner = board.winner();
    let plies = board.ply.index() as usize;
    let winner_code: u8 = match winner {
        Some(Player::One) => 1,
        Some(_) => 2,
        None => 0,
    };
    // Snapshot the final-board cell list and winning line once; each row
    // reprojects them into its own per-cluster window centre.
    let final_cells: Vec<((i32, i32), Cell)> = board.cells_iter().map(|(&qr, &c)| (qr, c)).collect();
    let winning_cells: Vec<(i32, i32)> = board.find_winning_line();

    // terminal_reason (Phase B' Class-3):
    //   0 = six_in_a_row : winner exists AND winning_cells non-empty
    //   1 = colony       : winner exists AND no winning_line
    //   2 = ply_cap      : no winner AND ply >= max_moves
    //   3 = other_draw   : no winner AND ply < max_moves
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

    // DRAW-MASK: `terminal_reason == 2` (ply-cap horizon truncation) masks its
    // fabricated `ply_cap_value` label out of the value loss.
    let value_valid: u8 = u8::from(terminal_reason != 2);
    let mut games_results = results_queue.lock().expect("results lock poisoned");
    for (feat, chain, pol, player, cq, cr, is_full_search, ply_index) in records_vec {
        // §178: split outcome by terminal_reason. Default cfg has both values
        // equal (`-0.1`), preserving pre-§178 behaviour byte-for-byte.
        let outcome = match winner {
            Some(p) => {
                if p as i8 == player as i8 {
                    1.0
                } else {
                    -1.0
                }
            }
            None => {
                if terminal_reason == 2 {
                    ply_cap_value
                } else {
                    draw_reward
                }
            }
        };

        // Per-row aux reprojection (ownership + winning_line) into this row's
        // per-cluster window centre.
        let mut aux_u8 = records::reproject_game_end_row(&final_cells, &winning_cells, cq, cr, n_cells);
        // §130: forward-scatter the aux pair into the same rotated frame as
        // state/chain/policy (reproject + scatter compose — both are pure
        // permutations on cell indices).
        if sym_idx != 0 {
            rotate_aux_inplace(&mut aux_u8, sym_idx, sym_tables, n_cells);
        }

        games_results.push_back((feat, chain, pol, outcome, plies, aux_u8, is_full_search, ply_index, value_valid));
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
        seeded,
        solver_fires,
    );

    // Cap the results queue to avoid memory explosion if the drain is slow;
    // dropped count tracked on `positions_dropped` for dashboard visibility.
    if games_results.len() > results_queue_cap {
        let to_drop = games_results.len() - results_queue_cap;
        for _ in 0..to_drop {
            games_results.pop_front();
        }
        positions_dropped.fetch_add(to_drop as u64, Ordering::Relaxed);
    }
}

/// Graph sibling of `finalize_game` (frozen `inner.rs:1767`). Reuses the winner /
/// `terminal_reason` / `version_seen` classification verbatim; stamps each
/// buffered `GraphRecord`'s `outcome`/`value_valid` via the §178 KEEP-verbatim
/// `records::finalize_graph_outcome`, sets `game_length` (compound-move count,
/// `plies.div_ceil(2)`), and drains into `graph_results_queue`. NO cell-geometry
/// reprojection (the graph net has no ownership/winning-line head). Pushes the
/// SAME `recent_game_results` metadata row and increments the same counters.
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
    seeded: bool,
    solver_fires: u32,
    graph_results_queue: &Mutex<VecDeque<GraphRecord>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    games_completed: &AtomicUsize,
    x_wins: &AtomicU64,
    o_wins: &AtomicU64,
    draws: &AtomicU64,
    positions_dropped: &AtomicU64,
) {
    // ── Game End: determine outcome (verbatim twin of finalize_game) ──
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

    let mut gq = graph_results_queue.lock().expect("graph_results_queue lock poisoned");
    for mut rec in graph_records {
        // §178 KEEP-verbatim split — reads rec.current_player / winner /
        // terminal_reason only, no cell geometry.
        let (outcome, value_valid_u8) =
            records::finalize_graph_outcome(rec.current_player, winner, terminal_reason, ply_cap_value, draw_reward);
        rec.outcome = outcome;
        rec.value_valid = value_valid_u8 != 0;
        rec.game_length = game_length;
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
        seeded,
        solver_fires,
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
fn bump_win_counters(winner: Option<Player>, x_wins: &AtomicU64, o_wins: &AtomicU64, draws: &AtomicU64) {
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
    seeded: bool,
    solver_fires: u32,
) {
    let (mv_min, mv_max, mv_distinct) = versions;
    let mut rg = recent_game_results.lock().expect("recent_game_results lock poisoned");
    rg.push_back((
        plies,
        winner_code,
        move_history,
        worker_id,
        terminal_reason,
        mv_min,
        mv_max,
        mv_distinct,
        u8::from(seeded),
        solver_fires,
    ));
    if rg.len() > 2000 {
        rg.pop_front();
    }
}

// ── P-03 focused finalize_game §178 outcome unit test (the VALUE branch invisible
//    to the drain tuple; see the module-header R8-justify). Placed LAST so no
//    non-test item follows the test module (clippy::items_after_test_module). ──
#[cfg(test)]
mod inv26_finalize_outcome_tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Mutex;

    use mantis_core::Board;
    use mantis_encoding::lookup;

    use crate::replay::sym::{sym_tables_for, SymTables};
    use crate::runner::record::RecordTuple;

    use super::finalize_game;
    use super::{GameResultRow, WorkerResultRow};

    const V6_N_CELLS: usize = 361;

    /// A no-winner board with exactly 6 scattered stones (winner == None). With so
    /// few stones a 6-in-a-row is impossible; we assert `winner().is_none()` so a
    /// mis-chosen coordinate fails loudly rather than silently flipping the branch.
    fn no_winner_board() -> Board {
        let mut b = Board::new();
        for &(q, r) in &[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)] {
            b.apply_move(q, r).expect("move within the default legal radius");
        }
        assert!(b.winner().is_none(), "6 scattered stones must not produce a winner");
        b
    }

    /// A single record row centred on the board window (the reproject target).
    fn one_record(board: &Board) -> RecordTuple {
        let (cq, cr) = board.window_center();
        (
            vec![0.0f32; 4],      // feat (contents irrelevant to the outcome branch)
            vec![0.0f32; 4],      // chain
            vec![0.0f32; 4],      // projected_policy
            board.current_player, // player
            cq,
            cr,
            true, // is_full_search
            board.ply.index() as u16,
        )
    }

    /// Call the REAL `finalize_game` and return the single pushed `WorkerResultRow`.
    /// `sym_idx = 0` ⇒ no aux rotation, so any `&'static SymTables` is inert here.
    fn finalize_single(max_moves: usize, draw_reward: f32, ply_cap_value: f32) -> WorkerResultRow {
        let board = no_winner_board();
        let sym_tables: &'static SymTables = sym_tables_for(lookup("v6").expect("v6 spec"));
        let results: Mutex<VecDeque<WorkerResultRow>> = Mutex::new(VecDeque::new());
        let recent: Mutex<VecDeque<GameResultRow>> = Mutex::new(VecDeque::new());
        let games_completed = AtomicUsize::new(0);
        let x_wins = AtomicU64::new(0);
        let o_wins = AtomicU64::new(0);
        let draws = AtomicU64::new(0);
        let positions_dropped = AtomicU64::new(0);

        finalize_game(
            &board,
            max_moves,
            vec![one_record(&board)],
            Vec::new(),
            &[],
            0,
            sym_tables,
            V6_N_CELLS,
            draw_reward,
            ply_cap_value,
            10_000,
            0,
            false,
            0,
            &results,
            &recent,
            &games_completed,
            &x_wins,
            &o_wins,
            &draws,
            &positions_dropped,
        );

        let mut guard = results.lock().expect("results lock");
        guard.pop_front().expect("exactly one row pushed")
    }

    /// The §178 branch the drain tuple cannot see: a ply-cap terminal
    /// (`plies >= max_moves`, winner None ⇒ `terminal_reason == 2`) pays
    /// `ply_cap_value`, NOT `draw_reward`, and `value_valid == 0` (DRAW-MASK).
    #[test]
    fn ply_cap_terminal_outcome_is_ply_cap_value_not_draw_reward() {
        // board has 6 plies; max_moves = 6 ⇒ plies >= max_moves ⇒ terminal_reason 2.
        let row = finalize_single(6, -0.1, -0.5);
        let outcome = row.3;
        let value_valid = row.8;
        assert!(
            (outcome - (-0.5)).abs() < 1e-9,
            "ply-cap outcome must equal ply_cap_value (-0.5); got {outcome}",
        );
        assert!(
            (outcome - (-0.1)).abs() > 1e-9,
            "ply-cap outcome must NOT equal draw_reward (-0.1); got {outcome}",
        );
        assert_eq!(value_valid, 0, "ply-cap (reason 2) masks the fabricated label (value_valid=0)");
    }

    /// The complementary branch: an organic non-terminal-by-cap board
    /// (`plies < max_moves`, winner None ⇒ `terminal_reason == 3`) pays
    /// `draw_reward` and `value_valid == 1`. Together with the test above this pins
    /// the split: the two outcomes DIVERGE exactly on `terminal_reason`.
    #[test]
    fn organic_draw_outcome_is_draw_reward_with_valid_value() {
        // max_moves = 100 ⇒ 6 plies < max_moves ⇒ terminal_reason 3.
        let row = finalize_single(100, -0.1, -0.5);
        let outcome = row.3;
        let value_valid = row.8;
        assert!(
            (outcome - (-0.1)).abs() < 1e-9,
            "organic draw outcome must equal draw_reward (-0.1); got {outcome}",
        );
        assert_eq!(value_valid, 1, "organic draw (reason 3) is a valid value target");
    }
}
