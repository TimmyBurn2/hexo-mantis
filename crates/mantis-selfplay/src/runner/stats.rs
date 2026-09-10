//! `WorkerStats` — per-worker accumulator bundle (WP6 D1/D13, LAW-18), ported
//! verbatim from the frozen `worker_loop/stats.rs`.
//!
//! `Arc<AtomicU*>` fire-rate / health accumulators cloned once per worker spawn (cheap
//! `Arc::clone`-per-field) and destructured at `game::run_worker_thread` entry. The solver,
//! seeded-corpus and K-cluster counters went with their levers (R346(f)).

use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct WorkerStats {
    pub(crate) games_completed: Arc<AtomicUsize>,
    pub(crate) positions_generated: Arc<AtomicUsize>,
    pub(crate) x_wins: Arc<AtomicU64>,
    pub(crate) o_wins: Arc<AtomicU64>,
    pub(crate) draws: Arc<AtomicU64>,
    pub(crate) positions_dropped: Arc<AtomicU64>,
    pub(crate) mcts_depth_accum: Arc<AtomicU64>,
    pub(crate) mcts_conc_accum: Arc<AtomicU64>,
    pub(crate) mcts_stat_count: Arc<AtomicU64>,
    pub(crate) mcts_quiescence_fires: Arc<AtomicU64>,
    /// R335(c) — the largest number of leaves ANY one search served. The budget
    /// is `n_simulations` (or the playout-cap arm's), so this must never exceed
    /// it; before the batch clamp it read `n_simulations + leaf_batch_size - 1`
    /// and the run served ~7 % more sims than the config named.
    pub(crate) max_sims_per_search: Arc<AtomicU64>,
    /// LAW-18 — playout-cap randomization's own fire rate, counted where the ARM IS DRAWN.
    /// The recorded row's `is_full_search` flag is the only other place the arm is visible,
    /// and with the forced-win and solver hooks deleted the draw is now the only writer.
    /// These two count the DRAW.
    pub(crate) pcr_full_moves: Arc<AtomicU64>,
    pub(crate) pcr_quick_moves: Arc<AtomicU64>,
    /// LAW-18 — the Gumbel halving round's WIDTH: leaves issued per inference round trip,
    /// as the two terms of a mean. A batching lever whose fire rate is not in the run
    /// cannot be told from one that has silently gone back to a leaf per round trip.
    pub(crate) gumbel_round_leaves: Arc<AtomicU64>,
    pub(crate) gumbel_rounds: Arc<AtomicU64>,
    // WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6).
    pub(crate) export_offwindow_mass_moves: Arc<AtomicU64>,
}
