//! `WorkerStats` — per-worker `Arc<AtomicU*>` fire-rate / health accumulators, cloned once per
//! worker spawn and destructured at `game::run_worker_thread` entry.

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
    /// The largest number of leaves ANY one search served; it must never exceed the
    /// budget, `n_simulations` (or the playout-cap arm's).
    pub(crate) max_sims_per_search: Arc<AtomicU64>,
    /// Playout-cap randomization's own fire rate: these two count the DRAW, where the ARM IS
    /// DRAWN; the row's `is_full_search` flag is the only other place the arm is visible.
    pub(crate) pcr_full_moves: Arc<AtomicU64>,
    pub(crate) pcr_quick_moves: Arc<AtomicU64>,
    /// The Gumbel halving round's WIDTH: leaves issued per inference round trip, as the two
    /// terms of a mean, so a lever silently back at one leaf per round trip shows in-run.
    pub(crate) gumbel_round_leaves: Arc<AtomicU64>,
    pub(crate) gumbel_rounds: Arc<AtomicU64>,
    /// Root Dirichlet applications, counted at the PUCT arm's mix-in site (0 under Gumbel).
    pub(crate) dirichlet_root_fires: Arc<AtomicU64>,
    // Target-integrity fire-rate counters.
    pub(crate) export_offwindow_mass_moves: Arc<AtomicU64>,
}
