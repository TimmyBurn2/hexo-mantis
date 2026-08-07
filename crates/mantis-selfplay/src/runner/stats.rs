//! `WorkerStats` — per-worker accumulator bundle (WP6 D1/D13, LAW-18), ported
//! verbatim from the frozen `worker_loop/stats.rs`.
//!
//! 23 `Arc<AtomicU*>` fire-rate / health accumulators cloned once per worker
//! spawn (cheap `Arc::clone`-per-field) and destructured at
//! `game::run_worker_thread` entry. The solver counters are incremented ONLY
//! under the `solver_enabled` / seeded branches, so an OFF (default) run leaves
//! the bench-gated hot path byte-identical.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use super::record::K_CLUSTER_HISTOGRAM_BUCKETS;

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
    pub(crate) cluster_value_std_accum: Arc<AtomicU64>,
    pub(crate) cluster_policy_disagreement_accum: Arc<AtomicU64>,
    pub(crate) cluster_variance_samples: Arc<AtomicU64>,
    // D-WS3V3 in-run solver fire-rate counters (cumulative since `start()`).
    pub(crate) solver_moves_eligible: Arc<AtomicU64>,
    pub(crate) solver_win_proven: Arc<AtomicU64>,
    pub(crate) solver_injected: Arc<AtomicU64>,
    pub(crate) solver_injected_offwindow: Arc<AtomicU64>,
    pub(crate) solver_budget_exhausted: Arc<AtomicU64>,
    pub(crate) solver_moves_eligible_seeded: Arc<AtomicU64>,
    pub(crate) solver_injected_seeded: Arc<AtomicU64>,
    pub(crate) seeded_games_started: Arc<AtomicU64>,
    // WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6).
    pub(crate) export_offwindow_mass_moves: Arc<AtomicU64>,
    pub(crate) gridls_zero_policy_rows: Arc<AtomicU64>,
    /// R256/ADJ-D37: proven forced wins swallowed by the LS coverage gate while
    /// the injecting lever was armed (both the O1 arm and the solver hook).
    pub(crate) uncovered_forced_win: Arc<AtomicU64>,
    /// Item 10(b) / R250: the in-run K histogram, one bucket per cluster-view
    /// count at the DENSE record path. Written only by `record::record_position`,
    /// which the graph arm never calls — so on a graph run every bucket stays 0
    /// and the emitter omits the field rather than publishing that zero.
    pub(crate) k_cluster_histogram: Arc<[AtomicU64; K_CLUSTER_HISTOGRAM_BUCKETS]>,
}
