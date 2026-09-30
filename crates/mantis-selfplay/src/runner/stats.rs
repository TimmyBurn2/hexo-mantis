//! `WorkerStats` — per-worker `Arc<AtomicU*>` fire-rate / health accumulators, cloned once per
//! worker spawn and destructured at `game::run_worker_thread` entry.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use mantis_search::mcts::TacticsCounters;

/// The rows a move adds beside its search's, in `MoveRow` order, full draws only: one kind a row, then two cross-counts.
pub const MOVE_TACTICS_ROWS: [&str; 6] = [
    "proven_root_rows",
    "decided_lost_rows",
    "vetoed_target_rows",
    "emptied_target_rows",
    "mixed_rows",
    "vetoed_all_rows",
];

/// A row a move records: a decided root's searched target, no policy at a lost or all-vetoed root, vetoes zeroed.
#[derive(Clone, Copy)]
pub(crate) enum MoveRow {
    ProvenRoot = 0,
    DecidedLost = 1,
    VetoedTarget = 2,
    EmptiedTarget = 3,
    Mixed = 4,
    VetoedAll = 5,
}

/// Every search's tactics rows summed (`TacticsCounters::rows` order), then the move rows.
pub(crate) struct TacticsTotals {
    slots: Vec<AtomicU64>,
}

impl TacticsTotals {
    pub(crate) fn new() -> Self {
        let n = TacticsCounters::ROWS + MOVE_TACTICS_ROWS.len();
        Self {
            slots: (0..n).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Add one search's rows.
    pub(crate) fn add_search(&self, rows: &TacticsCounters) {
        for (slot, (_, v)) in self.slots.iter().zip(rows.rows()) {
            if v > 0 {
                slot.fetch_add(v, Ordering::Relaxed);
            }
        }
    }

    /// Count one recorded row of `row`'s kind.
    pub(crate) fn add_move(&self, row: MoveRow) {
        self.slots[TacticsCounters::ROWS + row as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Every row by name, each read once with a `Relaxed` load.
    pub(crate) fn snapshot(&self) -> Vec<(&'static str, u64)> {
        let names = TacticsCounters::default()
            .rows()
            .map(|(name, _)| name)
            .into_iter()
            .chain(MOVE_TACTICS_ROWS);
        names
            .zip(&self.slots)
            .map(|(name, slot)| (name, slot.load(Ordering::Relaxed)))
            .collect()
    }
}

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
    /// The largest number of descents ANY one search spent; it must never exceed the
    /// budget, `n_simulations` (or the playout-cap arm's).
    pub(crate) max_sims_per_search: Arc<AtomicU64>,
    /// Searches that ended short of their budget, and the descents short: every descent counts, so both read 0.
    pub(crate) starved_searches: Arc<AtomicU64>,
    pub(crate) starved_descents: Arc<AtomicU64>,
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
    /// The tactics block's rows over every search, and the rows its moves recorded; all zero with tactics off.
    pub(crate) tactics_totals: Arc<TacticsTotals>,
}
