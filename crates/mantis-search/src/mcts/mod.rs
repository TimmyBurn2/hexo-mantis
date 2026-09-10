//! R8 justify: the tree's type, its pool constants and the derived bounds taken over them are one
//! unit — `MAX_ARMED_SIMS` is a function of `MAX_NODES` and `MAX_CHILDREN_PER_NODE`, and a bound
//! living apart from the constants it is derived from is a duplicate authority.
//!
//! Single-threaded PUCT MCTS tree with a flat pre-allocated node pool.
//!
//! - Virtual loss: `select_one_leaf` increments `virtual_loss_count` as it descends, decreasing
//!   the effective Q seen by later selections on the same path so batched `select_leaves(n)`
//!   spreads across branches; the penalty is reversed during `backup`.
//! - No per-node heap allocation: all nodes live in `pool: Vec<Node>`.
//! - Negamax value convention: `node.w_value` accumulates values from THAT node's
//!   player-to-move perspective, and backup flips sign when the player changes.
//! - PUCT: `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`, Q from the parent's perspective.

mod backup;
mod completed_q;
pub mod dirichlet;
pub mod gumbel_mctx;
pub mod kind;
pub mod node;
pub mod policy;
mod selection;
pub mod seq_halving;

pub use backup::{
    omitted_prior_stats, pool_overflow_count, take_omitted_prior_stats, take_pool_overflow_count,
    OmittedPriorStats,
};
pub use gumbel_mctx::MctxRootState;
pub use kind::SearchKind;
pub use node::{CachedPolicy, Node, TTEntry, MAX_NODES, VIRTUAL_LOSS_PENALTY};
pub use selection::{ForcedChildOutOfRange, SelectionDesync};

/// Maximum children created per leaf expansion: past this many legal moves only the top-K by NN
/// policy prior are expanded, tie-broken by `window_flat_idx` for determinism. What K costs is the
/// armed-sims ceiling, which `MAX_ARMED_SIMS` derives from this constant and `MAX_NODES`. The
/// interior legal set at radius 8 measures a median of 355 and a maximum of 8142, so the
/// omitted-prior telemetry is what says how often the cap still binds.
pub const MAX_CHILDREN_PER_NODE: usize = 1024;

/// The largest armed sim budget the node pool can serve, DERIVED from the pool's own two
/// constants: `select_leaves` expands TT-hit leaves without counting them against `n`, bounded
/// only by `max_attempts = 4n`, so one move's worst case is `4 × sims` expansions of up to
/// `MAX_CHILDREN_PER_NODE` children each. `finish_expansion`'s panic STAYS as the last line — an
/// overflowed pool has corrupted its own indices — and this constant stops a config reaching it.
pub const MAX_ARMED_SIMS: usize = MAX_NODES / (4 * MAX_CHILDREN_PER_NODE);

/// Root-only child cap under `SearchKind::Gumbel`, where the root expands its FULL legal set:
/// Gumbel-Top-k is a SAMPLER over the policy, so drawing it over the top-K-prior children instead
/// is a different sampler, not an approximation. The bound is `u16` because `Node::n_children` is,
/// and the measured legal set (355 median, 8 142 maximum at radius 8) reaches it in no real game.
pub const MAX_ROOT_CHILDREN: usize = u16::MAX as usize;

/// `MAX_ARMED_SIMS` under `SearchKind::Gumbel`, which spends up to `MAX_ROOT_CHILDREN` on its root.
/// A SECOND bound rather than a smaller shared one: folding the root's slots into `MAX_ARMED_SIMS`
/// would tighten the ceiling every PUCT config is validated against, where it is exact.
pub const MAX_ARMED_SIMS_GUMBEL: usize =
    (MAX_NODES - MAX_ROOT_CHILDREN) / (4 * MAX_CHILDREN_PER_NODE);

use fxhash::FxHashMap;
use mantis_core::board::{Board, BOARD_SIZE};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct MCTSTree {
    pub pool: Vec<Node>,
    pub(crate) next_free: u32,
    pub root_board: Board,
    pub(crate) c_puct: f32,
    pub(crate) virtual_loss: f32,
    /// KataGo-style dynamic FPU base: `fpu_value = parent_q - fpu_reduction *
    /// sqrt(explored_policy_mass)`, where the mass is the prior summed over visited children.
    /// 0.0 disables it (classical fixed FPU, Q=0 for unvisited).
    pub(crate) fpu_reduction: f32,
    pub selection_overlap_count: u32,
    pub max_depth_observed: u32,
    /// Pending leaves carry the fully-replayed leaf `Board` itself, captured at `select_one_leaf`
    /// exit, eliminating the per-leaf `root_board.clone() + N × apply_move` re-walk.
    pub(crate) pending: Vec<(u32, Board)>,
    pub transposition_table: FxHashMap<u128, TTEntry>,
    /// Enable quiescence value override at leaf nodes: ≥3 winning moves for the current player
    /// overrides the value to ±1.0. A game-specific theorem — each turn places 2 stones, so the
    /// opponent can block at most 2 winning cells per turn.
    pub(crate) quiescence_enabled: bool,
    /// Value blend amount for the 2-winning-moves case (strong but unproven).
    /// The NN value is nudged by ±quiescence_blend_2 toward ±1.0.
    pub(crate) quiescence_blend_2: f32,
    /// When set, `select_one_leaf` skips PUCT at the root and descends directly to this child pool
    /// index — Sequential Halving forcing simulations into one candidate's subtree.
    pub(crate) forced_root_child: Option<u32>,
    /// Accumulated leaf depth across all simulations since `new_game()`; divide by `sim_count`.
    pub(crate) depth_accum: u64,
    /// Number of simulations (calls to `select_one_leaf`) since last `new_game()`.
    pub(crate) sim_count: u32,
    /// Cumulative count of quiescence overrides/blends since `new_game()`, over all four branches.
    /// Atomic rather than `Cell`, because the bridge wraps the tree in a Send+Sync handle.
    pub quiescence_fire_count: AtomicU64,
    /// THIS SEARCH's omitted-prior counters, `(mass_micros, expansions_that_omitted,
    /// total_expansions)`.
    ///
    /// PER-SEARCH deliberately: the same quantities live in process-wide statics for the run-wide
    /// aggregate, so a caller bracketing ONE search used to have any concurrent search land inside
    /// its bracket. `AtomicU64` because the bridge's Send+Sync handle makes a `Cell` unusable.
    pub omitted_prior: OmittedPriorStats,
    /// Which search this tree runs. Set once per worker (`configure_search`), never
    /// per search.
    pub(crate) kind: SearchKind,
    /// Per-node backed-up-at-expansion value — the network's own estimate for a node before any
    /// child statistic entered it, held apart from the running mean `w_value / n_visits`.
    ///
    /// Post-quiescence, i.e. the value the search actually started from, since the children's Q
    /// are built from quiescence-corrected values too. EMPTY under `SearchKind::Puct`: a
    /// per-`Node` field would widen every pool slot for both kinds and add ~4 MB per worker to an
    /// arm that reads no raw value. NOT cleared by `new_game` — `finish_expansion` writes a node's
    /// entry as it expands it, so every read is preceded by its own write.
    pub(crate) raw_values: Vec<f32>,
    /// `c_visit` / `c_scale` for the interior selector, which runs inside `select_one_leaf` and
    /// has no config in hand — the SAME two config keys the export path takes as arguments.
    pub(crate) q_c_visit: f32,
    pub(crate) q_c_scale: f32,
    /// Children the ROOT may expand, as distinct from every other node's
    /// `MAX_CHILDREN_PER_NODE`. Set by `configure_search`; `MAX_CHILDREN_PER_NODE`
    /// under `SearchKind::Puct`, whose root is an ordinary PUCT node.
    pub(crate) root_children_cap: usize,
}

impl MCTSTree {
    /// Convenience constructor used by benches + tests; production hot path
    /// constructs via `new_full` directly (see the self-play worker init).
    pub fn new(c_puct: f32) -> Self {
        MCTSTree::new_full(c_puct, VIRTUAL_LOSS_PENALTY, 0.0)
    }

    pub fn new_full(c_puct: f32, virtual_loss: f32, fpu_reduction: f32) -> Self {
        let pool = vec![Node::uninit(); MAX_NODES];
        MCTSTree {
            pool,
            next_free: 1,
            root_board: Board::new(),
            c_puct,
            virtual_loss,
            fpu_reduction,
            selection_overlap_count: 0,
            max_depth_observed: 0,
            depth_accum: 0,
            sim_count: 0,
            pending: Vec::new(),
            transposition_table: FxHashMap::default(),
            quiescence_enabled: true,
            quiescence_blend_2: 0.3,
            forced_root_child: None,
            quiescence_fire_count: AtomicU64::new(0),
            omitted_prior: OmittedPriorStats::default(),
            kind: SearchKind::Puct,
            raw_values: Vec::new(),
            q_c_visit: 50.0,
            q_c_scale: 1.0,
            root_children_cap: MAX_CHILDREN_PER_NODE,
        }
    }

    pub fn new_game(&mut self, board: Board) {
        let mr = board.moves_remaining;
        self.root_board = board;
        self.pool[0] = Node::uninit();
        self.pool[0].moves_remaining = mr;
        self.next_free = 1;
        self.pending.clear();
        self.selection_overlap_count = 0;
        self.max_depth_observed = 0;
        self.depth_accum = 0;
        self.sim_count = 0;
        self.quiescence_fire_count.store(0, Ordering::Relaxed);
        self.omitted_prior.reset();
        self.forced_root_child = None;
        if let Some(root) = self.raw_values.first_mut() {
            *root = 0.0;
        }
        // Clear TT between games: positions do not repeat across games, and the `Vec<f32>` policy
        // entries accumulate unboundedly without this.
        self.transposition_table.clear();
    }

    pub fn root_visits(&self) -> u32 {
        self.pool[0].n_visits
    }

    /// First unallocated pool slot, so a test or audit can scan only the live portion of the pool.
    pub fn next_free_slot(&self) -> u32 {
        self.next_free
    }

    /// Set the Gumbel Sequential-Halving forced root child, or clear it. Pure state set — no
    /// search logic.
    ///
    /// # Errors
    /// `ForcedChildOutOfRange` — `child` is not a pool index inside the ROOT's child range.
    ///
    /// This used to be an unchecked store, and the value is read straight into
    /// `self.pool[best as usize]` on the next descent: an index past `MAX_NODES` index-panics,
    /// and any other in-range index descends into a node the root does not own — including an
    /// uninitialised slot, whose `action_idx` decodes to a cell an unbounded board ACCEPTS. That
    /// arm produced no panic and no error: it silently searched a subtree belonging to nothing.
    pub fn set_forced_root_child(
        &mut self,
        child: Option<u32>,
    ) -> Result<(), ForcedChildOutOfRange> {
        if let Some(idx) = child {
            let root = &self.pool[0];
            let first = root.first_child;
            let end = first.saturating_add(u32::from(root.n_children));
            if root.n_children == 0 || idx < first || idx >= end {
                return Err(ForcedChildOutOfRange {
                    child: idx,
                    first_child: first,
                    n_children: root.n_children,
                });
            }
        }
        self.forced_root_child = child;
        Ok(())
    }

    /// Configure quiescence once per worker from run config (mirrors the old worker's
    /// post-construction set). Pure state set — no search logic.
    pub fn configure_quiescence(&mut self, enabled: bool, blend_2: f32) {
        self.quiescence_enabled = enabled;
        self.quiescence_blend_2 = blend_2;
    }

    /// Select the search kind once per worker. Pure state set, surviving `new_game`.
    ///
    /// ONE setter and ONE stored kind: every surface that used to branch on a `gumbel_mcts` bool
    /// AND a dialect name AND a `completed_q_values` flag now reads this field, so the four
    /// cannot disagree about which search ran.
    pub fn configure_search(&mut self, kind: SearchKind, c_visit: f32, c_scale: f32) {
        self.kind = kind;
        self.q_c_visit = c_visit;
        self.q_c_scale = c_scale;
        self.root_children_cap = match kind {
            SearchKind::Puct => MAX_CHILDREN_PER_NODE,
            SearchKind::Gumbel => MAX_ROOT_CHILDREN,
        };
        // The per-node raw values exist only for the kind that reads them.
        self.raw_values = match kind {
            SearchKind::Puct => Vec::new(),
            SearchKind::Gumbel => vec![0.0; MAX_NODES],
        };
    }

    /// Read-and-reset THIS tree's omitted-prior counters — the per-search measurement bracket.
    /// The process-wide totals are NOT reset by it: those are the run-wide aggregate the bridge
    /// publishes, and a per-search read must not silently zero a run's telemetry.
    pub fn take_omitted_prior(&self) -> (u64, u64, u64) {
        self.omitted_prior.take()
    }

    /// The same three counters without resetting them.
    #[must_use]
    pub fn omitted_prior_stats(&self) -> (u64, u64, u64) {
        self.omitted_prior.read()
    }

    /// Children the ROOT may expand under this tree's kind.
    #[must_use]
    pub fn root_children_cap(&self) -> usize {
        self.root_children_cap
    }

    /// The search this tree runs.
    #[must_use]
    pub fn search_kind(&self) -> SearchKind {
        self.kind
    }

    /// The value backed up at root expansion (Mctx's `raw_values[root]`), or 0.0 under a
    /// kind that keeps none.
    #[must_use]
    pub fn root_raw_value(&self) -> f32 {
        self.raw_values.first().copied().unwrap_or(0.0)
    }

    pub fn reset(&mut self) {
        let mr = self.pool[0].moves_remaining;
        let board = self.root_board.clone();
        self.new_game(board);
        self.pool[0].moves_remaining = mr;
    }

    /// Value estimate at root from perspective of player to move.
    /// Returns Q = w_value / n_visits, or 0.0 if no visits.
    pub fn root_value(&self) -> f32 {
        let root = &self.pool[0];
        if root.n_visits == 0 {
            0.0
        } else {
            root.w_value / root.n_visits as f32
        }
    }

    pub fn root_n_children(&self) -> usize {
        if self.pool[0].is_expanded() {
            self.pool[0].n_children as usize
        } else {
            0
        }
    }

    /// Return `(mean_depth, root_concentration)` accumulated since the last `new_game()`: average
    /// depth descended per simulation, and max child visits over root total visits in `[0, 1]`.
    /// Both are 0.0 when no simulations have run. A once-per-search aggregation, never called from
    /// the inner sim loop.
    pub fn last_search_stats(&self) -> (f32, f32) {
        let mean_depth = if self.sim_count > 0 {
            self.depth_accum as f32 / self.sim_count as f32
        } else {
            0.0
        };
        let root_conc = {
            let root = &self.pool[0];
            let total = root.n_visits;
            if total == 0 || !root.is_expanded() {
                0.0
            } else {
                let first = root.first_child as usize;
                let n = root.n_children as usize;
                let max_v = (first..first + n)
                    .map(|i| self.pool[i].n_visits)
                    .max()
                    .unwrap_or(0);
                max_v as f32 / total as f32
            }
        };
        (mean_depth, root_conc)
    }

    /// Run `n` simulations using uniform priors (no NN). For benchmarking.
    ///
    /// `n_actions` is the 19-window-with-pass-slot stride: the bench harness carries plain
    /// geometry and no encoding, so the stride is the canonical `BOARD_SIZE² + 1`.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform_prior = 1.0 / n_actions as f32;
        let uniform_policy = vec![uniform_prior; n_actions];
        // Bench-fidelity: hoist the policies/values slot vecs ONCE outside the outer loop and
        // resize per iteration, removing bench-loop allocation noise. Bit-equivalent algorithm,
        // and production self-play does NOT enter this path.
        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(1);
        let mut values: Vec<f32> = Vec::with_capacity(1);
        for _ in 0..n {
            // A desync ENDS the bench loop rather than being retried: the bench measures a healthy
            // search. `run_simulations_cpu_only` has no production caller, so this is the one place
            // the error is legitimately dropped instead of propagated.
            let Ok(boards) = self.select_leaves(1) else {
                return;
            };
            if boards.is_empty() {
                continue;
            }
            policies.clear();
            for _ in 0..boards.len() {
                policies.push(uniform_policy.clone());
            }
            values.clear();
            values.resize(boards.len(), 0.0);
            self.expand_and_backup(&policies, &values);
        }
    }
}

#[cfg(test)]
mod tests;

// Completed-Q golden byte-identity harness (completed-Q sites S1/S2/S3/S4).
#[cfg(test)]
mod golden_tests;

// Mctx parity for the completed-Q math (GUMBEL-REPAIR-1).
#[cfg(test)]
mod parity_tests;
