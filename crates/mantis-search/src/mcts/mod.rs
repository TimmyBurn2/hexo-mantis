//! R8 justify: the tree's type, its pool constants and the derived bounds taken over them are
//! one unit — `MAX_ARMED_SIMS` is a function of `MAX_NODES` and `MAX_CHILDREN_PER_NODE`, and a
//! bound living apart from the constants it is derived from is the duplicate-authority class
//! this repo keeps closing.
//! Single-threaded PUCT MCTS tree with a flat pre-allocated node pool.
//!
//! Design notes:
//! - Virtual loss (Phase 2): when `select_one_leaf` descends through a node
//!   it increments `virtual_loss_count`. This decreases the effective Q seen
//!   by subsequent selections on the same path, so batched calls to
//!   `select_leaves(n)` naturally spread across different branches. The
//!   penalty is reversed during `backup`, leaving `w_value` / `n_visits`
//!   correct after the full round-trip.
//! - No per-node heap allocation: all nodes live in `pool: Vec<Node>`.
//! - Negamax value convention: `node.w_value` accumulates values from THAT
//!   node's player-to-move perspective. Backup flips sign when the player
//!   changes (which happens when `parent.moves_remaining == 1`).
//! - PUCT formula (AlphaZero):
//!   `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`
//!   where Q is from the *parent's* perspective.

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
};
pub use gumbel_mctx::MctxRootState;
pub use kind::SearchKind;
pub use node::{CachedPolicy, Node, TTEntry, MAX_NODES, VIRTUAL_LOSS_PENALTY};
pub use selection::{ForcedChildOutOfRange, SelectionDesync};

/// Maximum children created per leaf expansion.
///
/// Caps `expand_and_backup_single`: when the position has more than this many
/// legal moves, only the top-K by NN policy prior are expanded (tie-break by
/// `window_flat_idx` for determinism). Fewer-than-K positions take a fast
/// path with no sort.
///
/// Bound: pool nodes consumed per search ≈ n_simulations × leaf_batch × K.
/// At n_sims=400, leaf_batch=8, K=192 → ~614k slots, fits MAX_NODES=1M with
/// headroom for transposition-table re-expansions and root re-rooting.
///
/// Captures wide-board exploration during early training (legal_moves can
/// balloon past 1k cells once the board has 100+ stones spread out). Can drop
/// to 128 post-training-stabilisation if threat-probe shows no regression.
///
/// Subtree-reuse interaction: K is per-node, not per-tree. Children are
/// stable identity across re-roots since the chosen top-K set is determined
/// by local policy + flat_idx, both invariant under root rotation.
pub const MAX_CHILDREN_PER_NODE: usize = 192;

/// The largest armed sim budget the node pool can serve, DERIVED from the pool's own two
/// constants (AUDIT-1 F-21).
///
/// `finish_expansion` panics when `next_free + n_ch > pool.len()`, and `select_leaves`
/// expands TT-hit leaves WITHOUT counting them against `n` — bounded only by
/// `max_attempts = 4n`. So the worst case for one move is `4 × sims` expansions, each adding
/// up to `MAX_CHILDREN_PER_NODE` children, and the pool overflows from `n_simulations` alone
/// at roughly 1302. The schema declares `n_simulations: Field(ge=1)` with NO ceiling, and
/// `SelfPlayRunner::new` checked only `effective_standard == 0`, so a config could arm a
/// budget that halts the run at the first move that crosses the bound.
///
/// The panic STAYS as the last line — a pool that has actually overflowed has corrupted its
/// own indices and must not continue. This constant is what stops a config reaching it.
pub const MAX_ARMED_SIMS: usize = MAX_NODES / (4 * MAX_CHILDREN_PER_NODE);

/// Root-only child cap under `SearchKind::Gumbel`: the root expands its FULL legal set.
///
/// Gumbel-Top-k is a SAMPLER over the policy — every legal action's logit is perturbed and
/// the top m taken — so drawing it over the top-`MAX_CHILDREN_PER_NODE`-prior children
/// instead is not an approximation of that sampler, it is a different one: an action
/// outside the prior's top K can never be sampled however large its Gumbel draw.
///
/// THE BOUND IS `u16`, not a tuning choice: `Node::n_children` is a `u16`, so a root with
/// more legal moves than this could not record its own child count. The legal set is the
/// union of radius-r balls around every stone minus the occupied cells and grows with the
/// stone count — it is NOT a constant: measured at 355 median and 8 142 maximum at radius
/// 8 — so the cap binds only in a regime no measured game reaches, and when it does bind
/// the omitted-prior telemetry says so rather than the count wrapping.
pub const MAX_ROOT_CHILDREN: usize = u16::MAX as usize;

/// `MAX_ARMED_SIMS` under `SearchKind::Gumbel`, which spends up to `MAX_ROOT_CHILDREN` on
/// its root instead of `MAX_CHILDREN_PER_NODE`.
///
/// A SECOND bound rather than a smaller shared one: folding the root's slots into
/// `MAX_ARMED_SIMS` would tighten the ceiling every PUCT config is validated against, and
/// a PUCT root spends `MAX_CHILDREN_PER_NODE` like any other node, where the original
/// bound is exact.
pub const MAX_ARMED_SIMS_GUMBEL: usize =
    (MAX_NODES - MAX_ROOT_CHILDREN) / (4 * MAX_CHILDREN_PER_NODE);

use fxhash::FxHashMap;
use mantis_core::board::{Board, BOARD_SIZE};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Tree ─────────────────────────────────────────────────────────────────────

pub struct MCTSTree {
    pub pool: Vec<Node>,
    pub(crate) next_free: u32,
    pub root_board: Board,
    pub(crate) c_puct: f32,
    pub(crate) virtual_loss: f32,
    /// KataGo-style dynamic FPU base. FPU for unvisited children is computed as:
    ///   fpu_value = parent_q - fpu_reduction * sqrt(explored_policy_mass)
    /// where explored_policy_mass = sum of prior for all visited children.
    /// Set to 0.0 to disable (classical fixed-FPU behaviour: Q=0 for unvisited).
    pub(crate) fpu_reduction: f32,
    pub selection_overlap_count: u32,
    pub max_depth_observed: u32,
    /// Pending leaves carry the fully-replayed leaf `Board` itself (zobrist +
    /// ply state captured at `select_one_leaf` exit), eliminating the per-leaf
    /// `root_board.clone() + N × apply_move` re-walk previously done inside
    /// `expand_and_backup`.
    pub(crate) pending: Vec<(u32, Board)>,
    pub transposition_table: FxHashMap<u128, TTEntry>,
    /// Enable quiescence value override at leaf nodes.
    /// When true, if the current player has ≥3 winning moves the value is
    /// overridden to +1.0 (forced win); if the opponent has ≥3, to -1.0.
    /// This is a game-specific theorem: each turn places 2 stones, so the
    /// opponent can block at most 2 winning cells per turn.
    pub(crate) quiescence_enabled: bool,
    /// Value blend amount for the 2-winning-moves case (strong but unproven).
    /// The NN value is nudged by ±quiescence_blend_2 toward ±1.0.
    pub(crate) quiescence_blend_2: f32,
    /// When set, `select_one_leaf` skips PUCT at the root and descends
    /// directly to this child pool index. Used by Sequential Halving
    /// (Gumbel MCTS) to force simulations into a specific candidate's subtree.
    pub(crate) forced_root_child: Option<u32>,
    /// Accumulated leaf depth across all simulations since last `new_game()`.
    /// Divide by `sim_count` to get mean depth per simulation.
    pub(crate) depth_accum: u64,
    /// Number of simulations (calls to `select_one_leaf`) since last `new_game()`.
    pub(crate) sim_count: u32,
    /// Cumulative count of quiescence value overrides/blends since last `new_game()`.
    /// Tracks all 4 firing branches: ≥3 current wins (+1.0), ≥3 opponent wins (-1.0),
    /// 2 current wins (blend up), 2 opponent wins (blend down).
    /// Atomic (not Cell) because the FFI layer wraps MCTSTree in a Send+Sync
    /// handle in the bridge crate; Cell is `!Sync` and would break that bound.
    pub quiescence_fire_count: AtomicU64,
    /// Which search this tree runs. Set once per worker (`configure_search`), never
    /// per search.
    pub(crate) kind: SearchKind,
    /// Per-node backed-up-at-expansion value — Mctx's `tree.raw_values`, the network's
    /// own estimate for a node before any child statistic entered it. Held apart from
    /// `w_value / n_visits`, which is the running mean.
    ///
    /// Post-quiescence, i.e. the value the search actually started from: the children's
    /// Q are built from quiescence-corrected values too, so completing them against an
    /// uncorrected node would mix two scales.
    ///
    /// EMPTY under `SearchKind::Puct`, and that is the point — a per-`Node` field would
    /// widen every one of the pool's slots for both kinds and add ~4 MB per worker to the
    /// PUCT arm, which reads no raw value at all. Allocated by `configure_search` only
    /// when the Gumbel kind selects it.
    ///
    /// NOT cleared by `new_game`: `finish_expansion` writes a node's entry as it expands
    /// it, and a completed-Q read only ever reaches an EXPANDED node, so every read is
    /// preceded by its own write. Clearing a million floats per move would be real work
    /// for a value nothing can observe.
    pub(crate) raw_values: Vec<f32>,
    /// `c_visit` / `c_scale` for the interior selector, which runs inside
    /// `select_one_leaf` and has no config in hand. The SAME two config keys the export
    /// path takes as call arguments — one source, read at two depths.
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
            kind: SearchKind::Puct,
            raw_values: Vec::new(),
            q_c_visit: 50.0,
            q_c_scale: 1.0,
            root_children_cap: MAX_CHILDREN_PER_NODE,
        }
    }

    // ── Game lifecycle ────────────────────────────────────────────────────────

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
        self.forced_root_child = None;
        if let Some(root) = self.raw_values.first_mut() {
            *root = 0.0;
        }
        // Clear TT between games — positions don't repeat across games and
        // Vec<f32> policy entries accumulate unboundedly without this.
        self.transposition_table.clear();
    }

    pub fn root_visits(&self) -> u32 {
        self.pool[0].n_visits
    }

    /// First unallocated pool slot. Useful for tests/audits that want to
    /// scan only the live portion of the node pool without iterating
    /// `MAX_NODES` zeroed slots.
    pub fn next_free_slot(&self) -> u32 {
        self.next_free
    }

    /// Set the Gumbel Sequential-Halving forced root child (or clear it). The selfplay
    /// worker-loop driver steers Gumbel by forcing sims into a candidate then clearing;
    /// exposed for cross-crate driving (mantis-selfplay). Pure state set — no search logic.
    ///
    /// # Errors
    /// `ForcedChildOutOfRange` — `child` is not a pool index inside the ROOT's child range.
    ///
    /// AUDIT-1 F-02, second trigger. This used to be an unchecked store, and the value is read
    /// straight into `self.pool[best as usize]` on the next descent: an index at or beyond
    /// `MAX_NODES` index-panics, and any OTHER in-range index descends into a node the root
    /// does not own — including an uninitialised slot, whose `action_idx` of `u32::MAX`
    /// decodes to the axial cell `(32767, 32767)`, which an unbounded board ACCEPTS. That arm
    /// produced no panic and no error: it silently searched a subtree belonging to nothing.
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

    /// Select the search kind once per worker. Pure state set — no search logic.
    /// Survives `new_game`, which resets per-game state and not per-worker config.
    ///
    /// ONE setter and ONE stored kind: every surface that used to branch on a
    /// `gumbel_mcts` bool AND a dialect name AND a `completed_q_values` flag now reads
    /// this field, so the four cannot disagree about which search ran.
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

    /// Return search statistics accumulated since the last `new_game()`.
    ///
    /// Returns `(mean_depth, root_concentration)`:
    /// - `mean_depth`: average depth descended per simulation (leaf depth)
    /// - `root_concentration`: max child visits / root total visits ∈ [0.0, 1.0]
    ///
    /// Both values are 0.0 when no simulations have been run.
    /// This is a once-per-search aggregation — NOT called from the inner sim loop.
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
    /// `n_actions` is the 19-window-with-pass-slot stride: the bench harness
    /// (`MCTSTree::new()` + `Board::new()`) carries plain geometry and no
    /// encoding, so the stride is the canonical `BOARD_SIZE² + 1`. This is the
    /// sole construction shape the bench ever executes.
    pub fn run_simulations_cpu_only(&mut self, n: usize) {
        let n_actions = BOARD_SIZE * BOARD_SIZE + 1;
        let uniform_prior = 1.0 / n_actions as f32;
        let uniform_policy = vec![uniform_prior; n_actions];
        // Bench-fidelity: hoist policies/values slot vecs ONCE outside the outer
        // loop and resize per iteration. select_leaves(1) returns at most one
        // board, so capacity 1 suffices. Production self-play does NOT enter
        // this path. The change removes bench-loop allocation noise so the MCTS
        // sim/s metric better reflects algorithm cost. Bit-equivalent algorithm.
        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(1);
        let mut values: Vec<f32> = Vec::with_capacity(1);
        for _ in 0..n {
            // A desync ENDS the bench loop rather than being retried: the bench measures a
            // healthy search, and a tree that has desynchronised from its board is not one
            // (AUDIT-1 F-02). `run_simulations_cpu_only` has no production caller, so this
            // is the one place the error is legitimately dropped instead of propagated.
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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

// Completed-Q golden byte-identity harness (completed-Q sites S1/S2/S3/S4).
#[cfg(test)]
mod golden_tests;

// Mctx parity for the completed-Q math (GUMBEL-REPAIR-1).
#[cfg(test)]
mod parity_tests;
