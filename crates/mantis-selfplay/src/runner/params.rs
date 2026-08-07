//! `WorkerParams` + themed sub-flag bundles + the per-worker `WorkerGeometry`
//! (WP6 D1/D2), ported from the frozen `worker_loop/params.rs` + the
//! representation-resolution head of `worker_loop/mod.rs:130-193`.
//!
//! The 4 bundles (`WorkerStats`/`WorkerAtomics`/`WorkerChannels`/`WorkerParams`)
//! are cloned once per worker spawn and destructured at `game::run_worker_thread`
//! entry, so the per-sim hot path sees local scalars, never a `&RegistrySpec`
//! field access (`feedback_registryspec_by_ref_in_hotpath.md`).
//!
//! KILLs vs the frozen source (D7/D10):
//!   - `MoveConstraintFlags` collapses to a `zoi_enabled`-only bundle — the
//!     radius-jitter field is NEVER authored (D7);
//!   - `WorkerParams`'s interior-selection field is NOT authored (D10 —
//!     `MCTSTree` has no such field new-side).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use mantis_encoding::{PolicyPool, RegistrySpec, Representation};

use crate::queues::{DenseQueue, GraphQueue};
use crate::replay::hexg::GraphRecord;

use super::{GameResultRow, WorkerResultRow};

/// Per-worker geometry scalars resolved ONCE from the `RegistrySpec` at spawn
/// time (D2). `Copy` (~32 B); passed by value to `run_worker_thread` and
/// destructured into local scalar bindings at fn entry.
///
/// `is_graph` / `legal_set` are produced FROM the closed-match arms in
/// [`resolve_geometry`], NOT via the `spec.is_graph()` bool method (which carries
/// an implicit wildcard → dense-by-default for a future `Representation` variant,
/// the LAW-11 backdoor P-12 closes).
#[derive(Clone, Copy)]
pub struct WorkerGeometry {
    pub n_cells: usize,
    pub kept_planes: &'static [usize],
    pub policy_stride: usize,
    pub agg_trunk_sz: i32,
    pub has_pass_slot: bool,
    /// §D-MULTICLUSTER-S0: ragged legal-set MCTS prior / improved-policy target
    /// (no off-window drop) instead of the dense path.
    pub legal_set: bool,
    /// True when `spec.representation == Graph`. Gates the record/finalize
    /// dispatch between the dense K-cluster path and the whole-board HEXG path.
    pub is_graph: bool,
}

/// Resolve the per-worker geometry from a resolved `&'static RegistrySpec` (D2).
///
/// The kind dispatch is a **closed** `match spec.representation` with NO `_ =>`
/// arm: a new `Representation` variant fails WP6 compilation loudly. There is no
/// `None → v6` fallback — an absent spec is rejected as an error BEFORE this
/// point (`SelfPlayRunner::new`, LAW-11). `is_graph` / `legal_set` are set from
/// the arms (byte-identical to the frozen `worker_loop/mod.rs:168/176`).
#[must_use]
pub fn resolve_geometry(spec: &'static RegistrySpec) -> WorkerGeometry {
    match spec.representation {
        Representation::Grid => WorkerGeometry {
            n_cells: spec.n_cells(),
            kept_planes: spec.kept_plane_indices,
            policy_stride: spec.policy_stride(),
            agg_trunk_sz: spec.trunk_size as i32,
            has_pass_slot: spec.has_pass_slot,
            // Frozen: `legal_set = is_graph() || matches!(policy_pool, LegalSetScatterMax)`;
            // for Grid `is_graph()` is false, so it collapses to the policy_pool test.
            legal_set: matches!(spec.policy_pool, PolicyPool::LegalSetScatterMax),
            is_graph: false,
        },
        Representation::Graph => WorkerGeometry {
            n_cells: spec.n_cells(),
            kept_planes: spec.kept_plane_indices,
            policy_stride: spec.policy_stride(),
            agg_trunk_sz: spec.trunk_size as i32,
            has_pass_slot: spec.has_pass_slot,
            // Frozen: for Graph `is_graph()` is true, so `legal_set` is
            // unconditionally true.
            legal_set: true,
            is_graph: true,
        },
    }
}

#[derive(Clone)]
pub(crate) struct SearchFlags {
    pub(crate) quiescence_enabled: bool,
    pub(crate) completed_q_values: bool,
    pub(crate) gumbel_mcts: bool,
}

#[derive(Clone)]
pub(crate) struct ExplorationFlags {
    pub(crate) dirichlet_enabled: bool,
    pub(crate) selfplay_rotation_enabled: bool,
}

/// Move-constraint flags. Collapsed to `zoi_enabled` only — the frozen
/// radius-jitter sibling is KILLED (D7) and NEVER authored.
#[derive(Clone)]
pub(crate) struct MoveConstraintFlags {
    pub(crate) zoi_enabled: bool,
}

/// O1 forced-win → one-hot POLICY target knobs (default OFF).
#[derive(Clone, Copy)]
pub(crate) struct ForcedWinPolicy {
    pub(crate) enabled: bool,
    pub(crate) depth: u8,
    pub(crate) weight: f32,
}

/// D-WS3 L1 native solver-in-loop SOFT visit-injection knobs (default OFF).
#[derive(Clone, Copy)]
pub(crate) struct SolverInLoop {
    pub(crate) enabled: bool,
    pub(crate) depth: u32,
    pub(crate) node_budget: u64,
    pub(crate) neighbor_dist: i32,
    pub(crate) visit_weight: f32,
}

/// D-WS3V3 trap-corpus START-POSITION seeding bundle. `seed_fraction == 0.0` OR
/// an empty corpus keeps the rng stream byte-identical (no rng draw).
#[derive(Clone)]
pub(crate) struct SeedCorpus {
    pub(crate) corpus: Arc<Vec<Vec<(i32, i32)>>>,
    pub(crate) seed_fraction: f32,
}

/// Per-worker channel/queue bundle — the inference-queue producer handles the
/// worker submits to, plus the shared result queues it drains into. Replaces the
/// frozen `WorkerChannels` (which bundled the single `InferenceBatcher`); the
/// dense/graph inference queues are now the two disjoint pure-Rust handles (D4).
#[derive(Clone)]
pub(crate) struct WorkerChannels {
    pub(crate) dense_queue: DenseQueue,
    pub(crate) graph_queue: GraphQueue,
    pub(crate) results_queue: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    pub(crate) recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,
    /// Parallel graph-position results queue; only ever locked on the `is_graph`
    /// finalize branch (idle for every grid spec).
    pub(crate) graph_results_queue: Arc<Mutex<VecDeque<GraphRecord>>>,
}

#[derive(Clone)]
pub(crate) struct WorkerParams {
    pub(crate) max_moves: usize,
    pub(crate) leaf_batch_size: usize,
    pub(crate) c_puct: f32,
    pub(crate) fpu_reduction: f32,
    pub(crate) quiescence_blend_2: f32,
    pub(crate) fast_prob: f32,
    pub(crate) fast_sims: usize,
    pub(crate) standard_sims: usize,
    pub(crate) temp_threshold: usize,
    pub(crate) temp_min: f32,
    pub(crate) draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (distinct from `draw_reward`).
    pub(crate) ply_cap_value: f32,
    pub(crate) zoi_lookback: usize,
    pub(crate) zoi_margin: i32,
    pub(crate) c_visit: f32,
    pub(crate) c_scale: f32,
    pub(crate) gumbel_m: usize,
    pub(crate) gumbel_explore_moves: usize,
    pub(crate) dirichlet_alpha: f32,
    pub(crate) dirichlet_epsilon: f32,
    pub(crate) results_queue_cap: usize,
    pub(crate) full_search_prob: f32,
    pub(crate) n_sims_quick: usize,
    pub(crate) n_sims_full: usize,
    pub(crate) random_opening_plies: u32,
    /// DERIVED HEXG visit-slot capacity (R255/ADJ-D34) — composed once in
    /// `SelfPlayRunner::new`'s graph arm; `None` on grid runs (no visit slot),
    /// never a default.
    pub(crate) visit_capacity: Option<usize>,
    /// Resolved (never `None` — LAW-11) encoding spec, used by the per-game board
    /// construction (`init_per_game_board`, R2).
    pub(crate) registry_spec: &'static RegistrySpec,
    pub(crate) search_flags: SearchFlags,
    pub(crate) exploration_flags: ExplorationFlags,
    pub(crate) move_constraint_flags: MoveConstraintFlags,
    pub(crate) forced_win_policy: ForcedWinPolicy,
    pub(crate) solver_in_loop: SolverInLoop,
    pub(crate) seed_corpus: SeedCorpus,
}
