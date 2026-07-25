//! Native `SelfPlayRunnerConfig` — the pyo3-STRIPPED builder struct ported from
//! the frozen `game_runner/config.rs` (WP6 D7).
//!
//! Differences from the frozen source (all sanctioned by the zero-behaviour
//! contract, DESIGN §(d)):
//!   - the `#[pyclass]` / `#[pymethods]` / `#[pyo3(signature=…)]` 38-positional
//!     ctor is DROPPED (→ WP7 bridge);
//!   - the per-game radius-jitter knob is NOT authored (D7 KILL — dead for every
//!     registry spec; the None arm it guarded is itself killed);
//!   - the interior-selection knob is NOT authored (D10 KILL — the WP4-removed
//!     interior-selection type has no field new-side);
//!   - `feature_len` / `policy_len` caller-supplied shape overrides are DROPPED
//!     (C-1 / D2): shapes are spec-derived, and a `None encoding_name + explicit
//!     shapes` construct is UNREPRESENTABLE (LAW-11 — shapes do not tell Grid
//!     vs Graph; an absent identity key is an error even with shapes).
//!
//! **The `Default` impl below is TEST-SCAFFOLDING ONLY.** It exists so Rust
//! struct-literal callers (in-crate tests) can write
//! `SelfPlayRunnerConfig { encoding_name: Some("v6".into()), ..Default::default() }`.
//! It is NOT a config default-authority: the SOLE authoritative defaults live in
//! the WP8 Python schema (`extra="forbid"`, minted, R1 — code-side config
//! defaults are forbidden). P-01 pins that no field is dropped/mismerged and
//! that the radius-jitter field is absent.

/// Configuration for [`super::SelfPlayRunner`] — native (pyo3-free) fold of the
/// pre-cycle-3 kwarg constructor surface, MINUS the killed knobs (D7/D10) and the
/// caller-supplied shape overrides (C-1).
///
/// The bool fields mirror the user-tunable kwarg surface; the
/// `struct_excessive_bools` allow is a permanent KEEP (each flag is an
/// independent lever, not internal state — folding to an enum would lose the
/// per-flag ergonomics).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub struct SelfPlayRunnerConfig {
    pub n_workers: usize,
    pub max_moves_per_game: usize,
    pub n_simulations: usize,
    pub leaf_batch_size: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub fast_prob: f32,
    pub fast_sims: usize,
    pub standard_sims: usize,
    pub temp_threshold_compound_moves: usize,
    pub draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (winner=None AND ply ≥ max_moves).
    /// Split from `draw_reward` so organic draws and ply-cap truncations pay
    /// distinct value-head targets. Default `-0.1` matches `draw_reward` for
    /// back-compat (pre-§178 callers see identical outcomes).
    pub ply_cap_value: f32,
    pub quiescence_enabled: bool,
    pub quiescence_blend_2: f32,
    pub temp_min: f32,
    pub zoi_enabled: bool,
    pub zoi_lookback: usize,
    pub zoi_margin: i32,
    pub completed_q_values: bool,
    pub c_visit: f32,
    pub c_scale: f32,
    pub gumbel_mcts: bool,
    pub gumbel_m: usize,
    pub gumbel_explore_moves: usize,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub dirichlet_enabled: bool,
    pub results_queue_cap: usize,
    pub full_search_prob: f32,
    pub n_sims_quick: usize,
    pub n_sims_full: usize,
    pub random_opening_plies: u32,
    pub selfplay_rotation_enabled: bool,
    /// Registry-form encoding name (e.g. `"v6"`, `"gnn_axis_v1"`). Resolved to a
    /// `&'static RegistrySpec` at `SelfPlayRunner::new` via
    /// `mantis_encoding::lookup`. `None` = **error** (LAW-11 — absent identity
    /// key is never a grid/dense default; the frozen `None → v6` fallback is
    /// killed, D2).
    pub encoding_name: Option<String>,
    /// Inference-pool sizing hint — consumed by the WP7 producer face (the NN
    /// pool); the pure-Rust queues do not size a pool.
    pub inference_pool_size: Option<usize>,
    /// O1 forced-win → one-hot POLICY target knobs (default OFF).
    pub forced_win_policy_enabled: bool,
    pub forced_win_policy_depth: u8,
    pub forced_win_policy_weight: f32,
    /// D-WS3 L1 solver-in-loop SOFT visit-injection knobs (default OFF —
    /// `solver_enabled=false` makes the per-move hook a no-op).
    pub solver_enabled: bool,
    pub solver_depth: u32,
    pub solver_node_budget: u64,
    pub solver_neighbor_dist: i32,
    pub solver_visit_weight: f32,
    /// D-WS3V3 trap-corpus START-POSITION seeding (default OFF / empty).
    pub seed_fraction: f32,
    /// Move-prefix corpus (list-of-list-of-`(q, r)`). `None` / empty = no
    /// seeding. Dry-replay validated once at `SelfPlayRunner::new`.
    pub seed_corpus: Option<Vec<Vec<(i32, i32)>>>,
}

/// **TEST-SCAFFOLDING ONLY** (see the module doc). NOT a config default-authority
/// — the authoritative defaults live in the WP8 Python schema (R1). A *derived*
/// `Default` would give type-zeros (silently changing every caller), so this is a
/// manual impl mirroring the frozen semantic defaults MINUS the killed knobs.
impl Default for SelfPlayRunnerConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            max_moves_per_game: 128,
            n_simulations: 50,
            leaf_batch_size: 8,
            c_puct: 1.5,
            fpu_reduction: 0.25,
            fast_prob: 0.0,
            fast_sims: 50,
            standard_sims: 0,
            // D-TEMPDECAY C1: cosine-OFF default (was 15).
            temp_threshold_compound_moves: 0,
            draw_reward: -0.1,
            ply_cap_value: -0.1,
            quiescence_enabled: true,
            quiescence_blend_2: 0.3,
            // D-TEMPDECAY C1: anti-colony constant floor (was 0.05).
            temp_min: 0.5,
            zoi_enabled: false,
            zoi_lookback: 16,
            zoi_margin: 5,
            completed_q_values: false,
            c_visit: 50.0,
            c_scale: 1.0,
            gumbel_mcts: false,
            gumbel_m: 16,
            gumbel_explore_moves: 10,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            dirichlet_enabled: true,
            results_queue_cap: 10_000,
            full_search_prob: 0.0,
            n_sims_quick: 0,
            n_sims_full: 0,
            random_opening_plies: 0,
            selfplay_rotation_enabled: false,
            encoding_name: None,
            inference_pool_size: None,
            // O1 forced-win one-hot POLICY target — OFF by default.
            forced_win_policy_enabled: false,
            forced_win_policy_depth: 2,
            forced_win_policy_weight: 1.0,
            // D-WS3 L1 solver-in-loop — OFF by default (byte-identical self-play).
            solver_enabled: false,
            solver_depth: 16,
            solver_node_budget: 50_000,
            solver_neighbor_dist: 2,
            solver_visit_weight: 0.3,
            // D-WS3V3 start-position seeding — OFF by default.
            seed_fraction: 0.0,
            seed_corpus: None,
        }
    }
}
