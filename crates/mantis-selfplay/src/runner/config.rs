//! Native `SelfPlayRunnerConfig` — the pyo3-free builder struct; the ctor lives in the bridge.
//!
//! Shapes are spec-derived: there are no caller-supplied `feature_len` / `policy_len`
//! overrides, and a `None encoding_name` with explicit shapes is unrepresentable.
//!
//! **The `Default` impl below is TEST-SCAFFOLDING ONLY**, so in-crate struct-literal callers
//! can write `..Default::default()`. It is NOT a config default-authority: the sole
//! authoritative defaults live in the Python schema.

use mantis_search::SearchKind;

/// Configuration for [`super::SelfPlayRunner`], pyo3-free.
///
/// The `struct_excessive_bools` allow is a permanent KEEP: each flag is an independent lever,
/// not internal state, so folding them to an enum would lose the per-flag ergonomics.
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
    /// The terminal-via-ply-cap outcome (winner=None AND ply >= max_moves), split from
    /// `draw_reward` so organic draws and truncations pay distinct value-head targets.
    pub ply_cap_value: f32,
    pub quiescence_enabled: bool,
    pub quiescence_blend_2: f32,
    pub temp_min: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    /// Which search the workers run (`search.kind`). THE one key: it selects the root
    /// mechanism, the interior selector AND the exported target's semantics together.
    pub search_kind: SearchKind,
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
    /// Registry-form encoding name, resolved to a `&'static RegistrySpec` at
    /// `SelfPlayRunner::new`. `None` = **error**, never a grid/dense default.
    pub encoding_name: Option<String>,
}

/// **TEST-SCAFFOLDING ONLY**, not a config default-authority. Manual rather than derived: a
/// derived `Default` would hand out type-zeros and silently change every caller.
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
            // cosine-OFF
            temp_threshold_compound_moves: 0,
            draw_reward: -0.1,
            ply_cap_value: -0.1,
            quiescence_enabled: true,
            quiescence_blend_2: 0.3,
            // anti-colony constant floor
            temp_min: 0.5,
            c_visit: 50.0,
            c_scale: 1.0,
            search_kind: SearchKind::Puct,
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
            encoding_name: None,
        }
    }
}
