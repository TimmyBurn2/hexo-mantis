// Exceeds the 300-line soft cap (R8): the `SelfPlayRunnerConfig` builder pyclass
// (36-kwarg ctor mapping the NEW field set + the 10 post-ctor get/set attrs) and
// the `SelfPlayRunner` pyclass facade (collect_data 10-numpy marshal, the ~20
// counter getters incl. the 4 bridge-derived means, batcher, Drop=stop) are one
// line-auditable port unit against the frozen `game_runner/{config,mod}.rs`.
//! `SelfPlayRunnerConfig` + `SelfPlayRunner` pyclass surface over WP6
//! `mantis_selfplay::runner` (via the SEAM drain/snapshot faces).
//!
//! Both classes are Arc-based `Send + Sync` (NO `unsendable`): `SelfPlayRunner`
//! is `Send + Sync` and the bridge holds `Arc<SelfPlayRunner>`; workers own their
//! `Board` in Rust, never via Python.
//!
//! The runner's ~20 counter getters read a one-instant `RunnerStatsSnapshot`
//! (RAW atomics from the SEAM) and the bridge DERIVES the 4 means per DESIGN §c.6
//! (fixed-point ÷(count × 1e6); the root-concentration mean is f32 arithmetic, the
//! other three f64). ADJ-D32 / R249: the f64 derivation returns `Option<f32>` and
//! yields `None` at zero count — a mean over zero samples is not a measurement, and
//! the two cluster getters surface that `None` to Python verbatim. F-42:
//! `module = "mantis._engine"`.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_search::SearchKind;
use mantis_selfplay::runner::config::SelfPlayRunnerConfig;
use mantis_selfplay::runner::{GameResultRow, RunnerStatsSnapshot, SelfPlayRunner};

use crate::inference::PyInferenceBatcher;

/// Per-row tuple returned by `collect_graph_data`: the first NINE fields are
/// `HexgBuffer.push_graph_position`'s positional signature verbatim, and the tenth is the
/// runner-assigned `game_id` (R345(b)(6)), which that signature takes as a KEYWORD. The
/// split is what lets the Python push forward the nine unchanged and pass the tenth by name.
type GraphRecordRow = (
    Vec<(i16, i16, i8)>,
    Vec<(i16, i16, f32)>,
    i8,
    u8,
    u16,
    bool,
    f32,
    bool,
    u16,
    i64,
);

/// Derived fixed-point mean in f64 arithmetic (`mcts_mean_depth` /
/// `cluster_*_mean`): `accum / (count × 1e6)`, `count == 0 → None`. The `*_accum`
/// atomics are ×1_000_000 integers, so the ×1e6 divisor is load-bearing.
///
/// ADJ-D32 / R249: the zero-count arm used to return `0.0`. On the graph arm
/// `cluster_variance_samples` is permanently 0 — `search_drive.rs` returns into
/// `infer_and_expand_graph` before any variance code runs, and the atomics are not
/// even passed to it — so that guard published a hard `0.0` forever and a reader
/// could not tell a settled cluster ensemble from an instrument that never fired.
/// `None` is the only honest reading of a mean over zero samples; the caller decides
/// whether its field is publishable, and the emitter drops it.
fn derived_mean_f64(accum: u64, count: u64) -> Option<f32> {
    if count == 0 {
        None
    } else {
        Some((accum as f64 / (count as f64 * 1_000_000.0)) as f32)
    }
}

/// The two cluster-variance means from ONE snapshot, in field order:
/// `(value spread, top-1 policy disagreement)`.
///
/// Extracted from the two getters solely so the snapshot→getter mapping can be pinned
/// at all. The getters read `self.snapshot()`, whose atomics are `pub(crate)` to
/// `mantis-selfplay` with no bridge- or Python-reachable setter, so a transposition of
/// the two accumulators is unobservable from BOTH sides of the FFI — and it is the worst
/// class of telemetry defect: permanent, silent, and invisible in aggregate, because the
/// two series simply trade places for the whole run. `mantis-selfplay`'s
/// `stats_snapshot_reads_back_each_private_atomic` applies exactly this discipline to the
/// Derived fixed-point mean in f32 arithmetic (frozen
/// `mcts_mean_root_concentration` — NOTE: f32, not f64, per DESIGN §c.6 / O19).
fn derived_mean_f32(accum: u64, count: u64) -> f32 {
    if count == 0 {
        0.0
    } else {
        accum as f32 / (count as f32 * 1_000_000.0)
    }
}

/// Configuration builder for [`PySelfPlayRunner`] — a thin POD over the NEW
/// pure-Rust [`SelfPlayRunnerConfig`]. The `#[new]` maps the NEW field set only;
/// the WP4-killed `interior_selector` and the WP6-killed `legal_move_radius_jitter`
/// are ABSENT (LAW-11 / KILL), and the caller-supplied `feature_len`/`policy_len`
/// shape overrides are dropped (shapes are spec-derived). The 10 post-construction
/// O1 / solver / seed knobs are exposed as get/set (frozen attr surface).
#[pyclass(name = "SelfPlayRunnerConfig", module = "mantis._engine")]
#[derive(Clone)]
pub struct PySelfPlayRunnerConfig {
    inner: SelfPlayRunnerConfig,
}

impl PySelfPlayRunnerConfig {
    /// Crate-internal: the mapped native config (consumed by `PySelfPlayRunner`
    /// and read by the byte-equivalence field-set pin in `tests/bridge/`).
    pub(crate) fn to_rust(&self) -> SelfPlayRunnerConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PySelfPlayRunnerConfig {
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    #[new]
    #[pyo3(signature = (
        n_workers = 4,
        max_moves_per_game = 128,
        n_simulations = 50,
        leaf_batch_size = 8,
        c_puct = 1.5,
        fpu_reduction = 0.25,
        fast_prob = 0.0,
        fast_sims = 50,
        standard_sims = 0,
        temp_threshold_compound_moves = 0,
        draw_reward = -0.1,
        ply_cap_value = -0.1,
        quiescence_enabled = true,
        quiescence_blend_2 = 0.3,
        temp_min = 0.5,
        zoi_enabled = false,
        zoi_lookback = 16,
        zoi_margin = 5,
        c_visit = 50.0,
        c_scale = 1.0,
        gumbel_m = 16,
        gumbel_explore_moves = 10,
        dirichlet_alpha = 0.3,
        dirichlet_epsilon = 0.25,
        dirichlet_enabled = true,
        results_queue_cap = 10_000,
        full_search_prob = 0.0,
        n_sims_quick = 0,
        n_sims_full = 0,
        random_opening_plies = 0,
        encoding_name = None,
        inference_pool_size = None
    ))]
    pub fn new(
        n_workers: usize,
        max_moves_per_game: usize,
        n_simulations: usize,
        leaf_batch_size: usize,
        c_puct: f32,
        fpu_reduction: f32,
        fast_prob: f32,
        fast_sims: usize,
        standard_sims: usize,
        temp_threshold_compound_moves: usize,
        draw_reward: f32,
        ply_cap_value: f32,
        quiescence_enabled: bool,
        quiescence_blend_2: f32,
        temp_min: f32,
        zoi_enabled: bool,
        zoi_lookback: usize,
        zoi_margin: i32,
        c_visit: f32,
        c_scale: f32,
        gumbel_m: usize,
        gumbel_explore_moves: usize,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        dirichlet_enabled: bool,
        results_queue_cap: usize,
        full_search_prob: f32,
        n_sims_quick: usize,
        n_sims_full: usize,
        random_opening_plies: u32,
        encoding_name: Option<String>,
        inference_pool_size: Option<usize>,
    ) -> Self {
        // The 10 O1/solver/seed knobs come from Default (frozen `..Default::default()`);
        // Python sets the operative values as the get/set attributes below.
        PySelfPlayRunnerConfig {
            inner: SelfPlayRunnerConfig {
                n_workers,
                max_moves_per_game,
                n_simulations,
                leaf_batch_size,
                c_puct,
                fpu_reduction,
                fast_prob,
                fast_sims,
                standard_sims,
                temp_threshold_compound_moves,
                draw_reward,
                ply_cap_value,
                quiescence_enabled,
                quiescence_blend_2,
                temp_min,
                zoi_enabled,
                zoi_lookback,
                zoi_margin,
                c_visit,
                c_scale,
                gumbel_m,
                gumbel_explore_moves,
                dirichlet_alpha,
                dirichlet_epsilon,
                dirichlet_enabled,
                results_queue_cap,
                full_search_prob,
                n_sims_quick,
                n_sims_full,
                random_opening_plies,
                encoding_name,
                inference_pool_size,
                ..Default::default()
            },
        }
    }

    // ── The search kind (get/set) ──────────────────────────────────────────────
    //
    // Get/set rather than another constructor positional: `new` already carries 33, and
    // its own comment records that later knobs arrive this way. NO CONSTRUCTOR DEFAULT for
    // the kind is exposed through the setter — the `Default` the ctor folds in is Rust
    // test scaffolding, and `SelfPlayHParams` always writes this attribute from the
    // required `search.kind` config key (R1/LAW-11).

    #[getter]
    pub fn search_kind(&self) -> &'static str {
        self.inner.search_kind.as_config_str()
    }
    /// # Errors
    /// `ValueError` — `v` is not a search kind this build knows. REFUSED rather than
    /// defaulted: a typo'd kind silently falling back to `puct` is exactly the
    /// silent-fallback class LAW-11 closes.
    #[setter]
    pub fn set_search_kind(&mut self, v: &str) -> PyResult<()> {
        let parsed = SearchKind::from_config_str(v).ok_or_else(|| {
            PyValueError::new_err(format!(
                "search.kind={v:?} is not a known search kind \
                 (expected \"puct\" or \"gumbel\")"
            ))
        })?;
        self.inner.search_kind = parsed;
        Ok(())
    }



}

/// The ONE supervisor-facing wording for a latched run-fatal self-play defect.
///
/// R275(b) widened the latch's population: it carried only `TargetIntegrityError`
/// (record-dispatch refusals), and now also carries `InferenceSeamFailure` (a leaf
/// inference that failed on an open queue). The old text hard-coded
/// "target-integrity defect", which would have mislabelled every seam failure as a
/// target defect — the same class-confusion that cost F-816-9 its diagnosis. The
/// prefix now names the LAW, and the stored message leads with the variant name,
/// which is the stable grep anchor (`VisitSlotsExceeded`, `MassNotUnity`,
/// `EmptyTarget`, `ZeroVisitSearch`, `InferenceSeamFailure`).
fn fatal_defect_message(msg: &str) -> String {
    format!("self-play run-fatal defect (LAW-14): {msg}")
}

/// Pure-Rust self-play runner exposed to Python. Arc-based (`Send + Sync`); the
/// bridge holds an `Arc<SelfPlayRunner>` and a runner-linked `InferenceBatcher`.
#[pyclass(name = "SelfPlayRunner", module = "mantis._engine")]
pub struct PySelfPlayRunner {
    inner: Arc<SelfPlayRunner>,
    batcher: PyInferenceBatcher,
}

#[pymethods]
impl PySelfPlayRunner {
    /// Construct from a [`PySelfPlayRunnerConfig`]. Resolves + validates the
    /// encoding via the runner (absent/unknown `encoding_name` → `ValueError`,
    /// LAW-11), then builds a runner-linked batcher over the live queues.
    #[new]
    pub fn new(config: &PySelfPlayRunnerConfig) -> PyResult<Self> {
        let rust_config = config.to_rust();
        let encoding_name = rust_config.encoding_name.clone();
        let runner = SelfPlayRunner::new(rust_config).map_err(PyValueError::new_err)?;
        // `new()` succeeded ⇒ encoding_name is Some and resolvable.
        let spec = encoding_name
            .as_deref()
            .and_then(mantis_encoding::lookup)
            .expect("SelfPlayRunner::new validated the encoding_name resolves");
        let inner = Arc::new(runner);
        let batcher = PyInferenceBatcher::from_runner(
            spec,
            inner.graph_producer(),
            inner.clone(),
        );
        Ok(PySelfPlayRunner { inner, batcher })
    }

    /// Spawn `n_workers` self-play threads (idempotent).
    pub fn start(&self) {
        self.inner.start();
    }

    /// Flip running=false, close the inference queues, and join all workers.
    pub fn stop(&self) {
        self.inner.stop();
    }

    pub fn is_running(&self) -> bool {
        self.inner.is_running()
    }


    /// Drain all buffered graph-position records as a list of 9-tuples (field
    /// order = `HexgBuffer.push_graph_position`; no numpy — the records are
    /// variable-length). Grid runners return an empty list.
    ///
    /// WP12-R Phase T (DESIGN_T §3.4; LAW-14): raises the runner's stored
    /// fatal-defect message (a typed `TargetIntegrityError`, variant name in
    /// the text) so the Python pool drain loop dies loud — run-fatal, no
    /// silent except.
    ///
    /// # Errors
    /// `RuntimeError` when the runner's fatal-defect latch is set.
    pub fn collect_graph_data(&self) -> PyResult<Vec<GraphRecordRow>> {
        if let Some(msg) = self.inner.fatal_defect() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                fatal_defect_message(&msg),
            ));
        }
        Ok(self
            .inner
            .drain_graph_records()
            .into_iter()
            .map(|r| {
                (
                    r.stones,
                    r.visits,
                    r.current_player,
                    r.moves_remaining,
                    r.ply_index,
                    r.is_full_search,
                    r.outcome,
                    r.value_valid,
                    r.game_length,
                    // R345(b)(6): the game this position came from. Appended LAST so the
                    // leading nine stay exactly `HexgBuffer.push_graph_position`'s positional
                    // signature and the Python push can forward them verbatim.
                    r.game_id,
                )
            })
            .collect())
    }

    /// The runner-linked inference batcher (shares the runner's live queues +
    /// model-version atomic). Returns a clone sharing the Arc-backed state.
    #[getter]
    pub fn batcher(&self) -> PyInferenceBatcher {
        self.batcher.clone()
    }



    // ── win / throughput counters (RAW atomics via the snapshot) ───────────────
    #[getter]
    pub fn games_completed(&self) -> usize {
        self.snapshot().games_completed
    }
    #[getter]
    pub fn positions_generated(&self) -> usize {
        self.snapshot().positions_generated
    }
    #[getter]
    pub fn x_wins(&self) -> u64 {
        self.snapshot().x_wins
    }
    #[getter]
    pub fn o_wins(&self) -> u64 {
        self.snapshot().o_wins
    }
    #[getter]
    pub fn draws(&self) -> u64 {
        self.snapshot().draws
    }
    #[getter]
    pub fn positions_dropped(&self) -> u64 {
        self.snapshot().positions_dropped
    }

    pub fn get_win_stats(&self) -> (u64, u64, u64) {
        let s = self.snapshot();
        (s.x_wins, s.o_wins, s.draws)
    }

    // ── MCTS-health derived means (bridge-derived per DESIGN §c.6 / O19) ────────
    /// SCOPE NOTE (ADJ-D32 / R249): this getter keeps its `f32` shape and its
    /// zero-count `0.0`. `mcts_stat_count` increments once per search in
    /// `play_one_move`, path-independently, so its zero is TRANSIENT (a run before its
    /// first move) rather than the graph arm's permanent one — a different severity
    /// class, and the ruling's mandate is the cluster pair. Changing it is a separate
    /// decision, reported not taken.
    #[getter]
    pub fn mcts_mean_depth(&self) -> f32 {
        let s = self.snapshot();
        derived_mean_f64(s.mcts_depth_accum, s.mcts_stat_count).unwrap_or(0.0)
    }
    #[getter]
    pub fn mcts_mean_root_concentration(&self) -> f32 {
        let s = self.snapshot();
        derived_mean_f32(s.mcts_conc_accum, s.mcts_stat_count)
    }
    #[getter]
    pub fn mcts_quiescence_fires(&self) -> u64 {
        self.snapshot().mcts_quiescence_fires
    }
    /// R335(c)/LAW-18 — the largest leaf count ANY one search served, since `start()`.
    ///
    /// It must never exceed the run's sim budget. Before the batch clamp it read
    /// `n_simulations + leaf_batch_size − 1`, which is what put the ledger's
    /// `53.46 sims/move` against a configured 50; the lever under test therefore reports its
    /// own rate in-run, and re-measuring that line no longer needs a diagnostic rig branch.
    /// A run whose search budget is uniform reads exactly `n_simulations` here; the Gumbel arm
    /// reads LESS, because sequential halving never allocates its integer-division remainder.
    #[getter]
    pub fn max_sims_per_search(&self) -> u64 {
        self.snapshot().max_sims_per_search
    }


    // ── WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6) ────────
    #[getter]
    pub fn export_offwindow_mass_moves(&self) -> u64 {
        self.snapshot().export_offwindow_mass_moves
    }
    /// Fatal-defect latch fire count — must read 0 in a healthy run (the latch
    /// message itself surfaces through `collect_graph_data`'s typed raise).
    #[getter]
    pub fn target_integrity_defects(&self) -> u64 {
        self.snapshot().target_integrity_defects
    }

    /// R275(b) SEAM conjunct — leaf inferences that FAILED on an open queue and
    /// halted the run. Reads 0 in a healthy run, and a drain shutdown does NOT
    /// count: `stop()` closes both queues, and a failure arm reached with the
    /// queue closed is the §P22 skip, not a defect.
    ///
    /// Published on EVERY encoding (R250 absence does not apply, R256 mapping
    /// re-derived from code): both `infer_and_expand` arms — dense queue and
    /// graph queue — carry a failure leg, so the mechanism is live wherever
    /// self-play runs.
    #[getter]
    pub fn inference_failures_total(&self) -> u64 {
        self.snapshot().inference_failures_total
    }


    /// Worker threads that died by panic — must read 0 in a healthy run.
    ///
    /// A panicking worker used to be invisible from Python: the panic sat in its
    /// `JoinHandle`, `stop()` discarded it, and the pool kept reporting healthy while
    /// producing nothing. Non-zero here means self-play halted on a worker death, and it is
    /// the difference between diagnosing that and chasing "self-play got slow".
    #[getter]
    pub fn worker_panics(&self) -> u64 {
        self.snapshot().worker_panics
    }

    /// Drain and return all buffered game results since the last call.
    pub fn drain_game_results(&self) -> Vec<GameResultRow> {
        self.inner.drain_game_results()
    }

    /// Current model-version snapshot (reads the runner atomic; the batcher's
    /// `bump_model_version` writes through to it).
    #[getter]
    pub fn model_version(&self) -> u64 {
        self.inner.model_version()
    }

    /// WP7 NN seam — set the shared model-version snapshot workers read.
    pub fn set_model_version(&self, version: u64) {
        self.inner.set_model_version(version);
    }
}

impl PySelfPlayRunner {
    /// One-instant snapshot of the RAW counter atomics (SEAM).
    fn snapshot(&self) -> RunnerStatsSnapshot {
        self.inner.stats_snapshot()
    }
}

impl Drop for PySelfPlayRunner {
    /// Stop the runner when the Python object is dropped (frozen Drop=stop) — the
    /// runner-linked batcher may keep the `Arc<SelfPlayRunner>` alive, so the
    /// explicit stop is what joins the workers on runner GC (idempotent).
    fn drop(&mut self) {
        self.inner.stop();
    }
}

/// Register the `SelfPlayRunnerConfig` + `SelfPlayRunner` pyclasses into
/// `_engine`. Called by Slice ASM.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlayRunnerConfig>()?;
    m.add_class::<PySelfPlayRunner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v6_config() -> PySelfPlayRunnerConfig {
        PySelfPlayRunnerConfig {
            inner: SelfPlayRunnerConfig {
                encoding_name: Some("v6".to_string()),
                n_workers: 1,
                ..Default::default()
            },
        }
    }

    #[test]
    fn config_ctor_maps_new_field_set() {
        // The killed knobs are absent from the ctor; the mapped native config
        // carries the supplied identity key and the get/set defaults.
        let cfg = PySelfPlayRunnerConfig::new(
            2,
            64,
            30,
            8,
            1.5,
            0.25,
            0.0,
            50,
            0,
            0,
            -0.1,
            -0.1,
            true,
            0.3,
            0.5,
            false,
            16,
            5,
            50.0,
            1.0,
            16,
            10,
            0.3,
            0.25,
            true,
            10_000,
            0.0,
            0,
            0,
            0,
            false,
            Some("v6".to_string()),
            None,
        );
        let rust = cfg.to_rust();
        assert_eq!(rust.n_workers, 2);
        assert_eq!(rust.max_moves_per_game, 64);
        assert_eq!(rust.encoding_name.as_deref(), Some("v6"));
        // O1/solver/seed knobs come from Default.
        assert!(!rust.forced_win_policy_enabled);
        assert_eq!(rust.forced_win_policy_depth, 2);
        assert!(!rust.solver_enabled);
        assert!((rust.solver_visit_weight - 0.3).abs() < 1e-6);
    }

    #[test]
    fn config_getset_attrs_round_trip() {
        let mut cfg = v6_config();
        cfg.set_solver_enabled(true);
        cfg.set_solver_depth(24);
        cfg.set_forced_win_policy_weight(0.5);
        cfg.set_seed_fraction(0.1);
        cfg.set_seed_corpus(Some(vec![vec![(0, 0), (1, 0)]]));
        assert!(cfg.solver_enabled());
        assert_eq!(cfg.solver_depth(), 24);
        assert!((cfg.forced_win_policy_weight() - 0.5).abs() < 1e-6);
        assert!((cfg.seed_fraction() - 0.1).abs() < 1e-6);
        assert_eq!(cfg.to_rust().seed_corpus, Some(vec![vec![(0, 0), (1, 0)]]));
    }

    #[test]
    fn runner_constructs_and_lifecycle() {
        let r = PySelfPlayRunner::new(&v6_config()).expect("v6 runner constructs");
        assert!(!r.is_running());
        assert_eq!(r.games_completed(), 0);
        assert_eq!(r.model_version(), 0);
        r.set_model_version(5);
        assert_eq!(r.model_version(), 5);
        // The runner-linked batcher shares the model-version atomic.
        let b = r.batcher();
        assert_eq!(b.model_version(), 5);
        assert_eq!(b.bump_model_version(), 6);
        assert_eq!(
            r.model_version(),
            6,
            "batcher bump reaches the runner atomic"
        );
    }

    #[test]
    fn runner_missing_encoding_errors() {
        let cfg = PySelfPlayRunnerConfig::new(
            1, 64, 30, 8, 1.5, 0.25, 0.0, 50, 0, 0, -0.1, -0.1, true, 0.3, 0.5, false, 16, 5, 50.0,
            1.0, 16, 10, 0.3, 0.25, true, 10_000, 0.0, 0, 0, 0, false, None, None,
        );
        assert!(
            PySelfPlayRunner::new(&cfg).is_err(),
            "absent encoding_name is an error (LAW-11)"
        );
    }

    /// O19 — the derived-mean formulae on seeded accum/count.
    #[test]
    fn derived_means_match_fixed_point_formula() {
        // accum = mean × count × 1e6. depth mean 3.5 over 4 samples:
        // accum = 3.5 × 4 × 1e6 = 14_000_000.
        assert!((derived_mean_f64(14_000_000, 4).unwrap() - 3.5).abs() < 1e-6);
        // root-concentration mean 0.75 over 2 samples (f32 path):
        // accum = 0.75 × 2 × 1e6 = 1_500_000.
        assert!((derived_mean_f32(1_500_000, 2) - 0.75).abs() < 1e-6);
        // A single sample of 1.0 → accum 1_000_000, count 1 → 1.0.
        assert!((derived_mean_f64(1_000_000, 1).unwrap() - 1.0).abs() < 1e-6);
        // The f32 root-concentration form KEEPS its zero-guard (see the getter's scope
        // note): its count is `mcts_stat_count`, whose zero is transient.
        assert_eq!(derived_mean_f32(12_345, 0), 0.0);
    }

    /// ADJ-D32 / R249 MUTATION pin — a derived f64 mean over ZERO samples is `None`,
    /// never a number.
    ///
    /// This is the defect's root: with `cluster_variance_samples` pinned at 0 on the
    /// graph arm, a `0.0` here became `cluster_value_std_mean: 0.0` in every
    /// `iteration_complete` of the run — a fabricated measurement in the run's ONE
    /// event channel, indistinguishable from a real settled ensemble.
    ///
    /// FALSIFYING MUTATION: restore the zero-count arm to `Some(0.0)` (or revert the
    /// return type to `f32` with `0.0`, which reds this by compile error instead).
    /// Either MUST turn this test RED.
    #[test]
    fn zero_count_derived_mean_is_none_never_zero() {
        assert_eq!(
            derived_mean_f64(0, 0),
            None,
            "R249: an empty accumulator over zero samples is NOT a measured 0.0"
        );
        assert_eq!(
            derived_mean_f64(12_345, 0),
            None,
            "R249: zero count is None regardless of the accumulator's residue"
        );
        // The guard is on the COUNT alone — a genuine zero mean over real samples is a
        // measurement and must still be published.
        assert_eq!(derived_mean_f64(0, 4), Some(0.0));
    }

    /// ADJ-D32 closing pin — each cluster mean derives from its OWN accumulator.
    ///
    /// The two accumulators share a divisor and a type, so a transposition compiles,
    /// reads plausible, and is invisible in aggregate — the two series trade places for
    /// the whole run and no post-hoc analysis can separate them again. Every
    /// Python-side pin drives both means `None` (zero samples), where a swap is
    /// `None == None`; DISTINCT seeded values are the only instrument that can see it.
    /// The producing crate pins the atomic→snapshot half the same way
    /// (`mantis_selfplay::runner::tests::stats_snapshot_reads_back_each_private_atomic`);
    /// this is the snapshot→getter half.
    ///
    /// FALSIFYING MUTATION: swap `cluster_value_std_accum` and
    /// `cluster_policy_disagreement_accum` inside `cluster_means`. MUST turn this RED.
    #[test]
    fn cluster_means_read_their_own_accumulators() {
        let s = RunnerStatsSnapshot {
            // mean 0.5 over 4 samples = 0.5 × 4 × 1e6.
            cluster_value_std_accum: 2_000_000,
            // mean 1.5 over the SAME 4 samples = 1.5 × 4 × 1e6. Deliberately unequal and
            // deliberately not a permutation of the other, so a swap cannot read as noise.
            cluster_policy_disagreement_accum: 6_000_000,
            cluster_variance_samples: 4,
            ..RunnerStatsSnapshot::default()
        };

        let (value_std, disagreement) = cluster_means(&s);
        assert_eq!(
            value_std,
            Some(0.5),
            "cluster_value_std_mean must derive from cluster_value_std_accum"
        );
        assert_eq!(
            disagreement,
            Some(1.5),
            "cluster_policy_disagreement_mean must derive from \
             cluster_policy_disagreement_accum"
        );
    }

    /// A fresh runner's graph-record drain is empty (numpy-free — no numpy is
    /// built for `collect_graph_data`). The `collect_data` 10-numpy-array marshal
    /// (incl. the zero-row case) is pinned by the Python O-side tests post-ASM —
    /// the embedded cargo-test interpreter cannot load numpy's C-extension.
    #[test]
    fn collect_graph_data_empty_on_fresh_runner() {
        let r = PySelfPlayRunner::new(&v6_config()).unwrap();
        assert!(r
            .collect_graph_data()
            .expect("no fatal defect on a fresh runner")
            .is_empty());
    }
}
