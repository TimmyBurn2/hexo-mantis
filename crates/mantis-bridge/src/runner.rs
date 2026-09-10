// Exceeds the 300-line soft cap (R8): the `SelfPlayRunnerConfig` builder pyclass and the
// `SelfPlayRunner` pyclass facade (collect_data marshal, the ~20 counter getters incl. the
// bridge-derived means, batcher, Drop=stop) are one port unit against the frozen game_runner.
//! `SelfPlayRunnerConfig` + `SelfPlayRunner` pyclass surface over WP6 `mantis_selfplay::runner`.
//!
//! Both classes are Arc-based `Send + Sync` (NO `unsendable`) and workers own their `Board` in
//! Rust. The counter getters read a one-instant snapshot of RAW atomics and the bridge DERIVES
//! the 4 means (÷(count × 1e6), `None` at zero count).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mantis_search::SearchKind;
use mantis_selfplay::runner::config::SelfPlayRunnerConfig;
use mantis_selfplay::runner::{GameResultRow, RunnerStatsSnapshot, SelfPlayRunner};

use crate::inference::PyInferenceBatcher;

/// Per-row tuple from `collect_graph_data`: the first NINE fields are
/// `HexgBuffer.push_graph_position`'s positional signature verbatim, the tenth its keyword
/// `game_id`.
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

/// Derived fixed-point mean in f64: `accum / (count × 1e6)`, `count == 0 -> None`. The `*_accum`
/// atomics are ×1_000_000 integers, and `None` is the only honest reading of a zero-sample mean.
fn derived_mean_f64(accum: u64, count: u64) -> Option<f32> {
    if count == 0 {
        None
    } else {
        Some((accum as f64 / (count as f64 * 1_000_000.0)) as f32)
    }
}

/// The two cluster-variance means from ONE snapshot, in field order: `(value spread, top-1
/// policy disagreement)`. A transposition is unobservable from BOTH sides of the FFI.
fn derived_mean_f32(accum: u64, count: u64) -> f32 {
    if count == 0 {
        0.0
    } else {
        accum as f32 / (count as f32 * 1_000_000.0)
    }
}

/// Configuration builder for [`PySelfPlayRunner`] — a thin POD over the pure-Rust
/// [`SelfPlayRunnerConfig`]. Killed knobs are ABSENT, shapes are spec-derived.
#[pyclass(name = "SelfPlayRunnerConfig", module = "mantis._engine")]
#[derive(Clone)]
pub struct PySelfPlayRunnerConfig {
    inner: SelfPlayRunnerConfig,
}

impl PySelfPlayRunnerConfig {
    /// Crate-internal: the mapped native config, read by the field-set pin in `tests/bridge/`.
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
    ) -> Self {
        // The 10 O1/solver/seed knobs come from Default; Python sets them as attributes below.
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
                ..Default::default()
            },
        }
    }

    // Get/set rather than another constructor positional. NO CONSTRUCTOR DEFAULT for the kind
    // is exposed: `SelfPlayHParams` always writes it from the required `search.kind` config key.

    #[getter]
    pub fn search_kind(&self) -> &'static str {
        self.inner.search_kind.as_config_str()
    }
    /// # Errors
    /// `ValueError` — an unknown search kind, REFUSED rather than defaulted to `puct`.
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

/// The ONE supervisor-facing wording for a latched run-fatal self-play defect. The prefix names
/// the LAW rather than one defect class; the message leads with the variant name, the grep anchor.
fn fatal_defect_message(msg: &str) -> String {
    format!("self-play run-fatal defect (LAW-14): {msg}")
}

/// Pure-Rust self-play runner exposed to Python; Arc-based (`Send + Sync`).
#[pyclass(name = "SelfPlayRunner", module = "mantis._engine")]
pub struct PySelfPlayRunner {
    inner: Arc<SelfPlayRunner>,
    batcher: PyInferenceBatcher,
}

#[pymethods]
impl PySelfPlayRunner {
    /// Construct from a [`PySelfPlayRunnerConfig`], resolving and validating the encoding
    /// (absent or unknown -> `ValueError`) before building a runner-linked batcher.
    #[new]
    pub fn new(config: &PySelfPlayRunnerConfig) -> PyResult<Self> {
        let rust_config = config.to_rust();
        let encoding_name = rust_config.encoding_name.clone();
        let runner = SelfPlayRunner::new(rust_config).map_err(PyValueError::new_err)?;
        let spec = encoding_name
            .as_deref()
            .and_then(mantis_encoding::lookup)
            .expect("SelfPlayRunner::new validated the encoding_name resolves");
        let inner = Arc::new(runner);
        let batcher = PyInferenceBatcher::from_runner(spec, inner.graph_producer(), inner.clone());
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

    /// Drain all buffered graph-position records as a list of 10-tuples (no numpy — the records
    /// are variable-length); grid runners return an empty list.
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
                    // Appended LAST so the leading nine stay exactly
                    // `HexgBuffer.push_graph_position`'s positional signature.
                    r.game_id,
                )
            })
            .collect())
    }

    /// The runner-linked inference batcher, a clone sharing the Arc-backed queues and atomic.
    #[getter]
    pub fn batcher(&self) -> PyInferenceBatcher {
        self.batcher.clone()
    }

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

    /// SCOPE NOTE: this getter keeps its `f32` shape and its zero-count `0.0`, because
    /// `mcts_stat_count` increments once per search, so its zero is TRANSIENT.
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
    /// The largest leaf count ANY one search served since `start()`, which must never exceed the
    /// sim budget. Before the batch clamp it read `n_simulations + leaf_batch_size - 1`, which put
    /// the ledger's 53.46 sims/move against a configured 50. The Gumbel arm reads LESS.
    #[getter]
    pub fn max_sims_per_search(&self) -> u64 {
        self.snapshot().max_sims_per_search
    }

    #[getter]
    pub fn export_offwindow_mass_moves(&self) -> u64 {
        self.snapshot().export_offwindow_mass_moves
    }
    /// Fatal-defect latch fire count — must read 0 in a healthy run.
    #[getter]
    pub fn target_integrity_defects(&self) -> u64 {
        self.snapshot().target_integrity_defects
    }

    /// Leaf inferences that FAILED on an open queue and halted the run; 0 in a healthy run, and
    /// a drain shutdown does NOT count. Live on EVERY encoding: both arms carry a failure leg.
    #[getter]
    pub fn inference_failures_total(&self) -> u64 {
        self.snapshot().inference_failures_total
    }

    /// Worker threads that died by panic — 0 in a healthy run. Before this the panic sat in the
    /// `JoinHandle`, `stop()` discarded it, and the pool reported healthy while producing nothing.
    #[getter]
    pub fn worker_panics(&self) -> u64 {
        self.snapshot().worker_panics
    }

    /// Drain and return all buffered game results since the last call.
    pub fn drain_game_results(&self) -> Vec<GameResultRow> {
        self.inner.drain_game_results()
    }

    /// Current model-version snapshot; the batcher's `bump_model_version` writes through to it.
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
    /// Stop the runner on drop: the runner-linked batcher may keep the `Arc` alive, so the
    /// explicit stop is what joins the workers on runner GC (idempotent).
    fn drop(&mut self) {
        self.inner.stop();
    }
}

/// Register the `SelfPlayRunnerConfig` and `SelfPlayRunner` pyclasses into `_engine`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfPlayRunnerConfig>()?;
    m.add_class::<PySelfPlayRunner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_config() -> PySelfPlayRunnerConfig {
        PySelfPlayRunnerConfig {
            inner: SelfPlayRunnerConfig {
                encoding_name: Some("gnn_axis_v1".to_string()),
                n_workers: 1,
                ..Default::default()
            },
        }
    }

    #[test]
    fn config_ctor_maps_new_field_set() {
        // The killed knobs are absent from the ctor; the mapped config carries the identity key.
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
            Some("gnn_axis_v1".to_string()),
        );
        let rust = cfg.to_rust();
        assert_eq!(rust.n_workers, 2);
        assert_eq!(rust.max_moves_per_game, 64);
        assert_eq!(rust.encoding_name.as_deref(), Some("gnn_axis_v1"));
        // The knobs the ctor does NOT carry still come from Default.
        assert_eq!(rust.gumbel_m, 16);
        assert!(rust.dirichlet_enabled);
    }

    #[test]
    fn runner_constructs_and_lifecycle() {
        let r = PySelfPlayRunner::new(&graph_config()).expect("a graph runner constructs");
        assert!(!r.is_running());
        assert_eq!(r.games_completed(), 0);
        assert_eq!(r.model_version(), 0);
        r.set_model_version(5);
        assert_eq!(r.model_version(), 5);
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
            1, 64, 30, 8, 1.5, 0.25, 0.0, 50, 0, 0, -0.1, -0.1, true, 0.3, 0.5, 50.0, 1.0, 16, 10,
            0.3, 0.25, true, 10_000, 0.0, 0, 0, 0, None,
        );
        assert!(
            PySelfPlayRunner::new(&cfg).is_err(),
            "absent encoding_name is an error (LAW-11)"
        );
    }

    /// O19 — the derived-mean formulae on seeded accum/count.
    #[test]
    fn derived_means_match_fixed_point_formula() {
        // accum = mean × count × 1e6; depth mean 3.5 over 4 samples = 14_000_000.
        assert!((derived_mean_f64(14_000_000, 4).unwrap() - 3.5).abs() < 1e-6);
        // root-concentration mean 0.75 over 2 samples (f32 path) = 1_500_000.
        assert!((derived_mean_f32(1_500_000, 2) - 0.75).abs() < 1e-6);
        assert!((derived_mean_f64(1_000_000, 1).unwrap() - 1.0).abs() < 1e-6);
        // The f32 root-concentration form KEEPS its zero-guard: its count is transient.
        assert_eq!(derived_mean_f32(12_345, 0), 0.0);
    }

    /// MUTATION pin — a derived f64 mean over ZERO samples is `None`, never a number. With a
    /// sample count pinned at 0, a `0.0` here became a `cluster_value_std_mean: 0.0` in every
    /// `iteration_complete`: a fabricated measurement in the run's ONE event channel.
    ///
    /// FALSIFYING MUTATION: restore the zero-count arm to `Some(0.0)` (or revert the return type
    /// to `f32` with `0.0`, which reds this by compile error instead). Either MUST turn it RED.
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
        // The guard is on the COUNT alone — a genuine zero mean over real samples is published.
        assert_eq!(derived_mean_f64(0, 4), Some(0.0));
    }

    /// A fresh runner's graph-record drain is empty. The `collect_data` numpy marshal is pinned
    /// by the Python tests, since the embedded cargo-test interpreter cannot load numpy.
    #[test]
    fn collect_graph_data_empty_on_fresh_runner() {
        let r = PySelfPlayRunner::new(&graph_config()).expect("a graph runner constructs");
        assert!(r
            .collect_graph_data()
            .expect("no fatal defect on a fresh runner")
            .is_empty());
    }
}
