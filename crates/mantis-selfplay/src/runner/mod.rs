//! R8-justify: the pure-Rust `SelfPlayRunner` core (the ~40-field accumulator/
//! queue-owning struct + its resolving ctor + start/stop/drain lifecycle) is one
//! indivisible unit — the frozen `game_runner/mod.rs` was a single long module; the
//! phase bodies split into sibling modules (`spawn`/`game`/…) but the struct and its
//! lifecycle stay together so the ownership story is greppable in one file.
//!
//! Self-play runner core (WP6 D1) — the pyo3-STRIPPED half of the frozen
//! `game_runner/mod.rs`. Owns the shared `Arc` accumulators, the dense + graph
//! inference queues (`crate::queues`), the result queues, and the LAW-18
//! in-run fire counters. `start`/`stop`/
//! `is_running`/`drain_game_results` are the pure-Rust lifecycle; a producer
//! handle exposes the queues so a MOCK producer (tests) / the WP7 NN producer
//! face can `pop_batch` + `submit_results`.
//!
//! DROPPED to WP7 (R6/LAW-17 — pyo3 only in the bridge): the `#[pyclass]`
//! `SelfPlayRunner` face, every `#[getter]`, `collect_data` (10-numpy-array),
//! `collect_graph_data`, and the `batcher()` pymethod. The in-run fire counters'
//! READ getters are WP7-owed write-only debt (R9).

pub mod atomics;
pub mod config;
pub mod finalize;
pub mod game;
pub mod params;
pub mod record;
pub mod rotate;
pub mod search_drive;
pub mod spawn;
pub mod stats;

pub use config::SelfPlayRunnerConfig;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use mantis_encoding::{all_specs, lookup, RegistrySpec};

use crate::queues::{DenseQueue, GraphQueue};
use crate::replay::hexg::GraphRecord;

/// Per-row training tuple produced by self-play workers (frozen `mod.rs:44`).
/// Field order: `(feat, chain, policy, outcome, plies, combined_aux_u8,
/// is_full_search, ply_index, value_valid)`. The P-04 pin destructures this
/// carrier exhaustively — a carrier-type change bites.
pub type WorkerResultRow = (Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, Vec<u8>, bool, u16, u8);

/// Per-game result tuple consumed by [`SelfPlayRunner::drain_game_results`]
/// (frozen `mod.rs:54`). Field order: `(plies, winner_code, move_history,
/// worker_id, terminal_reason, model_version_min, model_version_max,
/// model_version_distinct, seeded, solver_fires)`.
pub type GameResultRow = (usize, u8, Vec<(i32, i32)>, usize, u8, u64, u64, u32, u8, u32);

/// Flat snapshot of the runner's 24 LAW-18 in-run counter atomics, each read once
/// via a single `Relaxed` load (the WP7-owed READ side of the write-only fire
/// counters — see the module doc). RAW cumulative counts ONLY: the fixed-point
/// ×1_000_000 accumulators (`*_accum`) are handed back UNDIVIDED so the WP7 bridge
/// derives the 4 means itself (`accum / (count × 1e6)`, `count == 0 → 0.0`; the
/// SEAM does NOT compute means). Every field maps 1:1 to a [`SelfPlayRunner`]
/// counter of the same name. Cumulative since `start()`; monotone across calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunnerStatsSnapshot {
    // ── win / throughput ──
    pub games_completed: usize,
    pub positions_generated: usize,
    pub x_wins: u64,
    pub o_wins: u64,
    pub draws: u64,
    pub positions_dropped: u64,
    // ── MCTS-health accumulators (`*_accum` are fixed-point ×1e6; the bridge
    //    derives `mcts_mean_depth` / `mcts_mean_root_concentration` from these) ──
    pub mcts_depth_accum: u64,
    pub mcts_conc_accum: u64,
    pub mcts_stat_count: u64,
    pub mcts_quiescence_fires: u64,
    // `cluster_value_std_accum` / `cluster_policy_disagreement_accum` are ×1e6;
    // `cluster_variance_samples` is the shared divisor count for both means.
    pub cluster_value_std_accum: u64,
    pub cluster_policy_disagreement_accum: u64,
    pub cluster_variance_samples: u64,
    // ── D-WS3V3 in-run solver fire-rate counters ──
    pub solver_moves_eligible: u64,
    pub solver_win_proven: u64,
    pub solver_injected: u64,
    pub solver_injected_offwindow: u64,
    pub solver_budget_exhausted: u64,
    pub solver_moves_eligible_seeded: u64,
    pub solver_injected_seeded: u64,
    pub seeded_games_started: u64,
    // ── WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6) ──
    /// Moves whose exported policy target carried off-window (overflow) mass.
    pub export_offwindow_mass_moves: u64,
    /// §3.5 zero-row fills, counted per recorded grid-ls cluster row.
    pub gridls_zero_policy_rows: u64,
    /// Fatal-defect latch fire count (must read 0 in a healthy run).
    pub target_integrity_defects: u64,
}

/// Pure-Rust self-play runner core. Spawns worker threads (`spawn.rs`) that run
/// full games, stream training rows into the result queues, and track win stats
/// plus MCTS/solver fire-rate counters. Every worker OWNS its `Board`
/// (`Board` is `Send + !Sync`, D3) — there is NO shared-board Arc.
pub struct SelfPlayRunner {
    /// Resolved encoding spec (never `None` — an absent identity key is rejected
    /// at `new()`, LAW-11).
    spec: &'static RegistrySpec,
    /// Runner config (with `standard_sims` already resolved to the effective
    /// budget; `seed_corpus` moved out to the `Arc` below).
    config: SelfPlayRunnerConfig,

    // ── inference queues (owned; producer handles exposed) ──
    dense_queue: DenseQueue,
    graph_queue: GraphQueue,

    // ── shared result queues ──
    results: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    graph_results: Arc<Mutex<VecDeque<GraphRecord>>>,
    recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,

    // ── control ──
    running: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// WP7 NN model-version snapshot source. Each worker reads this once per move and
    /// dedup-pushes it into `version_seen` (drain tuple `mv_min/mv_max/mv_distinct`).
    /// Defaults to 0 (no-NN); WP7 wires the real setter when the NN producer lands.
    model_version: Arc<AtomicU64>,

    /// Ctor-validated seed corpus (shared read-only across workers).
    seed_corpus: Arc<Vec<Vec<(i32, i32)>>>,

    // ── win / throughput accumulators ──
    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    x_wins: Arc<AtomicU64>,
    o_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    positions_dropped: Arc<AtomicU64>,

    // ── MCTS-health accumulators (LAW-18; read getters WP7-owed) ──
    mcts_depth_accum: Arc<AtomicU64>,
    mcts_conc_accum: Arc<AtomicU64>,
    mcts_stat_count: Arc<AtomicU64>,
    mcts_quiescence_fires: Arc<AtomicU64>,
    cluster_value_std_accum: Arc<AtomicU64>,
    cluster_policy_disagreement_accum: Arc<AtomicU64>,
    cluster_variance_samples: Arc<AtomicU64>,

    // ── D-WS3V3 in-run solver fire-rate counters (LAW-18) ──
    solver_moves_eligible: Arc<AtomicU64>,
    solver_win_proven: Arc<AtomicU64>,
    solver_injected: Arc<AtomicU64>,
    solver_injected_offwindow: Arc<AtomicU64>,
    solver_budget_exhausted: Arc<AtomicU64>,
    solver_moves_eligible_seeded: Arc<AtomicU64>,
    solver_injected_seeded: Arc<AtomicU64>,
    seeded_games_started: Arc<AtomicU64>,

    // ── WP12-R Phase T target-integrity surfaces (LAW-18 / LAW-14) ──
    export_offwindow_mass_moves: Arc<AtomicU64>,
    gridls_zero_policy_rows: Arc<AtomicU64>,
    target_integrity_defects: Arc<AtomicU64>,
    /// The fatal-defect latch (DESIGN_T §3.4): a worker panic is NOT loud —
    /// `stop()` swallows join results — so a `TargetIntegrityError` at the
    /// record dispatch stores its message here (store-then-`running=false`)
    /// and the bridge drain face raises it as a typed Python exception.
    fatal_defect: Arc<Mutex<Option<String>>>,
}

impl SelfPlayRunner {
    /// Construct a runner from a native [`SelfPlayRunnerConfig`]. Resolves the
    /// `encoding_name` to a `&'static RegistrySpec` (absent / unknown = `Err`,
    /// LAW-11), runs the effective-sim / playout-cap validations (error strings
    /// verbatim from the frozen pyo3 ctor, `PyValueError` stripped to
    /// `Result<_, String>`), and constructs the owned queues + accumulators.
    ///
    /// # Errors
    /// Returns `Err(msg)` when `encoding_name` is absent or unknown, or when a
    /// sim-budget / playout-cap invariant is violated.
    pub fn new(mut config: SelfPlayRunnerConfig) -> Result<Self, String> {
        // Resolve the identity key. Absent spec = error (LAW-11 — the frozen
        // `None → v6` fallback is killed, D2); unknown name = error naming the
        // bad name + the registry hint (error string verbatim, `PyValueError`
        // stripped).
        let spec: &'static RegistrySpec = match config.encoding_name.as_deref() {
            Some(name) => match lookup(name) {
                Some(spec) => spec,
                None => {
                    let mut known: Vec<&str> = all_specs().map(|s| s.name).collect();
                    known.sort_unstable();
                    return Err(format!(
                        "SelfPlayRunner: encoding_name {name:?} not in registry; known: {known:?}"
                    ));
                }
            },
            None => {
                return Err(
                    "SelfPlayRunner: encoding_name is required (an absent registry spec is an \
                     error — LAW-11: identity keys have no terminal default; the frozen \
                     None → v6 fallback is killed)"
                        .to_string(),
                );
            }
        };

        // Effective standard-search sim budget: `standard_sims` wins, else
        // `n_simulations`. Reject zero on the *effective* value.
        let effective_standard = if config.standard_sims == 0 {
            config.n_simulations
        } else {
            config.standard_sims
        };
        if effective_standard == 0 {
            return Err(
                "SelfPlayRunner: n_simulations (or standard_sims) must be > 0".to_string(),
            );
        }
        if config.fast_prob > 0.0 && config.fast_sims == 0 {
            return Err("SelfPlayRunner: fast_sims must be > 0 when fast_prob > 0".to_string());
        }
        if config.full_search_prob > 0.0 && (config.n_sims_quick == 0 || config.n_sims_full == 0) {
            let (n_sims_quick, n_sims_full) = (config.n_sims_quick, config.n_sims_full);
            return Err(format!(
                "SelfPlayRunner: n_sims_quick and n_sims_full must both be > 0 \
                 when full_search_prob > 0 (got n_sims_quick={n_sims_quick}, n_sims_full={n_sims_full})"
            ));
        }

        // ── WP12-R Phase T boot guards (DESIGN_T §3.4; read EXISTING keys only,
        // R120; armed VALUES are never set, R119). Scoped to GRAPH encodings:
        // the bound being enforced is the graph record's fixed MAX_VISITS slot
        // (`replay/hexg`); dense-362 records carry no such slot, and refusing a
        // grid config would be a behavior change no ruling ordered (PREREG_T
        // A-6 — grid exemption resolved HERE, grounds recorded in IMPL notes).
        if spec.is_graph() {
            use crate::replay::hexg::MAX_VISITS;
            // Guard 1 (overshoot-aware temperature-arm bound): the production
            // sim loops overshoot by up to leaf_batch_size - 1 (the uncapped
            // final batch), so refuse when
            // max(ARMED effective sim counts) + leaf_batch_size - 1 > MAX_VISITS.
            // Armed arms: standard (always), fast iff fast_prob > 0,
            // quick/full iff full_search_prob > 0.
            let mut max_armed = effective_standard;
            if config.fast_prob > 0.0 {
                max_armed = max_armed.max(config.fast_sims);
            }
            if config.full_search_prob > 0.0 {
                max_armed = max_armed.max(config.n_sims_quick).max(config.n_sims_full);
            }
            let worst_case = max_armed + config.leaf_batch_size.saturating_sub(1);
            if worst_case > MAX_VISITS {
                let lb = config.leaf_batch_size;
                return Err(format!(
                    "SelfPlayRunner: max armed sim budget {max_armed} + leaf_batch_size {lb} \
                     - 1 = {worst_case} exceeds MAX_VISITS ({MAX_VISITS}) — a graph record's \
                     visit target could carry more cells than the fixed HEXG slot holds and \
                     silent truncation is deleted (WP12-R Phase T); lower the armed sim \
                     budgets / leaf batch, or raise MAX_VISITS"
                ));
            }
            // Guard 2 (completed-Q ls/graph arm, F-1(b)): the post-fix improved
            // exporter places positive mass on ALL children, so its graph-record
            // support is child-count-wide — no sims bound exists. Honest
            // retirement-until-raised: delete this guard when MAX_VISITS is
            // raised to MAX_CHILDREN_PER_NODE.
            if config.completed_q_values
                && mantis_search::MAX_CHILDREN_PER_NODE > MAX_VISITS
            {
                return Err(format!(
                    "SelfPlayRunner: representation==graph with completed_q_values=true is \
                     refused while MAX_CHILDREN_PER_NODE ({}) > MAX_VISITS ({MAX_VISITS}): \
                     the completed-Q exporter places positive mass on every root child, so \
                     a record's support is child-count-wide and cannot fit the HEXG visit \
                     slot (WP12-R Phase T, DESIGN_T §3.4); set completed_q_values=false for \
                     graph runs, or raise MAX_VISITS to {} and retire this guard",
                    mantis_search::MAX_CHILDREN_PER_NODE,
                    mantis_search::MAX_CHILDREN_PER_NODE,
                ));
            }
        }

        // Move the seed corpus into the shared `Arc`. STAGE R2 dry-replay
        // validates every prefix ONCE here (needs the R2 spec→Board resolution in
        // `game::init_per_game_board`); the ctor-validated corpus lets the R2
        // replay hook trust it (a runtime failure there = debug_assert).
        let seed_corpus_vec = config.seed_corpus.take().unwrap_or_default();

        // Bake the resolved budget so the workers read the effective value.
        config.standard_sims = effective_standard;

        // Own both disjoint inference queues (D4). The dense queue's feature width
        // is the spec's state stride; the graph queue is a graph-spec's live seam
        // (idle for grid). The NN + pyo3 producer face defer to WP7. The graph queue
        // carries the spec's `graph_contract_version` (validate guarantees `Some(1)`
        // for a graph spec; grid specs are `None` → 1) for the batch-level die-loud
        // handshake in `submit_graph_and_wait` (frozen `inference_bridge.rs:425`).
        let dense_queue = DenseQueue::new(spec.state_stride());
        let graph_queue = GraphQueue::with_contract_version(spec.contract_version.unwrap_or(1));

        Ok(Self {
            spec,
            config,
            dense_queue,
            graph_queue,
            results: Arc::new(Mutex::new(VecDeque::new())),
            graph_results: Arc::new(Mutex::new(VecDeque::new())),
            recent_game_results: Arc::new(Mutex::new(VecDeque::new())),
            running: Arc::new(AtomicBool::new(false)),
            handles: Arc::new(Mutex::new(Vec::new())),
            model_version: Arc::new(AtomicU64::new(0)),
            seed_corpus: Arc::new(seed_corpus_vec),
            games_completed: Arc::new(AtomicUsize::new(0)),
            positions_generated: Arc::new(AtomicUsize::new(0)),
            x_wins: Arc::new(AtomicU64::new(0)),
            o_wins: Arc::new(AtomicU64::new(0)),
            draws: Arc::new(AtomicU64::new(0)),
            positions_dropped: Arc::new(AtomicU64::new(0)),
            mcts_depth_accum: Arc::new(AtomicU64::new(0)),
            mcts_conc_accum: Arc::new(AtomicU64::new(0)),
            mcts_stat_count: Arc::new(AtomicU64::new(0)),
            mcts_quiescence_fires: Arc::new(AtomicU64::new(0)),
            cluster_value_std_accum: Arc::new(AtomicU64::new(0)),
            cluster_policy_disagreement_accum: Arc::new(AtomicU64::new(0)),
            cluster_variance_samples: Arc::new(AtomicU64::new(0)),
            solver_moves_eligible: Arc::new(AtomicU64::new(0)),
            solver_win_proven: Arc::new(AtomicU64::new(0)),
            solver_injected: Arc::new(AtomicU64::new(0)),
            solver_injected_offwindow: Arc::new(AtomicU64::new(0)),
            solver_budget_exhausted: Arc::new(AtomicU64::new(0)),
            solver_moves_eligible_seeded: Arc::new(AtomicU64::new(0)),
            solver_injected_seeded: Arc::new(AtomicU64::new(0)),
            seeded_games_started: Arc::new(AtomicU64::new(0)),
            export_offwindow_mass_moves: Arc::new(AtomicU64::new(0)),
            gridls_zero_policy_rows: Arc::new(AtomicU64::new(0)),
            target_integrity_defects: Arc::new(AtomicU64::new(0)),
            fatal_defect: Arc::new(Mutex::new(None)),
        })
    }

    /// WP12-R Phase T fatal-defect latch (DESIGN_T §3.4; LAW-14): store the
    /// typed defect message (first defect wins — the latch is write-once),
    /// count the fire, THEN flip `running=false` (store-then-halt) so the
    /// supervisor-facing drain can always read the reason for the halt. A
    /// worker panic is NOT sufficient — `stop()` swallows join results.
    pub fn store_fatal_defect(&self, msg: String) {
        {
            let mut slot = self.fatal_defect.lock().expect("fatal_defect lock poisoned");
            if slot.is_none() {
                *slot = Some(msg);
            }
        }
        self.target_integrity_defects.fetch_add(1, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }

    /// Read the stored fatal defect, if any — the bridge drain face
    /// (`collect_graph_data`) raises this as a typed Python exception so the
    /// pool drain loop dies with the variant name (LAW-14, R152 posture).
    #[must_use]
    pub fn fatal_defect(&self) -> Option<String> {
        self.fatal_defect
            .lock()
            .expect("fatal_defect lock poisoned")
            .clone()
    }

    /// Spawn `n_workers` self-play threads (idempotent). See [`spawn`].
    pub fn start(&self) {
        self.start_impl();
    }

    /// Flip `running=false`, close both inference queues (waking blocked waiters
    /// with `Err`), and join all worker threads (drain-shutdown, D12). An
    /// in-progress game is DROPPED, never finalized as a draw.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.dense_queue.close();
        self.graph_queue.close();
        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Drain and return all buffered game results since the last call (pure-Rust;
    /// the frozen pymethod wrapper is dropped to WP7).
    pub fn drain_game_results(&self) -> Vec<GameResultRow> {
        let mut rg = self
            .recent_game_results
            .lock()
            .expect("recent_game_results lock poisoned");
        rg.drain(..).collect()
    }

    // ── WP7 SEAM (pure-additive, zero-behaviour) ────────────────────────────────
    // Narrow pub read/drain faces the WP7 `mantis-bridge` producer pyclasses build
    // over; the frozen `collect_data` / `collect_graph_data` / `#[getter]` /
    // `bump_model_version` pymethods are dropped to the bridge (R6/LAW-17). None of
    // these mutate self beyond the drain queues they own; existing behaviour is
    // untouched.

    /// Drain and return all buffered training rows since the last call — the
    /// `collect_data` producer face (frozen pymethod dropped to WP7). Rows arrive in
    /// FIFO push order. Mirrors [`Self::drain_game_results`].
    pub fn drain_training_rows(&self) -> Vec<WorkerResultRow> {
        let mut rows = self.results.lock().expect("results lock poisoned");
        rows.drain(..).collect()
    }

    /// Drain and return all buffered graph training records since the last call —
    /// the `collect_graph_data` producer face (frozen pymethod dropped to WP7).
    /// FIFO push order. Mirrors [`Self::drain_game_results`].
    pub fn drain_graph_records(&self) -> Vec<GraphRecord> {
        let mut rows = self.graph_results.lock().expect("graph_results lock poisoned");
        rows.drain(..).collect()
    }

    /// WP7 NN seam — set the shared model-version snapshot workers read once per
    /// move (dedup-pushed into the drain tuple `mv_min/mv_max/mv_distinct`). The
    /// frozen `InferenceBatcher.bump_model_version` writes through here. `0` = no-NN.
    pub fn set_model_version(&self, version: u64) {
        self.model_version.store(version, Ordering::SeqCst);
    }

    /// Current NN model-version snapshot (`0` = no-NN) — the read side of
    /// [`Self::set_model_version`] (frozen `InferenceBatcher.model_version` getter).
    #[must_use]
    pub fn model_version(&self) -> u64 {
        self.model_version.load(Ordering::SeqCst)
    }

    /// Snapshot the 24 LAW-18 in-run counter atomics with one `Relaxed` load each.
    /// Returns RAW cumulative counts (the `*_accum` fixed-point ×1e6 sums are NOT
    /// divided here — the WP7 bridge derives the 4 means, DESIGN §c.6). See
    /// [`RunnerStatsSnapshot`].
    #[must_use]
    pub fn stats_snapshot(&self) -> RunnerStatsSnapshot {
        RunnerStatsSnapshot {
            games_completed: self.games_completed.load(Ordering::Relaxed),
            positions_generated: self.positions_generated.load(Ordering::Relaxed),
            x_wins: self.x_wins.load(Ordering::Relaxed),
            o_wins: self.o_wins.load(Ordering::Relaxed),
            draws: self.draws.load(Ordering::Relaxed),
            positions_dropped: self.positions_dropped.load(Ordering::Relaxed),
            mcts_depth_accum: self.mcts_depth_accum.load(Ordering::Relaxed),
            mcts_conc_accum: self.mcts_conc_accum.load(Ordering::Relaxed),
            mcts_stat_count: self.mcts_stat_count.load(Ordering::Relaxed),
            mcts_quiescence_fires: self.mcts_quiescence_fires.load(Ordering::Relaxed),
            cluster_value_std_accum: self.cluster_value_std_accum.load(Ordering::Relaxed),
            cluster_policy_disagreement_accum: self
                .cluster_policy_disagreement_accum
                .load(Ordering::Relaxed),
            cluster_variance_samples: self.cluster_variance_samples.load(Ordering::Relaxed),
            solver_moves_eligible: self.solver_moves_eligible.load(Ordering::Relaxed),
            solver_win_proven: self.solver_win_proven.load(Ordering::Relaxed),
            solver_injected: self.solver_injected.load(Ordering::Relaxed),
            solver_injected_offwindow: self.solver_injected_offwindow.load(Ordering::Relaxed),
            solver_budget_exhausted: self.solver_budget_exhausted.load(Ordering::Relaxed),
            solver_moves_eligible_seeded: self.solver_moves_eligible_seeded.load(Ordering::Relaxed),
            solver_injected_seeded: self.solver_injected_seeded.load(Ordering::Relaxed),
            seeded_games_started: self.seeded_games_started.load(Ordering::Relaxed),
            export_offwindow_mass_moves: self.export_offwindow_mass_moves.load(Ordering::Relaxed),
            gridls_zero_policy_rows: self.gridls_zero_policy_rows.load(Ordering::Relaxed),
            target_integrity_defects: self.target_integrity_defects.load(Ordering::Relaxed),
        }
    }

    /// Spec-derived state (feature) stride — drives inv23 (P-02).
    #[must_use]
    pub fn feature_len(&self) -> usize {
        self.spec.state_stride()
    }

    /// Spec-derived policy stride — drives inv23 (P-02).
    #[must_use]
    pub fn policy_len(&self) -> usize {
        self.spec.policy_stride()
    }

    /// PRODUCER handle for both inference queues (mock producer in tests; WP7 NN
    /// producer face in prod). The queues are `Clone` (share one `Arc` inner), so
    /// this hands out live handles that `pop_batch` + `submit_results`.
    #[must_use]
    pub fn producer_handles(&self) -> (DenseQueue, GraphQueue) {
        (self.dense_queue.clone(), self.graph_queue.clone())
    }

    /// PRODUCER handle for the dense inference queue alone.
    #[must_use]
    pub fn dense_producer(&self) -> DenseQueue {
        self.dense_queue.clone()
    }

    /// PRODUCER handle for the graph inference queue alone.
    #[must_use]
    pub fn graph_producer(&self) -> GraphQueue {
        self.graph_queue.clone()
    }
}

impl Drop for SelfPlayRunner {
    fn drop(&mut self) {
        self.stop();
    }
}

/// WP7 SEAM round-trip gate — proves the pure-additive pub read/drain faces return
/// exactly what the private queues / counter atomics hold. No workers are spawned
/// (`new()` does not `start()`), so the private state is populated deterministically
/// in-test and read back through the new pub API only. Uses distinct per-field
/// values so a getter crosswired to the wrong atomic FAILS.
#[cfg(test)]
mod seam_roundtrip {
    use std::sync::atomic::Ordering;

    use crate::replay::hexg::GraphRecord;

    use super::{RunnerStatsSnapshot, SelfPlayRunner, SelfPlayRunnerConfig, WorkerResultRow};

    /// Minimal valid runner: only the identity key is required by `new()`; the
    /// default sim budget passes validation and no worker is started.
    fn runner() -> SelfPlayRunner {
        SelfPlayRunner::new(SelfPlayRunnerConfig {
            encoding_name: Some("v6".to_string()),
            ..Default::default()
        })
        .expect("v6 must resolve via the registry")
    }

    #[test]
    fn drain_training_rows_returns_pushed_rows_then_empties() {
        let r = runner();
        assert!(
            r.drain_training_rows().is_empty(),
            "fresh runner has no training rows"
        );

        let row0: WorkerResultRow =
            (vec![1.0, 2.0], vec![3.0], vec![0.5], 1.0, 7, vec![9u8], true, 4u16, 1u8);
        let row1: WorkerResultRow =
            (vec![-1.0], vec![], vec![0.25, 0.75], -0.1, 3, vec![], false, 2u16, 0u8);
        {
            let mut q = r.results.lock().expect("results lock poisoned");
            q.push_back(row0.clone());
            q.push_back(row1.clone());
        }

        assert_eq!(
            r.drain_training_rows(),
            vec![row0, row1],
            "drain returns the FIFO-ordered private rows"
        );
        assert!(
            r.drain_training_rows().is_empty(),
            "a second drain is empty (the queue was drained, not copied)"
        );
    }

    #[test]
    fn drain_graph_records_returns_pushed_records_then_empties() {
        let r = runner();
        assert!(r.drain_graph_records().is_empty());

        let g0 = GraphRecord {
            stones: vec![(1i16, 2i16, 1i8)],
            visits: vec![(1i16, 2i16, 0.5f32)],
            current_player: 1,
            moves_remaining: 2,
            ply_index: 4,
            is_full_search: true,
            outcome: 1.0,
            value_valid: true,
            game_length: 8,
        };
        let g1 = GraphRecord {
            current_player: -1,
            ply_index: 5,
            ..GraphRecord::default()
        };
        {
            let mut q = r.graph_results.lock().expect("graph_results lock poisoned");
            q.push_back(g0.clone());
            q.push_back(g1.clone());
        }

        assert_eq!(r.drain_graph_records(), vec![g0, g1]);
        assert!(r.drain_graph_records().is_empty());
    }

    #[test]
    fn stats_snapshot_reads_back_each_private_atomic() {
        let r = runner();
        assert_eq!(
            r.stats_snapshot(),
            RunnerStatsSnapshot::default(),
            "a fresh runner reports all-zero counters"
        );

        // Distinct values 1..=24, one per atomic, in struct-field order — a getter
        // wired to the wrong atomic would read the wrong number and fail.
        r.games_completed.store(1, Ordering::Relaxed);
        r.positions_generated.store(2, Ordering::Relaxed);
        r.x_wins.store(3, Ordering::Relaxed);
        r.o_wins.store(4, Ordering::Relaxed);
        r.draws.store(5, Ordering::Relaxed);
        r.positions_dropped.store(6, Ordering::Relaxed);
        r.mcts_depth_accum.store(7, Ordering::Relaxed);
        r.mcts_conc_accum.store(8, Ordering::Relaxed);
        r.mcts_stat_count.store(9, Ordering::Relaxed);
        r.mcts_quiescence_fires.store(10, Ordering::Relaxed);
        r.cluster_value_std_accum.store(11, Ordering::Relaxed);
        r.cluster_policy_disagreement_accum.store(12, Ordering::Relaxed);
        r.cluster_variance_samples.store(13, Ordering::Relaxed);
        r.solver_moves_eligible.store(14, Ordering::Relaxed);
        r.solver_win_proven.store(15, Ordering::Relaxed);
        r.solver_injected.store(16, Ordering::Relaxed);
        r.solver_injected_offwindow.store(17, Ordering::Relaxed);
        r.solver_budget_exhausted.store(18, Ordering::Relaxed);
        r.solver_moves_eligible_seeded.store(19, Ordering::Relaxed);
        r.solver_injected_seeded.store(20, Ordering::Relaxed);
        r.seeded_games_started.store(21, Ordering::Relaxed);
        r.export_offwindow_mass_moves.store(22, Ordering::Relaxed);
        r.gridls_zero_policy_rows.store(23, Ordering::Relaxed);
        r.target_integrity_defects.store(24, Ordering::Relaxed);

        let expected = RunnerStatsSnapshot {
            games_completed: 1,
            positions_generated: 2,
            x_wins: 3,
            o_wins: 4,
            draws: 5,
            positions_dropped: 6,
            mcts_depth_accum: 7,
            mcts_conc_accum: 8,
            mcts_stat_count: 9,
            mcts_quiescence_fires: 10,
            cluster_value_std_accum: 11,
            cluster_policy_disagreement_accum: 12,
            cluster_variance_samples: 13,
            solver_moves_eligible: 14,
            solver_win_proven: 15,
            solver_injected: 16,
            solver_injected_offwindow: 17,
            solver_budget_exhausted: 18,
            solver_moves_eligible_seeded: 19,
            solver_injected_seeded: 20,
            seeded_games_started: 21,
            export_offwindow_mass_moves: 22,
            gridls_zero_policy_rows: 23,
            target_integrity_defects: 24,
        };
        assert_eq!(
            r.stats_snapshot(),
            expected,
            "snapshot maps each atomic 1:1 with no crosswiring"
        );
    }

    #[test]
    fn model_version_setter_and_getter_roundtrip() {
        let r = runner();
        assert_eq!(r.model_version(), 0, "default model version is 0 (no-NN)");
        r.set_model_version(42);
        assert_eq!(r.model_version(), 42);
        r.set_model_version(7);
        assert_eq!(r.model_version(), 7, "a later set overwrites");
    }
}
