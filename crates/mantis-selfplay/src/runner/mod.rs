//! R8-justify: the pure-Rust `SelfPlayRunner` core (the ~40-field accumulator/
//! queue-owning struct + its resolving ctor + start/stop/drain lifecycle) is one
//! indivisible unit — the frozen `game_runner/mod.rs` was 1199 LOC; the phase
//! bodies split into sibling modules (`spawn`/`game`/…) but the struct and its
//! lifecycle stay together so the ownership story is greppable in one file.
//!
//! Self-play runner core (WP6 D1) — the pyo3-STRIPPED half of the frozen
//! `game_runner/mod.rs`. Owns the shared `Arc` accumulators, the dense + graph
//! inference queues (`crate::queues`), the result queues, the live curriculum
//! `radius_override`, and the LAW-18 in-run fire counters. `start`/`stop`/
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
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
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

    /// §174 live curriculum radius override (`-1` = no override). Non-jitter.
    radius_override: Arc<AtomicI32>,

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

        let radius_override = Arc::new(AtomicI32::new(config.radius_override.unwrap_or(-1)));

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
            radius_override,
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
        })
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

    /// §174 — update the per-game legal-move radius override live. `-1` clears the
    /// override (workers read this atomic at the start of each game). Non-jitter.
    pub fn set_radius_override(&self, radius: i32) {
        self.radius_override.store(radius, Ordering::SeqCst);
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
