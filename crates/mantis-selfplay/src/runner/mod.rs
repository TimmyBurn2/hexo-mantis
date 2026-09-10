//! R8-justify: the pure-Rust `SelfPlayRunner` core — the queue-owning accumulator struct, its
//! resolving ctor and the start/stop/drain lifecycle — is one indivisible unit: the phase bodies
//! split into sibling modules, but the struct and its lifecycle stay together so the ownership
//! story is greppable in one file. No pyo3 lives here.

pub mod atomics;
pub mod config;
pub mod finalize;
pub mod game;
pub mod params;
pub mod record;
pub mod search_drive;
pub mod spawn;
pub mod stats;

pub use config::SelfPlayRunnerConfig;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use mantis_encoding::{all_specs, lookup, RegistrySpec};

use crate::queues::GraphQueue;
use crate::replay::hexg::GraphRecord;

/// Per-row training tuple produced by self-play workers. The P-04 pin destructures this carrier
/// exhaustively, so a carrier-type change bites.
pub type WorkerResultRow = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    f32,
    usize,
    Vec<u8>,
    bool,
    u16,
    u8,
);

/// Per-game result tuple consumed by [`SelfPlayRunner::drain_game_results`].
pub type GameResultRow = (usize, u8, Vec<(i32, i32)>, usize, u8, u64, u64, u32);

/// Flat snapshot of the runner's in-run counter atomics, each read once via a `Relaxed` load.
/// RAW cumulative counts ONLY: the fixed-point accumulators are handed back UNDIVIDED so the
/// bridge can derive the means, since a cluster mean over zero samples is not a measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunnerStatsSnapshot {
    pub games_completed: usize,
    pub positions_generated: usize,
    pub x_wins: u64,
    pub o_wins: u64,
    pub draws: u64,
    pub positions_dropped: u64,
    // MCTS-health accumulators; `*_accum` are fixed-point x1e6 and the bridge derives the means.
    pub mcts_depth_accum: u64,
    pub mcts_conc_accum: u64,
    pub mcts_stat_count: u64,
    pub mcts_quiescence_fires: u64,
    /// The largest leaf count ANY one search served; must never exceed the search budget.
    pub max_sims_per_search: u64,
    /// Playout-cap randomization's fire rate, counted at the DRAW; at `full_search_prob == 0`
    /// every move counts `full`.
    pub pcr_full_moves: u64,
    pub pcr_quick_moves: u64,
    /// The Gumbel round's WIDTH as the two terms of a mean. BOTH zero on a PUCT run, whose
    /// reader publishes the ABSENCE rather than a 0/0.
    pub gumbel_round_leaves: u64,
    pub gumbel_rounds: u64,
    /// Moves whose exported policy target carried off-window (overflow) mass.
    pub export_offwindow_mass_moves: u64,
    /// Fatal-defect latch fire count (must read 0 in a healthy run).
    pub target_integrity_defects: u64,
    /// Leaf inferences that FAILED on an open queue and halted the run; a drain shutdown does
    /// NOT count here.
    pub inference_failures_total: u64,
    /// Worker threads that died by panic (must read 0 in a healthy run).
    pub worker_panics: u64,
}

/// Pure-Rust self-play runner core: spawns worker threads that run full games, stream training
/// rows into the result queues and track win stats plus fire-rate counters. Every worker OWNS its
/// `Board` — there is NO shared-board Arc.
pub struct SelfPlayRunner {
    /// Resolved encoding spec; never `None`, since an absent identity key is rejected at `new()`.
    spec: &'static RegistrySpec,
    /// Runner config, with `standard_sims` already resolved to the effective budget.
    config: SelfPlayRunnerConfig,
    /// HEXG visit-slot capacity, DERIVED once at composition from the sims regime. `None` on
    /// grid runs — dense-362 records carry no visit slot; never a default.
    visit_capacity: Option<usize>,

    graph_queue: GraphQueue,

    results: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    graph_results: Arc<Mutex<VecDeque<GraphRecord>>>,
    recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,

    running: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Worker threads that died by panic. MUST read 0 in a healthy run: before this counter a
    /// panicking worker was invisible, since the panic sat in its `JoinHandle`, `stop()`
    /// discarded it and `running` stayed `true`, so the pool ran with fewer workers and presented
    /// as "slow" rather than "broken".
    worker_panics: Arc<AtomicU64>,

    /// Model-version snapshot source, read once per move by each worker and dedup-pushed into
    /// the drain tuple. Defaults to 0 on a no-NN run.
    model_version: Arc<AtomicU64>,

    games_completed: Arc<AtomicUsize>,
    positions_generated: Arc<AtomicUsize>,
    x_wins: Arc<AtomicU64>,
    o_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    positions_dropped: Arc<AtomicU64>,

    mcts_depth_accum: Arc<AtomicU64>,
    mcts_conc_accum: Arc<AtomicU64>,
    mcts_stat_count: Arc<AtomicU64>,
    mcts_quiescence_fires: Arc<AtomicU64>,
    max_sims_per_search: Arc<AtomicU64>,
    pcr_full_moves: Arc<AtomicU64>,
    pcr_quick_moves: Arc<AtomicU64>,
    gumbel_round_leaves: Arc<AtomicU64>,
    gumbel_rounds: Arc<AtomicU64>,

    export_offwindow_mass_moves: Arc<AtomicU64>,
    target_integrity_defects: Arc<AtomicU64>,
    /// R275(b) SEAM conjunct fire count (see the snapshot field).
    inference_failures_total: Arc<AtomicU64>,
    /// R345(b)(6): the monotonic graph-game id source. See `WorkerAtomics::graph_game_seq`.
    graph_game_seq: Arc<AtomicU64>,
    /// The fatal-defect latch: a worker panic is NOT loud, since `stop()` swallows join results,
    /// so a `TargetIntegrityError` stores its message here and the drain face raises it typed.
    fatal_defect: Arc<Mutex<Option<String>>>,
}

impl SelfPlayRunner {
    /// Construct a runner from a native [`SelfPlayRunnerConfig`], resolving `encoding_name` to a
    /// `&'static RegistrySpec` and running the sim-budget / playout-cap validations.
    ///
    /// # Errors
    /// Returns `Err(msg)` when `encoding_name` is absent or unknown, or when a sim-budget /
    /// playout-cap invariant is violated.
    pub fn new(mut config: SelfPlayRunnerConfig) -> Result<Self, String> {
        // Absent spec = error (no `None -> v6` fallback); unknown name = error naming it.
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

        // Effective standard-search sim budget: `standard_sims` wins, else `n_simulations` — the
        // ONE resolution rule, and zero is rejected on the *effective* value.
        let effective_standard = crate::replay::hexg::effective_standard_sims(
            config.n_simulations,
            config.standard_sims,
        );
        if effective_standard == 0 {
            return Err("SelfPlayRunner: n_simulations (or standard_sims) must be > 0".to_string());
        }
        // The Gumbel kind reaches the root's FULL legal set, so it spends up to
        // `MAX_ROOT_CHILDREN` slots there and its ceiling is correspondingly lower.
        let (armed_ceiling, ceiling_name, ceiling_derivation) = match config.search_kind {
            mantis_search::SearchKind::Gumbel => (
                mantis_search::MAX_ARMED_SIMS_GUMBEL,
                "MAX_ARMED_SIMS_GUMBEL",
                "(MAX_NODES - MAX_ROOT_CHILDREN) / (4 * MAX_CHILDREN_PER_NODE)",
            ),
            mantis_search::SearchKind::Puct => (
                mantis_search::MAX_ARMED_SIMS,
                "MAX_ARMED_SIMS",
                "MAX_NODES / (4 * MAX_CHILDREN_PER_NODE)",
            ),
        };
        // The pool bound, checked at BOOT rather than at the first move that crosses it, over
        // EVERY sims knob the search can be driven at.
        for (name, sims) in [
            ("n_simulations", config.n_simulations),
            ("standard_sims", config.standard_sims),
            ("fast_sims", config.fast_sims),
            ("n_sims_quick", config.n_sims_quick),
            ("n_sims_full", config.n_sims_full),
        ] {
            if sims > armed_ceiling {
                return Err(format!(
                    "SelfPlayRunner: {name} = {sims} exceeds {ceiling_name} \
                     ({armed_ceiling}), derived as {ceiling_derivation}. \
                     `select_leaves` expands TT-hit leaves without counting them against the \
                     batch (bounded by max_attempts = 4n), so each move can add up to \
                     4 * sims * {} children and `finish_expansion` panics on pool overflow \
                     at the first move that crosses it",
                    mantis_search::MAX_CHILDREN_PER_NODE,
                ));
            }
        }
        if config.fast_prob > 0.0 && config.fast_sims == 0 {
            return Err("SelfPlayRunner: fast_sims must be > 0 when fast_prob > 0".to_string());
        }
        // `sample_dirichlet` builds `Gamma::new(alpha, 1.0).expect(...)` behind `debug_assert!`
        // guards dead in the shipped `.so`, so only pydantic's `gt=0` protected a MINTED config.
        // NaN-SAFE AND CLIPPY-CLEAN: `x <= 0.0` is FALSE for NaN and `!(x > 0.0)` trips
        // `clippy::neg_cmp_op_on_partial_ord`, while `partial_cmp` says it explicitly.
        if config.dirichlet_enabled
            && config.dirichlet_alpha.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        {
            return Err(format!(
                "SelfPlayRunner: dirichlet_alpha must be > 0 when dirichlet_enabled, got {} \
                 — the Gamma distribution behind the root noise cannot be built otherwise, \
                 and the guard inside `sample_dirichlet` is a debug_assert (absent in release)",
                config.dirichlet_alpha
            ));
        }
        if config.full_search_prob > 0.0 && (config.n_sims_quick == 0 || config.n_sims_full == 0) {
            let (n_sims_quick, n_sims_full) = (config.n_sims_quick, config.n_sims_full);
            return Err(format!(
                "SelfPlayRunner: n_sims_quick and n_sims_full must both be > 0 \
                 when full_search_prob > 0 (got n_sims_quick={n_sims_quick}, n_sims_full={n_sims_full})"
            ));
        }

        // Boot guard reading EXISTING keys only: capacity is DERIVED by the same authority the
        // mint-time validator calls, so this is defense in depth.
        let visit_capacity = Some(
            crate::replay::hexg::derived_visit_capacity(
                config.n_simulations,
                config.standard_sims,
                config.fast_prob,
                config.fast_sims,
                config.full_search_prob,
                config.n_sims_quick,
                config.n_sims_full,
                config.leaf_batch_size,
                config.gumbel_m,
                config.search_kind.as_config_str(),
            )
            .map_err(|e| format!("SelfPlayRunner: {e}"))?,
        );

        // Bake the resolved budget so the workers read the effective value.
        config.standard_sims = effective_standard;

        // The collector's saturation threshold is DERIVED from what this run can supply: a worker
        // blocks on its whole submitted batch, so `n_workers x leaf_batch_size` caps queue depth.
        let max_in_flight = config.n_workers.saturating_mul(config.leaf_batch_size);
        let graph_queue = GraphQueue::with_contract_version_and_supply(
            spec.contract_version.unwrap_or(1),
            max_in_flight,
        );

        Ok(Self {
            spec,
            config,
            visit_capacity,
            graph_queue,
            results: Arc::new(Mutex::new(VecDeque::new())),
            graph_results: Arc::new(Mutex::new(VecDeque::new())),
            recent_game_results: Arc::new(Mutex::new(VecDeque::new())),
            running: Arc::new(AtomicBool::new(false)),
            handles: Arc::new(Mutex::new(Vec::new())),
            worker_panics: Arc::new(AtomicU64::new(0)),
            model_version: Arc::new(AtomicU64::new(0)),
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
            max_sims_per_search: Arc::new(AtomicU64::new(0)),
            pcr_full_moves: Arc::new(AtomicU64::new(0)),
            pcr_quick_moves: Arc::new(AtomicU64::new(0)),
            gumbel_round_leaves: Arc::new(AtomicU64::new(0)),
            gumbel_rounds: Arc::new(AtomicU64::new(0)),
            export_offwindow_mass_moves: Arc::new(AtomicU64::new(0)),
            target_integrity_defects: Arc::new(AtomicU64::new(0)),
            inference_failures_total: Arc::new(AtomicU64::new(0)),
            graph_game_seq: Arc::new(AtomicU64::new(0)),
            fatal_defect: Arc::new(Mutex::new(None)),
        })
    }

    /// Store the typed defect message (first wins — the latch is write-once), count the fire,
    /// THEN flip `running=false`, so the drain can always read the reason for the halt.
    pub fn store_fatal_defect(&self, msg: String) {
        {
            let mut slot = self
                .fatal_defect
                .lock()
                .expect("fatal_defect lock poisoned");
            if slot.is_none() {
                *slot = Some(msg);
            }
        }
        self.target_integrity_defects.fetch_add(1, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }

    /// Read the stored fatal defect, if any — the bridge drain face raises it as a typed Python
    /// exception so the pool drain loop dies with the variant name.
    #[must_use]
    pub fn fatal_defect(&self) -> Option<String> {
        self.fatal_defect
            .lock()
            .expect("fatal_defect lock poisoned")
            .clone()
    }

    /// Worker threads that have died by panic. Reads 0 in a healthy run.
    #[must_use]
    pub fn worker_panics(&self) -> u64 {
        self.worker_panics.load(Ordering::SeqCst)
    }

    /// Spawn `n_workers` self-play threads (idempotent). See [`spawn`].
    pub fn start(&self) {
        self.start_impl();
    }

    /// Flip `running=false`, close both inference queues (waking blocked waiters with `Err`),
    /// and join all worker threads. An in-progress game is DROPPED, never finalized as a draw.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.graph_queue.close();
        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            // CHECKED, not discarded. `Err` here means the thread unwound OUT of the spawn
            // closure, which the normal path cannot do, so this double-counts nothing and is the
            // escape hatch for a panic raised outside the guard.
            if handle.join().is_err() {
                self.worker_panics.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Drain and return all buffered game results since the last call.
    pub fn drain_game_results(&self) -> Vec<GameResultRow> {
        let mut rg = self
            .recent_game_results
            .lock()
            .expect("recent_game_results lock poisoned");
        rg.drain(..).collect()
    }

    // Narrow pub read/drain faces the bridge producer pyclasses build over. None of these mutate
    // self beyond the drain queues they own.

    /// Drain and return all buffered training rows since the last call, in FIFO push order.
    pub fn drain_training_rows(&self) -> Vec<WorkerResultRow> {
        let mut rows = self.results.lock().expect("results lock poisoned");
        rows.drain(..).collect()
    }

    /// Drain and return all buffered graph training records since the last call, FIFO.
    pub fn drain_graph_records(&self) -> Vec<GraphRecord> {
        let mut rows = self
            .graph_results
            .lock()
            .expect("graph_results lock poisoned");
        rows.drain(..).collect()
    }

    /// Set the shared model-version snapshot workers read once per move and dedup-push into the
    /// drain tuple. `0` = no-NN.
    pub fn set_model_version(&self, version: u64) {
        self.model_version.store(version, Ordering::SeqCst);
    }

    /// Current NN model-version snapshot (`0` = no-NN).
    #[must_use]
    pub fn model_version(&self) -> u64 {
        self.model_version.load(Ordering::SeqCst)
    }

    /// Snapshot the in-run counter atomics with one `Relaxed` load each, returning RAW
    /// cumulative counts — the `*_accum` fixed-point sums are NOT divided here.
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
            max_sims_per_search: self.max_sims_per_search.load(Ordering::Relaxed),
            pcr_full_moves: self.pcr_full_moves.load(Ordering::Relaxed),
            pcr_quick_moves: self.pcr_quick_moves.load(Ordering::Relaxed),
            gumbel_round_leaves: self.gumbel_round_leaves.load(Ordering::Relaxed),
            gumbel_rounds: self.gumbel_rounds.load(Ordering::Relaxed),
            export_offwindow_mass_moves: self.export_offwindow_mass_moves.load(Ordering::Relaxed),
            target_integrity_defects: self.target_integrity_defects.load(Ordering::Relaxed),
            inference_failures_total: self.inference_failures_total.load(Ordering::Relaxed),
            worker_panics: self.worker_panics.load(Ordering::Relaxed),
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

    /// PRODUCER handle for the graph inference queue. The queue is `Clone` over one `Arc`
    /// inner, so this hands out a live handle.
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

/// Round-trip gate: the pub read/drain faces return exactly what the private queues and counter
/// atomics hold. No workers are spawned, so private state is populated deterministically in-test
/// and read back through the pub API only, with distinct per-field values so a crosswired getter
/// FAILS.
#[cfg(test)]
mod seam_roundtrip {
    use std::sync::atomic::Ordering;

    use crate::replay::hexg::GraphRecord;

    use super::{RunnerStatsSnapshot, SelfPlayRunner, SelfPlayRunnerConfig, WorkerResultRow};

    /// Minimal valid runner: only the identity key is required, and no worker is started.
    fn runner() -> SelfPlayRunner {
        SelfPlayRunner::new(SelfPlayRunnerConfig {
            encoding_name: Some("gnn_axis_v1".to_string()),
            ..Default::default()
        })
        .expect("gnn_axis_v1 must resolve via the registry")
    }

    #[test]
    fn drain_training_rows_returns_pushed_rows_then_empties() {
        let r = runner();
        assert!(
            r.drain_training_rows().is_empty(),
            "fresh runner has no training rows"
        );

        let row0: WorkerResultRow = (
            vec![1.0, 2.0],
            vec![3.0],
            vec![0.5],
            1.0,
            7,
            vec![9u8],
            true,
            4u16,
            1u8,
        );
        let row1: WorkerResultRow = (
            vec![-1.0],
            vec![],
            vec![0.25, 0.75],
            -0.1,
            3,
            vec![],
            false,
            2u16,
            0u8,
        );
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
            tail_mass: 0.0,
            current_player: 1,
            moves_remaining: 2,
            ply_index: 4,
            is_full_search: true,
            outcome: 1.0,
            value_valid: true,
            game_length: 8,
            game_id: -1,
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

    // Worker-panic propagation. The defect these pin: a panicking worker was parked in its
    // `JoinHandle`, `stop()` discarded the result, and `running` stayed true, so the pool
    // reported healthy while producing nothing. Every test below injects a REAL panic.

    /// The live arm, driving the SAME `guard_worker` the spawn closure calls.
    #[test]
    fn injected_worker_panic_is_counted_and_halts_the_run() {
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let panics = AtomicU64::new(0);
        let running = AtomicBool::new(true);

        crate::runner::spawn::guard_worker(&panics, &running, || {
            panic!("injected worker panic");
        });

        assert_eq!(
            panics.load(Ordering::SeqCst),
            1,
            "the panic was not COUNTED"
        );
        assert!(
            !running.load(Ordering::SeqCst),
            "the panic was counted but the run was not HALTED — the pool would keep \
             reporting healthy with a dead worker, which is the original defect"
        );
    }

    /// Mutation self-test: the arm stays silent on a clean worker.
    ///
    /// `guard_worker` fires only on `catch_unwind` returning `Err`. Without this, an arm that
    /// counted unconditionally would pass the test above while making the counter meaningless.
    #[test]
    fn a_clean_worker_neither_counts_nor_halts() {
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let panics = AtomicU64::new(0);
        let running = AtomicBool::new(true);

        crate::runner::spawn::guard_worker(&panics, &running, || { /* returns normally */ });

        assert_eq!(
            panics.load(Ordering::SeqCst),
            0,
            "counted a panic that never happened"
        );
        assert!(
            running.load(Ordering::SeqCst),
            "halted a run over a healthy worker"
        );
    }

    /// The escape arm: `stop()` must CHECK the join result, not discard it. A handle that
    /// panicked outside `guard_worker` is pushed straight onto the handle list `stop()` reads;
    /// before the fix this was `let _ = handle.join()` and the count stayed 0.
    #[test]
    fn stop_counts_a_panic_that_escaped_the_guard() {
        let r = runner();
        assert_eq!(r.worker_panics(), 0);

        r.handles
            .lock()
            .expect("handles lock")
            .push(std::thread::spawn(|| panic!("escaped worker panic")));

        r.stop();

        assert_eq!(
            r.worker_panics(),
            1,
            "stop() swallowed a join Err — a worker died and nothing recorded it"
        );
        assert_eq!(
            r.stats_snapshot().worker_panics,
            1,
            "the count did not reach the stats"
        );
    }

    /// A clean worker must not be counted by the `stop()` arm either.
    #[test]
    fn stop_does_not_count_a_worker_that_exited_cleanly() {
        let r = runner();
        r.handles
            .lock()
            .expect("handles lock")
            .push(std::thread::spawn(|| {}));
        r.stop();
        assert_eq!(
            r.worker_panics(),
            0,
            "a clean thread exit was counted as a panic"
        );
    }

    #[test]
    fn stats_snapshot_reads_back_each_private_atomic() {
        let r = runner();
        assert_eq!(
            r.stats_snapshot(),
            RunnerStatsSnapshot::default(),
            "a fresh runner reports all-zero counters"
        );

        // DISTINCT values, one per atomic, in struct-field order, never reused across fields
        // (the histogram occupies its own contiguous run), so an adjacent-field miswire cannot
        // pass by coincidence.
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
        r.max_sims_per_search.store(50, Ordering::Relaxed);
        r.pcr_full_moves.store(37, Ordering::Relaxed);
        r.pcr_quick_moves.store(38, Ordering::Relaxed);
        r.gumbel_round_leaves.store(39, Ordering::Relaxed);
        r.gumbel_rounds.store(40, Ordering::Relaxed);
        r.export_offwindow_mass_moves.store(22, Ordering::Relaxed);
        r.target_integrity_defects.store(24, Ordering::Relaxed);
        r.worker_panics.store(25, Ordering::Relaxed);
        r.inference_failures_total.store(36, Ordering::Relaxed);

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
            max_sims_per_search: 50,
            pcr_full_moves: 37,
            pcr_quick_moves: 38,
            gumbel_round_leaves: 39,
            gumbel_rounds: 40,
            export_offwindow_mass_moves: 22,
            target_integrity_defects: 24,
            inference_failures_total: 36,
            worker_panics: 25,
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
