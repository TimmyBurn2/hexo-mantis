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

/// Per-row training tuple produced by self-play workers (frozen `mod.rs:44`).
/// Field order: `(feat, chain, policy, outcome, plies, combined_aux_u8,
/// is_full_search, ply_index, value_valid)`. The P-04 pin destructures this
/// carrier exhaustively — a carrier-type change bites.
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

/// Per-game result tuple consumed by [`SelfPlayRunner::drain_game_results`]
/// (frozen `mod.rs:54`). Field order: `(plies, winner_code, move_history,
/// worker_id, terminal_reason, model_version_min, model_version_max,
/// model_version_distinct)`. The `seeded` / `solver_fires` slots went with the
/// seed-corpus and solver levers (R346(f)).
pub type GameResultRow = (usize, u8, Vec<(i32, i32)>, usize, u8, u64, u64, u32);

/// Flat snapshot of the runner's LAW-18 in-run counter atomics, each read once
/// via a single `Relaxed` load (the WP7-owed READ side of the write-only fire
/// counters — see the module doc). RAW cumulative counts ONLY: the fixed-point
/// ×1_000_000 accumulators (`*_accum`) are handed back UNDIVIDED so the WP7 bridge
/// derives the 4 means itself (`accum / (count × 1e6)`; the SEAM does NOT compute
/// means). ADJ-D32 / R249: the two cluster means are `None` at `count == 0` — a mean
/// over zero samples is not a measurement — while the two MCTS means keep a `0.0`
/// zero-guard, because their count advances on every arm and its zero is transient.
/// The distinction lives at the bridge; this snapshot carries only raw counts.
/// Every field maps 1:1 to a [`SelfPlayRunner`]
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
    /// R335(c) — the largest leaf count ANY one search served. Must never exceed the
    /// search budget; `search_drive::run_mcts_search` `fetch_max`es it per search.
    pub max_sims_per_search: u64,
    /// LAW-18 — playout-cap randomization's fire rate, counted at the DRAW. `full + quick`
    /// is every searched move; a run with `full_search_prob == 0` counts every move `full`,
    /// because the un-randomized arm searches the game budget.
    pub pcr_full_moves: u64,
    pub pcr_quick_moves: u64,
    /// LAW-18 — the Gumbel halving round's WIDTH, as the two terms of a mean:
    /// `gumbel_round_leaves / gumbel_rounds` is leaves per inference round trip. BOTH zero
    /// on a PUCT run, which issues no rounds — a reader publishes the ABSENCE, never a 0/0.
    pub gumbel_round_leaves: u64,
    pub gumbel_rounds: u64,
    // ── WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6) ──
    /// Moves whose exported policy target carried off-window (overflow) mass.
    pub export_offwindow_mass_moves: u64,
    /// Fatal-defect latch fire count (must read 0 in a healthy run).
    pub target_integrity_defects: u64,
    /// R275(b) SEAM conjunct — leaf inferences that FAILED on an open queue and
    /// halted the run (must read 0 in a healthy run; a drain shutdown does NOT
    /// count here, `search_drive::InferenceSeamFailure`). Encoding-INDEPENDENT
    /// (R250/R256 mapping re-derived from code): both `infer_and_expand` arms —
    /// the dense queue and the graph queue — have a failure leg, so the mechanism
    /// is live on every arm and the counter is published on every arm.
    pub inference_failures_total: u64,
    /// Worker threads that died by panic (must read 0 in a healthy run).
    pub worker_panics: u64,
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
    /// budget).
    config: SelfPlayRunnerConfig,
    /// HEXG visit-slot capacity, DERIVED once at composition from the sims
    /// regime (`replay::hexg::derived_visit_capacity`, R255/ADJ-D34). `None` on
    /// grid runs — dense-362 records carry no visit slot; never a default.
    visit_capacity: Option<usize>,

    graph_queue: GraphQueue,

    // ── shared result queues ──
    results: Arc<Mutex<VecDeque<WorkerResultRow>>>,
    graph_results: Arc<Mutex<VecDeque<GraphRecord>>>,
    recent_game_results: Arc<Mutex<VecDeque<GameResultRow>>>,

    // ── control ──
    running: Arc<AtomicBool>,
    handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    /// Worker threads that died by panic. MUST read 0 in a healthy run.
    ///
    /// Before this counter existed a panicking worker was invisible: `thread::spawn`
    /// captures the panic in its `JoinHandle`, `stop()` discarded that result
    /// (`let _ = handle.join()`), and `running` stayed `true` — so the pool silently ran
    /// with fewer workers, or none, and presented as "slow" rather than "broken". Written
    /// from two places: `spawn::WorkerPanicGuard` on the unwind itself (live, mid-run) and
    /// `stop()`'s join-result check (belt-and-braces, at shutdown).
    worker_panics: Arc<AtomicU64>,

    /// WP7 NN model-version snapshot source. Each worker reads this once per move and
    /// dedup-pushes it into `version_seen` (drain tuple `mv_min/mv_max/mv_distinct`).
    /// Defaults to 0 (no-NN); WP7 wires the real setter when the NN producer lands.
    model_version: Arc<AtomicU64>,

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
    max_sims_per_search: Arc<AtomicU64>,
    pcr_full_moves: Arc<AtomicU64>,
    pcr_quick_moves: Arc<AtomicU64>,
    gumbel_round_leaves: Arc<AtomicU64>,
    gumbel_rounds: Arc<AtomicU64>,

    // ── WP12-R Phase T target-integrity surfaces (LAW-18 / LAW-14) ──
    export_offwindow_mass_moves: Arc<AtomicU64>,
    target_integrity_defects: Arc<AtomicU64>,
    /// R275(b) SEAM conjunct fire count (see the snapshot field).
    inference_failures_total: Arc<AtomicU64>,
    /// R345(b)(6): the monotonic graph-game id source. See `WorkerAtomics::graph_game_seq`.
    graph_game_seq: Arc<AtomicU64>,
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
        // `n_simulations` — the ONE resolution rule, shared with the capacity
        // derivation below (`effective_standard_sims`). Reject zero on the
        // *effective* value.
        let effective_standard = crate::replay::hexg::effective_standard_sims(
            config.n_simulations,
            config.standard_sims,
        );
        if effective_standard == 0 {
            return Err("SelfPlayRunner: n_simulations (or standard_sims) must be > 0".to_string());
        }
        // The Gumbel kind reaches the root's FULL legal set, so it spends up to
        // `MAX_ROOT_CHILDREN` slots on the root instead of `MAX_CHILDREN_PER_NODE`, and its
        // ceiling is correspondingly lower. The PUCT bound is unchanged: a PUCT root is an
        // ordinary node and the original derivation is exact for it.
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
        // AUDIT-1 F-21: the pool bound, checked at BOOT rather than at the first move that
        // crosses it. Every sims knob the search can be driven at is checked, not only the
        // standard one, because a `fast_sims` or `n_sims_full` above the bound overflows the
        // same pool.
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
        // AUDIT-1 F-38. `sample_dirichlet` builds `Gamma::new(alpha, 1.0).expect(...)`, and
        // the guards above it are `debug_assert!` — dead in the shipped `.so`. Only pydantic's
        // `gt=0` protected a MINTED config; a hand-built runner spec (a test, a future
        // non-YAML source, an arithmetic slip upstream) reached the expect and panicked mid
        // self-play.
        //
        // NaN-SAFE AND CLIPPY-CLEAN. The obvious `x <= 0.0` is WRONG — it is FALSE for NaN, so
        // a NaN alpha would pass the guard and blow up inside `Gamma::new` anyway. The obvious
        // fix, `!(x > 0.0)`, is correct but trips `clippy::neg_cmp_op_on_partial_ord` (in
        // `clippy::all`, hence gate 2b). `partial_cmp` says the same thing explicitly: NaN
        // compares as `None`, which is not `Some(Greater)`, so it is refused.
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

        // WP12-R Phase T boot guard, re-ruled by R255/ADJ-D34 (read EXISTING keys only,
        // R120; armed VALUES are never set, R119). The slot capacity is DERIVED from the
        // configured sims regime by the ONE authority `derived_visit_capacity` — the SAME fn
        // the mint-time schema validator calls through the bridge, so an unsupported regime
        // REDs at config validation and this call is the defense-in-depth line.
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

        // The collector's saturation threshold is DERIVED from what this run can supply
        // (ledger F-1): a worker blocks on its whole submitted batch, so `n_workers x
        // leaf_batch_size` is a hard cap on queue depth and the threshold is clamped to it.
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

    /// WP12-R Phase T fatal-defect latch (DESIGN_T §3.4; LAW-14): store the
    /// typed defect message (first defect wins — the latch is write-once),
    /// count the fire, THEN flip `running=false` (store-then-halt) so the
    /// supervisor-facing drain can always read the reason for the halt. A
    /// worker panic is NOT sufficient — `stop()` swallows join results.
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

    /// Worker threads that have died by panic. Reads 0 in a healthy run.
    #[must_use]
    pub fn worker_panics(&self) -> u64 {
        self.worker_panics.load(Ordering::SeqCst)
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
        self.graph_queue.close();
        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        while let Some(handle) = handles.pop() {
            // CHECKED, not discarded. `Err` here means the thread unwound all the way OUT
            // of the spawn closure — which the normal path cannot do, because the closure
            // wraps `run_worker_thread` in `catch_unwind` and counts the panic itself
            // (`spawn.rs`). So this arm double-counts nothing; it is the escape hatch for a
            // panic raised outside that `catch_unwind` (in the closure's own prologue, or a
            // panic while panicking). If it ever fires, the count is still right and the
            // alternative is the old behaviour: silence.
            if handle.join().is_err() {
                self.worker_panics.fetch_add(1, Ordering::SeqCst);
            }
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
        let mut rows = self
            .graph_results
            .lock()
            .expect("graph_results lock poisoned");
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

    /// PRODUCER handle for the graph inference queue (mock producer in tests; the NN
    /// producer face in prod). The queue is `Clone` (shares one `Arc` inner), so this hands
    /// out a live handle that `pop_graph_batch` + `submit_graph_results`.
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

    // ── worker-panic propagation (item 3) ────────────────────────────────────────────
    //
    // The defect these pin: a panicking worker was parked in its `JoinHandle`, `stop()`
    // discarded the result, and `running` stayed true — so the pool reported healthy while
    // producing nothing. Every test below injects a REAL panic; none simulate one.

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

    /// Mutation self-test (LAW-07): the arm must stay silent on a clean worker.
    ///
    /// Mechanism: `guard_worker` fires only on `catch_unwind` returning `Err`, so a body
    /// that returns normally must leave both the counter and the flag untouched. Without
    /// this, an arm that counted unconditionally would pass the test above while making
    /// `worker_panics` meaningless — a counter that always reads non-zero reports nothing.
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

    /// The escape arm: `stop()` must CHECK the join result, not discard it.
    ///
    /// A handle that panicked outside `guard_worker` is pushed straight onto the runner's
    /// handle list — the one state `stop()` reads — and `stop()` must come back with the
    /// panic counted. Before the fix this was `let _ = handle.join()` and the count stayed 0.
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

        // DISTINCT values, one per atomic, in struct-field order — a getter wired
        // to the wrong atomic would read the wrong number and fail. The values are
        // never reused across fields (the histogram occupies its own contiguous
        // run), so an adjacent-field miswire cannot pass by coincidence.
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
