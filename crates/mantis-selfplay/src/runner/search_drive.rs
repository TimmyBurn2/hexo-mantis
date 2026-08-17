//! R8-justify: the per-move search phase (`play_one_move` → `run_mcts_search` →
//! `infer_and_expand{,_graph}` → `select_move`) plus its Copy arg-bundles is one
//! cohesive port unit lifted from the frozen `inner.rs` search-drive band
//! (`:742-1099`); splitting the target-policy build order (temperature →
//! completed-Q → O1 → solver) across files would scatter the load-bearing ORDER.
//!
//! Search-drive phase (WP6 D1) — drives the WP4 `mantis_search` primitives. The
//! Gumbel Sequential-Halving arm steers the tree via the amended
//! `set_forced_root_child` setter (per phase: force a candidate → select/expand →
//! clear → `halve_candidates`); the PUCT arm applies Dirichlet root noise (never
//! under Gumbel). Leaves encode to the dense queue OR build+submit to the graph
//! queue; the graph path is rotation-free at inference (D-seam-3). The
//! target-policy build ORDER is EXACTLY: temperature-annealed visit policy →
//! optional completed-Q improved policy → O1 forced-win one-hot → D-WS3 solver
//! soft-inject (frozen `:1157/1163/1241/1291`).
//!
//! The per-move model-version snapshot (frozen `:1214`) is RESTORED: the model
//! version is read once per move from a runner-owned `model_version` atomic
//! (`InferContext::model_version`, default 0 until WP7 wires the NN setter) and
//! dedup-pushed into `version_seen`, so a no-NN run's drain tuple is byte-identical
//! to the frozen `(mv_min, mv_max, mv_distinct) = (0, 0, 1)`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::atomic::AtomicBool;

use rand::prelude::IndexedRandom;
use rand::rngs::ThreadRng;
use rand::RngExt;

use mantis_core::board::hex_distance;
use mantis_core::Board;
use mantis_encoding::{encode_state_to_buffer_channels, RegistrySpec};
use mantis_search::{
    compute_move_temperature, ply_to_compound_move, GumbelSearchState, LegalSetPolicy,
    MCTSTree, Outcome, TacticalConfig, TacticalSolver,
};

use crate::queues::{build_leaf_graph, DenseQueue, GraphQueue};
use crate::records;
use crate::replay::hexg::GraphRecord;

/// K-cluster value aggregation: the value head takes the WORST cluster view (min).
///
/// CARD-MINPIN (WPSC Phase 4): lifted verbatim from the frozen inline loop so the
/// min/max asymmetry — value pools min while policy pools best-scoring
/// (`records::aggregate_policy*`) — has ONE named, pinned home. This asymmetry is a
/// flagged defect preserved pending the matched-FLOP dense arm (falsified.md F-04
/// scope note; registry.toml `value_pool = "min"` comments). Bit-for-bit parity with
/// the old loop is pinned by `tests/min_value_aggregation_pin.rs`. `pub` solely so
/// the pin can reach it.
///
/// # Panics
/// Panics on an empty slice — a cluster with zero leaf values is unrepresentable
/// upstream (every expanded leaf contributes exactly one value per cluster view).
pub fn aggregate_cluster_values_min(leaf_values: &[f32]) -> f32 {
    let mut min_v = leaf_values[0];
    for &v in leaf_values {
        if v < min_v {
            min_v = v;
        }
    }
    min_v
}
use crate::replay::sym::SymTables;

use super::record::{
    record_position, record_position_graph_dispatch, RecordTuple, K_CLUSTER_HISTOGRAM_BUCKETS,
};
use super::rotate::{rotate_policy_inplace, rotate_state_inplace};

// ── Copy arg-bundles (frozen `inner.rs:63-183`) ─────────────────────────────

/// NN queue handles + per-game symmetry context. Replaces the frozen single
/// `InferenceBatcher` with the two disjoint queues (D4) + the `is_graph` hoist
/// bool (set FROM the geometry match, not `spec.is_graph()`) + the resolved spec
/// (graph-build geometry). `Copy` — passed by value.
#[derive(Clone, Copy)]
pub(crate) struct InferContext<'a> {
    pub(crate) dense_queue: &'a DenseQueue,
    pub(crate) graph_queue: &'a GraphQueue,
    pub(crate) sym_tables: &'static SymTables,
    pub(crate) sym_idx: usize,
    pub(crate) inv_idx: usize,
    pub(crate) is_graph: bool,
    pub(crate) spec: &'static RegistrySpec,
    /// Runner-owned model-version snapshot source (frozen `inner.rs:1214` =
    /// `batcher.current_model_version()`). Read once per move and dedup-pushed into
    /// `version_seen`. Default 0 (no-NN) until WP7 wires the real setter.
    pub(crate) model_version: &'a AtomicU64,
}

/// I2 cluster-variance accumulator triplet (frozen `:82`).
#[derive(Clone, Copy)]
pub(crate) struct ClusterVarianceAtomics<'a> {
    pub(crate) value_std_accum: &'a AtomicU64,
    pub(crate) policy_disagreement_accum: &'a AtomicU64,
    pub(crate) variance_samples: &'a AtomicU64,
}

/// Per-move MCTS accumulators + `positions_generated` (frozen `:94`).
/// WP12-R Phase T adds the two LAW-18 target-integrity counters (DESIGN_T §3.6,
/// the solver_counters pattern): `export_offwindow_mass_moves` fires once per
/// move whose exported target carries overflow mass; `gridls_zero_policy_rows`
/// fires per recorded grid-ls cluster row filled with the §3.5 zero-row sentinel.
/// Item 10(b) adds `k_cluster_histogram`, the DENSE record path's K distribution
/// (R250: structurally unreachable on the graph arm — `record_position` is the
/// only writer and the graph branch below never calls it).
#[derive(Clone, Copy)]
pub(crate) struct MoveAccumulators<'a> {
    pub(crate) mcts_depth_accum: &'a AtomicU64,
    pub(crate) mcts_conc_accum: &'a AtomicU64,
    pub(crate) mcts_stat_count: &'a AtomicU64,
    pub(crate) mcts_quiescence_fires: &'a AtomicU64,
    pub(crate) positions_generated: &'a AtomicUsize,
    pub(crate) export_offwindow_mass_moves: &'a AtomicU64,
    pub(crate) gridls_zero_policy_rows: &'a AtomicU64,
    pub(crate) k_cluster_histogram: &'a [AtomicU64; K_CLUSTER_HISTOGRAM_BUCKETS],
    /// R256/ADJ-D37 — proven forced wins swallowed by the LS coverage gate while
    /// the injecting lever was armed (`apply_forced_win_one_hot_ls_counted`, both
    /// the O1 arm and the solver hook). LS-path mechanism: ticks wherever
    /// `legal_set` targets are built; the EMITTER publishes it on the graph arm
    /// only (R256 — see `events.uncovered_forced_win_block`).
    pub(crate) uncovered_forced_win: &'a AtomicU64,
}

/// WP12-R Phase T fatal-defect latch handle (DESIGN_T §3.4; LAW-14). Store the
/// typed message (first defect wins), count the fire, THEN flip `running=false`
/// (store-then-halt) — a worker panic is NOT loud (`stop()` swallows joins), so
/// this is what makes the typed raise reach the supervisor via the drain face.
///
/// TWO counters, ONE slot (R275(b)): the message channel is shared because there
/// is one supervisor-facing halt reason, but the fire counts stay separate —
/// `fires` counts target-integrity refusals at the record dispatch, and
/// `inference_failures` counts seam failures. Folding them would make the two
/// conjuncts of the F-816-9 class indistinguishable in the event stream, which is
/// the one thing the in-run instrument exists to prevent (LAW-18).
#[derive(Clone, Copy)]
pub(crate) struct FatalDefectLatch<'a> {
    pub(crate) slot: &'a std::sync::Mutex<Option<String>>,
    pub(crate) fires: &'a AtomicU64,
    pub(crate) inference_failures: &'a AtomicU64,
    pub(crate) running: &'a AtomicBool,
}

impl FatalDefectLatch<'_> {
    pub(crate) fn store(&self, msg: String) {
        self.store_counted(msg, self.fires);
    }

    /// R275(b) SEAM conjunct: latch a named inference failure. Same store-then-halt
    /// ordering, its OWN counter.
    pub(crate) fn store_inference_failure(&self, msg: String) {
        self.store_counted(msg, self.inference_failures);
    }

    fn store_counted(&self, msg: String, counter: &AtomicU64) {
        {
            let mut slot = self.slot.lock().expect("fatal_defect lock poisoned");
            if slot.is_none() {
                *slot = Some(msg);
            }
        }
        counter.fetch_add(1, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }
}

/// D-WS3V3 in-run solver fire-rate counter refs (frozen `:110`). Incremented ONLY
/// under the `solver_enabled` / seeded branches, so an OFF run is byte-identical.
#[derive(Clone, Copy)]
pub(crate) struct SolverCounters<'a> {
    pub(crate) moves_eligible: &'a AtomicU64,
    pub(crate) win_proven: &'a AtomicU64,
    pub(crate) injected: &'a AtomicU64,
    pub(crate) injected_offwindow: &'a AtomicU64,
    pub(crate) budget_exhausted: &'a AtomicU64,
    pub(crate) moves_eligible_seeded: &'a AtomicU64,
    pub(crate) injected_seeded: &'a AtomicU64,
    pub(crate) seeded_games_started: &'a AtomicU64,
}

/// Per-move scalar context (frozen `:141`). `Copy` — mirrors the flat
/// `WorkerParams` layout plus the per-game dynamics (`game_sims`, `is_fast_game`,
/// `sym_idx`, `game_start_ply`, `seeded`).
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct MovePlayContext {
    pub(crate) leaf_batch_size: usize,
    /// DERIVED HEXG visit capacity (R255) — `Some` iff this is a graph run.
    pub(crate) visit_capacity: Option<usize>,
    pub(crate) temp_threshold: usize,
    pub(crate) temp_min: f32,
    pub(crate) zoi_lookback: usize,
    pub(crate) zoi_margin: i32,
    pub(crate) c_visit: f32,
    pub(crate) c_scale: f32,
    pub(crate) gumbel_m: usize,
    pub(crate) gumbel_explore_moves: usize,
    pub(crate) dirichlet_alpha: f32,
    pub(crate) dirichlet_epsilon: f32,
    pub(crate) full_search_prob: f32,
    pub(crate) n_sims_quick: usize,
    pub(crate) n_sims_full: usize,
    pub(crate) game_sims: usize,
    pub(crate) is_fast_game: bool,
    pub(crate) sym_idx: usize,
    pub(crate) completed_q_values: bool,
    pub(crate) gumbel_mcts: bool,
    pub(crate) dirichlet_enabled: bool,
    pub(crate) zoi_enabled: bool,
    pub(crate) forced_win_enabled: bool,
    pub(crate) forced_win_depth: u8,
    pub(crate) forced_win_weight: f32,
    pub(crate) solver_enabled: bool,
    pub(crate) solver_depth: u32,
    pub(crate) solver_node_budget: u64,
    pub(crate) solver_neighbor_dist: i32,
    pub(crate) solver_visit_weight: f32,
    /// D-WS3V3: absolute ply the organic play begins at (0 organic; == seed
    /// prefix_len for a seeded game). Gates Gumbel exploration RELATIVE to start.
    pub(crate) game_start_ply: usize,
    pub(crate) seeded: bool,
}

/// Per-move policy — either the dense scatter_max vector (byte-identical dense
/// path) or the ragged legal-set policy (frozen `:713`).
#[derive(Clone)]
pub(crate) enum MovePolicy {
    Dense(Vec<f32>),
    Ls(LegalSetPolicy),
}

impl MovePolicy {
    /// Sample a move from `legal` proportional to this policy's mass at each coord
    /// (the ragged variant uses the `1/n` no-coverage floor).
    fn sample(&self, legal: &[(i32, i32)], board: &Board, trunk: i32) -> Option<(i32, i32)> {
        match self {
            MovePolicy::Dense(p) => records::sample_policy(p, legal, board, trunk),
            MovePolicy::Ls(ls) => {
                let floor = 1.0 / legal.len().max(1) as f32;
                records::sample_policy_ls(ls, legal, board, trunk, floor)
            }
        }
    }
}

/// Result of `play_one_move`: how the parent per-move loop should proceed.
pub(crate) enum MoveOutcome {
    /// Move played — continue the inner loop.
    Played,
    /// Break (terminal / failure / shutdown).
    Break,
    /// Continue to next iteration (root expansion failed).
    Continue,
}

/// Result of `run_mcts_search`.
enum McTSSearchResult {
    Completed(Option<GumbelSearchState>),
    RootExpansionFailed,
    /// R275(b) SEAM conjunct: a leaf inference FAILED. The search is abandoned
    /// here and never reports `Completed` — see [`InferenceSeamFailure`].
    InferenceFailed(InferenceSeamFailure),
}

/// R275(b) SEAM conjunct — a leaf inference that FAILED, as distinct from a
/// shutdown (F-816-9 Phase A §4 links 1-3, §7.3).
///
/// Pre-fix every failure arm of `infer_and_expand{,_graph}` collapsed to
/// `return 0`: the reason string travelled back from the waiter verbatim and was
/// dropped on the floor, the sim loop `break`ed on `n == 0`, and
/// `run_mcts_search` still returned `Completed`. A search that backed up ZERO
/// visits then reached the target exporter, which manufactured a policy target
/// out of the ε-noise-mixed priors — and the failure resurfaced 100+ plies later
/// as a target-integrity refusal naming neither the failure nor the leaf. The
/// silent degradation is the crime: LAW-14 makes a failed inference run-fatal and
/// NAMED, at the seam, with the reason carried.
///
/// **A DRAIN SHUTDOWN IS NOT A FAILURE.** `stop()` flips `running=false` and then
/// closes both queues, waking every in-flight waiter with `Err`
/// (`runner/mod.rs::stop`); the §P22/D12 drain-shutdown path depends on those
/// `Err`s being a skip, not a defect. The discriminator is read from LIVE STATE
/// (`queue.is_closed()`), never from the reason text — a string match on
/// "closed" would be one upstream re-word away from turning every clean stop into
/// a run-fatal defect. A genuine failure racing a concurrent close classifies as
/// a shutdown; that is conservative in the safe direction (the run is ending
/// anyway) and the EXPORTER conjunct independently refuses any target the
/// degraded search could still produce.
pub(crate) struct InferenceSeamFailure {
    arm: &'static str,
    stage: &'static str,
    reason: String,
}

impl InferenceSeamFailure {
    fn new(arm: &'static str, stage: &'static str, reason: impl Into<String>) -> Self {
        Self { arm, stage, reason: reason.into() }
    }
}

impl std::fmt::Display for InferenceSeamFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InferenceSeamFailure: {} leaf inference failed at {} — the batch backed up \
             nothing, so this search cannot be exported as if it had run (R275(b) seam \
             conjunct; LAW-14: run-fatal, never silently degraded). The queue was OPEN, so \
             this is not a drain shutdown. reason={}",
            self.arm, self.stage, self.reason
        )
    }
}

/// Classify one inference-failure arm: a CLOSED queue is the drain-shutdown skip
/// (`Ok(0)`), an OPEN one is the named run-fatal seam failure.
fn seam_or_shutdown(
    queue_closed: bool,
    arm: &'static str,
    stage: &'static str,
    reason: impl Into<String>,
) -> Result<usize, InferenceSeamFailure> {
    if queue_closed {
        Ok(0)
    } else {
        Err(InferenceSeamFailure::new(arm, stage, reason))
    }
}

// ── Inference + expansion (HOT path) ────────────────────────────────────────

/// Selects leaves, encodes per-cluster state, submits to the dense inference
/// queue, forward/inverse-scatters under the per-game symmetry, accumulates I2
/// cluster-variance metrics, aggregates per-leaf policies, and runs
/// `expand_and_backup`. Frozen `inner.rs:742`.
///
/// # Errors
/// R275(b) SEAM conjunct: [`InferenceSeamFailure`] when a leaf inference FAILS on
/// an OPEN queue. An empty leaf set is `Ok(0)` — that is search exhaustion, not a
/// failure — and so is any failure arm reached with the queue already closed (the
/// drain-shutdown path).
#[inline]
#[allow(clippy::too_many_arguments)]
fn infer_and_expand(
    tree: &mut MCTSTree,
    batch_size: usize,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    legal_set: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> Result<usize, InferenceSeamFailure> {
    // Graph-seam dispatch hoisted at the worker boundary (NOT per-sim). The graph
    // fn is `#[cold]`/`#[inline(never)]` so it never bloats the inlined dense path.
    if infer.is_graph {
        return infer_and_expand_graph(tree, batch_size, agg_trunk_sz, infer);
    }

    let leaves = tree.select_leaves(batch_size);
    if leaves.is_empty() {
        return Ok(0);
    }

    let mut all_batch_features: Vec<Vec<f32>> = Vec::new();
    let mut leaf_metadata: Vec<(usize, Vec<(i32, i32)>)> = Vec::with_capacity(leaves.len());

    for leaf in &leaves {
        let (views, centers) = leaf.get_cluster_views();
        let k = views.len();
        leaf_metadata.push((k, centers));
        for view in views {
            // No feature-buffer pool (WP7 owns pooling); allocate the state-stride
            // buffer directly. Width == kept_planes.len() * n_cells == state_stride.
            let mut buffer = vec![0.0f32; kept_planes.len() * n_cells];
            encode_state_to_buffer_channels(leaf, &view, &mut buffer, kept_planes, n_cells);
            // §130: forward-scatter the input planes to the rotated frame.
            if infer.sym_idx != 0 {
                rotate_state_inplace(&mut buffer, infer.sym_idx, infer.sym_tables);
            }
            all_batch_features.push(buffer);
        }
    }
    if all_batch_features.is_empty() {
        return Ok(0);
    }

    let total_clusters: usize = leaf_metadata.iter().map(|(k, _)| *k).sum();

    let (all_policies, all_values) = match infer.dense_queue.submit_batch_and_wait(all_batch_features) {
        Ok(results) => {
            let mut ps = Vec::with_capacity(results.len());
            let mut vs = Vec::with_capacity(results.len());
            for (mut p, v) in results {
                // §130: inverse-scatter the policy back to canonical frame.
                if infer.sym_idx != 0 {
                    rotate_policy_inplace(&mut p, infer.inv_idx, infer.sym_tables, n_cells);
                }
                ps.push(p);
                vs.push(v);
            }
            (ps, vs)
        }
        // R275(b): was "dense skip-on-Err (reason NOT consumed, D6): skip the
        // batch". The frozen dense surface collapses the waiter's reason to `()`
        // before it ever reaches here, so the arm and stage are all this leg can
        // name — which is precisely why it must be loud rather than silent.
        Err(()) => {
            return seam_or_shutdown(
                infer.dense_queue.is_closed(),
                "dense",
                "submit_batch_and_wait",
                "the dense queue surface collapses the waiter reason to `()` (D6); \
                 causes are a producer-submitted inference failure or a feature-length \
                 mismatch",
            )
        }
    };

    if all_policies.len() < total_clusters {
        return seam_or_shutdown(
            infer.dense_queue.is_closed(),
            "dense",
            "result-count",
            format!(
                "inference returned {} policies for {total_clusters} cluster requests",
                all_policies.len()
            ),
        );
    }

    // One arm allocates, the other is an empty `Vec::new()` — one policy_pool per run.
    let mut aggregated_policies: Vec<Vec<f32>> =
        if legal_set { Vec::new() } else { Vec::with_capacity(leaves.len()) };
    let mut aggregated_policies_ls: Vec<LegalSetPolicy> =
        if legal_set { Vec::with_capacity(leaves.len()) } else { Vec::new() };
    let mut aggregated_values = Vec::with_capacity(leaves.len());
    let mut curr = 0;

    for (i, (k, centers)) in leaf_metadata.iter().enumerate() {
        let leaf_policies = &all_policies[curr..curr + *k];
        let leaf_values = &all_values[curr..curr + *k];
        curr += *k;

        // I2 investigation metric: per-cluster value/policy variance (Q2/Q27).
        if *k >= 2 {
            let mean_v: f32 = leaf_values.iter().sum::<f32>() / *k as f32;
            let var_v: f32 = leaf_values.iter().map(|&v| (v - mean_v).powi(2)).sum::<f32>() / *k as f32;
            let std_v = var_v.sqrt();
            let mut top1 = Vec::with_capacity(*k);
            for p in leaf_policies {
                let mut bi = 0usize;
                let mut bv = p[0];
                for (ii, &pv) in p.iter().enumerate() {
                    if pv > bv {
                        bi = ii;
                        bv = pv;
                    }
                }
                top1.push(bi);
            }
            let mut max_c = 1usize;
            for &a in &top1 {
                let c = top1.iter().filter(|&&x| x == a).count();
                if c > max_c {
                    max_c = c;
                }
            }
            let disagree = 1.0f32 - (max_c as f32 / *k as f32);
            variance.value_std_accum.fetch_add((std_v * 1_000_000.0) as u64, Ordering::Relaxed);
            variance.policy_disagreement_accum.fetch_add((disagree * 1_000_000.0) as u64, Ordering::Relaxed);
            variance.variance_samples.fetch_add(1, Ordering::Relaxed);
        }

        aggregated_values.push(aggregate_cluster_values_min(leaf_values));
        if legal_set {
            aggregated_policies_ls.push(records::aggregate_policy_ls(
                policy_stride, has_pass_slot, agg_trunk_sz, &leaves[i], centers, leaf_policies,
            ));
        } else {
            aggregated_policies.push(records::aggregate_policy(
                policy_stride, has_pass_slot, agg_trunk_sz, &leaves[i], centers, leaf_policies,
            ));
        }
    }

    let n = leaves.len();
    if legal_set {
        tree.expand_and_backup_ls(&aggregated_policies_ls, &aggregated_values);
    } else {
        tree.expand_and_backup(&aggregated_policies, &aggregated_values);
    }
    Ok(n)
}

/// GNN counterpart of `infer_and_expand` (frozen `inner.rs:893`). Builds ONE axis
/// graph per evaluated leaf (F-19 build-once-per-leaf: no reuse, no patching),
/// submits the whole batch through the parallel graph queue in ONE
/// `submit_graphs_and_wait` (D4), and expands via `expand_and_backup_ls_at`
/// against the BUILDER's per-leaf `window_center`. Rotation-free at inference (v1
/// coord pre-rotation is WP5 sample-time aug).
///
/// # Errors
/// R275(b) SEAM conjunct: a build-guard trip or a graph-inference failure is a
/// named [`InferenceSeamFailure`], NOT the pre-fix silent `return 0`. This is the
/// exact leg F-816-9 died on — the graph waiter's `Err(reason)` travelled back
/// verbatim (D6) and was then discarded.
#[cold]
#[inline(never)]
fn infer_and_expand_graph(
    tree: &mut MCTSTree,
    batch_size: usize,
    agg_trunk_sz: i32,
    infer: InferContext,
) -> Result<usize, InferenceSeamFailure> {
    let leaves = tree.select_leaves(batch_size);
    if leaves.is_empty() {
        return Ok(0);
    }

    // Graph-build geometry from the resolved spec (graph specs define these).
    let win_length = infer.spec.win_length.expect("graph spec must define win_length") as u8;
    let radius = infer.spec.graph_radius.expect("graph spec must define graph_radius") as u16;

    let mut graphs = Vec::with_capacity(leaves.len());
    let mut centers: Vec<(i32, i32)> = Vec::with_capacity(leaves.len());
    for leaf in &leaves {
        // Stone list from the board's sparse cell map (order irrelevant — the
        // builder coordinate-sorts). `Cell`/`Player` are `#[repr(i8)]` (±1).
        let mut stones: Vec<(i64, i64, i64)> = Vec::new();
        for (&(q, r), &cell) in leaf.cells_iter() {
            stones.push((i64::from(q), i64::from(r), cell as i64));
        }
        let current_player = leaf.current_player as i64;
        let moves_remaining = i64::from(leaf.moves_remaining);
        match build_leaf_graph(&stones, current_player, moves_remaining, win_length, radius, agg_trunk_sz) {
            Ok(g) => {
                centers.push(g.window_center);
                graphs.push(g);
            }
            // Seam guard tripped (unreachable for a valid self-play board). R275(b):
            // the pre-fix response was a silent batch-skip, argued from D6 —
            // "nothing was enqueued, so there is no waiter to carry the reason to".
            // That argues only that the reason cannot travel the QUEUE; it never
            // argued for discarding it. The reason is right here, and a guard the
            // board cannot legitimately trip is a defect, not a degrade. NOT routed
            // through `seam_or_shutdown`: a build guard is a pure function of the
            // board, so a closed queue cannot cause it and cannot excuse it.
            Err(reason) => {
                return Err(InferenceSeamFailure::new("graph", "build_leaf_graph", reason))
            }
        }
    }

    // Submit the WHOLE leaf batch in one shot and block on the assembled
    // `(LegalSetPolicy, value)` of each. Q-FIND-1/R263: submitting one graph at a
    // time put exactly one leaf in flight per worker, so the collector's saturation
    // threshold was structurally unreachable and every forward carried a single
    // graph. The returned `Vec` is indexed by SUBMISSION ORDER — the same order as
    // `leaves` and `centers`, which `expand_and_backup_ls_at` below requires.
    let results = infer.graph_queue.submit_graphs_and_wait(graphs);
    // COLLECT-ALL-THEN-DECIDE: every waiter has already resolved by the time this
    // Vec exists, so the refusal below cannot orphan one. The waiter's
    // `Err(reason)` travels back verbatim (D6) — R275(b) stops it being collapsed
    // to a batch-skip and carries it into the named failure instead. This is the
    // line F-816-9 died at: `graph_inference_forward_failed` on the box became
    // `return 0` here, and the run's only symptom was a target-integrity refusal
    // 100+ plies later.
    let mut aggregated_ls: Vec<LegalSetPolicy> = Vec::with_capacity(results.len());
    let mut aggregated_values: Vec<f32> = Vec::with_capacity(results.len());
    for res in results {
        match res {
            Ok((ls, v)) => {
                aggregated_ls.push(ls);
                aggregated_values.push(v);
            }
            Err(reason) => {
                return seam_or_shutdown(
                    infer.graph_queue.is_closed(),
                    "graph",
                    "submit_graphs_and_wait",
                    reason,
                )
            }
        }
    }
    if aggregated_ls.len() < leaves.len() {
        return seam_or_shutdown(
            infer.graph_queue.is_closed(),
            "graph",
            "result-count",
            format!(
                "inference returned {} payloads for {} submitted graphs",
                aggregated_ls.len(),
                leaves.len()
            ),
        );
    }

    let n = leaves.len();
    // Expand frame trunk = spec.trunk_size (`agg_trunk_sz`) — the SAME trunk the
    // builder baked into `policy_scatter_index`. ALWAYS-ON tripwire (frozen
    // `inner.rs:945`, a real `assert!`): the threaded `agg_trunk_sz` MUST still equal
    // the spec's canonical graph trunk, else the expand frame drifts from the built
    // slot window and silently misreads every in-window slot. One integer compare per
    // leaf batch (release strips no `assert!`); die-loud on mismatch.
    assert_eq!(
        agg_trunk_sz,
        infer.spec.trunk_size as i32,
        "graph trunk mismatch: spec agg_trunk_sz vs spec graph trunk_size"
    );
    tree.expand_and_backup_ls_at(&aggregated_ls, &aggregated_values, &centers, agg_trunk_sz);
    Ok(n)
}

// ── MCTS search dispatch (HOT path) ─────────────────────────────────────────

/// Two-branch dispatcher: Gumbel sequential-halving (`gumbel_mcts=true`) steered
/// via `set_forced_root_child` (per candidate: force → expand → clear), or
/// standard PUCT with Dirichlet root noise. Dirichlet is PUCT-only (Gumbel-Top-k
/// IS the Gumbel root exploration). Frozen `inner.rs:970`.
#[allow(clippy::too_many_arguments)]
fn run_mcts_search(
    tree: &mut MCTSTree,
    board: &Board,
    move_sims: usize,
    leaf_batch_size: usize,
    gumbel_mcts: bool,
    dirichlet_enabled: bool,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    gumbel_m: usize,
    c_visit: f32,
    c_scale: f32,
    running: &AtomicBool,
    rng: &mut ThreadRng,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    legal_set: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
) -> McTSSearchResult {
    let mut gumbel_state: Option<GumbelSearchState> = None;

    if gumbel_mcts {
        // ── Gumbel MCTS with Sequential Halving ──
        let root_sims = match infer_and_expand(
            tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance,
        ) {
            Ok(n) => n,
            Err(e) => return McTSSearchResult::InferenceFailed(e),
        };
        if root_sims == 0 || !tree.pool[0].is_expanded() {
            return McTSSearchResult::RootExpansionFailed;
        }
        let mut sims_used = root_sims;

        // NO Dirichlet root noise under Gumbel (Gumbel-Top-k IS the mechanism).
        let effective_m = gumbel_m.min(move_sims).min(tree.root_n_children());
        if effective_m == 0 {
            let mut sims_done = sims_used;
            while sims_done < move_sims {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                let n = match infer_and_expand(
                    tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance,
                ) {
                    Ok(n) => n,
                    Err(e) => return McTSSearchResult::InferenceFailed(e),
                };
                if n == 0 {
                    break;
                }
                sims_done += n;
            }
            gumbel_state = None;
        } else {
            let mut gs = GumbelSearchState::new(tree, effective_m, c_visit, c_scale, rng);

            // Phase 3: Sequential Halving — allocate budget across phases.
            let num_phases = gs.num_phases;
            for phase in 0..num_phases {
                if sims_used >= move_sims {
                    break;
                }
                let remaining_budget = move_sims.saturating_sub(sims_used);
                let remaining_phases = num_phases - phase;
                let sims_per = (remaining_budget / (remaining_phases * gs.candidates.len())).max(1);

                let cands = gs.candidates.clone();
                for &cand_offset in &cands {
                    if sims_used >= move_sims {
                        break;
                    }
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }

                    let child_pool_idx = gs.first_child + cand_offset as u32;
                    tree.set_forced_root_child(Some(child_pool_idx));

                    let mut cand_sims = 0;
                    while cand_sims < sims_per && sims_used < move_sims {
                        if !running.load(Ordering::Relaxed) {
                            break;
                        }
                        // Cap batch to this candidate's remaining budget so we don't
                        // overshoot `sims_per`.
                        let batch = leaf_batch_size.min(sims_per.saturating_sub(cand_sims));
                        let n = match infer_and_expand(
                            tree, batch.max(1), kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance,
                        ) {
                            Ok(n) => n,
                            Err(e) => {
                                tree.set_forced_root_child(None);
                                return McTSSearchResult::InferenceFailed(e);
                            }
                        };
                        if n == 0 {
                            break;
                        }
                        cand_sims += n;
                        sims_used += n;
                    }
                    tree.set_forced_root_child(None);
                }

                if gs.candidates.len() <= 1 {
                    break;
                }
                gs.halve_candidates(tree);
            }
            tree.set_forced_root_child(None);
            gumbel_state = Some(gs);
        }
    } else {
        // ── Standard PUCT search with Dirichlet root noise ──
        let root_n = match infer_and_expand(
            tree, 1, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance,
        ) {
            Ok(n) => n,
            Err(e) => return McTSSearchResult::InferenceFailed(e),
        };
        if root_n == 0 {
            return McTSSearchResult::RootExpansionFailed;
        }
        let mut sims_done = root_n;

        let is_intermediate_ply = board.moves_remaining == 1 && board.ply.index() > 0;
        if dirichlet_enabled && !is_intermediate_ply && tree.pool[0].is_expanded() {
            let n_ch = tree.pool[0].n_children as usize;
            if n_ch > 0 {
                let noise = mantis_search::mcts::dirichlet::sample_dirichlet(dirichlet_alpha, n_ch, rng);
                tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
            }
        }

        while sims_done < move_sims {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let n = match infer_and_expand(
                tree, leaf_batch_size, kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set, infer, variance,
            ) {
                Ok(n) => n,
                Err(e) => return McTSSearchResult::InferenceFailed(e),
            };
            if n == 0 {
                break;
            }
            sims_done += n;
        }
    }

    McTSSearchResult::Completed(gumbel_state)
}

// ── Per-move dispatcher (warm/HOT path) ─────────────────────────────────────

/// Orchestrates one full move: playout-cap selection, MCTS search, per-move stat
/// accumulation, target-policy build (temperature → completed-Q → O1 → solver),
/// ZOI-filtered sampling, position recording (BEFORE apply), and apply-move.
/// Frozen `inner.rs:1099`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn play_one_move(
    tree: &mut MCTSTree,
    board: &mut Board,
    records_vec: &mut Vec<RecordTuple>,
    graph_records_vec: &mut Vec<GraphRecord>,
    move_history: &mut Vec<(i32, i32)>,
    version_seen: &mut Vec<u64>,
    rng: &mut ThreadRng,
    running: &AtomicBool,
    ctx: MovePlayContext,
    kept_planes: &'static [usize],
    n_cells: usize,
    policy_stride: usize,
    has_pass_slot: bool,
    agg_trunk_sz: i32,
    legal_set: bool,
    is_graph: bool,
    infer: InferContext,
    variance: ClusterVarianceAtomics,
    accumulators: MoveAccumulators,
    solver_counters: SolverCounters,
    solver_fires: &mut u32,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    // Move-level playout cap (orthogonal to game-level fast_prob).
    let (move_is_full_search, move_sims) = if ctx.full_search_prob > 0.0 {
        let full = rng.random::<f32>() < ctx.full_search_prob;
        let sims = if full { ctx.n_sims_full } else { ctx.n_sims_quick };
        (full, sims)
    } else {
        (true, ctx.game_sims)
    };

    // ── MCTS Search ──
    tree.new_game(board.clone());

    let gumbel_state = match run_mcts_search(
        tree, board, move_sims, ctx.leaf_batch_size, ctx.gumbel_mcts,
        ctx.dirichlet_enabled, ctx.dirichlet_alpha, ctx.dirichlet_epsilon,
        ctx.gumbel_m, ctx.c_visit, ctx.c_scale, running, rng,
        kept_planes, n_cells, policy_stride, has_pass_slot, agg_trunk_sz, legal_set,
        infer, variance,
    ) {
        McTSSearchResult::Completed(gs) => gs,
        McTSSearchResult::RootExpansionFailed => return MoveOutcome::Continue,
        // R275(b) SEAM conjunct: LAW-14 store-then-halt on its OWN counter. The
        // message rides the shared latch slot to the drain face, so the supervisor
        // reads the inference failure that killed the run instead of a
        // target-integrity refusal a hundred plies downstream.
        McTSSearchResult::InferenceFailed(err) => {
            fatal_latch.store_inference_failure(err.to_string());
            return MoveOutcome::Break;
        }
    };

    if !running.load(Ordering::Relaxed) {
        return MoveOutcome::Break;
    }

    // ── R275(b) EXPORTER conjunct: no visits, no target ──
    // BEFORE any exporter runs, and arm-independent — `records::refuse_zero_visit_export`
    // is the one place that decides a search is exportable. `policy` below feeds BOTH the
    // recorded target and the move actually played, so a zero-visit search would otherwise
    // also sample its move from the prior fallback.
    if let Err(err) = records::refuse_zero_visit_export(tree, board.ply.index() as u16) {
        fatal_latch.store(err.to_string());
        return MoveOutcome::Break;
    }

    // ── MCTS Policy with cosine-annealed temperature schedule ──
    let compound_move = ply_to_compound_move(board.ply.index() as usize);
    let temperature = if ctx.is_fast_game {
        1.0 // fast games: always exploratory
    } else {
        compute_move_temperature(compound_move, ctx.temp_threshold, ctx.temp_min)
    };
    let policy = if legal_set {
        MovePolicy::Ls(tree.get_policy_ls(temperature, policy_stride))
    } else {
        MovePolicy::Dense(tree.get_policy(temperature, policy_stride))
    };

    // Accumulate MCTS health stats once per search (not in the inner sim loop).
    {
        let (depth, conc) = tree.last_search_stats();
        accumulators.mcts_depth_accum.fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators.mcts_conc_accum.fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators.mcts_stat_count.fetch_add(1, Ordering::Relaxed);
        accumulators
            .mcts_quiescence_fires
            .fetch_add(tree.quiescence_fire_count.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    // Phase B' Class-1 (frozen `inner.rs:1214`): snapshot the model version once per
    // move and dedup-push into `version_seen`. The pure-Rust runner sources it from a
    // runner-owned `model_version` atomic (default 0 until WP7 wires the NN setter), so
    // a no-NN run's drain tuple `(mv_min, mv_max, mv_distinct)` is the frozen (0, 0, 1).
    {
        let v = infer.model_version.load(Ordering::Relaxed);
        if !version_seen.contains(&v) {
            version_seen.push(v);
        }
    }

    // Completed Q-values: compute improved policy for the training target.
    let mut target_policy = if ctx.completed_q_values {
        if legal_set {
            MovePolicy::Ls(tree.get_improved_policy_ls(policy_stride, ctx.c_visit, ctx.c_scale))
        } else {
            MovePolicy::Dense(tree.get_improved_policy(policy_stride, ctx.c_visit, ctx.c_scale))
        }
    } else {
        policy.clone()
    };

    // O1: forced-win → (near-)one-hot POLICY target (fires once per move at target
    // extraction, NOT in the per-sim hot path).
    let forced_win_fired = ctx.forced_win_enabled
        && match board.forced_win_move(ctx.forced_win_depth) {
            Some((wq, wr)) => match &mut target_policy {
                MovePolicy::Dense(t) => {
                    let action = board.window_flat_idx(wq, wr);
                    if action < policy_stride {
                        records::apply_forced_win_one_hot(t, action, ctx.forced_win_weight);
                        true
                    } else {
                        false
                    }
                }
                // Coverage-gated: a covered win one-hots; an uncovered win is a no-op
                // COUNTED by the R256 instrument (the one counted helper, shared with
                // the solver hook so mechanism and instrument cannot drift).
                MovePolicy::Ls(ls) => records::apply_forced_win_one_hot_ls_counted(
                    board, ls, (wq, wr), ctx.forced_win_weight, agg_trunk_sz,
                    accumulators.uncovered_forced_win,
                ),
            },
            None => false,
        };

    // D-WS3 L1: native solver-in-loop SOFT visit-injection (default-OFF =
    // byte-identical hot path).
    let solver_fired = run_solver_hook(
        board, &mut target_policy, &ctx, legal_set, policy_stride, agg_trunk_sz,
        solver_counters, solver_fires, accumulators.uncovered_forced_win,
    );

    let record_full_search = move_is_full_search || forced_win_fired || solver_fired;

    // LAW-18 (DESIGN_T §3.6): count a move whose exported target carries
    // off-window (overflow) mass — the restored-mass fire-rate; pre-Phase-T
    // this population was being truncated by the coverage gate.
    if let MovePolicy::Ls(ls) = &target_policy {
        if ls.overflow.values().any(|&p| p > 0.0) {
            accumulators.export_offwindow_mass_moves.fetch_add(1, Ordering::Relaxed);
        }
    }

    // ── Sample and apply move (ZOI-filtered legal set) ──
    let Some(move_idx) = select_move(board, move_history, &policy, gumbel_state, ctx, agg_trunk_sz, tree, rng) else {
        return MoveOutcome::Break;
    };

    // ── Record position (BEFORE apply_move; hoisted is_graph branch) ──
    if is_graph {
        let visit_capacity = ctx.visit_capacity.expect(
            "graph record dispatch requires the derived visit capacity — composed in \
             SelfPlayRunner::new's graph arm (R255)",
        );
        if let Err(err) = record_position_graph_dispatch(
            board, &target_policy, agg_trunk_sz, record_full_search, graph_records_vec,
            visit_capacity,
        ) {
            // LAW-14: a target-integrity defect is RUN-FATAL — latch the typed
            // message (variant name in Display) and halt; the bridge drain face
            // raises it to the supervisor (DESIGN_T §3.4).
            fatal_latch.store(err.to_string());
            return MoveOutcome::Break;
        }
    } else {
        record_position(
            board, kept_planes, n_cells, agg_trunk_sz, ctx.is_fast_game, ctx.completed_q_values,
            policy_stride, has_pass_slot, &target_policy, ctx.sym_idx, infer.sym_tables, record_full_search, records_vec,
            accumulators.gridls_zero_policy_rows,
            accumulators.k_cluster_histogram,
        );
    }

    if board.apply_move(move_idx.0, move_idx.1).is_err() {
        return MoveOutcome::Break;
    }
    move_history.push((move_idx.0, move_idx.1));
    accumulators.positions_generated.fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}

/// D-WS3 L1 solver hook (frozen `inner.rs:1291`). The net-free `mantis_search`
/// tactical solver proves the side-to-move's forced win and SOFT-injects visit
/// mass onto the proving move's first stone (`line[0]`) into the POLICY target.
/// `solver_enabled=false` (default) short-circuits before any solver work. Fires
/// once per move at target extraction, never in the per-sim hot path.
#[allow(clippy::too_many_arguments)]
fn run_solver_hook(
    board: &Board,
    target_policy: &mut MovePolicy,
    ctx: &MovePlayContext,
    legal_set: bool,
    policy_stride: usize,
    agg_trunk_sz: i32,
    solver_counters: SolverCounters,
    solver_fires: &mut u32,
    uncovered_forced_win: &AtomicU64,
) -> bool {
    if !ctx.solver_enabled {
        return false;
    }
    // Every move reaching here is "eligible" (the solver runs on every move).
    solver_counters.moves_eligible.fetch_add(1, Ordering::Relaxed);
    if ctx.seeded {
        solver_counters.moves_eligible_seeded.fetch_add(1, Ordering::Relaxed);
    }
    let cfg = TacticalConfig {
        cand_cap: 40,
        // legal_set (multi-window): None surfaces off-window forced wins (the
        // coverage gate gives them a ragged slot). DENSE (single-window): keep the
        // single-window guard so the solver only spends budget on the expressible
        // action space.
        window_half: if legal_set { None } else { Some((agg_trunk_sz - 1) / 2) },
        // Quiet-move widening; < 0 → None (threat-only).
        neighbor_dist: if ctx.solver_neighbor_dist < 0 {
            None
        } else {
            Some(ctx.solver_neighbor_dist)
        },
    };
    let proof = TacticalSolver::new(cfg).prove(board, ctx.solver_depth, ctx.solver_node_budget);
    if proof.budget_exhausted {
        solver_counters.budget_exhausted.fetch_add(1, Ordering::Relaxed);
    }
    if proof.result != Outcome::Win {
        return false;
    }
    solver_counters.win_proven.fetch_add(1, Ordering::Relaxed);
    let Some(&(wq, wr)) = proof.line.first() else {
        return false;
    };
    // (injected, off_window) — off_window is only reachable on the LS path.
    let (injected, off_window) = match target_policy {
        MovePolicy::Dense(t) => {
            let action = board.window_flat_idx(wq, wr);
            if action < policy_stride {
                records::apply_forced_win_one_hot(t, action, ctx.solver_visit_weight);
                (true, false)
            } else {
                (false, false)
            }
        }
        // Coverage-gated (the SAME counted helper as the O1 LS path — R256).
        MovePolicy::Ls(ls) => {
            let did = records::apply_forced_win_one_hot_ls_counted(
                board, ls, (wq, wr), ctx.solver_visit_weight, agg_trunk_sz, uncovered_forced_win,
            );
            // Off-window = injected into the ragged OVERFLOW target: the win maps
            // outside the dense global window (>= policy_stride).
            let (bcq, bcr) = board.window_center();
            let half = (agg_trunk_sz - 1) / 2;
            let off = did && Board::window_flat_idx_at_geom(wq, wr, bcq, bcr, agg_trunk_sz, half) >= policy_stride;
            (did, off)
        }
    };
    if injected {
        solver_counters.injected.fetch_add(1, Ordering::Relaxed);
        if ctx.seeded {
            solver_counters.injected_seeded.fetch_add(1, Ordering::Relaxed);
        }
        if off_window {
            solver_counters.injected_offwindow.fetch_add(1, Ordering::Relaxed);
        }
        *solver_fires += 1;
    }
    injected
}

/// D-WS3V3 Gumbel-explore gate on ply RELATIVE to game start (frozen
/// `inner.rs:1426`). `game_start_ply == 0` (organic) is byte-identical to the
/// absolute-ply test; a seeded game explores for `explore_moves` moves AFTER its
/// deep start.
#[inline]
fn relative_explore_gate(ply: usize, game_start_ply: usize, explore_moves: usize) -> bool {
    ply.saturating_sub(game_start_ply) >= explore_moves
}

/// Per-move legal-move sampler (frozen `inner.rs:1439`). ZOI-filters when enabled,
/// picks via Gumbel winner (post exploration gate) or visit-count sampling, falls
/// back to uniform random. `None` when no legal moves (caller breaks).
#[allow(clippy::too_many_arguments)]
fn select_move(
    board: &Board,
    move_history: &[(i32, i32)],
    policy: &MovePolicy,
    gumbel_state: Option<GumbelSearchState>,
    ctx: MovePlayContext,
    agg_trunk_sz: i32,
    tree: &MCTSTree,
    rng: &mut ThreadRng,
) -> Option<(i32, i32)> {
    let full_legal = board.legal_moves();
    if full_legal.is_empty() {
        return None;
    }

    // ZOI filtering: restrict move sampling to cells near recent moves.
    let legal = if ctx.zoi_enabled && move_history.len() >= 3 {
        let filtered: Vec<_> = full_legal
            .iter()
            .filter(|(q, r)| {
                move_history
                    .iter()
                    .rev()
                    .take(ctx.zoi_lookback)
                    .any(|(q0, r0)| hex_distance(*q, *r, *q0, *r0) <= ctx.zoi_margin)
            })
            .copied()
            .collect();
        if filtered.len() < 3 {
            full_legal
        } else {
            filtered
        }
    } else {
        full_legal
    };

    // Move selection: Gumbel winner or visit-count sampling. D-WS3V3 gates the
    // exploration on ply RELATIVE to game start.
    let use_gumbel_winner = gumbel_state.is_some()
        && relative_explore_gate(board.ply.index() as usize, ctx.game_start_ply, ctx.gumbel_explore_moves);
    let move_idx = if use_gumbel_winner {
        let mut gs = gumbel_state.unwrap();
        let best_pool = gs.best_action_pool_idx(tree);
        let val = tree.pool[best_pool as usize].action_idx;
        let mq = (val >> 16) as i32 - 32768;
        let mr = (val & 0xFFFF) as i32 - 32768;
        if legal.contains(&(mq, mr)) {
            (mq, mr)
        } else {
            match policy.sample(&legal, board, agg_trunk_sz) {
                Some(idx) => idx,
                None => *legal.choose(rng).unwrap(),
            }
        }
    } else {
        match policy.sample(&legal, board, agg_trunk_sz) {
            Some(idx) => idx,
            None => *legal.choose(rng).unwrap(),
        }
    };
    Some(move_idx)
}
