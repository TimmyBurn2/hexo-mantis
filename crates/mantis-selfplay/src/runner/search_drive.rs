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

use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use rand::prelude::IndexedRandom;
use rand::rngs::ThreadRng;
use rand::RngExt;

use mantis_core::Board;
use mantis_encoding::RegistrySpec;
use mantis_search::{
    compute_move_temperature, ply_to_compound_move, LegalSetPolicy, MCTSTree, MctxRootState,
    SearchKind,
};

use crate::queues::{build_leaf_graph, GraphQueue};
use crate::records;
use crate::replay::hexg::GraphRecord;

use super::record::record_position_graph_dispatch;

/// Graph queue handle + per-game symmetry context + the resolved spec (graph-build
/// geometry). `Copy` — passed by value.
#[derive(Clone, Copy)]
pub(crate) struct InferContext<'a> {
    pub(crate) graph_queue: &'a GraphQueue,
    pub(crate) spec: &'static RegistrySpec,
    /// Runner-owned model-version snapshot source (frozen `inner.rs:1214` =
    /// `batcher.current_model_version()`). Read once per move and dedup-pushed into
    /// `version_seen`. Default 0 (no-NN) until WP7 wires the real setter.
    pub(crate) model_version: &'a AtomicU64,
    /// R276(a): the runner's kill switch, read by `seam_or_shutdown` to tell OUR stop
    /// from a queue closed by anything else (a dying inference server, above all).
    pub(crate) running: &'a AtomicBool,
}

/// Per-move MCTS accumulators + `positions_generated` (frozen `:94`).
/// `export_offwindow_mass_moves` fires once per move whose exported target carries
/// overflow mass (LAW-18, DESIGN_T §3.6).
#[derive(Clone, Copy)]
pub(crate) struct MoveAccumulators<'a> {
    pub(crate) mcts_depth_accum: &'a AtomicU64,
    pub(crate) mcts_conc_accum: &'a AtomicU64,
    pub(crate) mcts_stat_count: &'a AtomicU64,
    pub(crate) mcts_quiescence_fires: &'a AtomicU64,
    /// R335(c) — `fetch_max`ed with each search's served-leaf count.
    pub(crate) max_sims_per_search: &'a AtomicU64,
    /// LAW-18 — the playout-cap arm as DRAWN, counted at the draw itself.
    pub(crate) pcr_full_moves: &'a AtomicU64,
    pub(crate) pcr_quick_moves: &'a AtomicU64,
    /// LAW-18 — the Gumbel round's width (see [`GumbelRoundCounters`]). Zero on a PUCT run,
    /// which issues no rounds; the reader omits the mean rather than publishing a 0/0.
    pub(crate) gumbel_round_leaves: &'a AtomicU64,
    pub(crate) gumbel_rounds: &'a AtomicU64,
    pub(crate) positions_generated: &'a AtomicUsize,
    pub(crate) export_offwindow_mass_moves: &'a AtomicU64,
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

/// Per-move scalar context (frozen `:141`). `Copy` — mirrors the flat
/// `WorkerParams` layout plus the per-game dynamics (`game_sims`, `is_fast_game`,
/// `game_start_ply`).
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct MovePlayContext {
    pub(crate) leaf_batch_size: usize,
    /// DERIVED HEXG visit capacity (R255) — `Some` iff this is a graph run.
    pub(crate) visit_capacity: Option<usize>,
    pub(crate) temp_threshold: usize,
    pub(crate) temp_min: f32,
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
    /// THE search authority: the root mechanism, the interior selector and the exported
    /// target's semantics all read this one field.
    pub(crate) search_kind: SearchKind,
    pub(crate) dirichlet_enabled: bool,
    /// Absolute ply the organic play begins at. Gates Gumbel exploration RELATIVE to start.
    pub(crate) game_start_ply: usize,
}

/// Per-move policy — the ragged legal-set policy (frozen `:713`). The dense scatter_max
/// vector went with the grid path (R346(f)).
#[derive(Clone)]
pub(crate) enum MovePolicy {
    Ls(LegalSetPolicy),
}

impl MovePolicy {
    /// Sample a move from `legal` proportional to this policy's mass at each coord
    /// (the ragged variant uses the `1/n` no-coverage floor).
    fn sample(&self, legal: &[(i32, i32)], board: &Board, trunk: i32) -> Option<(i32, i32)> {
        match self {
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

/// LAW-18 — the Gumbel round's WIDTH, in the two terms a mean is taken over.
///
/// `round_leaves / rounds` is leaves per inference round trip. At `m` candidates the first
/// round is `m` wide and each halving takes it down, so the mean over a search is well below
/// `m` and well above 1 — and 1 is exactly what a batching lever that had silently stopped
/// batching would read.
#[derive(Clone, Copy)]
pub(crate) struct GumbelRoundCounters<'a> {
    pub(crate) round_leaves: &'a AtomicU64,
    pub(crate) rounds: &'a AtomicU64,
}

/// How one inference round trip picks its leaves.
///
/// `Batch(n)` is PUCT's: `select_leaves(n)` descends `n` times from an unconstrained root,
/// spread by virtual loss. `Round(&[..])` is Gumbel's: one descent under EACH of the round's
/// surviving candidates, which is the halving phase's own width — `m` leaves, then `m/2`,
/// and so on — and needs no virtual loss across candidates because their subtrees are
/// disjoint.
#[derive(Clone, Copy)]
enum LeafSelection<'a> {
    Batch(usize),
    Round(&'a [u32]),
}

/// Result of `run_mcts_search`.
enum McTSSearchResult {
    /// The Gumbel root state (PUCT: `None`) and the leaves this search actually
    /// served — R335(c)'s per-search visit count, `fetch_max`ed by the caller.
    Completed(Option<MctxRootState>, usize),
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
/// **A DRAIN SHUTDOWN IS NOT A FAILURE, AND `is_closed()` IS THE WRONG WAY TO SAY
/// SO** (R276(a) merge-gate finding). `stop()` flips `running=false` and then closes
/// both queues, waking every in-flight waiter with `Err` (`runner/mod.rs::stop`);
/// the §P22/D12 drain-shutdown path depends on those `Err`s being a skip, not a
/// defect. The first shipped discriminator asked `queue.is_closed()`, and that
/// reads a state whose CAUSE is untyped: `close()` takes no reason, and the Python
/// inference server closes the batcher from a `finally` on ANY loop exit —
/// `inference_server.py`'s own comment says "Release blocked Rust waiters even if
/// this thread exits unexpectedly." So an inference-server DEATH closed the queue,
/// every in-flight failure then observed `closed`, and the failure re-entered
/// through the shutdown door: exactly the silent degrade this pin exists to kill.
///
/// The discriminator is therefore the RUNNER'S OWN KILL SWITCH, not the queue's
/// state. `SelfPlayRunner::stop()` stores `running=false` BEFORE either
/// `close()`, and `WorkerPool.stop()` calls `self._runner.stop()` (pool.py:393)
/// before `self._inference_server.stop()` (:394) — so on every clean stop the flag
/// is already false when the waiter wakes, and on a server death it is still true.
/// No queue is read at all, which also retires the wrong-queue hole
/// (`dense_queue` vs `graph_queue`) by construction rather than by oracle.
pub(crate) struct InferenceSeamFailure {
    arm: &'static str,
    stage: &'static str,
    reason: String,
}

impl InferenceSeamFailure {
    fn new(arm: &'static str, stage: &'static str, reason: impl Into<String>) -> Self {
        Self {
            arm,
            stage,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for InferenceSeamFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InferenceSeamFailure: {} leaf inference failed at {} — the batch backed up \
             nothing, so this search cannot be exported as if it had run (R275(b) seam \
             conjunct; LAW-14: run-fatal, never silently degraded). The runner was NOT \
             stopping, so this is not a drain shutdown — if the queue is closed, something \
             other than `stop()` closed it. reason={}",
            self.arm, self.stage, self.reason
        )
    }
}

/// Classify one inference-failure arm. `running == false` means OUR OWN `stop()` is
/// under way, which is the drain-shutdown skip (`Ok(0)`); anything else — including a
/// queue closed by a dying inference server — is the named run-fatal seam failure.
///
/// `SeqCst` matches the store side (`SelfPlayRunner::stop` and
/// `FatalDefectLatch::store_counted` both use `SeqCst`), so the flag a waking waiter
/// reads is ordered against the close that woke it. This is a cold path; the ordering
/// costs nothing and a weaker one would make the argument above unprovable.
fn seam_or_shutdown(
    running: &AtomicBool,
    arm: &'static str,
    stage: &'static str,
    reason: impl Into<String>,
) -> Result<usize, InferenceSeamFailure> {
    if running.load(Ordering::SeqCst) {
        Err(InferenceSeamFailure::new(arm, stage, reason))
    } else {
        Ok(0)
    }
}

// ── Inference + expansion (HOT path) ────────────────────────────────────────

/// One leaf-selection call, in whichever mode this round trip runs.
#[inline]
fn select_for(
    tree: &mut MCTSTree,
    selection: LeafSelection<'_>,
) -> Result<Vec<Board>, mantis_search::mcts::SelectionDesync> {
    match selection {
        LeafSelection::Batch(n) => tree.select_leaves(n),
        LeafSelection::Round(children) => tree.select_leaves_forced(children),
    }
}

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
// `#[cold]`/`#[inline(never)]` are DELETED with the dense arm they were paired against: they
// told LLVM to lay this out as the unlikely branch and optimize it for size, and it is now the
// only inference path there is.
fn infer_and_expand_graph(
    tree: &mut MCTSTree,
    selection: LeafSelection<'_>,
    agg_trunk_sz: i32,
    infer: InferContext,
) -> Result<usize, InferenceSeamFailure> {
    // AUDIT-1 F-02: a tree/board desync is a NAMED run-fatal seam failure, not a panic that
    // `guard_worker` converts into `running = false` with no reason latched. It routes through
    // the same channel every other leaf-inference failure does, so R275(b)'s instrument sees
    // it and `store_fatal_defect` names it.
    let leaves = select_for(tree, selection)
        .map_err(|desync| InferenceSeamFailure::new("graph", "selection", desync.to_string()))?;
    if leaves.is_empty() {
        return Ok(0);
    }

    // Graph-build geometry from the resolved spec (graph specs define these).
    let win_length = infer
        .spec
        .win_length
        .expect("graph spec must define win_length") as u8;
    let radius = infer
        .spec
        .graph_radius
        .expect("graph spec must define graph_radius") as u16;

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
        match build_leaf_graph(
            &stones,
            current_player,
            moves_remaining,
            win_length,
            radius,
            agg_trunk_sz,
        ) {
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
                return Err(InferenceSeamFailure::new(
                    "graph",
                    "build_leaf_graph",
                    reason,
                ))
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
                return seam_or_shutdown(infer.running, "graph", "submit_graphs_and_wait", reason)
            }
        }
    }
    if aggregated_ls.len() < leaves.len() {
        return seam_or_shutdown(
            infer.running,
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
        agg_trunk_sz, infer.spec.trunk_size as i32,
        "graph trunk mismatch: spec agg_trunk_sz vs spec graph trunk_size"
    );
    tree.expand_and_backup_ls_at(&aggregated_ls, &aggregated_values, &centers, agg_trunk_sz);
    Ok(n)
}

// ── MCTS search dispatch (HOT path) ─────────────────────────────────────────

/// Two-branch dispatcher on the ONE search kind: Gumbel (Gumbel-Top-k root sampling +
/// Sequential Halving, no Dirichlet — the Gumbel draw IS the root exploration) or PUCT
/// (Dirichlet root noise, PUCT descent).
///
/// THE ROOT'S OWN EVALUATION IS CHARGED under both kinds, and that is what makes `N`
/// mean `N leaves`. The deleted `gumbel_root_counts` key made the charge a config
/// choice, so "equal NN work at a fixed budget" was a claim a config could quietly
/// falsify; now the served count is the leaf count on both arms by construction.
#[allow(clippy::too_many_arguments)]
fn run_mcts_search(
    tree: &mut MCTSTree,
    board: &Board,
    move_sims: usize,
    leaf_batch_size: usize,
    search_kind: SearchKind,
    dirichlet_enabled: bool,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    gumbel_m: usize,
    c_visit: f32,
    c_scale: f32,
    running: &AtomicBool,
    rng: &mut ThreadRng,
    agg_trunk_sz: i32,
    infer: InferContext,
    rounds: GumbelRoundCounters,
) -> McTSSearchResult {
    // Both kinds open the same way: ONE leaf, which is the root itself, evaluated and
    // backed up. It is charged against the budget on both arms.
    let root_sims = match infer_and_expand_graph(tree, LeafSelection::Batch(1), agg_trunk_sz, infer)
    {
        Ok(n) => n,
        Err(e) => return McTSSearchResult::InferenceFailed(e),
    };
    if root_sims == 0 || !tree.pool[0].is_expanded() {
        return McTSSearchResult::RootExpansionFailed;
    }

    match search_kind {
        SearchKind::Gumbel => {
            let budget = move_sims.saturating_sub(root_sims);
            let state = MctxRootState::new(tree, gumbel_m, budget, rng);
            let mut spent = 0usize;
            while spent < budget {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // THE ROUND, not the simulation, is the unit. Every candidate alive at the
                // current considered visit level is descended into ONCE, and the whole set
                // goes to the producer as one batch — `m` leaves, then `m/2`, and so on.
                // The batch is re-derived from the tree's own visit counts each round, which
                // is what makes the halving Mctx's and not a phase allocation.
                let mut round = state.round_batch(tree, c_visit, c_scale);
                if round.is_empty() {
                    break;
                }
                // The budget is the last word: a round is never allowed to overspend it.
                round.truncate(budget - spent);
                // The indices come from the root state, but the range check is the SAME one
                // every other forced descent takes — a candidate the root does not own is a
                // bookkeeping defect and takes the run-fatal exit, not a silent descent into
                // a subtree belonging to nothing (AUDIT-1 F-02).
                for &child in &round {
                    if let Err(err) = tree.set_forced_root_child(Some(child)) {
                        let _ = tree.set_forced_root_child(None);
                        return McTSSearchResult::InferenceFailed(InferenceSeamFailure::new(
                            "gumbel",
                            "forced_root_child",
                            err.to_string(),
                        ));
                    }
                }
                let _ = tree.set_forced_root_child(None);

                let n = match infer_and_expand_graph(
                    tree,
                    LeafSelection::Round(&round),
                    agg_trunk_sz,
                    infer,
                ) {
                    Ok(n) => n,
                    Err(e) => return McTSSearchResult::InferenceFailed(e),
                };
                if n == 0 {
                    break;
                }
                // LAW-18: the round's own width, logged where it is decided. A batching
                // lever whose fire rate is not in the run cannot be told from a lever that
                // is issuing one leaf per round trip.
                rounds.round_leaves.fetch_add(n as u64, Ordering::Relaxed);
                rounds.rounds.fetch_add(1, Ordering::Relaxed);
                spent += n;
            }
            McTSSearchResult::Completed(Some(state), root_sims + spent)
        }
        SearchKind::Puct => {
            let mut sims_done = root_sims;

            let is_intermediate_ply = board.moves_remaining == 1 && board.ply.index() > 0;
            if dirichlet_enabled && !is_intermediate_ply && tree.pool[0].is_expanded() {
                let n_ch = tree.pool[0].n_children as usize;
                if n_ch > 0 {
                    let noise = mantis_search::mcts::dirichlet::sample_dirichlet(
                        dirichlet_alpha,
                        n_ch,
                        rng,
                    );
                    tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);
                }
            }

            while sims_done < move_sims {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // R335(c): the last batch is sized to the REMAINING budget. Unclamped, this
                // loop served 53-56 sims against a configured 50, which is what made the
                // ledger's served-sims line disagree with the config and every "fixed nodes"
                // claim unstatable.
                let batch = leaf_batch_size.min(move_sims - sims_done);
                let n = match infer_and_expand_graph(
                    tree,
                    LeafSelection::Batch(batch),
                    agg_trunk_sz,
                    infer,
                ) {
                    Ok(n) => n,
                    Err(e) => return McTSSearchResult::InferenceFailed(e),
                };
                if n == 0 {
                    break;
                }
                sims_done += n;
            }
            McTSSearchResult::Completed(None, sims_done)
        }
    }
}

// ── Per-move dispatcher (warm/HOT path) ─────────────────────────────────────

/// Orchestrates one full move: playout-cap selection, MCTS search, per-move stat
/// accumulation, target-policy build (temperature -> completed-Q), sampling, position
/// recording (BEFORE apply), and apply-move.
/// Frozen `inner.rs:1099`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) fn play_one_move(
    tree: &mut MCTSTree,
    board: &mut Board,
    graph_records_vec: &mut Vec<GraphRecord>,
    move_history: &mut Vec<(i32, i32)>,
    version_seen: &mut Vec<u64>,
    rng: &mut ThreadRng,
    running: &AtomicBool,
    ctx: MovePlayContext,
    policy_stride: usize,
    agg_trunk_sz: i32,
    infer: InferContext,
    accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
) -> MoveOutcome {
    // Move-level playout cap (orthogonal to game-level fast_prob).
    let (move_is_full_search, move_sims) = if ctx.full_search_prob > 0.0 {
        let full = rng.random::<f32>() < ctx.full_search_prob;
        let sims = if full {
            ctx.n_sims_full
        } else {
            ctx.n_sims_quick
        };
        (full, sims)
    } else {
        (true, ctx.game_sims)
    };
    // LAW-18, counted HERE — at the draw, before anything else can move the flag.
    if move_is_full_search {
        accumulators.pcr_full_moves.fetch_add(1, Ordering::Relaxed);
    } else {
        accumulators.pcr_quick_moves.fetch_add(1, Ordering::Relaxed);
    }

    // ── MCTS Search ──
    tree.new_game(board.clone());

    let gumbel_state = match run_mcts_search(
        tree,
        board,
        move_sims,
        ctx.leaf_batch_size,
        ctx.search_kind,
        ctx.dirichlet_enabled,
        ctx.dirichlet_alpha,
        ctx.dirichlet_epsilon,
        ctx.gumbel_m,
        ctx.c_visit,
        ctx.c_scale,
        running,
        rng,
        agg_trunk_sz,
        infer,
        GumbelRoundCounters {
            round_leaves: accumulators.gumbel_round_leaves,
            rounds: accumulators.gumbel_rounds,
        },
    ) {
        McTSSearchResult::Completed(gs, sims_served) => {
            // R335(c): the per-search visit count, as a MAX rather than a mean — a mean
            // hides a single overshooting search, and the property is an upper bound.
            accumulators
                .max_sims_per_search
                .fetch_max(sims_served as u64, Ordering::Relaxed);
            gs
        }
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
    let policy = MovePolicy::Ls(tree.get_policy_ls(temperature, policy_stride));

    // Accumulate MCTS health stats once per search (not in the inner sim loop).
    {
        let (depth, conc) = tree.last_search_stats();
        accumulators
            .mcts_depth_accum
            .fetch_add((depth * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators
            .mcts_conc_accum
            .fetch_add((conc * 1_000_000.0) as u64, Ordering::Relaxed);
        accumulators.mcts_stat_count.fetch_add(1, Ordering::Relaxed);
        accumulators.mcts_quiescence_fires.fetch_add(
            tree.quiescence_fire_count.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
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

    // The training target's semantics are the search kind's own answer — there is no
    // second flag that can disagree with the search that produced the tree.
    let target_policy = if ctx.search_kind.completed_q_target() {
        MovePolicy::Ls(tree.get_improved_policy_ls(policy_stride, ctx.c_visit, ctx.c_scale))
    } else {
        policy.clone()
    };

    let record_full_search = move_is_full_search;

    // LAW-18 (DESIGN_T §3.6): count a move whose exported target carries
    // off-window (overflow) mass — the restored-mass fire-rate; pre-Phase-T
    // this population was being truncated by the coverage gate.
    let MovePolicy::Ls(ls) = &target_policy;
    if ls.overflow.values().any(|&p| p > 0.0) {
        accumulators
            .export_offwindow_mass_moves
            .fetch_add(1, Ordering::Relaxed);
    }

    // ── Sample and apply move (ZOI-filtered legal set) ──
    let Some(move_idx) = select_move(
        board,
        move_history,
        &policy,
        gumbel_state,
        ctx,
        agg_trunk_sz,
        tree,
        rng,
    ) else {
        return MoveOutcome::Break;
    };

    // ── Record position (BEFORE apply_move) ──
    {
        let visit_capacity = ctx.visit_capacity.expect(
            "graph record dispatch requires the derived visit capacity — composed in \
             SelfPlayRunner::new's graph arm (R255)",
        );
        // R347(a) — the sparse row's explicit support is the search's OWN visited-candidate
        // set, read from the tree rather than inferred from the target: under Gumbel a
        // visited candidate can carry LESS mass than an unvisited one with a strong prior,
        // so "the top m by mass" is a different set and would put exact entries in the tail.
        let explicit_support = if ctx.search_kind.stores_sparse_rows() {
            Some(
                tree.visited_root_child_cells()
                    .into_iter()
                    .collect::<fxhash::FxHashSet<(i32, i32)>>(),
            )
        } else {
            None
        };
        if let Err(err) = record_position_graph_dispatch(
            board,
            &target_policy,
            agg_trunk_sz,
            record_full_search,
            graph_records_vec,
            visit_capacity,
            explicit_support.as_ref(),
        ) {
            // LAW-14: a target-integrity defect is RUN-FATAL — latch the typed
            // message (variant name in Display) and halt; the bridge drain face
            // raises it to the supervisor (DESIGN_T §3.4).
            fatal_latch.store(err.to_string());
            return MoveOutcome::Break;
        }
    }

    if board.apply_move(move_idx.0, move_idx.1).is_err() {
        return MoveOutcome::Break;
    }
    move_history.push((move_idx.0, move_idx.1));
    accumulators
        .positions_generated
        .fetch_add(1, Ordering::Relaxed);
    MoveOutcome::Played
}

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
    _move_history: &[(i32, i32)],
    policy: &MovePolicy,
    gumbel_state: Option<MctxRootState>,
    ctx: MovePlayContext,
    agg_trunk_sz: i32,
    tree: &MCTSTree,
    rng: &mut ThreadRng,
) -> Option<(i32, i32)> {
    let full_legal = board.legal_moves();
    if full_legal.is_empty() {
        return None;
    }

    let legal = full_legal;

    // Move selection: Gumbel winner or visit-count sampling, gated on ply RELATIVE to
    // game start.
    //
    // `gumbel_explore_moves` IS item 7's switch, and no new key is needed for it:
    // the paper's action selection is the Sequential-Halving winner and the
    // exploration comes from the Gumbel draw itself, so a Gumbel run that wants no
    // visit-count sampling mints `gumbel_explore_moves: 0` and takes the winner
    // from the first ply. A second boolean would be a second authority over one
    // behaviour.
    let use_gumbel_winner = gumbel_state.is_some()
        && relative_explore_gate(
            board.ply.index() as usize,
            ctx.game_start_ply,
            ctx.gumbel_explore_moves,
        );
    // Mctx's final action: the highest-scoring of the MOST-VISITED children, which is
    // Sequential Halving's answer rather than a visit-count sample.
    let winner_pool = if use_gumbel_winner {
        gumbel_state.and_then(|state| state.best_action(tree, ctx.c_visit, ctx.c_scale))
    } else {
        None
    };
    let sampled = |rng: &mut ThreadRng| match policy.sample(&legal, board, agg_trunk_sz) {
        Some(idx) => idx,
        None => *legal.choose(rng).unwrap(),
    };
    let move_idx = if let Some(best_pool) = winner_pool {
        let val = tree.pool[best_pool as usize].action_idx;
        let mq = (val >> 16) as i32 - 32768;
        let mr = (val & 0xFFFF) as i32 - 32768;
        if legal.contains(&(mq, mr)) {
            (mq, mr)
        } else {
            sampled(rng)
        }
    } else {
        sampled(rng)
    };
    Some(move_idx)
}

#[cfg(test)]
mod explore_gate_tests {
    use super::relative_explore_gate;

    /// ⊕ GUMBEL-REPAIR-1 item 7 — the first-N-ply visit sampling IS a config switch, and
    /// `selfplay.gumbel_explore_moves` is it. No second key is needed and none was added.
    ///
    /// The paper's action selection is the Sequential-Halving winner and the exploration
    /// comes from the Gumbel draw itself, so "visit sampling off" is `explore_moves: 0` —
    /// the gate then opens at the game's own first ply and every move is the winner. A
    /// second boolean would be a second authority over one behaviour (R1).
    #[test]
    fn zero_explore_moves_takes_the_winner_from_the_first_ply() {
        for ply in 0..4 {
            assert!(
                relative_explore_gate(ply, 0, 0),
                "at explore_moves 0 the gate is open at ply {ply}"
            );
        }
    }

    /// The shipped value samples for the first ten plies and takes the winner after.
    #[test]
    fn the_shipped_value_samples_for_exactly_its_span() {
        for ply in 0..10 {
            assert!(
                !relative_explore_gate(ply, 0, 10),
                "ply {ply} still samples"
            );
        }
        assert!(
            relative_explore_gate(10, 0, 10),
            "the gate opens AT the span, not past it"
        );
    }

    /// D-WS3V3: the span is RELATIVE to the game's own start, so a seeded game that begins
    /// deep in a replayed prefix explores for its own first `explore_moves` moves rather
    /// than opening the gate immediately because the absolute ply is already large.
    #[test]
    fn the_span_is_relative_to_the_games_start() {
        assert!(
            !relative_explore_gate(40, 35, 10),
            "5 moves into a game started at ply 35"
        );
        assert!(relative_explore_gate(45, 35, 10), "and open 10 moves in");
        // A start AFTER the ply saturates rather than wrapping.
        assert!(!relative_explore_gate(3, 35, 1));
    }
}
