//! R8-justify: the per-move search phase (`play_one_move` -> `run_mcts_search` ->
//! `infer_and_expand{,_graph}` -> `select_move`) plus its Copy arg-bundles is one unit;
//! splitting it would scatter the load-bearing target-policy build ORDER, which is EXACTLY:
//! temperature-annealed visit policy -> optional completed-Q improved policy -> forced-win
//! one-hot -> solver soft-inject.
//! The graph path is rotation-free at inference: the builder is passed no `sym_idx`,
//! pinned by `crates/mantis-selfplay/tests/rotation_parity.rs`.

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
    /// Read once per move and dedup-pushed into `version_seen`; default 0 on a no-NN run.
    pub(crate) model_version: &'a AtomicU64,
    /// The runner's kill switch: tells OUR stop from a queue closed by anything else.
    pub(crate) running: &'a AtomicBool,
}

/// Per-move MCTS accumulators. `export_offwindow_mass_moves` fires once per move whose
/// exported target carries overflow mass.
#[derive(Clone, Copy)]
pub(crate) struct MoveAccumulators<'a> {
    pub(crate) mcts_depth_accum: &'a AtomicU64,
    pub(crate) mcts_conc_accum: &'a AtomicU64,
    pub(crate) mcts_stat_count: &'a AtomicU64,
    pub(crate) mcts_quiescence_fires: &'a AtomicU64,
    /// `fetch_max`ed with each search's served-leaf count.
    pub(crate) max_sims_per_search: &'a AtomicU64,
    /// The playout-cap arm as DRAWN, counted at the draw itself.
    pub(crate) pcr_full_moves: &'a AtomicU64,
    pub(crate) pcr_quick_moves: &'a AtomicU64,
    /// The Gumbel round's width; zero on a PUCT run, whose reader omits the mean.
    pub(crate) gumbel_round_leaves: &'a AtomicU64,
    pub(crate) gumbel_rounds: &'a AtomicU64,
    pub(crate) positions_generated: &'a AtomicUsize,
    pub(crate) export_offwindow_mass_moves: &'a AtomicU64,
}

/// Fatal-defect latch handle: store the typed message (first defect wins), count the fire,
/// THEN flip `running=false` — a worker panic is NOT loud, so this is what makes the typed
/// raise reach the supervisor via the drain face. TWO counters, ONE slot: `fires` counts
/// target-integrity refusals and `inference_failures` counts seam failures, because folding
/// them would make the two conjuncts indistinguishable in the event stream.
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

    /// Latch a named inference failure: same store-then-halt ordering, its OWN counter.
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

/// Per-move scalar context, `Copy` — the flat `WorkerParams` layout plus per-game dynamics.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct MovePlayContext {
    pub(crate) leaf_batch_size: usize,
    /// DERIVED HEXG visit capacity — `Some` iff this is a graph run.
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

/// Per-move policy — the ragged legal-set policy.
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

/// The Gumbel round's WIDTH, in the two terms a mean is taken over. `round_leaves / rounds` is
/// leaves per inference round trip; the mean is well below `m` and well above 1, and 1 is what
/// a batching lever that had silently stopped batching would read.
#[derive(Clone, Copy)]
pub(crate) struct GumbelRoundCounters<'a> {
    pub(crate) round_leaves: &'a AtomicU64,
    pub(crate) rounds: &'a AtomicU64,
}

/// How one inference round trip picks its leaves. `Batch(n)` is PUCT's: `n` descents from an
/// unconstrained root, spread by virtual loss. `Round(&[..])` is Gumbel's: one descent under
/// EACH surviving candidate, needing no virtual loss because their subtrees are disjoint.
#[derive(Clone, Copy)]
enum LeafSelection<'a> {
    Batch(usize),
    Round(&'a [u32]),
}

/// Result of `run_mcts_search`.
enum McTSSearchResult {
    /// The Gumbel root state (PUCT: `None`) and the leaves served, `fetch_max`ed by the caller.
    Completed(Option<MctxRootState>, usize),
    RootExpansionFailed,
    /// A leaf inference FAILED. The search is abandoned here and never reports `Completed`.
    InferenceFailed(InferenceSeamFailure),
}

/// A leaf inference that FAILED, as distinct from a shutdown.
///
/// Pre-fix every failure arm collapsed to `return 0`: the reason was dropped, the sim loop
/// `break`ed on `n == 0`, and the search still reported `Completed`. A search that backed up
/// ZERO visits then reached the exporter, which manufactured a target out of the noise-mixed
/// priors, and the failure resurfaced 100+ plies later as an unrelated-looking refusal.
///
/// **A DRAIN SHUTDOWN IS NOT A FAILURE, AND `is_closed()` IS THE WRONG WAY TO SAY SO.**
/// `close()` takes no reason, and the Python inference server closes the batcher from a
/// `finally` on ANY loop exit, so a server DEATH closed the queue and every in-flight failure
/// re-entered through the shutdown door. The discriminator is the RUNNER'S OWN KILL SWITCH,
/// stored false BEFORE either `close()` and before the inference server is stopped: on a clean
/// stop the flag is already false when the waiter wakes, on a server death it is still true.
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

/// Classify one inference-failure arm: `running == false` is OUR OWN `stop()`, the
/// drain-shutdown skip (`Ok(0)`); anything else is the named run-fatal seam failure. `SeqCst`
/// matches the store side, so the flag a waking waiter reads is ordered against the close that
/// woke it.
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

/// Selects leaves, encodes per-cluster state, submits to the dense inference queue,
/// forward/inverse-scatters under the per-game symmetry, aggregates per-leaf policies, and runs
/// `expand_and_backup`.
///
/// # Errors
/// [`InferenceSeamFailure`] when a leaf inference FAILS on an OPEN queue. An empty leaf set is
/// `Ok(0)`, search exhaustion rather than failure, and so is a failure arm reached with the
/// queue already closed.
/// GNN counterpart. Builds ONE axis graph per evaluated leaf (no reuse, no patching), submits
/// the batch in ONE `submit_graphs_and_wait`, and expands against the BUILDER's per-leaf
/// `window_center`. Rotation-free at inference.
///
/// # Errors
/// A build-guard trip or a graph-inference failure is a named [`InferenceSeamFailure`].
// `#[cold]`/`#[inline(never)]` are DELETED with the dense arm: they told LLVM to optimize this
// as the unlikely branch, and it is now the only inference path there is.
fn infer_and_expand_graph(
    tree: &mut MCTSTree,
    selection: LeafSelection<'_>,
    agg_trunk_sz: i32,
    infer: InferContext,
) -> Result<usize, InferenceSeamFailure> {
    // A tree/board desync is a NAMED run-fatal seam failure, not a bare panic.
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
        // Order is irrelevant: the builder coordinate-sorts. `Cell`/`Player` are `#[repr(i8)]`.
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
            // Seam guard tripped (unreachable for a valid self-play board). NOT routed through
            // `seam_or_shutdown`: a build guard is a pure function of the board, so a closed
            // queue cannot cause it and cannot excuse it.
            Err(reason) => {
                return Err(InferenceSeamFailure::new(
                    "graph",
                    "build_leaf_graph",
                    reason,
                ))
            }
        }
    }

    // Submit the WHOLE leaf batch in one shot: one graph at a time put exactly one leaf in
    // flight per worker, making the collector's saturation threshold structurally unreachable.
    // The returned `Vec` is indexed by SUBMISSION ORDER, which `expand_and_backup_ls_at` requires.
    let results = infer.graph_queue.submit_graphs_and_wait(graphs);
    // COLLECT-ALL-THEN-DECIDE: every waiter has resolved by the time this Vec exists, so the
    // refusal below cannot orphan one, and `Err(reason)` is carried into the named failure.
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
    // Expand frame trunk = spec.trunk_size — the SAME trunk the builder baked into
    // `policy_scatter_index`; a drifted frame silently misreads every in-window slot.
    assert_eq!(
        agg_trunk_sz, infer.spec.trunk_size as i32,
        "graph trunk mismatch: spec agg_trunk_sz vs spec graph trunk_size"
    );
    tree.expand_and_backup_ls_at(&aggregated_ls, &aggregated_values, &centers, agg_trunk_sz);
    Ok(n)
}

/// Two-branch dispatcher on the ONE search kind: Gumbel (Gumbel-Top-k root sampling +
/// Sequential Halving, no Dirichlet — the Gumbel draw IS the root exploration) or PUCT.
///
/// THE ROOT'S OWN EVALUATION IS CHARGED under both kinds, which is what makes `N` mean `N
/// leaves`; a config key for the charge made "equal NN work at a fixed budget" falsifiable.
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
    // Both kinds open with ONE leaf, the root itself, charged against the budget on both arms.
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
                // THE ROUND, not the simulation, is the unit, and the batch is re-derived from
                // the tree's own visit counts each round — which is what makes the halving
                // Mctx's rather than a phase allocation.
                let mut round = state.round_batch(tree, c_visit, c_scale);
                if round.is_empty() {
                    break;
                }
                // The budget is the last word: a round is never allowed to overspend it.
                round.truncate(budget - spent);
                // A candidate the root does not own is a bookkeeping defect and takes the
                // run-fatal exit, via the same range check every other forced descent takes.
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
                // The round's own width, logged where it is decided: a lever whose fire rate is
                // not in the run cannot be told from one issuing a leaf per trip.
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
                // The last batch is sized to the REMAINING budget: unclamped, this loop served
                // 53-56 sims against a configured 50.
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

/// Orchestrate one full move: playout-cap selection, MCTS search, stat accumulation,
/// target-policy build, sampling, position recording (BEFORE apply), and apply-move.
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
            // A MAX rather than a mean: a mean hides a single overshooting search.
            accumulators
                .max_sims_per_search
                .fetch_max(sims_served as u64, Ordering::Relaxed);
            gs
        }
        McTSSearchResult::RootExpansionFailed => return MoveOutcome::Continue,
        // Store-then-halt on its OWN counter, so the supervisor reads the inference failure
        // that killed the run instead of a refusal a hundred plies downstream.
        McTSSearchResult::InferenceFailed(err) => {
            fatal_latch.store_inference_failure(err.to_string());
            return MoveOutcome::Break;
        }
    };

    if !running.load(Ordering::Relaxed) {
        return MoveOutcome::Break;
    }

    // No visits, no target — BEFORE any exporter runs and arm-independent. `policy` below feeds
    // BOTH the recorded target and the move played, so a zero-visit search would otherwise also
    // sample its move from the prior fallback.
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

    // Snapshot the model version once per move; a no-NN run's drain tuple is (0, 0, 1).
    {
        let v = infer.model_version.load(Ordering::Relaxed);
        if !version_seen.contains(&v) {
            version_seen.push(v);
        }
    }

    // The target's semantics are the search kind's own answer — no second flag can disagree.
    let target_policy = if ctx.search_kind.completed_q_target() {
        MovePolicy::Ls(tree.get_improved_policy_ls(policy_stride, ctx.c_visit, ctx.c_scale))
    } else {
        policy.clone()
    };

    let record_full_search = move_is_full_search;

    // The restored-mass fire-rate: this population used to be truncated by the coverage gate.
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
        // The sparse row's support is the search's OWN visited-candidate set, read from the
        // tree: under Gumbel a visited candidate can carry LESS mass than an unvisited one.
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
            // A target-integrity defect is RUN-FATAL: latch the typed message and halt.
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

/// Per-move legal-move sampler. ZOI-filters when enabled, picks via Gumbel winner (post
/// exploration gate) or visit-count sampling, falls back to uniform random.
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

    // Gated on ply RELATIVE to game start. `gumbel_explore_moves` is the whole switch: the
    // action selection is the Sequential-Halving winner and the exploration comes from the
    // Gumbel draw, so "no visit sampling" is `gumbel_explore_moves: 0`.
    let use_gumbel_winner = gumbel_state.is_some()
        && relative_explore_gate(
            board.ply.index() as usize,
            ctx.game_start_ply,
            ctx.gumbel_explore_moves,
        );
    // Mctx's final action: the highest-scoring of the MOST-VISITED children.
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

    /// The first-N-ply visit sampling IS this switch and needs no second key: the action
    /// selection is the Sequential-Halving winner, so `explore_moves: 0` opens the gate at the
    /// game's own first ply.
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

    /// RELATIVE to the game's own start, so a seeded deep-prefix game still explores.
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
