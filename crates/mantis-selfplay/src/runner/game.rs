//! R8-justify: the per-game control flow (`run_worker_thread` entry destructure +
//! `run_one_game` + `init_per_game_board` + the §P22 shutdown short-circuit + the
//! Copy per-game arg-bundles) is one cohesive orchestrator lifted from the frozen
//! `inner.rs` game band (`:275/:426/:448/:562/:618`); the fn-entry destructure
//! pattern must stay together with the game loop it feeds.
//!
//! Per-game control flow (WP6 D1/D2/D3/D11) — the worker-thread entry
//! `run_worker_thread` builds the tree once (`new_full` + `configure_quiescence`;
//! NO interior_selector, D10) and runs the outer game loop; `run_one_game` inits
//! a fresh per-game `Board` (each worker OWNS its board, `Send + !Sync`, D3),
//! runs the per-move loop, honours the §P22 shutdown short-circuit (drop an
//! in-progress game — never finalize a partial as a draw, D12), and dispatches
//! the hoisted `is_graph` finalize branch. Representation is resolved ONCE into a
//! Copy `WorkerGeometry` (D2); the per-move hot path sees cheap integer locals.
//!
//! `init_per_game_board` builds the board from the spec-derived `BoardGeometry`
//! (the frozen `Board::with_registry_spec` mapping — cluster window / threshold /
//! legal radius); the `None → v6` board arm is KILLED (D2, absent spec = error at
//! `new()`), and `legal_move_radius_jitter` is NEVER authored (D7 — the one
//! behavioural block is dead for every registry spec). The curriculum
//! per-game radius-override chain (A9, R25 commit B) is DELETED — dead weight,
//! no live caller once the Python-side curriculum plumbing is gone.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use rand::prelude::IndexedRandom;
use rand::rngs::ThreadRng;
use rand::{rng, RngExt};

use mantis_core::board::DEFAULT_CLUSTER_THRESHOLD;
use mantis_core::{Board, BoardGeometry};
use mantis_encoding::RegistrySpec;
use mantis_search::{MCTSTree, SearchKind, VIRTUAL_LOSS_PENALTY};

use crate::replay::hexg::GraphRecord;

use super::atomics::WorkerAtomics;
use super::finalize::finalize_game_graph;
use super::params::{
    ExplorationFlags, SearchFlags,
    WorkerChannels, WorkerGeometry, WorkerParams,
};
use super::search_drive::{
    play_one_move, FatalDefectLatch, InferContext, MoveAccumulators,
    MoveOutcome, MovePlayContext,
};
use super::stats::WorkerStats;
use super::{GameResultRow, WorkerResultRow};

/// Per-game-init scalar context (frozen `inner.rs:189`, MINUS the killed
/// `legal_move_radius_jitter` field, D7). `Copy`.
#[derive(Clone, Copy)]
struct PerGameInitCtx {
    max_moves: usize,
    random_opening_plies: u32,
    fast_prob: f32,
    fast_sims: usize,
    standard_sims: usize,
    draw_reward: f32,
    /// §178: terminal-via-ply-cap outcome (distinct from `draw_reward`).
    ply_cap_value: f32,
    results_queue_cap: usize,
    worker_id: usize,
}

/// Per-worker STATIC per-move scalar config, built ONCE at proto-build and copied
/// through the COLD layers into `MovePlayContext` (frozen `inner.rs:220`). `Copy`.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
struct WorkerMoveCfg {
    leaf_batch_size: usize,
    /// DERIVED HEXG visit capacity (R255) — `Some` iff this is a graph run.
    visit_capacity: Option<usize>,
    temp_threshold: usize,
    temp_min: f32,
    c_visit: f32,
    c_scale: f32,
    gumbel_m: usize,
    gumbel_explore_moves: usize,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    full_search_prob: f32,
    n_sims_quick: usize,
    n_sims_full: usize,
    search_kind: SearchKind,
    dirichlet_enabled: bool,
}

/// Per-game state outputs from `init_per_game_board` (frozen `inner.rs:593`).
struct PerGameInit {
    board: Board,
    /// Per-game graph-record accumulator — `Vec::new()` (no alloc) for grid games;
    /// only grows on the `is_graph` record branch.
    graph_records: Vec<GraphRecord>,
    move_history: Vec<(i32, i32)>,
    is_fast_game: bool,
    game_sims: usize,
}

/// Per-worker thread entry (frozen `inner.rs:275`). Owns its `MCTSTree`, RNG and
/// per-game `Board` (D3). Builds the tree once, then loops `run_one_game` until
/// `stop()` flips `running`.
pub(crate) fn run_worker_thread(
    worker_id: usize,
    stats: WorkerStats,
    atomics: WorkerAtomics,
    channels: WorkerChannels,
    params: WorkerParams,
    geometry: WorkerGeometry,
) {
    // Destructure geometry into local scalars so the per-sim hot path sees cheap
    // integers, never a `&RegistrySpec` field access (D2).
    let WorkerGeometry {
        policy_stride,
        agg_trunk_sz,
    } = geometry;
    let WorkerStats {
        games_completed,
        positions_generated,
        x_wins,
        o_wins,
        draws,
        positions_dropped,
        mcts_depth_accum,
        mcts_conc_accum,
        mcts_stat_count,
        mcts_quiescence_fires,
        max_sims_per_search,
        pcr_full_moves,
        pcr_quick_moves,
        gumbel_round_leaves,
        gumbel_rounds,
        export_offwindow_mass_moves,
    } = stats;
    let WorkerAtomics {
        running,
        model_version,
        fatal_defect,
        target_integrity_defects,
        inference_failures_total,
        graph_game_seq,
    } = atomics;
    let WorkerChannels {
        graph_queue,
        results_queue,
        recent_game_results,
        graph_results_queue,
    } = channels;
    let WorkerParams {
        max_moves,
        leaf_batch_size,
        c_puct,
        fpu_reduction,
        quiescence_blend_2,
        fast_prob,
        fast_sims,
        standard_sims,
        temp_threshold,
        temp_min,
        draw_reward,
        ply_cap_value,
        c_visit,
        c_scale,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        results_queue_cap,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        random_opening_plies,
        visit_capacity,
        registry_spec,
        search_flags:
            SearchFlags {
                quiescence_enabled,
                search_kind,
            },
        exploration_flags:
            ExplorationFlags {
                dirichlet_enabled,
            },
    } = params;

    let mut tree = MCTSTree::new_full(c_puct, VIRTUAL_LOSS_PENALTY, fpu_reduction);
    // Configure quiescence once per worker (the amended setter; D10 — NO
    // interior_selector, WP4 killed it).
    tree.configure_quiescence(quiescence_enabled, quiescence_blend_2);
    // Same posture as quiescence: per-WORKER configuration, set once, survives
    // `new_game`.
    tree.configure_search(search_kind, c_visit, c_scale);
    let mut rng = rng();
    // Per-move model-version snapshot (frozen `inner.rs:1214`): each `play_one_move`
    // dedup-pushes `model_version` (default 0 until WP7 wires the NN setter), so a
    // played-out game's drain tuple `(mv_min, mv_max, mv_distinct)` is (0, 0, 1).
    let mut version_seen: Vec<u64> = Vec::with_capacity(8);

    // Resolve the per-game `BoardGeometry` ONCE from the spec (the frozen
    // `Board::with_registry_spec` mapping — the None board arm is KILLED, D2).
    let board_geometry = BoardGeometry {
        legal_move_radius: registry_spec.legal_move_radius as i32,
        cluster_threshold: registry_spec.cluster_threshold.unwrap_or(
            DEFAULT_CLUSTER_THRESHOLD as usize,
        ) as i32,
        cluster_window_size: registry_spec.cluster_window_size.unwrap_or(
            registry_spec.board_size,
        ),
    };
    let move_accumulators = MoveAccumulators {
        mcts_depth_accum: &mcts_depth_accum,
        mcts_conc_accum: &mcts_conc_accum,
        mcts_stat_count: &mcts_stat_count,
        mcts_quiescence_fires: &mcts_quiescence_fires,
        max_sims_per_search: &max_sims_per_search,
        pcr_full_moves: &pcr_full_moves,
        pcr_quick_moves: &pcr_quick_moves,
        gumbel_round_leaves: &gumbel_round_leaves,
        gumbel_rounds: &gumbel_rounds,
        positions_generated: &positions_generated,
        export_offwindow_mass_moves: &export_offwindow_mass_moves,
    };
    // WP12-R Phase T fatal-defect latch (DESIGN_T §3.4; LAW-14).
    let fatal_latch = FatalDefectLatch {
        slot: &fatal_defect,
        fires: &target_integrity_defects,
        inference_failures: &inference_failures_total,
        running: &running,
    };
    let init_ctx = PerGameInitCtx {
        max_moves,
        random_opening_plies,
        fast_prob,
        fast_sims,
        standard_sims,
        draw_reward,
        ply_cap_value,
        results_queue_cap,
        worker_id,
    };
    let move_cfg = WorkerMoveCfg {
        leaf_batch_size,
        visit_capacity,
        temp_threshold,
        temp_min,
        c_visit,
        c_scale,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        search_kind,
        dirichlet_enabled,
    };
    // R345(b)(6) adds `graph_game_seq` as the sixth member: the game-id source travels with
    // the other finalize counters rather than as a seventh parameter, which is the shape the
    // tuple already exists to avoid.
    let finalize_counters: (
        &AtomicUsize,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
    ) = (
        &games_completed,
        &x_wins,
        &o_wins,
        &draws,
        &positions_dropped,
        &graph_game_seq,
    );

    while running.load(Ordering::Relaxed) {
        run_one_game(
            &mut tree,
            &mut rng,
            &mut version_seen,
            &running,
            &model_version,
            &graph_queue,
            registry_spec,
            board_geometry,
            init_ctx,
            policy_stride,
            agg_trunk_sz,
            move_cfg,
            move_accumulators,
            fatal_latch,
            &results_queue,
            &graph_results_queue,
            &recent_game_results,
            finalize_counters,
        );
    }
}

/// Per-game loop body (frozen `inner.rs:448`). Init board + per-game state, run
/// the inner move loop, honour the §P22 shutdown short-circuit, then dispatch the
/// hoisted `is_graph` finalize branch.
#[allow(clippy::too_many_arguments)]
fn run_one_game(
    tree: &mut MCTSTree,
    rng: &mut ThreadRng,
    version_seen: &mut Vec<u64>,
    running: &AtomicBool,
    model_version: &AtomicU64,
    graph_queue: &crate::queues::GraphQueue,
    registry_spec: &'static RegistrySpec,
    board_geometry: BoardGeometry,
    init_ctx: PerGameInitCtx,
    policy_stride: usize,
    agg_trunk_sz: i32,
    move_cfg: WorkerMoveCfg,
    move_accumulators: MoveAccumulators,
    fatal_latch: FatalDefectLatch,
    _results_queue: &Mutex<VecDeque<WorkerResultRow>>,
    graph_results_queue: &Mutex<VecDeque<GraphRecord>>,
    recent_game_results: &Mutex<VecDeque<GameResultRow>>,
    finalize_counters: (
        &AtomicUsize,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
        &AtomicU64,
    ),
) {
    let WorkerMoveCfg {
        leaf_batch_size,
        visit_capacity,
        temp_threshold,
        temp_min,
        c_visit,
        c_scale,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        search_kind,
        dirichlet_enabled,
    } = move_cfg;

    let PerGameInit {
        mut board,
        mut graph_records,
        mut move_history,
        is_fast_game,
        game_sims,
    } = init_per_game_board(board_geometry, init_ctx, rng, version_seen);

    let infer = InferContext {
        graph_queue,
        spec: registry_spec,
        model_version,
        running,
    };
    let play_ctx = MovePlayContext {
        game_start_ply: 0,
        leaf_batch_size,
        visit_capacity,
        temp_threshold,
        temp_min,
        c_visit,
        c_scale,
        gumbel_m,
        gumbel_explore_moves,
        dirichlet_alpha,
        dirichlet_epsilon,
        full_search_prob,
        n_sims_quick,
        n_sims_full,
        game_sims,
        is_fast_game,
        search_kind,
        dirichlet_enabled,
    };

    let move_iters = init_ctx.max_moves;
    for _ in 0..move_iters {
        if !running.load(Ordering::Relaxed) || board.check_win() || board.legal_move_count() == 0 {
            break;
        }

        // §115 random-opening plies: skip MCTS + recording for the first
        // `random_opening_plies` plies (skipped entirely for a seeded game).
        if board.ply.index() < init_ctx.random_opening_plies {
            let legal = board.legal_moves();
            if legal.is_empty() {
                break;
            }
            let (mq, mr) = *legal.choose(rng).unwrap();
            if board.apply_move(mq, mr).is_err() {
                break;
            }
            move_history.push((mq, mr));
            continue;
        }

        match play_one_move(
            tree,
            &mut board,
            &mut graph_records,
            &mut move_history,
            version_seen,
            rng,
            running,
            play_ctx,
            policy_stride,
            agg_trunk_sz,
            infer,
            move_accumulators,
            fatal_latch,
        ) {
            MoveOutcome::Played | MoveOutcome::Continue => {}
            MoveOutcome::Break => break,
        }
    }

    // §P22 — drain shutdown skip: if the move loop broke because `running` was
    // flipped false by `stop()`, the game is IN PROGRESS, not terminal. Returning
    // here short-circuits to the outer `while running…` guard (D12/P-05).
    if !running.load(Ordering::Relaxed) {
        return;
    }

    let (games_completed, x_wins, o_wins, draws, positions_dropped, graph_game_seq) =
        finalize_counters;
    finalize_game_graph(
        &board,
        init_ctx.max_moves,
        graph_records,
        move_history,
        version_seen,
        init_ctx.draw_reward,
        init_ctx.ply_cap_value,
        init_ctx.results_queue_cap,
        init_ctx.worker_id,
        graph_results_queue,
        recent_game_results,
        games_completed,
        x_wins,
        o_wins,
        draws,
        positions_dropped,
        graph_game_seq,
    );
}

/// Per-game board + state initializer (frozen `inner.rs:618`). Builds the board
/// from the spec-derived `BoardGeometry` (the `None → v6` arm is KILLED, D2),
/// pre-sizes the record vectors, dry-replays an optional seed prefix, samples
/// per-game rotation, and resolves the playout cap. The `legal_move_radius_jitter`
/// block is NEVER authored (D7 — dead for every registry spec).
fn init_per_game_board(
    board_geometry: BoardGeometry,
    init_ctx: PerGameInitCtx,
    rng: &mut ThreadRng,
    version_seen: &mut Vec<u64>,
) -> PerGameInit {
    // Spec is ALWAYS resolved (absent = error at `new()`, LAW-11) → build with the
    // spec-derived geometry. NO `Board::new()` fallback (D2).
    let board = Board::with_geometry(board_geometry);
    let move_history: Vec<(i32, i32)> = Vec::with_capacity(init_ctx.max_moves);
    version_seen.clear();

    // D7: `legal_move_radius_jitter` is KILLED (dead for every registry spec).

    // KataGo-style playout cap randomisation.
    let is_fast_game = init_ctx.fast_prob > 0.0 && rng.random::<f32>() < init_ctx.fast_prob;
    let game_sims = if is_fast_game {
        init_ctx.fast_sims
    } else {
        init_ctx.standard_sims
    };

    PerGameInit {
        board,
        graph_records: Vec::new(),
        move_history,
        is_fast_game,
        game_sims,
    }
}
