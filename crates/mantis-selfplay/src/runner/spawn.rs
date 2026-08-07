//! Worker spawn loop (WP6 D1/D2) — `SelfPlayRunner::start_impl`, ported from the
//! frozen `worker_loop/mod.rs`.
//!
//! Resolves the per-worker `WorkerGeometry` ONCE via the closed
//! [`super::params::resolve_geometry`] match (D2 — no `None → v6` fallback, no
//! `_ =>` arm), selects the spec-keyed `SymTables` singleton via
//! `sym_tables_for(spec)` (absent spec is an error caught at `new()`, LAW-11),
//! Arc-clones the SHARED accumulators (never fresh-per-worker), gives each worker
//! its dense/graph inference-queue producer handles, and spawns a thread running
//! [`super::game::run_worker_thread`].

use std::sync::atomic::Ordering;
use std::thread;

use crate::replay::sym::{sym_tables_for, SymTables};

use super::atomics::WorkerAtomics;
use super::params::{
    self, ExplorationFlags, ForcedWinPolicy, MoveConstraintFlags, SearchFlags, SeedCorpus,
    SolverInLoop, WorkerChannels, WorkerParams,
};
use super::stats::WorkerStats;
use super::{game, SelfPlayRunner};

/// Run one worker body, converting a panic into a COUNTED, run-halting event.
///
/// A worker that panicked used to vanish silently: `thread::spawn` parks the panic in the
/// `JoinHandle`, `SelfPlayRunner::stop()` threw that result away (`let _ = handle.join()`),
/// and `running` stayed `true`. The pool therefore kept reporting healthy while producing
/// nothing — the failure presented as "self-play is slow", which is the most expensive way
/// for it to present.
///
/// Catching HERE rather than reading join results at `stop()` is what makes the halt LIVE:
/// the run stops at the panic, not whenever someone gets round to shutting down. It also
/// leaves `join()` returning `Ok`, so `stop()`'s own join check counts only panics that
/// ESCAPED this function and the counter never double-counts one death.
///
/// Store-then-halt, matching `store_fatal_defect`'s ordering: the count is visible BEFORE
/// the flag flips, so a supervisor woken by `!is_running()` can always read a non-zero
/// `worker_panics` and attribute the halt instead of guessing.
///
/// `AssertUnwindSafe` is the honest annotation and not a workaround: the worker bundles are
/// moved in and dropped with the thread, and the only state touched after the catch is the
/// two atomics. Nothing observes a half-updated worker, because the run halts.
pub(crate) fn guard_worker<F: FnOnce()>(
    worker_panics: &std::sync::atomic::AtomicU64,
    running: &std::sync::atomic::AtomicBool,
    body: F,
) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err() {
        worker_panics.fetch_add(1, Ordering::SeqCst);
        running.store(false, Ordering::SeqCst);
    }
}

impl SelfPlayRunner {
    /// Spawn `n_workers` self-play threads. Idempotent: a second call while
    /// already running is a no-op (the `running.swap(true)` guard).
    ///
    /// Each worker owns its own `MCTSTree`, RNG and per-game `Board`. Shared state
    /// is reached through the `Arc` accumulators on `SelfPlayRunner`; all workers
    /// are joined by `stop()`.
    pub(crate) fn start_impl(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        // §100 defense-in-depth: game-level (`fast_prob`) and move-level
        // (`full_search_prob`) playout-cap randomisers must not both be active.
        assert!(
            !(self.config.fast_prob > 0.0 && self.config.full_search_prob > 0.0),
            "playout-cap mutex violated: fast_prob={} and full_search_prob={} \
             are both > 0 (§100 — game-level and move-level caps are mutually \
             exclusive)",
            self.config.fast_prob,
            self.config.full_search_prob,
        );

        // §130/§173: the spec-keyed 12-fold dihedral scatter tables (shared
        // `&'static SymTables`). No `None → v6` fallback — an absent spec is an
        // error at `new()` (LAW-11).
        // UNREAD ON THE GRAPH PATH (R28 rider, labeled at WPCLEAN): under `gnn_axis_v1`
        // this binding still materializes the size_19 dense scatter singleton, but the
        // graph record dispatch takes no tables and HEXG D6 augmentation rotates via the
        // shared `rotate_axial` primitive (`replay/sym.rs`), not these scatters. The
        // binding is kept unconditional because it is spec-keyed, cheap (shared static),
        // and the dense arms of the same worker loop do read it.
        let sym_tables_static: &'static SymTables = sym_tables_for(self.spec);

        // D2: resolve the per-worker geometry ONCE via the closed-match resolver
        // (`Copy`, ~32 B; copied into each spawned worker).
        let geometry = params::resolve_geometry(self.spec);

        let (stats_proto, atomics_proto, channels_proto, params_proto) =
            self.build_worker_prototypes();

        let mut handles = self.handles.lock().expect("runner handles lock poisoned");
        for worker_id in 0..self.config.n_workers {
            let stats = stats_proto.clone();
            let atomics = atomics_proto.clone();
            let channels = channels_proto.clone();
            let params = params_proto.clone();
            let sym_tables = sym_tables_static;
            // The two lifecycle Arcs the panic arm needs. Captured directly rather than
            // through the runner because the worker closure is `'static` and the runner is
            // borrowed here.
            let worker_panics = self.worker_panics.clone();
            let running = self.running.clone();
            let handle = thread::spawn(move || {
                guard_worker(&worker_panics, &running, || {
                    game::run_worker_thread(
                        worker_id, stats, atomics, channels, params, sym_tables, geometry,
                    );
                });
            });
            handles.push(handle);
        }
    }

    /// Build the per-worker capture prototype (4 bundles cloned once per worker
    /// spawn). Extracted so `start_impl` stays under the clippy line threshold.
    fn build_worker_prototypes(&self) -> (WorkerStats, WorkerAtomics, WorkerChannels, WorkerParams) {
        let c = &self.config;
        let stats_proto = WorkerStats {
            games_completed: self.games_completed.clone(),
            positions_generated: self.positions_generated.clone(),
            x_wins: self.x_wins.clone(),
            o_wins: self.o_wins.clone(),
            draws: self.draws.clone(),
            positions_dropped: self.positions_dropped.clone(),
            mcts_depth_accum: self.mcts_depth_accum.clone(),
            mcts_conc_accum: self.mcts_conc_accum.clone(),
            mcts_stat_count: self.mcts_stat_count.clone(),
            mcts_quiescence_fires: self.mcts_quiescence_fires.clone(),
            cluster_value_std_accum: self.cluster_value_std_accum.clone(),
            cluster_policy_disagreement_accum: self.cluster_policy_disagreement_accum.clone(),
            cluster_variance_samples: self.cluster_variance_samples.clone(),
            solver_moves_eligible: self.solver_moves_eligible.clone(),
            solver_win_proven: self.solver_win_proven.clone(),
            solver_injected: self.solver_injected.clone(),
            solver_injected_offwindow: self.solver_injected_offwindow.clone(),
            solver_budget_exhausted: self.solver_budget_exhausted.clone(),
            solver_moves_eligible_seeded: self.solver_moves_eligible_seeded.clone(),
            solver_injected_seeded: self.solver_injected_seeded.clone(),
            seeded_games_started: self.seeded_games_started.clone(),
            export_offwindow_mass_moves: self.export_offwindow_mass_moves.clone(),
            gridls_zero_policy_rows: self.gridls_zero_policy_rows.clone(),
            k_cluster_histogram: self.k_cluster_histogram.clone(),
        };
        let atomics_proto = WorkerAtomics {
            running: self.running.clone(),
            model_version: self.model_version.clone(),
            fatal_defect: self.fatal_defect.clone(),
            target_integrity_defects: self.target_integrity_defects.clone(),
        };
        let channels_proto = WorkerChannels {
            dense_queue: self.dense_queue.clone(),
            graph_queue: self.graph_queue.clone(),
            results_queue: self.results.clone(),
            recent_game_results: self.recent_game_results.clone(),
            graph_results_queue: self.graph_results.clone(),
        };
        let params_proto = WorkerParams {
            max_moves: c.max_moves_per_game,
            leaf_batch_size: c.leaf_batch_size,
            c_puct: c.c_puct,
            fpu_reduction: c.fpu_reduction,
            quiescence_blend_2: c.quiescence_blend_2,
            fast_prob: c.fast_prob,
            fast_sims: c.fast_sims,
            standard_sims: c.standard_sims,
            temp_threshold: c.temp_threshold_compound_moves,
            temp_min: c.temp_min,
            draw_reward: c.draw_reward,
            ply_cap_value: c.ply_cap_value,
            zoi_lookback: c.zoi_lookback,
            zoi_margin: c.zoi_margin,
            c_visit: c.c_visit,
            c_scale: c.c_scale,
            gumbel_m: c.gumbel_m,
            gumbel_explore_moves: c.gumbel_explore_moves,
            dirichlet_alpha: c.dirichlet_alpha,
            dirichlet_epsilon: c.dirichlet_epsilon,
            results_queue_cap: c.results_queue_cap,
            full_search_prob: c.full_search_prob,
            n_sims_quick: c.n_sims_quick,
            n_sims_full: c.n_sims_full,
            random_opening_plies: c.random_opening_plies,
            registry_spec: self.spec,
            search_flags: SearchFlags {
                quiescence_enabled: c.quiescence_enabled,
                completed_q_values: c.completed_q_values,
                gumbel_mcts: c.gumbel_mcts,
            },
            exploration_flags: ExplorationFlags {
                dirichlet_enabled: c.dirichlet_enabled,
                selfplay_rotation_enabled: c.selfplay_rotation_enabled,
            },
            // D7: `zoi_enabled` only — the radius-jitter sibling is killed.
            move_constraint_flags: MoveConstraintFlags {
                zoi_enabled: c.zoi_enabled,
            },
            forced_win_policy: ForcedWinPolicy {
                enabled: c.forced_win_policy_enabled,
                depth: c.forced_win_policy_depth,
                weight: c.forced_win_policy_weight,
            },
            solver_in_loop: SolverInLoop {
                enabled: c.solver_enabled,
                depth: c.solver_depth,
                node_budget: c.solver_node_budget,
                neighbor_dist: c.solver_neighbor_dist,
                visit_weight: c.solver_visit_weight,
            },
            seed_corpus: SeedCorpus {
                corpus: self.seed_corpus.clone(),
                seed_fraction: c.seed_fraction,
            },
        };
        (stats_proto, atomics_proto, channels_proto, params_proto)
    }
}
