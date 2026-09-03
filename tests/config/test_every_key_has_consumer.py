""">300 justify (R8): it is mostly DATA — the
184-entry `CONSUMER_REGISTRY` is the LAW-08 authority itself, one entry per schema leaf, and
splitting it would create a second registry copy to keep in sync (there is already exactly one
deliberate duplicate, `test_every_key_has_consumer_p2.py`). The five tests below are short by
comparison. WPMAIN moved the registry 170 -> 175 (R120 `eval_enabled`, R122's
`monitor.disk_guard.*` family, R126 `train.device`); WP12-R R178(a) moved it 175 -> 174, the
first DOWNWARD move (`train.buffer_save_interval` deleted as a dead knob under R116/LAW-08).
The registry figure is re-derived at HEAD from the live `len(CONSUMER_REGISTRY)`, never
transcribed (SF-7, REVIEW-impl F-4).

O15 — every-key-has-consumer bijection (LAW-08).

SCHEMA KEYS ONLY (not registered encodings — that is gate-8's disjoint concern). Enumerate
leaf key-paths of RunConfig.model_fields and assert the set equals an explicit CONSUMER_REGISTRY.
`selfplay.legal_move_radius_schedule`/`RadiusStage` are GONE (WPSC Phase 2 SC-A2 forced-fallout,
DESIGN_P2.md §5) — the NIT-3 "enumeration stops at a list[SubModel] field" note that used to
apply to `RadiusStage.step`/`.radius` no longer has a subject; `selfplay.mcts.*`/`selfplay.
playout_cap.*` are ordinary nested `StrictModel` leaves, fully enumerated like every other
section.

WPMINT DR-6 (R93): the walker now descends through OPTIONAL nested blocks too. The old stop
condition was blind to `Block | None` — the R79 arming idiom — and the reason recorded here
for that blindness was itself wrong; see `_nested_block` below.
"""
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from mantis.config.schema import RunConfig

# Each value names a REAL WP8/WP11-A reader; every "emit" reader genuinely appears in the
# O6 payload. WP11-A extends this registry with every eval.gate/eval.ladder leaf (design
# §c.1) — none of these are yet threaded into config/emit.py's resolved payload (that
# would break the pre-existing, non-oracle O6 9-knob pin in
# tests/config/test_resolved_config_emit.py, which this WP does not touch); their live
# consumer is the eval/bots/arena machinery cited below, not the emit payload.
CONSUMER_REGISTRY = {
    "schema_version": "loader version-pin + emit",
    "run_id": "mint header stamp + emit",
    "seed": (
        "seed_everything (mantis.util.determinism, R30a) -> mantis.run.build_run_collaborators, the ONE boot "
        "site for a RUN; mantis.diagnostics.worker_sweep seeds per rung from the same key (F-RESIT-10) + "
        "emit source-tag"
    ),
    "eval_enabled": (
        "mantis.run.compose_run -> the `wired_sources` eval_round declaration AND the"
        " build_eval_pipeline branch (R120; the deleted `compose_run(eval_enabled=…)`"
        " parameter's code-side default True died with it)"
    ),
    "allocator_posture": (
        "mantis.config.resolve.allocator_posture.resolve_allocator_posture -> the boot "
        "assertion in mantis.run.build_run_collaborators (the run process, before the first "
        "CUDA allocation) AND -> RoundSpec.allocator_posture -> mantis.eval.worker.run_round's "
        "first statement (the eval child, a SECOND allocator on the same card in its own "
        "process); audited by armed_aborts row `allocator_posture_minted` (RECAL-PREP, "
        "R308(g)(i))"
    ),
    "identity.encoding": "reconcile_encoding + encoding regime-parity (O11) + emit",
    "identity.representation": "resolve_amp_dtype + IdentityConfig runtime consistency guard (F1) + O11 + emit",
    "identity.arch_kind": "arch_from_spec_and_config via declared_arch_kind -> select_arch (the "
                          "arch-selector ROW, R330(e); OPTIONAL, absent until run6's mint writes it "
                          "per R323(b); absence resolves to the pinned incumbent)",
    "identity.warm_start.checkpoint": "resolve_bc_warm_start -> apply_bc_warm_start, called from "
                                      "init_trainer's FRESH branch (the BC warm-start ROW, R332(d) / "
                                      "AUDIT-1 F-19; OPTIONAL, absent until run6's mint writes it)",
    "identity.warm_start.net_hash": "resolve_bc_warm_start -> apply_bc_warm_start identity check: "
                                    "net_param_hash of the net rebuilt from the checkpoint's OWN "
                                    "stamp must equal this, else WarmStartIdentityError (R332(d))",
    "eval.random_model_sims": "resolve_eval_model_sims (random floor) + sims regime-parity (O9) + emit",
    "eval.sealbot_model_sims": "resolve_eval_model_sims (sealbot rungs) + sims regime-parity (O9) + emit",
    "eval.kraken_model_sims": "resolve_eval_model_sims (kraken rungs)",
    "eval.strix_model_sims": "resolve_eval_model_sims (strix rungs)",
    "eval.random_floor_games": "worker.py random-floor block game count",
    "eval.worker_device": "build_eval_pipeline child-process device",
    "eval.round_timeout_sec": "pipeline.py mid-round subprocess join bound",
    "eval.worker_kill_grace_sec": "pipeline.py terminate->kill grace",
    "eval.ply_cap_adjudication.criterion":
        "resolve_ply_cap_adjudication -> RoundSpec.ply_cap_adjudication -> worker.py "
        "_build_adjudicator -> arena/adjudicate.py PlyCapAdjudicator.measure",
    "eval.ply_cap_adjudication.min_margin":
        "resolve_ply_cap_adjudication -> RoundSpec.ply_cap_adjudication -> worker.py "
        "_build_adjudicator -> arena/adjudicate.py PlyCapAdjudicator.adjudicate award test",
    "eval.strength_floor.probe_games":
        "resolve_strength_floor -> RoundSpec.strength_floor -> worker.py _play_floor_probe "
        "game count",
    "eval.strength_floor.min_decisive_rate":
        "resolve_strength_floor -> RoundSpec.strength_floor -> floor_gate.py "
        "evaluate_strength_floor decisive-rate bar",
    "eval.strength_floor.min_winrate":
        "resolve_strength_floor -> RoundSpec.strength_floor -> floor_gate.py "
        "evaluate_strength_floor win-rate bar",
    "eval.gate.stride": "pipeline.py promotion-capable round stride",
    "eval.gate.screen_games": "worker.py gate screen block",
    "eval.gate.confirm_games": "worker.py gate confirm block",
    "eval.gate.promotion_winrate": "aggregate.py gate truth table",
    "eval.gate.screen_confirm_lo": "aggregate.py escalation test",
    "eval.gate.deploy_sims": "arena/deploy_head.py sims",
    "eval.gate.opening_book": "arena/books.py gate openings",
    "eval.gate.bootstrap_resamples": "aggregate.py bootstrap CI",
    "eval.gate.min_distinct_per_pair": "aggregate.py low-power guard",
    "eval.gate.seed_base": "aggregate.py bootstrap seed + worker.py opening seeds",
    "eval.ladder.rungs": "ladder.py LadderState rungs",
    "eval.ladder.round_games": "ladder.py allocate_games budget",
    "eval.ladder.min_games_per_active_rung": "ladder.py allocate_games floor",
    "eval.ladder.graduation_wr_lower_ci": "ladder.py graduation transition",
    "eval.ladder.graduation_consec_rounds": "ladder.py graduation streak",
    "eval.ladder.activation_wr_lower_ci": "ladder.py activation transition",
    "eval.ladder.calibration_every_k_rounds": "ladder.py calibration cadence",
    "eval.ladder.calibration_games": "ladder.py calibration allocation",
    "eval.ladder.bootstrap_resamples": (
        "pipeline.py RoundSpec.ladder_bootstrap_resamples -> worker.py aggregate_rung (M-2)"
    ),
    "eval.ladder.bootstrap_ci_level": (
        "pipeline.py RoundSpec.ladder_bootstrap_ci_level -> worker.py aggregate_rung (M-2)"
    ),
    "eval.ladder.bt_prior_games": "bt.py fit_bt prior",
    "eval.ladder.bootstrap_seed": (
        "pipeline.py RoundSpec.ladder_bootstrap_seed -> worker.py aggregate_rung (M-2)"
    ),
    # WPSC Phase 2 SC-A1 (R-TRAINCONFIG-SCHEMA closure): every TrainConfig leaf's live
    # consumer is TrainHParams.from_config (trainer/core.py), which reads config["train"]
    # directly (no flat-key fallback).
    "train.lr": "TrainHParams.from_config -> optimizer ctor (trainer/core.py)",
    "train.weight_decay": "TrainHParams.from_config -> build_param_groups (trainer/core.py)",
    "train.grad_clip": "TrainHParams.from_config -> fp16_backward_step max_grad_norm",
    "train.fp16": "TrainHParams.from_config -> Trainer fp16/scaler gate",
    "train.device":
        "torch.device(config.train.device) in mantis.run.build_run_collaborators ->"
        " init_trainer(device=…) AND WorkerPool(device=…) (R126; the retired --device flag"
        " on both callers)",
    "train.amp_dtype": (
        "resolve_amp_dtype (R30b single authority) -> amp_dtype_for -> "
        "Trainer/InferenceServer/cuda_warmup autocast dtype"
    ),
    "train.lr_schedule": "TrainHParams.from_config -> Trainer._build_scheduler",
    "train.total_steps": "TrainHParams.from_config -> Trainer._build_scheduler T_max fallback",
    "train.scheduler_t_max": "TrainHParams.from_config -> Trainer._build_scheduler T_max",
    "train.eta_min": "TrainHParams.from_config -> Trainer._build_scheduler eta_min",
    "train.min_lr": "TrainHParams.from_config -> Trainer._build_scheduler eta_min fallback",
    "train.checkpoint_interval": "TrainHParams.from_config ->"
                                " Trainer._maybe_periodic_checkpoint, the ONE periodic-save gate,"
                                " called by BOTH the dense and the graph step tail (R173)",
    "train.actor_sync_cadence_steps": "resolve_actor_sync_cadence -> compose_run -> ActorSync.maybe_sync (WP-UNFREEZE K1)",
    "train.max_train_steps":
        "resolve_max_train_steps -> compose_run -> StepCoordinatorConfig.stop_step",
    # WPAX Phase D (R65/R80) registered the BLOCK as one leaf, because `_leaf_paths` used to
    # recurse only into `isinstance(ann, type) and issubclass(ann, BaseModel)`. WPMINT DR-6
    # (R93) closes that: the cause was OPTIONALITY, not nesting — `DrawRateAbortConfig |
    # None` is a `UnionType`, while a NON-optional nested block (`monitor.drain.*`) was
    # descended into all along. `Block | None` is the house arming idiom (R79), so the hole
    # was generic to every future arming block: a fourth, wholly unconsumed key inside
    # `DrawRateAbortConfig` passed the full tier plus gates 7 and 12 green. The three inner
    # keys are registered individually now, each at the call site that reads it.
    "train.draw_rate_abort.threshold":
        "resolve_draw_rate_abort -> StepCoordinatorConfig.draw_rate_abort -> step.py"
        " _run_hard_abort_gates -> check_draw_rate_collapse(threshold=)",
    "train.draw_rate_abort.min_step":
        "resolve_draw_rate_abort -> StepCoordinatorConfig.draw_rate_abort -> step.py"
        " _run_hard_abort_gates -> check_draw_rate_collapse(min_step=)",
    "train.draw_rate_abort.N_pool_min":
        "resolve_draw_rate_abort -> StepCoordinatorConfig.draw_rate_abort -> step.py"
        " _run_hard_abort_gates -> pooled_draw_rate(N_pool_min=) [R92]",
    "train.draw_rate_abort.consec":
        "resolve_draw_rate_abort -> StepCoordinatorConfig.draw_rate_abort -> step.py"
        " _run_hard_abort_gates -> check_draw_rate_collapse(consec=) [WPMINT K-B]",
    # WPMINT Phase K-B (CARD-COORD-KNOBS, R78/R80): the 18 step-coordinator knobs (19 until
    # R178(a) deleted `train.buffer_save_interval`, whose only consumer chain ended in the
    # no-op `_try_save_buffer` D4 arm — F-CS-2 measured it production-dead). Every
    # citation below names the path from the ONE resolver to the line that READS the value,
    # and every one was verified BY MUTATION per R93 (set the knob, drive the production
    # path, observe the consumer move) in tests/config/test_coordinator_knobs_wiring.py —
    # never by grep, because DR-11 proved a grep cannot tell a reader from a `pop`.
    "train.eval_interval":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py _maybe_kick_eval"
        " round boundary (+ promotion_capable_rounds)",
    "train.log_interval":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py _run_log_interval"
        " boundary (payload events + WARN rules + both hard-abort gates + monitor_gates)",
    "train.min_buf_size":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py O4 warmup floor",
    "train.replay_capacity":
        "resolve_coordinator_knobs -> _step_coordinator_config ->"
        " StepCoordinatorConfig.capacity -> step.py buffer_capacity (warmup event + axis"
        " payload) and mantis.run.build_run_collaborators buffer sizing",
    "train.replay_capacity_schedule":
        "resolve_coordinator_knobs -> _step_coordinator_config ->"
        " StepCoordinatorConfig.buffer_schedule -> step.py D1 buffer.resize ramp",
    "train.training_steps_per_game":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py O6"
        " _steps_budget(new_games, this, max_train_burst)",
    "train.max_train_burst":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py O6 _steps_budget"
        " ceiling",
    "train.batch_size":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " _run_training_step batch size (replaced the train_cfg/full_config dict lookup whose"
        " literal 256 was the real production value, WPMINT K-A)",
    "train.microbatch_caps.max_edges":
        "resolve_microbatch_caps -> MicrobatchCapsSpec.max_edges ->"
        " StepCoordinator._microbatch_caps (threaded as caps_provider) ->"
        " dispatch.py::_graph_step edge-term partition (graph route only)",
    "train.microbatch_caps.max_nodes":
        "resolve_microbatch_caps -> MicrobatchCapsSpec.max_nodes ->"
        " StepCoordinator._microbatch_caps (threaded as caps_provider) ->"
        " dispatch.py::_graph_step node-term partition (graph route only)",
    "train.augment":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py _run_training_step"
        " -> trainer.train_step(augment=) / assemble_mixed_batch(augment=)",
    "train.recency_weight":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py _run_training_step"
        " -> assemble_mixed_batch recency window weighting",
    "train.mixing_initial_w":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " _compute_pretrained_weight(initial_w=) (mixed batch + axis payload)",
    "train.mixing_min_w":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " _compute_pretrained_weight(min_w=)",
    "train.mixing_decay_steps":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " _compute_pretrained_weight(decay_steps=)",
    "train.hard_gn_threshold":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py D3"
        " grad_norm_hard_abort comparison (+ armed_aborts DEFERRED row, WPMINT K-B)",
    "train.hard_gn_min_steps":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py D3"
        " grad_norm_hard_abort consecutive-step count",
    "train.terminal_eval_enabled":
        "resolve_coordinator_knobs -> _step_coordinator_config -> coordinator/drain.py"
        " run_terminal_eval close-out gate",
    "train.bot_batch_share":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " _run_training_step n_bot batch slots",
    "train.selfplay_stall_timeout_sec":
        "resolve_coordinator_knobs -> _step_coordinator_config -> step.py"
        " StallWatchdog(timeout_sec=) (LAW-16 always-armed guard)",
    "train.completed_q_values": "TrainHParams.from_config -> CE-vs-KL policy loss switch",
    "train.value_target": "TrainHParams.from_config single-variant assertion (T-D)",
    "train.policy_target": "TrainHParams.from_config cross-validated vs completed_q_values (T-B)",
    "train.draw_reward": "SelfPlayHParams.from_config cross-section read (SC-A2)",
    "train.ply_cap_value": "SelfPlayHParams.from_config cross-section read (SC-A2)",
    "train.policy_prune_frac": "TrainHParams.from_config -> _prune_policy_targets",
    "train.entropy_reg_weight": "TrainHParams.from_config -> entropy bonus weight (R37)",
    "train.aux_opp_reply_weight": "TrainHParams.from_config -> aux opp-reply loss weight",
    "train.uncertainty_weight": "TrainHParams.from_config -> uncertainty loss weight",
    "train.ownership_weight": "TrainHParams.from_config -> ownership loss weight",
    "train.threat_weight": "TrainHParams.from_config -> threat loss weight",
    "train.aux_chain_weight": "TrainHParams.from_config -> chain loss weight",
    "train.ply_index_weight": "TrainHParams.from_config -> ply-index loss weight",
    "train.threat_pos_weight": "TrainHParams.from_config -> threat pos_weight tensor",
    # WPSC Phase 2 SC-A2 (R-SELFPLAYCONFIG-SCHEMA closure): every SelfplayConfig/MctsConfig/
    # PlayoutCapConfig/InferenceConfig leaf's live consumer is SelfPlayHParams.from_config /
    # InferenceHParams.from_config (mantis.selfplay.hparams), which read the nested
    # `selfplay`/`selfplay.mcts`/`selfplay.playout_cap`/`inference` sections directly.
    "selfplay.n_workers": "SelfPlayHParams.from_config -> WorkerPool worker count",
    "selfplay.leaf_batch_size": "SelfPlayHParams.from_config -> runner leaf_batch_size",
    "selfplay.max_game_moves": "SelfPlayHParams.from_config -> runner max_moves_per_game",
    "selfplay.inference_pool_size": "SelfPlayHParams.from_config -> runner inference_pool_size",
    "selfplay.completed_q_values": "SelfPlayHParams.from_config -> runner completed_q_values",
    "selfplay.c_visit": "SelfPlayHParams.from_config -> runner c_visit",
    "selfplay.c_scale": "SelfPlayHParams.from_config -> runner c_scale",
    "selfplay.gumbel_mcts": "SelfPlayHParams.from_config -> runner gumbel_mcts + WorkerPool.gumbel_mcts",
    "selfplay.gumbel_m": "SelfPlayHParams.from_config -> runner gumbel_m",
    "selfplay.gumbel_explore_moves": "SelfPlayHParams.from_config -> runner gumbel_explore_moves",
    "selfplay.results_queue_cap": "SelfPlayHParams.from_config -> runner results_queue_cap",
    "selfplay.random_opening_plies": "SelfPlayHParams.from_config -> runner random_opening_plies",
    "selfplay.rotation_enabled": "SelfPlayHParams.from_config -> runner selfplay_rotation_enabled",
    "selfplay.forced_win_policy_enabled": "SelfPlayHParams.from_config -> runner.forced_win_policy_enabled",
    "selfplay.forced_win_policy_depth": "SelfPlayHParams.from_config -> runner.forced_win_policy_depth",
    "selfplay.forced_win_policy_weight": "SelfPlayHParams.from_config -> runner.forced_win_policy_weight",
    "selfplay.solver_enabled": "SelfPlayHParams.from_config -> runner.solver_enabled",
    "selfplay.solver_depth": "SelfPlayHParams.from_config -> runner.solver_depth",
    "selfplay.solver_node_budget": "SelfPlayHParams.from_config -> runner.solver_node_budget",
    "selfplay.solver_neighbor_dist": "SelfPlayHParams.from_config -> runner.solver_neighbor_dist",
    "selfplay.solver_visit_weight": "SelfPlayHParams.from_config -> runner.solver_visit_weight",
    "selfplay.seed_fraction": "SelfPlayHParams.from_config -> runner.seed_fraction",
    "selfplay.seed_corpus_path": "SelfPlayHParams.from_config -> _load_seed_corpus",
    "selfplay.log_investigation_metrics": "SelfPlayHParams.from_config -> pool investigation logging",
    "selfplay.instrumentation_enabled": "SelfPlayHParams.from_config -> pool instrumentation gate",
    "selfplay.mcts.n_simulations": "SelfPlayHParams.from_config -> runner n_simulations",
    "selfplay.mcts.c_puct": "SelfPlayHParams.from_config -> runner c_puct",
    "selfplay.mcts.fpu_reduction": "SelfPlayHParams.from_config -> runner fpu_reduction",
    "selfplay.mcts.quiescence_enabled": "SelfPlayHParams.from_config -> runner quiescence_enabled",
    "selfplay.mcts.quiescence_blend_2": "SelfPlayHParams.from_config -> runner quiescence_blend_2",
    "selfplay.mcts.dirichlet_alpha": "SelfPlayHParams.from_config -> runner dirichlet_alpha",
    "selfplay.mcts.dirichlet_epsilon": "SelfPlayHParams.from_config -> runner dirichlet_epsilon",
    "selfplay.mcts.dirichlet_enabled": "SelfPlayHParams.from_config -> runner dirichlet_enabled",
    "selfplay.playout_cap.fast_sims": "SelfPlayHParams.from_config -> runner fast_sims",
    "selfplay.playout_cap.fast_prob": "SelfPlayHParams.from_config -> runner fast_prob",
    "selfplay.playout_cap.standard_sims": "SelfPlayHParams.from_config -> runner standard_sims",
    "selfplay.playout_cap.full_search_prob": "SelfPlayHParams.from_config -> runner full_search_prob",
    "selfplay.playout_cap.n_sims_quick": "SelfPlayHParams.from_config -> runner n_sims_quick",
    "selfplay.playout_cap.n_sims_full": "SelfPlayHParams.from_config -> runner n_sims_full",
    "selfplay.playout_cap.zoi_enabled": "SelfPlayHParams.from_config -> runner zoi_enabled",
    "selfplay.playout_cap.zoi_lookback": "SelfPlayHParams.from_config -> runner zoi_lookback",
    "selfplay.playout_cap.zoi_margin": "SelfPlayHParams.from_config -> runner zoi_margin",
    "selfplay.playout_cap.temperature_threshold_compound_moves": (
        "SelfPlayHParams.from_config -> runner temp_threshold_compound_moves"
    ),
    "selfplay.playout_cap.temp_min": "SelfPlayHParams.from_config -> runner temp_min",
    "inference.inference_batch_size": "InferenceHParams.from_config -> inference_server batch size",
    "inference.inference_max_wait_ms": "InferenceHParams.from_config -> inference_server max wait",
    "inference.trace_inference": "InferenceHParams.from_config -> inference_server tracing gate",
    "inference.compile_inference": "InferenceHParams.from_config -> inference_server compile gate",
    "inference.compile_inference_mode": "InferenceHParams.from_config -> inference_server compile mode",
    "inference.compile_inference_dynamic": "InferenceHParams.from_config -> inference_server compile dynamic",
    "inference.perf_timing": "InferenceHParams.from_config -> perf timing diagnostics",
    "inference.perf_sync_cuda": "InferenceHParams.from_config -> perf CUDA-sync diagnostics",
    # F-816-10 (R276(f)): the GRAPH inference forward's memory bound. GRAPH-ROUTE-SCOPED,
    # and the rows say so: the resolver is called from the graph branch of
    # `InferenceServer.__init__` alone, so a grid run structurally cannot reach either key.
    "inference.fused_graph_caps.max_fused_edges":
        "resolve_fused_graph_caps -> FusedGraphCapsSpec.max_fused_edges ->"
        " InferenceServer.__init__ (graph branch only, eager) -> _run_graph_loop"
        " plan_fused_forwards edge-term partition; also threaded parent-side into"
        " RoundSpec.fused_graph_caps -> the eval child's LocalInferenceEngine",
    "inference.fused_graph_caps.max_fused_nodes":
        "resolve_fused_graph_caps -> FusedGraphCapsSpec.max_fused_nodes ->"
        " InferenceServer.__init__ (graph branch only, eager) -> _run_graph_loop"
        " plan_fused_forwards node-term partition; also threaded parent-side into"
        " RoundSpec.fused_graph_caps -> the eval child's LocalInferenceEngine",
    # WPSC Phase 2 SC-A3 (R-MONITORCONFIG-SCHEMA closure): every MonitorSchemaConfig leaf's
    # live consumer is resolve_monitor_config (mantis.config.resolve.monitor), the pure 1:1
    # field-copy onto mantis.monitor.config.MonitorConfig; the 4 monitor.drain.* leaves feed
    # DrainCaps (run.py) / drain_budget_sec + _run_terminal_sync (eval/pipeline.py) through
    # their own resolver, mantis.config.resolve.drain.resolve_drain_caps (WPMINT K-A).
    # ── monitor.gate_interval (R242 / ADJ-D12) — schema-only, like `drain`/`disk_guard`:
    # named directly by compose_run (one leaf, no shape to resolve) and threaded into
    # StepCoordinatorConfig. It is the ARMING cadence; train.log_interval is narration.
    "monitor.gate_interval":
        "mantis.run.compose_run -> _step_coordinator_config ->"
        " StepCoordinatorConfig.gate_interval -> step.py _run_gate_interval"
        " (hard-abort gates + the LAW-18 monitor_gates summary)",
    "monitor.alert_entropy_min": "resolve_monitor_config -> monitor/rules.py entropy WARN",
    "monitor.collapse_threshold_nats": "resolve_monitor_config -> monitor/rules.py collapse threshold",
    "monitor.alert_grad_norm_max": "resolve_monitor_config -> monitor/rules.py grad-norm WARN",
    "monitor.alert_loss_increase_window": "resolve_monitor_config -> monitor/rules.py loss-increase window",
    "monitor.wr_hard_abort_enabled": "resolve_monitor_config -> sealbot-WR hard-abort disposition wrapper",
    "monitor.wr_rolling_consecutive_evals": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_rolling_threshold": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_rolling_min_step": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_collapse_from_peak_ratio": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_collapse_min_step": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_collapse_consecutive_evals": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_early_death_threshold": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.wr_early_death_min_step": "resolve_monitor_config -> sealbot_wr_trajectory_alert",
    "monitor.axis_warn": "resolve_monitor_config -> train/events.py emit_axis_distribution",
    "monitor.axis_alert": "resolve_monitor_config -> train/events.py emit_axis_distribution",
    "monitor.heartbeat_deadline_train_step_sec": "resolve_monitor_config -> heartbeat_watchdog.py per-source deadline",
    "monitor.heartbeat_deadline_inference_dispatch_sec": "resolve_monitor_config -> heartbeat_watchdog.py deadline",
    "monitor.heartbeat_deadline_selfplay_drain_sec": "resolve_monitor_config -> heartbeat_watchdog.py deadline",
    "monitor.heartbeat_deadline_eval_round_sec": "resolve_monitor_config -> heartbeat_watchdog.py eval-poller deadline",
    "monitor.heartbeat_poll_interval_sec": "resolve_monitor_config -> heartbeat_watchdog.py poll cadence",
    "monitor.heartbeat_file_interval_sec": "resolve_monitor_config -> heartbeat_watchdog.py file-write cadence",
    "monitor.heartbeat_close_out_deadline_sec": "resolve_monitor_config -> disarm_staleness() teardown budget",
    "monitor.heartbeat_fire_effect_timeout_sec": "resolve_monitor_config -> heartbeat fire-path effect timeout",
    "monitor.actor_lag_threshold_steps": "resolve_monitor_config -> build_run_safety -> ActorLagSpec.threshold_steps (WP-UNFREEZE K2)",
    "monitor.actor_lag_abort_enabled": "resolve_monitor_config -> build_run_safety -> ActorLagSpec.abort_enabled (WP-UNFREEZE K3)",
    "monitor.supervisor_stale_after_sec":
        "monitor/supervise.py::main --config -> load_config"
        " -> resolve_monitor_config -> Supervisor(stale_after_sec=) -> LivenessTracker staleness deadline",
    "monitor.supervisor_poll_interval_sec":
        "monitor/supervise.py::main --config -> load_config"
        " -> resolve_monitor_config -> Supervisor(poll_interval_sec=) -> the watch loop's sleep",
    "monitor.supervisor_kill_grace_sec":
        "monitor/supervise.py::main --config -> load_config"
        " -> resolve_monitor_config -> Supervisor(kill_grace_sec=) -> stop_child_cooperatively grace_sec",
    "monitor.supervisor_max_relaunches":
        "monitor/supervise.py::main --config -> load_config"
        " -> resolve_monitor_config -> Supervisor(max_relaunches=) -> the relaunch budget",
    # WPMINT Phase K-A (R93): these four citations were FALSE until this phase — the block
    # was popped by resolve_monitor_config and never reached the functions named below, which
    # a grep could not tell from a read (DR-11). The path is now named end to end and is
    # verified BY MUTATION, per key, in tests/config/test_drain_caps_wiring.py.
    "monitor.disk_guard.interval_sec":
        "resolve_disk_guard -> DiskGuard(interval_sec=…) -> the guard thread's poll cadence"
        " (mantis.run.compose_run, LAW-16 leg 3)",
    "monitor.disk_guard.warn_gb":
        "resolve_disk_guard -> DiskGuard(warn_gb=…) -> disk_alert level=warn threshold",
    "monitor.disk_guard.fail_gb":
        "resolve_disk_guard -> DiskGuard(fail_gb=…) -> disk_alert level=critical + SIGTERM",
    "monitor.drain.final_eval_drain_timeout_sec": "resolve_drain_caps -> _step_coordinator_config -> DrainCaps -> drain_budget_sec (eval/pipeline.py)",
    "monitor.drain.eval_final_drain_safety_factor": "resolve_drain_caps -> _step_coordinator_config -> DrainCaps -> drain_budget_sec (eval/pipeline.py)",
    "monitor.drain.eval_final_drain_hard_cap_sec": "resolve_drain_caps -> _step_coordinator_config -> DrainCaps -> drain_budget_sec (eval/pipeline.py)",
    "monitor.drain.terminal_eval_hard_cap_sec": "resolve_drain_caps -> _step_coordinator_config -> DrainCaps -> _run_terminal_sync budget_sec (eval/pipeline.py)",
}


def _nested_block(ann: Any) -> type[BaseModel] | None:
    """The nested config BLOCK an annotation names, seen THROUGH `Optional` / `Block | None`.

    WPMINT DR-6 (R93). The predecessor tested `isinstance(ann, type) and issubclass(ann,
    BaseModel)` and stopped there, so an OPTIONAL block was one opaque leaf. The stated
    reason ("the BLOCK is one leaf") was wrong: non-optional nested blocks — `monitor.drain`
    — were descended into all along, and the real cause was that `DrawRateAbortConfig |
    None` is a `types.UnionType`, not a `type`. Since `Block | None` is the house arming
    idiom (R79), that blindness was generic to every future arming block, and it was
    measured: a fourth, entirely unconsumed key inside `DrawRateAbortConfig` passed the full
    tier and gates 7 and 12 green.

    NIT-3 is UNCHANGED and is a different rule: `list[SubModel]` (`eval.ladder.rungs`) has a
    `list` origin, not a union one, so it stays ONE leaf. Only an `Optional`/union whose
    non-`None` arms name exactly ONE BaseModel is descended — a genuine sum of two different
    blocks has no single key-path to hand out and is left as a leaf rather than guessed at.
    """
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    if get_origin(ann) in (Union, UnionType):
        blocks = [arm for arm in get_args(ann)
                  if isinstance(arm, type) and issubclass(arm, BaseModel)]
        if len(blocks) == 1:
            return blocks[0]
    return None


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Leaf key-paths of a StrictModel; recurse into nested models (including OPTIONAL ones,
    DR-6) but STOP at any field that does not name exactly one nested block (NIT-3 —
    list[SubModel] is one leaf)."""
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        nested = _nested_block(field.annotation)
        if nested is not None:
            out.extend(_leaf_paths(nested, path))
        else:
            out.append(path)
    return out


def test_schema_leaves_equal_consumer_registry_bijection():
    leaves = set(_leaf_paths(RunConfig))
    registered = set(CONSUMER_REGISTRY)
    assert leaves == registered, (
        f"schema-only (unregistered): {leaves - registered}; "
        f"registry-only (no schema field): {registered - leaves}"
    )


def test_registry_matches_the_live_leaf_count():
    # WP11-A extends the O15 registry 8 -> 36 leaves (design §c.1: eval.gate.* + eval.
    # ladder.* + 6 new eval.* scalars). WPSC Phase 2 SC-A1 extends it further 36 -> 61
    # (design §2: 25 new `train.*` leaves, R-TRAINCONFIG-SCHEMA closure). SC-A2 removes 1
    # (`selfplay.legal_move_radius_schedule`, forced fallout of DESIGN_P2.md §5) and adds 52
    # (25 `selfplay.*` + 8 `selfplay.mcts.*` + 11 `selfplay.playout_cap.*` + 8 `inference.*`,
    # R-SELFPLAYCONFIG-SCHEMA closure): 61 - 1 + 52 = 112. SC-A3 adds 31 (27 `monitor.*` + 4
    # `monitor.drain.*`, R-MONITORCONFIG-SCHEMA closure): 112 + 31 = 143. The historical
    # test name is retired here (it said `eight` while asserting 146); the bijection test
    # above is the live invariant this file exists to hold, not this count.
    # WP-UNFREEZE adds 3 (K1 train.actor_sync_cadence_
    # steps + K2/K3 monitor.actor_lag_*): 143 + 3 = 146. WPAX S-4 adds 1
    # (train.max_train_steps, the run-length authority): 146 + 1 = 147.
    # WPAX Phase D adds 1 (train.draw_rate_abort, registered as ONE opaque block leaf):
    # 147 + 1 = 148. WPMINT DR-6 (R93) then makes `_leaf_paths` descend through OPTIONAL
    # blocks, which retires that one block leaf and surfaces its three inner keys:
    # 148 - 1 + 3 = 150. Measured, not derived: `train.draw_rate_abort` is the ONLY
    # `Block | None` field anywhere in `RunConfig`, so those three are the whole blast
    # radius; `eval.ladder.rungs` is a `list[LadderRung]` and stays one leaf under NIT-3.
    # WPMINT Phase DS (R92) SWAPS one of those three (`min_samples` -> `N_pool_min`) and the
    # count is unchanged at 150 — recorded because a swap that silently dropped or doubled a
    # key would move this number, and the bijection above would then be the only witness.
    # WPMINT Phase K-B (CARD-COORD-KNOBS, R78/R80) adds 20: the 19 flat `train.*`
    # step-coordinator knobs plus `train.draw_rate_abort.consec`, the fourth abort term R80
    # left with this card. 150 + 20 = 170. SIX further coordinator fields were DELETED rather
    # than authored (adjudication call K-a) precisely so this count would not rise by 26 with
    # six of them naming nothing — a registry entry for a dead field is the LAW-08 violation
    # this test exists to catch, written down by hand. `train.replay_capacity_schedule` is a
    # `list[ReplayCapacityStage]` and stays ONE leaf under NIT-3, exactly like
    # `eval.ladder.rungs`, so its two inner names are covered by the schedule's own entry.
    # WPMAIN (CARD-RUN-MAIN) adds 5 and the count becomes 175: `eval_enabled` (R120, the
    # code-side `compose_run` default promoted to a typed top-level key), the three
    # `monitor.disk_guard.*` leaves (R122's granted FAMILY — one block, one resolver, three
    # typed leaves, replacing four dead `dict.get` literals in a function with zero callers)
    # and `train.device` (R126, the CLI-only run input on both callers promoted to a typed
    # `train.*` key). 170 + 1 + 3 + 1 = 175. The nested `monitor.disk_guard` block itself is
    # DESCENDED by `_leaf_paths`, so it contributes three leaves and not four.
    # WP12-R R178(a) SUBTRACTS 1 — the first DOWNWARD move this count has ever made:
    # `train.buffer_save_interval` is deleted under R116/LAW-08 because F-CS-2 measured its
    # only consumer chain (`_try_save_buffer`) production-dead on every leg, and a minted key
    # with no reachable effect is the dead-knob class R1 exists to kill. 175 - 1 = 174.
    # WP12-R F2 (CARD-RUN5-GPU-OOM, R179) ADDS 2: `train.microbatch_caps.{max_edges,
    # max_nodes}` — ONE block, ONE fact ("how big may one micro-batch be"), sized TOGETHER
    # from one measured cost model. Their consumer is GRAPH-ROUTE-SCOPED and the registry
    # rows say so: the provider is threaded to `_graph_step` alone and `_grid_step` is not
    # given it, so a grid run structurally cannot reach them. 174 + 2 = 176.
    # R242 (ADJ-D12) ADDS 1: `monitor.gate_interval`, the ARMING cadence split off the
    # NARRATION cadence `train.log_interval`. It is a monitor leaf and a SCHEMA-ONLY one
    # (a third enumerated drop in `resolve_monitor_config`, beside `drain` and
    # `disk_guard`), so the runtime `MonitorConfig` field count does not move with it.
    # 176 + 1 = 177.
    # The eval-posture bundle (F-R-P2B-5) ADDS 5: `eval.ply_cap_adjudication.{criterion,
    # min_margin}` and `eval.strength_floor.{probe_games, min_decisive_rate, min_winrate}` —
    # TWO `Block | None` blocks, both minted `null` in every committed config. The walker
    # DESCENDS optional blocks (WPMINT DR-6/R93), so all five inner leaves surface here while
    # each YAML file carries exactly one line per block; the count below is the SCHEMA's, not
    # the YAML's, and the two differ by design. This also RETIRES the line above that called
    # `train.draw_rate_abort` the only `Block | None` field in `RunConfig` — it was true when
    # written and is not any more; the sentence is corrected here rather than left to be read
    # as evidence (R192(e)). 177 + 5 = 182.
    # F-816-10 (R276(f)) ADDS 2: `inference.fused_graph_caps.{max_fused_edges,
    # max_fused_nodes}` — ONE block, ONE fact ("how big may one fused inference forward
    # be"), sized TOGETHER from one measured fit against one budget, and the training cap's
    # partner over the same card. Members are NOT spelled `max_edges`/`max_nodes`: the
    # train-side OF2-9 census freezes those names and distinct names keep the two budgets
    # unconfusable. Shipped `null` in both production configs — the R119 placeholder, which
    # is schema-valid and runtime-REFUSED, never an off state. 182 + 2 = 184.
    # RECAL-PREP (R308(g)(i)) ADDS 1 and the count becomes 185: `allocator_posture`, a
    # TOP-LEVEL leaf naming the CUDA caching allocator's REGIME. TOP-LEVEL for `eval_enabled`'s
    # own recorded grounds — a root-composition fact spanning more than one section's surface,
    # with three consumers in two processes (trainer, self-play/inference, and the eval child,
    # which is a SECOND allocator on the same card). It is a `Literal | None` SCALAR, not a
    # block, so the walker contributes exactly one leaf and each YAML file carries exactly one
    # line; there is no descend-into-optional-block arithmetic here of the kind the
    # eval-posture bundle above needed. Shipped `null` in EVERY committed config — R119's
    # placeholder, schema-valid and runtime-REFUSED — because R308(g)(i) reserves the VALUE
    # for the re-calibration sitting under R282(b) and a token minted by a dispatcher would be
    # a regime nobody measured. 184 + 1 = 185.
    # R330(e): + 1 = 186 — `identity.arch_kind`, the arch-selector row, the first OPTIONAL leaf
    # this walker has counted. One leaf, one YAML line when minted; absent from every committed
    # config until run6's mint (R323(b)), read by ONE function (`arch_from_spec_and_config`).
    # R332(d): + 2 = 188 — `identity.warm_start`'s two members. ONE optional BLOCK, two leaves
    # under the walker: both are REQUIRED inside the block, which is what makes a checkpoint
    # path with no expected net hash unconstructible (AUDIT-1 F-19).
    assert len(CONSUMER_REGISTRY) == 188
    assert len(_leaf_paths(RunConfig)) == 188


def test_no_forward_reference_strings_in_registry():
    # V-NOOP strengthening (R40, WPSC Phase 3 SC-B4, DESIGN_P3.md §0/§5.1 item 3): the
    # bijection above is a pure key-SET diff — it never checks that a registry string names
    # a function that actually reads the field, so an honest-but-unconsumed entry (like
    # `train.amp_dtype` sat through all of Phase 2, DESIGN_P2 STOP CANDIDATE 4) can pass it
    # forever. This bans forward-reference-shaped language in CONSUMER_REGISTRY values.
    banned = ("SC-B3 wires", "TODO", "will be")
    hits = [
        (key, value)
        for key, value in CONSUMER_REGISTRY.items()
        for token in banned
        if token in value
    ]
    assert hits == [], f"forward-reference-shaped registry strings: {hits}"


def test_the_walker_descends_into_an_OPTIONAL_block_not_only_a_required_one():
    """WPMINT DR-6 (R93) — the walker's own predicate, driven on all three shapes at once.

    The defect this is the sole witness for: `Block | None` is the house arming idiom (R79),
    and the pre-DR-6 walker treated such a block as ONE opaque leaf, so a key added inside it
    had no LAW-08 obligation at all. Measured at Phase DR: a fourth, wholly unconsumed key
    inside `DrawRateAbortConfig` passed the full tier plus gates 7 and 12, all green. The
    bijection test above cannot witness this on its own — with the block opaque, the walker
    and the registry agreed with each other about a key neither could see.

    NIT-3 is asserted in the same breath, because the fix must not swallow it: a
    `list[SubModel]` field has no single key-path to hand out and stays ONE leaf. Without
    this arm a "descend into anything that mentions a BaseModel" implementation would pass.
    """
    class _Inner(BaseModel):
        a: int
        b: int

    class _Outer(BaseModel):
        required_block: _Inner
        optional_block: _Inner | None
        block_list: list[_Inner]
        scalar: int

    assert _leaf_paths(_Outer) == [
        "required_block.a", "required_block.b",
        "optional_block.a", "optional_block.b",
        "block_list", "scalar",
    ], (
        "an OPTIONAL nested block must be descended into exactly like a required one — "
        "optionality, not nesting, was the cause of the LAW-08 hole (DR-6) — while a "
        f"list[SubModel] stays one leaf (NIT-3); got {_leaf_paths(_Outer)}"
    )

    leaves = set(_leaf_paths(RunConfig))
    assert {"train.draw_rate_abort.threshold", "train.draw_rate_abort.min_step",
            "train.draw_rate_abort.N_pool_min"} <= leaves, (
        "the real arming block's three inner keys must each carry their own LAW-08 "
        "obligation now, not hide behind the block key"
    )
    assert "train.draw_rate_abort" not in leaves, (
        "the block itself is no longer a leaf — leaving it registered as well would put two "
        "registry entries on one fact and re-open the bijection to an unconsumed inner key"
    )


def test_bijection_bites_on_a_real_schema_mutation():
    # F5: a genuine schema mutation (a throwaway subclass adding a leaf field with no registry
    # entry) must make the bijection fail — enumeration picks up the new leaf, not set-algebra.
    class _MutatedRunConfig(RunConfig):
        phantom_leaf: int  # new schema field, no CONSUMER_REGISTRY entry

    leaves = set(_leaf_paths(_MutatedRunConfig))
    assert "phantom_leaf" in leaves
    assert leaves != set(CONSUMER_REGISTRY), "bijection must break when a new leaf is unregistered"
