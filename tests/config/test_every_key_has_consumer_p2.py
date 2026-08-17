""">300 justify (R8): it is mostly DATA — the
SECOND copy of the 184-entry `CONSUMER_REGISTRY`, deliberately duplicated so the two copies must
agree, plus the same walker. It crossed the cap at WPMINT Phase K-B, which added 20 leaves;
WPMAIN moved the registry 170 -> 175 (R120 `eval_enabled`, R122's
`monitor.disk_guard.*` family, R126 `train.device`); WP12-R R178(a) moved the registry
175 -> 174, the first DOWNWARD move (`train.buffer_save_interval` deleted as a dead knob under
R116/LAW-08). The registry figure is re-derived at HEAD from the live
`len(CONSUMER_REGISTRY)`, never transcribed (SF-7, REVIEW-impl F-4).
Splitting it would create a THIRD copy to keep in sync, which is the drift the duplication
exists to detect. The three tests below are short by comparison.

SC-A1..A4 oracle — O15 every-key-has-consumer bijection, re-asserted against the FULL
POST-Phase-2 leaf set (DESIGN_P2.md §1.1/§1.2/§4.2/§4.3 / PREREG_P2.md suite #10; edit-
target was tests/config/test_every_key_has_consumer.py).

DEVIATION FROM PREREG PATH (logged in ORACLE_NOTES_P2.md): PREREG marks this suite as an
edit to the existing file. ORACLE-WRITE's writable surface is NEW files only — this is a
new file carrying the FULL post-Phase-2 registry; IMPL retires the old 36-entry registry
at port time. RED-at-import until IMPL lands TrainConfig/SelfplayConfig(expanded)/
MonitorSchemaConfig/DrainCapsConfig/InferenceConfig.

Leaf-count arithmetic (computed mechanically from DESIGN_P2.md's field lists, not
guessed): 36 (current) - 1 (selfplay.legal_move_radius_schedule, removed) + 25 (train.*) +
25 (selfplay.* scalars) + 8 (selfplay.mcts.*) + 11 (selfplay.playout_cap.*) +
8 (inference.*) + 27 (monitor.* scalars) + 4 (monitor.drain.*) = 143;
WP-UNFREEZE adds 3 (K1/K2/K3 actor-sync knobs) = 146; WPAX S-4 adds 1
(train.max_train_steps) = 147; WPAX Phase D adds 1 (train.draw_rate_abort, registered as
ONE opaque block leaf) = 148; WPMINT DR-6 (R93) makes `_leaf_paths` descend through
OPTIONAL blocks, retiring that block leaf and surfacing its three inner keys = 150;
WPMINT Phase K-B (CARD-COORD-KNOBS, R78/R80) adds 20 — the 19 flat `train.*`
step-coordinator knobs plus `train.draw_rate_abort.consec` — = 170. Six further coordinator
fields were DELETED, not authored (call K-a): they had no reader in `src/`, so a registry
entry for them would have been the exact defect this bijection exists to catch.

This file is the SECOND copy of the registry and the walker; both copies must agree, so
every DR-6 edit landed here byte-for-byte in meaning (`tests/config/
test_every_key_has_consumer.py` carries the long-form rationale and the walker's own
oracle).

Every consumer string below cites the real read site named in DESIGN_P2.md §1.1 (train)/
§1.2 (selfplay/inference)/§4.2-4.3 (monitor/drain) — not placeholder text.
"""
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from mantis.config.schema import RunConfig

CONSUMER_REGISTRY: dict[str, str] = {
    # ── unchanged from the pre-Phase-2 registry (35 of the original 36; radius dropped) ──
    "schema_version": "loader version-pin + emit",
    "run_id": "mint header stamp + emit",
    "seed": "emit source-tag (acting RNG consumer: mantis.train.determinism.seed_everything, SC-A6)",
    "eval_enabled": (
        "mantis.run.compose_run -> the `wired_sources` eval_round declaration AND the"
        " build_eval_pipeline branch (R120; the deleted `compose_run(eval_enabled=…)`"
        " parameter's code-side default True died with it)"
    ),
    "identity.encoding": "reconcile_encoding + encoding regime-parity (O11) + emit",
    "identity.representation": "resolve_amp_dtype + IdentityConfig runtime consistency guard (F1) + O11 + emit",
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
    "eval.ladder.bootstrap_resamples": "pipeline.py RoundSpec.ladder_bootstrap_resamples -> worker.py aggregate_rung",
    "eval.ladder.bootstrap_ci_level": "pipeline.py RoundSpec.ladder_bootstrap_ci_level -> worker.py aggregate_rung",
    "eval.ladder.bt_prior_games": "bt.py fit_bt prior",
    "eval.ladder.bootstrap_seed": "pipeline.py RoundSpec.ladder_bootstrap_seed -> worker.py aggregate_rung",
    # ── train.* (SC-A1; TrainHParams read sites, DESIGN_P2.md §1.1) ──────────────────────
    "train.lr": "TrainHParams.lr -> core.py:190 AdamW ctor; resume-owned (orchestrator.py:29)",
    "train.weight_decay": "TrainHParams.weight_decay -> core.py:189 AdamW ctor",
    "train.grad_clip": "TrainHParams.grad_clip -> core.py:409,476 fp16_backward_step max_grad_norm",
    "train.fp16": "TrainHParams.fp16 -> core.py:176-183 CUDA-only scaler/autocast gate",
    "train.device":
        "torch.device(config.train.device) in mantis.run.build_run_collaborators ->"
        " init_trainer(device=…) AND WorkerPool(device=…) (R126; the retired --device flag"
        " on both callers)",
    "train.amp_dtype": "TrainConfig.amp_dtype -> grid-path AMP dtype (schema now, model/amp.py wiring SC-B3, R30b)",
    "train.lr_schedule": "TrainHParams.lr_schedule -> core.py:222-235 _build_scheduler",
    "train.total_steps": "TrainHParams.total_steps -> core.py:227 _build_scheduler T_max fallback",
    "train.scheduler_t_max": "TrainHParams.scheduler_t_max -> core.py:227 _build_scheduler T_max",
    "train.eta_min": "TrainHParams.eta_min -> core.py:230 _build_scheduler eta_min",
    "train.min_lr": "TrainHParams.min_lr -> core.py:230 _build_scheduler eta_min fallback",
    "train.checkpoint_interval": "TrainHParams.checkpoint_interval ->"
                                " Trainer._maybe_periodic_checkpoint (THE one reader; both the dense"
                                " and the graph step tail call it — R173/CARD-CS2)",
    "train.actor_sync_cadence_steps": "resolve_actor_sync_cadence -> compose_run -> ActorSync.maybe_sync (WP-UNFREEZE K1)",
    "train.max_train_steps":
        "resolve_max_train_steps -> compose_run -> StepCoordinatorConfig.stop_step",
    # WPAX Phase D (R65/R80) registered the BLOCK as one leaf; WPMINT DR-6 (R93) makes the
    # walker descend through OPTIONAL blocks — the cause of the old blindness was
    # optionality, not nesting — so the three inner keys are registered individually now,
    # each at the call site that reads it.
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
    "train.completed_q_values": "TrainHParams.completed_q_values -> core.py:347 CE-vs-KL loss switch; cross-validated",
    "train.value_target": "TrainHParams.from_config single-variant assertion (T-D)",
    "train.policy_target": "RunConfig cross-section validator (policy_target/completed_q_values consistency, §2)",
    "train.draw_reward": "SelfPlayHParams.from_config reads config['train']['draw_reward'] (cross-section)",
    "train.ply_cap_value": "SelfPlayHParams.from_config reads config['train']['ply_cap_value'] (cross-section)",
    "train.policy_prune_frac": "TrainHParams.policy_prune_frac -> core.py:305 _prune_policy_targets gate",
    "train.entropy_reg_weight": "TrainHParams.entropy_reg_weight -> core.py:289,362-365 / losses.py:285-286",
    "train.aux_opp_reply_weight": "TrainHParams.aux_opp_reply_weight -> core.py:283,313-314,320,334,358-361",
    "train.uncertainty_weight": "TrainHParams.uncertainty_weight -> core.py:284,314,321,336,366-369",
    "train.ownership_weight": "TrainHParams.ownership_weight -> core.py:285,315,338,371-374",
    "train.threat_weight": "TrainHParams.threat_weight -> core.py:286,316,340,375-379",
    "train.aux_chain_weight": "TrainHParams.aux_chain_weight -> core.py:287,322,380-385",
    "train.ply_index_weight": "TrainHParams.ply_index_weight -> core.py:288,323,343-344,386-389",
    "train.threat_pos_weight": "TrainHParams.threat_pos_weight -> core.py:211-214 _threat_pos_weight tensor",
    # ── selfplay.* scalars (SC-A2; SelfPlayHParams read sites, DESIGN_P2.md §1.2) ────────
    "selfplay.n_workers": "SelfPlayHParams.n_workers -> pool worker count",
    "selfplay.leaf_batch_size": "SelfPlayHParams.leaf_batch_size -> runner leaf_batch_size ctor kwarg",
    "selfplay.max_game_moves": "SelfPlayHParams.max_moves_per_game -> runner max_moves_per_game ctor kwarg",
    "selfplay.inference_pool_size": "SelfPlayHParams.inference_pool_size -> runner inference_pool_size ctor kwarg",
    "selfplay.completed_q_values": "SelfPlayHParams.completed_q_values -> runner ctor kwarg; cross-validated",
    "selfplay.c_visit": "SelfPlayHParams.c_visit -> runner c_visit ctor kwarg",
    "selfplay.c_scale": "SelfPlayHParams.c_scale -> runner c_scale ctor kwarg",
    "selfplay.gumbel_mcts": "SelfPlayHParams.gumbel_mcts -> runner gumbel_mcts ctor kwarg (R23)",
    "selfplay.gumbel_m": "SelfPlayHParams.gumbel_m -> runner gumbel_m ctor kwarg (R23)",
    "selfplay.gumbel_explore_moves": "SelfPlayHParams.gumbel_explore_moves -> runner ctor kwarg (R23)",
    "selfplay.results_queue_cap": "SelfPlayHParams.results_queue_cap -> runner results_queue_cap ctor kwarg",
    "selfplay.random_opening_plies": "SelfPlayHParams.random_opening_plies -> runner ctor kwarg",
    "selfplay.rotation_enabled": "SelfPlayHParams.rotation_enabled -> runner selfplay_rotation_enabled ctor kwarg",
    "selfplay.forced_win_policy_enabled": "SelfPlayHParams.forced_win_policy_enabled -> runner post-ctor attr",
    "selfplay.forced_win_policy_depth": "SelfPlayHParams.forced_win_policy_depth -> runner post-ctor attr",
    "selfplay.forced_win_policy_weight": "SelfPlayHParams.forced_win_policy_weight -> runner post-ctor attr",
    "selfplay.solver_enabled": "SelfPlayHParams.solver_enabled -> runner post-ctor attr",
    "selfplay.solver_depth": "SelfPlayHParams.solver_depth -> runner post-ctor attr",
    "selfplay.solver_node_budget": "SelfPlayHParams.solver_node_budget -> runner post-ctor attr",
    "selfplay.solver_neighbor_dist": "SelfPlayHParams.solver_neighbor_dist -> runner post-ctor attr",
    "selfplay.solver_visit_weight": "SelfPlayHParams.solver_visit_weight -> runner post-ctor attr",
    "selfplay.seed_fraction": "SelfPlayHParams.seed_fraction -> runner post-ctor attr + _load_seed_corpus gate",
    "selfplay.seed_corpus_path": "SelfPlayHParams.seed_corpus_path -> _load_seed_corpus path arg",
    "selfplay.log_investigation_metrics": "SelfPlayHParams.log_investigation_metrics -> investigation-metrics gate",
    "selfplay.instrumentation_enabled": "SelfPlayHParams.instrumentation_enabled -> instrumentation gate",
    # ── selfplay.mcts.* (8) ───────────────────────────────────────────────────────────────
    "selfplay.mcts.n_simulations": "SelfPlayHParams.n_simulations -> runner n_simulations ctor kwarg",
    "selfplay.mcts.c_puct": "SelfPlayHParams.c_puct -> runner c_puct ctor kwarg",
    "selfplay.mcts.fpu_reduction": "SelfPlayHParams.fpu_reduction -> runner fpu_reduction ctor kwarg",
    "selfplay.mcts.quiescence_enabled": "SelfPlayHParams.quiescence_enabled -> runner ctor kwarg",
    "selfplay.mcts.quiescence_blend_2": "SelfPlayHParams.quiescence_blend_2 -> runner ctor kwarg",
    "selfplay.mcts.dirichlet_alpha": "SelfPlayHParams.dirichlet_alpha -> runner ctor kwarg",
    "selfplay.mcts.dirichlet_epsilon": "SelfPlayHParams.dirichlet_epsilon -> runner ctor kwarg (mcts.epsilon key pin)",
    "selfplay.mcts.dirichlet_enabled": "SelfPlayHParams.dirichlet_enabled -> runner ctor kwarg",
    # ── selfplay.playout_cap.* (11) ───────────────────────────────────────────────────────
    "selfplay.playout_cap.fast_sims": "SelfPlayHParams.fast_sims -> runner fast_sims ctor kwarg (REQUIRED)",
    "selfplay.playout_cap.fast_prob": "SelfPlayHParams.fast_prob -> runner fast_prob ctor kwarg; mutual-exclusion",
    "selfplay.playout_cap.standard_sims": "SelfPlayHParams.standard_sims -> runner standard_sims ctor kwarg",
    "selfplay.playout_cap.full_search_prob": "SelfPlayHParams.full_search_prob -> runner ctor kwarg; mutual-exclusion",
    "selfplay.playout_cap.n_sims_quick": "SelfPlayHParams.n_sims_quick -> runner n_sims_quick ctor kwarg",
    "selfplay.playout_cap.n_sims_full": "SelfPlayHParams.n_sims_full -> runner n_sims_full ctor kwarg",
    "selfplay.playout_cap.zoi_enabled": "SelfPlayHParams.zoi_enabled -> runner zoi_enabled ctor kwarg",
    "selfplay.playout_cap.zoi_lookback": "SelfPlayHParams.zoi_lookback -> runner zoi_lookback ctor kwarg",
    "selfplay.playout_cap.zoi_margin": "SelfPlayHParams.zoi_margin -> runner zoi_margin ctor kwarg",
    "selfplay.playout_cap.temperature_threshold_compound_moves": "SelfPlayHParams.temp_threshold_compound_moves -> runner ctor kwarg",
    "selfplay.playout_cap.temp_min": "SelfPlayHParams.temp_min -> runner temp_min ctor kwarg",
    # ── inference.* (8; InferenceHParams read sites) ─────────────────────────────────────
    "inference.inference_batch_size": "InferenceHParams.inference_batch_size -> inference_server.py:74 ctor",
    "inference.inference_max_wait_ms": "InferenceHParams.inference_max_wait_ms -> inference_server.py:74 ctor",
    "inference.trace_inference": "InferenceHParams.trace_inference -> inference_server.py:74 ctor",
    "inference.compile_inference": "InferenceHParams.compile_inference -> inference_server.py:74 ctor",
    "inference.compile_inference_mode": "InferenceHParams.compile_inference_mode -> inference_server.py:74 ctor",
    "inference.compile_inference_dynamic": "InferenceHParams.compile_inference_dynamic -> inference_server.py:74 ctor",
    "inference.perf_timing": "InferenceHParams.perf_timing -> inference_server.py:74 ctor (diagnostics ns)",
    "inference.perf_sync_cuda": "InferenceHParams.perf_sync_cuda -> inference_server.py:74 ctor (diagnostics ns)",
    # F-816-10 (R276(f)): the GRAPH inference forward's memory bound — this copy states
    # the chain independently of its twin (two independently-maintained registries by
    # design). Route-scoped: the resolve happens inside the graph branch of the ctor.
    "inference.fused_graph_caps.max_fused_edges":
        "FusedGraphCapsSpec.max_fused_edges (resolve_fused_graph_caps) -> the graph arm of"
        " InferenceServer.__init__ -> _run_graph_loop plan_fused_forwards edge bound; and"
        " RoundSpec.fused_graph_caps -> LocalInferenceEngine in the eval subprocess",
    "inference.fused_graph_caps.max_fused_nodes":
        "FusedGraphCapsSpec.max_fused_nodes (resolve_fused_graph_caps) -> the graph arm of"
        " InferenceServer.__init__ -> _run_graph_loop plan_fused_forwards node bound; and"
        " RoundSpec.fused_graph_caps -> LocalInferenceEngine in the eval subprocess",
    # ── monitor.* (27; resolve_monitor_config -> MonitorConfig, DESIGN_P2.md §4.2) ───────
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
    "monitor.supervisor_stale_after_sec": "resolve_monitor_config -> monitor/supervise.py staleness flag",
    "monitor.supervisor_poll_interval_sec": "resolve_monitor_config -> monitor/supervise.py poll cadence",
    "monitor.supervisor_kill_grace_sec": "resolve_monitor_config -> monitor/supervise.py kill grace",
    "monitor.supervisor_max_relaunches": "resolve_monitor_config -> monitor/supervise.py relaunch cap",
    # ── monitor.drain.* (4; DrainCapsConfig, DESIGN_P2.md §4.3) ──────────────────────────
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
    """The nested block an annotation names, seen THROUGH `Optional` / `Block | None`
    (WPMINT DR-6, R93). NIT-3 is unchanged: `list[SubModel]` has a `list` origin, not a
    union one, and stays ONE leaf."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    if get_origin(ann) in (Union, UnionType):
        blocks = [arm for arm in get_args(ann)
                  if isinstance(arm, type) and issubclass(arm, BaseModel)]
        if len(blocks) == 1:
            return blocks[0]
    return None


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
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
    # 143 post-SC-A3 + 3 WP-UNFREEZE knobs (K1/K2/K3) + 1 WPAX S-4 knob
    # (train.max_train_steps, the run-length authority) = 147.
    # WPAX Phase D adds 1 (train.draw_rate_abort, registered as ONE opaque block leaf):
    # 147 + 1 = 148. WPMINT DR-6 (R93) retires that block leaf and surfaces its three inner
    # keys: 148 - 1 + 3 = 150. `train.draw_rate_abort` is the ONLY `Block | None` field in
    # `RunConfig`, so those three are the whole blast radius. This count and the sibling
    # copy's must stay equal.
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
    assert len(CONSUMER_REGISTRY) == 184
    assert len(_leaf_paths(RunConfig)) == 184


def test_bijection_bites_on_a_real_schema_mutation():
    class _MutatedRunConfig(RunConfig):
        phantom_leaf: int

    leaves = set(_leaf_paths(_MutatedRunConfig))
    assert "phantom_leaf" in leaves
    assert leaves != set(CONSUMER_REGISTRY), "bijection must break when a new leaf is unregistered"
