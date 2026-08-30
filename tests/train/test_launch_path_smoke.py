"""O-SMOKE — end-to-end launch-path smoke (WP10 Slice-2 gate, INTEGRATION tier).

A minimal end-to-end launch of `run_training_loop` on a CPU synthetic config: build the trainer
via `build_net(arch)`, run ≈2 steps, write an envelope-v2 checkpoint, resume from it, and shut
down clean on a simulated signal. Bites the F-10 class — a launch-path wiring break unit tests
miss. Tier ruling (DISPATCHER CORRECTION of rev-1): the launch-path smoke homes in the
INTEGRATION tier (operator brief + repo_design §8: "integration … includes at least one launch-
path smoke"); carries `@pytest.mark.integration`, reached via `make test.integration`. Kept
minimal (N≈2 steps, CPU synthetic, tiny net). IMPL-authored gating oracle (non-⊕⊕).
"""
from __future__ import annotations

import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.encoding import lookup
from mantis.model import CnnArch, arch_from_spec_and_config, build_net
from mantis.train.checkpoints import CHECKPOINT_SCHEMA_VERSION, resume_trainer
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers
from mantis.train.loop import run_training_loop
from mantis.train.trainer.core import Trainer

pytestmark = pytest.mark.integration

ENCODING = "v6_live2_ls"


def _synthetic_batch(spec):
    b = 3
    planes, hw = int(spec.n_planes), int(spec.board_size)
    n_actions = hw * hw + 1
    states = np.zeros((b, planes, hw, hw), dtype=np.float32)
    policies = np.full((b, n_actions), 1.0 / n_actions, dtype=np.float32)
    outcomes = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    return states, policies, outcomes


def _eval_block():
    # WP11-A schema extension: eval.gate/eval.ladder are now required (design §c.1).
    return {
        "random_model_sims": 1, "sealbot_model_sims": 1, "kraken_model_sims": 1,
        "strix_model_sims": 1, "random_floor_games": 0, "worker_device": "cpu",
        "round_timeout_sec": 1.0, "worker_kill_grace_sec": 1.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 1, "confirm_games": 1, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 1, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1, "min_distinct_per_pair": 1, "seed_base": 1,
        },
        "ladder": {
            "rungs": [{"name": "r0", "bot": "random", "variant": "raw", "depth": None,
                      "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
                      "deploy_matched": True, "games_max": 1}],
            "round_games": 1, "min_games_per_active_rung": 1, "graduation_wr_lower_ci": 0.9,
            "graduation_consec_rounds": 1, "activation_wr_lower_ci": 0.5,
            "calibration_every_k_rounds": 1, "calibration_games": 1,
            "bootstrap_resamples": 1, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1,
        },
    }


#: WPMINT Phase K-A stage 0: the complete `train:` payload, DERIVED from a MINTED config
#: rather than restated — eleven files carried a hand-written copy, so a new `train.*` key
#: cost eleven edits. `dev_example.yaml`'s resolved block was measured byte-identical to this
#: file's census except for `fp16`, which this smoke path pins itself below.
_MINTED_TRAIN: dict = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


#: Every config this file builds is a GRID config, and it says so once. The block builders read
#: it so the arch-scoped blocks are dropped AT SOURCE (R322(d)): `train:` here is derived from a
#: GRAPH config's dump, so it arrives carrying `microbatch_caps`, which a grid `RunConfig`
#: refuses. Driven from `ARCH_SCOPED_KEYS` — the schema's own partition — rather than by name.
_REPRESENTATION = "grid"


def _drop_foreign_arch_keys(section: str, block: dict) -> dict:
    """`block` without the arch-scoped keys `_REPRESENTATION` does not have."""
    for key in ARCH_SCOPED_KEYS:
        if key.section == section and key.arch != _REPRESENTATION:
            block.pop(key.field, None)
    return block


def _train_block():
    # WPSC Phase 2 SC-A1: `train:` is now a required RunConfig section (DESIGN_P2.md §2).
    # `fp16=False` is this file's own delta: the smoke path runs on CPU.
    return _drop_foreign_arch_keys("train", dict(_MINTED_TRAIN, fp16=False))


def _selfplay_block():
    # WPSC Phase 2 SC-A2: `selfplay:` is now the expanded nested shape (DESIGN_P2.md §3);
    # `legal_move_radius_schedule` is gone (DESIGN_P2.md §5).
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "inference_pool_size": None, "completed_q_values": False, "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_mcts": False, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0, "rotation_enabled": True,
        "forced_win_policy_enabled": False, "forced_win_policy_depth": 2,
        "forced_win_policy_weight": 1.0, "solver_enabled": False, "solver_depth": 16,
        "solver_node_budget": 50_000, "solver_neighbor_dist": 2, "solver_visit_weight": 0.3,
        "seed_fraction": 0.0, "seed_corpus_path": None, "log_investigation_metrics": True,
        "instrumentation_enabled": False,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "zoi_enabled": False, "zoi_lookback": 16, "zoi_margin": 5,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _inference_block():
    return _drop_foreign_arch_keys("inference", {
        "inference_batch_size": 64, "inference_max_wait_ms": 10, "trace_inference": True,
        "compile_inference": False, "compile_inference_mode": "default",
        "compile_inference_dynamic": True, "perf_timing": False, "perf_sync_cuda": False,
        # `fused_graph_caps` is ARCH-SCOPED to graph (R322(d)) and this is a GRID config, so it
        # is stripped by the helper above. Left in the literal so the strip is visible here.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    })


def _monitor_block():
    return {
        # R242 (ADJ-D12): the ARMING cadence, schema-only and required.
        "gate_interval": 1000,
        "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
        "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
        "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
        "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
        "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
        "axis_warn": 0.45, "axis_alert": 0.50,
        "heartbeat_deadline_train_step_sec": 1800.0,
        "heartbeat_deadline_inference_dispatch_sec": 1800.0,
        "heartbeat_deadline_selfplay_drain_sec": 1800.0,
        "heartbeat_deadline_eval_round_sec": 1800.0,
        "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
        "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
        "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
        "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
        "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
        "drain": {
            "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
            "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
        },
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }


def _config():
    return {
        "schema_version": 1, "run_id": "smoke", "seed": 20260722,
        "eval_enabled": True,
        # RECAL-PREP (R308(g)(i)): a REQUIRED top-level leaf. `null` is R119's
        # placeholder — refused at boot on a cuda process, valued only by the
        # re-calibration sitting under R282(b).
        "allocator_posture": None,
        "identity": {"encoding": ENCODING, "representation": _REPRESENTATION},
        "eval": _eval_block(),
        "train": _train_block(),
        "selfplay": _selfplay_block(),
        "inference": _inference_block(),
        "monitor": _monitor_block(),
    }


def test_launch_path_smoke(tmp_path, full_train_hparams):
    """Build → run ≈2 steps → write envelope-v2 ckpt → resume → clean shutdown on a signal."""
    spec = lookup(ENCODING)
    # a tiny CNN (filters=16, res_blocks=1) — CPU-cheap, real build_net(arch) net.
    arch = CnnArch(board_size=int(spec.board_size), in_channels=int(spec.n_planes),
                   filters=16, res_blocks=1)
    net = build_net(arch)
    config = _config()
    # Default cosine schedule (TrainConfig-minted "cosine") so the saved envelope carries a
    # scheduler_state and the resumed Trainer (which re-defaults to cosine) restores it.
    hp = full_train_hparams(fp16=False, checkpoint_interval=0)
    tr = Trainer(net, config, arch=arch, checkpoint_dir=tmp_path, train_hparams=hp)

    states, policies, outcomes = _synthetic_batch(spec)

    # ── run ≈2 steps through run_training_loop, then request save-then-exit ──────────────
    state = ShutdownState()
    seen = {"n": 0}

    def one_step():
        tr.train_step_from_tensors(states, policies, outcomes)
        seen["n"] += 1
        if seen["n"] >= 2:
            state.shutdown_save = True  # request the final save; the loop observes it

    run_training_loop(trainer=tr, shutdown_state=state, step_fn=one_step, max_steps=10)
    assert tr.step == 2, "the loop must have driven exactly 2 training steps"

    # ── the loop wrote a FINAL envelope-v2 checkpoint on shutdown_save ──────────────────
    ckpts = list(tmp_path.glob("*.ckpt"))
    assert ckpts, "run_training_loop must write an envelope-v2 checkpoint on shutdown_save"
    ckpt = ckpts[0]
    assert ckpt.name.startswith("smoke_00000002_"), f"unexpected v2 filename {ckpt.name}"
    payload = torch.load(ckpt, weights_only=True)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 2
    assert payload["kind"] == "full"

    # ── resume from it (build_net(metadata.arch) + restore optim/scaler/step) ───────────
    tr2 = resume_trainer(Trainer, ckpt, fallback_config=config)
    assert tr2.loaded_from_full_checkpoint is True
    assert tr2.step == 2
    assert len(tr2.optimizer.param_groups) == 2

    # ── clean shutdown on a SIMULATED signal (save-then-exit choreography) ──────────────
    orig_int = signal.getsignal(signal.SIGINT)
    try:
        state2 = ShutdownState()
        install_signal_handlers(state2)
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)  # simulate one SIGINT
        assert state2.shutdown_save is True and state2.running is False
        # a 0-step loop over the shutdown-flagged state saves once and returns clean.
        final = run_training_loop(trainer=tr2, shutdown_state=state2)
        assert final.shutdown_save is True
        assert len(list(tmp_path.glob("smoke_00000002_*.ckpt"))) >= 1
    finally:
        signal.signal(signal.SIGINT, orig_int)
