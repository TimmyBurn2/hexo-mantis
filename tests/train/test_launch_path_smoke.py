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

import numpy as np
import pytest
import torch

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


def _train_block():
    # WPSC Phase 2 SC-A1: `train:` is now a required RunConfig section (DESIGN_P2.md §2).
    return {
        "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "fp16": False, "amp_dtype": "fp16",
        "lr_schedule": "cosine", "total_steps": 1_000_000, "scheduler_t_max": None,
        "eta_min": 5e-4, "min_lr": None, "checkpoint_interval": 0, "completed_q_values": False,
        "value_target": "pure_outcome_z", "policy_target": "raw_visit_distribution",
        "draw_reward": -0.5, "ply_cap_value": -0.5, "policy_prune_frac": 0.0,
        "entropy_reg_weight": 0.0, "aux_opp_reply_weight": 0.0, "uncertainty_weight": 0.0,
        "ownership_weight": 0.0, "threat_weight": 0.0, "aux_chain_weight": 0.0,
        "ply_index_weight": 0.0, "threat_pos_weight": 1.0,
    }


def _config():
    return {
        "schema_version": 1, "run_id": "smoke", "seed": 20260722,
        "identity": {"encoding": ENCODING, "representation": "grid"},
        "eval": _eval_block(),
        "train": _train_block(),
        "selfplay": {"legal_move_radius_schedule": None},
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
