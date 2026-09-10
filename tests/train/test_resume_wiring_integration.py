"""`resume_trainer` actually CALLS `apply_config_overrides_f1` / `resolve_lr_provenance`
end-to-end.

`RunConfig` is `extra="forbid"` at every level, so a real baked config cannot carry bare flat
legacy keys; this suite therefore treats `"train"`, the whole nested section, as the
declared/base-inherited unit — the only key shape that is both a real top-level `RunConfig`
key and round-trips through the flat dict-key comparison unmodified. The lr-specific loud
warning is pinned assuming the baked lr is read NESTED as `baked_config["train"]["lr"]` while
the declared lr comes from a bare flat `"lr"` override; a flat-only baked read would make the
warning permanently unreachable.
"""
from __future__ import annotations

from pathlib import Path

from mantis.config.schema import ARCH_SCOPED_KEYS
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import build_net
from mantis.train.checkpoints import resume_trainer, save_checkpoint
from mantis.train.trainer.core import Trainer

ENCODING = "gnn_axis_v1"

_LADDER_RUNGS = [
    {"name": "sealbot_d5", "bot": "sealbot", "variant": "d5", "depth": 5,
     "opponent_sims": None, "opening_book": "book_v1_s20260625_p4",
     "deploy_matched": True, "games_max": 32},
]


def _eval_block() -> dict:
    return {
        "random_model_sims": 1, "sealbot_model_sims": 1, "random_floor_games": 0, "worker_device": "cpu",
        "round_timeout_sec": 1.0, "worker_kill_grace_sec": 1.0,
        "ply_cap_adjudication": None, "strength_floor": None,
        "gate": {
            "stride": 1, "screen_games": 1, "confirm_games": 1, "promotion_winrate": 0.55,
            "screen_confirm_lo": 0.44, "deploy_sims": 1, "opening_book": "book_v1_s20260625_p4",
            "bootstrap_resamples": 1, "min_distinct_per_pair": 1, "seed_base": 1,
        },
        "ladder": {
            "rungs": [dict(r) for r in _LADDER_RUNGS], "round_games": 1,
            "min_games_per_active_rung": 1, "graduation_wr_lower_ci": 0.9,
            "graduation_consec_rounds": 1, "activation_wr_lower_ci": 0.5,
            "calibration_every_k_rounds": 1, "calibration_games": 1,
            "bootstrap_resamples": 1, "bootstrap_ci_level": 0.95,
            "bt_prior_games": 1.0, "bootstrap_seed": 1,
        },
    }


#: The complete `train:` payload, DERIVED from a MINTED config rather than restated: eleven
#: files carried a hand-written copy, so a new `train.*` key cost eleven edits. `lr_schedule`
#: is the one leaf this file pins itself below.
_MINTED_TRAIN: dict = load_config(
    Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train.model_dump()


#: Every config this file builds is a GRAPH config, and it says so once. The block builders read
#: it so arch-scoped blocks are dropped AT SOURCE: these oracles compare an OVERRIDE block
#: against the BAKED one, so a builder emitting a key the assembled config strips would make the
#: comparison measure this file's own inconsistency.
_REPRESENTATION = "graph"


def _drop_foreign_arch_keys(section: str, block: dict) -> dict:
    """`block` without the arch-scoped keys `_REPRESENTATION` does not have."""
    for key in ARCH_SCOPED_KEYS:
        if key.section == section and key.arch != _REPRESENTATION:
            block.pop(key.field, None)
    return block


def _train_block(*, lr: float = 1e-3) -> dict:
    # This file's own delta, `lr_schedule="none"`: the resume oracles compare optimizer/LR state
    # across a save->load, and a live schedule would move the number they compare.
    return _drop_foreign_arch_keys(
        "train", dict(_MINTED_TRAIN, lr=lr, lr_schedule="none")
    )


def _selfplay_block() -> dict:
    return {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }


def _inference_block() -> dict:
    return _drop_foreign_arch_keys("inference", {
        "inference_batch_size": 64, "inference_max_wait_ms": 10,
        # `fused_graph_caps` is ARCH-SCOPED, so the helper above strips it when it does not
        # belong to `_REPRESENTATION`. Left in the literal so the strip is visible here.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    })


def _monitor_block() -> dict:
    return {
        # The ARMING cadence, schema-only and required.
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
        "drain": {"final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
                 "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0},
        "disk_guard": {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0},
    }


def _full_config(*, lr: float = 1e-3) -> dict:
    config = {
        "schema_version": 1, "run_id": "resume_wiring", "seed": 20260725,
        "eval_enabled": True,
        # A REQUIRED top-level leaf; `null` is the placeholder, refused at boot on a cuda
        # process and valued only by the re-calibration.
        "allocator_posture": None,
        "identity": {"encoding": ENCODING, "representation": "graph"},
        "eval": _eval_block(),
        "train": _train_block(lr=lr),
        "search": {"kind": "puct"},
        "selfplay": _selfplay_block(),
        "inference": _inference_block(),
        "monitor": _monitor_block(),
    }
    assert config["identity"]["representation"] == _REPRESENTATION
    return config


def _save(tmp_path: Path, *, lr: float, tiny_net, optim_scaler_sched, metadata_kwargs) -> Path:
    opt, scaler, sched = optim_scaler_sched
    return save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=100,
        config=_full_config(lr=lr), metadata_kwargs=metadata_kwargs,
        checkpoint_dir=tmp_path, kind="full",
    )


def test_declared_train_section_wins_and_reaches_resumed_trainer_hp(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs
):
    path = _save(tmp_path, lr=1e-3, tiny_net=tiny_net, optim_scaler_sched=optim_scaler_sched,
                metadata_kwargs=metadata_kwargs)
    overrides = {"train": _train_block(lr=5e-4)}
    tr = resume_trainer(Trainer, path, config_overrides=overrides,
                        declared_keys=frozenset({"train"}))
    assert tr.hp.lr == 5e-4, "a DECLARED train section must WIN over the baked one (E0)"
    assert tr.f1_deferred_keys == frozenset()


def test_base_inherited_train_section_defers_to_baked_and_emits_deferred_event(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs, spy_sink
):
    path = _save(tmp_path, lr=1e-3, tiny_net=tiny_net, optim_scaler_sched=optim_scaler_sched,
                metadata_kwargs=metadata_kwargs)
    overrides = {"train": _train_block(lr=9e-4)}  # differs from baked, NOT declared
    tr = resume_trainer(Trainer, path, config_overrides=overrides,
                        declared_keys=frozenset(), sink=spy_sink)
    assert tr.hp.lr == 1e-3, "a base-inherited (non-declared) key must DEFER to baked"
    assert tr.f1_deferred_keys == frozenset({"train"})
    events = spy_sink.named("resume_base_default_deferred_to_baked")
    assert events and events[-1]["knob"] == "train"


def test_matching_base_inherited_train_section_does_not_defer(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs, spy_sink
):
    path = _save(tmp_path, lr=1e-3, tiny_net=tiny_net, optim_scaler_sched=optim_scaler_sched,
                metadata_kwargs=metadata_kwargs)
    overrides = {"train": _train_block(lr=1e-3)}  # identical to baked
    tr = resume_trainer(Trainer, path, config_overrides=overrides,
                        declared_keys=frozenset(), sink=spy_sink)
    assert tr.f1_deferred_keys == frozenset()
    assert not spy_sink.named("resume_base_default_deferred_to_baked")


def test_declared_lr_ignored_on_full_resume_emits_loud_warning(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs, spy_sink
):
    path = _save(tmp_path, lr=1e-3, tiny_net=tiny_net, optim_scaler_sched=optim_scaler_sched,
                metadata_kwargs=metadata_kwargs)
    tr = resume_trainer(Trainer, path, config_overrides={"lr": 5e-4},
                        declared_keys=frozenset({"lr"}), sink=spy_sink)
    assert tr.hp.lr == 1e-3, "lr is resume-state-owned: a bare declared lr must be IGNORED"
    events = spy_sink.named("resume_lr_override_ignored")
    assert events, "resume_trainer must emit the loud lr-ignored warning"
    ev = events[-1]
    assert ev["declared"] == 5e-4
    assert ev["baked"] == 1e-3
    assert ev["effective"] == 1e-3


def test_no_config_overrides_leaves_baked_train_section_untouched(
    tmp_path, tiny_net, optim_scaler_sched, metadata_kwargs
):
    path = _save(tmp_path, lr=1e-3, tiny_net=tiny_net, optim_scaler_sched=optim_scaler_sched,
                metadata_kwargs=metadata_kwargs)
    tr = resume_trainer(Trainer, path)
    assert tr.hp.lr == 1e-3
    assert tr.f1_deferred_keys == frozenset()


def test_resume_trainer_docstring_no_longer_asserts_unimplemented_f1_e0_semantics():
    # A light introspection oracle: the real behavior is pinned by the five tests above. The
    # docstring must describe the F1(A)/E0 override rule and must not hedge it as aspirational,
    # which it once did while neither rule was wired.
    doc = (resume_trainer.__doc__ or "").lower()
    assert "f1(a)" in doc or "e0" in doc, (
        "the docstring must still describe the F1(A)/E0 override-application rule"
    )
    for hedge in ("todo", "not yet", "future work", "not implemented", "no-op"):
        assert hedge not in doc, f"docstring still hedges the F1(A)/E0 claim with {hedge!r}"
