"""O-F1E0 — resume-precedence semantics as an integrated round-trip (WP10 Slice-2 gate).

The F1/E0/scheduler/lr resume gate SEMANTICS composed end-to-end through
`orchestrator.build_resume_config_overrides → checkpoints.resume_trainer` (T-CK-14..20 pin the
units; this exercises the whole path). The OLD code IS the spec — the literal expectations are
pinned in `tests/fixtures/train/resume_goldens.json` (the #C3 dispatcher capture). IMPL-authored
gating oracle (non-⊕⊕); PREREG "Non-⊕⊕ oracle verdicts".
"""
from __future__ import annotations

import pytest

from mantis.train.checkpoints import (
    apply_config_overrides_f1,
    resolve_lr_provenance,
    resume_trainer,
    save_checkpoint,
)
from mantis.train.orchestrator import (
    RESUME_CHECKPOINT_OWNED_KEYS,
    build_resume_config_overrides,
)
from mantis.train.trainer.core import Trainer


def _overrides(result):
    return dict(getattr(result, "overrides", result))


# ── declared-wins / base-defers (composed through the reconciler) ────────────────────────
def test_declared_wins_base_defers(resume_goldens, spy_sink):
    """O-F1E0(a) — a DECLARED key wins over the checkpoint-baked value; a base-INHERITED key
    defers to baked + warns; a declared null travels (F1(A)/E0/B3)."""
    g = resume_goldens["T-CK-16_declared_wins_base_defers"]
    inp, exp = g["inputs"], g["expected_output"]
    resolved, deferred = apply_config_overrides_f1(
        dict(inp["baked_config"]), dict(inp["config_overrides"]),
        set(inp["declared_keys"]), sink=spy_sink,
    )
    assert resolved == exp["resolved_config"]
    assert set(deferred) == set(exp["deferred_keys"])
    assert spy_sink.has(exp["warning_event"])


# ── frozen-owned keys never enter the launch overrides ───────────────────────────────────
def test_frozen_owned_keys_excluded_from_overrides():
    """O-F1E0(b) — the launch config wins for non-frozen knobs, but every
    RESUME_CHECKPOINT_OWNED_KEY (encoding/arch/optim/scheduler/lr) is EXCLUDED from the
    resume overrides (the checkpoint state is authoritative)."""
    baked = {"aux_chain_weight": 0.5, "lr": 0.001, "encoding": "v6_live2_ls", "grad_clip": 1.0}
    launch = {"aux_chain_weight": 0.2, "lr": 0.002, "encoding": "v6", "grad_clip": 2.0}
    ov = _overrides(build_resume_config_overrides(baked, launch))
    assert ov.get("aux_chain_weight") == 0.2      # non-frozen → launch wins
    assert ov.get("grad_clip") == 2.0             # non-frozen → launch wins
    for frozen in RESUME_CHECKPOINT_OWNED_KEYS:
        assert frozen not in ov, f"frozen key {frozen} leaked into the resume overrides"


# ── scheduler-horizon gate ───────────────────────────────────────────────────────────────
def test_scheduler_horizon_gate(resume_goldens):
    """O-F1E0(c) — without --override-scheduler-horizon the horizon keys stay owned (excluded);
    with it, total_steps/scheduler_t_max re-enter and re-horizon the anneal."""
    g = resume_goldens["T-CK-17_scheduler_horizon_gate"]
    baked, launch = g["inputs"]["baked_config_A"], g["inputs"]["launch_variant_B"]
    exp = g["expected_output"]
    off = _overrides(build_resume_config_overrides(dict(baked), dict(launch),
                                                   override_scheduler_horizon=False))
    assert ("total_steps" in off) is exp["override_scheduler_horizon_FALSE"]["overrides_contains_total_steps"]
    on = _overrides(build_resume_config_overrides(dict(baked), dict(launch),
                                                  override_scheduler_horizon=True))
    assert on.get("total_steps") == exp["override_scheduler_horizon_TRUE"]["overrides_total_steps"]
    assert on.get("scheduler_t_max") == exp["override_scheduler_horizon_TRUE"]["overrides_scheduler_t_max"]


# ── lr resume-owned (loud on an ignored declared override) ────────────────────────────────
def test_lr_resume_owned_is_loud(resume_goldens):
    """O-F1E0(d) — a declared lr differing from the baked lr on a full resume is IGNORED
    (resume-state-owned) and flagged loud; a matching declared lr is not flagged."""
    g = resume_goldens["T-CK-20_lr_resume_owned"]["expected_output"]
    ign = g["resolve_lr_provenance_override_ignored_case"]
    assert resolve_lr_provenance(declared=ign["declared"], baked=ign["baked"],
                                 effective=ign["effective"]).override_ignored is True
    assert resolve_lr_provenance(declared=0.001, baked=0.001, effective=0.001).override_ignored is False


# ── the full round-trip: save → resume restores optim/scaler/step (frozen-owned honored) ──
def test_full_save_resume_roundtrip_restores_state(tmp_path, tiny_net, optim_scaler_sched,
                                                    valid_config, metadata_kwargs, resume_goldens):
    """O-F1E0(e) — a full save → resume through resume_trainer restores optimizer/scaler/step
    (the frozen optimizer/scheduler state is authoritative), composing the whole path."""
    opt, scaler, sched = optim_scaler_sched
    path = save_checkpoint(model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched,
                           step=750, config=valid_config, metadata_kwargs=metadata_kwargs,
                           checkpoint_dir=tmp_path, kind="full")
    exp = resume_goldens["T-CK-19_full_vs_weights_restore"]["expected_output"]["full_resume"]
    tr = resume_trainer(Trainer, path, fallback_config=valid_config)
    assert tr.loaded_from_full_checkpoint is exp["loaded_from_full_checkpoint"]
    assert tr.step == exp["resumed_step"]
    assert len(tr.optimizer.param_groups) == exp["optimizer_param_groups_restored"]
    assert tr.scaler is not None
