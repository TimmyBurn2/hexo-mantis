"""A resumed Trainer's carried config is RunConfig-PURE.

The first --resume-from burn died at its first periodic-checkpoint boundary:
`build_resume_config_overrides` unconditionally injected the LEGACY `torch_compile` knob, the
merge wrote it into the carried config, and the one writer's write-time validation correctly
REJECTED the save, so LAW-14 re-raised and the loop died. Fresh runs were immune, so no
pre-existing oracle saw it. The carried config now holds EXACTLY the schema keys, derived from
`RunConfig.model_fields`; the strip is scoped to the resume machinery's own directives — never
widening into a general unknown-key launder, never naming a real schema key — while a directive
the caller asked for still travels in the OVERRIDES.
"""
from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from mantis.config.schema import RunConfig
from mantis.train.checkpoints import (
    RESUME_DIRECTIVE_KEYS,
    apply_config_overrides_f1,
    resume_trainer,
    save_checkpoint,
)
from mantis.train.orchestrator import build_resume_config_overrides, init_trainer
from mantis.train.trainer.core import Trainer


def _resume_cfg(mk_config, *, checkpoint_interval: int = 25) -> dict:
    """A complete schema-valid nested config with the periodic cadence armed."""
    cfg = mk_config()
    cfg["train"]["checkpoint_interval"] = checkpoint_interval
    return cfg


def _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch, *, step: int = 33):
    opt, scaler, sched = optim_scaler_sched
    return save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=step,
        config=cfg, metadata_kwargs=mk_meta(tiny_arch), checkpoint_dir=tmp_path, kind="full",
    )


def test_resume_then_first_periodic_boundary_saves_and_emits(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch, spy_sink
):
    """The exact death shape: save at 33 (interval 25), resume via the production entry, reach
    boundary 50, fire the ONE periodic seam. Pre-fix `_write_v2_payload` raised on it."""
    cfg = _resume_cfg(mk_config)
    path = _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch)

    trainer = init_trainer(config=cfg, device=torch.device("cpu"),
                           checkpoint_path=str(path), sink=spy_sink)
    assert trainer.step == 33, "full resume restores the baked step"

    trainer.step = 50  # the next periodic boundary after 33 at interval 25
    out = trainer._maybe_periodic_checkpoint(None)

    assert out is not None and out.exists(), (
        "the first post-resume periodic save must WRITE — pre-F-R-P4-1-fix it raised the "
        "extra_forbidden ValidationError on the injected legacy torch_compile key"
    )
    saves = spy_sink.named("periodic_checkpoint_save")
    assert len(saves) == 1 and saves[0]["step"] == 50, (
        f"the periodic seam emits its own event once, at the boundary; got {saves}"
    )


@pytest.mark.parametrize("flags", [
    {},
    {"allow_fresh_scheduler": True},
    {"override_scheduler_horizon": True},
], ids=["plain", "allow_fresh_scheduler", "override_scheduler_horizon"])
def test_resumed_carried_config_is_exactly_the_runconfig_key_set(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch, flags
):
    """After a production-shaped resume the carried config holds EXACTLY the RunConfig top-level
    keys and re-validates, under every resume-directive flag. `allow_fresh_scheduler` witnesses
    the strip independently: it puts a real directive into the overrides."""
    cfg = _resume_cfg(mk_config)
    path = _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch)

    trainer = init_trainer(config=cfg, device=torch.device("cpu"),
                           checkpoint_path=str(path), **flags)

    assert set(trainer.config) == set(RunConfig.model_fields), (
        "carried-config key set must equal the schema's own field set (derived, never "
        f"transcribed); extra={sorted(set(trainer.config) - set(RunConfig.model_fields))} "
        f"missing={sorted(set(RunConfig.model_fields) - set(trainer.config))}"
    )
    RunConfig.model_validate(dict(trainer.config))  # values re-validate, not just key names


def test_override_builder_injects_no_directive_keys_for_a_nested_launch_config(mk_config):
    """A nested launch config yields overrides with no SPURIOUS resume-directive key — RED on a
    re-added `torch_compile`-class injection — while a flagged call still carries its directive;
    the owned-values record is the one legitimate directive (its consumer is the loud ignore)."""
    cfg = mk_config()
    plain = build_resume_config_overrides(cfg, cfg)
    assert set(plain) & RESUME_DIRECTIVE_KEYS <= {"resume_owned_launch_values"}, (
        f"spurious directive injection: {sorted(set(plain) & RESUME_DIRECTIVE_KEYS)}"
    )
    flagged = build_resume_config_overrides(cfg, cfg, allow_fresh_scheduler=True)
    assert flagged.get("allow_fresh_scheduler") is True, (
        "the legitimate directive must still travel in the overrides — the fix protects "
        "the carried config, not the mechanism"
    )


def test_legacy_explicit_torch_compile_override_never_reaches_the_carried_config(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch
):
    """A legacy caller passing flat `torch_compile` in `config_overrides` gets a Trainer whose
    carried config does NOT hold it: both loader surfaces route through `resume_trainer`."""
    cfg = _resume_cfg(mk_config)
    path = _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch)

    directives = {
        "torch_compile": False,
        "torch_compile_mode": "default",
        "total_steps": 5000,
        "scheduler_t_max": 5000,
    }  # every RESUME_DIRECTIVE_KEYS member a legacy caller can pass alongside a real resume
    trainer = resume_trainer(
        Trainer, path, fallback_config=cfg,
        config_overrides=directives, declared_keys=None,
        device=torch.device("cpu"),
    )
    for key in directives:
        assert key not in trainer.config, (
            f"directive {key!r} leaked into the carried config — removing it from "
            "RESUME_DIRECTIVE_KEYS would resurrect the F-R-P4-1 class for this key"
        )
    RunConfig.model_validate(dict(trainer.config))


def test_write_time_validation_still_raises_on_a_non_directive_unknown_key(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch
):
    """The INVERSE of the repro: a bogus key no schema owns SURVIVES the directive-scoped strip,
    is carried, and the FIRST periodic save still RAISES at the ONE writer. Widening the strip to
    `k in RunConfig.model_fields` would launder it; the conformance pin cannot see this, because
    it saves directly rather than through `resume_trainer`."""
    cfg = _resume_cfg(mk_config)
    path = _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch)

    trainer = resume_trainer(
        Trainer, path, fallback_config=cfg,
        config_overrides={"a_key_no_schema_owns": 7}, declared_keys=None,
        device=torch.device("cpu"),
    )
    assert "a_key_no_schema_owns" in trainer.config, (
        "the strip laundered a NON-directive unknown key — it has widened beyond "
        "RESUME_DIRECTIVE_KEYS and is now softening write-time validation (LAW-14)"
    )
    trainer.step = 50
    with pytest.raises(ValidationError, match="a_key_no_schema_owns"):
        trainer._maybe_periodic_checkpoint(None)


def test_mechanism_directives_force_declare_and_win_over_baked():
    """A mechanism directive in the overrides WINS over a baked flat value even undeclared, and
    nothing defers: the force-declare set derives from `RESUME_DIRECTIVE_KEYS`, and a divergent
    local set omitting `total_steps` would defer to the baked value with only a warning."""
    resolved, deferred = apply_config_overrides_f1(
        {"total_steps": 1000}, {"total_steps": 5000}, frozenset(), sink=None,
    )
    assert resolved["total_steps"] == 5000, (
        f"the mechanism directive lost the merge: resolved {resolved!r}"
    )
    assert deferred == frozenset(), (
        f"a force-declared mechanism key may never defer-to-baked; got {deferred!r}"
    )


def test_resume_directive_keys_never_name_a_runconfig_key():
    """`RESUME_DIRECTIVE_KEYS` ∩ RunConfig fields must stay EMPTY, so a future schema adopting
    one of these names fires here rather than at the next resumed run's first periodic save."""
    overlap = RESUME_DIRECTIVE_KEYS & RunConfig.model_fields.keys()
    assert not overlap, (
        f"RESUME_DIRECTIVE_KEYS overlaps the live schema: {sorted(overlap)} — the strip "
        "would silently drop a REAL config key; re-scope the directive set instead"
    )
