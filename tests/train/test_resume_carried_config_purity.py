"""⊕ F-R-P4-1 (SD-FIX F-P4) — a resumed Trainer's carried config is RunConfig-PURE.

The first --resume-from burn died at its first periodic-checkpoint boundary (step 50,
interval 25): `build_resume_config_overrides` unconditionally injected the LEGACY
`torch_compile` knob ("pre-E0 default path"), the F1 merge wrote it into the carried
config, and the ONE writer's write-time validation (`RunConfig.model_validate`, R1
`extra="forbid"`) correctly REJECTED the save; LAW-14 re-raised and the loop died
(FINDINGS_R-P4 F-R-P4-1, verbatim traceback). Fresh runs were immune — only resumes
carried the poison, so no pre-existing oracle saw it.

One invariant, five witnesses — the carried config holds EXACTLY the RunConfig schema
keys, derived from `RunConfig.model_fields` (R98: derived at point of use, never
transcribed):

- the REPRO — resume through the production entry (`init_trainer(checkpoint_path=...)`,
  the exact `run.py:477` shape: nested launch config, no `declared_keys`), step to the
  next periodic boundary, and the periodic save WRITES + EMITS. Pre-fix this raised
  `ValidationError: torch_compile … extra_forbidden` from `_write_v2_payload`.
- the PURITY oracle, parametrized over the resume-directive flags —
  `allow_fresh_scheduler=True` is the arm that keeps the boundary strip honest
  independently of the deleted injection (the flag legitimately puts a directive into
  `config_overrides`; without the strip it would poison the carried config).
- the BUILDER pin — a nested launch config yields overrides with NO directive keys
  (kills a re-added injection independently of the strip), while a directive the caller
  actually asked for still travels in the OVERRIDES (the mechanism is preserved, only
  the carried config is protected).
- the LEGACY-CALLER pin — an explicit flat `torch_compile` in `config_overrides`
  (the legacy launch shape) is consumed/ignored by the machinery, never carried.
- the DISJOINTNESS guard — `RESUME_DIRECTIVE_KEYS` may never name a real RunConfig key,
  so the strip can only ever remove the machinery's OWN directives;
- the NOT-SOFTENED pin — the strip may never widen into a general unknown-key launder: a
  bogus NON-directive key smuggled through `config_overrides` is carried and the first
  periodic save still RAISES at the one writer. This is the RESUME-path authority for
  LAW-14 staying un-softened (red-team M7 proved the conformance pin
  `test_config_snapshot_schema_validated_on_write` cannot see it — that pin saves
  directly and never routes through `resume_trainer`);
- the FORCE-DECLARE pin — a mechanism directive present in the overrides WINS over a
  baked flat value even when not operator-declared, with nothing deferred (red-team
  M11a: a divergent local mechanism set that omits a member would silently defer an
  operator's horizon override to the baked value, announced only by a warning event no
  test asserted).
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
    """A complete schema-valid nested config: fp16 off (CPU rig), periodic cadence armed."""
    cfg = mk_config()
    cfg["train"]["fp16"] = False
    cfg["train"]["checkpoint_interval"] = checkpoint_interval
    return cfg


def _save(tmp_path, cfg, tiny_net, optim_scaler_sched, mk_meta, tiny_arch, *, step: int = 33):
    opt, scaler, sched = optim_scaler_sched
    return save_checkpoint(
        model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=step,
        config=cfg, metadata_kwargs=mk_meta(tiny_arch), checkpoint_dir=tmp_path, kind="full",
    )


# ── the REPRO: resume → next periodic boundary → the save WRITES (it used to raise) ──────
def test_resume_then_first_periodic_boundary_saves_and_emits(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch, spy_sink
):
    """The box's exact death shape: save at step 33 (interval 25), resume via the
    production entry, reach boundary 50, fire the ONE periodic seam. Pre-fix:
    `pydantic ValidationError — torch_compile: Extra inputs are not permitted` raised at
    `checkpoints._write_v2_payload` (the verbatim F-R-P4-1 traceback). Post-fix: the save
    writes through the ONE stamped writer and the LAW-18 event rides the spy sink."""
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


# ── the PURITY oracle (R98: key set derived from RunConfig itself) ───────────────────────
@pytest.mark.parametrize("flags", [
    {},
    {"allow_fresh_scheduler": True},
    {"override_scheduler_horizon": True},
], ids=["plain", "allow_fresh_scheduler", "override_scheduler_horizon"])
def test_resumed_carried_config_is_exactly_the_runconfig_key_set(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch, flags
):
    """After a production-shaped resume the carried config holds EXACTLY the RunConfig
    top-level keys and re-validates — under every resume-directive flag. The
    `allow_fresh_scheduler` arm is the strip's independent witness: the flag puts a real
    directive into the overrides, and only the resume-path boundary strip keeps it out of
    the config every future periodic save re-validates."""
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


# ── the BUILDER pin: no spurious directive injection; a real directive still travels ─────
def test_override_builder_injects_no_directive_keys_for_a_nested_launch_config(mk_config):
    """A nested (production-shaped) launch config yields overrides carrying ZERO
    resume-directive keys — this is the pin a re-added `torch_compile` injection turns
    RED, independently of the downstream strip. The flagged call still carries its
    directive in the OVERRIDES dict (consumed by `resume_trainer`, never by the carried
    config): the mechanism survives, only the injection is dead."""
    cfg = mk_config()
    plain = build_resume_config_overrides(cfg, cfg)
    assert not (set(plain) & RESUME_DIRECTIVE_KEYS), (
        f"spurious directive injection: {sorted(set(plain) & RESUME_DIRECTIVE_KEYS)}"
    )
    flagged = build_resume_config_overrides(cfg, cfg, allow_fresh_scheduler=True)
    assert flagged.get("allow_fresh_scheduler") is True, (
        "the legitimate directive must still travel in the overrides — the fix protects "
        "the carried config, not the mechanism"
    )


# ── the LEGACY-CALLER pin: an explicit flat directive is consumed, never carried ─────────
def test_legacy_explicit_torch_compile_override_never_reaches_the_carried_config(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch
):
    """A legacy caller that passes flat `torch_compile` in `config_overrides` (the
    pre-E0 launch shape, `declared_keys=None` verbatim-merge branch) gets a Trainer whose
    carried config does NOT hold it — the strip covers BOTH loader surfaces because both
    route through `resume_trainer` (O3b: one loader)."""
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


# ── the NOT-SOFTENED pin: the strip never launders a non-directive unknown key ───────────
def test_write_time_validation_still_raises_on_a_non_directive_unknown_key(
    tmp_path, tiny_net, optim_scaler_sched, mk_config, mk_meta, tiny_arch
):
    """Red-team M7 (H1), the exact INVERSE of the repro: a bogus key no schema owns,
    smuggled through `config_overrides` on the legacy verbatim branch, SURVIVES the
    boundary strip (which is directive-scoped, never a general filter), is carried, and
    the FIRST periodic save still RAISES at the ONE writer. A strip widened to
    `k in RunConfig.model_fields` would launder the key, let the save WRITE, and this
    REDs — the resume-path witness that LAW-14's write-time validation stays the error
    surface. (The conformance pin `test_config_snapshot_schema_validated_on_write`
    structurally cannot see this: it saves directly, never through `resume_trainer`.)"""
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


# ── the FORCE-DECLARE pin: mechanism directives win the merge, one authority ─────────────
def test_mechanism_directives_force_declare_and_win_over_baked():
    """Red-team M11a (H2): a mechanism directive present in the overrides WINS over a
    baked flat value even when the operator never declared it, and nothing defers — the
    merge's force-declare set derives from `RESUME_DIRECTIVE_KEYS` (one authority,
    review SHOULD-3). A divergent local set omitting `total_steps` would DEFER the
    operator's `--override-scheduler-horizon` value to the baked one (announced only by
    a warning event nothing asserted) and this REDs."""
    resolved, deferred = apply_config_overrides_f1(
        {"total_steps": 1000}, {"total_steps": 5000}, frozenset(), sink=None,
    )
    assert resolved["total_steps"] == 5000, (
        f"the mechanism directive lost the merge: resolved {resolved!r}"
    )
    assert deferred == frozenset(), (
        f"a force-declared mechanism key may never defer-to-baked; got {deferred!r}"
    )


# ── the DISJOINTNESS guard: the strip may only ever name machinery directives ────────────
def test_resume_directive_keys_never_name_a_runconfig_key():
    """`RESUME_DIRECTIVE_KEYS` ∩ RunConfig fields must stay EMPTY: the boundary strip is
    scoped to the resume machinery's own directives and can never launder a real schema
    key. If a future schema adopts one of these names, this guard fires at the schema
    change, not at the next resumed run's first periodic save."""
    overlap = RESUME_DIRECTIVE_KEYS & RunConfig.model_fields.keys()
    assert not overlap, (
        f"RESUME_DIRECTIVE_KEYS overlaps the live schema: {sorted(overlap)} — the strip "
        "would silently drop a REAL config key; re-scope the directive set instead"
    )
