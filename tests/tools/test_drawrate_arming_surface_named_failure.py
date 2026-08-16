"""⊕ WPAX Phase D ORACLE — O-D4: an unresolvable `config_path` is a NAMED failure, for ANY
row, not an unnamed rc 1 (DESIGN_D §5.4, §5.5; RED-TEAM_P's F-4, fixed to its class per R71).

RED-at-import until IMPL lands `ArmingSurfaceMissingError`.

**F-4, reproduced at HEAD (DESIGN_D §0, re-driven this stage).** Flip the shipped
`draw_rate_collapse` row to REQUIRED and audit a real `configs/run5.yaml` and `_dotted`
raises `AttributeError: 'TrainConfig' object has no attribute 'step_coordinator'`, which
`main`'s bare `except Exception` collapses into **rc 1 `PreflightInternalError`** — the one
outcome `preflight_mint.py:79` ("every outcome NAMED") and `:1270` ("the tool's own failure
is NAMED, never bare") both say cannot exist. Two shipped claims, falsified by one route.
This route has NO producer at HEAD.

**Written to the CLASS, not to this row (R71).** F-4's own words are *"the same route swallows
a typo in any row's `config_path`"*, so the arms below drive a typo on the ACTOR-LAG row as
well — a fix fitted to the draw-rate row alone would pass every draw-rate arm and leave the
next row's typo landing on rc 1 exactly as before. That is MF-7's failure shape, and R71 is
the law written from it.

**§5.5's asymmetry, both arms, pinned rather than discovered later.** The block shape makes
`_dotted`'s behaviour on an EXPLICITLY DISARMED config load-bearing: the shipped walker
raises `AttributeError: 'NoneType' object has no attribute 'threshold'` on
`train.draw_rate_abort: null` (measured), so a legitimately disarmed config would rc-31
rather than report "disarmed". A `None` met mid-walk must therefore short-circuit to `None`
while a MISSING attribute still raises. The residual is disclosed by the last arm: a typo
*after* a legitimately-`None` segment reports "disarmed" rather than raising. It is caught
where it gates — `PRODUCTION_CONFIGS` includes `"configs/run5.yaml"` and run5 is ARMED, so
the walk reaches the leaf and the typo raises.

Everything below drives `audit_arming`, the walker's only consumer, rather than `_dotted`
directly: the row identity in the message comes from the row, so pinning `_dotted`'s own
parameter list here would constrain IMPL's shape without adding a witness.

R7 / gate 6: the tool's report goes to `tmp_path`, never inside the tree.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mantis.config.armed_aborts import (
    EXEMPT_CONFIGS,
    MANIFEST,
    PRODUCTION_CONFIGS,
    ArmedAbort,
    ArmingSurfaceMissingError,  # RED anchor — F-4's named arm (R71 class fix)
    Status,
    audit_arming,
)
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"


def _load_tool():
    """`tests/tools/test_preflight_mint_process.py:80-92`'s precedent — absolute path, no
    `sys.path` write (R5 / LAW-17)."""
    spec = importlib.util.spec_from_file_location("preflight_mint_for_wpax_d4", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _retyped(name: str, config_path: str) -> tuple[ArmedAbort, ...]:
    """The shipped manifest with ONE row's `config_path` replaced. Every other field is the
    row's own, so the only variable is resolvability."""
    out = []
    for row in MANIFEST:
        out.append(row if row.name != name else ArmedAbort(
            name=row.name, config_path=config_path, mechanism=row.mechanism,
            status=row.status, exit_code=row.exit_code, owner=row.owner,
            source_pin=row.source_pin, note=row.note))
    assert any(r.config_path == config_path for r in out), f"no row named {name!r} to re-point"
    return tuple(out)


def _run5() -> RunConfig:
    return load_config(REPO_ROOT / "configs" / "run5.yaml")


def _disarmed_run5() -> RunConfig:
    dumped = _run5().model_dump()
    dumped["train"]["draw_rate_abort"] = None
    return RunConfig.model_validate(dumped)


def test_an_unresolvable_config_path_is_a_NAMED_failure_not_an_unnamed_rc_1() -> None:
    """The module half. Every arm names the same three things, because each answers a
    different operator question: WHICH ROW is broken, WHAT PATH it declared, and WHICH
    SEGMENT of that path does not exist. An `AttributeError` from pydantic's
    `BaseModel.__getattr__` carries only the last one, and `main` then loses even that.

    MF-2 note (c), taken: the walk must be wrapped in `try/except AttributeError` PER SEGMENT
    rather than pre-checked with `hasattr` — only the former can name which segment failed,
    and `_dotted`'s AttributeError comes from `BaseModel.__getattr__`, not from a plain
    lookup.
    """
    assert issubclass(ArmingSurfaceMissingError, AttributeError), (
        "subclassing AttributeError preserves every existing caller's behaviour — a caller "
        "that catches AttributeError today must not start leaking this one"
    )

    config = _run5()
    cases = {
        "a leaf typo on the draw-rate row": (
            "draw_rate_collapse", "train.draw_rate_abort.thrshold", "thrshold"),
        "a mid-path typo on the draw-rate row": (
            "draw_rate_collapse", "train.draw_rate_abrt.threshold", "draw_rate_abrt"),
        "F-4's own shipped path — the section this delta rules AGAINST creating (§2)": (
            "draw_rate_collapse", "train.step_coordinator.draw_rate_threshold",
            "step_coordinator"),
        "R71: the SAME route on the OTHER row — the fix is to the class, not to this row": (
            "actor_lag", "monitor.actor_lag_abort_enabuled", "actor_lag_abort_enabuled"),
        "R71: a typo in the FIRST segment, on the other row": (
            "actor_lag", "moniter.actor_lag_abort_enabled", "moniter"),
    }
    for reason, (row_name, path, segment) in cases.items():
        with pytest.raises(ArmingSurfaceMissingError) as caught:
            audit_arming(config, manifest=_retyped(row_name, path))
        message = str(caught.value)
        for needle, what in ((row_name, "the ROW, so the operator knows which line to fix"),
                             (path, "the FULL dotted path the row declared"),
                             (segment, "the SEGMENT that does not exist")):
            assert needle in message, (
                f"{reason}: the named failure must carry {what}; {needle!r} is missing from "
                f"{message!r}"
            )


def test_an_explicitly_disarmed_block_reports_DISARMED_and_never_raises() -> None:
    """§5.5, measured on the shipped walker: `_dotted` on `train.draw_rate_abort: null`
    raises `AttributeError: 'NoneType' object has no attribute 'threshold'` today. Left as
    is, a legitimately disarmed config — the posture R59 explicitly permits for smoke runs,
    and the posture four of the five committed configs will carry — would fail gate 12 at
    rc 31 instead of being reported disarmed.

    So the F-4 fix carries one more conjunct: a `None` met MID-WALK is an explicitly disarmed
    block and short-circuits to `None`; a MISSING attribute still raises. Both arms are here
    because a walker that short-circuits on ANY failure would satisfy this arm and silently
    convert every typo above into "disarmed".
    """
    disarmed = _disarmed_run5()
    result = audit_arming(disarmed)
    assert [row.name for row in result.disarmed] == ["draw_rate_collapse"], (
        "a `null` block must report DISARMED through `Mechanism.CONFIG_THRESHOLD_GT_ZERO`'s "
        f"non-numeric arm, with zero change to `Mechanism`; got {result.disarmed}"
    )
    assert list(audit_arming(_run5()).disarmed) == [], (
        "…and the same walk on the ARMED committed config must reach the leaf and find 0.25 "
        "— a short-circuit that fired on the armed path would report run5 disarmed"
    )

    with pytest.raises(ArmingSurfaceMissingError):
        audit_arming(disarmed, manifest=_retyped("draw_rate_collapse", "train.draw_rate_abrt"))

    typo_after_none = audit_arming(
        disarmed, manifest=_retyped("draw_rate_collapse", "train.draw_rate_abort.thrshold"))
    assert [row.name for row in typo_after_none.disarmed] == ["draw_rate_collapse"], (
        "THE DISCLOSED RESIDUAL (§5.5), pinned so it is not rediscovered as a bug: a typo "
        "AFTER a legitimately-None segment reports 'disarmed' rather than raising, because "
        "the walk short-circuits before it can reach the bad segment. It is caught where it "
        "gates — PRODUCTION_CONFIGS includes 'configs/run5.yaml' and run5 is ARMED, so the "
        "walk reaches the leaf and the typo raises (the arm above)"
    )


def test_the_tool_maps_the_named_arm_to_rc_31_and_never_to_the_unnamed_rc_1(
    tmp_path, monkeypatch, capsys,
) -> None:
    """The tool half, and the whole point of F-4. `preflight_mint.py:1270`'s bare
    `except Exception` turns any AttributeError from the walk into rc 1
    `PreflightInternalError`, a code whose own docstring says it cannot happen. The fix maps
    `ArmingSurfaceMissingError` onto the already-defined `PreflightManifestError` — rc 31,
    the code every other manifest-integrity failure already uses — so gate 12's operator sees
    a manifest problem rather than "the tool broke".

    Driven through `main()`, not through the helper, because the rc is produced by main's
    handler chain and that chain is what F-4 defeats. `audit_arming`'s `manifest=` default is
    bound at DEF time, so the tool's `audit_arming(_load(path))` call reads the default rather
    than `TOOL.MANIFEST` — DESIGN §8.4 measured that exact trap on its own plugin. Both are
    therefore rebound, and the assertion below is on rc, not on a mocked call.
    """
    assert TOOL.PreflightManifestError.rc == 31 and TOOL.PreflightInternalError.rc == 1, (
        "harness precondition: the two codes this test distinguishes must be the shipped ones"
    )
    assert PRODUCTION_CONFIGS and EXEMPT_CONFIGS, (
        "harness precondition: the audit must have a scope, or it fails at the vacuity guard "
        "for an unrelated reason"
    )

    bad = _retyped("draw_rate_collapse", "train.draw_rate_abort.thrshold")
    monkeypatch.setattr(TOOL, "MANIFEST", bad)
    monkeypatch.setitem(audit_arming.__kwdefaults__, "manifest", bad)

    rc = TOOL.main(["--audit-only", "--out-dir", str(tmp_path / "report")])
    err = capsys.readouterr().err
    assert rc == 31, (
        "a row whose arming surface does not resolve on a real RunConfig must be rc 31 "
        f"PreflightManifestError. rc 1 is the F-4 route: an unnamed PreflightInternalError, "
        f"the outcome preflight_mint.py:79 and :1270 both claim is impossible. Got rc {rc}\n"
        f"{err[-2000:]}"
    )
    assert "PreflightManifestError" in err and "PreflightInternalError" not in err, (
        f"the failure must be NAMED as a manifest problem, not as the tool breaking; got "
        f"{err[-2000:]}"
    )
    for needle in ("draw_rate_collapse", "train.draw_rate_abort.thrshold", "thrshold"):
        assert needle in err, (
            "the rc-31 message must still carry the row, the path and the failing segment — "
            f"a named code with an unnamed cause is half the fix; missing {needle!r} from "
            f"{err[-2000:]}"
        )


def test_the_shipped_manifest_still_audits_green_so_the_rc_31_arm_is_not_vacuous(
    tmp_path,
) -> None:
    """The control. Every arm above reads a NON-zero outcome off a perturbed manifest; if the
    UNPERTURBED tool were already red, all of them would pass for the wrong reason. This is
    `test_the_mini_tree_rig_is_green_before_it_is_perturbed`'s discipline applied to the
    monkeypatch rig.

    It also states the post-flip fact plainly: with `draw_rate_collapse` REQUIRED and armed on
    run5 at R82's 0.25, gate 12's audit mode is GREEN — the flip does not need a waiver.
    """
    assert TOOL.main(["--audit-only", "--out-dir", str(tmp_path / "control")]) == 0, (
        "the SHIPPED manifest must audit the real tree green after the flip: run5 arms both "
        "required rows, so rc 0 here is the state Phase D lands in"
    )
    # WPMINT Phase K-B (call K-c): the shipped manifest no longer holds zero deferred rows —
    # `grad_norm_hard_abort` is one, which is the deferred mechanism finally being fed the
    # kind of row R81 kept it alive for. The CONTROL's real subject is unchanged: rc 0 above
    # is the whole point, and a DEFERRED row cannot change it because deferred rows print and
    # do not gate. That is asserted directly here rather than inferred from an empty list.
    # R265 / ADJ-D38 adds `sealbot_wr_abort` to that set and the transcribed row list here
    # became a tally that had to be re-edited for a change it has no opinion about (R192(e)).
    # DERIVED now, and stating the claim the control actually rests on: the deferred set is
    # non-empty (so "deferred rows do not gate" has a subject) and rc 0 above held anyway.
    deferred = [row for row in MANIFEST if row.status is Status.DEFERRED]
    assert deferred, (
        "…the control needs at least one deferred row, or 'a deferred row prints loudly and "
        "gates nothing' is being asserted about nothing while rc 0 above proves only that "
        "the required rows are armed"
    )
    deferred_names = {row.name for row in deferred}
    assert "draw_rate_collapse" not in deferred_names, (
        "the draw-rate row must NOT be deferred — Phase D flipped it REQUIRED and a quiet "
        "demotion would stop it gating a production mint while this control stayed green"
    )
    assert all(row.status is Status.REQUIRED for row in MANIFEST
               if row.name not in deferred_names), (
        "every row outside the declared deferred set must be REQUIRED — `status` is the only "
        "thing that decides, and there is no third posture"
    )
