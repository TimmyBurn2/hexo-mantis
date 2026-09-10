"""An unresolvable `config_path` is a NAMED failure, for ANY row, not an unnamed rc 1.

Flip the shipped `draw_rate_collapse` row to REQUIRED, audit a real config, and `_dotted` raises
`AttributeError`, which `main`'s bare `except Exception` collapses into rc 1
`PreflightInternalError` — the one outcome the tool twice claims cannot exist. Written to the
CLASS: the arms drive a typo on the ACTOR-LAG row too, since a fix fitted to the draw-rate row
alone would leave the next row's typo on rc 1 exactly as before. The disarmed asymmetry is
pinned both ways — a `None` met mid-walk short-circuits, a MISSING attribute still raises — and
the residual, a typo AFTER a legitimately-`None` segment reporting "disarmed", is caught where
it gates, since run5 is ARMED and the walk reaches the leaf. Everything drives `audit_arming`,
the walker's only consumer; the tool's report goes to `tmp_path`, never inside the tree.
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
    ArmingSurfaceMissingError,
    Status,
    audit_arming,
)
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"


def _load_tool():
    """Load the tool from its absolute path, with no `sys.path` write."""
    spec = importlib.util.spec_from_file_location("preflight_mint_for_wpax_d4", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _retyped(name: str, config_path: str) -> tuple[ArmedAbort, ...]:
    """The shipped manifest with ONE row's `config_path` replaced; the only variable is
    resolvability."""
    out = []
    for row in MANIFEST:
        out.append(row if row.name != name else ArmedAbort(
            name=row.name, config_path=config_path, mechanism=row.mechanism,
            status=row.status, exit_code=row.exit_code, owner=row.owner,
            source_pin=row.source_pin, note=row.note))
    assert any(r.config_path == config_path for r in out), f"no row named {name!r} to re-point"
    return tuple(out)


def _run5() -> RunConfig:
    return load_config(REPO_ROOT / "configs" / "run6.yaml")


def _disarmed_run5() -> RunConfig:
    dumped = _run5().model_dump()
    dumped["train"]["draw_rate_abort"] = None
    return RunConfig.model_validate(dumped)


def test_an_unresolvable_config_path_is_a_NAMED_failure_not_an_unnamed_rc_1() -> None:
    """The module half: each arm names WHICH ROW is broken, WHAT PATH it declared and WHICH
    SEGMENT does not exist — pydantic's `AttributeError` carries only the last, and `main` loses
    even that. Hence `try/except AttributeError` PER SEGMENT rather than a `hasattr` pre-check."""
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
    """`_dotted` on `train.draw_rate_abort: null` raises `'NoneType' object has no attribute
    'threshold'`, so a legitimately disarmed config would fail gate 12 at rc 31 rather than be
    reported disarmed: a `None` met MID-WALK short-circuits, a MISSING attribute still raises.
    Both arms, because a walker short-circuiting on ANY failure would satisfy just one."""
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
        "gates — PRODUCTION_CONFIGS includes 'configs/run6.yaml' and run5 is ARMED, so the "
        "walk reaches the leaf and the typo raises (the arm above)"
    )


def test_the_tool_maps_the_named_arm_to_rc_31_and_never_to_the_unnamed_rc_1(
    tmp_path, monkeypatch, capsys,
) -> None:
    """The tool half: a bare `except Exception` turns any AttributeError from the walk into rc 1
    `PreflightInternalError`, a code whose own docstring says it cannot happen; the named arm maps
    onto `PreflightManifestError`, rc 31. Driven through `main()` because the rc comes from its
    handler chain, and `audit_arming`'s DEF-time `manifest=` default is rebound alongside
    `TOOL.MANIFEST` so the tool cannot read the unperturbed one."""
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
    """The control: every arm above reads a NON-zero outcome off a PERTURBED manifest, so if the
    unperturbed tool were already red they would all pass for the wrong reason. It also states
    the post-flip fact plainly — with `draw_rate_collapse` REQUIRED and armed, the audit is GREEN.
    """
    assert TOOL.main(["--audit-only", "--out-dir", str(tmp_path / "control")]) == 0, (
        "the SHIPPED manifest must audit the real tree green after the flip: run5 arms both "
        "required rows, so rc 0 here is the state Phase D lands in"
    )
    # Derived, not transcribed: the deferred set must be non-empty so "deferred rows print and
    # do not gate" has a subject, and rc 0 above must hold anyway.
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
