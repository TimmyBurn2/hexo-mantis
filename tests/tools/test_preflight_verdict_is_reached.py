"""The preflight evidence report never publishes a verdict it did not reach.

The skeleton report is born saying `verdict: "pass", rc: 0` and nothing on the success path
sets it, so a `BaseException` unwinding into the `finally` that writes the report published a
PASS while every assertion block said `not_run`. `_finalise_verdict` is the converse of the
tool's "a reached verdict is never overwritten" rule, and it runs inside `_write_report` so no
write path can forget it.

Second half: an interrupt inside `_run_child`'s `proc.communicate` left `report["child"]`
unassigned, so the artifact claimed no boot was spawned for a child that was. The child record
is now written BEFORE the wait.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"


def _load_tool() -> Any:
    """Load the tool by absolute path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("_pfm_verdict_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_the_skeleton_report_is_born_claiming_a_pass_it_has_not_earned() -> None:
    """Prove the skeleton report is born claiming a pass it has not earned.

    If this reds because the skeleton stopped saying `pass`, the defect moved upstream and
    `_finalise_verdict` is belt-and-braces rather than the load-bearing guard.
    """
    skeleton = TOOL._new_report("preflight")
    assert skeleton["verdict"] == "pass" and skeleton["rc"] == 0
    for name in ("a_sync", "b_lag", "c_arming"):
        assert skeleton["assertions"][name]["verdict"] == "not_run"


@pytest.mark.parametrize("mode", ["preflight", "audit"])
def test_an_unearned_pass_is_DOWNGRADED_at_write_time(mode: str) -> None:
    """Prove an unearned pass is downgraded at write time."""
    report = TOOL._new_report(mode)
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "not_reached", report
    assert report["rc"] == TOOL.PreflightVerdictUnreachedError.rc
    assert report["failure"] == "PreflightVerdictUnreachedError"
    assert report["verdict_unreached"], "the downgrade must NAME the blocks that fell short"
    for name in report["verdict_unreached"]:
        assert name in TOOL.MODE_REQUIRED_ASSERTIONS[mode]


def test_a_mode_whose_assertions_ALL_passed_keeps_its_pass() -> None:
    """Prove the finaliser downgrades only, and keeps a genuinely earned pass."""
    report = TOOL._new_report("preflight")
    for name in TOOL.MODE_REQUIRED_ASSERTIONS["preflight"]:
        report["assertions"][name] = {"verdict": "pass"}
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "pass" and report["rc"] == 0 and report["failure"] is None


def test_audit_mode_does_NOT_require_the_two_boot_assertions() -> None:
    """Prove audit mode does not require the two boot assertions: it spawns no child."""
    report = TOOL._new_report("audit")
    report["assertions"]["c_arming"] = {"verdict": "pass"}
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "pass", report
    assert report["assertions"]["a_sync"]["verdict"] == "not_run"
    assert TOOL.MODE_REQUIRED_ASSERTIONS["audit"] == ("c_arming",)


def test_a_RECORDED_failure_is_never_rewritten_by_the_finaliser() -> None:
    """Prove the finaliser rewrites nothing that already reached a verdict."""
    report = TOOL._new_report("preflight")
    report.update(verdict="fail", rc=34, failure="PreflightWatchdogFiredError")
    TOOL._finalise_verdict(report)
    assert (report["verdict"], report["rc"], report["failure"]) == (
        "fail", 34, "PreflightWatchdogFiredError")


def test_an_unknown_mode_is_a_NAMED_internal_failure_not_a_fallback() -> None:
    """Prove an unknown mode is a named internal failure, never a borrowed requirement set."""
    report = TOOL._new_report("audit")
    report["mode"] = "sideways"
    with pytest.raises(TOOL.PreflightInternalError, match="sideways"):
        TOOL._finalise_verdict(report)


def test_the_WRITE_path_downgrades_so_no_call_site_can_skip_it(tmp_path: Path) -> None:
    """Prove the downgrade happens in the write path, so no call site can skip it."""
    report = TOOL._new_report("preflight")
    TOOL._write_report(tmp_path, report)
    written = sorted(tmp_path.glob("preflight_*.json"))
    assert len(written) == 1, written
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["verdict"] == "not_reached", on_disk
    assert on_disk["rc"] != 0


def test_an_interrupt_inside_the_run_stamps_the_report_and_RERAISES(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove an interrupt stamps the report and reraises, so the shell semantics are unchanged."""
    def _boom(*_a: Any, **_k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(TOOL, "_run_audit", _boom)
    with pytest.raises(KeyboardInterrupt):
        TOOL.main(["--audit-only", "--out-dir", str(tmp_path)])

    written = sorted(tmp_path.glob("preflight_*.json"))
    assert len(written) == 1, written
    report = json.loads(written[0].read_text(encoding="utf-8"))
    assert report["verdict"] != "pass", report
    assert report["failure"] == "PreflightInterruptedError"
    assert report["rc"] == TOOL.PreflightInterruptedError.rc
    assert report["interrupted_by"] == "KeyboardInterrupt"


def test_the_interrupt_rc_stays_out_of_the_bands_the_run_reserves() -> None:
    """Prove the preflight rcs stay out of the run's reserved bands.

    A preflight failure wearing 42-47 would be read as a watchdog or an armed abort.
    """
    for err in (TOOL.PreflightInterruptedError, TOOL.PreflightVerdictUnreachedError):
        assert err.rc not in TOOL.RESERVED_CODES, err
        assert err.rc not in TOOL.WATCHDOG_CODES, err
        assert err.rc != TOOL.RELAUNCH_BUDGET_CODE, err


def test_the_child_record_is_assigned_BEFORE_the_blocking_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove the child record is assigned before the blocking wait.

    An interrupt in `proc.communicate` used to leave it None, so the report claimed no boot
    was spawned about a child that was.
    """
    report = TOOL._new_report("preflight")
    seen: dict[str, Any] = {}

    class _Proc:
        pid = 4242
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            # The ordinary place for an interrupt to land: record what the report says here.
            seen["child_at_wait"] = report.get("child")
            raise KeyboardInterrupt

    monkeypatch.setattr(TOOL.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(TOOL, "_child_argv", lambda _args: ["true"])
    args = type("A", (), {"timeout_sec": 1.0, "out_dir": str(tmp_path)})()
    with pytest.raises(KeyboardInterrupt):
        TOOL._run_child(args, report)

    assert seen["child_at_wait"] is not None, (
        "the report still claimed no child while the child was running — the exact state "
        "`_not_run_reason` mis-describes"
    )
    assert seen["child_at_wait"]["spawned"] is True
    assert seen["child_at_wait"]["pid"] == 4242
    assert seen["child_at_wait"]["outcome"] == "in_flight"


def test_the_not_run_reason_no_longer_claims_no_boot_for_a_spawned_child() -> None:
    """Prove the not-run reason no longer claims no boot for a spawned child."""
    report = TOOL._new_report("preflight")
    assert TOOL.NOT_BOOTED_REASON in TOOL._not_run_reason(report)
    report["child"] = {"spawned": True, "pid": 4242, "rc": None, "outcome": "in_flight"}
    reason = TOOL._not_run_reason(report)
    assert TOOL.NOT_BOOTED_REASON not in reason, reason
    assert TOOL.BOOTED_REASON in reason, reason
