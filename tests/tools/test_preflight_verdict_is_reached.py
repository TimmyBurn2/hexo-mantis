"""AUDIT-1 F-03 — the preflight evidence report never publishes a verdict it did not reach.

THE DEFECT. `_new_report` constructs the report already saying `verdict: "pass", rc: 0`, and
nothing on the success path ever SETS that verdict — `main`'s two `except` arms only overwrite
it on failure. A `BaseException` (a `KeyboardInterrupt` during a long burst, a callee's
`SystemExit`) unwinds through both arms into the `finally` that writes the report, and the
artifact a mint sign-off reads then says PASS while every assertion block says `not_run`.

The tool's contract #10 already says "a verdict that was REACHED is never overwritten". It had
no CONVERSE. `_finalise_verdict` is that converse, and it runs in `_write_report` beside
`_finalise_not_run` and `_finalise_tier` so no future write path can forget it.

SECOND HALF. If the interrupt lands inside `_run_child`'s `proc.communicate` — the ordinary
place, since that is where the burst is waited on — `report["child"]` was never assigned, so
`_not_run_reason` read `child is None` and published "NO boot was spawned" for a child that
was spawned and might still be holding the card. The record is now written BEFORE the wait.
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
    """By absolute path — `tools/` is not an importable package (the house convention every
    other gate test in this directory follows)."""
    spec = importlib.util.spec_from_file_location("_pfm_verdict_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


# ── the finaliser, driven on the tool's OWN skeleton ──────────────────────────────────

def test_the_skeleton_report_is_born_claiming_a_pass_it_has_not_earned() -> None:
    """The finding's premise, re-derived rather than asserted. If this ever reds because the
    skeleton stops saying `pass`, the defect was fixed upstream and `_finalise_verdict`
    becomes belt-and-braces rather than the load-bearing guard."""
    skeleton = TOOL._new_report("preflight")
    assert skeleton["verdict"] == "pass" and skeleton["rc"] == 0
    for name in ("a_sync", "b_lag", "c_arming"):
        assert skeleton["assertions"][name]["verdict"] == "not_run"


@pytest.mark.parametrize("mode", ["preflight", "audit"])
def test_an_unearned_pass_is_DOWNGRADED_at_write_time(mode: str) -> None:
    """THE PIN. A report whose assertions never reached a verdict cannot be written as one."""
    report = TOOL._new_report(mode)
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "not_reached", report
    assert report["rc"] == TOOL.PreflightVerdictUnreachedError.rc
    assert report["failure"] == "PreflightVerdictUnreachedError"
    assert report["verdict_unreached"], "the downgrade must NAME the blocks that fell short"
    for name in report["verdict_unreached"]:
        assert name in TOOL.MODE_REQUIRED_ASSERTIONS[mode]


def test_a_mode_whose_assertions_ALL_passed_keeps_its_pass() -> None:
    """The control. The finaliser downgrades only — it must not red a genuinely green run,
    which is what `tests/tools/test_preflight_armed_smoke.py` measures end to end."""
    report = TOOL._new_report("preflight")
    for name in TOOL.MODE_REQUIRED_ASSERTIONS["preflight"]:
        report["assertions"][name] = {"verdict": "pass"}
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "pass" and report["rc"] == 0 and report["failure"] is None


def test_audit_mode_does_NOT_require_the_two_boot_assertions() -> None:
    """Audit mode spawns no child, so (a) and (b) are `not_run` BY CONSTRUCTION. Requiring
    them would red gate 12 on every commit — the table is what keeps the two modes' verdicts
    derived from their own subjects."""
    report = TOOL._new_report("audit")
    report["assertions"]["c_arming"] = {"verdict": "pass"}
    TOOL._finalise_verdict(report)
    assert report["verdict"] == "pass", report
    assert report["assertions"]["a_sync"]["verdict"] == "not_run"
    assert TOOL.MODE_REQUIRED_ASSERTIONS["audit"] == ("c_arming",)


def test_a_RECORDED_failure_is_never_rewritten_by_the_finaliser() -> None:
    """Contract #10's own half, still standing: the finaliser touches nothing that already
    reached a verdict, so a raising arm's `failure` name and rc survive verbatim."""
    report = TOOL._new_report("preflight")
    report.update(verdict="fail", rc=34, failure="PreflightWatchdogFiredError")
    TOOL._finalise_verdict(report)
    assert (report["verdict"], report["rc"], report["failure"]) == (
        "fail", 34, "PreflightWatchdogFiredError")


def test_an_unknown_mode_is_a_NAMED_internal_failure_not_a_fallback() -> None:
    """R1 at the derivation: a mode with no entry must refuse, never borrow another mode's
    requirements — the ADJ-13 F-3 class one field over."""
    report = TOOL._new_report("audit")
    report["mode"] = "sideways"
    with pytest.raises(TOOL.PreflightInternalError, match="sideways"):
        TOOL._finalise_verdict(report)


# ── the write path carries it (no future writer can forget) ───────────────────────────

def test_the_WRITE_path_downgrades_so_no_call_site_can_skip_it(tmp_path: Path) -> None:
    """`_finalise_verdict` lives in `_write_report`, not at a call site, for the reason its
    two siblings do: the invariant must hold for every write path there will ever be."""
    report = TOOL._new_report("preflight")
    TOOL._write_report(tmp_path, report)
    written = sorted(tmp_path.glob("preflight_*.json"))
    assert len(written) == 1, written
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["verdict"] == "not_reached", on_disk
    assert on_disk["rc"] != 0


# ── the interrupt path: what an operator's Ctrl-C actually lands ──────────────────────

def test_an_interrupt_inside_the_run_stamps_the_report_and_RERAISES(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit's PIN, at `main`. The interrupt keeps its own semantics at the shell — this
    arm changes what the report SAYS, never what the process does."""
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
    """36/37 sit in the parent-side band. 42-47 belong to the run's own machinery, and a
    preflight failure wearing one of those would be read as a watchdog or an armed abort."""
    for err in (TOOL.PreflightInterruptedError, TOOL.PreflightVerdictUnreachedError):
        assert err.rc not in TOOL.RESERVED_CODES, err
        assert err.rc not in TOOL.WATCHDOG_CODES, err
        assert err.rc != TOOL.RELAUNCH_BUDGET_CODE, err


# ── the child record exists before the blocking wait ──────────────────────────────────

def test_the_child_record_is_assigned_BEFORE_the_blocking_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-03's second half. An interrupt in `proc.communicate` used to leave `child` None, so
    `_not_run_reason` said "NO boot was spawned" about a child that was."""
    report = TOOL._new_report("preflight")
    seen: dict[str, Any] = {}

    class _Proc:
        pid = 4242
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            # the ordinary place for an interrupt to land: record what the report says HERE
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
    """The consequence, read off the sentence the artifact actually carries."""
    report = TOOL._new_report("preflight")
    assert TOOL.NOT_BOOTED_REASON in TOOL._not_run_reason(report)
    report["child"] = {"spawned": True, "pid": 4242, "rc": None, "outcome": "in_flight"}
    reason = TOOL._not_run_reason(report)
    assert TOOL.NOT_BOOTED_REASON not in reason, reason
    assert TOOL.BOOTED_REASON in reason, reason
