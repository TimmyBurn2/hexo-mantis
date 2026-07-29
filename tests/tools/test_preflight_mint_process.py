"""WPAX Phase P — the mint preflight's PROCESS half. **NOT part of the frozen oracle set.**

Why this file exists, and why it is not frozen. `tests/tools/test_preflight_mint.py` is a
byte-frozen ORACLE-WRITE artefact (`bd8e65e682c6a2dc`): it was written before the tool and
cannot receive tests without an R43 event. REVIEW_IMPL_P's core finding is that the freeze
left a clean split in the tool's coverage — the *predicate* half is densely pinned (23 of 23
mutations died) while the *process* half has essentially no producer at all, because no test
in the repo drives a preflight past `_run_child`. Eighteen of forty-nine resurrection routes
stayed green and all but two of them live in that half.

So this file carries the process-half producers, written AFTER ORACLE-WRITE in response to a
review finding. The precedent is Phase S's `tests/train/test_actor_lag_wiring_live.py`, which
was non-frozen by the same deliberate design. Nothing here re-states a frozen oracle's
subject; every test below is the FIRST producer of the thing it drives.

What each block is the only witness to (LAW-07):

- **MF-1** `_classify_child`'s arm order. The shipped order tested the stderr tail BEFORE the
  pass-through range, so a child that exited 12 with `object has no attribute` anywhere in its
  stderr exited the parent 32 — re-creating MF-6, the defect revision loop 1 closed. The whole
  seven-arm classifier had zero producer (RR-11b).
- **MF-3** the R56 source-pin scan, driven THROUGH the audit path. `verify_source_pins` had a
  direct producer (M10) but nothing drove it from inside `_audit_manifest_and_configs`, and
  `source_pins_ok` was a literal — so the scan was deletable in one line with the report still
  claiming it ran, at full tier (RR-08). This is Phase D's forcing function.
- **MF-4** `_build_buffer`'s LAW-11 raise. Replaceable by a silent grid default at full tier
  (RR-12); gate 11's `SCAN_ROOTS` excludes `tools/`.
- **MF-5** rc 41 `PreflightReportUnwritableError`. LAW-14 is *persistence-fatal, no silent
  excepts*, and the `except OSError` could become a silent `return` at full tier (RR-34).
- **MF-6** the assertion-verdict -> exit-code seam. The tool's verdict mechanism, with no
  producer at all (RR-10, RR-32, RR-33).
- **MF-7** gate 12's SCOPE — the config-declaration partition and the AUDIT/PREFLIGHT union.
  Both escapes were demonstrated at rc 0 by REVIEW-impl.
- **MF-8** `b0`'s `>= 2` floor and `a4`, each satisfied by a constant (RR-39, RR-44).
- **SF-I2** the evidence block's integrity claim and the segment glob's run scoping.
- **SF-I3** rc 22, the burst-completeness refusal (RR-14).
- **ADJ-12** the arithmetic that decides run5's expected preflight outcome, rc 23 vs rc 25.
- **CARD-D-BURST-FLOOR** (WPMINT Phase B) the report's MINT TIER — which tier the accepted
  burst was and what it does NOT prove. The only witness to three things nothing else pins:
  that `_burst_tier` reads the config's own floor rows rather than "cleared the max" (which
  would claim draw-rate reachability on a config that arms no draw-rate abort), that coverage
  tracks the OUTCOME rather than the burst length (at HEAD every child dies at TD-4, so
  `covered` is `[]` and both tiers stay OWED), and that tier `sync_lag` is UNREACHABLE on a
  production config — the measured ground for the card's one deviation.

R64 posture: nothing here patches, fakes or monkeypatches anything INSIDE the tool. The
mini-tree rig copies the real tool byte-for-byte to a scratch root so that its own
`REPO_ROOT = parents[2]` resolves there — the tool is unmodified and unaware. The two
stand-ins that do appear (`_ChildOutcome`, `_Identity`) stand in for a *subprocess result
dict* and for a config's identity leaf, neither of which is a production object the tool
constructs; O-2's ban is on stand-ins in the TOOL.

>300 justify (R8): one tool, one process half. Every test below drives the same tool module
loaded once by absolute path, and four of the eight blocks share the one mini-tree rig
(`_mini_tree`) that makes the tool's `REPO_ROOT` addressable at all — that rig is the only
way the audit path's scope, its source-pin scan and its manifest integrity can be driven
without writing inside the repo (R7). Splitting by MUST-FIX would fork the rig four ways and
give each copy its own way to be wrong. Roughly half the length is the per-test "what defect
is this the only witness to" rationale LAW-07 requires.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.config.armed_aborts import (
    EXEMPT_CONFIGS,
    MANIFEST,
    PRODUCTION_CONFIGS,
    ArmedAbort,
    Mechanism,
    Status,
    audit_arming,
)
from mantis.config.loader import discover_configs, load_config
from mantis.monitor.sink import JsonlEventSink
from mantis.train.actor_sync import ActorSync
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"

#: run5's own constants, read from the file rather than restated (§14 item 17 / ADJ-12).
RUN5 = REPO_ROOT / "configs" / "run5.yaml"
_N = 101
#: WPAX Phase D: the burst floor for `configs/run5.yaml` MOVED. run5 now arms the draw-rate
#: abort at `min_step: 25000`, and the same cross-field rule that binds
#: `monitor.actor_lag_threshold_steps` inside the run binds this floor too — so a burst the
#: run5 step floor never reaches is refused at rc 11, exactly as a burst below the lag
#: threshold is. `_N` stays 101 for every drive that is about the a/b assertion arithmetic;
#: the drives that push a REAL `configs/run5.yaml` through `_apply_burst_override` use this.
#: (Measured, not assumed: `max(100, 1, 25000) + 1`.)
_RUN5_BURST = 25001


def _load_tool():
    """Load the gate script by absolute path — ZERO `sys.path` mutation (R5 / LAW-17)."""
    spec = importlib.util.spec_from_file_location("preflight_mint_process", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _run_tool(*args, cwd: Path = REPO_ROOT, tool: Path = TOOL_PATH, timeout: int = 300):
    return subprocess.run([sys.executable, str(tool), *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)


# ══ MF-1 — the §6.3a classification order ══════════════════════════════════════════════
def _child(rc: int, *, tail: str = "", timed_out: bool = False) -> dict:
    """The `_run_child` result dict, in its exact shipped shape (`preflight_mint.py:767`).

    A dict, not a production object: `_classify_child`'s whole input is this dict, so driving
    it directly is driving the real function on its real argument.
    """
    child = {"rc": rc, "timed_out": timed_out, "stderr_tail": tail, "stdout_tail": ""}
    if rc < 0:
        import signal as _signal

        child["signal"] = -rc
        child["signal_name"] = _signal.Signals(-rc).name
    return child


_ATTR = ("Traceback (most recent call last):\n"
         "AttributeError: 'Trainer' object has no attribute 'train_step'\n")


@pytest.mark.parametrize("rc", [10, 11, 12, 13, 30, 41])
def test_a_child_that_named_its_own_outcome_is_believed_over_the_stderr_sniff(rc: int) -> None:
    """MF-1, and the only producer of §6.3a arm 4 at all.

    DESIGN §6.3a's total order puts `rc in [10, 41] -> PROPAGATED UNCHANGED` at arm **4** and
    the missing-attribute stderr sniff at arm **5**. The shipped implementation ran 5 before
    4, and it is reachable rather than theoretical: `_load` wraps any loader exception as
    `load_config(...) raised: {exc}` and `_apply_burst_override` appends
    `Validator said: {exc}`, so an underlying `'X' object has no attribute 'y'` from pydantic
    or yaml lands in the tail of a child that exited with its OWN code. Under the reversed
    order that child exited the parent 32 and lost its name — which is MF-6 verbatim, the
    defect revision loop 1 closed.

    Both stderr arms are driven for every pass-through code, because the defect is precisely
    that the two disagreed.
    """
    for tail in ("", _ATTR):
        with pytest.raises(TOOL.PreflightChildOutcomeError) as caught:
            TOOL._classify_child(_child(rc, tail=tail))
        assert caught.value.rc == rc, (
            f"§6.3a arm 4: a child rc in [10, 41] PROPAGATES UNCHANGED whatever its stderr "
            f"says. With tail={tail!r} the parent reported rc {caught.value.rc}, not {rc} — "
            "a child that named its own outcome must be believed about it"
        )


def test_the_stderr_sniff_still_classifies_a_child_that_did_NOT_name_itself() -> None:
    """The other half of MF-1: moving arm 5 behind arm 4 must not disable it.

    rc 1 is outside the pass-through range — an uncaught traceback, which is exactly the
    child that has nothing to say about itself. That is the child the sniff is for, and it is
    the one HEAD actually produces (TD-4 exits the child 1).
    """
    with pytest.raises(TOOL.PreflightTreeDefectError) as caught:
        TOOL._classify_child(_child(1, tail=_ATTR))
    assert caught.value.rc == 32
    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._classify_child(_child(1, tail="MissingEncodingError: config has no 'encoding'"))
    assert caught.value.rc == 33, (
        "a nonzero child with no missing-attribute signature is rc 33 — this is the wall "
        "HEAD actually hits (TD-4), and it must not be silently promoted to 32"
    )


def test_the_classifier_evaluates_arms_1_to_3_before_anything_else() -> None:
    """RR-11b: the whole seven-arm classifier had zero producer. Arms 1-3 and 6-7 here.

    Arm 1 before arm 2 is load-bearing and is why the timeout row is driven with a NEGATIVE
    rc: a timeout kill also produces a negative `returncode`, so a classifier that checked
    the sign first would report every timeout as a signal death (rc 35) and lose the one fact
    that distinguishes them.
    """
    with pytest.raises(TOOL.PreflightTimeoutError) as caught:
        TOOL._classify_child(_child(-15, timed_out=True))
    assert caught.value.rc == 40, "arm 1 (timed_out) precedes arm 2 (rc < 0)"

    with pytest.raises(TOOL.PreflightChildSignaledError) as caught:
        TOOL._classify_child(_child(-9))
    assert caught.value.rc == 35 and "SIGKILL" in str(caught.value), (
        "N-4: `Popen.returncode` is NEGATIVE on signal death, never 128+N, and the message "
        f"must name the signal; got {caught.value!r}"
    )

    for reserved in TOOL.WATCHDOG_CODES:
        with pytest.raises(TOOL.PreflightWatchdogFiredError) as caught:
            TOOL._classify_child(_child(reserved))
        assert caught.value.rc == 34, f"child rc {reserved} is the run's own watchdog"

    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._classify_child(_child(TOOL.RELAUNCH_BUDGET_CODE))
    assert caught.value.rc == 33, (
        "44 is the supervisor's RELAUNCH_BUDGET_EXIT_CODE and cannot legitimately be raised "
        "by a preflight child — it is 33 with that note, never 34"
    )

    assert TOOL._classify_child(_child(0)) is None, (
        "arm 7: only a child that exited 0 reaches the predicates (the anti-evasion rule)"
    )


def test_a_child_rc_46_is_the_runs_own_ARMED_ABORT_and_is_never_collapsed_to_33() -> None:
    """X-6 — CARD-ABORT-EXIT (R84) in the rc taxonomy, driven rather than declared.

    Before WPMINT Phase X, 46 sat in a hole: outside `PASS_THROUGH` ([10, 41]) and outside the
    reserved set, so `_classify_child` fell through every arm to the final
    `PreflightBootFailedError` and a child that exited with the AUTHORED abort code reported
    **rc 33** — the tool meant to surface the signal would have destroyed it. That is the
    failure this test measures, and it measures it at the number, not at the concept.

    The rc PROPAGATES rather than being rewritten to a parent code, which is the difference
    between this and the rc-34 watchdog arm above: a watchdog fire is `os._exit` mid-run and
    the parent's own diagnosis (34) is the useful thing; an armed abort is COOPERATIVE — the
    run decided, unwound, saved, and returned the manifest's number — so a supervisor must
    read the same number on both sides of this tool.
    """
    for code in TOOL.ARMED_ABORT_CODES:
        with pytest.raises(TOOL.PreflightArmedAbortFiredError) as caught:
            TOOL._classify_child(_child(code))
        assert caught.value.rc == code, (
            f"an armed abort's authored rc must propagate UNCHANGED; got {caught.value.rc} "
            f"for child rc {code}"
        )
        assert caught.value.rc != TOOL.PreflightBootFailedError.rc, (
            "and it must NOT be the rc 33 it collapsed to before the taxonomy was extended"
        )
        assert code not in TOOL.PASS_THROUGH, (
            "the premise: 46 cannot ride arm 4, which is why it needed an arm of its own"
        )
    assert TOOL.ARMED_ABORT_CODES == (46,), (
        f"the authored code is 46 and it comes from the ONE authority; got "
        f"{TOOL.ARMED_ABORT_CODES!r}"
    )
    assert TOOL.DRAW_RATE_COLLAPSE_EXIT_CODE == 46 and 46 not in TOOL.WATCHDOG_CODES, (
        "46 is a reserved code but NOT a watchdog code — it is not delivered by `os._exit` "
        "and must not be diagnosed as a stall"
    )


def test_the_boot_childs_rc_is_decided_by_whether_an_abort_fired() -> None:
    """X-6's other half — `_abort_rc`, the card's one real process boundary.

    `_boot_main` hands `RunHandles.shutdown.abort_rule` to this function and returns what it
    says. Three outcomes, driven directly because the function IS the boundary:

    * no rule fired -> 0 (a clean run is still a clean run);
    * a rule fired with an authored code -> that code, taken from the manifest;
    * a rule fired with NO authored code -> a NAMED failure. Never 0 — an aborted run
      reported as a clean boot is the defect R84 opened the card on — and never an invented
      number, which is the thing R84 refused for `grad_norm_hard_abort`.
    """
    assert TOOL._abort_rc(None) == 0, "no rule fired is the ONLY thing that means rc 0"
    assert TOOL._abort_rc("draw_rate_collapse") == TOOL.DRAW_RATE_COLLAPSE_EXIT_CODE

    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._abort_rc("grad_norm_hard_abort")
    assert "grad_norm_hard_abort" in str(caught.value) and caught.value.rc == 33, (
        "an abort with no authored code is a named failure that NAMES THE RULE, not a silent "
        f"rc 0 and not a fabricated code; got rc {caught.value.rc}"
    )


def test_the_pass_through_range_is_the_designs_range_and_is_not_empty() -> None:
    """RR-33: `PASS_THROUGH = range(0, 0)` kills arm 4 silently — every named child outcome
    collapses to 33 — and the parametrized test above would then fail one code at a time.
    Pinned as a set so the boundaries are explicit: 10 and 41 are IN, 9 and 42 are OUT (42 is
    the run's own watchdog code and must never be mistaken for a preflight outcome)."""
    passing = set(TOOL.PASS_THROUGH)
    assert passing == set(range(10, 42)), f"§6.3a arm 4's range is [10, 41]; got {sorted(passing)}"
    assert 9 not in passing and 42 not in passing


# ══ MF-2 — the docstring states what the tool MEASURABLY does ══════════════════════════
def test_the_module_docstring_names_the_wall_the_boot_actually_hits() -> None:
    """MF-2. The shipped docstring asserted that mode PREFLIGHT terminates on
    CARD-TRAINSTEP-ADAPTER at `train/coordinator/step.py:573`. IMPL measured that false and
    REVIEW-impl re-produced it independently (rc 33, child rc 1, 1.388 s): the boot dies
    EARLIER, at `WorkerPool` construction, on `MissingEncodingError` (TD-4). A gate whose own
    docstring states a measured-false fact is the first thing the next reader believes about
    it, which is SF-7's ruling applied to a docstring instead of an R8 line.

    The measurement itself is `test_the_real_boot_terminates_where_the_docstring_says`
    below (integration tier); this test is the cheap consistency pin that goes red in the
    default tier if the two are ever edited apart.
    """
    doc = TOOL.__doc__ or ""
    assert "CARD-POOL-ENCODING-BRIDGE" in doc and "MissingEncodingError" in doc, (
        "the docstring must name the wall the boot actually hits at HEAD (TD-4)"
    )
    assert "terminates on CARD-TRAINSTEP-ADAPTER" not in doc, (
        "the falsified sentence, verbatim as shipped. TD-1 is BEHIND TD-4, not in front of it"
    )
    assert "measured false" in doc, (
        "the docstring still mentions the old wall (it has to, to say the correction "
        "happened) — so it must also say, in words, that the old claim was MEASURED false"
    )


@pytest.mark.integration
def test_the_real_boot_terminates_where_the_docstring_says(tmp_path) -> None:
    """MF-2's actual producer: the real boot, on the real tree, in production posture.

    This is the only test in the repo that drives a preflight child to completion. It is the
    integration tier because it imports torch and builds a real `Trainer` (~1.4 s of child
    plus interpreter start), and because its subject is the state of the TREE, not of the
    tool: when CARD-POOL-ENCODING-BRIDGE lands, this test is what tells the next reader that
    the docstring's HEAD claim has expired.
    """
    out_dir = tmp_path / "boot"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "240", "--device", "cpu")
    assert result.returncode == 33, (
        "the documented HEAD outcome is rc 33 PreflightBootFailedError on TD-4; got "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-3000:]}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert len(reports) == 1, f"the evidence report is written ALWAYS (§9.1); found {reports}"
    report = json.loads(reports[0].read_text())
    assert report["failure"] == "PreflightBootFailedError"
    assert report["child"]["rc"] == 1 and report["child"]["timed_out"] is False
    assert "MissingEncodingError" in report["child"]["stderr_tail"], (
        "the wall the docstring names must be the wall in the evidence; got tail "
        f"{report['child']['stderr_tail'][-600:]!r}"
    )
    assert "train_step" not in report["child"]["stderr_tail"], (
        "TD-1 is BEHIND TD-4: the child never reaches `step.py:573`, which is exactly what "
        "DESIGN §3.4 got wrong and what MF-2 corrects in the docstring"
    )


# ══ the mini-tree rig: the tool's own REPO_ROOT, addressable ═══════════════════════════
def _mini_tree(tmp_path: Path) -> Path:
    """A scratch root the real tool resolves as its own `REPO_ROOT`.

    `REPO_ROOT = Path(os.path.abspath(__file__)).resolve().parents[2]`, so a byte-identical
    copy of the tool at `<root>/tools/ci_gates/preflight_mint.py` reads `<root>` as the repo.
    The tool is NOT modified, patched or subclassed — it is the shipped file, run as itself
    against a different tree. That is what makes the audit path drivable at all without
    writing inside the real repo (R7 / gate 6).

    The tree carries exactly what the audit path reads: every `configs/*.yaml` (the
    declaration partition's left-hand side), `armed_aborts.py` (hashed into the report), and
    the deferred row's pinned source file (the R56 scan's subject). `MANIFEST`,
    `PRODUCTION_CONFIGS` and `EXEMPT_CONFIGS` still come from the INSTALLED package, so the
    rig varies the tree and never the manifest.
    """
    root = tmp_path / "tree"
    (root / "tools" / "ci_gates").mkdir(parents=True)
    shutil.copy2(TOOL_PATH, root / "tools" / "ci_gates" / "preflight_mint.py")
    (root / "configs").mkdir()
    # ADJ-13 F-1: the rig copies what the ONE discovery authority finds, not a third glob of
    # its own. A rig that enumerated configs differently from the gate it drives would go on
    # passing after exactly the divergence F-1 was.
    for config in discover_configs(REPO_ROOT / "configs"):
        target = root / "configs" / config.relative_to(REPO_ROOT / "configs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, target)
    for rel in ["src/mantis/config/armed_aborts.py",
                *[row.source_pin[0] for row in MANIFEST if row.source_pin]]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, root / rel)
    return root


def _mini_audit(root: Path, *args):
    """`--audit-only` against a mini tree. `cwd` is the mini root so nothing resolves back
    into the real repo, and the copied tool is the executable."""
    return _run_tool("--audit-only", *args, cwd=root,
                     tool=root / "tools" / "ci_gates" / "preflight_mint.py")


def test_the_mini_tree_rig_is_green_before_it_is_perturbed(tmp_path) -> None:
    """The rig's own vacuity floor. Every MF-3 / MF-7 test below reads a NON-zero rc off a
    perturbed mini tree; if the unperturbed tree were already red, all of them would pass for
    the wrong reason. Driven first, and asserted to be byte-for-byte the same tool."""
    root = _mini_tree(tmp_path)
    copied = (root / "tools" / "ci_gates" / "preflight_mint.py").read_bytes()
    assert copied == TOOL_PATH.read_bytes(), "the rig must run the SHIPPED tool, unmodified"
    result = _mini_audit(root)
    assert result.returncode == 0, (
        "an unperturbed mini tree must be as green as the real one; got "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-3000:]}"
    )


# ══ MF-3 — the R56 source-pin scan, driven THROUGH the audit path ══════════════════════
def test_the_source_pin_scan_runs_inside_the_live_audit_path(tmp_path) -> None:
    """MF-3 / RR-08. M10 drives `verify_source_pins` DIRECTLY; nothing drove it through
    `_audit_manifest_and_configs`, and `"source_pins_ok": True` was a hardcoded literal — so
    deleting the call left the whole default tier green (1773 passed) with the report still
    claiming the scan had run.

    §8.4 called this scan "the forcing function that makes Phase D's flip unforgettable", and
    it did that job: the pin fired on the deletion of `draw_rate_threshold: float = 0.0`.
    WPAX Phase D has now landed, so the pin's subject MOVED with it rather than retiring —
    it binds to the THREADING at the construction site
    (`src/mantis/run.py`, `draw_rate_abort=resolve_draw_rate_abort(config.train)`), which is
    what keeps the newly-REQUIRED row tamper-evident while it gates a production mint. The
    claim driven here is unchanged and is read off the manifest, never hardcoded: delete the
    pinned text from the tree and the GATE fails, not merely a helper function. (N-1: a
    REQUIRED row MAY carry a pin — `__post_init__` never forbade it, whatever the class
    docstring used to say.)
    """
    root = _mini_tree(tmp_path)
    pinned = [row for row in MANIFEST if row.source_pin is not None]
    assert pinned, "no pinned row means this test has no subject"
    rel, text = pinned[0].source_pin
    target = root / rel
    original = target.read_text()
    assert text in original, f"the pin {text!r} must be present before it is deleted"
    target.write_text(original.replace(text, "# Phase D deleted the pinned literal\n"))

    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        "deleting the pinned literal must fail GATE 12 by name (rc 31 "
        f"PreflightManifestError), not merely a helper; got {result.returncode}\n"
        f"{output[-3000:]}"
    )
    assert "PreflightManifestError" in output and pinned[0].name in output, (
        f"the failure must name the broken row so the operator knows which; got {output[-1500:]}"
    )
    assert "R56" in output, "…and cite the rule that forbids editing the pin instead"


def test_the_report_publishes_the_pins_the_scan_ACTUALLY_covered(tmp_path) -> None:
    """MF-3's second half. `source_pins_ok` was a literal; it and `source_pins_scanned` are
    now both derived from the scan's own result, so a deleted call is a `NameError` rather
    than a quiet green. The scanned list is what makes the field a MEASUREMENT: a report that
    says `ok` while naming zero pins is a green over nothing, which is the shape MF-3, MF-7
    and b0 are all instances of."""
    result = _run_tool("--audit-only", "--out-dir", str(tmp_path / "out"))
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    reports = sorted((tmp_path / "out").glob("preflight_*.json"))
    assert len(reports) == 1
    manifest = json.loads(reports[0].read_text())["manifest"]
    assert manifest["source_pins_ok"] is True
    assert manifest["source_pins_scanned"] == [
        row.name for row in MANIFEST if row.source_pin is not None
    ], (
        "the report must name every pin the scan covered, and the list must be non-empty — "
        f"got {manifest.get('source_pins_scanned')!r}"
    )
    assert manifest["source_pins_scanned"], "a scan that covered nothing is not a scan"


# ══ MF-7 — gate 12's SCOPE ═════════════════════════════════════════════════════════════
def test_a_config_declared_by_neither_tuple_fails_the_gate(tmp_path) -> None:
    """MF-7 (i), the escape REVIEW-impl DEMONSTRATED at rc 0.

    A production config that is simply not listed in `PRODUCTION_CONFIGS` was never audited:
    `sed 's/actor_lag_abort_enabled: true/…: false/' configs/run5.yaml > configs/run6.yaml`
    then `--audit-only` returned **0**, with the one required abort disarmed on a config
    sitting in `configs/`. Nothing pinned `configs/*.yaml ⊆ PRODUCTION_CONFIGS ∪ EXEMPT`, and
    R59's "smoke configs may legally be disarmed" was expressed by ABSENCE — which made
    "deliberately exempt" and "forgotten" the same observable.

    The fix is the partition, and this is its producer: the same planted `run6.yaml`, and the
    gate now refuses to guess.
    """
    root = _mini_tree(tmp_path)
    run6 = root / "configs" / "run6.yaml"
    run6.write_text(RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                             "actor_lag_abort_enabled: false"))
    assert "actor_lag_abort_enabled: false" in run6.read_text(), (
        "the planted config must really be disarmed, or this test is vacuous"
    )
    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        "a config on disk that neither tuple names must FAIL the gate — it was rc 0 before "
        f"(never audited at all); got {result.returncode}\n{output[-3000:]}"
    )
    assert "configs/run6.yaml" in output and "UNDECLARED" in output, (
        f"the failure must name the undeclared config and say what to do; got {output[-2000:]}"
    )


def test_a_declaration_that_names_a_missing_config_fails_the_gate(tmp_path) -> None:
    """MF-7 (i), the other direction — gate 11's stale-`KNOWN_DEBT` rule
    (`silent_encoding_gate.py:338-344`) applied to the config set.

    R65's Phase D re-mints run5. If the re-mint lands under a new filename, an unchecked
    tuple goes on auditing a file nobody will run — the register rots into decoration while
    still reading green. Removing a declared config from disk must therefore be as fatal as
    adding an undeclared one.
    """
    root = _mini_tree(tmp_path)
    victim = EXEMPT_CONFIGS[0][0]
    (root / victim).unlink()
    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        f"a declaration naming an absent config must fail; got {result.returncode}\n"
        f"{output[-3000:]}"
    )
    assert "STALE" in output and victim in output, (
        f"the failure must name the stale declaration; got {output[-2000:]}"
    )


def test_the_declaration_partition_holds_on_the_REAL_tree() -> None:
    """The forward direction, on the tree that ships. Without this the two tests above are
    satisfied by a partition that is wrong in the same way everywhere."""
    undeclared, stale, overlapping = TOOL._config_declaration_drift()
    assert (undeclared, stale, overlapping) == ([], [], []), (
        "configs/*.yaml must be partitioned exactly by PRODUCTION_CONFIGS and "
        f"EXEMPT_CONFIGS; undeclared={undeclared} stale={stale} overlapping={overlapping}"
    )
    assert set(TOOL._discovered_configs()) == set(PRODUCTION_CONFIGS) | {
        rel for rel, _reason in EXEMPT_CONFIGS
    }
    assert all(reason.strip() for _rel, reason in EXEMPT_CONFIGS), (
        "an exemption with no written reason is an exemption nobody can justify later — "
        "the reason is DATA and the tool prints it on the failure path"
    )


def test_naming_a_config_ADDS_scrutiny_and_never_replaces_the_production_set(tmp_path) -> None:
    """MF-7 (ii), the second demonstrated escape.

    `_run_audit` REPLACED the production set when `--config` was given while `_run_preflight`
    UNIONED it, so `--audit-only --config X` and the full preflight returned **rc 0 and rc 30
    on the same tree** — two authorities for one law, in one tool, undocumented everywhere.

    The rig makes the production config itself disarmed and then names a DIFFERENT, healthy
    config on the command line. Under the old replace semantics that is rc 0 and the disarmed
    production config is never looked at; under union it is rc 30. The named config lives
    outside `configs/` so the declaration partition is untouched and the only variable is the
    scope rule.
    """
    root = _mini_tree(tmp_path)
    production = root / PRODUCTION_CONFIGS[0]
    production.write_text(production.read_text().replace("actor_lag_abort_enabled: true",
                                                         "actor_lag_abort_enabled: false"))
    healthy = tmp_path / "healthy.yaml"
    healthy.write_text(RUN5.read_text())

    bare = _mini_audit(root)
    assert bare.returncode == 30, (
        f"the disarmed production config must fail on its own; got {bare.returncode}\n"
        f"{(bare.stdout + bare.stderr)[-2000:]}"
    )
    named = _mini_audit(root, "--config", str(healthy))
    output = named.stdout + named.stderr
    assert named.returncode == 30, (
        "naming a healthy config on the command line must NOT excuse the production set — "
        f"replace semantics returned 0 here; got {named.returncode}\n{output[-3000:]}"
    )
    assert "actor_lag" in output and PRODUCTION_CONFIGS[0].split("/")[-1] in output, (
        f"the failure must still name the disarmed production config; got {output[-2000:]}"
    )


def test_both_modes_compute_the_audit_scope_from_the_same_function() -> None:
    """The structural half of MF-7 (ii): one rule, so the two modes cannot drift apart
    again. `_audit_paths` is the only place either mode may derive its config set."""
    source = TOOL_PATH.read_text()
    assert source.count("_audit_paths(") == 3, (
        "exactly one definition and exactly two call sites (one per mode); a third caller or "
        "a second derivation is how the asymmetry comes back"
    )
    named = REPO_ROOT / "configs" / "dev_example.yaml"
    assert TOOL._audit_paths(None) == sorted(TOOL._resolve_production_configs())
    assert TOOL._audit_paths(named) == sorted({named, *TOOL._resolve_production_configs()}), (
        "naming a config UNIONS it with the production set — union is the safe direction "
        "because naming a config can only ever add scrutiny, never remove it"
    )


def test_the_manifest_vacuity_guard_fires_before_anything_indexes_the_paths(
    monkeypatch, tmp_path,
) -> None:
    """RR-20 + RR-21 + SF-I9, all three of which live in the same four lines.

    RR-20: the vacuity guard was belt-and-braces behind the frozen O-6 and had no producer
    in the TOOL — deleting it left the tier green. An empty required set audits EVERY config
    green and an empty `PRODUCTION_CONFIGS` binds no config at all, so this guard is what
    stops "assertion (c) passed" meaning "assertion (c) had nothing to say".

    SF-I9: `_run_audit` indexed `paths[0]` BEFORE the guard ran, so an empty production set
    raised `IndexError` -> the generic handler -> an unnamed rc 1, not the named rc 31. The
    manifest audit now runs first; this drives that ordering.

    `monkeypatch` rebinds a module-level constant in the TOOL MODULE OBJECT, in a test. O-2's
    ban is a census over the tool's SOURCE — the tool contains no monkeypatch and this does
    not put one there.
    """
    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS", ())
    with pytest.raises(TOOL.PreflightManifestError) as caught:
        TOOL._audit_manifest_and_configs([])
    assert caught.value.rc == 31 and "vacuous" in str(caught.value)

    report = TOOL._new_report("audit")
    args = SimpleNamespace(config=None, out_dir=None)
    with pytest.raises(TOOL.PreflightManifestError):
        TOOL._run_audit(args, report)

    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS", tuple(PRODUCTION_CONFIGS))
    # WPAX Phase D: this used to filter the SHIPPED manifest down to its deferred rows. After
    # the flip that filter yields `()`, so the arm would have proved the EMPTY case — which
    # the arm above already proves — and silently lost its own subject: "a manifest that has
    # rows, none of them REQUIRED". The synthetic row restores exactly that subject.
    deferred_only = (_SYNTHETIC_DEFERRED,)
    assert deferred_only and not [row for row in deferred_only
                                  if row.status.value == "required"], (
        "harness precondition: the subject is a NON-EMPTY manifest with no required row"
    )
    monkeypatch.setattr(TOOL, "MANIFEST", deferred_only)
    with pytest.raises(TOOL.PreflightManifestError) as caught:
        TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert "vacuous" in str(caught.value), (
        "a manifest with no REQUIRED row audits every config green — the other half of the "
        f"guard; got {caught.value!s}"
    )


#: WPAX Phase D re-basing (R81's shape, applied to this family). The flip left the SHIPPED
#: manifest with ZERO deferred rows, so `assert deferred, "no deferred row means this test
#: has no subject"` below went red — the four pins lost their subject, not their point.
#: `_print_deferred_rows` SURVIVES (R81 rules it explicitly: CARD-COORD-KNOBS will feed it
#: rows), so the mechanism is driven on a synthetic row through the `manifest=` keyword the
#: tool now exposes — the same seam `audit_arming` already had. Keeping a shipped row
#: deferred so these assertions stayed true was REJECTED: it would shape the manifest a mint
#: reads to suit a test.
_SYNTHETIC_DEFERRED = ArmedAbort(
    name="_synthetic_deferred_probe",
    config_path="train.does_not_exist",
    mechanism=Mechanism.CONFIG_BOOL,
    status=Status.DEFERRED,
    exit_code=None,
    owner="CARD-COORD-KNOBS (R78)",
    source_pin=("src/mantis/run.py", "def compose_run"),
    note="synthetic subject for R56's loud-debt mechanism; not a shipped row.",
)


@pytest.fixture(scope="module")
def audit_stdout() -> str:
    """R56's loud print, driven on the synthetic deferred row, shared by the four field pins.

    In-process rather than through a subprocess `--audit-only`: after the flip the shipped
    manifest prints nothing at all, so a real run carries no subject. What is under test is
    the PRINT, and it is driven directly at the seam that exists for it.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        TOOL._print_deferred_rows(manifest=(_SYNTHETIC_DEFERRED,))
    printed = buffer.getvalue()
    assert "DEFERRED" in printed, (
        "the loud print must fire on a manifest that DOES carry debt — if this is empty the "
        f"mechanism is dead and all four pins below are vacuous; got {printed!r}"
    )
    return printed


#: field -> (what the print must carry, why dropping it makes the row un-chaseable).
_DEFERRED_FIELDS = {
    "owner": "deferred debt with nobody chasing it IS the status quo (R56)",
    "config_path": "without the arming surface nobody knows what would close the row",
    "source_pin": "without the pin verbatim the row is not tamper-evident from the print",
    "note": "the note is WHERE 'why is this row deferred' lives — it had NO live consumer "
            "at all before this pass (SF-I4 / R4 / LAW-08)",
}


@pytest.mark.parametrize("field", sorted(_DEFERRED_FIELDS))
def test_the_deferred_print_carries_the_field(field: str, audit_stdout: str) -> None:
    """RR-38 / RR-45 / SF-I5, and the LAW-08 half of SF-I4.

    The dispatcher asked whether R56's DEFERRED print is "loud and tamper-evident or merely
    present". Measured: it was loud by CONVENTION only — the frozen M13 asserts the substrings
    `DEFERRED` and `draw_rate_collapse` and nothing else, so the tool could drop `owner=`, the
    arming-surface line and the `pinned to …` line with the whole default tier green.

    Parametrized per field on purpose (R-P2): with the four folded into one test, all four
    ways of hollowing the print share ONE failure signature and the report cannot say which
    field went missing. One pin per field, so each drop dies alone.
    """
    # WPMINT Phase K-B: the shipped manifest holds ONE deferred row now
    # (`grad_norm_hard_abort`, call K-c). The subject below stays SYNTHETIC anyway, and
    # deliberately: this test hollows the print one field at a time, and a shipped row would
    # make each drop's failure depend on that row's own field values rather than on the
    # print. What is re-pointed is only the premise, which is now the weaker and true one.
    assert [row.name for row in MANIFEST if row.status.value == "deferred"] == [
        "grad_norm_hard_abort"
    ], ("the shipped deferred set is the grad-norm row; the synthetic subject below is "
        "additional to it, never a stand-in for an empty manifest (WPAX Phase D / R81)")
    deferred = [_SYNTHETIC_DEFERRED]
    for row in deferred:
        if field == "owner":
            needle = f"{row.name}  owner={row.owner}"
        elif field == "source_pin":
            rel, text = row.source_pin
            assert rel in audit_stdout, f"{_DEFERRED_FIELDS[field]}; missing {rel!r}"
            needle = text
        else:
            needle = getattr(row, field)
        assert needle in audit_stdout, (
            f"the DEFERRED print must carry {row.name}'s {field}: "
            f"{_DEFERRED_FIELDS[field]}. Missing {needle!r} from:\n{audit_stdout}"
        )


def test_the_report_publishes_the_RESOLVED_coordinator_config(tmp_path) -> None:
    """R78's first design question, ANSWERED YES and driven (WPMINT Phase K-B).

    R78 made "the preflight's JSON dump of the resolved coordinator config" where
    CARD-COORD-KNOBS starts. Phase K's census measured the answer at HEAD as NO: the tool
    built a coordinator config and read exactly `.capacity` off it, so the ~30 knobs that
    decide a run's shape were invisible in the artifact a mint sign-off reads. K-B authors
    them and publishes them.

    Three arms, because the block is only worth anything if all three hold:

    * it is PRESENT and complete — every field of the two resolved specs, by NAME, taken from
      the dataclasses rather than from a list written here (a hand-written key census would
      agree with the code by maintenance, which is the shape this whole card exists to kill);
    * it AGREES with the config on disk, key for key. A block that published defaults would
      be `_not_run_reason`'s named defect — "a shipped constant asserting something the run
      measured otherwise, in the evidence artifact a mint sign-off reads";
    * it MOVES when the config moves. The audit runs on a real minted config, so the drive
      compares two different ones rather than mutating a shipped file.
    """
    import dataclasses

    from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
    from mantis.config.resolve.drain import DrainCapsSpec

    _run_tool("--audit-only", "--config", "configs/run5.yaml",
              "--out-dir", str(tmp_path / "coord"))
    report = json.loads(sorted((tmp_path / "coord").glob("preflight_*.json"))[0].read_text())
    block = report["coordinator"]
    assert block is not None, (
        "the resolved coordinator config must be IN the evidence artifact — R78's rider, and "
        "the census measured its absence"
    )

    config = load_config(REPO_ROOT / "configs" / "run5.yaml")
    assert set(block["knobs"]) == {f.name for f in dataclasses.fields(CoordinatorKnobsSpec)}
    assert set(block["drain_caps"]) == {f.name for f in dataclasses.fields(DrainCapsSpec)}
    assert block["knobs"] == json.loads(json.dumps(
        dataclasses.asdict(resolve_coordinator_knobs(config.train)))), (
        "the published knobs must be the RESOLVED ones, field for field — a block built from "
        f"anything else is a restated literal; got {block['knobs']}"
    )
    assert block["stop_step"] == int(config.train.max_train_steps)
    assert block["draw_rate_abort"] == {"threshold": 0.25, "min_step": 25000,
                                        "N_pool_min": 50, "consec": 3}, (
        "run5's armed terms, as the run will really see them — the four travel together"
    )

    _run_tool("--audit-only", "--config", "configs/smoke_gnn.yaml",
              "--out-dir", str(tmp_path / "smoke"))
    other = json.loads(
        sorted((tmp_path / "smoke").glob("preflight_*.json"))[0].read_text())["coordinator"]
    assert other["stop_step"] == 2000 != block["stop_step"], (
        "the block must MOVE with the config it was resolved from; a constant would report "
        f"the same run length for both, got {other['stop_step']} and {block['stop_step']}"
    )
    assert other["draw_rate_abort"] is None, (
        "a DISARMED config must publish an explicit `null`, not an omitted key: absence would "
        "be indistinguishable from a block the tool forgot to fill"
    )


def test_the_report_publishes_the_audits_own_deferred_and_required_rows(
    tmp_path, monkeypatch,
) -> None:
    """RR-07 + the `exit_code` half of SF-I4.

    `AuditResult.deferred` had **no live consumer**: the tool re-derived the list from
    `MANIFEST` directly, so the field could return `()` with the entire tier green. It is now
    read from the audit's own result, so the published block and the audit that produced it
    cannot disagree. `ArmedAbort.exit_code` — the code the abort FIRES with, which is what
    maps a run's exit status back to a manifest row — was read only by the oracle's flip
    simulation and is now published too.
    """
    _run_tool("--audit-only", "--out-dir", str(tmp_path / "rows"))
    report = json.loads(sorted((tmp_path / "rows").glob("preflight_*.json"))[0].read_text())
    manifest = report["manifest"]
    assert [row["name"] for row in manifest["deferred"]] == [
        row.name for row in MANIFEST if row.status.value == "deferred"
    ] == ["grad_norm_hard_abort"], (
        "the report's deferred block must be the SHIPPED manifest's, row for row. WPMINT "
        "Phase K-B (call K-c) gives it a real subject — the grad-norm row — where WPAX "
        "Phase D left it empty; the equality is against the manifest either way, so the "
        f"published block and the audit that produced it cannot disagree; got {manifest['deferred']!r}"
    )
    # …and the field-completeness claim moves onto a manifest that HAS a deferred row. Left
    # against the shipped one it became `all(...)` over an empty list — an assertion whose
    # subject vanished, which passes forever and witnesses nothing (LAW-07's own species,
    # and the shape REVIEW-impl caught in Phase P). `audit_arming`'s `manifest=` default is
    # bound at DEF time, so BOTH the module attribute and the kwdefault are rebound (F-2).
    probe_manifest = (*MANIFEST, _SYNTHETIC_DEFERRED)
    monkeypatch.setattr(TOOL, "MANIFEST", probe_manifest)
    monkeypatch.setitem(audit_arming.__kwdefaults__, "manifest", probe_manifest)
    with_debt = TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert [row["name"] for row in with_debt["deferred"]] == [
        "grad_norm_hard_abort", _SYNTHETIC_DEFERRED.name
    ], (
        "the published deferred block is read from `AuditResult.deferred`, so a manifest "
        "carrying debt must publish ALL of it — the shipped grad-norm row (WPMINT K-B) and "
        f"the probe, in manifest order; got {with_debt['deferred']!r}"
    )
    assert all(row["note"] and row["owner"] and row["source_pin"]
               for row in with_debt["deferred"]), (
        "every published deferred row must carry the three fields that make it chaseable: "
        f"without them the report records a row nobody can act on; got {with_debt['deferred']!r}"
    )
    assert [row["name"] for row in manifest["required_rows"]] == manifest["required"], (
        "the two required views are one authority and must not drift"
    )
    assert [row["exit_code"] for row in manifest["required_rows"]] == [
        row.exit_code for row in MANIFEST if row.status.value == "required"
    ], "a required row's firing exit code must reach the evidence report"
    assert manifest["audited_configs"] and manifest["exempt_configs"], (
        "the report must state the SCOPE it audited, in both directions — MF-7's whole "
        "subject is that the scope was not knowable from the report"
    )


# ══ MF-4 — LAW-11 in the buffer selector ═══════════════════════════════════════════════
def _identity(representation: str, encoding: str = "gnn_axis_v1"):
    """The two leaves `_build_buffer` reads. Not a stand-in for a production object the tool
    constructs (O-2's subject) — it is the argument, and building a `RunConfig` whose
    `identity.representation` is unknown is impossible by construction, which is the point."""
    return SimpleNamespace(identity=SimpleNamespace(representation=representation,
                                                    encoding=encoding))


def test_an_unknown_representation_raises_and_is_never_a_dense_default() -> None:
    """MF-4 / RR-12 / LAW-11. Replacing this raise with a silent `ReplayBuffer` default left
    the whole default tier green (1773 passed): O-9 asserts only that the TOKENS `HexgBuffer`,
    `ReplayBuffer`, `identity` and `representation` appear in the source, and all four survive
    the mutation. Gate 11 cannot see it either — `SCAN_ROOTS = ("src", "crates")` excludes
    `tools/` (ruling and measurement recorded at `preflight_mint.py:_build_buffer`).

    So the raise gets a behavioural producer. Both the absent case and the unknown case are
    driven, because LAW-11's rule is that ABSENT and UNKNOWN are the same error, never a
    default: `"an absent or unknown representation is an ERROR, never a dense default"`.
    """
    for representation in ("hexagonal", "", "dense", "GRAPH", "none"):
        with pytest.raises(TOOL.PreflightConfigError) as caught:
            TOOL._build_buffer(_identity(representation), 8)
        assert caught.value.rc == 10
        assert "LAW-11" in str(caught.value) and repr(representation) in str(caught.value), (
            "the refusal must name the law and the value it refused; got "
            f"{caught.value!s}"
        )


def test_the_two_declared_representations_select_their_own_real_buffer() -> None:
    """The inverse arm — a selector that only ever raises is as useless as one that never
    does. These are the REAL `mantis._engine` buffers, selected off the declared
    representation and never sniffed off a live module."""
    from mantis._engine import HexgBuffer, ReplayBuffer

    graph = TOOL._build_buffer(_identity("graph"), 8)
    grid = TOOL._build_buffer(_identity("grid", encoding="v6"), 8)
    assert isinstance(graph, HexgBuffer) and isinstance(grid, ReplayBuffer), (
        f"graph -> HexgBuffer, grid -> ReplayBuffer; got {type(graph)} / {type(grid)}"
    )
    assert load_config(RUN5).identity.representation == "graph", (
        "run5 is the graph arm, so the graph branch is the one the mint actually takes — "
        "pinned here so a config change that flips it is visible"
    )


# ══ MF-5 — LAW-14: the report write is fatal, never silent ═════════════════════════════
def test_an_unwritable_out_dir_is_rc_41_and_never_a_silent_return(tmp_path) -> None:
    """MF-5 / RR-34 / LAW-14 (*persistence-fatal, no silent excepts*).

    rc 41 `PreflightReportUnwritableError` is N-3's own named outcome and had no producer, so
    the `except OSError` could be turned into a silent `return` with the whole default tier
    green. It is the one outcome a `finally` cannot cover, and it is the outcome that decides
    whether "the report is written ALWAYS" is a fact or a hope.

    The rig makes `--out-dir` an existing regular FILE, so `out_dir.mkdir(parents=True,
    exist_ok=True)` raises `FileExistsError` (an `OSError`). That is a real filesystem
    failure, not a permission trick that a root-run CI would skip.
    """
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("this path is a file, so mkdir on it fails\n")
    result = _run_tool("--audit-only", "--out-dir", str(blocker))
    output = result.stdout + result.stderr
    assert result.returncode == 41, (
        "the audit itself SUCCEEDS here — rc 41 is the report write failing in the `finally`, "
        f"and it must be fatal and named; got {result.returncode}\n{output[-2000:]}"
    )
    assert "PreflightReportUnwritableError" in output and str(blocker) in output, (
        f"the message must name the outcome and the intended path; got {output[-1500:]}"
    )
    assert TOOL.PreflightReportUnwritableError.rc == 41


def test_a_writable_out_dir_still_writes_exactly_one_report(tmp_path) -> None:
    """The inverse arm of MF-5: rc 41 must not be reachable on a healthy write, or the
    `except` has simply been widened into an unconditional failure."""
    _run_tool("--audit-only", "--out-dir", str(tmp_path / "ok"))
    assert len(sorted((tmp_path / "ok").glob("preflight_*.json"))) == 1


# ══ MF-6 — the verdict -> exit-code seam ═══════════════════════════════════════════════
def test_a_failing_assertion_block_raises_with_the_blocks_OWN_exit_code() -> None:
    """MF-6 / RR-10 / RR-32. The tool's verdict mechanism, with no producer at all: no test
    reached `_run_preflight` past `_run_child`, so `if block["verdict"] != "pass"` could be
    made never to raise, and `FAILURE_CODES = {}` could collapse rc 20-29 into 33, both with
    the full tier green.

    MF-3 and MF-7 both reduce to "rc 0 must mean what it says"; this is the code that makes
    the rc track the verdict. Every entry of the table is driven, because a table with one
    tested row is a table that rots in the other eight.
    """
    assert TOOL.FAILURE_CODES, "RR-32: an empty table collapses every named outcome into 33"
    for name, rc in TOOL.FAILURE_CODES.items():
        blocks = {"a_sync": {"verdict": "pass", "failure": None},
                  "b_lag": {"verdict": "fail", "failure": name, "sub_reason": "probe"}}
        with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
            TOOL._verdict_exit(blocks)
        assert caught.value.rc == rc, (
            f"the report's `failure` and the process rc have ONE authority; {name} must exit "
            f"{rc}, got {caught.value.rc}"
        )
        assert caught.value.failure_name == name and name in str(caught.value)


def test_the_failure_code_table_is_the_designs_table() -> None:
    """§6.3's rc column, transcribed once. A table that drifts from the design is a second
    authority for what a CI log's number means, which is the class MF-6 named."""
    assert TOOL.FAILURE_CODES == {
        "PreflightSyncAbsentError": 20,
        "PreflightSyncCadenceError": 21,
        "PreflightBurstIncompleteError": 22,
        "PreflightInversionUndiscriminatedError": 23,
        "PreflightLagUnobservableError": 25,
        "PreflightLagFrozenError": 26,
        "PreflightLagArithmeticError": 27,
        "PreflightLagSourceMismatchError": 28,
        "PreflightLagInvertedError": 29,
    }
    assert 24 not in TOOL.FAILURE_CODES.values(), (
        "§6.3 keeps 24 free so 23 and 25 stay visually distinct in a CI log"
    )
    # WPMINT Phase X: the reserved band grew to 42–46 (46 = the authored draw-rate abort code)
    # and this assertion moved with it. Pinned against `RESERVED_CODES` rather than a re-typed
    # `{42, 43, 44, 45, 46}`: a hand-written set here would go stale exactly as the four-element
    # one just did, silently, because a set literal cannot notice a sixth reserved code.
    assert set(TOOL.FAILURE_CODES.values()).isdisjoint(set(TOOL.RESERVED_CODES)), (
        "the codes the run's OWN machinery reserves must never be an assertion outcome"
    )
    assert TOOL.RESERVED_CODES == (42, 43, 44, 45, 46), (
        "and the band the docstring declares is 42–46; a code that joins the family without "
        f"joining this tuple is one the parent will collapse. Got {TOOL.RESERVED_CODES!r}"
    )


def test_the_a_side_verdict_is_evaluated_before_the_b_side() -> None:
    """`_verdict_exit` reports the FIRST failing block, so a run that breaks both is named by
    (a). Pinned because "which of two failures is reported" is exactly the determinism F-3
    settled for the predicate tables and left unstated for the blocks."""
    blocks = {"a_sync": {"verdict": "fail", "failure": "PreflightSyncAbsentError",
                         "sub_reason": None},
              "b_lag": {"verdict": "fail", "failure": "PreflightLagFrozenError",
                        "sub_reason": "both"}}
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit(blocks)
    assert caught.value.rc == 20 and caught.value.failure_name == "PreflightSyncAbsentError"


def test_two_passing_blocks_raise_nothing() -> None:
    """The inverse: a seam that always raises is a gate that can never go green."""
    assert TOOL._verdict_exit({"a_sync": {"verdict": "pass", "failure": None},
                               "b_lag": {"verdict": "pass", "failure": None}}) is None


def test_an_unknown_failure_name_falls_back_to_a_NAMED_boot_failure() -> None:
    """The table's miss arm. An assertion block naming an outcome the table does not carry
    must still exit nonzero with a name — never 0, and never an unnamed 1."""
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit({"a_sync": {"verdict": "pass", "failure": None},
                            "b_lag": {"verdict": "fail", "failure": "PreflightSomethingNew",
                                      "sub_reason": None}})
    assert caught.value.rc == TOOL.PreflightBootFailedError.rc == 33


# ══ the event streams the remaining blocks drive ═══════════════════════════════════════
_P = 5.0
_STEP_SEC = 0.5
_SAMPLE_TS = (0.0, 15.0, 30.0, 45.0)
_THRESHOLD = 100


class _SyncTarget:
    """`ActorSync.__init__` requires a target (`actor_sync.py:34`) and `maybe_sync` calls two
    methods on it (`:69-70`). The object under test is the real `ActorSync`."""

    def sync_inference_weights(self, state_dict) -> None:
        pass

    def update_checkpoint_step(self, step: int) -> None:
        pass


def _real_syncs(tmp_path: Path, tag: str, steps, *, cadence: int = 1) -> list[dict]:
    """A REAL `ActorSync` through a REAL `JsonlEventSink`, read back off disk. `ts` is
    re-based onto the modelled 0.5 s/step clock, which makes b4c's window HARDER, never
    easier."""
    sink = JsonlEventSink(log_dir=tmp_path / f"sync_{tag}", run_id=f"proc_{tag}")
    learner = {"v": 0}
    sync = ActorSync(target=_SyncTarget(), state_dict_fn=lambda: {},
                     step_fn=lambda: learner["v"], cadence_steps=cadence, sink=sink,
                     run_id=f"proc_{tag}")
    for step in steps:
        learner["v"] = step
        sync.maybe_sync(step)
    events = [json.loads(line) for line in sink.path.read_text().splitlines() if line.strip()]
    syncs = [event for event in events if event.get("event") == "actor_sync"]
    for event in syncs:
        event["ts"] = _STEP_SEC * float(event["step"])
    return syncs


def _model_samples(readings) -> list[dict]:
    """The sample payload shape, licensed by the frozen oracle's
    `test_the_modelled_sample_stream_is_what_the_REAL_watchdog_emits`."""
    return [{"event": "actor_lag_sample", "seq": index, "ts": _SAMPLE_TS[index],
             "learner_step": learner, "actor_ckpt_step": actor,
             "lag_steps": learner - actor, "threshold_steps": _THRESHOLD}
            for index, (learner, actor) in enumerate(readings)]


def _stream(syncs, samples, *, final_step: int = _N) -> list[dict]:
    save = [{"event": "shutdown_save", "step": final_step,
             "ts": _STEP_SEC * float(final_step) + 0.5}]
    return sorted(syncs + samples + save, key=lambda event: float(event["ts"]))


def _assertions(events, *, cadence: int = 1, burst: int = _N):
    return TOOL.evaluate_assertions(events, cadence_steps=cadence, burst_steps=burst,
                                    poll_interval_sec=_P)


# ══ MF-8 — the two predicates a constant satisfies ═════════════════════════════════════
def test_b0_needs_TWO_samples_and_one_is_not_enough(tmp_path) -> None:
    """MF-8 / RR-39. `b0` gates every other b-predicate — DESIGN's own words, *"reporting
    b1…b5a True over an absent measurement is a green over nothing"* — and its `>= 2` floor
    was unpinned: relaxing it to `>= 1` left the full tier green, because the only oracle in
    the corpus drives ZERO samples (M2).

    One sample is the interesting case and it is the one nobody drove: with a single reading
    every transport predicate is trivially satisfiable and NOTHING about the transport has
    been observed — you cannot see a value move by looking at it once. Both sides of the
    floor are asserted, so neither `>= 1` nor `>= 3` survives.
    """
    syncs = _real_syncs(tmp_path, "b0", range(1, _N + 1))
    one = _assertions(_stream(syncs, _model_samples(((0, 0),))))["b_lag"]
    assert one["samples"] == 1, "the rig must really carry exactly one sample"
    assert one["b0"] is False, "one sample is not a measurement of a transport"
    assert one["failure"] == "PreflightLagUnobservableError", (
        "the floor's own named outcome (rc 25) — a relaxed floor reports a DIFFERENT failure "
        f"(b2, frozen learner) and this is what catches it; got {one.get('failure')!r}"
    )
    assert all(one[key] is None for key in TOOL.B_KEYS[1:]), (
        f"b0 gates the rest; reporting them at all is a green over nothing: {one!r}"
    )

    two = _assertions(_stream(syncs, _model_samples(((0, 0), (30, 29)))))["b_lag"]
    assert two["b0"] is True and two["samples"] == 2, (
        "…and exactly two samples must CLEAR the floor, or the floor has merely been raised "
        f"instead of pinned; got {two!r}"
    )


def test_a4_discriminates_a_LOST_SINK_LINE_from_a_missed_sync(tmp_path) -> None:
    """MF-8 / RR-44, and MF-4's restated `a4` — which had no producer at all.

    `a4` is not a cadence predicate: a contiguous `sync_count` holds for any sequence one
    `ActorSync` produces, which is why the frozen oracle pins it TRUE on both a thinned stream
    (M7) and an over-firing one (M14). What it pins is *single-producer / no sink line loss*
    — and the suite asserted `a4 is True` in three places and drove no line-loss stream at
    all, so the constant `True` satisfied every one.

    The pair below is the discrimination itself. Both streams have the SAME observed sync
    steps, so a1 and a2 are identical on them; the only difference is whether the missing
    event was never emitted (the run missed a sync) or emitted and lost (the sink dropped a
    line). `sync_count` is the only witness that can tell those apart, and that is a4's whole
    value.
    """
    missed = _real_syncs(tmp_path, "missed", [s for s in range(1, _N + 1) if s != 50])
    lost = [event for event in _real_syncs(tmp_path, "lost", range(1, _N + 1))
            if event["step"] != 50]
    assert [e["step"] for e in missed] == [e["step"] for e in lost], (
        "the two streams must be indistinguishable on step alone, or the pair proves nothing"
    )
    assert [e["sync_count"] for e in missed] == list(range(1, _N)), (
        "a run that MISSED a sync still counts contiguously — the counter is the producer's"
    )
    assert 50 not in [e["sync_count"] for e in lost], (
        "a LOST line leaves a hole in the counter; that hole is the only observable"
    )

    missed_block = _assertions(_stream(missed, _model_samples(((0, 0), (30, 29)))))["a_sync"]
    lost_block = _assertions(_stream(lost, _model_samples(((0, 0), (30, 29)))))["a_sync"]
    assert (missed_block["a1"], missed_block["a2"]) == (False, False)
    assert (lost_block["a1"], lost_block["a2"]) == (False, False)
    assert missed_block["a4"] is True, "a missed sync is not line loss"
    assert lost_block["a4"] is False, (
        "a lost sink line MUST flip a4 — with a4 constant True the two streams are "
        f"indistinguishable and MF-4's second half has no witness; got {lost_block!r}"
    )


def test_a3_is_a_real_echo_of_the_configs_cadence_and_not_a_constant(tmp_path) -> None:
    """RR-30b / SF-I6. `a3` is in exactly `a4`'s position: the suite asserted `a3 is True` in
    three places and no oracle drove a stream whose `cadence_steps` disagrees with the config,
    so a constant `True` satisfied every one.

    What a3 pins is that the syncs the parent is reading were produced by the cadence the
    parent BOOTED — the burst override rewrites `train.max_train_steps`, and a run whose
    `ActorSync` was constructed from a different config than the one the preflight validated
    is the exact provenance failure the two-stage split exists to make visible. Driven at
    cadence 1 with the events' own echo tampered, so a1/a2/a4 stay True and a3 dies alone.
    """
    syncs = _real_syncs(tmp_path, "a3", range(1, _N + 1))
    for event in syncs:
        event["cadence_steps"] = 7        # the events say 7; the parent booted 1
    block = _assertions(_stream(syncs, _model_samples(((0, 0), (30, 29)))))["a_sync"]
    assert (block["a1"], block["a2"], block["a4"]) == (True, True, True), (
        f"only the echo was tampered, so every other a-predicate must hold: {block!r}"
    )
    assert block["a3"] is False, (
        "a stream that reports a cadence the parent did not boot must fail a3 — with a3 "
        "constant True nothing in the repo can tell the two apart"
    )
    assert block["sub_reason"] == "cadence" and block["failure"] == "PreflightSyncCadenceError"


def test_the_a_side_sub_reason_precedence_is_the_declared_table_order(tmp_path) -> None:
    """RR-42 / SF-I6 — F-3's a-side order, which was unpinned while the b-side was pinned
    hard (reversing the b table kills M3 and the regressing-source row).

    When more than one a-predicate falls, exactly one name reaches the operator, and WHICH
    one is what F-3 settled: the first false in `(a1, a2, a3, a4)`. Two multi-flip streams
    are driven, because a single-flip stream cannot distinguish any ordering from any other.
    """
    assert TOOL.A_KEYS == ("a1", "a2", "a3", "a4"), "the declared table order itself"

    # a1 + a2 + a3 all fall: a missed boundary AND a tampered cadence echo.
    both = _real_syncs(tmp_path, "prec_a", [s for s in range(1, _N + 1) if s != 50])
    for event in both:
        event["cadence_steps"] = 7
    block = _assertions(_stream(both, _model_samples(((0, 0), (30, 29)))))["a_sync"]
    assert (block["a1"], block["a2"], block["a3"]) == (False, False, False), (
        f"the rig must really flip three predicates or it proves no ordering: {block!r}"
    )
    assert block["sub_reason"] == "missed", (
        "a1/a2 precede a3, so the operator is told the run MISSED a sync — the cadence echo "
        f"is the lesser diagnosis; got {block.get('sub_reason')!r}"
    )

    # a3 + a4 both fall: a tampered echo AND a lost sink line, with the steps intact.
    lost = [event for event in _real_syncs(tmp_path, "prec_b", range(1, _N + 1))
            if event["step"] != 50]
    for event in lost:
        event["cadence_steps"] = 7
    lost.append({"event": "actor_sync", "step": 50, "ts": _STEP_SEC * 50.0,
                 "cadence_steps": 7, "sync_count": 999})
    lost.sort(key=lambda event: event["step"])
    block = _assertions(_stream(lost, _model_samples(((0, 0), (30, 29)))))["a_sync"]
    assert (block["a1"], block["a2"]) == (True, True), "the step list is intact here"
    assert (block["a3"], block["a4"]) == (False, False), (
        f"both of the trailing predicates must fall for the order to be observable: {block!r}"
    )
    assert block["sub_reason"] == "cadence", (
        "a3 precedes a4; a reversed table reports `counter` and sends the operator to the "
        f"sink instead of to the config; got {block.get('sub_reason')!r}"
    )


def test_a_burst_that_stopped_short_is_its_own_named_outcome(tmp_path) -> None:
    """SF-I3 / RR-14: rc 22 had no producer, so deleting the burst-completeness check let a
    TRUNCATED burst pass assertion (a) — a run that stopped at step 40 of 101 syncs
    "correctly" for the 40 steps it took. The check is what makes (a) a statement about the
    burst that was ASKED for rather than the one that happened."""
    syncs = _real_syncs(tmp_path, "short", range(1, 41))
    block = _assertions(_stream(syncs, [], final_step=40))["a_sync"]
    assert block["failure"] == "PreflightBurstIncompleteError", (
        f"N=40 against --burst-steps 101 is rc 22, by name; got {block.get('failure')!r}"
    )
    assert block["N"] == 40 and all(block[key] is None for key in TOOL.A_KEYS), (
        "the four sub-predicates are NOT evaluated on a burst that did not finish — "
        f"reporting a cadence over a truncated run is the green-over-nothing shape: {block!r}"
    )
    assert TOOL.FAILURE_CODES[block["failure"]] == 22


# ══ SF-I2 — the evidence block's integrity claim ═══════════════════════════════════════
def _plant(log_dir: Path, name: str, events) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def test_the_evidence_hash_covers_every_segment_that_was_READ(tmp_path) -> None:
    """SF-I2, half two. `events.lines` counted ALL segments while `events.sha256` hashed only
    the LAST — an integrity claim strictly broader than what it hashed, so a report could
    evaluate events its own hash did not cover. The hash is now over exactly the bytes read,
    and every segment that went into it is NAMED."""
    import hashlib

    log_dir = tmp_path / "logs"
    first = _plant(log_dir, "events_r_seg0000.jsonl", [{"event": "actor_sync", "step": 1}])
    second = _plant(log_dir, "events_r_seg0001.jsonl", [{"event": "shutdown_save", "step": 2}])
    segments, events = TOOL._read_segment(log_dir, run_id="r")
    block = TOOL._events_block(segments, events)

    assert block["lines"] == 2 and len(block["segments"]) == 2, (
        f"both segments must be read and both must be named; got {block!r}"
    )
    expected = hashlib.sha256(first.read_bytes() + second.read_bytes()).hexdigest()
    assert block["sha256"] == expected, (
        "the hash must cover the concatenation of every segment consumed, not just the last"
    )
    assert block["sha256"] != hashlib.sha256(second.read_bytes()).hexdigest(), (
        "…and the old last-segment-only hash must no longer satisfy it"
    )


def test_a_foreign_runs_segment_is_not_read_as_this_runs_evidence(tmp_path) -> None:
    """SF-I2, half one — Rig C, demonstrated by REVIEW-impl against a planted directory.

    `_read_segment` globbed `events_*.jsonl` with no run scope, no mtime floor and no
    emptiness requirement on `--out-dir`, so a stale segment left behind by any other run was
    concatenated into THIS run's evidence and evaluated by the predicates. The moment TD-4 and
    TD-1 land and a child can exit 0, that is a green built partly on somebody else's run.
    """
    log_dir = tmp_path / "logs"
    _plant(log_dir, "events_SOMEONE_ELSE_seg0000.jsonl",
           [{"event": "actor_sync", "step": 7, "owner": "SOMEONE_ELSE"}])
    _plant(log_dir, "events_THIS_seg0000.jsonl",
           [{"event": "shutdown_save", "step": 101, "owner": "THIS"}])
    segments, events = TOOL._read_segment(log_dir, run_id="THIS")
    assert [event["owner"] for event in events] == ["THIS"], (
        f"only this run's segments may be consumed; got {[e.get('owner') for e in events]}"
    )
    assert [path.name for path in segments] == ["events_THIS_seg0000.jsonl"]
    assert TOOL._read_segment(log_dir, run_id="ABSENT") == ([], []), (
        "a run with no segment at all reads nothing — never the nearest thing on disk"
    )


# ══ ADJ-12 — rc 23 vs rc 25 for run5, the arithmetic, MEASURED ═════════════════════════
def test_the_second_lag_sample_costs_a_full_file_interval_of_WALL_CLOCK(tmp_path) -> None:
    """ADJ-12's determination, driven through the REAL watchdog at run5's REAL constants.

    ADJ-12 records run5's expected preflight outcome as **rc 23**
    (`PreflightInversionUndiscriminatedError`). That outcome presupposes `b0` — at least TWO
    `actor_lag_sample` events — because b0 GATES every other b-predicate and its own failure
    is **rc 25** `PreflightLagUnobservableError`. So which of the two run5 produces is decided
    by one arithmetic fact, and this test is its producer.

    Under Remedy A the sample interval IS `monitor.heartbeat_file_interval_sec` (the watchdog
    reuses the interval already in the object — one rule, two consumers), and the poll loop
    polls FIRST and then waits `heartbeat_poll_interval_sec`
    (`heartbeat_watchdog.py:418-426`). run5 sets file 15.0 / poll 5.0, so: sample #1 lands on
    the first armed poll, and sample #2 lands on the first poll at or after **t + 15.0 s**.

    Consequence, stated in the notes and NOT resolvable here: a 101-step burst yields two
    samples only if it runs for >= 15.0 s of armed wall clock, i.e. only if a step costs
    >= ~148.5 ms. The step/wall ratio is UNMEASURED at HEAD (DESIGN §14 item 17) and TD-4
    blocks the boot that would measure it, so **rc 25 is live and rc 23 is unproduced**.
    """
    config = load_config(RUN5)
    file_interval = float(config.monitor.heartbeat_file_interval_sec)
    poll_interval = float(config.monitor.heartbeat_poll_interval_sec)
    assert (file_interval, poll_interval) == (15.0, 5.0), (
        "run5's own sampling constants, read from the file; if they change this "
        f"determination changes with them. got file={file_interval} poll={poll_interval}"
    )

    clock = {"t": 0.0}
    sink = JsonlEventSink(log_dir=tmp_path / "adj12", run_id="adj12")
    watchdog = HeartbeatWatchdog(
        registry=SimpleNamespace(sources=("train_step",), ages=lambda: {"train_step": 0.0},
                                 beaten_sources=lambda: frozenset({"train_step"}),
                                 arm=lambda: None),
        deadlines={"train_step": 0.0}, sink=sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=file_interval,
        poll_interval_sec=poll_interval, clock=lambda: clock["t"],
        save_snapshot=lambda: None, exit_fn=lambda code: None, snapshot_timeout_sec=2.0,
        wired_sources=["train_step"],
        actor_lag=ActorLagSpec(learner_step_fn=lambda: int(clock["t"]),
                               actor_ckpt_step_fn=lambda: int(clock["t"]),
                               threshold_steps=_THRESHOLD, abort_enabled=False),
    )

    emitted: list[float] = []
    for tick in range(0, 5):
        clock["t"] = tick * poll_interval
        watchdog.poll_once()
        samples = [json.loads(line) for line in sink.path.read_text().splitlines()
                   if line.strip()]
        count = len([s for s in samples if s.get("event") == "actor_lag_sample"])
        while len(emitted) < count:
            emitted.append(clock["t"])

    assert emitted[0] == 0.0, "the first poll emits a sample (there is no previous one)"
    assert len(emitted) >= 2, "five polls at 5.0 s span 20.0 s and must clear the interval"
    assert emitted[1] == file_interval, (
        "the SECOND sample lands on the first poll at or after one full "
        f"heartbeat_file_interval_sec; got {emitted[1]} with polls at "
        f"{[t * poll_interval for t in range(5)]}"
    )
    assert emitted[1] - emitted[0] >= file_interval, (
        "b0 (>= 2 samples) therefore costs a full 15.0 s of armed wall clock on run5. A "
        "101-step burst shorter than that is rc 25 PreflightLagUnobservableError, NOT the "
        "rc 23 ADJ-12 filed — and the step/wall ratio that would settle it is unmeasured "
        "(TD-4 blocks the boot). The mint checklist must carry both outcomes."
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ADJ-13 (R71/R72) — the RED-TEAM findings, each closed at its CLASS boundary
#
# R71's class-fix law exists because MF-7's fix was fitted to the reviewer's demonstration
# input: the reviewer showed a disarmed `configs/run6.yaml`, the fix closed
# `configs/run6.yaml`, and `configs/run6.yml` plus `configs/prod/run6.yaml` walked straight
# through. So every block below names its CLASS in one sentence and its flip-set covers the
# class boundary rather than the input that demonstrated the defect.
# ══════════════════════════════════════════════════════════════════════════════════════

# ══ F-1 — CLASS: "what is a config file on disk" answered by more than one glob ═════════
#: The class boundary, not the demo input. `run6.yaml` is the input RED-TEAM's predecessor
#: demonstrated and the only one MF-7's fix closed; the other three are the ways a config can
#: enter `configs/` that gate 7 blesses and gate 12 could not see. `tools/mint_config.py --out`
#: takes a free path, so every one of these is a supported output of the repo's own tool.
#:
#: R75 widened this list to **every escape this class has ever produced**, because the
#: protection is now the shared-authority invariant rather than a loader refusal: each of these
#: is loadable, so each must be discovered, so each must be UNDECLARED. `run6.txt` / `run6.YAML`
#: (recheck R-2), `run6.yaml.bak`, `.yaml` and `run6.yamlx` (RED-TEAM) were previously caught by
#: `load_config` refusing them; they are caught HERE now, which is the whole re-ruling.
_F1_PLANT_PATHS = ("run6.yaml", "run6.yml", "prod/run6.yaml", "prod/nested/run6.yml",
                   "run6.txt", "run6.YAML", "run6.yaml.bak", ".yaml", "run6.yamlx")


def _plant_disarmed(root: Path, rel: str) -> Path:
    """A really-disarmed copy of run5 at `configs/<rel>` inside a mini tree."""
    target = root / "configs" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                               "actor_lag_abort_enabled: false"))
    assert "actor_lag_abort_enabled: false" in target.read_text(), (
        "the planted config must really be disarmed, or this row is vacuous"
    )
    return target


@pytest.mark.parametrize("rel", _F1_PLANT_PATHS)
def test_an_undeclared_config_fails_the_gate_at_ANY_suffix_and_ANY_depth(tmp_path, rel) -> None:
    """ADJ-13 F-1, forward half. **Class: gate 12 held its own answer to "what is a config"
    (a flat `configs/*.yaml` glob) beside gate 7's (`**/*.yaml` + `**/*.yml`), so a file one
    gate blessed was invisible to the other.**

    Measured before the fix, on the real tree: `configs/run6.yml` and `configs/prod/run6.yaml`
    each validated under gate 7 (`OK configs/run6.yml`) and returned gate 12 **rc 0 with no
    UNDECLARED line** — the actor-lag abort disarmed, in `configs/`, invisible. Only
    `run6.yaml` — the exact filename the previous reviewer demonstrated — was caught.

    The parametrisation IS the fix's evidence: `.yaml` alone would pass against the old code
    too, so a flip-set containing only the demo input cannot distinguish this fix from its
    predecessor. `.yml`, one subdirectory level and two subdirectory levels are the class.
    """
    root = _mini_tree(tmp_path)
    _plant_disarmed(root, rel)
    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        f"a disarmed config at configs/{rel} must FAIL the gate; it was rc 0 for every "
        f"shape but `run6.yaml` before ADJ-13. got {result.returncode}\n{output[-3000:]}"
    )
    assert f"configs/{rel}" in output and "UNDECLARED" in output, (
        "the failure must name the undeclared config by the SAME relative path a declaration "
        f"would use, subdirectory components included; got {output[-2000:]}"
    )


def test_a_declared_SUBDIRECTORY_config_is_audited_and_never_reported_STALE(
    monkeypatch, tmp_path,
) -> None:
    """ADJ-13 F-1, inverse half — the one that makes it worse than a scope miss.

    With `configs/prod/run6.yaml` on disk AND named in `PRODUCTION_CONFIGS`, the old flat glob
    discovered `['configs/dev_example.yaml', …]` with no `prod/` entry, so the declaration was
    reported **STALE — "declared, absent from disk"** for a file the tool was looking straight
    at: rc 31 carrying a false statement. A subdirectory config could not be legally declared
    at all, which means the forward half could not even be REMEDIED the way the error message
    instructs.

    So both directions are pinned: the path is discoverable, the declaration resolves, and the
    config is really AUDITED (rc 30, disarmed) rather than merely tolerated.

    `monkeypatch` rebinds module-level constants in the TOOL MODULE OBJECT, in a test — the
    same licence `test_the_manifest_vacuity_guard_fires_before_anything_indexes_the_paths`
    takes. O-2's ban is a census over the tool's SOURCE and this puts nothing there.
    """
    root = _mini_tree(tmp_path)
    _plant_disarmed(root, "prod/run6.yaml")
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS",
                        (*PRODUCTION_CONFIGS, "configs/prod/run6.yaml"))

    assert "configs/prod/run6.yaml" in TOOL._discovered_configs(), (
        "discovery must find a subdirectory config, or no declaration of it can ever be "
        f"anything but STALE; got {TOOL._discovered_configs()}"
    )
    undeclared, stale, overlapping = TOOL._config_declaration_drift()
    assert (undeclared, stale, overlapping) == ([], [], []), (
        "a declared, present subdirectory config is a LEGAL state; it was reported STALE "
        f"while sitting on disk. got undeclared={undeclared} stale={stale} "
        f"overlapping={overlapping}"
    )
    with pytest.raises(TOOL.PreflightArmingAuditError) as caught:
        TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert caught.value.rc == 30 and "run6.yaml" in str(caught.value), (
        "…and the declaration must actually BIND it — a subdirectory config that is declared "
        f"but not audited is the same hole wearing a declaration; got {caught.value!s}"
    )


#: **The class boundary the first two fixes both missed, and the third fix got backwards.**
#: MF-7's fix closed the reviewer's `configs/run6.yaml`; the ADJ-13 fix closed `run6.yml` and
#: `configs/prod/`; the recheck walked `configs/run6.txt` and `configs/run6.YAML` straight
#: through both gates. The corrective pass then closed it by making `load_config` REFUSE those
#: suffixes — and **R75 DECLINED that**: a run may be launched from a path of any shape. The
#: protection is the shared-authority invariant instead (loader accepts => audit sees), so the
#: rows below assert the OPPOSITE of what they asserted at `4d11147`: each of these files is
#: loadable, therefore DISCOVERED, therefore UNDECLARED, therefore gate 12 is RED.
#:
#: Still the complement of an enumeration rather than another enumeration: a plain unknown
#: suffix, a CASE variant of a known one, no suffix at all, a known suffix that is not final,
#: and a dotfile named like a suffix.
_F1_UNRECOGNISED = ("run6.txt", "run6.YAML", "run6", "run6.yaml.bak", "run6.YML", "run6.yamL",
                    ".yaml", "run6.yamlx")


@pytest.mark.parametrize("rel", _F1_UNRECOGNISED)
def test_a_config_shaped_file_at_an_UNRECOGNISED_suffix_is_DISCOVERED_and_AUDITED(
    tmp_path, monkeypatch, rel,
) -> None:
    """R71's novel-extension row, driven at the boundary rather than at a demo input, and
    INVERTED by R75.

    Each planted file is a **byte-for-byte copy of run5 with the one REQUIRED armed-abort row
    disarmed** — so anything in the repo that will read it reads a production config with the
    actor-lag hard abort off, which is precisely MF-7's hazard. Measured before ADJ-13's
    corrective pass, on `configs/run6.txt`: schema-valid, `audit_arming` reporting `actor_lag`
    DISARMED, gate 7 **rc 0**, gate 12 **rc 0**, mintable, launchable.

    The loader still reads every one of them (R75). What closes the class is that discovery no
    longer filters by name, so "loadable" and "audited" cannot come apart: the file is
    enumerated, reported UNDECLARED, and gate 12 exits 31. The three assertions are the
    invariant's chain — loadable, discovered, red — because a row that only checked rc could
    pass on a gate that went red for some unrelated reason.
    """
    root = _mini_tree(tmp_path)
    planted = _plant_disarmed(root, rel)
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    relposix = planted.relative_to(root).as_posix()

    assert load_config(planted).run_id == "run5", (
        f"{rel} must still LOAD — R75 declined the accept-set narrowing, so the protection has "
        "to come from the audit seeing it, not from the loader refusing it"
    )
    assert relposix in TOOL._discovered_configs(), (
        f"{rel} is loadable, so discovery MUST enumerate it — that is the shared-authority "
        f"invariant, and its failure is ADJ-13 F-1; got {TOOL._discovered_configs()}"
    )
    undeclared, _stale, _overlapping = TOOL._config_declaration_drift()
    assert relposix in undeclared, (
        f"{rel} is a launchable, disarmed config nobody declared; it must be UNDECLARED rather "
        f"than silently exempt; got {undeclared}"
    )
    result = _run_tool("--audit-only", cwd=root,
                       tool=root / "tools" / "ci_gates" / "preflight_mint.py")
    assert result.returncode == 31, (
        "gate 12 must go RED on a launchable config it cannot account for; got "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-2000:]}"
    )


def test_the_LAUNCH_route_accepts_any_shape_and_the_gates_SEE_it(tmp_path) -> None:
    """`src/mantis/run.py:252` calls `load_config(argv[0])` on a FREE path. That was one of the
    three facts that made `configs/run6.txt` a real hazard; R75 rules it is **not** the fact to
    change — the operator keeps a free launch path, and the audit is what must not have a blind
    spot.

    So this row measures both halves at once: the entry point ACCEPTS the odd shape (the
    backout of `4d11147` at the surface an operator actually types), and the same file under
    `configs/` is red at gate 12 (the protection that replaced the refusal). The control arm is
    the same drive on a real config, so a row that passed by breaking `mantis.run` outright
    would be red.
    """
    disarmed_body = RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                             "actor_lag_abort_enabled: false")
    odd = tmp_path / "run6.txt"
    odd.write_text(disarmed_body)
    launched = subprocess.run([sys.executable, "-m", "mantis.run", str(odd)],
                              cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert launched.returncode == 0, (
        "R75: a run may be launched from a path of any shape; the loader accept-set narrowing "
        f"is out. got rc {launched.returncode}\n{(launched.stdout + launched.stderr)[-2000:]}"
    )
    assert "run_id=run5" in launched.stdout, launched.stdout[-2000:]
    assert "ConfigSuffixError" not in launched.stderr, (
        "the refusal must be gone from the launch path entirely, not merely downgraded"
    )
    allowed = subprocess.run([sys.executable, "-m", "mantis.run", str(RUN5)],
                             cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert allowed.returncode == 0, (
        "the control arm: a real config must still launch, or this row passes by breaking the "
        f"entry point. got {allowed.returncode}\n{(allowed.stdout + allowed.stderr)[-2000:]}"
    )


def test_the_MINT_route_is_free_and_the_PREFLIGHT_covers_it_SHAPE_AGNOSTICALLY(tmp_path) -> None:
    """R75's second consequence, and the claim it rests on — verified rather than asserted.

    `tools/mint_config.py --out` is a free path again: the suffix guard the corrective pass put
    there is out, because the ruling is that *the preflight covers the mint path
    shape-agnostically*. That sentence is only true if `--config <path>` really audits a minted
    config of any shape, so this row mints one at `.txt` and drives the audit at it.

    Both arms are load-bearing. Mint-succeeds alone would pass on a tool that wrote an
    unauditable file; audit-is-red alone would pass on a tool that could not mint at all.
    """
    mint = REPO_ROOT / "tools" / "mint_config.py"
    odd = tmp_path / "minted.txt"
    minted = subprocess.run([sys.executable, str(mint), "--template", "dev", "--out", str(odd),
                             "--set", "run_id=x"], cwd=str(REPO_ROOT), capture_output=True,
                            text=True, timeout=300)
    assert minted.returncode == 0 and odd.is_file(), (
        "R75: `--out` carries no shape constraint; the guard is out. got rc "
        f"{minted.returncode}\n{(minted.stdout + minted.stderr)[-2000:]}"
    )
    assert load_config(odd).run_id == "x", "…and the minted file must be readable"

    audited = _run_tool("--audit-only", "--config", str(odd))
    assert audited.returncode == 30, (
        "the preflight must AUDIT a named config whatever its shape — that is what 'the "
        "preflight covers the mint path shape-agnostically' means, and the dev template is "
        f"disarmed by design (R59). got rc {audited.returncode}\n"
        f"{(audited.stdout + audited.stderr)[-2000:]}"
    )
    assert "minted.txt" in (audited.stdout + audited.stderr), (
        "…and it must name the file it audited, or the rc is about something else"
    )


def test_there_is_NO_excluded_class_left_under_configs(monkeypatch, tmp_path) -> None:
    """ADJ-13 F-1's structural half — inverted once by the corrective pass, and again by R75.

    The row two revisions ago asserted `configs/run6.conf` must NOT be reported UNDECLARED,
    planted as a two-line stub so it never asked whether a `.conf` file could be a real config;
    it **required the escape to stay open**. The corrective pass made the silence defensible by
    having the loader refuse the file. R75 declines that, so the silence goes instead: there is
    no excluded class under `configs/` at all, and the argument that used to justify one — "a
    gate that goes red on a README is a gate operators route around" — is answered by putting
    the README somewhere else. `configs/` is the audit root; it holds configs.

    The tree carries every shape at once: flat `.yaml`, flat `.yml`, nested `.yaml`, a disarmed
    `.conf`, a disarmed `.txt`, and a genuine non-config that is not even valid YAML for a
    config — the hardest case for this ruling, and it is red on purpose.
    """
    root = _mini_tree(tmp_path)
    planted = [_plant_disarmed(root, rel)
               for rel in ("run6.yml", "prod/run6.yaml", "run6.conf", "run6.txt")]
    notes = root / "configs" / "NOTES.md"
    notes.write_text("not a config\n")
    planted.append(notes)

    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    discovered = TOOL._discovered_configs()
    authority = [path.relative_to(root).as_posix()
                 for path in discover_configs(root / "configs")]
    assert discovered == authority, (
        "gate 12's audit set IS the loader's discovery enumeration — not a copy of it and not "
        f"a second glob (R71). got {discovered} vs {authority}"
    )
    undeclared, _stale, _overlapping = TOOL._config_declaration_drift()
    for path in planted:
        rel = path.relative_to(root).as_posix()
        assert rel in discovered, (
            f"{rel} is under the audit root and is not a directory, so it is discovered — "
            f"there is no name-shaped exclusion left to hide behind; got {discovered}"
        )
        assert rel in undeclared, (
            f"{rel} is on disk and in neither declaration tuple; UNDECLARED is the only honest "
            f"report. got {undeclared}"
        )
    shutil.copy2(REPO_ROOT / "tools" / "ci_gates" / "validate_configs.py",
                 root / "tools" / "ci_gates" / "validate_configs.py")
    gate7 = _run_tool(cwd=root, tool=root / "tools" / "ci_gates" / "validate_configs.py")
    assert gate7.returncode == 1 and "FAIL configs/NOTES.md" in gate7.stderr, (
        "the MEASURED COST of the ruling, driven rather than argued: a stray non-config under "
        f"configs/ is a loud gate-7 failure. got rc {gate7.returncode}\n{gate7.stderr[-2000:]}"
    )


#: Recheck R-4 — a REGRESSION the ADJ-13 delta briefly introduced in a gate that was green:
#: an `is_file()` conjunct in `discover_configs` made gate 7 stop rejecting shapes it rejected
#: at `c3ab028`. Under R75 the exclusion is by TYPE, not by name, so the two shapes part
#: company and the row says which is which:
#:
#: * a dangling symlink is a broken FILE reference — the loader's refusal is an accident of the
#:   target's absence, not a property of the path's type — so it stays enumerated and gate 7 is
#:   LOUD, as at `c3ab028`;
#: * a real DIRECTORY is refused by `read_text` unconditionally and is walked THROUGH by
#:   `rglob`, so skipping it cannot hide anything. It is skipped uniformly — `configs/prod/` and
#:   `configs/adir.yaml` are treated alike, where HEAD treated them differently on their NAMES,
#:   which is the shape R75 removes. Gate 7's silence about a directory is now a rule with a
#:   reason rather than a side effect of a glob.
_R4_BROKEN = ("broken_symlink", "symlinked_directory")


@pytest.mark.parametrize("kind", _R4_BROKEN)
def test_a_config_SHAPED_but_BROKEN_path_is_a_LOUD_gate_7_failure_and_not_silence(
    tmp_path, kind,
) -> None:
    """Driven through gate 7 as a process, in a mini tree, because rc is the observable.

    The `symlinked_directory` arm is the input just outside the exclusion's boundary (R71):
    `rglob` will not walk through it, so if discovery ALSO dropped it the subtree behind it
    would be invisible to both gates. Enumerated, it is a loud gate-7 failure that names the
    path an operator has to look at.
    """
    root = _mini_tree(tmp_path)
    shutil.copy2(REPO_ROOT / "tools" / "ci_gates" / "validate_configs.py",
                 root / "tools" / "ci_gates" / "validate_configs.py")
    broken = root / "configs" / "broken.yaml"
    if kind == "broken_symlink":
        broken.symlink_to(tmp_path / "nowhere" / "target.yaml")
    else:
        hidden = tmp_path / "hidden_subtree"
        hidden.mkdir()
        (hidden / "run6.yaml").write_text(RUN5.read_text())
        broken.symlink_to(hidden)

    result = _run_tool(cwd=root, tool=root / "tools" / "ci_gates" / "validate_configs.py")
    assert result.returncode == 1, (
        f"a {kind} at configs/broken.yaml must be a LOUD gate-7 failure. got rc "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-2000:]}"
    )
    assert "FAIL configs/broken.yaml" in result.stderr, result.stderr[-2000:]
    assert result.stdout.count("OK ") == len(discover_configs(REPO_ROOT / "configs")), (
        "…and every real config must still validate, so the row cannot pass by breaking the "
        f"gate outright. got {result.stdout!r}"
    )


def test_a_REAL_directory_is_skipped_UNIFORMLY_and_never_by_its_name(tmp_path, monkeypatch):
    """The other half of R-4, stated as a rule rather than left as a behaviour change.

    At `c3ab028` gate 7 globbed `**/*.yaml`, so a DIRECTORY named `configs/adir.yaml` was red
    while `configs/prod/` was silent — the same path type, two answers, chosen by the name. R75
    removes name from the question: both are skipped, and nothing is lost because `rglob` walks
    through both and enumerates everything loadable inside them.
    """
    root = _mini_tree(tmp_path)
    (root / "configs" / "adir.yaml").mkdir()
    (root / "configs" / "adir.yaml" / "inner.yaml").write_text(RUN5.read_text())
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)

    discovered = TOOL._discovered_configs()
    assert "configs/adir.yaml" not in discovered, (
        f"a real directory is refused by read_text by TYPE, so it is skipped; got {discovered}"
    )
    assert "configs/adir.yaml/inner.yaml" in discovered, (
        "…and the skip hides nothing, because rglob walked through it. A loadable config "
        f"inside a config-NAMED directory is still enumerated; got {discovered}"
    )


def test_gate_7_and_gate_12_enumerate_the_SAME_files_on_the_REAL_tree() -> None:
    """The cross-gate equality, driven through BOTH real gates on the tree that ships.

    The mini-tree rows above could all be satisfied by two globs that happen to agree today.
    This one runs gate 7 as a process, reads the files it says it validated off its own
    stdout, and compares that set to gate 12's audit set. A divergence of any kind — a suffix,
    a depth, a sort order — is red here, which is the only assertion that would have caught
    F-1 the day it landed.
    """
    result = _run_tool(cwd=REPO_ROOT, tool=REPO_ROOT / "tools" / "ci_gates" /
                       "validate_configs.py")
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    gate7 = sorted(line[len("OK "):].strip() for line in result.stdout.splitlines()
                   if line.startswith("OK "))
    assert gate7, "gate 7 printing no OK line means this comparison has no subject"
    assert gate7 == TOOL._discovered_configs(), (
        "gate 7 validates a config gate 12 never audits (or the reverse) — one authority, "
        f"two answers. gate7={gate7} gate12={TOOL._discovered_configs()}"
    )


# ══ R-6 / R-7 — the corrective pass's own two CARD closes ══════════════════════════════
def test_one_config_reached_two_ways_is_audited_ONCE_and_not_twice(tmp_path, monkeypatch) -> None:
    """Recheck R-6 — F-2's class at a site F-2's own census missed.

    F-2 was censused over `os.path.abspath` call SITES; the class it declared is *a path-identity
    comparison whose two sides normalise differently*, and set membership is one. `_audit_paths`
    unioned `_resolve_production_configs()` (a plain `REPO_ROOT / rel`) with `named` (arriving
    `.resolve()`d from `_resolve_config_path`), so under a symlinked `configs/run5.yaml` the set
    held two SPELLINGS of one config: audited twice, published twice in `audited_configs`.

    Fail-safe in direction, which is why it survived — and exactly why it needs a producer: a
    defect whose only symptom is duplicated work has nothing to make it visible. The rig
    replaces run5 with a symlink to a file outside the tree, which is the shape the recheck
    measured.
    """
    root = _mini_tree(tmp_path)
    real = tmp_path / "elsewhere"
    real.mkdir()
    target = real / "run5.yaml"
    target.write_text((root / "configs" / "run5.yaml").read_text())
    (root / "configs" / "run5.yaml").unlink()
    (root / "configs" / "run5.yaml").symlink_to(target)

    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    named = TOOL._resolve_config_path(str(root / "configs" / "run5.yaml"))
    paths = TOOL._audit_paths(named)
    assert len(paths) == len(set(paths)) == len(PRODUCTION_CONFIGS), (
        "one config reached by two spellings must be ONE entry — a set of paths that "
        f"normalise differently is a set of spellings, not of configs; got {paths}"
    )
    assert paths == [target], f"…and both spellings must collapse onto the target; got {paths}"
    bare = TOOL._audit_paths(None)
    assert bare == [target], (
        "the production side alone must normalise the same way, or the union is still "
        f"comparing two schemes; got {bare}"
    )


def test_the_probe_sweep_survives_a_SYMLINK_and_never_takes_the_suite_with_it(tmp_path) -> None:
    """Recheck R-7 — N-2's loud sweep, at the one filesystem shape that inverted it.

    `Path.is_dir()` FOLLOWS symlinks; `shutil.rmtree` REFUSES them. So a symlink at either
    probe path sent the loud sweep into its own `RuntimeError` from a session-scoped autouse
    fixture: **195 collection errors across all of `tests/tools/`**, including every row of gate
    11's corpus, which has nothing to do with the preflight. Measured, both ways — 195 errors
    without the symlink arm, 193 passed with it.

    The conftest is loaded by absolute path (ZERO `sys.path` mutation, R5 / LAW-17) and its
    `PROBES` are redirected into `tmp_path`, so this row never touches the real probe paths the
    session fixture owns.
    """
    spec = importlib.util.spec_from_file_location(
        "preflight_probe_conftest", Path(__file__).parent / "conftest.py")
    assert spec is not None and spec.loader is not None
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep_me.txt").write_text("the symlink target must NOT be removed\n")
    link = tmp_path / "probe_symlink"
    link.symlink_to(elsewhere)
    directory = tmp_path / "probe_dir"
    (directory / "logs").mkdir(parents=True)
    absent = tmp_path / "probe_absent"

    conftest.PROBES = (link, directory, absent)
    conftest._sweep()  # must not raise — that is the whole finding

    assert not link.exists() and not link.is_symlink(), "the symlink probe must be unlinked"
    assert elsewhere.is_dir() and (elsewhere / "keep_me.txt").is_file(), (
        "…by `unlink`, which removes the LINK — a sweep that followed it into the target "
        "would delete a directory it was never pointed at"
    )
    assert not directory.exists(), "the directory arm must still be swept (N-2's own subject)"
    assert not absent.exists()

    unremovable = tmp_path / "unremovable"
    (unremovable / "child").mkdir(parents=True)
    unremovable.chmod(0o500)  # r-x: rmtree cannot unlink the child it contains
    conftest.PROBES = (unremovable,)
    try:
        with pytest.raises(RuntimeError, match="by hand"):
            conftest._sweep()
    finally:
        unremovable.chmod(0o700)
    assert unremovable.is_dir(), (
        "…and LOUDNESS is kept: N-2's finding was a sweep whose failure mode is silence, and "
        "the symlink arm must not have quietly restored `ignore_errors=True`"
    )


# ══ F-2 — CLASS: a path-identity comparison whose two sides normalise differently ═══════
#: The class boundary. `abspath` normalises `..` and makes absolute TEXTUALLY, so rows 1-3
#: were refused before ADJ-13 and prove nothing about the fix; rows 4-6 need the filesystem
#: and every one of them escaped. `_symlink` rows are the defect RED-TEAM demonstrated
#: (`?? reports_redteam_probe/` inside the working tree, rc 33 from the boot wall instead of
#: rc 13 from this guard).
_F2_INSIDE = ("absolute", "dotdot", "toplevel_itself", "symlink", "symlink_two_hops",
              "symlink_to_toplevel")


def _f2_inside_path(kind: str, tmp_path: Path) -> str:
    target = REPO_ROOT / "_preflight_symlink_probe"
    if kind == "absolute":
        return str(target)
    if kind == "dotdot":
        return str(REPO_ROOT / "configs" / ".." / "_preflight_symlink_probe")
    if kind == "toplevel_itself":
        return str(REPO_ROOT)
    if kind == "symlink":
        link = tmp_path / "outlink"
        link.symlink_to(target)
        return str(link)
    if kind == "symlink_two_hops":
        first = tmp_path / "hop1"
        first.symlink_to(target)
        second = tmp_path / "hop2"
        second.symlink_to(first)
        return str(second)
    link = tmp_path / "toplink"
    link.symlink_to(REPO_ROOT)
    return str(link)


@pytest.mark.parametrize("kind", _F2_INSIDE)
def test_an_out_dir_that_reaches_the_repo_BY_ANY_ROUTE_is_refused(tmp_path, kind) -> None:
    """ADJ-13 F-2. **Class: `--out-dir` containment compared an `abspath`-normalised path
    (symlinks NOT followed) against a `.resolve()`d git toplevel — two different naming
    schemes for one filesystem location, so any route that needs the filesystem escaped.**

    The frozen oracle's `test_an_out_dir_inside_the_repo_is_refused` drives the plain
    absolute-inside form only, which `abspath` handles textually — so it stayed green across
    the entire defect. That is why this flip-set is six rows and not one: three routes that
    were always refused, and three that were not, in the same parametrisation, so a
    regression to `abspath` cannot pass by satisfying the easy half.

    The guard's own docstring claims "the refusal lands BEFORE anything is created", so that
    is asserted too — it was false through a symlink, where the tool went on to `mkdir` the
    child's `logs/` and `checkpoints/` and drop its evidence report inside the working tree.
    """
    probe = REPO_ROOT / "_preflight_symlink_probe"
    raw = _f2_inside_path(kind, tmp_path)
    try:
        with pytest.raises(TOOL.PreflightOutDirInsideRepoError) as caught:
            TOOL._checked_out_dir(raw)
        assert caught.value.rc == 13, f"the named outcome is rc 13; got {caught.value.rc}"
        assert not probe.exists(), (
            "the refusal must land BEFORE anything is created — the guard's own docstring "
            f"says so and it was FALSE through a symlink; {probe} exists"
        )
    finally:
        if probe.is_dir():
            shutil.rmtree(probe)


@pytest.mark.parametrize("kind", ["symlink", "symlink_two_hops"])
def test_an_out_dir_reached_through_a_symlink_OUTSIDE_the_repo_is_still_ALLOWED(
    tmp_path, kind,
) -> None:
    """The inverse, and the reason the fix is `.resolve()` rather than "refuse symlinks".

    A guard that refused every symlinked `--out-dir` would pass every row above while making
    the tool unusable on any machine whose scratch directory is a symlink — which is the
    normal shape of `/tmp` under systemd, and exactly how RED-TEAM's own rig was built. The
    class is asymmetric normalisation, not symlinks, and the flip-set has to say so.
    """
    outside = tmp_path / "real_out"
    outside.mkdir()
    link = tmp_path / "link1"
    link.symlink_to(outside)
    if kind == "symlink_two_hops":
        second = tmp_path / "link2"
        second.symlink_to(link)
        link = second
    assert TOOL._checked_out_dir(str(link)) == outside.resolve(), (
        "a symlink whose target is outside the repo is a legal --out-dir and must resolve to "
        "its target"
    )


def test_the_symlink_refusal_is_reached_by_the_REAL_CLI_and_writes_nothing(tmp_path) -> None:
    """F-2 end to end. The unit rows above drive `_checked_out_dir`; this one drives the
    shipped process, because the defect's consequence was a FILE in the working tree and only
    a process can produce that. `--audit-only` keeps it cheap: `main` checks the out-dir
    before either mode runs, so the guard is reached without a boot."""
    probe = REPO_ROOT / "_preflight_symlink_probe"
    link = tmp_path / "outlink"
    link.symlink_to(probe)
    try:
        result = _run_tool("--audit-only", "--out-dir", str(link))
        assert result.returncode == 13, (
            "the real CLI must refuse a symlinked --out-dir; RED-TEAM measured rc 33 from the "
            f"BOOT WALL with the report already written inside the tree. got "
            f"{result.returncode}\n{(result.stdout + result.stderr)[-2000:]}"
        )
        assert "PreflightOutDirInsideRepoError" in (result.stdout + result.stderr)
        assert not probe.exists(), f"{probe} was created by a gate that exists to prevent it"
    finally:
        if probe.is_dir():
            shutil.rmtree(probe)


# ══ F-3 — CLASS: a shipped constant asserting what the run measured otherwise ═══════════
#: **The class, RESTATED by the corrective pass, because the first fix landed inside it.**
#: *A shipped constant asserting something the run measured otherwise, in the evidence artefact
#: a mint sign-off reads.* The first fix keyed the `not_run` disclaimer on `mode` — which is the
#: run's INTENT, not its history — so every mode-PREFLIGHT failure landing before `_run_child`
#: (rc 10 config-path, rc 11 burst-below-floor, rc 30 arming, rc 31 manifest) published *"a boot
#: was spawned and a burst attempted"* beside `"child": null`, and the sentence's own pointer
#: ("see `child` … and `events`") aimed at two null fields. Same class, opposite falsehood: the
#: pre-fix string was false about the mode and true about the boot; the post-fix string was true
#: about the mode and false about the boot. Measured by the recheck at rc 11 — on the fix's OWN
#: headline producer drive, which asserted that the string named the right MODE and never that
#: it was TRUE.
#:
#: So the class boundary is not "which mode" but "what did the run actually do", and the rows
#: below are driven at BOTH answers to that question, against the field that records it.
def test_the_not_run_reason_NAMES_the_mode_the_report_was_written_in() -> None:
    """The mode-naming half — still a real property, and still the one MF-2's recurrence broke.

    `NOT_RUN_REASON` was ONE constant, `"mode=audit — no boot, no burst"`, written
    unconditionally by `_new_report`, so a PREFLIGHT report published the AUDIT disclaimer.
    The flip-set is every mode the tool can write, not just the one that was wrong: a reason
    that names the wrong mode, or the same reason in both modes, is red here.

    What this row deliberately does NOT do any more is treat mode-agreement as truth. That is
    `test_the_not_run_reason_is_DERIVED_from_the_reports_own_child_block`'s subject.
    """
    assert set(TOOL.REPORT_MODES) == {"audit", "preflight"}, (
        "every mode `_new_report` can be called with must be declared; an undeclared mode is "
        f"a named refusal, not a fallback. got {sorted(TOOL.REPORT_MODES)}"
    )
    reasons = set()
    for mode in TOOL.REPORT_MODES:
        report = TOOL._new_report(mode)
        assert report["mode"] == mode
        for name in ("a_sync", "b_lag"):
            block = report["assertions"][name]
            assert block["verdict"] == "not_run"
            assert f"mode={mode}" in block["reason"], (
                f"the {name} not_run reason must name the mode the report was written in; "
                f"mode={mode!r} got {block['reason']!r}"
            )
            reasons.add(block["reason"])
    assert len(reasons) == len(TOOL.REPORT_MODES), (
        "two modes sharing one reason is the defect with an extra dict key — one of them is "
        f"asserting the other's facts. got {reasons}"
    )


def test_an_unknown_report_mode_is_REFUSED_and_never_defaulted() -> None:
    """R1 at the report boundary. A `.get(mode, <some mode>)` would satisfy every assertion
    above while re-creating F-3 for any mode added later — the fallback IS the defect, so
    there is no fallback."""
    with pytest.raises(TOOL.PreflightInternalError) as caught:
        TOOL._new_report("dry-run")
    assert "no code-side default" in str(caught.value)
    with pytest.raises(TOOL.PreflightInternalError):
        TOOL._not_run_reason({"mode": "dry-run", "child": None})


#: The two answers to "what did this run actually do". Not two modes — the recheck's finding is
#: precisely that mode does not answer it.
_F3_HISTORIES = ("no_child", "child")


@pytest.mark.parametrize("history", _F3_HISTORIES)
def test_the_not_run_reason_is_DERIVED_from_the_reports_own_child_block(history) -> None:
    """**The class boundary, both sides of it.** `_finalise_not_run` re-derives the disclaimer
    from `report["child"]` immediately before the write, so the sentence and the field it
    describes cannot disagree — one is computed from the other.

    Driven at `child is None` (every failure before `_run_child`: rc 10 / 11 / 30 / 31) and at a
    populated `child` (rc 32 / 33 / 34 / 35 / 40, and every assertion failure). A constant here —
    in EITHER direction — fails one of these two rows, which is the property the mode-keyed fix
    did not have: it was a constant per mode, and one of its two constants was false on four
    exit codes.
    """
    report = TOOL._new_report("preflight")
    if history == "child":
        report["child"] = {"rc": 33, "timed_out": False}
    TOOL._finalise_not_run(report)
    for name in ("a_sync", "b_lag"):
        reason = report["assertions"][name]["reason"]
        booted_claim = TOOL.BOOTED_REASON in reason
        assert booted_claim is (report["child"] is not None), (
            f"the {name} not_run reason claims booted={booted_claim} while the report's own "
            f"`child` block is {report['child']!r}. A disclaimer that disagrees with the field "
            f"it points at is ADJ-13 F-3 with the falsehood inverted. got {reason!r}"
        )
        assert (TOOL.NOT_BOOTED_REASON in reason) is (report["child"] is None)
        if history == "child":
            assert "child rc 33" in reason, (
                "the booted disclaimer must carry the child's OWN rc, so the sentence is "
                f"checkable against the block beside it; got {reason!r}"
            )


def test_a_verdict_that_was_REACHED_is_never_overwritten_by_the_disclaimer(tmp_path) -> None:
    """The inverse of the row above, and the reason `_finalise_not_run` is not a blanket
    rewrite: a block that reached a real verdict has a MEASUREMENT in it, and stamping a
    not_run disclaimer over it would destroy the evidence the report exists to carry."""
    report = TOOL._new_report("preflight")
    report["child"] = {"rc": 0, "timed_out": False}
    report["assertions"]["a_sync"] = {"verdict": "fail", "failure": "PreflightSyncAbsentError"}
    TOOL._finalise_not_run(report)
    assert report["assertions"]["a_sync"] == {"verdict": "fail",
                                              "failure": "PreflightSyncAbsentError"}, (
        "a block that reached a verdict must be left exactly as measured; got "
        f"{report['assertions']['a_sync']!r}"
    )
    assert TOOL.BOOTED_REASON in report["assertions"]["b_lag"]["reason"], (
        "…while the block that is still not_run does get the derived disclaimer"
    )


def test_a_real_PREFLIGHT_report_never_claims_a_boot_ITS_OWN_child_block_denies(tmp_path) -> None:
    """F-3 through the shipped process, on a real config, writing a real report — and the
    assertion is on TRUTH, not on mode-agreement.

    Driven at a burst below the config's own cross-field floor so it exits rc 11 **without
    spawning a child**. The report is still written from the `finally` (LAW-14), still carries
    `"mode": "preflight"`, and still carries the two `not_run` blocks. That is the artefact a
    mint sign-off reads, and this is the exact drive the first fix cited as its producer while
    checking only that the string named the mode — with the string claiming a boot that this
    very run did not attempt.
    """
    out = tmp_path / "out"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", "5",
                       "--out-dir", str(out), "--timeout-sec", "60", "--device", "cpu")
    assert result.returncode == 11, (result.stdout + result.stderr)[-2000:]
    reports = sorted(out.glob("preflight_*.json"))
    assert len(reports) == 1, f"the evidence report is written ALWAYS; found {reports}"
    report = json.loads(reports[0].read_text())
    assert report["mode"] == "preflight"
    assert report["child"] is None, (
        "the rig is only a witness if this run really did stop before `_run_child`; got "
        f"{report['child']!r}"
    )
    for name in ("a_sync", "b_lag"):
        reason = report["assertions"][name]["reason"]
        assert "mode=audit" not in reason, (
            f"a PREFLIGHT report published the AUDIT disclaimer for {name}: {reason!r}"
        )
        assert "mode=preflight" in reason, f"got {reason!r}"
        assert TOOL.NOT_BOOTED_REASON in reason and TOOL.BOOTED_REASON not in reason, (
            f"NO boot was spawned on this run — `child` is null — and the {name} disclaimer "
            f"claims one was. That is the finding, in the artefact: {reason!r}"
        )


@pytest.mark.integration
def test_a_BOOTED_preflight_reports_a_boot_and_names_its_childs_own_rc(tmp_path) -> None:
    """The other side of the boundary through the shipped process. Integration-tiered for the
    same reason `test_the_real_boot_terminates_where_the_docstring_says` is: it builds a real
    `Trainer` and imports torch. The default tier covers this arm through
    `test_the_not_run_reason_is_DERIVED_from_the_reports_own_child_block[child]`; this row is
    what proves the derived sentence survives the real process, on a real artefact.
    """
    out = tmp_path / "boot"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out), "--timeout-sec", "240", "--device", "cpu")
    assert result.returncode == 33, (result.stdout + result.stderr)[-3000:]
    report = json.loads(sorted(out.glob("preflight_*.json"))[0].read_text())
    assert report["child"] is not None and report["child"]["rc"] == 1
    for name in ("a_sync", "b_lag"):
        reason = report["assertions"][name]["reason"]
        assert TOOL.BOOTED_REASON in reason and TOOL.NOT_BOOTED_REASON not in reason, (
            f"a boot WAS spawned on this run and the {name} disclaimer denies it: {reason!r}"
        )
        assert f"child rc {report['child']['rc']}" in reason


# ══ R72 (recheck R-3) — conjuncts the first enumeration did not list ════════════════════
def test_a_report_with_no_config_block_is_still_NAMED_and_never_unnamed(tmp_path) -> None:
    """`_report_name`'s `or "unknown"` run_id fallback — recheck R-3 / X6: deletable with the
    full default tier green.

    It is live on every report written before `report["config"]` is populated, which is every
    rc 10 / 11 / 30 / 31 preflight — exactly the runs whose evidence a failed mint reads. With
    the fallback gone the filename interpolates `None`, so the artefact is named
    `preflight_None_*.json`; with the whole expression gone it is a `TypeError` inside the
    `finally`, i.e. rc 41 over a report that exists.
    """
    report = TOOL._new_report("preflight")
    assert report["config"] is None
    assert TOOL._report_name(report).startswith("preflight_unknown_"), (
        "a report with no config block must still carry a NAME a reader can file; got "
        f"{TOOL._report_name(report)!r}"
    )
    named = TOOL._new_report("audit")
    named["config"] = {"run_id": "run5"}
    assert TOOL._report_name(named).startswith("preflight_run5_"), (
        "…and when the config block IS populated the run_id must come from it, or the "
        f"fallback is a constant. got {TOOL._report_name(named)!r}"
    )
    out = tmp_path / "out"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", "5",
                       "--out-dir", str(out), "--timeout-sec", "60", "--device", "cpu")
    assert result.returncode == 11
    assert [path.name for path in sorted(out.glob("*.json"))][0].startswith(
        "preflight_run5_"), (
        "the rc-11 route populates `config` before `_apply_burst_override` raises, so the "
        f"real artefact is run5-named; got {sorted(path.name for path in out.glob('*.json'))}"
    )


#: `raised_by` records WHICH side of the process boundary named the outcome: a child rc inside
#: `PASS_THROUGH` is the child's own named code (§6.3a arm 4), anything else was diagnosed by
#: the parent. Recheck R-3 / X7 — reducible to the constant `"parent"` with the full tier
#: green, which is F-3's own species (an evidence-artefact field that asserts nothing) in the
#: same report, and unenumerated by the first R72 pass.
_X7_CHILDREN = ((12, "child"), (50, "parent"), (0, "parent"))


@pytest.mark.parametrize(("child_rc", "expected"), _X7_CHILDREN)
def test_the_reports_raised_by_field_records_WHICH_SIDE_named_the_outcome(
    monkeypatch, tmp_path, child_rc, expected,
) -> None:
    """Driven through the real `_run_child` — a real `Popen`, a real join, a real rc.

    Only `_child_argv` is redirected, on the TOOL MODULE OBJECT (the licence the mini-tree
    rows already take; O-2's ban is a census over the TOOL's source, and nothing inside the
    tool is patched). Redirecting it is what makes the child's exit code controllable at all:
    the shipped argv re-execs this same file, whose rc is a property of the tree.
    """
    monkeypatch.setattr(TOOL, "_child_argv",
                        lambda args: [sys.executable, "-c", f"raise SystemExit({child_rc})"])
    report = TOOL._new_report("preflight")
    child = TOOL._run_child(SimpleNamespace(timeout_sec=60.0), report)
    assert child["rc"] == child_rc
    assert child["raised_by"] == expected, (
        f"child rc {child_rc} is {'inside' if child_rc in TOOL.PASS_THROUGH else 'outside'} "
        f"PASS_THROUGH {TOOL.PASS_THROUGH}, so `raised_by` must be {expected!r} — a constant "
        f"here makes the field decoration in an EVIDENCE artefact; got {child['raised_by']!r}"
    )
    assert report["child"] is child, "the block must be published into the report, not returned"


# ══ F-5 — CLASS: a two-directional property measured in one direction only ══════════════
def test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored(tmp_path) -> None:
    """ADJ-13 F-5, and DESIGN §11 **rig 2** as a test. **Class: `_audit_paths` unions two
    sets, and only the direction that catches REPLACE was produced — the direction that
    catches IGNORE had no oracle at all.**

    Measured: replacing `named = _resolve_config_path(args.config) if args.config else None`
    with `named = None` — i.e. dropping `--config` on the floor — left the **whole default
    tier green** while DESIGN §11 rig 2 returned rc 0 instead of rc 30.
    `test_naming_a_config_ADDS_scrutiny_and_never_replaces_the_production_set` cannot see it:
    it names a HEALTHY config against an ALREADY-DISARMED production set, so rc 30 arrives
    from the production set whether the named config is unioned in or thrown away.

    This is the inverse arm — **healthy production set, DISARMED named config** — which is
    what an operator types when preflighting a candidate before it joins `PRODUCTION_CONFIGS`,
    and the only arm that distinguishes union from "ignore `named`". The named config lives
    outside `configs/` so the declaration partition is untouched and the only variable is
    whether `--config` is honoured.

    Compounded by N-3: `configs/dev_example.yaml` is now EXEMPT, so `--config` is the only
    route by which gate 12 can go red on a disarmed config the operator hands it.
    """
    root = _mini_tree(tmp_path)
    bare = _mini_audit(root)
    assert bare.returncode == 0, (
        "the production set must be HEALTHY here or this test is the old one again; got "
        f"{bare.returncode}\n{(bare.stdout + bare.stderr)[-2000:]}"
    )
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                                  "actor_lag_abort_enabled: false"))
    named = _mini_audit(root, "--config", str(candidate))
    output = named.stdout + named.stderr
    assert named.returncode == 30, (
        "a DISARMED config named on the command line must be audited — `named = None` "
        f"returns 0 here with the whole tier green; got {named.returncode}\n{output[-3000:]}"
    )
    assert "candidate.yaml" in output and "actor_lag" in output, (
        f"the failure must name the config the operator asked about; got {output[-2000:]}"
    )


# ══ F-6 — CLASS: a conjunct of a shipped predicate that no mutation row exercises ═══════
def test_an_inversion_on_a_NON_SAMPLING_poll_is_caught_only_by_b5as_negatives_conjunct(
    tmp_path,
) -> None:
    """ADJ-13 F-6. **Class: `b5a = (not negatives) and all(lag >= 0)` — the first conjunct had
    no corpus row, and deleting it left the whole default tier green (1826 passed).**

    The obvious defence is that the conjuncts are redundant: the sample carries the same
    `detail` dict as the negative event, so a negative lag is visible in both. **Measured
    false at run5's own constants.** The sample is gated on `heartbeat_file_interval_sec`
    (15.0) while `actor_lag_negative` is latched per episode and polls run at
    `heartbeat_poll_interval_sec` (5.0) — so two of every three polls emit no sample, and an
    inversion that begins and ends between samples is invisible to `all(lag >= 0)`.

    Under that configuration the conjunct is not belt-and-braces: it is the PRIMARY arm for
    DESIGN §11 rig 3(α), and it had no test. This is the rig RED-TEAM drove — a real
    `HeartbeatWatchdog`, a real `ActorLagSpec`, a real `JsonlEventSink`, seven polls spanning
    t = 0…30, with the inversion visible only at t = 5 and t = 10.
    """
    config = load_config(RUN5)
    file_interval = float(config.monitor.heartbeat_file_interval_sec)
    poll_interval = float(config.monitor.heartbeat_poll_interval_sec)
    assert (file_interval, poll_interval) == (15.0, 5.0), (
        "run5's own constants, read from the file — this finding IS the ratio between them; "
        f"got file={file_interval} poll={poll_interval}"
    )

    #: (learner, actor) per poll at t = 0, 5, 10, 15, 20, 25, 30. The actor overtakes the
    #: learner at t=5 and t=10 ONLY — both non-sampling polls.
    readings = ((10, 0), (20, 25), (30, 35), (40, 30), (45, 30), (50, 40), (60, 50))
    times = tuple(poll_interval * index for index in range(len(readings)))
    cursor = {"i": 0}
    clock = {"t": 0.0}
    sink = JsonlEventSink(log_dir=tmp_path / "f6", run_id="f6")
    watchdog = HeartbeatWatchdog(
        registry=SimpleNamespace(sources=("train_step",), ages=lambda: {"train_step": 0.0},
                                 beaten_sources=lambda: frozenset({"train_step"}),
                                 arm=lambda: None),
        deadlines={"train_step": 0.0}, sink=sink, counters_fn=lambda: 0,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=file_interval,
        poll_interval_sec=poll_interval, clock=lambda: clock["t"],
        save_snapshot=lambda: None, exit_fn=lambda code: None, snapshot_timeout_sec=2.0,
        wired_sources=["train_step"],
        actor_lag=ActorLagSpec(
            learner_step_fn=lambda: readings[cursor["i"]][0],
            actor_ckpt_step_fn=lambda: readings[cursor["i"]][1],
            threshold_steps=_THRESHOLD, abort_enabled=False),
    )

    samples: list[dict] = []
    negatives: list[dict] = []
    seen = 0
    for index, now in enumerate(times):
        cursor["i"], clock["t"] = index, float(now)
        watchdog.poll_once()
        events = [json.loads(line) for line in sink.path.read_text().splitlines()
                  if line.strip()]
        for event in events[seen:]:
            if event.get("event") == "actor_lag_sample":
                event["ts"] = float(now)
                samples.append(event)
            elif event.get("event") == "actor_lag_negative":
                event["ts"] = float(now)
                negatives.append(event)
        seen = len(events)

    # ── the measurement F-6 rests on, asserted rather than assumed ────────────────────
    assert len(samples) == 3, (
        "seven polls at 5.0 s over a 15.0 s sample interval emit 3 samples (t=0, 15, 30) — if "
        f"this changes, the finding changes with it; got {len(samples)}"
    )
    assert len(negatives) == 1, (
        f"the inversion must be reported once per episode (latched); got {len(negatives)}"
    )
    assert all(int(sample["lag_steps"]) >= 0 for sample in samples), (
        "THE POINT: not one sample carries the inversion, because both inverted polls fell "
        f"between samples. got {[s['lag_steps'] for s in samples]}"
    )

    syncs = _real_syncs(tmp_path, "f6", [30, 50])
    events = sorted([*syncs, *samples, *negatives], key=lambda event: float(event["ts"]))
    block = _assertions(events)["b_lag"]
    assert block["b5a"] is False, (
        "b5a must fail on an inversion that no sample witnessed — `all(lag >= 0)` alone "
        f"returns True here, which is the mutation the whole tier missed. got {block!r}"
    )
    assert block["failure"] == "PreflightLagInvertedError", (
        f"…and it must be the REPORTED failure, not shadowed by an earlier key; got {block!r}"
    )
    assert all(block[key] is True for key in ("b0", "b1", "b2", "b3", "b4a", "b4b", "b4c")), (
        "every other predicate must hold, or this row is not testing the negatives conjunct; "
        f"got {block!r}"
    )


def test_b5as_two_conjuncts_are_each_INDEPENDENTLY_sufficient(tmp_path) -> None:
    """F-6's other half — the isolation that proves NEITHER conjunct is redundant.

    The row above isolates `(not negatives)`: an inversion in the event stream with no
    negative sample. This one isolates `all(lag >= 0)`: a negative SAMPLE with no
    `actor_lag_negative` event, which is what a stream carries when the sink loses the latched
    event or the inversion is still open at the first sample. Both must fail b5a, so a fix
    that drops either conjunct has a red row rather than a green tier.
    """
    syncs = _real_syncs(tmp_path, "b5aiso", [40])
    samples = _model_samples(((0, 0), (30, 40)))
    assert [sample["lag_steps"] for sample in samples] == [0, -10], (
        f"the rig must carry a genuinely negative SAMPLE; got {samples!r}"
    )
    events = sorted([*syncs, *samples], key=lambda event: float(event["ts"]))
    assert not [e for e in events if e.get("event") == "actor_lag_negative"], (
        "…and NO actor_lag_negative event, or this row is the other conjunct again"
    )
    block = _assertions(events)["b_lag"]
    assert block["b5a"] is False and block["failure"] == "PreflightLagInvertedError", (
        f"a negative sample alone must fail b5a; got {block!r}"
    )


# ══ N-1 / N-3 — CLASS: a declaration whose SHAPE is pinned but whose CONTENT is not ═════
def test_run5_is_bound_BY_NAME_and_is_not_freely_exemptable(monkeypatch, tmp_path) -> None:
    """ADJ-13 N-1 and N-3. **Class: every check on the declaration is structural — the two
    tuples partition the tree, every exemption carries a reason — and structure is preserved
    by moving the mint subject from one side to the other.**

    Nothing anywhere pinned `configs/run5.yaml ∈ PRODUCTION_CONFIGS`. Moving it to
    `EXEMPT_CONFIGS` with a written reason and promoting an armed smoke config in its place
    keeps the partition exact, keeps every reason non-blank, and yields **gate 12 rc 0 with
    run5 disarmed** — the run the operator is about to mint, unaudited, with every existing
    assertion satisfied. The second half below drives exactly that swap so the literal pin
    above is a measurement rather than a restatement.

    N-3's half: with `configs/dev_example.yaml` now EXEMPT, gate 12's ability to go red on the
    REAL `configs/` tree rests on run5's membership here — the `--config` route is F-5's. Both
    are now pinned, so neither is the sole witness.
    """
    exempt = {rel for rel, _reason in EXEMPT_CONFIGS}
    assert "configs/run5.yaml" in PRODUCTION_CONFIGS, (
        "the config the operator is about to mint must be bound BY NAME — absence from this "
        f"tuple is not a red gate, it is silence. got {PRODUCTION_CONFIGS}"
    )
    assert "configs/run5.yaml" not in exempt

    # …and the escape the pin exists to refuse, driven.
    root = _mini_tree(tmp_path)
    production = root / "configs" / "run5.yaml"
    production.write_text(production.read_text().replace("actor_lag_abort_enabled: true",
                                                         "actor_lag_abort_enabled: false"))
    bare = _mini_audit(root)
    assert bare.returncode == 30, (
        "a disarmed run5 must fail gate 12 with NO --config in sight — this is the whole of "
        f"gate 12's red-capability on the real tree (N-3); got {bare.returncode}\n"
        f"{(bare.stdout + bare.stderr)[-2000:]}"
    )

    # The swap, exactly as an unwitting editor would write it: run5 moves to EXEMPT with a
    # written reason, and an ARMED smoke config takes its place on the production side.
    smoke = root / "configs" / "smoke_gnn.yaml"
    # WPAX Phase D: "an ARMED smoke config" now means armed on BOTH required rows — the
    # draw-rate row joined the manifest, and a smoke config ships it `null` (R59). Arming it
    # here keeps the ESCAPE this test demonstrates intact: if the promoted config were
    # disarmed on either row the swap would fail the audit for its OWN reason and the escape
    # would look closed by something other than the by-name pin. `min_step` is 1 because this
    # config's `max_train_steps` is 2000 and the twin cross-validator binds it inside the run.
    smoke.write_text(smoke.read_text()
                     .replace("actor_lag_abort_enabled: false",
                              "actor_lag_abort_enabled: true")
                     .replace("draw_rate_abort: null",
                              "draw_rate_abort:\n"
                              "    threshold: 0.25\n"
                              "    min_step: 1\n"
                              "    N_pool_min: 50\n"
                              "    consec: 3"))
    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS", ("configs/smoke_gnn.yaml",))
    monkeypatch.setattr(TOOL, "EXEMPT_CONFIGS",
                        (*[row for row in EXEMPT_CONFIGS if row[0] != "configs/smoke_gnn.yaml"],
                         ("configs/run5.yaml", "moved with a written reason")))
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    assert TOOL._config_declaration_drift() == ([], [], []), (
        "the swap keeps the partition EXACT — which is why no structural check catches it"
    )
    assert all(reason.strip() for _rel, reason in TOOL.EXEMPT_CONFIGS), (
        "…and every exemption still carries a written reason, so that check does not catch "
        "it either"
    )
    TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))  # green, run5 disarmed on disk
    assert "actor_lag_abort_enabled: false" in production.read_text(), (
        "THE ESCAPE: assertion (c) just passed while the config the operator is minting sits "
        "on disk with its hard abort off. The only thing standing between that state and this "
        "tree is the by-name pin at the top of this test."
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# R72 — "every conjunct of every shipped predicate appears in some flip-set"
#
# The ADJ-13 enumeration mutated all 68 conjuncts of the tool's and the manifest's shipped
# predicates one at a time against the full default tier. F-5 and F-6 were the two RED-TEAM
# named; the ten below were found by that enumeration and had no flip-set row of any kind.
# Each block states which conjunct it is the first witness to, because that is the only thing
# that makes it worth its line count.
# ══════════════════════════════════════════════════════════════════════════════════════

#: (events, expected source, expected value). The witness ladder in `_step_ground_truth`, in
#: its shipped precedence order. Only the FIRST rung had a producer.
_GROUND_TRUTH_LADDER = (
    ("shutdown_save", [{"event": "shutdown_save", "step": 101, "ts": 1.0}], "shutdown_save",
     101),
    ("terminal_eval", [{"event": "terminal_eval", "step": 77, "ts": 1.0}], "terminal_eval", 77),
    ("samples_only", [], "actor_lag_sample", 60),
    ("nothing", [], "absent", 0),
)


@pytest.mark.parametrize("name,events,source,value", _GROUND_TRUTH_LADDER,
                         ids=[row[0] for row in _GROUND_TRUTH_LADDER])
def test_every_rung_of_the_step_ground_truth_LADDER_is_a_live_witness(
    name, events, source, value,
) -> None:
    """R72 gap (G2, G3). **Conjuncts: `_step_ground_truth`'s `terminal_eval` and
    `actor_lag_sample` arms — both deletable with the full default tier green (1826 passed).**

    §7.3's whole argument is that there is no per-step event in a 101-step burst
    (`log_interval=1000`), so N comes from an INDEPENDENT witness and the report NAMES which
    one spoke. Every corpus row carries a `shutdown_save`, so only the first rung was ever
    exercised: a run that dies before close-out — which is every run this preflight is
    designed to catch — falls to the second or third rung, and neither had a producer.

    `a1`/`a2` are computed against `n` from this ladder, so a wrong rung is a wrong N and a
    wrong `expected` set: the sync assertion silently measures against the wrong burst length
    rather than failing. The `absent` rung matters for the same reason in the other direction
    — `n = 0` must make `a_sync` fail loudly (rc 22) rather than pass vacuously.
    """
    samples = _model_samples(((30, 20), (60, 50))) if name == "samples_only" else []
    ground = TOOL._step_ground_truth([*events, *samples], samples)
    assert ground == {"source": source, "value": value}, (
        f"rung {name!r} must be reached and NAMED in the report; got {ground!r}"
    )


def test_the_absent_ground_truth_rung_fails_the_burst_rather_than_passing_it() -> None:
    """The `absent` rung's consequence, which is the only reason it is not simply `0`.

    With no witness at all `n = 0`, and `n != burst_steps` must make the a-side fail by name.
    A rung that returned the requested burst length instead would turn "we could not measure
    the burst" into "the burst completed", which is the class this whole gate exists to refuse.
    """
    block = TOOL.evaluate_assertions([], cadence_steps=1, burst_steps=_N,
                                     poll_interval_sec=_P)["a_sync"]
    assert block["step_ground_truth"] == {"source": "absent", "value": 0}
    assert block["failure"] == "PreflightBurstIncompleteError", (
        f"an unmeasurable burst is rc 22, never a pass; got {block!r}"
    )


def test_b2_fails_a_FROZEN_LEARNER_and_is_not_satisfied_by_a_constant(tmp_path) -> None:
    """R72 gap (L4). **Conjunct: `b2`'s `max(learners) > min(learners)` — no corpus row flips
    b2 at all, so the whole predicate was deletable with the tier green.**

    b3 (the frozen ACTOR) has M3 and M8; b2 is its mirror on the learner side and had nothing.
    A frozen learner is not hypothetical: it is what a stuck training loop looks like to the
    watchdog while the actor goes on being resynced, and without b2 the lag block reports a
    healthy transport over a learner that never moved.
    """
    syncs = _real_syncs(tmp_path, "b2frozen", [30, 50])
    samples = _model_samples(((50, 0), (50, 30), (50, 50)))
    block = _assertions(sorted([*syncs, *samples], key=lambda e: float(e["ts"])))["b_lag"]
    assert block["b2"] is False, f"a learner that never moves must fail b2; got {block!r}"
    assert block["failure"] == "PreflightLagFrozenError" and block["sub_reason"] == "learner", (
        f"…and must be diagnosed as the LEARNER side, not the actor's; got {block!r}"
    )
    assert block["b1"] is True, "the arithmetic is self-consistent — only the learner froze"


def test_b2s_positivity_conjunct_is_load_bearing_and_not_decoration(tmp_path) -> None:
    """R72 gap (L5). **Conjunct: `b2`'s `max(learners) >= 1`.**

    Stated honestly, because a flip-set row that cannot be reached is worse than none: for
    NON-NEGATIVE learner steps this conjunct is implied by `max > min`, so no physically
    producible stream isolates it. It is a floor against a learner counter that is not a step
    count at all — a sentinel, an uninitialised negative, a sign-flipped delta — and the flip
    row is the stream that has one. Without this row the conjunct is deletable in one
    character with the whole tier green, which is what the enumeration measured.
    """
    syncs = _real_syncs(tmp_path, "b2pos", [30])
    samples = _model_samples(((-3, -9), (-1, -5)))
    assert [sample["lag_steps"] for sample in samples] == [6, 4], (
        f"the rig's arithmetic must stay self-consistent so b1 holds; got {samples!r}"
    )
    block = _assertions(sorted([*syncs, *samples], key=lambda e: float(e["ts"])))["b_lag"]
    assert block["b1"] is True and block["b2"] is False, (
        "a learner counter that moves but never reaches 1 is not a step count; b2's "
        f"positivity floor is the only witness. got {block!r}"
    )
    assert block["failure"] == "PreflightLagFrozenError"


def test_a_config_in_BOTH_tuples_fails_the_gate(monkeypatch, tmp_path) -> None:
    """R72 gap (C6). **Conjunct: `_config_declaration_drift`'s `overlapping` term.**

    The partition's third arm — a config named by `PRODUCTION_CONFIGS` AND `EXEMPT_CONFIGS`,
    i.e. two answers to one question — was deletable with the tier green. `undeclared` and
    `stale` both had producers; the arm that catches "audited AND excused" did not, and it is
    the arm an editor trips by adding a row without removing the other.
    """
    root = _mini_tree(tmp_path)
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    monkeypatch.setattr(TOOL, "EXEMPT_CONFIGS",
                        (*EXEMPT_CONFIGS, ("configs/run5.yaml", "excused as well as audited")))
    assert TOOL._config_declaration_drift()[2] == ["configs/run5.yaml"], (
        "a config in both tuples must be reported as OVERLAPPING"
    )
    with pytest.raises(TOOL.PreflightManifestError) as caught:
        TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert caught.value.rc == 31 and "IN BOTH TUPLES" in str(caught.value), (
        f"…and it must fail the GATE by name, not merely the helper; got {caught.value!s}"
    )


#: (value, armed, why). The threshold mechanism's type guard, which decides whether a value
#: is a threshold at all before it decides whether the threshold is positive.
_THRESHOLD_TYPE_ROWS = (
    (True, False, "a bool is not a threshold — `float(True) > 0.0` is True"),
    (False, False, "…and neither is the other bool"),
    ("0.35", False, "a string that looks like a threshold is not one"),
    (None, False, "an absent value never arms (LAW-11's shape)"),
    (0.35, True, "…while a real positive threshold does"),
    (0.0, False, "…and a zero one does not"),
)


@pytest.mark.parametrize("value,armed,why", _THRESHOLD_TYPE_ROWS,
                         ids=[repr(row[0]) for row in _THRESHOLD_TYPE_ROWS])
def test_the_threshold_mechanisms_TYPE_guard_is_a_real_predicate(value, armed, why) -> None:
    """R72 gap (A2, A3). **Conjuncts: `Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed`'s
    `isinstance(value, bool)` exclusion and its `not isinstance(value, (int, float))`
    exclusion.** Both deletable with the full tier green.

    The frozen manifest oracle drives `is_armed(0.15) / (0.0) / (-1.0)` — the numeric axis
    only. Dropping the bool exclusion makes `is_armed(True)` return **True**, so R65's Phase D
    could wire `draw_rate_threshold` to a boolean arming flag and the manifest would report the
    row ARMED without a threshold existing. Dropping the numeric-type exclusion makes a string
    raise `ValueError` inside the audit, which `main`'s generic handler collapses to an unnamed
    rc 1 — the one outcome the tool's own docstring says cannot happen.

    Both directions are here, so the guard cannot be satisfied by returning a constant.
    """
    from mantis.config.armed_aborts import Mechanism

    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(value) is armed, why


def test_the_relaunch_budget_code_is_refused_BY_NAME_and_not_as_a_generic_boot_failure(
) -> None:
    """R72 gap (CC4). **Conjunct: `_classify_child`'s `rc == RELAUNCH_BUDGET_CODE` arm.**

    Deleting the arm leaves rc 44 falling through to the final `PreflightBootFailedError`, so
    the exception TYPE is unchanged and every type-level assertion stays green — which is
    exactly why it had no producer. What is lost is the diagnosis: 44 is the supervisor's
    `RELAUNCH_BUDGET_EXIT_CODE`, reserved by the run's own machinery, and a preflight child
    that produced it means something is wired to a supervisor the preflight never started.
    The assertion is therefore on the MESSAGE, which is the only observable that differs.
    """
    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._classify_child(_child(TOOL.RELAUNCH_BUDGET_CODE))
    assert "RELAUNCH_BUDGET_EXIT_CODE" in str(caught.value), (
        "rc 44 must be diagnosed as the reserved supervisor code, not reported as a generic "
        f"boot failure; got {caught.value!s}"
    )
    assert TOOL.RELAUNCH_BUDGET_CODE not in TOOL.PASS_THROUGH, (
        "…and it must not be inside the pass-through range, or the arm is unreachable"
    )


def test_both_arms_of_the_config_path_resolver_are_live(monkeypatch, tmp_path) -> None:
    """R72 gap (RS1, RS2). **Conjuncts: `_resolve_config_path`'s cwd-relative arm and its
    `REPO_ROOT` fallback arm.** Each was deletable with the full tier green, because every
    existing caller runs with `cwd == REPO_ROOT`, where the two arms return the same path.

    They stop agreeing the moment an operator preflights from anywhere else — which is the
    normal case for the MANUAL mint gate, since `--out-dir` must be outside the repo. The
    rows below are driven from a cwd that is NOT the repo, so each arm is the only one that
    can answer.
    """
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "local.yaml"
    local.write_text(RUN5.read_text())
    assert TOOL._resolve_config_path("local.yaml") == local.resolve(), (
        "the cwd-relative arm: a config beside the operator, which REPO_ROOT cannot find"
    )
    assert TOOL._resolve_config_path("configs/run5.yaml") == RUN5.resolve(), (
        "the REPO_ROOT fallback arm: a repo-relative path from a foreign cwd, which the "
        "cwd-relative arm cannot find"
    )
    with pytest.raises(TOOL.PreflightConfigError) as caught:
        TOOL._resolve_config_path("nope.yaml")
    assert caught.value.rc == 10, "…and neither arm matching is a NAMED refusal"


# ══ R72 CLOSING PASS — the ten conjuncts CARD-R72-EVIDENCE-STRING-FALLBACKS named ═══════
#
# CLASS (R71), stated at the level the ten actually share: **an evidence-report field whose
# value is a FALLBACK — a constant, or a branch nothing drives — so the report can assert
# something the run never measured.** That is ADJ-13 F-3's species in the same artefact, and
# the corrective pass's own re-verification proved the ten survive every mutation at the full
# default tier. The rows below are the first producers of all of them.
#
# The boundary is NOT "does the happy path print the reason". It is the COMPLEMENT: every
# posture in which the fallback fires, driven so the fallback's own sentence is checked
# against the report's own measurement of the same run.


#: `_git_toplevel` believes git only when it BOTH answered (rc 0) AND said something
#: (`stdout.strip()`). The two conjuncts short-circuit, so the only inputs that tell them
#: apart are the OFF-DIAGONAL ones — rc 0 with empty output, and a nonzero rc WITH output.
#: A test that only drives "git works" / "no repo here" cannot distinguish either conjunct
#: from `True`, which is why both were UNCOVERED. The shim is a REAL `git` on `PATH`, so a
#: real `subprocess.run` is driven and nothing inside the tool is patched (R64).
_GIT_POSTURES = (
    ("rc 0 WITH a toplevel — the only posture git is believed in", 0, "/shimmed/top\n", True),
    ("rc 0 and SILENT — answered, said nothing", 0, "", False),
    ("rc 0 and WHITESPACE — `.strip()`, not truthiness, is the predicate", 0, "  \n", False),
    ("rc 1 WITH output — the row that tells `returncode == 0` from `True`", 1, "/shimmed/top\n",
     False),
    ("rc 1 and silent — not a repo", 1, "", False),
)


@pytest.mark.parametrize(("why", "rc", "stdout", "believed"), _GIT_POSTURES,
                         ids=[posture[0].split(" —")[0] for posture in _GIT_POSTURES])
def test_git_is_believed_ONLY_when_it_BOTH_answered_AND_named_a_toplevel(
    monkeypatch, tmp_path, why, rc, stdout, believed,
) -> None:
    """R72 closing pass. **Conjuncts: `_git_toplevel`'s `result.returncode == 0` and
    `result.stdout.strip()`** (the recheck's X3/X4, UNCOVERED under R72-C).

    `_git_toplevel` decides what counts as "inside the working tree" for
    `_checked_out_dir` — the guard that stops gate 12 writing JSONL into the repo it gates
    (ADJ-13 F-2, rc 13). Both conjuncts were replaceable by `True` with the full tier green,
    because every existing drive runs inside a healthy checkout where the two agree.

    The fallback is `REPO_ROOT`, and it is CORRECT rather than merely unpinned: `REPO_ROOT`
    is `Path(os.path.abspath(__file__)).resolve().parents[2]` (`preflight_mint.py:107`), so
    both branches return a `.resolve()`d path and F-2's two-normalisation-schemes defect
    cannot come back through the fallback. What was missing was any row that reached it.
    """
    root = tmp_path / "root"
    root.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "git"
    shim.write_text(f"#!{sys.executable}\nimport sys\n"
                    f"sys.stdout.write({stdout!r})\nraise SystemExit({rc})\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(TOOL, "REPO_ROOT", root.resolve())
    expected = Path(stdout.strip()).resolve() if believed else root.resolve()
    assert TOOL._git_toplevel() == expected, (
        f"{why}: git must be believed IFF it answered rc 0 AND named something; a conjunct "
        f"replaced by `True` here returns {'the shim path' if not believed else 'REPO_ROOT'} "
        f"and the repo-containment guard moves with it. got {TOOL._git_toplevel()}"
    )


#: The retired constant. It is asserted ABSENT from every message below, because the defect
#: this block closes is that it was printed in a posture where it is FALSE.
_RETIRED_WATCHDOG_CONSTANT = "reason not found in the segment"

#: `_watchdog_reason`'s three arms, each named by what the RUN did rather than by intent.
_WATCHDOG_POSTURES = (
    ("the run read a reason", {"fired_reason": "actor_lag_exceeded",
                               "segments_scanned": ["events_run5_seg0000.jsonl"]},
     "actor_lag_exceeded"),
    ("no scan is recorded on the block at all", {}, "no segment scan is recorded"),
    ("the scan RAN and read nothing", {"segments_scanned": []}, "NO segment was read"),
    ("the scan read segments and none carried a reason",
     {"segments_scanned": ["a.jsonl", "b.jsonl"]}, "2 segment(s) were read"),
)


@pytest.mark.parametrize(("why", "extra", "expected"), _WATCHDOG_POSTURES,
                         ids=[posture[0] for posture in _WATCHDOG_POSTURES])
@pytest.mark.parametrize("rc", TOOL.WATCHDOG_CODES)
def test_the_watchdog_reason_is_DERIVED_from_what_the_run_actually_READ(
    why, extra, expected, rc,
) -> None:
    """R72 closing pass. **Conjuncts: the two leaves of
    `child.get('fired_reason') or 'reason not found in the segment'`, and the three arms of
    the `_watchdog_reason` that REPLACES them.**

    This is F-3's class a second time in the same report, and it is a WRONG string, not
    merely an unpinned one. Producer for the wrongness — the shipped `_run_preflight` past a
    real `_run_child`, one watchdog rc, three postures
    (`scratchpad/r72c/probe_fired.py`):

        segment carries the event + reason -> "(actor_lag_exceeded)"
        segment read, event NOT in it      -> "(reason not found in the segment)"   TRUE
        `out_dir/logs` never written       -> "(reason not found in the segment)"   FALSE

    In the third the SAME report's `events` block reads `segments: [], lines: 0,
    sha256: null` — no segment was read, so nothing could be "not found in" one. Two
    different facts printed the same sentence and one of them was false, which is exactly
    what F-3 was.

    The fix is F-3's own shape: the sentence is computed from `child["segments_scanned"]`,
    the list the post-child scan publishes of what it actually read, so the sentence and the
    `events` block are two views of one measurement and cannot disagree. A constant in ANY
    of the three arms collapses two postures into one and fails a row here.

    Driven at every code in `WATCHDOG_CODES` so the arm is not pinned to one rc.
    """
    child = {"rc": rc, "timed_out": False, "stderr_tail": "", "stdout_tail": "", **extra}
    with pytest.raises(TOOL.PreflightWatchdogFiredError) as caught:
        TOOL._classify_child(child)
    message = str(caught.value)
    assert expected in message, (
        f"{why}: the parenthetical must be DERIVED from the run's own scan record "
        f"{child.get('segments_scanned')!r} / reason {child.get('fired_reason')!r}; a "
        f"constant makes the evidence report assert a search that may never have happened. "
        f"got {message!r}"
    )
    assert _RETIRED_WATCHDOG_CONSTANT not in message, (
        "…and the retired constant must never come back: it claims a segment was SEARCHED, "
        f"which is false whenever nothing was read. got {message!r}"
    )


#: The three postures of the post-child scan, as a real child process. `mode` is handed to
#: the stub child, which writes the segment (or does not) and then exits with a watchdog rc.
_SCAN_CHILD = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "out, mode = Path(sys.argv[1]), sys.argv[2]\n"
    "if mode != 'nologs':\n"
    "    logs = out / 'logs'\n"
    "    logs.mkdir(parents=True, exist_ok=True)\n"
    "    rows = ([{'event': 'heartbeat_watchdog_fired', 'reason': 'actor_lag_exceeded',\n"
    "              'code': 42}] if mode == 'fired' else [])\n"
    "    rows.append({'event': 'actor_lag_sample', 'learner_step': 1,\n"
    "                 'actor_ckpt_step': 0, 'lag_steps': 1})\n"
    "    (logs / 'events_run5_seg0000.jsonl').write_text(\n"
    "        ''.join(json.dumps(row) + chr(10) for row in rows))\n"
    "raise SystemExit(42)\n"
)

_SCAN_POSTURES = (
    ("fired", "actor_lag_exceeded", 1),
    ("quiet", "1 segment(s) were read", 1),
    ("nologs", "NO segment was read", 0),
)


@pytest.mark.parametrize(("mode", "expected", "segments"), _SCAN_POSTURES,
                         ids=[posture[0] for posture in _SCAN_POSTURES])
def test_the_POST_CHILD_segment_scan_is_driven_and_agrees_with_the_reports_OWN_events_block(
    monkeypatch, tmp_path, mode, expected, segments,
) -> None:
    """R72 closing pass. **Conjuncts: `_run_preflight`'s `log_dir.is_dir()` and its
    `event.get("event") == "heartbeat_watchdog_fired"` comprehension filter.**

    These are the two the corrective pass could not close, and the reason it could not is
    stated in this file's own header: **no default-tier test drives a preflight past
    `_run_child`**, because the shipped `_child_argv` re-execs the tool as a full `--_boot`
    child (torch, `init_trainer`, `WorkerPool`, `compose_run`) and every such child dies at
    TD-4 long before a segment exists. So the whole post-child half — the scan, the fired
    filter, and the reason it feeds — had no producer at all.

    They are reached HERE the way `test_the_reports_raised_by_field_records_WHICH_SIDE…`
    already reaches `_run_child`: `_child_argv` is redirected on the TOOL MODULE OBJECT and
    nothing else is. `_run_child` is the real one — a real `Popen`, a real join, a real rc —
    and the stub child is a real process that writes a real JSONL segment through the real
    filename convention. `_run_preflight` itself is unmodified and unaware: it does its own
    `_resolve_config_path`, `_load`, `_audit_manifest_and_configs` and
    `_apply_burst_override` on the shipping `configs/run5.yaml`.

    The assertion is the BICONDITIONAL, not the sentence alone: the parenthetical and
    `report["events"]["segments"]` are two views of one scan, so a mutation that changes what
    the scan reads must change both, and a mutation that changes only the sentence is caught
    by the disagreement. `log_dir.is_dir()` forced to `False` turns the `fired` row into
    "NO segment was read" while its segment sits on disk; the fired filter forced to `False`
    turns it into "1 segment(s) were read and none of them carries a reason" while the event
    is in the file. Each is a report contradicting the tree it just measured.

    The `fired` segment deliberately carries a TRAILING non-watchdog event, so the filter
    forced to `True` also fails: `fired[-1]` becomes the trailing row, which has no `reason`.
    """
    out_dir = tmp_path / "out"
    monkeypatch.setattr(TOOL, "_child_argv",
                        lambda args: [sys.executable, "-c", _SCAN_CHILD, str(out_dir), mode])
    report = TOOL._new_report("preflight")
    args = SimpleNamespace(config=str(RUN5), burst_steps=_RUN5_BURST, out_dir=str(out_dir),
                           timeout_sec=120.0, device="cpu")
    with pytest.raises(TOOL.PreflightWatchdogFiredError) as caught:
        TOOL._run_preflight(args, report, out_dir)
    message = str(caught.value)
    assert report["child"]["rc"] == 42, (
        "the rig is only a witness if a REAL child really ran and this really is the "
        f"post-child half; got child={report['child']!r}"
    )
    assert expected in message, (
        f"posture {mode!r}: the watchdog parenthetical must report what the POST-CHILD scan "
        f"read; got {message!r}"
    )
    assert len(report["events"]["segments"]) == segments, (
        f"posture {mode!r}: the report's own events block must record the same scan the "
        f"sentence describes; got {report['events']}"
    )
    assert report["child"]["segments_scanned"] == report["events"]["segments"], (
        "the child block's scan record and the events block are two views of ONE scan — if "
        "they can disagree, the sentence derived from one of them is unfalsifiable. got "
        f"{report['child'].get('segments_scanned')!r} vs {report['events']['segments']!r}"
    )
    assert _RETIRED_WATCHDOG_CONSTANT not in message, (
        f"…and never the retired constant. got {message!r}"
    )


@pytest.mark.parametrize("extra", [{}, {"stderr_tail": ""}, {"stderr_tail": None}],
                         ids=["absent", "empty", "null"])
def test_a_child_with_NO_stderr_tail_reports_an_EMPTY_tail_and_never_a_PLACEHOLDER(
    extra,
) -> None:
    """R72 closing pass. **Conjunct: the `""` fallback in
    `tail = str(child.get("stderr_tail") or "")`.**

    Every existing drive of `_classify_child` supplies a tail, so the fallback had no
    producer and was replaceable by any constant with the full tier green. It is *merely
    unpinned* rather than wrong — `str(None or "")` is `""`, which is the truthful rendering
    of "the child said nothing" — but a constant here writes that constant into the
    operator-facing message of every silent child, and `tail` is ALSO what §6.3a arm 5 sniffs
    for `object has no attribute`, so a non-empty fallback can manufacture or suppress an
    rc-32 tree-defect diagnosis from nothing.

    Asserted on the WHOLE message rather than a substring, because the defect a substring
    check misses is exactly a placeholder appended to it.
    """
    child = {"rc": 1, "timed_out": False, **extra}
    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._classify_child(child)
    assert str(caught.value) == "child exited 1:\n", (
        "a child that produced no stderr must render an EMPTY tail — a fallback constant "
        f"lands verbatim in the report's failure message. got {str(caught.value)!r}"
    )


def test_the_child_block_carries_the_childs_OWN_stdout_AND_stderr_tails(monkeypatch) -> None:
    """R72 closing pass. **Conjunct: the `stdout` leaf of
    `"stdout_tail": (stdout or "")[-4000:]`.**

    `stderr` and both `""` halves are already covered (a mutated `""` makes the slice a
    `TypeError`), but `stdout` itself was replaceable by `False` — i.e. `stdout_tail` pinned
    to `""` — with the full tier green, because the one existing `_run_child` drive uses a
    child that prints nothing. `stdout_tail` is the half of the evidence artefact that
    carries a child's own diagnostics when it exits with a code that does NOT name a reason,
    so a pinned-empty half is an evidence report that silently drops half the evidence.

    Both streams are asserted in one row, so a fix that wires stdout to stderr is red too.
    """
    monkeypatch.setattr(TOOL, "_child_argv", lambda args: [
        sys.executable, "-c",
        "import sys; sys.stdout.write('OUT-MARKER'); sys.stderr.write('ERR-MARKER'); "
        "raise SystemExit(7)"])
    report = TOOL._new_report("preflight")
    child = TOOL._run_child(SimpleNamespace(timeout_sec=60.0), report)
    assert child["rc"] == 7, "the rig is only a witness if the real child really ran"
    assert child["stdout_tail"] == "OUT-MARKER", (
        "the child's OWN stdout must reach the report — a constant here empties half the "
        f"evidence artefact with every type-level assertion still green. got "
        f"{child['stdout_tail']!r}"
    )
    assert child["stderr_tail"] == "ERR-MARKER", (
        f"…and the two streams must not be crossed. got {child['stderr_tail']!r}"
    )


def _lag_blocks(inversion_reason):
    """The `_verdict_exit` input for a failing (b), with (a) passing so (b) is reached."""
    return {"a_sync": {"verdict": "pass", "failure": None},
            "b_lag": {"verdict": "fail", "failure": "PreflightLagFrozenError",
                      "sub_reason": "both", "inversion_reason": inversion_reason}}


def test_the_verdict_message_carries_the_INVERSION_REASON_when_there_is_one() -> None:
    """R72 closing pass. **Conjunct: `block.get('inversion_reason')` in `_verdict_exit`.**

    The inversion axis is MF-3's whole point — rc 23/26 exist because a SWAPPED-OPERAND lag
    wiring can look healthy — and `inversion_reason` is the sentence that tells an operator
    WHICH of the two it was. It was replaceable by `False` (i.e. always dropped) with the
    full tier green, because every existing `_verdict_exit` drive asserts the failure NAME
    and the rc, never the reason text.
    """
    reason = ("actor_sync_cadence_steps == 4: the learner is structurally ahead between "
              "syncs")
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit(_lag_blocks(reason))
    assert reason in str(caught.value), (
        "the inversion reason is the only thing that distinguishes rc 26 'frozen on BOTH "
        "sides' from rc 23 'undiscriminated'; dropped, the operator gets a name and no "
        f"diagnosis. got {str(caught.value)!r}"
    )


def test_a_verdict_with_NO_inversion_reason_ends_cleanly_and_never_prints_a_PLACEHOLDER(
) -> None:
    """R72 closing pass. **Conjunct: the `''` fallback beside it.**

    The inverse arm. `inversion_reason` is `None` on every failure that is not the inversion
    axis (`_lag_block` initialises it to `None` at `preflight_mint.py:486`), which is the
    common case — so the fallback is what MOST failure messages end with, and it had no
    producer. Merely unpinned rather than wrong, but a constant here appends itself to every
    non-inversion failure message in the artefact.

    Asserted as EQUALITY on the whole message, because a trailing placeholder is precisely
    what a substring assertion cannot see.
    """
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit(_lag_blocks(None))
    assert str(caught.value) == "PreflightLagFrozenError sub_reason='both'", (
        "with no inversion reason the message must END at the sub_reason — the `.strip()` "
        f"exists for exactly that. got {str(caught.value)!r}"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# WPMINT Phase B — CARD-D-BURST-FLOOR: the MINT TIER the report publishes
#
# CLASS: **an evidence artifact that reports a run's LENGTH without reporting what that
# length bought.** run5's minimum legal burst is 25001 because arming
# `train.draw_rate_abort` at `min_step: 25000` puts a third row in `_burst_floors`. The floor
# cannot be shrunk (a run5 armed value, mint-prereg-only, R82/R85) and a shorter burst that
# pretended to cover the draw-rate axis is out under R64 — so the honest move is for the
# report to say which tier ran and what that tier does NOT prove.
#
# Every row below drives the SHIPPED functions on the REAL committed configs. Nothing here
# patches the tool, and nothing here estimates a wall-clock: at HEAD no burst of any length
# has ever run (TD-4), which is precisely the fact `covered: []` publishes.
# ══════════════════════════════════════════════════════════════════════════════════════
def _tier_config(path: Path):
    return load_config(path)


def test_every_mint_tier_has_a_NOT_PROVEN_entry_and_there_is_NO_default() -> None:
    """R1 at the tier boundary — `REPORT_MODES`' discipline, one field over.

    `_new_report`'s mode table has no default because "falling back would publish some other
    mode's disclaimer, which is ADJ-13 F-3 itself". A tier disclaimer inherits that verbatim:
    a `.get(tier, <some tier>)` would satisfy every other row in this block while publishing
    the `full` tier's reachability claim under a `sync_lag` run. The fallback IS the defect,
    so there is no fallback.
    """
    tiers = {TOOL.TIER_NONE, TOOL.TIER_SYNC_LAG, TOOL.TIER_FULL}
    assert set(TOOL.TIER_NOT_PROVEN) == tiers, (
        "every tier `_tier_disclaimer` can be called with must declare what it does NOT "
        f"prove; got {sorted(TOOL.TIER_NOT_PROVEN)}"
    )
    assert len(set(TOOL.TIER_NOT_PROVEN.values())) == len(tiers), (
        "two tiers sharing one disclaimer is the overclaim with an extra dict key — one of "
        "them is publishing the other's coverage"
    )
    assert tuple(TOOL.MINT_REQUIRED_TIERS) == (TOOL.TIER_SYNC_LAG, TOOL.TIER_FULL), (
        "the card requires BOTH tiers for a mint; got "
        f"{tuple(TOOL.MINT_REQUIRED_TIERS)}"
    )
    with pytest.raises(TOOL.PreflightInternalError) as caught:
        TOOL._tier_disclaimer({"tier": {"tier": "quick"}})
    assert "no code-side default" in str(caught.value)


def test_the_burst_tier_is_DERIVED_from_the_configs_OWN_floor_rows() -> None:
    """`_burst_tier` reads `_burst_floors`, so the tier and the refusal message an operator
    was shown come off ONE row set. Driven on the real committed configs at three lengths.

    The third assertion is the one that matters and the one a "cleared the max floor"
    implementation gets wrong: on a config whose `train.draw_rate_abort` is absent, ANY burst
    clears every floor the config has — and calling that `full` would claim draw-rate
    reachability on a config that arms no draw-rate abort at all. That is the overclaim this
    whole block exists to prevent, so it is pinned at a burst LONGER than run5's own floor.
    """
    run5 = _tier_config(RUN5)
    minimum = TOOL._minimum_legal_burst(run5)
    assert minimum == _RUN5_BURST, f"run5's floor moved: {minimum}"
    assert TOOL._burst_tier(run5, minimum) == TOOL.TIER_FULL
    assert TOOL._burst_tier(run5, minimum - 1) == TOOL.TIER_NONE, (
        "a burst below the max floor is not a shorter tier — it is a burst the validators "
        "refuse, and no tier ran at all"
    )
    unarmed = [path for path in discover_configs(REPO_ROOT / "configs")
               if _tier_config(path).train.draw_rate_abort is None]
    assert unarmed, "this row is vacuous unless some committed config leaves the row absent"
    for path in unarmed:
        config = _tier_config(path)
        assert TOOL._burst_tier(config, TOOL._minimum_legal_burst(config)) == TOOL.TIER_SYNC_LAG
        assert TOOL._burst_tier(config, _RUN5_BURST) == TOOL.TIER_SYNC_LAG, (
            f"{path.name} declares no {TOOL.DRAW_RATE_FLOOR_KEY} row, so no burst length on "
            "it can reach the draw-rate abort's first firing step. Tier `full` here would be "
            "a reachability claim about an abort this config does not arm"
        )


def test_a_PRODUCTION_config_can_never_be_preflighted_in_the_SHORT_tier() -> None:
    """**The measured ground for deviating from the card's presumptive two-RUN shape.**

    The card offered "a short burst asserts sync/lag/arming-audit, a long tier asserts
    draw-rate live-fire reachability, BOTH required for mint". On a production config the
    short tier is not merely unrun — it is UNREACHABLE, and this is the producer for that
    claim rather than an argument for it. Three links, each driven:

    1. `draw_rate_collapse` is a REQUIRED manifest row, so a production config that disarms
       it fails assertion (c) at rc 30 (`audit_arming` here; the rc-30 drive is F-5's);
    2. an armed row puts `min_step + 1` into `_burst_floors`;
    3. `_apply_burst_override` refuses every burst below the max at rc 11.

    So the ONLY route to tier `sync_lag` on run5 is to disarm the row the mint exists to arm —
    a change to a run5 armed value (R82/R85, hard stop) and a faked axis (R64). The two tiers
    are two COVERAGE claims, not two runs, and `full` covers `sync_lag`.
    """
    assert PRODUCTION_CONFIGS, "vacuous unless something is declared production"
    required = [row.name for row in MANIFEST if row.status is Status.REQUIRED]
    assert "draw_rate_collapse" in required, (
        "link 1: if the draw-rate row stops being REQUIRED, a production config may be minted "
        "with it disarmed and the short tier becomes reachable again"
    )
    for rel in PRODUCTION_CONFIGS:
        config = _tier_config(REPO_ROOT / rel)
        keys = [key for key, _value, _floor in TOOL._burst_floors(config)]
        assert TOOL.DRAW_RATE_FLOOR_KEY in keys, f"link 2 broken for {rel}: {keys}"
        minimum = TOOL._minimum_legal_burst(config)
        assert TOOL._burst_tier(config, minimum) == TOOL.TIER_FULL
        with pytest.raises(TOOL.PreflightBurstTooShortError) as caught:
            TOOL._apply_burst_override(config, minimum - 1)
        assert int(caught.value.rc) == 11 and "MINIMUM legal burst" in str(caught.value), (
            "link 3: the only burst that would tier as `sync_lag` on a production config is "
            "one the config's own cross-field validators refuse"
        )


#: The two answers to "did this run actually cover a tier". Not two tiers — a tier is
#: REQUESTED by the burst and COVERED by the outcome, and at HEAD those never coincide.
_TIER_HISTORIES = ("no_verdict", "verdict")


@pytest.mark.parametrize("history", _TIER_HISTORIES)
def test_a_tier_is_COVERED_only_when_the_run_reached_a_verdict(history) -> None:
    """`_finalise_tier` re-derives coverage from the report's own (a)/(b) verdicts, so the
    coverage claim and the assertion blocks beside it cannot disagree — one is computed from
    the other, exactly as `_finalise_not_run` computes the boot disclaimer from `child`.

    A constant in EITHER direction fails one of these two rows. The `no_verdict` row is the
    HEAD posture: every mode-PREFLIGHT child dies at TD-4 with (a) and (b) still `not_run`,
    so a `covered` that tracked the burst length instead of the outcome would publish full
    mint coverage for a run that never took a step.
    """
    report = TOOL._new_report("preflight")
    report["tier"] = TOOL._tier_block(_tier_config(RUN5), _RUN5_BURST)
    assert report["tier"]["tier"] == TOOL.TIER_FULL
    if history == "verdict":
        for name in ("a_sync", "b_lag"):
            report["assertions"][name] = {"verdict": "pass"}
    TOOL._finalise_tier(report)
    block = report["tier"]
    if history == "verdict":
        assert block["covered"] == [TOOL.TIER_SYNC_LAG, TOOL.TIER_FULL], (
            "`full` clears every floor `sync_lag` clears and one more, so one green `full` "
            f"run discharges both required tiers; got {block['covered']}"
        )
        assert block["owed"] == [] and "still OWED: (none)" in block["does_not_prove"]
    else:
        assert block["covered"] == [], (
            "the burst was long enough for tier `full`, but the run reached no verdict — "
            f"coverage tracks the OUTCOME, not the argument. got {block['covered']}"
        )
        assert block["owed"] == list(TOOL.MINT_REQUIRED_TIERS), (
            "a tier that could not be run today must say so and stay OWED, never report "
            f"`not_run` as though it were optional. got {block['owed']}"
        )
        assert "NOTHING in this tier is demonstrated" in block["does_not_prove"]


def test_the_tier_disclaimer_is_RE_DERIVED_at_write_time_and_never_the_prediction(
        tmp_path) -> None:
    """The tier block stamped by `_new_report` is a PREDICTION (`none`, before any burst was
    accepted). `_write_report` calls `_finalise_tier` beside `_finalise_not_run` so the
    prediction is replaced by the measurement for every write path there will ever be.

    Driven through the real writer, asserted against the bytes ON DISK — a re-derivation that
    happens only at the call site is one a second caller can forget, and the artifact is the
    only thing a mint sign-off reads.
    """
    report = TOOL._new_report("preflight")
    assert report["tier"]["tier"] == TOOL.TIER_NONE and report["tier"]["burst_steps"] is None
    stale = report["tier"]["does_not_prove"]
    report["tier"] = TOOL._tier_block(_tier_config(RUN5), _RUN5_BURST)
    assert report["tier"]["does_not_prove"] is None, (
        "`_tier_block` must not compose the disclaimer — the run has not happened yet"
    )
    TOOL._write_report(tmp_path, report)
    written = json.loads(next(iter(tmp_path.glob("preflight_*.json"))).read_text())
    assert written["tier"]["does_not_prove"] != stale
    assert written["tier"]["does_not_prove"].startswith(f"tier={TOOL.TIER_FULL} ")
    assert TOOL.TIER_NOT_PROVEN[TOOL.TIER_FULL] in written["tier"]["does_not_prove"]
    assert written["tier"]["floors"] == [
        {"key": key, "value": value, "floor": floor, "cleared": True}
        for key, value, floor in TOOL._burst_floors(_tier_config(RUN5))
    ], "the block must name WHICH rule made the tier what it is, row by row"


def test_the_none_tier_disclaimer_is_TRUE_in_mode_AUDIT_and_not_only_at_rc_11() -> None:
    """ADJ-13 F-3, caught inside this card's own first draft and pinned so it stays caught.

    The `none` disclaimer was first written as *"no burst SURVIVED the config's own
    cross-field validators"*. Measured false on the very next drive: mode AUDIT requests no
    burst, so nothing was refused — the sentence told an AUDIT reader their burst had been
    rejected. Same class as the `not_run` disclaimer that was keyed on `mode`, committed
    inside the fix for it. The wording is now pinned to the FIELD that records the fact.
    """
    for mode in TOOL.REPORT_MODES:
        sentence = TOOL._new_report(mode)["tier"]["does_not_prove"]
        assert "tier.burst_steps` is null" in sentence, (
            "the `none` disclaimer must point at the field that records the fact rather than "
            f"assert a story about how the run got there; mode={mode} got {sentence!r}"
        )
        assert "survived" not in sentence, (
            "the measured-false draft, verbatim: it asserts a refusal that never happened in "
            "mode AUDIT"
        )


def test_a_refused_burst_publishes_tier_none_and_owes_BOTH_tiers(tmp_path) -> None:
    """The rc-11 posture, driven through the REAL tool as a process.

    A burst the validators refuse is not a shorter tier, and the artifact must not read as
    though some coverage was obtained. Pinned end-to-end because the tier is stamped in
    `_run_preflight` AFTER `_apply_burst_override` returns — a stamp one line earlier would
    publish `tier: sync_lag` for a run that never started.
    """
    out_dir = tmp_path / "refused"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_RUN5_BURST - 1),
                       "--out-dir", str(out_dir), "--timeout-sec", "60", "--device", "cpu")
    assert result.returncode == 11, (result.stdout + result.stderr)[-2000:]
    report = json.loads(next(iter(out_dir.glob("preflight_*.json"))).read_text())
    assert report["child"] is None and report["override"] is None
    assert report["tier"]["tier"] == TOOL.TIER_NONE
    assert report["tier"]["burst_steps"] is None and report["tier"]["floors"] is None
    assert report["tier"]["owed"] == list(TOOL.MINT_REQUIRED_TIERS)
    assert report["tier"]["does_not_prove"] in result.stdout, (
        "the sentence printed to the operator and the sentence on disk must be the same "
        "bytes — a second composition site is a second thing that can disagree with the "
        "artifact"
    )


@pytest.mark.integration
def test_the_real_preflight_publishes_the_tier_it_RAN_and_what_it_does_NOT_prove(
        tmp_path) -> None:
    """The card's headline requirement, measured on the real tool at run5's real floor.

    This is `test_the_real_boot_terminates_where_the_docstring_says`'s twin: that row pins
    WHERE the boot stops, this one pins what the artifact then CLAIMS. At HEAD the child dies
    at TD-4, so the honest report is tier `full` REQUESTED, `covered: []`, both tiers still
    OWED — and when CARD-POOL-ENCODING-BRIDGE lands this row is what tells the next reader
    that a green burst may finally cover something.
    """
    out_dir = tmp_path / "tiered"
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "240", "--device", "cpu")
    assert result.returncode == 33, (result.stdout + result.stderr)[-3000:]
    report = json.loads(next(iter(out_dir.glob("preflight_*.json"))).read_text())
    block = report["tier"]
    assert block["tier"] == TOOL.TIER_FULL and block["burst_steps"] == _RUN5_BURST
    assert all(row["cleared"] for row in block["floors"])
    assert block["covered"] == [] and block["owed"] == list(TOOL.MINT_REQUIRED_TIERS), (
        "the burst cleared every floor, but the run never took a step — the report must not "
        f"read as coverage. got covered={block['covered']} owed={block['owed']}"
    )
    assert "REACHABILITY and nothing else" in block["does_not_prove"], (
        "even a COMPLETED `full` burst proves the run reaches the abort's first firing step "
        "and nothing more; the artifact has to say so"
    )
    assert "NOTHING in this tier is demonstrated" in block["does_not_prove"]
