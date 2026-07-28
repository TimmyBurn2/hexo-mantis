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

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.config.armed_aborts import EXEMPT_CONFIGS, MANIFEST, PRODUCTION_CONFIGS
from mantis.config.loader import (
    CONFIG_SUFFIXES,
    ConfigSuffixError,
    discover_configs,
    is_config_path,
    load_config,
)
from mantis.monitor.sink import JsonlEventSink
from mantis.train.actor_sync import ActorSync
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"

#: run5's own constants, read from the file rather than restated (§14 item 17 / ADJ-12).
RUN5 = REPO_ROOT / "configs" / "run5.yaml"
_N = 101


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
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_N),
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

    §8.4 calls this scan "the forcing function that makes Phase D's flip unforgettable": when
    R65's Phase D deletes `draw_rate_threshold: float = 0.0`, gate 12 must go RED. This test
    is that claim, driven end to end — delete the pinned literal from the tree and the GATE
    fails, not merely a helper function.
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
    monkeypatch.setattr(TOOL, "MANIFEST",
                        tuple(row for row in MANIFEST if row.status.value == "deferred"))
    with pytest.raises(TOOL.PreflightManifestError) as caught:
        TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert "vacuous" in str(caught.value), (
        "a manifest with no REQUIRED row audits every config green — the other half of the "
        f"guard; got {caught.value!s}"
    )


@pytest.fixture(scope="module")
def audit_stdout() -> str:
    """One `--audit-only` drive, shared by the four field pins below."""
    result = _run_tool("--audit-only")
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return result.stdout


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
    deferred = [row for row in MANIFEST if row.status.value == "deferred"]
    assert deferred, "no deferred row means this test has no subject"
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


def test_the_report_publishes_the_audits_own_deferred_and_required_rows(tmp_path) -> None:
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
    ], f"the report's deferred block must be the audit's; got {manifest['deferred']!r}"
    assert all(row["note"] and row["owner"] and row["source_pin"]
               for row in manifest["deferred"])
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
    assert set(TOOL.FAILURE_CODES.values()).isdisjoint({42, 43, 44, 45}), (
        "the four codes the run's OWN machinery reserves must never be an assertion outcome"
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
_F1_PLANT_PATHS = ("run6.yaml", "run6.yml", "prod/run6.yaml", "prod/nested/run6.yml")


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


#: **The class boundary the first two fixes both missed.** MF-7's fix closed the reviewer's
#: `configs/run6.yaml`; the ADJ-13 fix closed `run6.yml` and `configs/prod/`; the recheck walked
#: `configs/run6.txt` and `configs/run6.YAML` straight through both gates. The escape recurred
#: because the fixes kept widening ONE side: discovery filtered by extension while `load_config`
#: was `yaml.load(Path(path).read_text())` — no filter at all — so the LAUNCHABLE set stayed
#: strictly larger than the DISCOVERED set and the complement of every enumeration was the next
#: exploit. Rows below are the complement, not another enumeration: a plain unknown suffix, a
#: CASE variant of a known one, no suffix at all, and a known suffix that is not final.
_F1_UNRECOGNISED = ("run6.txt", "run6.YAML", "run6", "run6.yaml.bak", "run6.YML", "run6.yamL")


@pytest.mark.parametrize("rel", _F1_UNRECOGNISED)
def test_a_config_shaped_file_at_an_UNRECOGNISED_suffix_is_not_a_config_ANYWHERE(
    tmp_path, monkeypatch, rel,
) -> None:
    """R71's novel-extension row, driven at the boundary rather than at a demo input.

    Each planted file is a **byte-for-byte copy of run5 with the one REQUIRED armed-abort row
    disarmed** — so if anything in the repo will read it, it is a production config with the
    actor-lag hard abort off, which is precisely MF-7's hazard. Measured before this pass, on
    `configs/run6.txt`: schema-valid, `audit_arming` reporting `actor_lag` DISARMED, gate 7
    **rc 0**, gate 12 **rc 0**, mintable via `mint_config.py --out`, launchable via
    `python -m mantis.run <path>`.

    The fix is not a wider enumeration — it is that `load_config` now REFUSES what discovery
    rejects, from the one predicate both call. So the assertion is the whole biconditional: the
    gates' silence about this file is CORRECT exactly because nothing can read it.
    """
    root = _mini_tree(tmp_path)
    planted = _plant_disarmed(root, rel)
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)

    assert not is_config_path(planted), f"{rel} must not be a config to the one predicate"
    with pytest.raises(ConfigSuffixError):
        load_config(planted)
    assert planted.relative_to(root).as_posix() not in TOOL._discovered_configs(), (
        f"{rel} is not loadable, so it must not be enumerated either — discovery and the "
        "loader are the same predicate or they are two authorities again"
    )
    undeclared, _stale, _overlapping = TOOL._config_declaration_drift()
    assert planted.relative_to(root).as_posix() not in undeclared, (
        "…and the exclusion must be SILENT, not a false UNDECLARED red. That silence is only "
        "defensible because the loader refuses the file, which the two assertions above "
        f"measure; got {undeclared}"
    )
    result = _run_tool("--audit-only", cwd=root,
                       tool=root / "tools" / "ci_gates" / "preflight_mint.py")
    assert result.returncode == 0, (
        "gate 12 must stay green over a file nothing can read; got "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-2000:]}"
    )


def test_the_LAUNCH_route_refuses_what_the_gates_cannot_see(tmp_path) -> None:
    """The third of the three facts that made `configs/run6.txt` a real hazard rather than a
    contrived one: `src/mantis/run.py:254` calls `load_config(argv[0])` on a FREE path, so a
    file no gate enumerates was one command away from being the config a run booted from.

    Driven through the real entry point as a process, because that is the surface an operator
    actually types. The control arm is the same drive on a real config, so a row that passed by
    breaking `mantis.run` outright would be red.
    """
    unreadable = tmp_path / "run6.txt"
    unreadable.write_text(RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                                   "actor_lag_abort_enabled: false"))
    refused = subprocess.run([sys.executable, "-m", "mantis.run", str(unreadable)],
                             cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert refused.returncode != 0, (
        "a schema-valid, actor-lag-DISARMED config at an unrecognised suffix launched a run; "
        f"got rc 0\n{refused.stdout[-2000:]}"
    )
    assert "ConfigSuffixError" in refused.stderr, refused.stderr[-2000:]
    allowed = subprocess.run([sys.executable, "-m", "mantis.run", str(RUN5)],
                            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert allowed.returncode == 0, (
        "the control arm: a real config must still launch, or this row passes by breaking the "
        f"entry point. got {allowed.returncode}\n{(allowed.stdout + allowed.stderr)[-2000:]}"
    )


def test_the_MINT_route_refuses_an_out_path_the_loader_would_refuse(tmp_path) -> None:
    """The second of the three facts: `tools/mint_config.py --out` is a free path with no
    suffix constraint, so the repo's OWN minting tool would produce the unreadable file on
    request. Refused from the same predicate rather than from a fourth copy of the answer.
    """
    mint = REPO_ROOT / "tools" / "mint_config.py"
    bad = tmp_path / "minted.txt"
    refused = subprocess.run([sys.executable, str(mint), "--template", "dev", "--out", str(bad),
                              "--set", "run_id=x"], cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=300)
    assert refused.returncode == 2, (refused.stdout + refused.stderr)[-2000:]
    assert not bad.exists(), "refused BEFORE anything was written, or the refusal is cosmetic"
    for suffix in CONFIG_SUFFIXES:
        good = tmp_path / f"minted{suffix}"
        allowed = subprocess.run([sys.executable, str(mint), "--template", "dev", "--out",
                                  str(good), "--set", "run_id=x"], cwd=str(REPO_ROOT),
                                 capture_output=True, text=True, timeout=300)
        assert allowed.returncode == 0 and good.is_file(), (
            f"the inverse: EVERY suffix in CONFIG_SUFFIXES must still mint, or the fix is "
            f"'refuse everything but .yaml'. {suffix}: rc {allowed.returncode}\n"
            f"{(allowed.stdout + allowed.stderr)[-1500:]}"
        )


def test_the_exclusion_BOTH_gates_share_is_backed_by_the_LOADER_refusing_the_file(
    monkeypatch, tmp_path,
) -> None:
    """ADJ-13 F-1's structural half — INVERTED by the corrective pass.

    The row this replaces asserted `configs/run6.conf` must not be reported UNDECLARED, and
    planted it as a two-line stub so it never asked whether a `.conf` file could be a real
    config. As written it **required the escape to stay open**: a defect with a green test
    defending it. The silence is still the right behaviour — a gate that goes red on a README
    is a gate operators route around — but it is only defensible if the excluded file is not a
    config anywhere, so that is what this row now measures, on a REAL disarmed config rather
    than on a stub.

    The tree carries every shape at once: flat `.yaml`, flat `.yml`, nested `.yaml`, a disarmed
    `.conf`, a disarmed `.txt`, and a genuine non-config.
    """
    root = _mini_tree(tmp_path)
    _plant_disarmed(root, "run6.yml")
    _plant_disarmed(root, "prod/run6.yaml")
    excluded = [_plant_disarmed(root, "run6.conf"), _plant_disarmed(root, "run6.txt")]
    notes = root / "configs" / "NOTES.md"
    notes.write_text("not a config\n")
    excluded.append(notes)

    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    discovered = TOOL._discovered_configs()
    authority = [path.relative_to(root).as_posix()
                 for path in discover_configs(root / "configs")]
    assert discovered == authority, (
        "gate 12's audit set IS the loader's discovery authority — not a copy of it and not a "
        f"second glob (R71). got {discovered} vs {authority}"
    )
    assert "configs/run6.yml" in discovered and "configs/prod/run6.yaml" in discovered
    undeclared, _stale, _overlapping = TOOL._config_declaration_drift()
    for path in excluded:
        rel = path.relative_to(root).as_posix()
        assert rel not in discovered, f"{rel} is outside CONFIG_SUFFIXES; got {discovered}"
        assert rel not in undeclared, (
            f"a false UNDECLARED red on {rel} is as much a divergence as a false green; got "
            f"{undeclared}"
        )
        with pytest.raises(ConfigSuffixError):
            load_config(path)
    assert set(CONFIG_SUFFIXES) == {".yaml", ".yml"}, (
        "the authority's own contents, pinned: widening this tuple widens discovery AND the "
        f"loader together, which is the property F-1 broke; got {CONFIG_SUFFIXES}"
    )


#: Recheck R-4 — a REGRESSION the ADJ-13 delta introduced in a gate that was green. Adding an
#: `is_file()` conjunct to `discover_configs` made gate 7 stop rejecting two shapes it rejected
#: at `c3ab028` (HEAD rc 1 on each; delta rc 0, both gates silent). The predicate is a NAME
#: test again, so a config-shaped-and-broken file is a loud gate-7 failure rather than a
#: filtered-away one.
_R4_BROKEN = ("directory", "broken_symlink")


@pytest.mark.parametrize("kind", _R4_BROKEN)
def test_a_config_SHAPED_but_BROKEN_path_is_a_LOUD_gate_7_failure_and_not_silence(
    tmp_path, kind,
) -> None:
    """Driven through gate 7 as a process, in a mini tree, because rc is the observable."""
    root = _mini_tree(tmp_path)
    shutil.copy2(REPO_ROOT / "tools" / "ci_gates" / "validate_configs.py",
                 root / "tools" / "ci_gates" / "validate_configs.py")
    broken = root / "configs" / "broken.yaml"
    if kind == "directory":
        broken.mkdir()
    else:
        broken.symlink_to(tmp_path / "nowhere" / "target.yaml")

    result = _run_tool(cwd=root, tool=root / "tools" / "ci_gates" / "validate_configs.py")
    assert result.returncode == 1, (
        f"a {kind} at configs/broken.yaml is config-SHAPED and BROKEN — gate 7 rejected it at "
        f"HEAD and must still. got rc {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-2000:]}"
    )
    assert "FAIL configs/broken.yaml" in result.stderr, result.stderr[-2000:]
    assert result.stdout.count("OK ") == len(discover_configs(REPO_ROOT / "configs")), (
        "…and every real config must still validate, so the row cannot pass by breaking the "
        f"gate outright. got {result.stdout!r}"
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
    result = _run_tool("--config", "configs/run5.yaml", "--burst-steps", str(_N),
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
    smoke.write_text(smoke.read_text().replace("actor_lag_abort_enabled: false",
                                               "actor_lag_abort_enabled: true"))
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
    args = SimpleNamespace(config=str(RUN5), burst_steps=_N, out_dir=str(out_dir),
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
