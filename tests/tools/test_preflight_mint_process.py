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
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.config.armed_aborts import EXEMPT_CONFIGS, MANIFEST, PRODUCTION_CONFIGS
from mantis.config.loader import load_config
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
    for config in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        shutil.copy2(config, root / "configs" / config.name)
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
