"""The mint preflight's PROCESS half — the producers for everything past `_run_child`.

>300 justify (R8): one tool, one process half. Every test drives the same tool module loaded
once by absolute path, and four of the blocks share the one `_mini_tree` rig that makes the
tool's `REPO_ROOT` addressable without writing inside the repo. Splitting by subject would
fork that rig four ways and give each copy its own way to be wrong.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mantis.config.armed_aborts import (
    EARLIEST_FIRE_FRACTION,
    EXEMPT_CONFIGS,
    MANIFEST,
    PRODUCTION_CONFIGS,
    ArmedAbort,
    Mechanism,
    SampleClockNotDerivableError,
    Status,
    audit_arming,
)
from mantis.config.loader import config_identity_sha256, discover_configs, load_config
from mantis.config.preflight_stamp import write_stamp
from mantis.monitor.sink import JsonlEventSink
from mantis.train.actor_sync import ActorSync
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
#: The tool loads its parent half off its own directory, so a rig that relocates the tool
#: must carry the sibling with it.
PARENT_PATH = TOOL_PATH.with_name("preflight_mint_parent.py")


#: run5's own constants, read from the file rather than restated.
RUN5 = REPO_ROOT / "configs" / "run6.yaml"
_N = 101
#: The burst floor for `configs/run6.yaml`: `max(100, 1, 25000) + 1`, measured not assumed —
#: run5 arms the draw-rate abort at `min_step: 25000`, so a shorter burst is refused at rc 11.
#: `_N` stays 101 for every drive that is about the a/b assertion arithmetic.
_RUN5_BURST = 25001


def _load_tool():
    """Load the gate script by absolute path — ZERO `sys.path` mutation (R5 / LAW-17)."""
    spec = importlib.util.spec_from_file_location("preflight_mint_process", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()

#: Read from the generator that stamps the header separator rather than restating the format.
_MINT_SPEC = importlib.util.spec_from_file_location(
    "_mint_config_for_run5_twin", REPO_ROOT / "tools" / "mint_config.py")
assert _MINT_SPEC is not None and _MINT_SPEC.loader is not None
MINT_CONFIG = importlib.util.module_from_spec(_MINT_SPEC)
_MINT_SPEC.loader.exec_module(MINT_CONFIG)


def _run_tool(*args, cwd: Path = REPO_ROOT, tool: Path = TOOL_PATH, timeout: int = 300,
              env: dict[str, str] | None = None):
    return subprocess.run([sys.executable, str(tool), *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout, env=env)


def _cuda_is_available() -> bool:
    """Is THIS box a CUDA box? Asked once, at import, so the host-dependence of the two
    device-sensitive rows below is DECLARED in a skip marker instead of being discovered as
    a mystery failure. torch is already a module-level transitive import here (`ActorSync`)."""
    import torch

    return bool(torch.cuda.is_available())


#: Declared once. `configs/run6.yaml` mints `train.device: cuda`, so what a real run5 boot
#: does is a property of the host; both halves are pinned rather than one left to chance.
_CUDA_BOX = _cuda_is_available()



#: The leaves a CPU twin of run6 is ALLOWED to differ in, and nothing else. The warm-start
#: block collapses to one null because run6's BC checkpoint lives under untracked `checkpoints/`.
FORCED_TWIN_LEAVES: frozenset[str] = frozenset({
    "run_id", "train.device", "eval.worker_device",
    "identity.warm_start", "identity.warm_start.checkpoint", "identity.warm_start.net_hash",
})


def _flat_leaves(config) -> dict[str, object]:
    """A validated config's leaves as dotted paths — the same shape gate 13's walker uses."""
    def walk(node, prefix: str) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                out.update(walk(value, f"{path}."))
            else:
                out[path] = value
        return out
    return walk(config.model_dump(), "")


def _mint_run5_cpu_bootable_twin(out_dir: Path) -> Path:
    """Mint run5's CPU twin PLUS a valued fused-graph cap — the launch-surface row's target only.

    The cap pair is read OFF THE TEMPLATE, so the day the template's non-binding derivation
    moves, this twin moves with it.
    """
    template = yaml.safe_load(
        (REPO_ROOT / "tools" / "config_templates" / "dev.yaml").read_text(encoding="utf-8")
    )["inference"]["fused_graph_caps"]
    dest = _mint_run5_cpu_twin(out_dir, name="run5_cpu_bootable", extra_deltas=[
        (f"inference.fused_graph_caps={{max_fused_edges: {template['max_fused_edges']}, "
         f"max_fused_nodes: {template['max_fused_nodes']}}}"),
    ])
    return dest


def _run5_header_deltas() -> list[str]:
    """Re-issue run5's own stamped `# delta:` lines as `mint_config.py --set` arguments.

    Derived from the stamped header, never transcribed: a leaf inside a template block that
    ships `null` is not addressable by `--set` at all, so a leaf diff cannot produce a
    replayable delta set. The separator is imported from the generator that writes it.

    Raises:
        AssertionError: if run5's header carries no delta lines, or a line does not carry the
            separator — either means the stamped provenance is unreadable, and a twin minted
            from an unreadable header is not run5's twin.
    """
    deltas: list[str] = []
    for line in RUN5.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# delta:"):
            continue
        key, _, rest = line[len("# delta:"):].strip().partition(": ")
        old, sep, new = rest.partition(MINT_CONFIG.HEADER_SEP)
        assert sep, f"unreadable delta line (no {MINT_CONFIG.HEADER_SEP!r}): {line!r}"
        assert key and old, f"unreadable delta line (no key or old slot): {line!r}"
        deltas.append(f"{key}={new}")
    assert deltas, (
        f"{RUN5} carries no `# delta:` header lines — the twin has nothing to replay, and a "
        "config minted from the bare template is not run5"
    )
    return deltas


def _mint_run5_cpu_twin(out_dir: Path, *, name: str = "run5_cpu_boot",
                        extra_deltas: list[str] | None = None) -> Path:
    """Mint run5's CPU twin: run5's own header deltas replayed, plus `run_id` and the two
    device leaves, minus the warm start.

    `identity.warm_start` is dropped because run6 declares a BC checkpoint under
    `checkpoints/`, which is never tracked, so no drive here can supply the file. Minted
    per-drive into `tmp_path` rather than committed, so no near-clone of run5 sits in the
    audit root where an operator could preflight it believing it was run5.
    """
    run5 = load_config(RUN5)
    draw = run5.train.draw_rate_abort
    assert draw is not None, "premise: run5 arms the draw-rate abort (the tier-full floor row)"
    dest = out_dir / f"{name}.yaml"
    deltas = [
        # The warm start is replayed by nobody: its checkpoint is an untracked artifact.
        *(d for d in _run5_header_deltas() if not d.startswith("identity.warm_start=")),
        "run_id=run5_cpu_boot",
        "train.device=cpu",
        "eval.worker_device=cpu",
        *(extra_deltas or ()),
    ]
    # `--set` REFUSES a key the template omits and `--mint-row` is the flag for those, so
    # which flag a delta needs is derived from the template's own leaves.
    def _raw_leaves(node, prefix=""):
        out = []
        for key, value in (node or {}).items():
            path = f"{prefix}{key}"
            out.extend(_raw_leaves(value, f"{path}.") if isinstance(value, dict) else [path])
        return out

    template_leaves = _raw_leaves(yaml.safe_load(
        (REPO_ROOT / "tools" / "config_templates" / "dev.yaml").read_text(encoding="utf-8")))
    argv = [sys.executable, str(REPO_ROOT / "tools" / "mint_config.py"),
            "--template", "dev", "--out", str(dest)]
    for delta in deltas:
        key = delta.split("=", 1)[0]
        in_template = any(leaf == key or leaf.startswith(key + ".") for leaf in template_leaves)
        argv += ["--set" if in_template else "--mint-row", delta]
    minted = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert minted.returncode == 0, (
        "the twin must be MINTED, not hand-varied (R130/R103); mint_config exited "
        f"{minted.returncode}\n{(minted.stdout + minted.stderr)[-2000:]}"
    )
    twin = load_config(dest)
    base, other = _flat_leaves(run5), _flat_leaves(twin)
    absent = object()
    keys = base.keys() | other.keys()
    differing = {key for key in keys if base.get(key, absent) != other.get(key, absent)}
    if extra_deltas:
        # The bounded-difference guarantee binds the UNNAMED twin alone; the named variant
        # states its own extra leaves.
        return dest
    assert differing == FORCED_TWIN_LEAVES, (
        "the twin must be run6 with the device this box has, minus a warm start no checkout "
        "can hold, and NOTHING else — anything more and these drives stop being evidence "
        f"about run6's own boot. Differing leaves: {sorted(differing)}"
    )
    assert twin.train.device == "cpu" and run5.train.device == "cuda", (
        f"got twin device {twin.train.device!r} against run5 {run5.train.device!r}"
    )
    return dest


def _launch_until_boot_identity(config_path: Path, out_dir: Path, *, deadline_sec: float = 180.0,
                                env: dict[str, str] | None = None):
    """Drive the production launcher and stop it the moment the run publishes `run_boot_identity`.

    Teardown is SIGTERM to the child's own process group, which lands on the lifecycle
    handlers and lets the run save-then-exit; SIGKILL is the backstop.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "mantis.run", "--config", str(config_path),
         "--out-dir", str(out_dir)],
        cwd=str(REPO_ROOT), start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    witness = None
    started = time.monotonic()
    try:
        while witness is None and time.monotonic() - started < deadline_sec:
            for segment in sorted((out_dir / "logs").glob("events_*.jsonl")):
                for line in segment.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A tailing reader can catch a half-flushed final line; the next
                        # poll re-reads the file whole.
                        continue
                    if row.get("event") == "run_boot_identity":
                        witness = row
                        break
                if witness is not None:
                    break
            if witness is None and proc.poll() is not None:
                break  # the launcher exited without ever publishing — the caller reports it
            if witness is None:
                time.sleep(0.25)
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
    return SimpleNamespace(witness=witness, rc=proc.returncode, stdout=stdout, stderr=stderr)


def _child(rc: int, *, tail: str = "", timed_out: bool = False) -> dict:
    """The `_run_child` result dict in its exact shipped shape — a dict, not a production object."""
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
    """A child that named its own outcome is classified by its rc BEFORE the stderr sniff, so
    it keeps its name; both stderr arms are driven for every pass-through code."""
    for tail in ("", _ATTR):
        with pytest.raises(TOOL.PreflightChildOutcomeError) as caught:
            TOOL._classify_child(_child(rc, tail=tail))
        assert caught.value.rc == rc, (
            f"§6.3a arm 4: a child rc in [10, 41] PROPAGATES UNCHANGED whatever its stderr "
            f"says. With tail={tail!r} the parent reported rc {caught.value.rc}, not {rc} — "
            "a child that named its own outcome must be believed about it"
        )


def test_the_stderr_sniff_still_classifies_a_child_that_did_NOT_name_itself() -> None:
    """Moving the stderr sniff behind the pass-through arm must not disable it — rc 1 is the
    child that has nothing to say about itself."""
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
    """Classifier arms 1-3 and 6-7; the timeout arm runs before the sign arm, because a
    timeout kill also produces a negative returncode."""
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
    """A child rc of 46 is the run's own cooperative armed abort: it PROPAGATES rather than
    collapsing to the generic boot failure 33, so a supervisor reads one number on both sides."""
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
    assert TOOL.ARMED_ABORT_CODES == (46, 47, 48), (
        f"the authored codes are 46 (draw-rate) and 47 (disk guard) and BOTH come from the ONE "
        f"authority (`monitor/heartbeat.py`), never re-typed here; got "
        f"{TOOL.ARMED_ABORT_CODES!r}"
    )
    assert not set(TOOL.ARMED_ABORT_CODES) & set(TOOL.WATCHDOG_CODES), (
        "both are reserved codes but NEITHER is a watchdog code — they are not delivered by "
        "`os._exit` and must not be diagnosed as a stall"
    )
    assert TOOL.DRAW_RATE_COLLAPSE_EXIT_CODE == 46


def test_the_boot_childs_rc_is_decided_by_whether_an_abort_fired() -> None:
    """`_abort_rc` returns 0 when no rule fired, the manifest's code for an authored abort, and
    a NAMED failure for an abort with no authored code — never 0 and never an invented number."""
    assert TOOL._abort_rc(None) == 0, "no rule fired is the ONLY thing that means rc 0"
    assert TOOL._abort_rc("draw_rate_collapse") == TOOL.DRAW_RATE_COLLAPSE_EXIT_CODE

    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._abort_rc("grad_norm_hard_abort")
    assert "grad_norm_hard_abort" in str(caught.value) and caught.value.rc == 33, (
        "an abort with no authored code is a named failure that NAMES THE RULE, not a silent "
        f"rc 0 and not a fabricated code; got rc {caught.value.rc}"
    )


def test_the_pass_through_range_is_the_designs_range_and_is_not_empty() -> None:
    """The pass-through range is the design's and is not empty: 10 and 41 are IN, 9 and 42 are
    OUT — 42 is the run's own watchdog code and is never a preflight outcome."""
    passing = set(TOOL.PASS_THROUGH)
    assert passing == set(range(10, 42)), f"§6.3a arm 4's range is [10, 41]; got {sorted(passing)}"
    assert 9 not in passing and 42 not in passing


def test_the_module_docstring_names_the_wall_the_boot_actually_hits() -> None:
    """The tool's module docstring must name the wall the boot actually hits; the integration
    row below is the measurement and this is the cheap default-tier consistency pin."""
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
@pytest.mark.usefixtures("local_puller")
def test_the_real_boot_terminates_where_the_docstring_says(tmp_path) -> None:
    """The real boot, on the real tree, in production posture — the only test that drives a
    preflight child to completion, so the child's rc is read off the report and never restated."""
    out_dir = tmp_path / "boot"
    result = _run_tool("--config", str(_mint_run5_cpu_twin(tmp_path)),
                       "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "45", "--receipt-wait-sec", "0")
    assert result.returncode == 40, (
        "post-TD-4, and post-mint, the boot runs until the timeout kills it: rc 40 "
        "PreflightTimeoutError. An rc 33 here means run5's minted caps stopped resolving. "
        f"got {result.returncode}\n{(result.stdout + result.stderr)[-3000:]}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert len(reports) == 1, f"the evidence report is written ALWAYS (§9.1); found {reports}"
    report = json.loads(reports[0].read_text())
    assert report["failure"] == "PreflightTimeoutError"
    assert report["child"]["timed_out"] is True
    tail = report["child"]["stderr_tail"]
    assert "UncalibratedFusedGraphCapsError" not in tail, (
        "run5 is CALIBRATED since 2026-08-18; a refusal here means the minted pair stopped "
        f"reaching the resolver. got tail {tail[-600:]!r}"
    )
    assert "MissingEncodingError" not in tail, (
        "CARD-POOL-ENCODING-BRIDGE has landed; the pool resolves `identity.encoding` through "
        f"the ONE resolver. A MissingEncodingError here is that card regressing. got {tail[-600:]!r}"
    )
    assert "train_step" not in tail, (
        "TD-1 is not reached on a CPU box — it sits BEHIND the warmup gate, and the buffer "
        "never fills. If this ever fires, the box got far enough to need "
        "CARD-TRAINSTEP-ADAPTER, which would be news worth reading."
    )
    # The positive half: the run's own segment is the witness that the child ran, not the
    # tool's say-so.
    segments = sorted((out_dir / "logs").glob("events_*.jsonl"))
    assert segments, f"a booted run writes its own segment; found {list(out_dir.rglob('*'))}"
    events = {json.loads(line)["event"] for line in segments[0].read_text().splitlines() if line}
    assert {"run_segment_started", "heartbeat_watchdog_armed",
            "selfplay_stall_watchdog_armed"} <= events, (
        f"the boot must reach an ARMED training loop, not just construct objects; saw {events}"
    )


@pytest.mark.integration
@pytest.mark.usefixtures("local_puller")
def test_an_UNCALIBRATED_twin_is_refused_by_the_ARMING_AUDIT_before_it_can_boot(tmp_path) -> None:
    """An uncalibrated production config is refused by the ARMING AUDIT before a child is ever
    spawned, so the audit shadows the composition seam the refusal used to be measured at."""
    out_dir = tmp_path / "boot_uncalibrated"
    twin = _mint_run5_cpu_twin(tmp_path, name="run5_cpu_uncalibrated", extra_deltas=[
        "inference.fused_graph_caps={max_fused_edges: null, max_fused_nodes: null}",
    ])
    result = _run_tool("--config", str(twin), "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "45", "--receipt-wait-sec", "0")
    assert result.returncode == 30, (
        "an UNCALIBRATED production config must be refused by the ARMING AUDIT before any "
        f"boot: rc 30 PreflightArmingAuditError. got {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-3000:]}"
    )
    blob = result.stdout + result.stderr
    assert "PreflightArmingAuditError" in blob and "fused_graph_caps_calibrated" in blob, (
        "the refusal must name the ROW, or an operator cannot tell which of the required "
        f"aborts is disarmed. got {blob[-1500:]!r}"
    )
    assert "inference.fused_graph_caps.max_fused_edges" in blob, (
        f"…and the KEY, which is the thing they have to mint. got {blob[-1500:]!r}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert len(reports) == 1, f"the evidence report is written ALWAYS (§9.1); found {reports}"
    report = json.loads(reports[0].read_text())
    assert report["child"] is None, (
        "rc 30 is decided BEFORE the child is spawned; a child block here means the audit "
        "stopped running first, which is the ordering the cheap failure depends on")


@pytest.mark.integration
@pytest.mark.usefixtures("local_puller")
def test_the_real_boot_still_reaches_an_ARMED_loop_on_a_CALIBRATED_config(tmp_path) -> None:
    """The tool's SUCCESS path: an otherwise-identical config that HAS a cap boots clean and
    arms both watchdogs, so the refusal above is caused by the missing value and nothing else."""
    out_dir = tmp_path / "boot_calibrated"
    result = _run_tool("--config", str(_mint_run5_cpu_bootable_twin(tmp_path)),
                       "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "45", "--receipt-wait-sec", "0")
    assert result.returncode == 40, (
        "with the cap VALUED the boot runs until the timeout kills it: rc 40 "
        "PreflightTimeoutError. An rc 33 here means the caps are refused even when present, "
        f"i.e. the resolver reads something other than the config. got {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-3000:]}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert len(reports) == 1, f"the evidence report is written ALWAYS (§9.1); found {reports}"
    report = json.loads(reports[0].read_text())
    assert report["failure"] == "PreflightTimeoutError"
    assert report["child"]["timed_out"] is True
    assert "UncalibratedFusedGraphCapsError" not in report["child"]["stderr_tail"], (
        "a VALUED cap must not refuse. This firing means the resolver rejects a legal pair — "
        f"got tail {report['child']['stderr_tail'][-600:]!r}"
    )
    # The positive half: the run's own segment is the witness that the child ran, not the
    # tool's say-so.
    segments = sorted((out_dir / "logs").glob("events_*.jsonl"))
    assert segments, f"a booted run writes its own segment; found {list(out_dir.rglob('*'))}"
    events = {json.loads(line)["event"] for line in segments[0].read_text().splitlines() if line}
    assert {"run_segment_started", "heartbeat_watchdog_armed",
            "selfplay_stall_watchdog_armed"} <= events, (
        f"the boot must reach an ARMED training loop, not just construct objects; saw {events}"
    )


@pytest.mark.integration
@pytest.mark.usefixtures("local_puller")
@pytest.mark.skipif(
    _CUDA_BOX,
    reason="asserts what a CUDA-MINTED run5 does on a NON-CUDA host; this box has CUDA, so "
           "the boot legitimately proceeds instead of refusing. The binding measurement for "
           "run5 on a CUDA box is the box preflight (CARD-RUN5-GPU-OOM, R130) — not this row, "
           "which exists to keep the device false-clear dead on every CPU box in the fleet.",
)
def test_booting_run5_on_a_non_CUDA_box_fails_LOUD_in_init_trainer(tmp_path) -> None:
    """A cuda-minted run6 on a non-CUDA host fails LOUD and BEFORE any boot: since R347(d) the
    START halt rc 17 (`PreflightCudaBuildError`) fires ahead of the child, where the old rc 33
    in `init_trainer` used to be the first wall. The device stays a config fact either way, so a
    cpu preflight can never false-clear the GPU memory wall."""
    out_dir = tmp_path / "run5_on_cpu"
    assert load_config(RUN5).train.device == "cuda", (
        "PREMISE: run6 mints `train.device: cuda`. If it is ever re-minted to cpu this row "
        "is testing nothing and must be re-adjudicated, not adjusted"
    )
    from mantis.config.resolve.allocator_posture import resolve_allocator_posture

    full_config = load_config(RUN5).model_dump()
    env = {**os.environ, **resolve_allocator_posture(full_config).required_env()}
    result = _run_tool("--config", str(RUN5), "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "45", "--receipt-wait-sec", "0", env=env)
    assert result.returncode == 17, (
        "run6 on a non-CUDA box must HALT by name before the boot: rc 17 "
        f"PreflightCudaBuildError. got {result.returncode}\n"
        f"{(result.stdout + result.stderr)[-3000:]}"
    )
    report = json.loads(sorted(out_dir.glob("preflight_*.json"))[0].read_text())
    assert report["failure"] == "PreflightCudaBuildError" and report["verdict"] == "fail"
    assert report.get("child") is None, (
        "the halt lands BEFORE `_run_child`: a child block here means a boot was attempted on "
        f"a torch that cannot compute on a GPU; got {report.get('child')!r}"
    )
    assert "CPU-ONLY build" in (result.stdout + result.stderr), (
        "the refusal names the cause — a CPU-only torch — rather than a bare rc"
    )


def _mini_tree(tmp_path: Path) -> Path:
    """A scratch root the real tool resolves as its own `REPO_ROOT`.

    The tool is the shipped file run as itself against a different tree. The tree carries
    exactly what the audit path reads; `MANIFEST`, `PRODUCTION_CONFIGS` and `EXEMPT_CONFIGS`
    still come from the installed package, so the rig varies the tree and never the manifest.
    """
    root = tmp_path / "tree"
    (root / "tools" / "ci_gates").mkdir(parents=True)
    shutil.copy2(TOOL_PATH, root / "tools" / "ci_gates" / "preflight_mint.py")
    # The tool's loader keys the sibling off `__file__`, so the copied tool loads the COPIED
    # parent half.
    shutil.copy2(PARENT_PATH, root / "tools" / "ci_gates" / "preflight_mint_parent.py")
    (root / "configs").mkdir()
    # The rig copies what the ONE discovery authority finds; a rig with a glob of its own
    # would go on passing after exactly the divergence it exists to catch.
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
    """`--audit-only` against a mini tree."""
    return _run_tool("--audit-only", *args, cwd=root,
                     tool=root / "tools" / "ci_gates" / "preflight_mint.py")


def test_the_mini_tree_rig_is_green_before_it_is_perturbed(tmp_path) -> None:
    """The rig's own vacuity floor: green before it is perturbed."""
    root = _mini_tree(tmp_path)
    copied = (root / "tools" / "ci_gates" / "preflight_mint.py").read_bytes()
    assert copied == TOOL_PATH.read_bytes(), "the rig must run the SHIPPED tool, unmodified"
    copied_parent = (root / "tools" / "ci_gates" / "preflight_mint_parent.py").read_bytes()
    assert copied_parent == PARENT_PATH.read_bytes(), (
        "the rig must carry the SHIPPED parent half, unmodified — a diverged sibling would "
        "make the copied tool a different tool"
    )
    result = _mini_audit(root)
    assert result.returncode == 0, (
        "an unperturbed mini tree must be as green as the real one; got "
        f"{result.returncode}\n{(result.stdout + result.stderr)[-3000:]}"
    )


def test_the_source_pin_scan_runs_inside_the_live_audit_path(tmp_path) -> None:
    """The source-pin scan runs inside the live audit path and `source_pins_ok` is derived from
    its result, so deleting the call fails the gate rather than leaving a green report."""
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
    """The report publishes the pins the scan ACTUALLY covered, so it cannot say `ok` while
    naming zero pins."""
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


def test_a_config_declared_by_neither_tuple_fails_the_gate(tmp_path) -> None:
    """A config sitting in `configs/` that NEITHER tuple declares fails the gate, so
    "deliberately exempt" and "forgotten" stop being the same observable."""
    root = _mini_tree(tmp_path)
    rel = f"{_F1_PLANT_STEM}.yaml"
    plant = root / "configs" / rel
    plant.write_text(RUN5.read_text().replace("actor_lag_abort_enabled: true",
                                              "actor_lag_abort_enabled: false"))
    assert "actor_lag_abort_enabled: false" in plant.read_text(), (
        "the planted config must really be disarmed, or this test is vacuous"
    )
    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        "a config on disk that neither tuple names must FAIL the gate — it was rc 0 before "
        f"(never audited at all); got {result.returncode}\n{output[-3000:]}"
    )
    assert f"configs/{rel}" in output and "UNDECLARED" in output, (
        f"the failure must name the undeclared config and say what to do; got {output[-2000:]}"
    )


def test_a_declaration_that_names_a_missing_config_fails_the_gate(tmp_path) -> None:
    """A declaration naming a config absent from disk fails the gate, so a stale declaration is
    never silently tolerated."""
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
    """Every config on the tree that ships is declared by exactly one of the two tuples."""
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
    """`--config X` UNIONS with the production set rather than replacing it, so a disarmed
    production config is still audited when a different config is named."""
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
    """Both modes compute the audit scope from one function, so they cannot drift apart again."""
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
    """The manifest vacuity guard fires before anything indexes the paths, so an empty
    production set is the named rc 31 rather than an IndexError collapsed to an unnamed rc 1."""
    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS", ())
    with pytest.raises(TOOL.PreflightManifestError) as caught:
        TOOL._audit_manifest_and_configs([])
    assert caught.value.rc == 31 and "vacuous" in str(caught.value)

    report = TOOL._new_report("audit")
    args = SimpleNamespace(config=None, out_dir=None)
    with pytest.raises(TOOL.PreflightManifestError):
        TOOL._run_audit(args, report)

    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS", tuple(PRODUCTION_CONFIGS))
    # The synthetic row restores this arm's subject — a manifest that HAS rows, none of them
    # REQUIRED. Filtering the shipped manifest down to its deferred rows now yields ().
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


def test_an_interval_that_outruns_the_run_REDS_the_real_gate(tmp_path) -> None:
    """A `monitor.gate_interval` that outruns the run reds the real gate: a threshold whose gate
    boundaries fall past the end of the run is armed in the config and unread in the run."""
    root = _mini_tree(tmp_path)
    production = root / PRODUCTION_CONFIGS[0]
    original = production.read_text()
    assert original.count("gate_interval: 1000\n") == 1, (
        "the rig rewrites exactly one key; if run5's gate_interval spelling moved, this "
        "perturbation is no longer the one the defect needs"
    )
    production.write_text(original.replace("gate_interval: 1000\n",
                                           "gate_interval: 1000000000\n"))
    assert "draw_rate_abort" in production.read_text(), (
        "the draw-rate row must still be ARMED, or this test is about the arming audit"
    )

    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 30, (
        "an armed abort that cannot fire inside its own run must FAIL assertion (c) — this "
        f"tree was rc 0 before R251; got {result.returncode}\n{output[-3000:]}"
    )
    assert "CADENCE-DISARMED" in output and "draw_rate_collapse" in output, (
        f"the failure must name the row and the class; got {output[-2000:]}"
    )
    assert "3000000000" in output and "250000.0" in output, (
        "…and it must name the COMPUTED step and the BOUND, or an operator cannot tell "
        f"which key to move; got {output[-2000:]}"
    )
    assert "NEVER a sanctioned disarm" in output, (
        "…and say that a large interval is not a legal way to disarm a row, which is the "
        f"whole ruling; got {output[-2000:]}"
    )


def test_the_green_audit_PUBLISHES_the_cadence_it_computed(tmp_path) -> None:
    """The green audit publishes every judged row with its computed step, the bound it cleared
    and the fraction it used — a check whose only output is its own failure witnesses nothing."""
    result = _run_tool("--audit-only", "--out-dir", str(tmp_path / "cadence"))
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    report = json.loads(
        sorted((tmp_path / "cadence").glob("preflight_*.json"))[0].read_text())["manifest"]
    assert report["cadence_fraction"] == EARLIEST_FIRE_FRACTION
    judged = {row["name"] for row in report["cadence"]}
    armed_required = {row.name for row in MANIFEST if row.status is Status.REQUIRED}
    assert judged == armed_required, (
        "every REQUIRED row armed on the production config must appear with a verdict; got "
        f"{sorted(judged)} against {sorted(armed_required)}"
    )
    assert all(row["within"] for row in report["cadence"])
    draw = [row for row in report["cadence"] if row["name"] == "draw_rate_collapse"][0]
    assert draw["earliest_fire_step"] == 25000.0 and draw["bound"] == 250000.0, (
        f"the published numbers must be the computed ones; got {draw!r}"
    )


class _NeuteredMember:
    """A cadence member whose arithmetic answers a constant and which ANSWERS rather than raises
    on a `None` period — the mutation the self-test exists to catch, driving arms A-D and F."""

    @staticmethod
    def earliest_fire_step(values, *, period_steps=None):
        del values, period_steps
        return 0.0


class _NeuteredCadence:
    GATE_INTERVAL_CONSEC = _NeuteredMember
    STEP_LAG_THRESHOLD = _NeuteredMember
    EVAL_ROUND_CONSEC = _NeuteredMember


class _Clock:
    """One sample-clock member's shape as `_cadence_self_test`'s arm E reads it: a name and a
    period path. A plain class, so instances stay hashable (arm E keys a dict by member)."""

    def __init__(self, value: str, period_path: str) -> None:
        self.value = value
        self.period_path = period_path


#: Every sample clock naming ONE period key: every arithmetic arm stays green because the
#: numbers are all correct and only the KEY they are taken from is wrong, so arm E is the
#: only thing that can see it.
_COLLAPSED_CLOCKS = (_Clock("gate_boundary", "monitor.gate_interval"),
                     _Clock("eval_round", "monitor.gate_interval"))


def test_the_TOOLS_OWN_fraction_is_the_one_the_audit_compares(monkeypatch, tmp_path) -> None:
    """The audit compares against the TOOL'S OWN fraction, not the callee default — at HEAD the
    two are the same object, so every published number agrees with every compared number by
    coincidence.

    `1.0` is the one fraction that leaves the self-test green, so the pin is provably about the
    call site and not about the trigger. Two properties: the published bound is taken at the
    tool's number, and a config whose earliest fire sits between the two bounds FLIPS verdict.
    """
    assert not TOOL._cadence_self_test(), "the unmutated self-test must be green"
    run5_length = load_config(RUN5).train.max_train_steps
    # Earliest fire 100000 * max(3, ceil(25000/100000)) = 300000 — above run5's 0.25 bound
    # (250000) and inside a 1.0 bound (1000000). Written OUTSIDE configs/ so the declaration
    # partition is untouched and the only variable is the fraction.
    between = tmp_path / "between_the_bounds.yaml"
    between.write_text(RUN5.read_text().replace("gate_interval: 1000\n",
                                                "gate_interval: 100000\n"))
    with pytest.raises(TOOL.PreflightArmingAuditError):
        TOOL._audit_manifest_and_configs([between])

    monkeypatch.setattr(TOOL, "EARLIEST_FIRE_FRACTION", 1.0)
    assert not TOOL._cadence_self_test(), (
        "fraction 1.0 must leave the self-test green, or this pin is testing the trigger"
    )
    block = TOOL._audit_manifest_and_configs([between])
    assert block["cadence_fraction"] == 1.0
    for row in block["cadence"]:
        assert row["bound"] == 1.0 * run5_length, (
            "the PUBLISHED fraction must be the fraction the comparison USED — under the "
            f"callee default this row's bound would still read {0.25 * run5_length}, and the "
            f"report would say one number while the audit compared another; got {row!r}"
        )


def test_the_self_tests_arm_A_fires_ALONE_when_the_bound_would_refuse_a_healthy_row(
    monkeypatch,
) -> None:
    """The self-test's arm A fires ALONE when the bound would refuse a healthy row — the half
    that protects against the gate refusing a healthy config had no producer.

    Driven by shrinking the self-test's own synthetic run length rather than the fraction: at
    run length 1 the bound is 0.25, which the healthy operands' 25000 exceeds while the vacuous
    operands still exceed it too and the lag member still computes 101.
    """
    monkeypatch.setattr(TOOL, "_SELF_TEST_RUN_LENGTH", 1)
    failures = TOOL._cadence_self_test()
    joined = "\n".join(failures)
    assert "arm A" in joined, (
        "a bound that refuses the HEALTHY operands must fire arm A — without this the "
        f"false-positive half of the trigger has no producer; got {failures!r}"
    )
    assert "arm B" not in joined and "arm C" not in joined, (
        "…and it must fire ALONE, or the arms are not independently observable and a "
        f"one-armed self-test reads the same as a two-armed one; got {failures!r}"
    )
    assert TOOL.main(["--audit-only"]) == 31, (
        "…and the gate must refuse to publish a verdict, by name, rather than auditing the "
        "tree with a trigger that would reject every healthy config"
    )


@pytest.mark.parametrize("attribute,value", [("EARLIEST_FIRE_FRACTION", float("inf")),
                                             ("Cadence", _NeuteredCadence),
                                             ("SampleClock", _COLLAPSED_CLOCKS)])
def test_neutering_the_cadence_check_REDS_the_gates_own_self_test(
    monkeypatch, attribute, value,
) -> None:
    """All three ways the cadence check can rot — an infinite bound, a constant earliest fire,
    and every sample clock naming the same period key — reach the process boundary as the named
    rc 31 rather than a quiet green."""
    assert not TOOL._cadence_self_test(), (
        "the unmutated self-test must pass, or the mutation below proves nothing"
    )
    monkeypatch.setattr(TOOL, attribute, value)
    assert TOOL._cadence_self_test(), (
        f"neutering {attribute} must be VISIBLE to the self-test — an instrument that cannot "
        "notice its own arithmetic being replaced is not an instrument"
    )
    assert TOOL.main(["--audit-only"]) == 31, (
        "…and the gate must refuse to publish a verdict, by name (rc 31), rather than "
        "auditing the tree with a dead check"
    )


def test_an_underivable_sample_clock_is_the_NAMED_rc_31_never_the_tools_rc_1(
    monkeypatch,
) -> None:
    """An underivable sample clock arrives as `PreflightManifestError` rc 31 with its message
    carried through, never as `main`'s bare-except rc 1 "the tool broke"."""
    def _raise(*_args, **_kwargs):
        raise SampleClockNotDerivableError("synthetic: train.eval_interval resolved to None")

    assert TOOL.main(["--audit-only"]) == 0, "the unmutated gate must be green"
    monkeypatch.setattr(TOOL, "audit_cadence", _raise)
    assert TOOL.main(["--audit-only"]) == 31, (
        "an underivable sample clock must reach the boundary as the NAMED manifest rc 31, "
        "not as the tool's own unnamed internal error"
    )


def test_the_deferred_rows_cadence_is_PRINTED_which_is_its_ONLY_live_consumer() -> None:
    """The deferred rows' `cadence` field reaches an operator on every gate run, which is its
    only live consumer until the flip to REQUIRED; driven through the real CLI, since only the
    process output can witness that."""
    result = _run_tool("--audit-only")
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    deferred = [row for row in MANIFEST if row.status is Status.DEFERRED]
    assert deferred, "no deferred row means this test has no subject"
    for row in deferred:
        assert row.cadence is not None, (
            f"deferred row {row.name!r} declares no cadence, so the flip to REQUIRED is not "
            "the one-field data edit the row's own comment claims"
        )
        assert f"earliest-fire cadence: {row.cadence.value}" in result.stdout, (
            f"the gate must PRINT {row.name!r}'s declared cadence — it is that field's only "
            f"live consumer until the row flips; got {result.stdout[-2000:]}"
        )
        for path in row.cadence_paths:
            assert path in result.stdout, (
                f"…and the operands too: {path!r} is what makes the printed cadence auditable "
                "rather than a bare enum name"
            )


#: The shipped manifest carries no deferred rows, so the four pins below drive the print
#: mechanism on a SYNTHETIC row through the `manifest=` keyword the tool exposes. Keeping a
#: shipped row deferred to suit a test was refused.
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
    """The tool's loud DEFERRED print, driven on the synthetic row and shared by the four pins."""
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
    """The deferred print carries the field, one pin per field, so each way of hollowing the
    print dies alone instead of sharing one failure signature."""
    # The subject stays SYNTHETIC deliberately: this test hollows the print one field at a
    # time, and a shipped row would make each drop's failure depend on that row's own field
    # values rather than on the print. All this test needs of the shipped set is NON-EMPTY.
    assert [row.name for row in MANIFEST if row.status.value == "deferred"], (
        "the shipped manifest holds NO deferred row, so the synthetic subject below would be "
        "standing in for an empty manifest rather than adding to a real one (R81)"
    )
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
    """The report publishes the RESOLVED coordinator config: present and complete by NAME off
    the dataclasses, agreeing with the config on disk key for key, and MOVING when it moves."""
    import dataclasses

    from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
    from mantis.config.resolve.drain import DrainCapsSpec

    _run_tool("--audit-only", "--config", "configs/run6.yaml",
              "--out-dir", str(tmp_path / "coord"))
    report = json.loads(sorted((tmp_path / "coord").glob("preflight_*.json"))[0].read_text())
    block = report["coordinator"]
    assert block is not None, (
        "the resolved coordinator config must be IN the evidence artifact — R78's rider, and "
        "the census measured its absence"
    )

    config = load_config(REPO_ROOT / "configs" / "run6.yaml")
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
        "run6's armed terms, as the run will really see them — the four travel together"
    )

    # The two arms need two DIFFERENT properties: the smoke profile is the only config whose
    # run length differs from run6's, and `dev_example` the only DISARMED one.
    _run_tool("--audit-only", "--config", "configs/smoke_preflight_armed.yaml",
              "--out-dir", str(tmp_path / "smoke"))
    other = json.loads(
        sorted((tmp_path / "smoke").glob("preflight_*.json"))[0].read_text())["coordinator"]
    smoke_stop = int(load_config(REPO_ROOT / "configs" / "smoke_preflight_armed.yaml")
                     .train.max_train_steps)
    assert other["stop_step"] == smoke_stop != block["stop_step"], (
        "the block must MOVE with the config it was resolved from; a constant would report "
        f"the same run length for both, got {other['stop_step']} and {block['stop_step']}"
    )

    _run_tool("--audit-only", "--config", "configs/dev_example.yaml",
              "--out-dir", str(tmp_path / "disarmed"))
    disarmed = json.loads(
        sorted((tmp_path / "disarmed").glob("preflight_*.json"))[0].read_text())["coordinator"]
    assert disarmed["draw_rate_abort"] is None, (
        "a DISARMED config must publish an explicit `null`, not an omitted key: absence would "
        "be indistinguishable from a block the tool forgot to fill"
    )


def test_the_report_publishes_the_audits_own_deferred_and_required_rows(
    tmp_path, monkeypatch,
) -> None:
    """The report publishes the audit's OWN deferred and required rows, read from the audit's
    result rather than re-derived from `MANIFEST`, so the two cannot disagree."""
    _run_tool("--audit-only", "--out-dir", str(tmp_path / "rows"))
    report = json.loads(sorted((tmp_path / "rows").glob("preflight_*.json"))[0].read_text())
    manifest = report["manifest"]
    shipped_deferred = [row.name for row in MANIFEST if row.status.value == "deferred"]
    assert [row["name"] for row in manifest["deferred"]] == shipped_deferred, (
        "the report's deferred block must be the SHIPPED manifest's, row for row. WPMINT "
        "Phase K-B (call K-c) gave it a real subject where WPAX Phase D left it empty and "
        "R265 / ADJ-D38 added a second (`sealbot_wr_abort`); the equality is against the "
        "manifest either way, so the published block and the audit that produced it cannot "
        f"disagree; got {manifest['deferred']!r}"
    )
    assert shipped_deferred, (
        "…and the shipped set must be non-empty, or the equality above is `[] == []` and "
        "witnesses nothing (LAW-07's vanished-subject species). This assertion is DERIVED, "
        "not a transcribed row list: the previous `== ['grad_norm_hard_abort']` tally had to "
        "be re-edited for a change it has no opinion about (R192(e))"
    )
    # The field-completeness claim needs a manifest that HAS a deferred row; against the
    # shipped one it is `all(...)` over an empty list. `audit_arming`'s `manifest=` default is
    # bound at DEF time, so BOTH the module attribute and the kwdefault are rebound.
    probe_manifest = (*MANIFEST, _SYNTHETIC_DEFERRED)
    monkeypatch.setattr(TOOL, "MANIFEST", probe_manifest)
    monkeypatch.setitem(audit_arming.__kwdefaults__, "manifest", probe_manifest)
    with_debt = TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))
    assert [row["name"] for row in with_debt["deferred"]] == [
        *shipped_deferred, _SYNTHETIC_DEFERRED.name
    ], (
        "the published deferred block is read from `AuditResult.deferred`, so a manifest "
        "carrying debt must publish ALL of it — every shipped deferred row and the probe, in "
        f"manifest order; got {with_debt['deferred']!r}"
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


# `_build_buffer` lives in the composition root as `mantis.run._select_buffer`; the raise class
# is `RepresentationRouteError`, which carries no rc, because a `src/` exception must not carry
# a CI tool's exit code.
def _identity(representation: str, encoding: str = "gnn_axis_v1"):
    """The leaves `_select_buffer` reads; the graph arm derives its ring capacity from the
    sims-regime leaves, so the stub carries run5's minted shape (50 sims, leaf 8, PCR disarmed)."""
    return SimpleNamespace(
        identity=SimpleNamespace(representation=representation, encoding=encoding),
        # The selector seeds the ring's sampler from `config.seed`, so only the attribute's
        # presence is load-bearing here.
        seed=20260719,
        search=SimpleNamespace(kind="puct"),
        selfplay=SimpleNamespace(
            leaf_batch_size=8,
            gumbel_m=16,
            mcts=SimpleNamespace(n_simulations=50),
            playout_cap=SimpleNamespace(
                standard_sims=0,
                fast_prob=0.0,
                fast_sims=50,
                full_search_prob=0.0,
                n_sims_quick=0,
                n_sims_full=0,
            ),
        ),
    )


def test_an_unknown_representation_raises_and_is_never_a_dense_default() -> None:
    """An unknown OR absent representation raises and is never a dense default — both cases are
    driven, because absent and unknown are the same error."""
    from mantis.run import _select_buffer
    from mantis.train.coordinator.dispatch import RepresentationRouteError

    # `grid` must be refused BY NAME like any other unknown, or deleting the dense arm left a
    # silent default behind.
    for representation in ("hexagonal", "", "dense", "grid", "GRAPH", "none"):
        with pytest.raises(RepresentationRouteError) as caught:
            _select_buffer(_identity(representation), 8)
        assert "LAW-11" in str(caught.value) and repr(representation) in str(caught.value), (
            "the refusal must name the law and the value it refused; got "
            f"{caught.value!s}"
        )


def test_the_ONE_declared_representation_selects_its_own_real_buffer() -> None:
    """The ONE declared representation selects its own REAL engine buffer, off the declared
    representation and never sniffed off a live module — the inverse of the refusal arm above."""
    from mantis._engine import HexgBuffer
    from mantis.run import _select_buffer

    graph = _select_buffer(_identity("graph"), 8)
    assert isinstance(graph, HexgBuffer), f"graph -> HexgBuffer; got {type(graph)}"
    assert load_config(RUN5).identity.representation == "graph", (
        "run6 is the graph arm, so the graph branch is the one the mint actually takes — "
        "pinned here so a config change that flips it is visible"
    )


def test_an_unwritable_out_dir_is_rc_41_and_never_a_silent_return(tmp_path) -> None:
    """An unwritable `--out-dir` is rc 41 and never a silent return; the rig makes it an
    existing regular file, so `mkdir` raises a real `OSError` rather than a permission trick."""
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
    """A writable out-dir still writes exactly one report, so rc 41 is not reachable on a
    healthy write."""
    _run_tool("--audit-only", "--out-dir", str(tmp_path / "ok"))
    assert len(sorted((tmp_path / "ok").glob("preflight_*.json"))) == 1


def test_a_failing_assertion_block_raises_with_the_blocks_OWN_exit_code() -> None:
    """A failing assertion block raises with that block's OWN exit code, and every entry of the
    table is driven — a table with one tested row rots in the other eight."""
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
    """The failure-code table is the design's table."""
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
    # Pinned against `RESERVED_CODES` rather than a re-typed set literal: a hand-written set
    # cannot notice a newly reserved code, which is how the four-element one went stale.
    assert set(TOOL.FAILURE_CODES.values()).isdisjoint(set(TOOL.RESERVED_CODES)), (
        "the codes the run's OWN machinery reserves must never be an assertion outcome"
    )
    assert TOOL.RESERVED_CODES == (42, 43, 44, 45, 46, 47, 48), (
        "and the band the docstring declares is 42–47; a code that joins the family without "
        f"joining this tuple is one the parent will collapse. Got {TOOL.RESERVED_CODES!r}"
    )


def test_the_a_side_verdict_is_evaluated_before_the_b_side() -> None:
    """`_verdict_exit` reports the FIRST failing block, so a run that breaks both is named by (a)."""
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
    """An unknown failure name falls back to a NAMED boot failure."""
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit({"a_sync": {"verdict": "pass", "failure": None},
                            "b_lag": {"verdict": "fail", "failure": "PreflightSomethingNew",
                                      "sub_reason": None}})
    assert caught.value.rc == TOOL.PreflightBootFailedError.rc == 33


_P = 5.0
_STEP_SEC = 0.5
_SAMPLE_TS = (0.0, 15.0, 30.0, 45.0)
_THRESHOLD = 100


class _SyncTarget:
    """`ActorSync.__init__` requires a target and `maybe_sync` calls two methods on it."""

    def sync_inference_weights(self, state_dict) -> None:
        pass

    def update_checkpoint_step(self, step: int) -> None:
        pass


def _real_syncs(tmp_path: Path, tag: str, steps, *, cadence: int = 1) -> list[dict]:
    """A REAL `ActorSync` through a REAL `JsonlEventSink`, read back off disk."""
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
    """The sample payload shape the real watchdog emits."""
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


def test_b0_needs_TWO_samples_and_one_is_not_enough(tmp_path) -> None:
    """`b0` needs TWO samples: with a single reading every transport predicate is trivially
    satisfiable and nothing about the transport has been observed. Both sides of the floor."""
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
    """`a4` discriminates a LOST SINK LINE from a missed sync: both streams carry the same
    observed sync steps, so `sync_count` is the only witness that can tell them apart."""
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
    """`a3` is a real echo of the config's cadence and not a constant: it pins that the syncs
    being read were produced by the cadence the parent BOOTED. Driven so a3 dies alone."""
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
    """The a-side sub-reason precedence is the declared table order."""
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
    """A burst that stopped short is its own named outcome, so a truncated burst cannot pass
    assertion (a) for the steps it did take."""
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


def _plant(log_dir: Path, name: str, events) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def test_the_evidence_hash_covers_every_segment_that_was_READ(tmp_path) -> None:
    """The evidence hash covers exactly the segments that were READ, and every segment that
    went into it is NAMED — `lines` once counted all segments while `sha256` hashed the last."""
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
    """A foreign run's segment is not read as this run's evidence: the glob is scoped to the
    booted run, so a stale segment cannot be concatenated into this report."""
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


def test_the_second_lag_sample_costs_a_full_file_interval_of_WALL_CLOCK(tmp_path) -> None:
    """The second lag sample costs a full `heartbeat_file_interval_sec` of wall clock, which is
    what decides rc 23 vs rc 25 for run5.

    The watchdog reuses the file interval as the sample interval and the poll loop polls FIRST,
    then waits `heartbeat_poll_interval_sec`. run5 sets file 15.0 / poll 5.0, so sample #2 lands
    on the first poll at or after t + 15.0 s — a 101-step burst yields two samples only if a
    step costs >= ~148.5 ms.
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


# Each block below names its CLASS in one sentence, and its flip-set covers the class boundary
# rather than the input that demonstrated the defect.

#: The class boundary, not the demo input: every way a config can enter `configs/` that gate 7
#: blesses and gate 12 could not see, each of them loadable and therefore required to be
#: discovered. The stem is ASSERTED undeclared by its own row below rather than chosen and
#: hoped for; the history in the docstrings still names `run6.yaml`, because that is what
#: was demonstrated.
_F1_PLANT_STEM = "run7"
_F1_PLANT_PATHS = (f"{_F1_PLANT_STEM}.yaml", f"{_F1_PLANT_STEM}.yml",
                   f"prod/{_F1_PLANT_STEM}.yaml", f"prod/nested/{_F1_PLANT_STEM}.yml",
                   f"{_F1_PLANT_STEM}.txt", f"{_F1_PLANT_STEM}.YAML",
                   f"{_F1_PLANT_STEM}.yaml.bak", ".yaml", f"{_F1_PLANT_STEM}.yamlx")


def test_the_F1_plant_stem_is_declared_by_NEITHER_tuple() -> None:
    """The premise every plant row rests on, asserted instead of assumed: the plant stem is
    declared by NEITHER tuple."""
    declared = {*PRODUCTION_CONFIGS, *(path for path, _why in EXEMPT_CONFIGS)}
    collides = sorted(d for d in declared if Path(d).name.split(".")[0] == _F1_PLANT_STEM)
    assert not collides, (
        f"the F1 plant stem {_F1_PLANT_STEM!r} is declared by {collides} — a plant at a "
        "DECLARED name tests rc 30 (disarmed) and not rc 31 (undeclared). Move the stem"
    )


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
    """An undeclared config fails the gate at ANY suffix and ANY depth — `.yaml` alone would
    pass against the old flat glob too, so the parametrisation IS the fix's evidence."""
    root = _mini_tree(tmp_path)
    _plant_disarmed(root, rel)
    result = _mini_audit(root)
    output = result.stdout + result.stderr
    assert result.returncode == 31, (
        f"a disarmed config at configs/{rel} must FAIL the gate; it was rc 0 for every "
        f"shape but the flat `.yaml` before ADJ-13. got {result.returncode}\n{output[-3000:]}"
    )
    assert f"configs/{rel}" in output and "UNDECLARED" in output, (
        "the failure must name the undeclared config by the SAME relative path a declaration "
        f"would use, subdirectory components included; got {output[-2000:]}"
    )


def test_a_declared_SUBDIRECTORY_config_is_audited_and_never_reported_STALE(
    monkeypatch, tmp_path,
) -> None:
    """A declared SUBDIRECTORY config is discovered, its declaration resolves, and it is really
    AUDITED — never reported STALE for a file the tool is looking straight at."""
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


#: Each of these files is loadable, therefore DISCOVERED, therefore UNDECLARED, therefore gate
#: 12 is RED. The complement of an enumeration rather than another enumeration: an unknown
#: suffix, a CASE variant, no suffix, a known suffix that is not final, and a dotfile.
_F1_UNRECOGNISED = ("run6.txt", "run6.YAML", "run6", "run6.yaml.bak", "run6.YML", "run6.yamL",
                    ".yaml", "run6.yamlx")


@pytest.mark.parametrize("rel", _F1_UNRECOGNISED)
def test_a_config_shaped_file_at_an_UNRECOGNISED_suffix_is_DISCOVERED_and_AUDITED(
    tmp_path, monkeypatch, rel,
) -> None:
    """A config-shaped file at an UNRECOGNISED suffix is DISCOVERED and AUDITED: the loader still
    reads it, so "loadable" and "audited" cannot come apart. Loadable, discovered, red."""
    root = _mini_tree(tmp_path)
    planted = _plant_disarmed(root, rel)
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    relposix = planted.relative_to(root).as_posix()

    assert load_config(planted).run_id == "run6", (
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
    """`mantis.run`'s launcher calls `load_config` on a FREE path, and both gates see the file."""
    canonical = _mint_run5_cpu_bootable_twin(tmp_path)
    odd = tmp_path / "run6.txt"
    odd.write_bytes(canonical.read_bytes())
    identity = config_identity_sha256(load_config(odd))
    # R348(c): the launcher demands a stamp for this identity on this tree; one stamp covers
    # both shapes because they are the same bytes.
    state_home = tmp_path / "state"
    env = {**os.environ, "XDG_STATE_HOME": str(state_home)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_STATE_HOME", str(state_home))
        write_stamp(config=load_config(odd), config_path=odd, tree_root=REPO_ROOT,
                    halts={"workspace": {"verdict": "MIRRORED", "run_dir": str(tmp_path),
                                         "bundle": {"step": 0, "files": {}},
                                         "shard": {"name": "drive", "sha256": ""}},
                           "cuda_build": {"verdict": "not_run"}},
                    booted_config_sha256="drive", burst_steps=0, report_path=tmp_path / "r.json")

    launched = _launch_until_boot_identity(odd, tmp_path / "odd_shape", env=env)
    assert "ConfigSuffixError" not in launched.stderr, (
        "the refusal must be gone from the launch path entirely, not merely downgraded\n"
        f"{launched.stderr[-2000:]}"
    )
    assert launched.witness is not None, (
        "R75: a run may be launched from a path of any shape; the loader accept-set narrowing "
        f"is out. The launcher published no `run_boot_identity`, so it never booted this file. "
        f"rc {launched.rc}\n{(launched.stdout + launched.stderr)[-2000:]}"
    )
    assert launched.witness["config_sha256"] == identity, (
        "the booted process must have loaded THE FILE AT THE ODD PATH — a matching run_id "
        "alone would also be produced by a launcher that read some other copy. got "
        f"{launched.witness['config_sha256']!r} against {identity!r}"
    )

    control = _launch_until_boot_identity(canonical, tmp_path / "canonical_shape", env=env)
    assert control.witness is not None, (
        "the control arm: the same bytes at a canonical `.yaml` shape must still launch, or "
        f"this row passes by breaking the entry point. rc {control.rc}\n"
        f"{(control.stdout + control.stderr)[-2000:]}"
    )
    assert control.witness["config_sha256"] == identity, (
        "…and the two shapes are the same config, so they boot the same identity: "
        f"{control.witness['config_sha256']!r} vs {identity!r}"
    )


def test_the_MINT_route_is_free_and_the_PREFLIGHT_covers_it_SHAPE_AGNOSTICALLY(tmp_path) -> None:
    """The MINT route takes a free `--out` path and `--config <path>` audits the result
    shape-agnostically; both arms are load-bearing."""
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
    """There is NO excluded class left under `configs/`: every shape is enumerated, including a
    genuine non-config that is not even valid YAML — red on purpose."""
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


#: A dangling symlink is a broken FILE reference, so it stays enumerated and gate 7 is LOUD; a
#: real DIRECTORY is refused by `read_text` and walked THROUGH by `rglob`, so skipping it can
#: hide nothing and it is skipped by TYPE rather than by name.
_R4_BROKEN = ("broken_symlink", "symlinked_directory")


@pytest.mark.parametrize("kind", _R4_BROKEN)
def test_a_config_SHAPED_but_BROKEN_path_is_a_LOUD_gate_7_failure_and_not_silence(
    tmp_path, kind,
) -> None:
    """A config-shaped but BROKEN path is a loud gate-7 failure naming the path, not silence:
    `rglob` will not walk through a symlinked directory, so dropping it would hide the subtree."""
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
    """A real DIRECTORY is skipped UNIFORMLY and never by its name — `configs/adir.yaml` and
    `configs/prod/` are the same path type and get the same answer."""
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
    """Gate 7 and gate 12 enumerate the SAME files on the real tree: a divergence of suffix,
    depth or sort order is red here, which is the only assertion that catches the split early."""
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


def test_one_config_reached_two_ways_is_audited_ONCE_and_not_twice(tmp_path, monkeypatch) -> None:
    """One config reached two ways is audited ONCE: `_audit_paths` unioned a plain `REPO_ROOT /
    rel` with a `.resolve()`d path, so a symlinked config held two spellings of one file."""
    root = _mini_tree(tmp_path)
    real = tmp_path / "elsewhere"
    real.mkdir()
    target = real / "run6.yaml"
    target.write_text((root / "configs" / "run6.yaml").read_text())
    (root / "configs" / "run6.yaml").unlink()
    (root / "configs" / "run6.yaml").symlink_to(target)

    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    named = TOOL._resolve_config_path(str(root / "configs" / "run6.yaml"))
    paths = TOOL._audit_paths(named)
    assert len(paths) == len(set(paths)) == len(PRODUCTION_CONFIGS), (
        "one config reached by two spellings must be ONE entry — a set of paths that "
        f"normalise differently is a set of spellings, not of configs; got {paths}"
    )
    # The expectation is DERIVED from the declaration at point of use; the subject stays
    # "two spellings collapse onto one".
    others = sorted((root / rel).resolve() for rel in PRODUCTION_CONFIGS
                    if rel != "configs/run6.yaml")
    assert paths == sorted([target, *others]), (
        f"…and both spellings must collapse onto the target; got {paths}"
    )
    bare = TOOL._audit_paths(None)
    assert bare == sorted([target, *others]), (
        "the production side alone must normalise the same way, or the union is still "
        f"comparing two schemes; got {bare}"
    )


def test_the_probe_sweep_survives_a_SYMLINK_and_never_takes_the_suite_with_it(tmp_path) -> None:
    """The probe sweep survives a SYMLINK: `Path.is_dir()` follows symlinks and `shutil.rmtree`
    refuses them, so without the symlink arm a session-scoped fixture takes the whole directory
    of tests down with it."""
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
    if os.access(unremovable, os.W_OK):
        # Measured on the box (runs as root): CAP_DAC_OVERRIDE ignores permission bits, so
        # rmtree succeeds and "DID NOT RAISE" is this arm's only possible outcome. The
        # capability is probed, not the uid.
        unremovable.chmod(0o700)
        pytest.skip("permission bits do not bind this user (root / CAP_DAC_OVERRIDE): the "
                    "loudness arm is unobservable here and runs wherever the suite runs "
                    "unprivileged")
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


#: The class boundary. `abspath` normalises `..` and makes absolute TEXTUALLY, so rows 1-3 were
#: always refused; rows 4-6 need the filesystem and every one of them escaped.
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
    """An out-dir reaching the repo BY ANY ROUTE is refused, and refused BEFORE anything is
    created — through a symlink the tool went on to `mkdir` inside the working tree."""
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
    """An out-dir reached through a symlink OUTSIDE the repo is still allowed, which is why the
    fix is `.resolve()` and not "refuse symlinks"."""
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
    """The symlink refusal is reached by the REAL CLI and writes nothing; `--audit-only` keeps it
    cheap, since the out-dir is checked before either mode runs."""
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


#: The class: an evidence-report field asserting something the run measured otherwise. Keying
#: the disclaimer on `mode` — the run's INTENT rather than its history — published "a boot was
#: spawned" beside a null child, so the rows are driven at BOTH answers to "what did the run
#: actually do".
def test_the_not_run_reason_NAMES_the_mode_the_report_was_written_in() -> None:
    """The not_run reason NAMES the mode the report was written in."""
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
    """An unknown report mode is REFUSED and never defaulted: a `.get(mode, <some mode>)` would
    publish some other mode's disclaimer, so there is no fallback."""
    with pytest.raises(TOOL.PreflightInternalError) as caught:
        TOOL._new_report("dry-run")
    assert "no code-side default" in str(caught.value)
    with pytest.raises(TOOL.PreflightInternalError):
        TOOL._not_run_reason({"mode": "dry-run", "child": None})


#: The two answers to "what did this run actually do". Not two modes — mode does not answer it.
_F3_HISTORIES = ("no_child", "child")


@pytest.mark.parametrize("history", _F3_HISTORIES)
def test_the_not_run_reason_is_DERIVED_from_the_reports_own_child_block(history) -> None:
    """The not_run reason is DERIVED from the report's own `child` block immediately before the
    write, so the sentence and the field it describes cannot disagree."""
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
    """A verdict that was REACHED is never overwritten by the disclaimer: a block with a real
    measurement in it must keep the evidence the report exists to carry."""
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
    """A real PREFLIGHT report never claims a boot its own child block denies — driven through
    the shipped process on a real config, and asserted on TRUTH rather than mode-agreement."""
    out = tmp_path / "out"
    result = _run_tool("--config", "configs/run6.yaml", "--burst-steps", "5",
                       "--out-dir", str(out), "--timeout-sec", "60", "--receipt-wait-sec", "0")
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
@pytest.mark.usefixtures("local_puller")
def test_a_BOOTED_preflight_reports_a_boot_and_names_its_childs_own_rc(tmp_path) -> None:
    """A BOOTED preflight reports a boot and names its child's own rc, whatever the child did."""
    out = tmp_path / "boot"
    result = _run_tool("--config", str(_mint_run5_cpu_twin(tmp_path)),
                       "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out), "--timeout-sec", "45", "--receipt-wait-sec", "0")
    # The child's rc is read off the report and never restated here: a run that spawned a child
    # must not carry the NOT_BOOTED disclaimer, whatever the child then did.
    assert result.returncode == 40, (result.stdout + result.stderr)[-3000:]
    report = json.loads(sorted(out.glob("preflight_*.json"))[0].read_text())
    assert report["child"] is not None and report["child"]["timed_out"] is True
    for name in ("a_sync", "b_lag"):
        reason = report["assertions"][name]["reason"]
        assert TOOL.BOOTED_REASON in reason and TOOL.NOT_BOOTED_REASON not in reason, (
            f"a boot WAS spawned on this run and the {name} disclaimer denies it: {reason!r}"
        )
        assert f"child rc {report['child']['rc']}" in reason


def test_a_report_with_no_config_block_is_still_NAMED_and_never_unnamed(tmp_path) -> None:
    """A report with no config block is still NAMED and never unnamed — the `or "unknown"`
    run_id fallback was deletable with the whole default tier green."""
    report = TOOL._new_report("preflight")
    assert report["config"] is None
    assert TOOL._report_name(report).startswith("preflight_unknown_"), (
        "a report with no config block must still carry a NAME a reader can file; got "
        f"{TOOL._report_name(report)!r}"
    )
    named = TOOL._new_report("audit")
    named["config"] = {"run_id": "run6"}
    assert TOOL._report_name(named).startswith("preflight_run6_"), (
        "…and when the config block IS populated the run_id must come from it, or the "
        f"fallback is a constant. got {TOOL._report_name(named)!r}"
    )
    out = tmp_path / "out"
    result = _run_tool("--config", "configs/run6.yaml", "--burst-steps", "5",
                       "--out-dir", str(out), "--timeout-sec", "60", "--receipt-wait-sec", "0")
    assert result.returncode == 11
    assert [path.name for path in sorted(out.glob("*.json"))][0].startswith(
        "preflight_run6_"), (
        "the rc-11 route populates `config` before `_apply_burst_override` raises, so the "
        f"real artefact is run6-named; got {sorted(path.name for path in out.glob('*.json'))}"
    )


#: `raised_by` records WHICH side of the process boundary named the outcome: a child rc inside
#: `PASS_THROUGH` is the child's own named code, anything else was diagnosed by the parent.
_X7_CHILDREN = ((12, "child"), (50, "parent"), (0, "parent"))


@pytest.mark.parametrize(("child_rc", "expected"), _X7_CHILDREN)
def test_the_reports_raised_by_field_records_WHICH_SIDE_named_the_outcome(
    monkeypatch, tmp_path, child_rc, expected,
) -> None:
    """The report's `raised_by` field records which side named the outcome, driven through the
    real `_run_child` — a real Popen, a real join, a real rc."""
    monkeypatch.setattr(TOOL, "_child_argv",
                        lambda args: [sys.executable, "-c", f"raise SystemExit({child_rc})"])
    report = TOOL._new_report("preflight")
    # `_run_child` spools the full child streams beside the report, which is what out_dir is for.
    child = TOOL._run_child(SimpleNamespace(timeout_sec=60.0, out_dir=str(tmp_path)), report)
    assert child["rc"] == child_rc
    assert child["raised_by"] == expected, (
        f"child rc {child_rc} is {'inside' if child_rc in TOOL.PASS_THROUGH else 'outside'} "
        f"PASS_THROUGH {TOOL.PASS_THROUGH}, so `raised_by` must be {expected!r} — a constant "
        f"here makes the field decoration in an EVIDENCE artefact; got {child['raised_by']!r}"
    )
    assert report["child"] is child, "the block must be published into the report, not returned"


def test_naming_a_DISARMED_config_is_AUDITED_and_never_ignored(tmp_path) -> None:
    """Naming a DISARMED config is AUDITED and never ignored: a healthy production set with a
    disarmed NAMED config is the only arm that distinguishes union from "ignore `--config`"."""
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


def test_an_inversion_on_a_NON_SAMPLING_poll_is_caught_only_by_b5as_negatives_conjunct(
    tmp_path,
) -> None:
    """An inversion on a NON-SAMPLING poll is caught only by b5a's negatives conjunct.

    The conjuncts are not redundant at run5's own constants: samples are gated on
    `heartbeat_file_interval_sec` (15.0) while polls run at `heartbeat_poll_interval_sec` (5.0),
    so two of every three polls emit no sample and an inversion between samples is invisible to
    `all(lag >= 0)`.
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
    """b5a's two conjuncts are each INDEPENDENTLY sufficient: the row above isolates
    `(not negatives)`, this one a negative SAMPLE with no negative event."""
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


def test_run5_is_bound_BY_NAME_and_is_not_freely_exemptable(monkeypatch, tmp_path) -> None:
    """run5 is bound BY NAME and is not freely exemptable: moving it to `EXEMPT_CONFIGS` with a
    written reason keeps every structural check satisfied while the run ships unaudited."""
    exempt = {rel for rel, _reason in EXEMPT_CONFIGS}
    assert "configs/run6.yaml" in PRODUCTION_CONFIGS, (
        "the config the operator is about to mint must be bound BY NAME — absence from this "
        f"tuple is not a red gate, it is silence. got {PRODUCTION_CONFIGS}"
    )
    assert "configs/run6.yaml" not in exempt
    assert PRODUCTION_CONFIGS == ("configs/run6.yaml",), (
        "R346(f) took run5 and the shakedown config, so run6 is the whole production side. A "
        "member added without a by-name pin of its own is F-P2B's escape reopened, and an "
        f"empty tuple is N-1's silence. got {PRODUCTION_CONFIGS}"
    )

    # …and the escape the pin exists to refuse, driven.
    root = _mini_tree(tmp_path)
    production = root / "configs" / "run6.yaml"
    production.write_text(production.read_text().replace("actor_lag_abort_enabled: true",
                                                         "actor_lag_abort_enabled: false"))
    bare = _mini_audit(root)
    assert bare.returncode == 30, (
        "a disarmed run5 must fail gate 12 with NO --config in sight — this is the whole of "
        f"gate 12's red-capability on the real tree (N-3); got {bare.returncode}\n"
        f"{(bare.stdout + bare.stderr)[-2000:]}"
    )

    # The swap, exactly as an unwitting editor would write it: run6 moves to EXEMPT with a
    # written reason and a config armed on both required rows takes its place.
    smoke = root / "configs" / "dev_example.yaml"
    # Armed on BOTH, so the escape cannot look closed by something other than the by-name pin:
    # `min_step: 1` with `consec: 3` at this config's `gate_interval: 1000` reaches its third
    # observation at step 3000, inside a 1,000,000-step run.
    smoke_text = smoke.read_text(encoding="utf-8")
    assert smoke_text.count("gate_interval: 1000\n") == 1, (
        "the fixture reads exactly one interval key; if the promoted config's gate_interval "
        f"spelling moved, the arming below is no longer the one the swap needs. got "
        f"{smoke_text.count('gate_interval: 1000')} loose match(es)"
    )
    promoted_length = load_config(smoke).train.max_train_steps
    assert 1 + 3 * 1000 < promoted_length, (
        "…and the armed abort must be able to FIRE inside the promoted run, or the swap is "
        f"refused for R251's reason instead of passing to expose the escape. got "
        f"{promoted_length}"
    )
    smoke.write_text(smoke_text
                     .replace("actor_lag_abort_enabled: false",
                              "actor_lag_abort_enabled: true")
                     .replace("draw_rate_abort: null",
                              "draw_rate_abort:\n"
                              "    threshold: 0.25\n"
                              "    min_step: 1\n"
                              "    N_pool_min: 50\n"
                              "    consec: 3"), encoding="utf-8")
    monkeypatch.setattr(TOOL, "PRODUCTION_CONFIGS",
                        ("configs/dev_example.yaml",
                         *[rel for rel in PRODUCTION_CONFIGS if rel != "configs/run6.yaml"]))
    monkeypatch.setattr(TOOL, "EXEMPT_CONFIGS",
                        (*[row for row in EXEMPT_CONFIGS
                           if row[0] != "configs/dev_example.yaml"],
                         ("configs/run6.yaml", "moved with a written reason")))
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    assert TOOL._config_declaration_drift() == ([], [], []), (
        "the swap keeps the partition EXACT — which is why no structural check catches it"
    )
    assert all(reason.strip() for _rel, reason in TOOL.EXEMPT_CONFIGS), (
        "…and every exemption still carries a written reason, so that check does not catch "
        "it either"
    )
    TOOL._audit_manifest_and_configs(TOOL._audit_paths(None))  # green, run6 disarmed on disk
    assert "actor_lag_abort_enabled: false" in production.read_text(), (
        "THE ESCAPE: assertion (c) just passed while the config the operator is minting sits "
        "on disk with its hard abort off. The only thing standing between that state and this "
        "tree is the by-name pin at the top of this test."
    )


# Each block below states which conjunct of a shipped predicate it is the first witness to.

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
    """Every rung of the step-ground-truth ladder is a live witness: a wrong rung is a wrong N
    and a wrong `expected` set, so the sync assertion would measure the wrong burst length."""
    samples = _model_samples(((30, 20), (60, 50))) if name == "samples_only" else []
    ground = TOOL._step_ground_truth([*events, *samples], samples)
    assert ground == {"source": source, "value": value}, (
        f"rung {name!r} must be reached and NAMED in the report; got {ground!r}"
    )


def test_the_absent_ground_truth_rung_fails_the_burst_rather_than_passing_it() -> None:
    """The absent ground-truth rung fails the burst loudly rather than passing it vacuously."""
    block = TOOL.evaluate_assertions([], cadence_steps=1, burst_steps=_N,
                                     poll_interval_sec=_P)["a_sync"]
    assert block["step_ground_truth"] == {"source": "absent", "value": 0}
    assert block["failure"] == "PreflightBurstIncompleteError", (
        f"an unmeasurable burst is rc 22, never a pass; got {block!r}"
    )


def test_b2_fails_a_FROZEN_LEARNER_and_is_not_satisfied_by_a_constant(tmp_path) -> None:
    """`b2` fails a FROZEN LEARNER and is not satisfied by a constant — without it the lag block
    reports a healthy transport over a learner that never moved."""
    syncs = _real_syncs(tmp_path, "b2frozen", [30, 50])
    samples = _model_samples(((50, 0), (50, 30), (50, 50)))
    block = _assertions(sorted([*syncs, *samples], key=lambda e: float(e["ts"])))["b_lag"]
    assert block["b2"] is False, f"a learner that never moves must fail b2; got {block!r}"
    assert block["failure"] == "PreflightLagFrozenError" and block["sub_reason"] == "learner", (
        f"…and must be diagnosed as the LEARNER side, not the actor's; got {block!r}"
    )
    assert block["b1"] is True, "the arithmetic is self-consistent — only the learner froze"


def test_b2s_positivity_conjunct_is_load_bearing_and_not_decoration(tmp_path) -> None:
    """`b2`'s `max(learners) >= 1` is a floor against a learner counter that is not a step count
    at all — a sentinel, an uninitialised negative, a sign-flipped delta."""
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
    """A config named by BOTH tuples fails the gate: "audited AND excused" is the arm an editor
    trips by adding a row without removing the other."""
    root = _mini_tree(tmp_path)
    monkeypatch.setattr(TOOL, "REPO_ROOT", root)
    monkeypatch.setattr(TOOL, "EXEMPT_CONFIGS",
                        (*EXEMPT_CONFIGS, ("configs/run6.yaml", "excused as well as audited")))
    assert TOOL._config_declaration_drift()[2] == ["configs/run6.yaml"], (
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
    """The threshold mechanism's TYPE guard is a real predicate in both directions: a bool must
    not read as an armed threshold, and a string must not raise inside the audit as an rc 1."""
    from mantis.config.armed_aborts import Mechanism

    assert Mechanism.CONFIG_THRESHOLD_GT_ZERO.is_armed(value) is armed, why


def test_the_relaunch_budget_code_is_refused_BY_NAME_and_not_as_a_generic_boot_failure(
) -> None:
    """The relaunch-budget code is refused BY NAME: dropping the arm leaves the exception TYPE
    unchanged, so the assertion is on the MESSAGE, the only observable that differs."""
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
    """Both arms of the config-path resolver are live — they agree only when `cwd == REPO_ROOT`,
    so the rows are driven from a cwd that is not the repo."""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "local.yaml"
    local.write_text(RUN5.read_text())
    assert TOOL._resolve_config_path("local.yaml") == local.resolve(), (
        "the cwd-relative arm: a config beside the operator, which REPO_ROOT cannot find"
    )
    assert TOOL._resolve_config_path("configs/run6.yaml") == RUN5.resolve(), (
        "the REPO_ROOT fallback arm: a repo-relative path from a foreign cwd, which the "
        "cwd-relative arm cannot find"
    )
    with pytest.raises(TOOL.PreflightConfigError) as caught:
        TOOL._resolve_config_path("nope.yaml")
    assert caught.value.rc == 10, "…and neither arm matching is a NAMED refusal"



#: `_git_toplevel` believes git only when it BOTH answered (rc 0) AND said something. The
#: conjuncts short-circuit, so only the OFF-DIAGONAL inputs tell them apart; the shim is a REAL
#: `git` on PATH, so nothing inside the tool is patched.
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
    """git is believed ONLY when it both answered and named a toplevel; the fallback is
    `REPO_ROOT`, which is already `.resolve()`d, so the two-normalisation defect cannot return."""
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


#: The retired constant, asserted ABSENT from every message below — it was printed in a posture
#: where it is FALSE.
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
    """The watchdog reason is DERIVED from `child["segments_scanned"]`, so the sentence and the
    `events` block are two views of one measurement and cannot disagree.

    "reason not found in the segment" was printed both when a segment was read without the
    event and when no segment was read at all — two different facts, one of them false.
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


#: The three postures of the post-child scan, as a real child process. `mode` is handed to the
#: stub child, which writes the segment (or does not) and then exits with a watchdog rc.
_SCAN_CHILD = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "out, mode, run_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]\n"
    "if mode != 'nologs':\n"
    "    logs = out / 'logs'\n"
    "    logs.mkdir(parents=True, exist_ok=True)\n"
    "    rows = ([{'event': 'heartbeat_watchdog_fired', 'reason': 'actor_lag_exceeded',\n"
    "              'code': 42}] if mode == 'fired' else [])\n"
    "    rows.append({'event': 'actor_lag_sample', 'learner_step': 1,\n"
    "                 'actor_ckpt_step': 0, 'lag_steps': 1})\n"
    "    (logs / ('events_' + run_id + '_seg0000.jsonl')).write_text(\n"
    "        ''.join(json.dumps(row) + chr(10) for row in rows), encoding='utf-8')\n"
    "raise SystemExit(42)\n"
)

_SCAN_POSTURES = (
    ("fired", "actor_lag_exceeded", 1),
    ("quiet", "1 segment(s) were read", 1),
    ("nologs", "NO segment was read", 0),
)


@pytest.mark.parametrize(("mode", "expected", "segments"), _SCAN_POSTURES,
                         ids=[posture[0] for posture in _SCAN_POSTURES])
@pytest.mark.usefixtures("local_puller")
def test_the_POST_CHILD_segment_scan_is_driven_and_agrees_with_the_reports_OWN_events_block(
    monkeypatch, tmp_path, mode, expected, segments,
) -> None:
    """The POST-CHILD segment scan is driven and agrees with the report's OWN events block.

    `_child_argv` is redirected on the tool module object and nothing else is: `_run_child` is
    the real one and the stub child is a real process writing a real JSONL segment. The
    assertion is the BICONDITIONAL, so a mutation changing only the sentence is caught by the
    disagreement; the `fired` segment carries a trailing non-watchdog event so a filter forced
    to `True` fails too.
    """
    out_dir = tmp_path / "out"
    target = _mint_run5_cpu_twin(tmp_path)
    run_id = load_config(target).run_id
    monkeypatch.setattr(TOOL, "_child_argv",
                        lambda args: [sys.executable, "-c", _SCAN_CHILD, str(out_dir), mode,
                                      run_id])
    report = TOOL._new_report("preflight")
    args = SimpleNamespace(config=str(target), burst_steps=_RUN5_BURST, out_dir=str(out_dir),
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
    """A child with NO stderr tail reports an EMPTY tail and never a placeholder — `tail` is also
    what the stderr sniff reads, so a non-empty fallback manufactures a tree-defect diagnosis."""
    child = {"rc": 1, "timed_out": False, **extra}
    with pytest.raises(TOOL.PreflightBootFailedError) as caught:
        TOOL._classify_child(child)
    assert str(caught.value) == "child exited 1:\n", (
        "a child that produced no stderr must render an EMPTY tail — a fallback constant "
        f"lands verbatim in the report's failure message. got {str(caught.value)!r}"
    )


def test_the_child_block_carries_the_childs_OWN_stdout_AND_stderr_tails(monkeypatch, tmp_path) -> None:
    """The child block carries the child's OWN stdout AND stderr tails; both are asserted in one
    row, so a fix that wires stdout to stderr is red too."""
    monkeypatch.setattr(TOOL, "_child_argv", lambda args: [
        sys.executable, "-c",
        "import sys; sys.stdout.write('OUT-MARKER'); sys.stderr.write('ERR-MARKER'); "
        "raise SystemExit(7)"])
    report = TOOL._new_report("preflight")
    # out_dir carries the child's spooled streams beside the report.
    child = TOOL._run_child(SimpleNamespace(timeout_sec=60.0, out_dir=str(tmp_path)), report)
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
    """The verdict message carries the INVERSION REASON when there is one — the sentence that
    tells an operator which of the two inversion outcomes it was."""
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
    """A verdict with NO inversion reason ends cleanly and never prints a placeholder, asserted
    as equality on the whole message because a trailing placeholder is what a substring misses."""
    with pytest.raises(TOOL.PreflightAssertionsFailedError) as caught:
        TOOL._verdict_exit(_lag_blocks(None))
    assert str(caught.value) == "PreflightLagFrozenError sub_reason='both'", (
        "with no inversion reason the message must END at the sub_reason — the `.strip()` "
        f"exists for exactly that. got {str(caught.value)!r}"
    )


# CLASS: an evidence artifact that reports a run's LENGTH without reporting what that length
# bought. Every row below drives the shipped functions on the real committed configs.
def _tier_config(path: Path):
    return load_config(path)


def test_every_mint_tier_has_a_NOT_PROVEN_entry_and_there_is_NO_default() -> None:
    """Every mint tier has a NOT-PROVEN entry and there is NO default: a `.get(tier, …)` would
    publish one tier's reachability claim under another tier's run."""
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
    """The burst tier is DERIVED from the config's OWN floor rows, so a config that arms no
    draw-rate abort can never be called `full` merely for clearing every floor it has."""
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
    """A PRODUCTION config can never be preflighted in the SHORT tier: the required draw-rate row
    puts `min_step + 1` into the floors and a shorter burst is refused at rc 11."""
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


#: The two answers to "did this run actually cover a tier". Not two tiers — a tier is REQUESTED
#: by the burst and COVERED by the outcome.
_TIER_HISTORIES = ("no_verdict", "verdict")


@pytest.mark.parametrize("history", _TIER_HISTORIES)
def test_a_tier_is_COVERED_only_when_the_run_reached_a_verdict(history) -> None:
    """A tier is COVERED only when the run reached a verdict, re-derived from the report's own
    (a)/(b) blocks rather than from the burst length."""
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
    """The tier disclaimer is RE-DERIVED at write time and is never the prediction `_new_report`
    stamped."""
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
    """The `none` tier disclaimer is TRUE in mode AUDIT and not only at rc 11: mode AUDIT requests
    no burst, so a sentence about a refused burst would be false there."""
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
    """A refused burst publishes tier `none` and owes BOTH tiers."""
    out_dir = tmp_path / "refused"
    result = _run_tool("--config", "configs/run6.yaml", "--burst-steps", str(_RUN5_BURST - 1),
                       "--out-dir", str(out_dir), "--timeout-sec", "60", "--receipt-wait-sec", "0")
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
@pytest.mark.usefixtures("local_puller")
def test_the_real_preflight_publishes_the_tier_it_RAN_and_what_it_does_NOT_prove(
        tmp_path) -> None:
    """The real preflight publishes the tier it RAN and what that tier does NOT prove; the tier
    arithmetic is run5's own floor, carried through the twin's identical draw-rate block."""
    out_dir = tmp_path / "tiered"
    result = _run_tool("--config", str(_mint_run5_cpu_twin(tmp_path)),
                       "--burst-steps", str(_RUN5_BURST),
                       "--out-dir", str(out_dir), "--timeout-sec", "45", "--receipt-wait-sec", "0")
    # The TIER ARITHMETIC does not move with the outcome: the tier block is published on every
    # terminating preflight, and a run that proved LESS must still say what it did not prove.
    assert result.returncode == 40, (result.stdout + result.stderr)[-3000:]
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


# Driving gate 12 against a PERTURBED run5 needs the mini-tree rig, so these two rows live here
# rather than beside the other gate-12 tests.
def test_a_production_config_with_the_terminal_eval_off_fails_gate_12(tmp_path) -> None:
    """A production config with the terminal eval switched off fails gate 12 — the unperturbed
    tree must be green and the perturbed one red by name, or neither arm proves anything."""
    import yaml

    root = _mini_tree(tmp_path)
    healthy = _mini_audit(root)
    assert healthy.returncode == 0, (
        "premise: an unperturbed mini tree is as green as the real one, so the red below is "
        f"the flag and not the rig; got {healthy.returncode}\n"
        f"{(healthy.stdout + healthy.stderr)[-2000:]}"
    )

    run5_copy = root / "configs" / "run6.yaml"
    document = yaml.safe_load(run5_copy.read_text(encoding="utf-8"))
    assert document["train"]["terminal_eval_enabled"] is True, (
        "premise: run5 mints the terminal eval ON, which is what makes the row REQUIRED "
        f"rather than DEFERRED; got {document['train']['terminal_eval_enabled']!r}"
    )
    document["train"]["terminal_eval_enabled"] = False
    run5_copy.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    perturbed = _mini_audit(root)
    output = perturbed.stdout + perturbed.stderr
    assert perturbed.returncode == TOOL.PreflightArmingAuditError.rc == 30, (
        "a REQUIRED armed-abort row disarmed on a production config is gate 12's whole "
        f"subject and must be rc 30; got {perturbed.returncode}\n{output[-3000:]}"
    )
    assert "terminal_eval_broken" in output, (
        f"…and the failure must NAME the disarmed row so an operator knows which key to "
        f"restore; got {output[-2000:]}"
    )
    assert "train.terminal_eval_enabled" in output, (
        f"…and the arming surface, which is the thing they actually edit; got {output[-2000:]}"
    )


def test_a_child_rc_48_is_the_runs_own_ARMED_ABORT_and_is_never_collapsed_to_33() -> None:
    """A child rc of 48 is the run's own armed abort and never collapses to 33; the constant is
    imported from the one authority, never re-typed here."""
    from mantis.monitor.heartbeat import TERMINAL_EVAL_BROKEN_EXIT_CODE

    assert TERMINAL_EVAL_BROKEN_EXIT_CODE in TOOL.ARMED_ABORT_CODES, (
        "48 is a COOPERATIVE armed abort — the run decided, unwound, saved and returned the "
        "manifest's number — so the parent must classify it with 46 and 47 and not as a boot "
        f"failure; got {TOOL.ARMED_ABORT_CODES!r}"
    )
    with pytest.raises(TOOL.PreflightArmedAbortFiredError) as caught:
        TOOL._classify_child(_child(TERMINAL_EVAL_BROKEN_EXIT_CODE))
    assert caught.value.rc == TERMINAL_EVAL_BROKEN_EXIT_CODE, (
        "an armed abort's authored rc must PROPAGATE unchanged — a supervisor must read the "
        f"same number on both sides of this tool; got {caught.value.rc}"
    )
    assert caught.value.rc != TOOL.PreflightBootFailedError.rc, (
        "…and it must not be the rc 33 an unregistered code collapses to"
    )
    assert TERMINAL_EVAL_BROKEN_EXIT_CODE not in TOOL.PASS_THROUGH, (
        "the premise: 48 cannot ride the [10, 41] pass-through arm, which is why it needs "
        f"the armed-abort arm; got PASS_THROUGH={TOOL.PASS_THROUGH!r}"
    )
    assert TERMINAL_EVAL_BROKEN_EXIT_CODE not in TOOL.WATCHDOG_CODES, (
        "…and it is NOT a watchdog code: it is not delivered by `os._exit` from a fire path "
        "and must not be diagnosed as a stall"
    )
    assert TERMINAL_EVAL_BROKEN_EXIT_CODE in TOOL.RESERVED_CODES, (
        "…while still being inside the reserved 42–47+ band the tool's docstring declares, "
        f"which is DERIVED from the three tuples; got {TOOL.RESERVED_CODES!r}"
    )
