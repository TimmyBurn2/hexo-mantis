"""The preflight child boots the composition root, proved at the process boundary.

`run_boot_identity` and `resolved_config` are emitted by `compose_run` and by nothing else in
the tree, so their presence in the CHILD's own log directory is evidence no AST census can
forge: the parent re-execs a fresh interpreter, and what that process did is observable only
from what it left behind.

INTEGRATION tier: a real ~30 s CPU boot + burst. No fakes — real tool, subprocess, config.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.schema import RunConfig

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("local_puller")]

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "ci_gates" / "preflight_mint.py"
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"

#: The burst length `test_preflight_armed_smoke.py` drives — one authority.
_BURST_STEPS = 16

#: The events `compose_run` alone publishes into a run's own segment.
_COMPOSER_EVENTS = ("run_boot_identity", "resolved_config")


@pytest.fixture(scope="module")
def preflight_child(tmp_path_factory, preflight_budget_sec, preflight_harness_ceiling_sec):
    """Spawn ONE real preflight, shared by every assertion below."""
    out_dir = tmp_path_factory.mktemp("preflight_convergence")
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--config", str(_CONFIG),
         "--burst-steps", str(_BURST_STEPS), "--out-dir", str(out_dir),
         "--timeout-sec", str(preflight_budget_sec), "--receipt-wait-sec", "120"],
        cwd=str(_REPO), capture_output=True, text=True,
        timeout=preflight_harness_ceiling_sec,
    )
    return proc, out_dir


def _code_text(path: Path) -> str:
    """Return source with COMMENT / STRING / f-string-literal tokens removed.

    FSTRING_MIDDLE is 3.12+ (PEP 701); on the 3.11 floor f-strings lex as STRING.
    """
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(tok.string for tok in tokenize.tokenize(handle.readline)
                         if tok.type not in skip)


def _child_events(out_dir: Path) -> list[dict]:
    """Return the CHILD's own event stream — only `compose_run` writes here."""
    segments = sorted((out_dir / "logs").glob("*.jsonl"))
    assert segments, (
        f"the child left no JSONL segment under {out_dir / 'logs'} — it never reached "
        "`build_run_safety`, which means it never reached the composition root"
    )
    return [json.loads(line) for segment in segments
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_child_boots_green_with_no_device_flag_on_the_argv(preflight_child) -> None:
    """The child boots the CONFIG's own device, with no `--device` flag on the argv.

    Killer: restore the flag — a `--device cpu` invocation can then false-clear a cuda run's
    memory wall. This is the only oracle that RUNS the argv the parent builds.
    """
    proc, out_dir = preflight_child
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, (
        f"the preflight must be green on the new path (rc {proc.returncode}):\n{tail}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert reports, f"no evidence report written:\n{tail}"
    report = json.loads(reports[-1].read_text())
    assert report["verdict"] == "pass" and report["child"]["rc"] == 0


def test_the_child_process_left_the_composition_roots_own_boot_events(preflight_child) -> None:
    """The child's segment carries the composition root's own boot events, exactly once each.

    Killer: re-point the child at a shim that rebuilds the composition inline — rc stays 0
    and the evidence report stays `pass`, so no other oracle observes the child PROCESS.
    """
    _proc, out_dir = preflight_child
    names = [event.get("event") for event in _child_events(out_dir)]
    for event in _COMPOSER_EVENTS:
        assert names.count(event) == 1, (
            f"the child's segment must carry exactly one {event} — the composition root's "
            f"own boot record; got {names.count(event)} in {sorted(set(names))}"
        )


def test_the_childs_published_identity_is_the_config_it_actually_composed(
    preflight_child,
) -> None:
    """The identity the child publishes is of the config it actually composed.

    The hash is recomputed from the config the child was ASKED to boot, with the burst
    override applied the way the child applies it, so a child that read a different file is a
    named mismatch. The run id must be the MINTED one, never a default a log cannot attribute.

    Killer: publish the pre-override config's hash, or compose a second config.
    """
    _proc, out_dir = preflight_child
    raw = load_config(_CONFIG).model_dump()
    raw["train"]["max_train_steps"] = _BURST_STEPS
    booted = RunConfig.model_validate(raw)

    identity = next(event for event in _child_events(out_dir)
                    if event.get("event") == "run_boot_identity")
    assert identity["config_sha256"] == config_identity_sha256(booted), (
        "the child published the identity of a DIFFERENT config than the one it was asked "
        "to boot — the F-B1 defect, at the seam F-B1 closed"
    )
    assert identity["run_id"] == booted.run_id == "smoke_preflight_armed", (
        f"the published run id must be the config's own; got {identity['run_id']!r}"
    )


def test_the_tool_no_longer_builds_a_single_collaborator_for_itself() -> None:
    """The tool constructs none of the run's collaborators itself.

    The scan is over CODE with comment/string tokens removed: the tool's prose goes on NAMING
    these builders, and a raw-text census would flag that and teach people to word comments
    around a gate.

    Killer: leave one construction behind "just for the preflight" — a tool that builds even
    one collaborator differently preflights a run nobody will launch.
    """
    source = _code_text(_TOOL)
    for token in ("init_trainer", "WorkerPool", "HexgBuffer", "ReplayBuffer",
                  "build_run_safety", "StepCoordinatorConfig"):
        assert token not in source, (
            f"the tool still names {token!r}: the boot lives at `mantis.run` now, and a CI "
            "gate that builds its own collaborators is the one-authority violation this WP "
            "exists to end (R121(a))"
        )
