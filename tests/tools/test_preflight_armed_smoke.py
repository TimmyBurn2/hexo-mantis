"""The armed-smoke burst oracle: the LIVE CONSUMER of `configs/smoke_preflight_armed.yaml`.

The real preflight tool, as a subprocess, boots the real tree off the minted armed smoke and
runs a real 16-step burst; rc 0 with `tier.covered == ["sync_lag", "full"]` is the whole learner
half executing. It is the one non-run config that arms BOTH required abort rows, which is what
makes a fast preflight rehearsal target exist. Integration tier.

Every row here reads ONE module-scoped boot. The child's own log directory is evidence no AST
census can forge: `run_boot_identity` and `resolved_config` are emitted by `compose_run` and by
nothing else, and what the re-exec'd interpreter did is observable only from what it left behind.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.preflight_stamp import (
    read_stamp,
    require_preflight_stamp,
)
from _code_text import code_text

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("local_puller")]

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
CONFIG = REPO_ROOT / "configs" / "smoke_preflight_armed.yaml"

#: 16 is the minimum legal burst for this config's guard values plus one:
#: `monitor.actor_lag_threshold_steps` 14 (floor 15), `train.draw_rate_abort.min_step` 10
#: (floor 11), `train.actor_sync_cadence_steps` 2 (floor 3).
BURST_STEPS = 16

#: The events `compose_run` alone publishes into a run's own segment.
_COMPOSER_EVENTS = ("run_boot_identity", "resolved_config")

#: Another run's segment, planted in the out-dir before the boot: the refusal is scoped to THIS
#: run_id, so the boot must proceed past it.
_FOREIGN_SEGMENT = "events_some_other_run_seg0000.jsonl"


@pytest.fixture(scope="module")
def armed_preflight(local_puller, tmp_path_factory, preflight_budget_sec,
                    preflight_harness_ceiling_sec) -> SimpleNamespace:
    """Spawn ONE real preflight over an out-dir holding a foreign run's segment, shared by every row below."""
    root = tmp_path_factory.mktemp("armed_preflight")
    out_dir = root / "out"
    # The stamp store is redirected so a test burst never stamps the host's real store.
    state_home = root / "state"
    litter = out_dir / "logs" / _FOREIGN_SEGMENT
    litter.parent.mkdir(parents=True)
    litter.write_text('{"event":"x"}\n', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--config", str(CONFIG),
         "--burst-steps", str(BURST_STEPS), "--out-dir", str(out_dir),
         "--timeout-sec", str(preflight_budget_sec), "--receipt-wait-sec", "120"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=preflight_harness_ceiling_sec,
        env={**os.environ, "XDG_STATE_HOME": str(state_home)},
    )
    # Moved aside once the boot has run past it, so the segment readers below see the child's alone.
    litter.rename(root / _FOREIGN_SEGMENT)
    return SimpleNamespace(proc=proc, root=root, out_dir=out_dir, state_home=state_home)


def _child_events(out_dir: Path) -> list[dict]:
    """Return the CHILD's own event stream — only `compose_run` writes here."""
    segments = sorted((out_dir / "logs").glob("*.jsonl"))
    assert segments, (
        f"the child left no JSONL segment under {out_dir / 'logs'} — it never reached "
        "`build_run_safety`, which means it never reached the composition root"
    )
    return [json.loads(line) for segment in segments
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_armed_smoke_config_completes_a_bounded_burst_through_the_real_preflight(
    armed_preflight, monkeypatch
):
    state_home = armed_preflight.state_home
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    proc = armed_preflight.proc
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, f"preflight not green (rc {proc.returncode}):\n{tail}"

    reports = sorted(armed_preflight.out_dir.glob("preflight_*.json"))
    assert reports, f"no report written:\n{tail}"
    report = json.loads(reports[-1].read_text(encoding="utf-8"))

    assert report["verdict"] == "pass" and report["rc"] == 0
    assert report["config"]["run_id"] == "smoke_preflight_armed"
    assert report["config"]["representation"] == "graph", (
        "the proof must run run5's representation — the graph route is TD-1's subject"
    )
    # (c) both REQUIRED abort rows ARMED on a non-run5 config, audited.
    assert report["assertions"]["c_arming"]["verdict"] == "pass"
    # (a) the burst COMPLETED: the independent witness (terminal_eval / shutdown_save —
    # never the actor_sync stream auditing itself) saw exactly BURST_STEPS learner steps.
    a = report["assertions"]["a_sync"]
    assert a["verdict"] == "pass"
    assert int(a["step_ground_truth"]["value"]) == BURST_STEPS
    assert a["step_ground_truth"]["source"] != "absent"
    # (b) lag transport measured on live samples.
    assert report["assertions"]["b_lag"]["verdict"] == "pass"
    # The two mint tiers this tool can demonstrate are BOTH demonstrated by this drive.
    assert report["tier"]["tier"] == "full"
    assert report["tier"]["covered"] == ["sync_lag", "full"]
    assert report["child"]["rc"] == 0 and report["child"]["timed_out"] is False
    # A green preflight leaves a stamp carrying the halts' readings; the local
    # puller receipted the burst's bundle and first shard, so `mantis.run` accepts the stamp.
    config = load_config(CONFIG)
    stamp = read_stamp(config_identity_sha256(config))
    assert Path(report["preflight_stamp"]).is_relative_to(state_home)
    assert stamp["halts"]["workspace"] == report["workspace"]
    assert stamp["halts"]["cuda_build"] == report["cuda_build"]
    workspace = stamp["halts"]["workspace"]
    assert workspace["verdict"] == "MIRRORED"
    # A clean completion writes a checkpoint and NO bundle, so the artefact
    # the loop was proven on is the burst's checkpoint of record, at the burst's own step.
    assert "bundle" not in workspace, workspace
    assert workspace["checkpoint"]["name"].startswith(
        f"smoke_preflight_armed_{BURST_STEPS:08d}_"), workspace
    assert workspace["shard"]["name"].startswith("games_smoke_preflight_armed_seg")
    accepted = require_preflight_stamp(config, tree_root=REPO_ROOT)
    assert accepted["config_sha256"] == stamp["config_sha256"]


def test_a_foreign_run_ids_litter_does_not_trip_the_refusal(armed_preflight):
    """The discriminating negative: the refusal is scoped to THIS run_id's segments —
    foreign litter proceeds to the boot (witnessed by the run reaching a real verdict,
    rc 0, exactly as on a clean dir).

    The budget comes from `conftest.PREFLIGHT_BUDGET_SEC` rather than
    from a literal here. It used to read `300`, which passes on this host with ~46% margin and
    went red on the migration box — the grounds, and the three measurements behind the value,
    are in the conftest beside the constant."""
    res = armed_preflight.proc
    assert res.returncode == 0, res.stdout + res.stderr
    out = armed_preflight.out_dir
    report = json.loads(sorted(out.glob("preflight_*.json"))[-1].read_text(encoding="utf-8"))
    assert Path(report["preflight_stamp"]).is_relative_to(armed_preflight.root), (
        f"the green burst stamped {report['preflight_stamp']}, outside tmp_path: the host's store"
    )


def test_the_child_boots_green_with_no_device_flag_on_the_argv(
    armed_preflight, tmp_path_factory
) -> None:
    """The child boots the CONFIG's own device, with no `--device` flag on the argv.

    Killer: restore the flag — a `--device cpu` invocation can then false-clear a cuda run's
    memory wall. This is the only oracle that RUNS the argv the parent builds.
    """
    proc, out_dir = armed_preflight.proc, armed_preflight.out_dir
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, (
        f"the preflight must be green on the new path (rc {proc.returncode}):\n{tail}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert reports, f"no evidence report written:\n{tail}"
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["verdict"] == "pass" and report["child"]["rc"] == 0
    stamp = Path(report["preflight_stamp"])
    assert stamp.is_relative_to(tmp_path_factory.getbasetemp()), (
        f"the green burst stamped {stamp}, outside the test's tmp tree: the host's store"
    )


def test_the_child_process_left_the_composition_roots_own_boot_events(armed_preflight) -> None:
    """The child's segment carries the composition root's own boot events, exactly once each.

    Killer: re-point the child at a shim that rebuilds the composition inline — rc stays 0
    and the evidence report stays `pass`, so no other oracle observes the child PROCESS.
    """
    names = [event.get("event") for event in _child_events(armed_preflight.out_dir)]
    for event in _COMPOSER_EVENTS:
        assert names.count(event) == 1, (
            f"the child's segment must carry exactly one {event} — the composition root's "
            f"own boot record; got {names.count(event)} in {sorted(set(names))}"
        )


def test_the_childs_published_identity_is_the_config_it_actually_composed(
    armed_preflight,
) -> None:
    """The identity the child publishes is the MINTED config's own — the burst is a stop bound
    over it, never a mutation (CARD-STAMP-FLOOR) — so a child that read a different file, or a
    tool that shaped a burst config, is a named mismatch. Killer: a `max_train_steps` override.
    """
    booted = load_config(CONFIG)

    identity = next(event for event in _child_events(armed_preflight.out_dir)
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
    source = code_text(TOOL)
    for token in ("init_trainer", "WorkerPool", "HexgBuffer", "ReplayBuffer",
                  "build_run_safety", "StepCoordinatorConfig"):
        assert token not in source, (
            f"the tool still names {token!r}: the boot lives at `mantis.run` now, and a CI "
            "gate that builds its own collaborators is the one-authority violation this WP "
            "exists to end (R121(a))"
        )
