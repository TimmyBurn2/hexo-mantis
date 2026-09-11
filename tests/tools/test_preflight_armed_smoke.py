"""The armed-smoke burst oracle: the LIVE CONSUMER of `configs/smoke_preflight_armed.yaml`.

The real preflight tool, as a subprocess, boots the real tree off the minted armed smoke and
runs a real 16-step burst; rc 0 with `tier.covered == ["sync_lag", "full"]` is the whole learner
half executing. It is the one non-run config that arms BOTH required abort rows, which is what
makes a fast preflight rehearsal target exist. Integration tier.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mantis.config.loader import load_config
from mantis.config.preflight_stamp import require_preflight_stamp

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("planted_durable_mounts")]

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
CONFIG = REPO_ROOT / "configs" / "smoke_preflight_armed.yaml"

#: 16 is the minimum legal burst for this config's guard values plus one:
#: `monitor.actor_lag_threshold_steps` 14 (floor 15), `train.draw_rate_abort.min_step` 10
#: (floor 11), `train.actor_sync_cadence_steps` 2 (floor 3).
BURST_STEPS = 16


def test_armed_smoke_config_completes_a_bounded_burst_through_the_real_preflight(
    tmp_path, monkeypatch, preflight_budget_sec, preflight_harness_ceiling_sec
):
    # The R348(c) stamp store is redirected so a test burst never stamps the host's real store.
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--config", str(CONFIG),
         "--burst-steps", str(BURST_STEPS), "--out-dir", str(tmp_path),
         "--timeout-sec", str(preflight_budget_sec)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=preflight_harness_ceiling_sec,
        env={**os.environ, "XDG_STATE_HOME": str(state_home)},
    )
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, f"preflight not green (rc {proc.returncode}):\n{tail}"

    reports = sorted(tmp_path.glob("preflight_*.json"))
    assert reports, f"no report written:\n{tail}"
    report = json.loads(reports[-1].read_text())

    assert report["verdict"] == "pass" and report["rc"] == 0
    assert report["config"]["run_id"] == "smoke_preflight_armed"
    assert report["config"]["representation"] == "graph", (
        "the proof must run run5's representation — the graph route is TD-1's subject"
    )
    # (c) both REQUIRED abort rows ARMED on a non-run5 config — the R103 grant, audited.
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
    # R348(c): a green preflight leaves the stamp `mantis.run` will demand, on THIS tree.
    stamp = require_preflight_stamp(load_config(CONFIG), tree_root=REPO_ROOT)
    assert Path(report["preflight_stamp"]).is_relative_to(state_home)
    assert stamp["halts"]["workspace"] == report["workspace"]
    assert stamp["halts"]["cuda_build"] == report["cuda_build"]
