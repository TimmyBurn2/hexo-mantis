"""WPTS Phase F — the armed-smoke burst oracle (ADJ-24 / R103), and TD-1's end-to-end proof.

THE LIVE CONSUMER of `configs/smoke_preflight_armed.yaml` (LAW-08; the config's
`EXEMPT_CONFIGS` row names this file). One drive, everything real (R64 posture): the REAL
`tools/ci_gates/preflight_mint.py` as a subprocess boots the REAL tree off the minted armed
smoke config — real `Trainer` (GnnNet), real `WorkerPool` self-play on CPU, real graph
replay buffer — clears every arming floor, fills the buffer with REAL games, runs a REAL
16-step graph training burst through the coordinator's declared dispatcher, and exits GREEN.

Why this is TD-1's end-to-end proof (R103's closing clause): the burst cannot complete
without a working training step. Before WPTS Phase T, `step.py`'s straight arm called a
`trainer.train_step` that did not exist — this exact drive would die at the first
post-warmup step. rc 0 with `tier.covered == ["sync_lag", "full"]` is therefore the whole
learner half executing, not a boot smoke.

Why rc 0 is reachable off-run5 at all: every other non-run5 config deliberately disarms the
REQUIRED abort rows (R59) and is refused at the arming audit (rc 30, measured by WPBRIDGE).
`smoke_preflight_armed.yaml` arms BOTH required rows at burst-scale guard values (R103's
grant), which is what makes a fast preflight rehearsal target exist.

INTEGRATION tier: a real ~30 s CPU boot + burst + terminal-eval witness.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
CONFIG = REPO_ROOT / "configs" / "smoke_preflight_armed.yaml"

#: 16 is the minimum legal burst for this config's guard values plus one:
#: `monitor.actor_lag_threshold_steps` 14 (floor 15), `train.draw_rate_abort.min_step` 10
#: (floor 11), `train.actor_sync_cadence_steps` 2 (floor 3).
BURST_STEPS = 16


def test_armed_smoke_config_completes_a_bounded_burst_through_the_real_preflight(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--config", str(CONFIG),
         "--burst-steps", str(BURST_STEPS), "--out-dir", str(tmp_path),
         "--timeout-sec", "400", "--device", "cpu"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=500,
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
