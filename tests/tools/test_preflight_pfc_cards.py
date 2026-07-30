"""Phase PFC card oracles (WPCLEAN): CARD-PREFLIGHT-OUTDIR-REUSE + CARD-PREFLIGHT-CHILD-
STDERR-BUDGET, driven on the REAL tool (subprocess / in-process module load — never a
stand-in). Deliberately NOT in the frozen set (the process-file precedent: non-frozen so
process-half fixes stay editable). The two R43-gated cards (SPLIT-PARENT-HALF,
ORACLE-OUTDIR-CLEANUP) are QUEUED, not tested here — see wp/WPCLEAN/ADJUDICATION_QUEUE.md.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "preflight_mint.py"
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke_preflight_armed.yaml"
SMOKE_RUN_ID = "smoke_preflight_armed"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("preflight_mint_pfc_test", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_tool(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL_PATH), *argv],
                          capture_output=True, text=True, check=False, cwd=REPO_ROOT)


def test_out_dir_reuse_error_carries_rc_15_uniquely(tool):
    assert issubclass(tool.PreflightOutDirReusedError, tool.PreflightError)
    assert tool.PreflightOutDirReusedError.rc == 15
    taken = {cls.rc for name in dir(tool)
             if isinstance((cls := getattr(tool, name)), type)
             and issubclass(cls, tool.PreflightError)
             and cls is not tool.PreflightOutDirReusedError}
    assert 15 not in taken


@pytest.mark.integration
def test_a_dirty_same_run_id_out_dir_is_refused_before_the_boot(tmp_path):
    """The card's hole: `_read_segment` scopes by run_id, so a same-run_id reuse reads a
    PREVIOUS burst's events as this run's evidence. The refusal must land BEFORE any boot —
    pinned by wall-clock shape: rc 15 in far less time than a torch boot could take."""
    out = tmp_path / "reused"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / f"events_{SMOKE_RUN_ID}_seg0000.jsonl").write_text('{"event":"x"}\n')
    res = _run_tool("--config", str(SMOKE_CONFIG), "--burst-steps", "16",
                    "--out-dir", str(out), "--timeout-sec", "60")
    assert res.returncode == 15, res.stdout + res.stderr
    assert SMOKE_RUN_ID in res.stdout + res.stderr
    assert "seg0000" in res.stdout + res.stderr


@pytest.mark.integration
def test_a_foreign_run_ids_litter_does_not_trip_the_refusal(tmp_path):
    """The discriminating negative: the refusal is scoped to THIS run_id's segments —
    foreign litter proceeds to the boot (witnessed by the run reaching a real verdict,
    rc 0, exactly as on a clean dir)."""
    out = tmp_path / "littered"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "events_some_other_run_seg0000.jsonl").write_text('{"event":"x"}\n')
    res = _run_tool("--config", str(SMOKE_CONFIG), "--burst-steps", "16",
                    "--out-dir", str(out), "--timeout-sec", "300")
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.integration
def test_child_streams_spool_in_full_beside_the_report(tool, tmp_path):
    """CARD-PREFLIGHT-CHILD-STDERR-BUDGET, the spool arm: the report's tails are a VIEW,
    the spools are the record. Driven through the real `_run_child` with a child that dies
    on a config its own loader refuses (fast, deterministic): the spool files must exist,
    carry the FULL streams, and each tail must be exactly the spool's last 4000 chars."""
    bad_config = tmp_path / "not_a_config.yaml"
    bad_config.write_text("this is: [not, a, run, config\n")
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(config=str(bad_config), burst_steps=16,
                           out_dir=str(out), timeout_sec=120.0, device="cpu")
    report: dict = {}
    child = tool._run_child(args, report)
    assert child["rc"] != 0
    stdout_spool = Path(child["stdout_spool"])
    stderr_spool = Path(child["stderr_spool"])
    assert stdout_spool.is_file() and stderr_spool.is_file()
    assert stdout_spool.parent == out and stderr_spool.parent == out
    assert child["stderr_tail"] == stderr_spool.read_text(encoding="utf-8")[-4000:]
    assert child["stdout_tail"] == stdout_spool.read_text(encoding="utf-8")[-4000:]
    assert json.dumps(report["child"]) # the block is JSON-serializable with the new keys
