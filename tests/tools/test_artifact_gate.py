"""Test the tester: gate 6's size thresholds must arm (LAW-07 mutation self-test).

Runs the REAL tools/ci_gates/artifact_gate.py via subprocess against a throwaway git repo
built in tmp. Pins BOTH sides of the R8 ruling: the tests/fixtures/ carve-out is a raised
10 MB ceiling, not an exemption (a fixture over it must be rejected), while the 1 MB rule
outside fixtures and the jsonl carve-out are unchanged by that restructure.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "ci_gates" / "artifact_gate.py"

MAX_ADDED_BYTES = 1_000_000
MAX_FIXTURE_BYTES = 10_000_000


def _git_env() -> dict[str, str]:
    # Detach from the operator's global/system git config so identity, hooks and templates
    # cannot leak into the fixture repo.
    env = dict(os.environ)
    env.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_AUTHOR_NAME="gate test",
        GIT_AUTHOR_EMAIL="gate@test.invalid",
        GIT_COMMITTER_NAME="gate test",
        GIT_COMMITTER_EMAIL="gate@test.invalid",
    )
    return env


def _run_gate(tree: Path, added: dict[str, bytes]) -> subprocess.CompletedProcess:
    """Build base-commit + one commit ADDing `added`, then run the gate over that range."""
    env = _git_env()
    tree.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tree, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (tree / "base.txt").write_text("base\n")
    git("add", "-A")
    git("commit", "-qm", "base")

    for rel, blob in added.items():
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    git("add", "-A")
    git("commit", "-qm", "under test")

    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", "HEAD~1"],
        cwd=tree, env=env, capture_output=True, text=True, check=False,
    )


def test_fixture_at_the_ceiling_passes(tmp_path):
    res = _run_gate(tmp_path, {"tests/fixtures/bank.bin": b"\0" * MAX_FIXTURE_BYTES})
    assert res.returncode == 0, res.stdout + res.stderr


def test_fixture_over_the_ceiling_is_rejected(tmp_path):
    res = _run_gate(tmp_path, {"tests/fixtures/bank.bin": b"\0" * (MAX_FIXTURE_BYTES + 1)})
    assert res.returncode == 1, res.stdout + res.stderr
    assert "VIOLATION oversize-fixture: tests/fixtures/bank.bin" in res.stdout


def test_over_1mb_outside_fixtures_still_rejected(tmp_path):
    res = _run_gate(tmp_path, {"src/mantis/blob.bin": b"\0" * (MAX_ADDED_BYTES + 1)})
    assert res.returncode == 1, res.stdout + res.stderr
    assert "VIOLATION large-file: src/mantis/blob.bin" in res.stdout


def test_between_the_two_ceilings_only_fixtures_pass(tmp_path):
    """The carve-out still carves: 5 MB is fine in fixtures, fatal anywhere else."""
    ok = _run_gate(tmp_path / "a", {"tests/fixtures/mid.bin": b"\0" * 5_000_000})
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = _run_gate(tmp_path / "b", {"docs/mid.bin": b"\0" * 5_000_000})
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "VIOLATION large-file: docs/mid.bin" in bad.stdout


def test_jsonl_carve_out_unchanged(tmp_path):
    ok = _run_gate(tmp_path / "a", {"tests/fixtures/probe.jsonl": b"{}\n"})
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = _run_gate(tmp_path / "b", {"reports/probe.jsonl": b"{}\n"})
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "VIOLATION jsonl-outside-fixtures: reports/probe.jsonl" in bad.stdout


def test_artifact_dirs_rejected_at_any_size(tmp_path):
    res = _run_gate(tmp_path, {"checkpoints/tiny.pt": b"x"})
    assert res.returncode == 1, res.stdout + res.stderr
    assert "VIOLATION artifact-dir: checkpoints/tiny.pt" in res.stdout
