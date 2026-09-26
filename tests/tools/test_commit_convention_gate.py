"""Gate 19: every commit since the base is one subject line with an empty body and no trailer."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[2] / "tools" / "ci_gates" / "commit_convention_gate.py"


#: The host's git config (signing, hooks, showSignature) must not reach the fixture or the gate.
_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, env=_ENV).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.email", "gate@example.com")
    _git(tmp_path, "config", "user.name", "gate")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base: whatever it says\n\nwith a body")
    _git(tmp_path, "branch", "base")
    return tmp_path


def _commit(repo: Path, message: str) -> None:
    _git(repo, "commit", "-q", "--allow-empty", "--cleanup=verbatim", "-m", message)


def _run(repo: Path, base: str = "base") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["python3", str(GATE), "--base", base], cwd=repo, capture_output=True,
                          text=True, env=_ENV)


def test_one_line_commits_pass_and_the_base_itself_is_not_judged(repo: Path) -> None:
    _commit(repo, "fix(x): one line that says why")
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 commit" in proc.stdout


@pytest.mark.parametrize("message", [
    "fix(x): subject\n\nCo-Authored-By: someone <a@example.com>",
    "fix(x): subject\n\nClaude-Session: https://example.com/s",
    "fix(x): subject\n\na body paragraph",
    "fix(x): a subject wrapped\nonto a second line",
])
def test_a_body_or_a_trailer_reds(repo: Path, message: str) -> None:
    _commit(repo, "fix(x): fine")
    _commit(repo, message)
    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "VIOLATION" in proc.stdout and "fine" not in proc.stdout


def test_an_empty_range_says_so_and_passes(repo: Path) -> None:
    proc = _run(repo)
    assert proc.returncode == 0 and "0 commits" in proc.stdout


def test_an_unresolvable_base_is_an_error_never_a_green(repo: Path) -> None:
    assert _run(repo, "no-such-ref").returncode == 2


def test_an_all_zeros_base_widens_to_origin_dev(repo: Path) -> None:
    _git(repo, "update-ref", "refs/remotes/origin/dev", "base")
    _commit(repo, "fix(x): one line")
    proc = _run(repo, "0" * 40)
    assert proc.returncode == 0 and "origin/dev" in proc.stdout, proc.stdout + proc.stderr
