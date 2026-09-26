"""Gate 18: every Rust file touched since the base is rustfmt-clean; an untouched file is never swept."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "tools" / "ci_gates" / "rustfmt_touched_gate.py"
_CHANNEL = re.search(r'channel\s*=\s*"([^"]+)"', (REPO_ROOT / "rust-toolchain.toml").read_text(
    encoding="utf-8")).group(1)  # pyright: ignore[reportOptionalMemberAccess]
_CLEAN = "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n"
_DIRTY = "fn main(){let x=1;println!(\"{x}\");}\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.email", "gate@example.com")
    _git(tmp_path, "config", "user.name", "gate")
    (tmp_path / "old.rs").write_text(_DIRTY, encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "branch", "base")
    return tmp_path


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["python3", str(GATE), "--base", "base"], cwd=repo, capture_output=True,
                          text=True, env={**os.environ, "RUSTUP_TOOLCHAIN": _CHANNEL})


def test_an_unformatted_touched_file_reds(repo: Path) -> None:
    (repo / "new.rs").write_text(_DIRTY, encoding="utf-8")
    _git(repo, "add", "new.rs")
    _git(repo, "commit", "-q", "-m", "touch")
    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "new.rs" in proc.stdout and "old.rs" not in proc.stdout


def test_an_uncommitted_edit_counts_as_touched(repo: Path) -> None:
    (repo / "old.rs").write_text(_DIRTY + "\n", encoding="utf-8")
    assert _run(repo).returncode == 1


def test_formatted_touched_files_pass_and_the_untouched_one_is_never_swept(repo: Path) -> None:
    (repo / "new.rs").write_text(_CLEAN, encoding="utf-8")
    _git(repo, "add", "new.rs")
    _git(repo, "commit", "-q", "-m", "touch")
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 touched Rust file" in proc.stdout
    assert (repo / "old.rs").read_text(encoding="utf-8") == _DIRTY


def test_a_deleted_file_is_not_checked_and_an_empty_scope_says_so(repo: Path) -> None:
    _git(repo, "rm", "-q", "old.rs")
    _git(repo, "commit", "-q", "-m", "drop")
    proc = _run(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 touched Rust files" in proc.stdout


def test_an_unresolvable_base_is_an_error_never_a_green(repo: Path) -> None:
    proc = subprocess.run(["python3", str(GATE), "--base", "no-such-ref"], cwd=repo,
                          capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr

