"""Test the tester: gate 8 run for real in a tmp tree, with the registry absent, matching and drifted.

The present rows plant the registry at the path the script reads and run it through a `uv` shim
that execs this interpreter, so an rc is the handshake's verdict and never an environment failure.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "ci_gates" / "registry_gate.sh"
REGISTRY_REL = Path("crates") / "mantis-encoding" / "src" / "registry.toml"


def _run_in(tree: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=tree, capture_output=True, text=True, check=False, env=env
    )


def _plant_registry(tree: Path, rel: Path, extra: bytes = b"") -> None:
    dest = tree / rel
    dest.parent.mkdir(parents=True)
    dest.write_bytes((REPO_ROOT / REGISTRY_REL).read_bytes() + extra)


def _shim_env(tmp_path: Path) -> dict[str, str]:
    """PATH with a `uv` that runs `uv run python ARGS` as this interpreter and refuses anything else."""
    bindir = tmp_path / "shim"
    bindir.mkdir()
    shim = bindir / "uv"
    shim.write_text(
        f'#!/bin/sh\n[ "$1 $2" = "run python" ] || exit 97\nshift 2\nexec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}


def test_present_matching_registry_passes_the_handshake(tmp_path):
    tree = tmp_path / "tree"
    _plant_registry(tree, REGISTRY_REL)
    res = _run_in(tree, _shim_env(tmp_path))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "handshake ARMED + PASS" in res.stdout, res.stdout + res.stderr


def test_present_drifted_registry_fails_the_handshake(tmp_path):
    tree = tmp_path / "tree"
    _plant_registry(tree, REGISTRY_REL, extra=b"\n# drift planted by the test\n")
    res = _run_in(tree, _shim_env(tmp_path))
    assert res.returncode != 0, res.stdout + res.stderr
    assert "on-disk registry.toml sha != compiled" in res.stderr, res.stdout + res.stderr


def test_hard_fail_when_registry_absent(tmp_path):
    (tmp_path / "crates" / "mantis-encoding").mkdir(parents=True)
    res = _run_in(tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "expected registry" in (res.stdout + res.stderr)

