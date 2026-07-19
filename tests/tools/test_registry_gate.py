"""Test the tester: gate 8's existence trigger must arm (LAW-07 mutation self-test).

Runs the REAL tools/ci_gates/registry_gate.sh via subprocess in a tmp tree. With a dummy
crates/mantis-encoding/registry.toml present, the script MUST exit nonzero while the
audit CLI is absent (registry present + no audit = loud failure — proves the `[ ! -f ]`
path constant arms). With no registry, the armed-stub path exits 0 with its message.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "ci_gates" / "registry_gate.sh"


def _run_in(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=tree, capture_output=True, text=True, check=False
    )


def test_trigger_arms_when_registry_present(tmp_path):
    (tmp_path / "crates" / "mantis-encoding").mkdir(parents=True)
    (tmp_path / "crates" / "mantis-encoding" / "registry.toml").write_text("[dummy]\n")
    res = _run_in(tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "registry not yet ported" not in res.stdout


def test_hard_fail_when_registry_absent(tmp_path):
    # WP7 (1cbc89a) armed gate 8: the registry is ported (WP3) at crates/mantis-encoding/
    # src/registry.toml, so its ABSENCE is a hard failure, not the retired pre-port stub-pass.
    (tmp_path / "crates" / "mantis-encoding").mkdir(parents=True)
    res = _run_in(tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "expected registry" in (res.stdout + res.stderr)
