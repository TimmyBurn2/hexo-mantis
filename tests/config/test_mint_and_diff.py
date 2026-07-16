"""Mint/diff tool behavior tests (exit-code contracts per the config tooling design)."""
import subprocess
import sys
from pathlib import Path

from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
DIFF = REPO_ROOT / "tools" / "config_diff.py"
TEMPLATE = REPO_ROOT / "tools" / "config_templates" / "dev.yaml"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=False)


def _mint(tmp_path: Path, name: str, *deltas: str):
    out = tmp_path / name
    argv = [str(MINT), "--template", "dev", "--out", str(out)]
    for delta in deltas:
        argv += ["--set", delta]
    return out, _run(*argv)


def test_mint_output_validates(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "run_id=mint_check")
    assert proc.returncode == 0, proc.stderr
    cfg = load_config(out)
    assert cfg.run_id == "mint_check"


def test_mint_stamps_template_and_delta_header(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "run_id=mint_check")
    assert proc.returncode == 0, proc.stderr
    head = out.read_text().splitlines()[:3]
    assert head[0] == "# minted-by: tools/mint_config.py"
    assert head[1] == "# template: dev"
    assert head[2] == "# delta: run_id: template_dev -> mint_check"


def test_mint_rejects_unknown_delta_key(tmp_path):
    out, proc = _mint(tmp_path, "minted.yaml", "identity.bogus=1")
    assert proc.returncode == 2
    assert "unknown delta key" in proc.stderr
    assert not out.exists()


def test_diff_exit_0_on_exactly_claimed_one_key_diff(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "MATCH" in res.stdout


def test_diff_exit_1_on_unclaimed_extra_diff(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check", "seed=999")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id")
    assert res.returncode == 1
    assert "UNCLAIMED diff on seed" in res.stdout


def test_diff_exit_1_when_claimed_key_identical(tmp_path):
    out, proc = _mint(tmp_path, "b.yaml", "run_id=diff_check")
    assert proc.returncode == 0, proc.stderr
    res = _run(str(DIFF), str(TEMPLATE), str(out), "--expect", "run_id", "--expect", "seed")
    assert res.returncode == 1
    assert "expected diff on seed but values are identical" in res.stdout
