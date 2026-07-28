"""O8b — lying-header self-check (tools/config_diff.py --from-header; B3, red-team #3).

Parses a committed config's stamped header (# template + # delta lines), re-diffs the
config against tools/config_templates/<t>.yaml, and asserts {claimed} == {actual}. Makes
the delta-antidote structural (CI-runnable), not manual --expect diligence.
"""
import subprocess
import sys
from pathlib import Path

import yaml

from mantis.config.loader import discover_configs

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
DIFF = REPO_ROOT / "tools" / "config_diff.py"
TEMPLATE_DIR = REPO_ROOT / "tools" / "config_templates"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, check=False)


def _from_header(config: Path) -> subprocess.CompletedProcess:
    return _run(str(DIFF), "--from-header", str(config))


def _write(path: Path, *, template: str, claimed: list[str], body_overrides: dict) -> Path:
    """Write a config with an explicit header (claimed deltas) + a body derived from the
    named template with body_overrides applied. Lets a test make header and body disagree."""
    data = yaml.safe_load((TEMPLATE_DIR / f"{template}.yaml").read_text())
    for dotted, value in body_overrides.items():
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    header = ["# minted-by: tools/mint_config.py", f"# template: {template}"]
    header += [f"# delta: {k}: x -> y" for k in claimed]
    path.write_text("\n".join(header) + "\n" + yaml.safe_dump(data, sort_keys=False))
    return path


def _mint(out: Path, *deltas: str) -> subprocess.CompletedProcess:
    argv = [str(MINT), "--template", "dev", "--out", str(out)]
    for d in deltas:
        argv += ["--set", d]
    return _run(*argv)


def test_truthful_minted_config_exits_0(tmp_path):
    out = tmp_path / "c.yaml"
    assert _mint(out, "run_id=truthful").returncode == 0
    res = _from_header(out)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "MATCH" in res.stdout


def test_header_omits_a_real_delta_exits_1(tmp_path):
    out = _write(
        tmp_path / "c.yaml",
        template="dev",
        claimed=["run_id"],
        body_overrides={"run_id": "liar", "seed": 999},
    )
    res = _from_header(out)
    assert res.returncode == 1
    assert "seed" in res.stdout


def test_header_claims_unchanged_key_exits_1(tmp_path):
    out = _write(
        tmp_path / "c.yaml",
        template="dev",
        claimed=["run_id", "seed"],
        body_overrides={"run_id": "liar"},
    )
    res = _from_header(out)
    assert res.returncode == 1
    assert "seed" in res.stdout


def test_missing_template_exits_2(tmp_path):
    out = _write(
        tmp_path / "c.yaml",
        template="dev",
        claimed=["run_id"],
        body_overrides={"run_id": "liar"},
    )
    # rewrite the template line to name a template that does not exist
    text = out.read_text().replace("# template: dev", "# template: nonexistent_template")
    out.write_text(text)
    assert _from_header(out).returncode == 2


def test_unparseable_header_exits_2(tmp_path):
    # a config with no "# template:" line -> the header is unparseable
    out = tmp_path / "c.yaml"
    assert _mint(out, "run_id=x").returncode == 0
    body = "\n".join(l for l in out.read_text().splitlines() if not l.startswith("# template:"))
    out.write_text(body)
    assert _from_header(out).returncode == 2


def test_invalid_config_exits_2(tmp_path):
    out = _write(
        tmp_path / "c.yaml",
        template="dev",
        claimed=["seed"],
        body_overrides={"seed": "not-an-int"},  # str->int fails strict validation
    )
    assert _from_header(out).returncode == 2


# ── F2 — arm --from-header structurally over the SHIPPED configs (CI test tier, gate 3) ──
def test_every_committed_config_header_is_truthful():
    # ADJ-13 F-1 corrective pass (recheck R-5): the ONE discovery authority, not a
    # sixth flat glob. A flat `*.yaml` census is blind to `configs/prod/run6.yaml`,
    # which gate 7 and gate 12 both now make legal.
    configs = discover_configs(REPO_ROOT / "configs")
    assert configs, "no committed configs found"
    for cfg in configs:
        res = _from_header(cfg)
        assert res.returncode == 0, f"{cfg.name}: {res.stdout}{res.stderr}"
        assert "MATCH" in res.stdout


def test_committed_config_body_lie_would_be_caught(tmp_path):
    # A real mutation (not tautological): flip a run5 body key NOT listed in its header -> exit 1.
    src = (REPO_ROOT / "configs" / "run5.yaml").read_text()
    assert "random_model_sims: 96" in src  # not in run5's header (only run_id + seed are)
    lie = tmp_path / "run5_lie.yaml"
    lie.write_text(src.replace("random_model_sims: 96", "random_model_sims: 64"))
    res = _from_header(lie)
    assert res.returncode == 1
    assert "random_model_sims" in res.stdout


def test_mutation_self_test_bites(tmp_path):
    out = tmp_path / "c.yaml"
    assert _mint(out, "run_id=honest").returncode == 0
    original = out.read_text()
    assert _from_header(out).returncode == 0
    # hand-edit an UNLISTED key in the body -> the check must bite
    tampered = original.replace("seed: 20260716", "seed: 424242")
    assert tampered != original
    out.write_text(tampered)
    assert _from_header(out).returncode == 1
    # revert -> clean again
    out.write_text(original)
    assert _from_header(out).returncode == 0
