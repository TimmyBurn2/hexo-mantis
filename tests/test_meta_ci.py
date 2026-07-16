"""Meta-CI pins (repo_design §8): make targets, integration-tier reachability, ci.yml wiring."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_makefile_has_exactly_the_nine_dispatched_targets():
    text = (REPO_ROOT / "Makefile").read_text()
    targets = {
        m.group(1) for m in re.finditer(r"^([A-Za-z][A-Za-z0-9_.]*):", text, flags=re.MULTILINE)
    }
    assert targets == {
        "build", "build.native", "test", "test.integration",
        "bench", "bench.baseline", "check.wasm", "vendor", "clean",
    }


def test_integration_tier_reachable_from_make():
    text = (REPO_ROOT / "Makefile").read_text()
    m = re.search(r"^test\.integration:\n((?:\t.*\n)+)", text, flags=re.MULTILINE)
    assert m, "test.integration target missing"
    assert "-m integration" in m.group(1)


def test_ci_yaml_pins_tiers_and_gate_scripts():
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    for needle in (
        'pytest -m "not integration and not slow"',
        "pytest -m integration",
        "uv sync --locked",
        "cargo test --workspace --locked",
        "cargo clippy --workspace --all-targets --locked -- -D clippy::all",
        "make check.wasm",
        "make bench",
        "tools/ci_gates/test_count_gate.sh",
        "tools/ci_gates/artifact_gate.py",
        "tools/ci_gates/validate_configs.py",
        "tools/ci_gates/registry_gate.sh",
        "tools/check_import_dag.py src/mantis",
        "tools/ci_gates/check_tracked_refs.py",
    ):
        assert needle in text, f"ci.yml missing: {needle}"
