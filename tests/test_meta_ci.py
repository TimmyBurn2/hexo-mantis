"""Meta-CI pins (repo_design §8): make targets, integration-tier reachability, ci.yml wiring."""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_makefile_dispatches_exactly_the_declared_target_set():
    """The set stays EXACT so a stray target still reds this pin, and nothing states its SIZE:
    a transcribed tally must be re-edited on every change to the thing it counts."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        m.group(1) for m in re.finditer(r"^([A-Za-z][A-Za-z0-9_.]*):", text, flags=re.MULTILINE)
    }
    assert targets == {
        "build", "build.native", "test", "test.integration", "lint", "lint.rust",
        # `gates` is the everyday set; `gates.exit` adds the slow tier, which BOTH pytest
        # tiers deselect; `dashboard` renders a run record.
        "gates", "gates.exit", "dashboard",
        "bench", "bench.baseline", "check.wasm", "vendor", "vendor.sealbot", "clean",
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
        # The python job's workspace must still be built from the lockfile: gate 1 syncs only
        # inside its own temp clone, so without this a later bare `uv run` would re-lock.
        "uv sync --locked",
        "cargo test --workspace --locked",
        "cargo clippy --workspace --all-targets --locked -- -D clippy::all",
        "make check.wasm",
        "make bench",
        "tools/ci_gates/gate_01_fresh_sync.sh",
        "tools/ci_gates/test_count_gate.sh",
        "tools/ci_gates/artifact_gate.py",
        "tools/ci_gates/validate_configs.py",
        "tools/ci_gates/registry_gate.sh",
        "tools/check_import_dag.py src/mantis",
        "tools/ci_gates/check_tracked_refs.py",
    ):
        assert needle in text, f"ci.yml missing: {needle}"


def _ci_run_commands() -> list[str]:
    """Every `run:` body in ci.yml. Parsed, not grepped: a substring search cannot tell a
    command that EXECUTES a script from a step name or a `with:` value naming its path."""
    spec = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return [
        step["run"]
        for job in spec.get("jobs", {}).values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    ]


def _is_invocable(path: Path) -> bool:
    """A file CI can INVOKE: every .sh, and any .py with a `__main__` guard or a shebang.

    Derived from file content, never a hand list, so a library module beside a gate is exempt
    and a real gate script written tomorrow cannot use this arm to hide.
    """
    if path.suffix == ".sh":
        return True
    text = path.read_text(encoding="utf-8", errors="replace")
    return "__main__" in text or text.startswith("#!")


def _gate_scripts() -> list[str]:
    """Repo-relative paths of every gate script CI must invoke: `tools/ci_gates/` recursively,
    so a script in a subdirectory cannot hide, plus gate 9's script under `tools/`."""
    gate_dir = REPO_ROOT / "tools" / "ci_gates"
    found = {
        str(p.relative_to(REPO_ROOT))
        for p in gate_dir.rglob("*")
        if p.is_file() and p.suffix in {".sh", ".py"} and _is_invocable(p)
    }
    found.add("tools/check_import_dag.py")
    return sorted(found)


def test_every_ci_gate_script_is_invoked_by_ci_yaml():
    """No orphaned gate scripts, derived from the filesystem rather than a hand list, and by
    INVOCATION: only executed `run:` bodies are searched, so a step merely named for a script
    fails.
    """
    scripts = _gate_scripts()
    assert scripts, "no gate scripts found — the census itself is broken"

    # Three invocation sites count, all executed rather than mentioned: a `run:` body in
    # `ci.yml`, a `Makefile` recipe line, and another gate SCRIPT.
    commands = _ci_run_commands()
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    gate_bodies = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "tools" / "ci_gates").iterdir())
        if path.is_file() and path.suffix in {".sh", ".py"}
    )
    orphans = [
        s for s in scripts
        if not any(s in cmd for cmd in commands)
        and Path(s).name not in makefile
        and Path(s).name not in gate_bodies
    ]
    assert not orphans, (
        f"gate scripts present but executed by no ci.yml `run:` step, no Makefile recipe and "
        f"no sibling gate: {orphans}. A gate nothing runs is not a gate."
    )


def test_lint_and_type_gate_is_blocking_and_self_tested():
    """Exactly one lint/type step, running the gate script, with NO `|| true` escape and the
    self-test armed on every run — a permanently-advisory green is fog."""
    commands = _ci_run_commands()
    gate_steps = [c for c in commands if "lint_gate.sh" in c]
    assert len(gate_steps) == 1, (
        f"expected exactly one lint_gate.sh step in ci.yml, found {len(gate_steps)}"
    )
    step = gate_steps[0]
    assert "--self-test" in step, "gate 14 must arm its own trigger on every CI run"
    assert "|| true" not in step, "gate 14 is a GATE (R98); an advisory escape defeats it"
    # One authority for the lint verdict: no standalone advisory invocation beside the gate.
    strays = [c for c in commands
              if ("ruff check" in c or "pyright" in c) and "lint_gate.sh" not in c]
    assert not strays, f"standalone ruff/pyright steps beside the gate: {strays}"


def test_gate_01_script_actually_fresh_clones_and_syncs():
    """Gate 1 must CLONE: `ci.yml` previously inlined a sync of the existing checkout, which
    cannot prove that a fresh clone builds."""
    text = (REPO_ROOT / "tools" / "ci_gates" / "gate_01_fresh_sync.sh").read_text()
    assert "git clone" in text, "gate 1 must clone; syncing the checkout proves nothing"
    assert "uv sync --locked" in text, "gate 1 must build the extension from the lockfile"
    # The call form, not the bare word: a substring check for "hello()" would flag the
    # script's own header comment explaining the rot.
    assert "_engine.hello" not in text, (
        "hello() was deleted at WP7 (8198016); gate 1 must not assert it"
    )
