"""Meta-CI pins (repo_design §8): make targets, integration-tier reachability, ci.yml wiring."""
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_makefile_dispatches_exactly_the_declared_target_set():
    """The set stays EXACT so a stray target still reds this pin.

    `lint` joined at WPCLEAN Phase LG as CI gate 14's local runner (R98); `vendor.sealbot`
    joined at R324(c), because a vendor BUILD step reachable only by hand is a second
    vendoring mechanism beside `make vendor`, which CLAUDE.md declares to be the one.

    THE COUNT LEFT THIS NAME at R324(c) and the reason is the repo's own (R192(e),
    derive-or-delete): a name that transcribes a tally must be re-edited on every change to
    the thing it counts, and is then read as evidence by someone who did not re-derive it.
    The SET below is the authority; nothing states its size."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        m.group(1) for m in re.finditer(r"^([A-Za-z][A-Za-z0-9_.]*):", text, flags=re.MULTILINE)
    }
    assert targets == {
        "build", "build.native", "test", "test.integration", "lint",
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
        # The python job's workspace must still be built from the lockfile. Gate 1 syncs
        # only inside its own temp clone, so without this the later steps' bare `uv run`
        # would re-lock — an existing CI property REVIEW-impl caught being dropped.
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
    """Every `run:` body in ci.yml, across all jobs and steps.

    Parsed, not grepped: a substring search over the raw file cannot tell a command
    that EXECUTES a script from a step *name*, a comment, or a `with:` value that
    merely mentions its path. That distinction is the whole point of this pin.
    """
    spec = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return [
        step["run"]
        for job in spec.get("jobs", {}).values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    ]


def _is_invocable(path: Path) -> bool:
    """A file CI can INVOKE: every .sh, and any .py with a `__main__` guard or a shebang.

    WPBOX Phase Q: the preflight split leaves a LIBRARY module beside its gate
    (`preflight_mint_parent.py`, loaded by `preflight_mint.py` off its own directory).
    A library has no meaningful "invoked by ci.yml" obligation — its executable half is
    the gate that loads it, and THAT file stays censused. The discrimination is derived
    from file content, never a hand list (this test's own R44 lesson), so a real gate
    script written tomorrow — which needs a `__main__` guard or shebang to be a gate at
    all — cannot use this arm to hide.
    """
    if path.suffix == ".sh":
        return True
    text = path.read_text(encoding="utf-8", errors="replace")
    return "__main__" in text or text.startswith("#!")


def _gate_scripts() -> list[str]:
    """Repo-relative paths of every gate script whose logic CI must invoke.

    `tools/ci_gates/` recursively (so a script tucked in a subdirectory cannot hide),
    plus gate 9, whose script is the one that lives directly under `tools/`.
    """
    gate_dir = REPO_ROOT / "tools" / "ci_gates"
    found = {
        str(p.relative_to(REPO_ROOT))
        for p in gate_dir.rglob("*")
        if p.is_file() and p.suffix in {".sh", ".py"} and _is_invocable(p)
    }
    found.add("tools/check_import_dag.py")
    return sorted(found)


def test_every_ci_gate_script_is_invoked_by_ci_yaml():
    """No orphaned gate scripts (R44 / LAW-07).

    The hand-written needle list above enumerated nine of the ten gate scripts and
    silently omitted `gate_01_fresh_sync.sh`. That omission is why gate 1 could sit
    unreferenced by any workflow step, any make target, and any test from WP0 until
    WPUF-2 — while `ci.yml` ran a divergent inline reimplementation under gate 1's name
    that never exercised the stale symbol the real script asserted. A hand-maintained
    list cannot catch the gate it forgot, so this derives the expectation from the
    filesystem instead.

    Invocation, not mention: REVIEW-impl rebuilt the original bug against an earlier
    version of this test — a step *named* for the script but with an inline `run:` body
    passed, and so did commenting the `run:` line out. Both now fail, because only
    executed `run:` bodies are searched.
    """
    scripts = _gate_scripts()
    assert scripts, "no gate scripts found — the census itself is broken"

    commands = _ci_run_commands()
    orphans = [s for s in scripts if not any(s in cmd for cmd in commands)]
    assert not orphans, (
        f"gate scripts present but executed by no ci.yml `run:` step: {orphans}. "
        "A gate nothing runs is not a gate."
    )


def test_lint_and_type_gate_is_blocking_and_self_tested():
    """The R57 advisory pin, RE-POINTED by its own designed path (WPCLEAN Phase LG, R98).

    The old pin's docstring said re-blocking "must happen by burning the findings down, not
    by flipping the flag" — Phase LT burned the configured surface to zero (CENSUS_LT), so
    the flip is now legal and this pin inverts to guard the NEW posture: exactly one
    lint/type step, running the gate script (invocation, not mention), with NO `|| true`
    escape and the self-test armed on every run. An advisory `|| true` creeping back is
    what this test now reds on — a permanently-advisory green is the fog R98 ended.
    """
    commands = _ci_run_commands()
    gate_steps = [c for c in commands if "lint_gate.sh" in c]
    assert len(gate_steps) == 1, (
        f"expected exactly one lint_gate.sh step in ci.yml, found {len(gate_steps)}"
    )
    step = gate_steps[0]
    assert "--self-test" in step, "gate 14 must arm its own trigger on every CI run"
    assert "|| true" not in step, "gate 14 is a GATE (R98); an advisory escape defeats it"
    # The old standalone advisory invocations must not linger beside the gate — one
    # authority for the lint verdict, not a gate plus a shadow report.
    strays = [c for c in commands
              if ("ruff check" in c or "pyright" in c) and "lint_gate.sh" not in c]
    assert not strays, f"standalone ruff/pyright steps beside the gate: {strays}"


def test_gate_01_script_actually_fresh_clones_and_syncs():
    """repo_design §9.1: 'clone-and-run is the product', so gate 1 must CLONE.

    Pins the two behaviours that `ci.yml` previously inlined incorrectly — it synced the
    existing checkout, which cannot prove a fresh clone builds.
    """
    text = (REPO_ROOT / "tools" / "ci_gates" / "gate_01_fresh_sync.sh").read_text()
    assert "git clone" in text, "gate 1 must clone; syncing the checkout proves nothing"
    assert "uv sync --locked" in text, "gate 1 must build the extension from the lockfile"
    # The call form, not the bare word: the script's own header comment explains the
    # hello() rot, and a substring check for "hello()" would flag that explanation.
    assert "_engine.hello" not in text, (
        "hello() was deleted at WP7 (8198016); gate 1 must not assert it"
    )
