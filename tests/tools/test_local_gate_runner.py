"""AUDIT-1 F-09 — every CI gate has a local equivalent, and the runner enumerates them all.

THE DEFECT. R311(b) suspended remote CI and made LOCAL GREEN the gate. There was no local
runner. `make test` runs the default tier plus `cargo test --workspace --locked`, which does
NOT compile `[[bench]]` targets; `make bench` compiles exactly one of the eight. And
`cargo clippy --workspace --all-targets --locked -- -D clippy::all` appeared ONLY in
`.github/workflows/ci.yml`. So every "full local gate set" since the suspension excluded
`-D clippy::all` — including `incompatible_msrv`, the guard on the `rust-version = "1.87"`
floor — and never compiled the seven bench targets standing behind the 28 floors in
`tools/bench_floors.toml`. CLAUDE.md's own rule: "nothing lives only in workflow YAML".

WHAT THESE ROWS ARE. Not a second copy of what each gate checks — they assert the SET. The
workflow is parsed for its `gate N:` steps and every one must have a local invocation in
`tools/ci_gates/run_all.sh`. A gate added to CI and not to the runner reds here, which is the
only way "local green" can keep meaning what R311(b) says it means.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RUNNER = REPO_ROOT / "tools" / "ci_gates" / "run_all.sh"
MAKEFILE = REPO_ROOT / "Makefile"

#: `- name: "gate 2b: clippy (...)"` → the gate's number-plus-letter key.
_GATE_STEP = re.compile(r'^\s*-\s*name:\s*"gate\s*([0-9]+[a-z]?)\s*:', re.M)
#: The runner's own rows: `run_gate "gate 2b: clippy (...)" \`
_RUNNER_ROW = re.compile(r'^\s*run_gate\s+"gate\s*([0-9]+[a-z]?)\s*:', re.M)


def _workflow_gates() -> set[str]:
    return set(_GATE_STEP.findall(WORKFLOW.read_text(encoding="utf-8")))


def _runner_gates() -> set[str]:
    return set(_RUNNER_ROW.findall(RUNNER.read_text(encoding="utf-8")))


def test_the_runner_exists_and_is_executable() -> None:
    assert RUNNER.is_file(), "there is no local gate runner"
    assert RUNNER.stat().st_mode & 0o111, f"{RUNNER} is not executable"


def test_the_parse_finds_a_plausible_number_of_gates() -> None:
    """The non-vacuity floor. A regex that matched nothing would make every row below pass —
    the phantom-gate class this file exists to close, one layer up."""
    gates = _workflow_gates()
    assert len(gates) >= 17, f"only {len(gates)} gate steps parsed out of ci.yml: {sorted(gates)}"
    assert len(_runner_gates()) >= 17, sorted(_runner_gates())


def test_every_CI_gate_has_a_local_invocation() -> None:
    """THE PIN. A gate that lives only in the workflow is a gate local green does not run."""
    missing = _workflow_gates() - _runner_gates()
    assert not missing, (
        f"gate(s) {sorted(missing)} run in CI and in no local command. R311(b) makes local "
        f"green the gate, so a CI-only gate is a gate nothing checks — remote CI is SUSPENDED."
    )


def test_the_runner_invents_no_gate_CI_does_not_have() -> None:
    """The converse: the runner is a roster of the real gates, not a second authority that
    could drift into checking something the workflow never agreed to."""
    extra = _runner_gates() - _workflow_gates()
    assert not extra, f"the runner runs gate(s) {sorted(extra)} that ci.yml does not define"


def test_clippy_is_reachable_from_a_make_target_not_only_from_the_workflow() -> None:
    """The specific gate F-09 measured: `-D clippy::all` was in ci.yml and nowhere under
    `Makefile`/`tools/`."""
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cargo clippy" in makefile, "no make target runs clippy"
    assert "--all-targets" in makefile, (
        "clippy runs without --all-targets, so the seven non-smoke bench targets are still "
        "compiled by no local command"
    )
    assert "-D clippy::all" in makefile
    assert "lint.rust:" in makefile and "gates:" in makefile


def test_the_all_targets_flag_is_what_compiles_the_bench_targets() -> None:
    """The mechanism, named against the tree: eight `[[bench]]` targets exist, `make bench`
    builds ONE, and `cargo test` builds none of them."""
    benches = set()
    for manifest in sorted((REPO_ROOT / "crates").glob("*/Cargo.toml")):
        text = manifest.read_text(encoding="utf-8")
        benches |= set(re.findall(r'\[\[bench\]\]\s*\nname\s*=\s*"([^"]+)"', text))
    assert len(benches) >= 8, sorted(benches)

    makefile = MAKEFILE.read_text(encoding="utf-8")
    bench_recipe = makefile.split("\nbench:", 1)[1].split("\n\n", 1)[0]
    built_by_make_bench = {name for name in benches if name in bench_recipe}
    assert built_by_make_bench == {"smoke_bench"}, (
        f"`make bench` builds {sorted(built_by_make_bench)}; the other "
        f"{len(benches) - 1} are reached only through clippy --all-targets"
    )


def test_gate_1_is_declared_opt_in_rather_than_silently_absent() -> None:
    """The one gate the runner does not run by default must SAY it does not, or the summary
    line is a claim about a set that is one short."""
    body = RUNNER.read_text(encoding="utf-8")
    assert "--with-fresh-sync" in body
    assert "gate 1" in body
    assert "gate 1 NOT RUN" in body, (
        "the summary must state the omission; a runner that quietly skips a gate is the "
        "thing this file is about"
    )


def test_the_runner_runs_every_gate_even_after_one_reds() -> None:
    """`set -e` would stop at the first failure and report one gate when seventeen were
    asked about."""
    body = RUNNER.read_text(encoding="utf-8")
    assert "set -uo pipefail" in body and "set -euo pipefail" not in body


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_the_runner_answers_for_its_own_interface(flag: str) -> None:
    proc = subprocess.run(["bash", str(RUNNER), flag], capture_output=True, text=True,
                          cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "--with-fresh-sync" in proc.stdout and "--base" in proc.stdout


def test_an_unknown_argument_is_refused_rather_than_ignored() -> None:
    proc = subprocess.run(["bash", str(RUNNER), "--nope"], capture_output=True, text=True,
                          cwd=REPO_ROOT)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "unknown argument" in proc.stderr


def test_the_only_filter_can_select_a_single_gate() -> None:
    """The seam that makes the runner usable while iterating — and it must still be the
    production runner, not a second path."""
    proc = subprocess.run(
        ["bash", str(RUNNER), "--only", "gate 16"], capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "gate 16" in proc.stdout
    assert "gate 2a" not in proc.stdout, "the filter did not filter"
    assert "green: 1" in proc.stdout
