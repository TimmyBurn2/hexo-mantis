"""AUDIT-1 F-12 — a deselecting marker cannot remove a test from every tier in silence.

THE DEFECT. Gate 3c counts COLLECTED tests: `pytest --collect-only -q -m ''` walks the whole
tree, and DESELECTION IS NOT COLLECTION. `make test` runs `-m "not integration and not slow"`;
`make test.integration` runs `-m integration`. So one `@pytest.mark.slow` line — or a
host-true `@pytest.mark.skipif`, or a module-level `pytestmark` — takes a test out of every
tier the repo actually runs while the collected-test floor, the suite's own green, and every
other gate stay rc 0. The 4519 floor counted tests that EXIST, not tests that RUN.

`tests/model/conformance/` already refuses this for its own suite, after RED-TEAM planted one
`slow` line above a cross-crate legal-set claim and nothing in the repo reported it. This is
the tree-wide twin, plus gate 3c's missing third arm: `tree_floor <= collected`.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "ci_gates" / "tier_census.py"
DECLARATION = REPO_ROOT / "tools" / "ci_gates" / "tier_declaration.txt"
GATE = REPO_ROOT / "tools" / "ci_gates" / "test_count_gate.sh"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("_tier_census_probe", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load()


# ── the census reads the tree ─────────────────────────────────────────────────────────

def test_the_committed_tree_is_green_and_the_declaration_is_not_empty() -> None:
    """R98: a gate is adopted only over a clean baseline — and an EMPTY declaration would
    make every row below pass vacuously, which is the phantom-gate class one layer up."""
    observed = TOOL.census(REPO_ROOT / "tests")
    declared = TOOL.load_declaration(DECLARATION)
    assert len(declared) >= 50, f"only {len(declared)} declared rows"
    assert TOOL.compare(observed, declared) == ([], [])


def test_both_marker_spellings_are_seen(tmp_path: Path) -> None:
    """The decorator is the obvious form; a module-level `pytestmark` takes a whole FILE out
    in one line and carries no decorator at all."""
    suite = tmp_path / "tests"
    suite.mkdir()
    # The decorator is assembled rather than written inline: a literal "\n" immediately
    # before "@pytest.mark" matches CI gate 17's `user@host` class, and a fixture that
    # trips a gate teaches the next reader to add a hatch reflexively.
    mark = "@pytest.mark"
    (suite / "test_decorated.py").write_text(
        f"import pytest\n\n\n{mark}.slow\ndef test_a() -> None:\n    pass\n",
        encoding="utf-8")
    (suite / "test_module_marked.py").write_text(
        "import pytest\n\npytestmark = [pytest.mark.integration]\n\n\n"
        "def test_b() -> None:\n    pass\n",
        encoding="utf-8")
    assert TOOL.census(suite) == {
        ("tests/test_decorated.py", "test_a", "slow"),
        ("tests/test_module_marked.py", TOOL.MODULE_SCOPE, "integration"),
    }


def test_a_non_deselecting_marker_is_not_censused(tmp_path: Path) -> None:
    """The control. `parametrize` changes HOW a test runs, never WHETHER — declaring it would
    make the declaration a list of every marker in the repo and stop being reviewable."""
    suite = tmp_path / "tests"
    suite.mkdir()
    # The decorator is assembled rather than written inline: `'n'@pytest.mark…` matches CI
    # gate 17's `user@host` pattern class, and a fixture that trips a gate teaches the next
    # reader to add a hatch (rule 7's own reasoning about reflexive escapes).
    mark = "@pytest.mark.parametrize"
    (suite / "test_p.py").write_text(
        f"import pytest\n\n\n{mark}(\"n\", [1])\n"
        "def test_a(n: int) -> None:\n    pass\n", encoding="utf-8")
    assert TOOL.census(suite) == set()


def test_an_unparseable_test_module_REFUSES_rather_than_being_skipped(tmp_path: Path) -> None:
    """A census that silently skipped a file it could not read would report a clean tier over
    exactly the file most likely to be hiding something."""
    suite = tmp_path / "tests"
    suite.mkdir()
    (suite / "test_broken.py").write_text("def test_a(  :\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        TOOL.census(suite)


# ── the gate's two directions ─────────────────────────────────────────────────────────

def test_an_UNDECLARED_marker_is_refused() -> None:
    """THE PIN. This is RED-TEAM's plant, one directory wider."""
    observed = TOOL.census(REPO_ROOT / "tests") | {
        ("tests/train/test_losses.py", "test_chain_loss_math_is_smooth_l1", "slow")}
    undeclared, stale = TOOL.compare(observed, TOOL.load_declaration(DECLARATION))
    assert undeclared == [
        ("tests/train/test_losses.py", "test_chain_loss_math_is_smooth_l1", "slow")]
    assert stale == []


def test_a_STALE_declaration_is_refused_too() -> None:
    """A line with no marker behind it is a standing licence nobody is using, and the next
    real deselection hides behind it."""
    declared = TOOL.load_declaration(DECLARATION) | {("tests/nope.py", "test_ghost", "slow")}
    undeclared, stale = TOOL.compare(TOOL.census(REPO_ROOT / "tests"), declared)
    assert undeclared == []
    assert stale == [("tests/nope.py", "test_ghost", "slow")]


def test_an_empty_declaration_is_a_FAILURE_not_a_vacuous_pass(tmp_path: Path) -> None:
    path = tmp_path / "decl.txt"
    path.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="phantom-gate"):
        TOOL.load_declaration(path)


def test_a_malformed_declaration_row_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "decl.txt"
    path.write_text("tests/a.py::test_b slow\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed row"):
        TOOL.load_declaration(path)


def test_the_tools_own_self_test_fires_every_control() -> None:
    proc = subprocess.run(["python3", str(TOOL_PATH), "--self-test"],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all controls fire" in proc.stdout
    assert "SELF-TEST FAILED" not in proc.stdout


# ── gate 3c carries it, and gained its third count arm ────────────────────────────────

def test_gate_3c_runs_the_census() -> None:
    """The census is not a gate of its own — it rides 3c, because 'no test was lost' is
    exactly 3c's claim and a deselected test is a lost test the count cannot see."""
    body = GATE.read_text(encoding="utf-8")
    assert "tier_census.py" in body


def test_gate_3c_runs_BOTH_arms_even_when_the_first_reds() -> None:
    """A run that stops at the count reports one fact when two were asked about."""
    body = GATE.read_text(encoding="utf-8")
    assert "count_rc=0 census_rc=0" in body
    assert 'verdict "$collected" "$ref_floor" "$tree_floor" "$ref" || count_rc=$?' in body


def test_gate_3c_refuses_a_floor_ratcheted_PAST_the_collection() -> None:
    """F-12's third arm. The two original arms compare the count against the REF floor and
    the floors against each other; nothing compared the tree's own floor against what the
    tree collects, so a floor over tests that do not exist stayed green."""
    body = GATE.read_text(encoding="utf-8")
    assert "FAIL (over-ratchet)" in body
    proc = subprocess.run(["bash", str(GATE), "--self-test"], capture_output=True, text=True,
                          cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all correct" in proc.stdout


def test_the_self_tests_arm_tally_is_DERIVED_not_transcribed() -> None:
    """R192(e), derive-or-delete. The line read "4 clean arms + 7 firing arms" as a literal;
    adding an arm made it wrong, and a wrong tally is read as evidence."""
    body = GATE.read_text(encoding="utf-8")
    assert "4 clean arms + 7 firing arms" not in body
    assert "grep -cE" in body and 'clean arms + $fired firing arms' in body
