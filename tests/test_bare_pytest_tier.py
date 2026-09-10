"""A bare `pytest` resolves to the DEFAULT TIER, and the other invocations still mean what they
meant.

The witness is a SUBPROCESS, not an assertion on the live `markexpr`: the mechanism under test is
pytest's own argument resolution against this repo's `pyproject.toml`, and an in-process
assertion would pin the invocation instead. Each arm collects THIS file — which carries one
plain, one `integration`-marked and one `slow`-marked function — and reads pytest's summary line.

PLANTED BREAK: remove `-m ...` from `addopts` and the bare arm reads the whole tree with no
deselection, failing the first test below.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve().relative_to(REPO_ROOT)
DEFAULT_TIER = "not integration and not slow"

_SUMMARY = re.compile(
    r"(?:(?P<selected>\d+)/)?(?P<collected>\d+) tests? collected(?: \((?P<deselected>\d+) deselected\))?"
)


def _collect(*extra: str) -> tuple[int, int, int]:
    """Collect this file in a pytest CHILD against the repo's own pyproject.

    Returns (selected, collected, deselected) read off pytest's summary line.

    Raises:
        AssertionError: the child printed no summary line this parser recognises.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *extra, str(THIS)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if "collected" in ln]
    assert lines, f"no collection summary in child output:\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}"
    m = _SUMMARY.search(lines[-1])
    assert m, f"unparsed summary line: {lines[-1]!r}"
    collected = int(m.group("collected"))
    deselected = int(m.group("deselected") or 0)
    selected = int(m.group("selected") or collected)
    return selected, collected, deselected


def test_a_bare_pytest_applies_the_default_tier_to_the_whole_invocation() -> None:
    """With no `-m` typed, the default tier deselects this file's integration and slow arms."""
    selected, collected, deselected = _collect()
    assert deselected == 2 and collected - selected == 2, (
        f"a bare `pytest` collected {selected}/{collected} with {deselected} deselected; the default "
        f"tier must deselect exactly the `integration` and `slow` functions this file carries. If "
        f"deselected is 0 the superset is running — `addopts` in pyproject.toml has lost `-m "
        f"'{DEFAULT_TIER}'`, which is the three-sitting defect this file exists to hold shut."
    )


def test_an_explicit_integration_tier_still_overrides_the_default() -> None:
    """A later `-m integration` must win, or the CI integration tier runs the default tier twice."""
    selected, collected, deselected = _collect("-m", "integration")
    assert selected == 1 and deselected == collected - 1, (
        f"`-m integration` selected {selected}/{collected}; it must select exactly the one "
        "`integration`-marked function here"
    )


def test_an_empty_marker_expression_counts_the_whole_tree() -> None:
    """An empty `-m` clears the default and deselects nothing, so a tree count is not a tier count."""
    selected, collected, deselected = _collect("-m", "")
    assert deselected == 0 and selected == collected, (
        f"`-m ''` still deselected {deselected} of {collected}; gate 3c's whole-tree count would "
        "be a tier count"
    )


def test_the_default_tier_expression_is_the_one_make_and_ci_pin() -> None:
    """`addopts` and `make test` name the SAME tier string, read from both files."""
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert f"-m '{DEFAULT_TIER}'" in addopts, addopts
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert f'pytest -m "{DEFAULT_TIER}"' in makefile, "make test no longer names the default tier"


def test_the_run_header_states_the_tier_it_resolved(request: pytest.FixtureRequest) -> None:
    """The header states the tier it resolved, derived from the live marker expression."""
    markexpr = request.config.getoption("markexpr") or ""
    # No `-q`: quiet mode suppresses the header this arm reads.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-m", markexpr, str(THIS)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    header = [ln for ln in proc.stdout.splitlines() if ln.startswith("TIER:")]
    assert header, f"no `TIER:` header line in:\n{proc.stdout[:1200]}"
    expected = f"-m {markexpr!r}" if markexpr else "NONE (whole tree, every marker)"
    assert expected in header[0], (header[0], expected)


# The witnesses the arms above collect: one of each tier, deliberately trivial.
def test_plain_witness_is_in_the_default_tier() -> None:
    assert True


@pytest.mark.integration
def test_integration_witness_is_outside_the_default_tier() -> None:
    assert True


@pytest.mark.slow
def test_slow_witness_is_outside_the_default_tier() -> None:
    assert True
