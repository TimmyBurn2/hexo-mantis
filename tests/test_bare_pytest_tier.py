"""A bare `pytest` resolves to the DEFAULT TIER, and the other invocations still mean what they
meant (R330(g)).

THE DEFECT, measured three times before it was fixed. The marker expression
`-m "not integration and not slow"` lived ONLY in the Makefile's `test` target; `pyproject.toml`'s
`addopts` carried none. A bare `uv run pytest -q` therefore ran the integration superset — ~35
minutes against ~3 for the tier — and three consecutive box sittings each disclosed having done
exactly that (sitting 4 Δ10.13, sitting 5 item 1, sitting 6 §11.1). "The lesson has now failed to
transfer three times, which is itself the finding: it is not a memory problem, it is a missing
guard" — this file is the guard's witness.

WHY THE WITNESS IS A SUBPROCESS AND NOT AN ASSERTION ON `config.option.markexpr`. An in-process
assertion on the live marker expression would read whatever THIS run was invoked with, so it would
be green under `make test`, green under a bare `pytest`, deselected under `-m integration`, and RED
under any legitimate targeted `-m` a developer types — a pin on the invocation, not on the
mechanism. The mechanism is pytest's own argument resolution against THIS repo's `pyproject.toml`,
and only a real pytest child process exercises it. Each arm below collects THIS file — which
deliberately carries one plain, one `integration`-marked and one `slow`-marked function — and reads
pytest's own summary line: `N/M tests collected (K deselected)` is the tier applying, `M tests
collected` is the whole tree.

THE PLANTED BREAK. Remove `-m ...` from `addopts` and the bare arm reads `M tests collected` with
no deselection: the first test below fails, naming the superset. Put the expression back without
the quoting pytest's shlex split needs and every arm fails at collection. Neither the Makefile nor
CI changes: both keep their EXPLICIT `-m`, and `test_meta_ci.py` still pins them — this file adds
what happens when nobody typed one.
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
    """Run `pytest --collect-only -q <extra> <this file>` as a CHILD against the repo's own
    pyproject; return (selected, collected, deselected) read off pytest's summary line.

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
    """No `-m` typed → the default tier's marker expression deselects the integration and slow
    functions in this file, and pytest says so on its own summary line."""
    selected, collected, deselected = _collect()
    assert deselected == 2 and collected - selected == 2, (
        f"a bare `pytest` collected {selected}/{collected} with {deselected} deselected; the default "
        f"tier must deselect exactly the `integration` and `slow` functions this file carries. If "
        f"deselected is 0 the superset is running — `addopts` in pyproject.toml has lost `-m "
        f"'{DEFAULT_TIER}'`, which is the three-sitting defect this file exists to hold shut."
    )


def test_an_explicit_integration_tier_still_overrides_the_default() -> None:
    """`make test.integration` passes `-m integration`; the later expression must WIN, or the CI
    integration tier would silently run the default tier twice."""
    selected, collected, deselected = _collect("-m", "integration")
    assert selected == 1 and deselected == collected - 1, (
        f"`-m integration` selected {selected}/{collected}; it must select exactly the one "
        "`integration`-marked function here"
    )


def test_an_empty_marker_expression_counts_the_whole_tree() -> None:
    """Gate 3c passes `-m ''` so its collected-test count is of the TREE, not of a tier; an empty
    expression must clear the default and deselect nothing."""
    selected, collected, deselected = _collect("-m", "")
    assert deselected == 0 and selected == collected, (
        f"`-m ''` still deselected {deselected} of {collected}; gate 3c's whole-tree count would "
        "be a tier count"
    )


def test_the_default_tier_expression_is_the_one_make_and_ci_pin() -> None:
    """The expression in `addopts` and the one in `make test` are the SAME string, read from both
    files, so the two paths cannot name two different tiers and call them the default."""
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert f"-m '{DEFAULT_TIER}'" in addopts, addopts
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert f'pytest -m "{DEFAULT_TIER}"' in makefile, "make test no longer names the default tier"


def test_the_run_header_states_the_tier_it_resolved(request: pytest.FixtureRequest) -> None:
    """The root conftest prints `TIER: ...` derived from the LIVE marker expression, so a reader of
    any run's output can see which tier ran without reconstructing it from the command typed."""
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


# ── the witnesses the arms above collect: one of each tier, deliberately trivial ─────────
def test_plain_witness_is_in_the_default_tier() -> None:
    assert True


@pytest.mark.integration
def test_integration_witness_is_outside_the_default_tier() -> None:
    assert True


@pytest.mark.slow
def test_slow_witness_is_outside_the_default_tier() -> None:
    assert True
