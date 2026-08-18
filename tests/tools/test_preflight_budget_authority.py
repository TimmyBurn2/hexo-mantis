"""ONE authority for the preflight wall-clock budget (R46 loop under R284(f); R1's shape).

The defect this closes, measured rather than asserted: THREE tests drove the real preflight tool
with three separately-transcribed `--timeout-sec` constants (300, 400, 400), plus a fourth number
(`subprocess.run(timeout=500)`) coupled to one of them. Four authorities for one quantity. The
300 went red on the migration box while passing here with ~46% margin, and nothing in the tree
tied the four numbers together, so fixing one would have left three.

This file is the flip-set for the CLASS (R71): it fails if any `tests/tools/` module re-grows a
literal `--timeout-sec` for a REAL tool drive, and it fails if the harness ceiling stops
exceeding the tool budget. It does NOT pin the budget's VALUE — the value's grounds live beside
it in `conftest.py`, and a test that re-transcribed the number here would be the fifth authority.

Deliberately NOT covered: the short refusal-path budgets (45 / 60 / 120) that several rows use
for drives which must die BEFORE a boot — those are asserting "this fails fast", so a small
literal is the claim, not a transcription of machine speed. The allowlist below names them, and
naming them is the point: an unlisted literal is a new one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_TOOLS_TESTS = Path(__file__).resolve().parent

#: Budgets that deliberately stay small literals: each bounds a drive that must fail BEFORE a
#: real boot, so the number IS the assertion ("fails fast"), not a guess about the host.
_FAST_REFUSAL_BUDGETS = {"45", "45.0", "60", "60.0", "120", "120.0"}

#: Every spelling of a `--timeout-sec` LITERAL that has appeared in this tree, plus the ones a
#: near-miss would produce. The first version matched only `"--timeout-sec", "300"` — a comma —
#: and therefore could not see `{..., "--timeout-sec": "60"}`, which is a shape ALREADY PRESENT
#: in `test_preflight_mint.py`. A scan whose pattern is narrower than the code it guards reports
#: clean for the wrong reason.
_TIMEOUT_LITERAL = re.compile(
    r"""--timeout-sec(?:=(?P<eq>[0-9.]+)|["']\s*[,:]\s*["'](?P<sep>[0-9.]+)["'])"""
)


def _literals(text: str) -> set[str]:
    return {m.group("eq") or m.group("sep") for m in _TIMEOUT_LITERAL.finditer(text)}


def test_the_harness_ceiling_always_exceeds_the_tool_budget(
    preflight_budget_sec, preflight_harness_ceiling_sec
) -> None:
    """If `subprocess.run(timeout=...)` fires first the tool never writes the report the tests
    read, and a tool verdict is reported as a harness timeout — the wrong failure, attributed to
    the wrong thing.

    Read through the FIXTURES, not by importing the conftest: R5 bars cross-test imports, and a
    module-level `from tests.tools.conftest import ...` here would be exactly that (it also only
    resolves by accident of the rootdir being on `sys.path`)."""
    assert preflight_harness_ceiling_sec > preflight_budget_sec


def test_the_PATTERN_matches_something_not_merely_the_substring() -> None:
    """LAW-07, and the first version got this wrong in the way that matters.

    It asserted the SUBSTRING `--timeout-sec` appeared in at least three files, which says
    nothing about whether the compiled pattern matches anything. With a broken regex every
    parametrized row below would pass vacuously — the exact failure this file is named for. So
    the self-test now asserts the PATTERN itself yields literals, and asserts it against a
    synthetic sample carrying all three spellings, so a regex that silently stops matching one
    of them reds here rather than going quiet."""
    sample = (
        'run(["x", "--timeout-sec", "300"])\n'
        '{"--config": c, "--timeout-sec": "60"}\n'
        'argv = ["--timeout-sec=45"]\n'
    )
    assert _literals(sample) == {"300", "60", "45"}, _literals(sample)

    found = {p.name: _literals(p.read_text(encoding="utf-8")) for p in _scanned()}
    matched = {k: v for k, v in found.items() if v}
    assert len(matched) >= 3, (
        f"the compiled pattern matched literals in only {len(matched)} file(s): {matched}. "
        "A scan that matches nothing passes every row below for free."
    )


def _scanned() -> list[Path]:
    """The files under scan. THIS file is excluded, and the exclusion is load-bearing rather than
    convenient: the self-test above carries a synthetic sample containing all three literal
    spellings on purpose, and a scanner that flagged its own test data would have to choose
    between testing its pattern and passing itself."""
    return sorted(p for p in _TOOLS_TESTS.glob("test_preflight*.py")
                  if p.name != Path(__file__).name)


@pytest.mark.parametrize("path", _scanned(), ids=lambda p: p.name)
def test_no_module_re_transcribes_a_full_drive_budget(path: Path) -> None:
    rogue = sorted(_literals(path.read_text(encoding="utf-8")) - _FAST_REFUSAL_BUDGETS)
    assert not rogue, (
        f"{path.name} carries literal --timeout-sec {rogue}. A budget that bounds a REAL boot + "
        "burst + terminal eval encodes how fast the machine is, which no test can transcribe: "
        "take it from the `preflight_budget_sec` fixture. If this drive is meant to die BEFORE "
        "a boot, add its value to _FAST_REFUSAL_BUDGETS with the reason."
    )
