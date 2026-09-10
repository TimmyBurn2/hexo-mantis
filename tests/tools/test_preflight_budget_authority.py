"""One authority for the preflight wall-clock budget.

This file fails if any `tests/tools/` module re-grows a literal `--timeout-sec` for a real tool
drive, or if the harness ceiling stops exceeding the tool budget. It does NOT pin the budget's
value: that lives with its grounds in `conftest.py`, and re-transcribing it here would make this
file a second authority for it.

The short refusal-path budgets are deliberately allowlisted below: they bound drives that must
die before a boot, so the small literal is the claim rather than a guess about machine speed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_TOOLS_TESTS = Path(__file__).resolve().parent

#: Budgets that deliberately stay small literals: each bounds a drive that must fail BEFORE a
#: real boot, so the number IS the assertion ("fails fast"), not a guess about the host.
_FAST_REFUSAL_BUDGETS = {"45", "45.0", "60", "60.0", "120", "120.0"}

#: Every spelling of a `--timeout-sec` literal in this tree: a pattern narrower than the code it
#: guards reports clean for the wrong reason, and both the comma and colon shapes are present.
_TIMEOUT_LITERAL = re.compile(
    r"""--timeout-sec(?:=(?P<eq>[0-9.]+)|["']\s*[,:]\s*["'](?P<sep>[0-9.]+)["'])"""
)


def _literals(text: str) -> set[str]:
    return {m.group("eq") or m.group("sep") for m in _TIMEOUT_LITERAL.finditer(text)}


def test_the_harness_ceiling_always_exceeds_the_tool_budget(
    preflight_budget_sec, preflight_harness_ceiling_sec
) -> None:
    """Prove the harness ceiling exceeds the tool budget.

    If `subprocess.run(timeout=...)` fires first, the tool never writes the report the tests read
    and a tool verdict is reported as a harness timeout. Both numbers arrive through fixtures
    because a cross-test import of the conftest is barred.
    """
    assert preflight_harness_ceiling_sec > preflight_budget_sec


def test_the_PATTERN_matches_something_not_merely_the_substring() -> None:
    """Prove the compiled pattern yields literals, on a sample carrying all three spellings.

    Asserting the substring instead would say nothing about the pattern, and a regex that
    matched nothing would pass every parametrized row below for free.
    """
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
    """Return the files under scan, excluding this one.

    The exclusion is load-bearing: the self-test above carries a sample containing all three
    literal spellings on purpose.
    """
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
