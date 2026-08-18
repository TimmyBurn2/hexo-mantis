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

_TIMEOUT_LITERAL = re.compile(r'"--timeout-sec",\s*"([0-9.]+)"')


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


def test_the_scan_reaches_the_files_it_claims_to_guard() -> None:
    """LAW-07: a scan matching nothing passes vacuously."""
    hits = [p.name for p in _TOOLS_TESTS.glob("test_preflight*.py")
            if "--timeout-sec" in p.read_text(encoding="utf-8")]
    assert len(hits) >= 3, f"the scan reached only {hits}"


@pytest.mark.parametrize(
    "path", sorted(_TOOLS_TESTS.glob("test_preflight*.py")), ids=lambda p: p.name
)
def test_no_module_re_transcribes_a_full_drive_budget(path: Path) -> None:
    literals = set(_TIMEOUT_LITERAL.findall(path.read_text(encoding="utf-8")))
    rogue = sorted(literals - _FAST_REFUSAL_BUDGETS)
    assert not rogue, (
        f"{path.name} carries literal --timeout-sec {rogue}. A budget that bounds a REAL boot + "
        "burst + terminal eval encodes how fast the machine is, which no test can transcribe: "
        "take it from the `preflight_budget_sec` fixture. If this drive is meant to die BEFORE "
        "a boot, add its value to _FAST_REFUSAL_BUDGETS with the reason."
    )
