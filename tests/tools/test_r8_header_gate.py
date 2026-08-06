# >300 justify (R8): ONE gate's producer suite. The MUST-FIRE and MUST-NOT-FIRE tables are a
# single discrimination claim, not two lists — every row of each is a near-miss of the other
# (a terse `L`-unit size is a tally; a bare number beside a deleted module's path is not; the
# cap token written in LOC form is silent while a smaller LOC figure is a count), and that is
# exactly the pairing a reviewer must see on one screen before widening either. Splitting them
# lets a new MUST-FIRE row silently contradict a MUST-NOT-FIRE row in the other file, which is
# the false-positive class that gets a gate disabled. The presence rows and the whole-tree
# baseline drive the SAME spec-loaded `GATE` object through the SAME `check_file`, and R5 bars
# the cross-test import that would rejoin a split. NOTE for anyone editing above line 80: this
# file quotes banned headers as data, so its own justification must stay the FIRST marker in
# the file, and must itself quote no figure, or the gate fails itself.
"""CI gate 15's producer test (LAW-07): the R8 justification detector must BITE.

Both halves are exercised through `r8_header_gate.check_file` -- the SAME function the gate's
scan calls. An oracle that re-implemented the decision could drift from the thing it certifies,
which is the defect class this repo keeps finding (gate 11's docstring says so in its own words).

The no-count corpus below is not invented. Every MUST-FIRE string is a real header from this
tree at the adoption commit, and every MUST-NOT-FIRE string is a real clause that a naive
detector flags: `r153 line-dispersal` and `a stale size in an R8 line` are the two false
positives the first draft produced, and the deleted-predecessor sizes are the case the rule
deliberately permits (a number that counts a frozen file can never need re-editing).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "r8_header_gate.py"


def _load_gate():
    """Load the gate by PATH.

    R5/LAW-17 ban `sys.path` mutation and `tools/` is not an importable package, so the gate is
    spec-loaded from its file exactly as the other gate oracles do it.
    """
    spec = importlib.util.spec_from_file_location("_r8_header_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _body(n: int) -> list[str]:
    return [f"x = {i}" for i in range(n)]


# --- half 1: the no-count rule ------------------------------------------------------------

MUST_FIRE = [
    pytest.param(
        "# >300 justify (R8), stated at this file's MEASURED size of 488 lines (was 430 at K-B",
        id="measured-size-of-N-lines",
    ),
    pytest.param(
        "# R8 >300 justify (697, re-measured by `wc -l` at WP12-R Phase O; was 609 at WPMAIN",
        id="bare-parenthetical-plus-chronicle",
    ),
    pytest.param(
        "# R8 >300 justify (697): the manifest ROWS are data and their reason text IS the row",
        id="bare-parenthetical-no-unit-word",
    ),
    pytest.param(
        "// R8 >300 justify (MEASURED 310 lines, `wc -l`): one oracle -- the call sequence",
        id="MEASURED-N-lines",
    ),
    pytest.param(
        "# R8 >300 justify (402, MEASURED by `wc -l`; was 330 at Phase O): the file builds both",
        id="idiom-without-adjacent-digits",
    ),
    pytest.param(
        "# >300 justify (R8, by 8 lines): O-D3 and O-D4 are two instruments on ONE family",
        id="overage-clause",
    ),
    pytest.param(
        "//! R8-justify (662 lines): the relocated O-16..O-30 HEXG oracle roster binds ONE type",
        id="rust-marker-parenthetical",
    ),
    pytest.param(
        "# >300 justify: SC-A2's dict literal is what pushed this file from 292 to 303 lines",
        id="growth-chronicle",
    ),
    pytest.param(
        "# >300 justify: the per-field error-collection loop is >100 LOC by design",
        id="portion-count-in-LOC",
    ),
    pytest.param(
        "# >300 justify: one cohesive surface (the old `training/anchor.py`, 659 L)",
        id="terse-L-unit",
    ),
]


@pytest.mark.parametrize("header", MUST_FIRE)
def test_the_no_count_rule_fires(header: str) -> None:
    violations, _over, _marker = GATE.check_file("f.py", [header, *_body(400)])
    assert violations, (
        f"gate 15 MISSED a stated line count: {header!r}. This is the class that let "
        "run.py claim 867 against 1024."
    )
    assert "states a line count" in violations[0]


MUST_NOT_FIRE = [
    pytest.param(
        "# >300 justify (R8): one seam, one set of drivable fakes.",
        id="plain-reason",
    ),
    pytest.param(
        "// Exceeds the 300-line soft cap (R8): the full PyBoard surface (~40 methods)",
        id="cap-token-hyphenated",
    ),
    pytest.param(
        "//! R8: >300 LOC by design -- the save path and the two-pass atomic load are one unit",
        id="cap-token-in-LOC-form",
    ),
    pytest.param(
        "# R8 justification: 300+ lines. R181 requires one artifact to carry four things",
        id="cap-token-with-plus",
    ),
    pytest.param(
        "# >300 justify: NEAR-VERBATIM port of THE batch contract reader. The 18 named "
        "contract errors, the resolver and the two check layers are ONE contract",
        id="digits-that-are-not-a-tally",
    ),
    pytest.param(
        "# >300 justify: combines three old modules (`training/warmstart_launch.py` 190, "
        "`training/warmstart_value_head.py` 197, `training/gnn_warmstart.py` 144) into ONE",
        id="deleted-predecessors-sizes",
    ),
    pytest.param(
        "// R8 >300 justify: a 40-ply DISPERSED seed prefix (the r153 line-dispersal rule)",
        id="register-id-then-the-word-line",
    ),
    pytest.param(
        "# >300 justify: a stale size in an R8 line is a false statement, so none is stated",
        id="the-rules-own-name-then-line",
    ),
    pytest.param(
        "# >300 justify (R8): O-T1..O-T7 are one cohesive contract over ONE seam",
        id="dotted-oracle-ids",
    ),
]


@pytest.mark.parametrize("header", MUST_NOT_FIRE)
def test_the_no_count_rule_is_silent_on_legitimate_prose(header: str) -> None:
    violations, _over, marker = GATE.check_file("f.py", [header, *_body(400)])
    assert marker is True, f"gate 15 did not even see a justification in {header!r}"
    assert not violations, (
        f"gate 15 FALSE POSITIVE on a legitimate reason: {header!r} -> {violations}. "
        "A gate that fires on correct code gets disabled within a week."
    )


def test_the_rule_holds_across_a_wrapped_multi_line_clause() -> None:
    """The real headers wrap mid-clause, so `N` and `lines` land on different rows."""
    header = [
        "# >300 justify: SC-A2's explicit dict literal pushed this file from 292 to 303",
        "# lines; WPCLEAN Phase LT's type-visibility guards took it further.",
    ]
    violations, _over, _marker = GATE.check_file("f.py", [*header, *_body(400)])
    assert violations, "a count split across two comment rows must still be caught"


def test_a_count_outside_the_justification_paragraph_is_not_the_headers_problem() -> None:
    """The block ends at the paragraph. Prose further down the docstring is not a justification.

    This is deliberate and it is what keeps the gate off unrelated documentation -- including
    this gate's own docstring, which quotes every banned form as an example.
    """
    lines = [
        '"""Module summary.',
        "",
        ">300 justify: one seam, one harness.",
        "",
        "The sizing budget below was measured at 9.431 GiB over 240 lines of fixture.",
        '"""',
        *_body(400),
    ]
    violations, _over, marker = GATE.check_file("f.py", lines)
    assert marker is True
    assert not violations


# --- half 2: the presence rule ------------------------------------------------------------


def test_exactly_at_the_cap_needs_no_justification() -> None:
    violations, over, _marker = GATE.check_file("f.py", _body(GATE.CAP))
    assert over is False
    assert not violations


def test_one_line_over_the_cap_needs_one() -> None:
    violations, over, _marker = GATE.check_file("f.py", _body(GATE.CAP + 1))
    assert over is True
    assert violations and "no R8 justification" in violations[0]


def test_an_empty_file_is_not_a_violation() -> None:
    violations, over, marker = GATE.check_file("f.py", [])
    assert (over, marker) == (False, False)
    assert not violations


@pytest.mark.parametrize(
    ("name", "header"),
    [
        ("py-hash-comment", ["# >300 justify (R8): one unit, one harness."]),
        ("py-shebang-then-comment",
         ["#!/usr/bin/env python3", "# >300 justify (R8): one unit, one harness."]),
        ("py-docstring", ['"""Summary.', "", ">300 justify: one unit, one harness.", '"""']),
        ("rs-line-comment", ["// Exceeds the 300-line soft cap (R8): one auditable unit."]),
        ("rs-inner-doc", ["//! R8-justify: one cohesive port unit lifted from the frozen band."]),
        ("rs-loc-form", ["//! R8: >300 LOC by design -- one indivisible format unit."]),
    ],
)
def test_every_house_style_of_header_is_recognised(name: str, header: list[str]) -> None:
    """Six spellings exist in this tree. A gate that knew one would fail the other five."""
    violations, _over, marker = GATE.check_file("f.py", [*header, *_body(400)])
    assert marker is True, f"{name}: gate 15 did not recognise the justification"
    assert not violations, f"{name}: {violations}"


def test_a_justification_below_the_marker_window_does_not_count() -> None:
    """80 lines, not the whole file. The auditor's `grep the file` criterion called two files
    justified on the strength of an `R8` token in a test body 400 lines down."""
    lines = [*_body(GATE.MARKER_WINDOW + 5), "# >300 justify (R8): too late to be a header.",
             *_body(400)]
    violations, _over, marker = GATE.check_file("f.py", lines)
    assert marker is False
    assert violations and "no R8 justification" in violations[0]


def test_the_deepest_real_marker_in_the_tree_is_inside_the_window() -> None:
    """The window is 80 because the corpus put a real justification at line 75, not because 80
    is a round number. If a legitimate header ever sits deeper, this fails rather than the file."""
    deepest = 0
    for path in GATE.source_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        marker = GATE.find_marker(lines)
        if marker is not None:
            deepest = max(deepest, marker + 1)
    assert 0 < deepest <= GATE.MARKER_WINDOW, f"deepest marker is at line {deepest}"


# --- the gate as a whole ------------------------------------------------------------------


def _scratch_tree(root: Path, rel: str, text: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_a_non_utf8_source_is_reported_not_crashed_on(tmp_path: Path) -> None:
    """Gates 9 and 10 BOTH crashed with UnicodeDecodeError for want of `encoding=`. A gate that
    dies on the input it is meant to judge reports nothing at all (S-19)."""
    _scratch_tree(tmp_path, "src/bad.py", "# >300 justify\n".encode() + b"x = '\xff\xfe'\n")
    violations, _over, _headers, _scanned = GATE.scan(tmp_path)
    assert violations and "not UTF-8" in violations[0]


def test_the_scan_finds_a_planted_defect_in_a_scratch_tree(tmp_path: Path) -> None:
    _scratch_tree(tmp_path, "src/a.py", "\n".join(_body(400)) + "\n")
    _scratch_tree(
        tmp_path,
        "tests/b.py",
        "# >300 justify (R8), at this file's MEASURED size of 12 lines.\n"
        + "\n".join(_body(400))
        + "\n",
    )
    violations, over_cap, headers, _scanned = GATE.scan(tmp_path)
    assert over_cap == 2
    assert headers == 1
    assert len(violations) == 2
    assert any("no R8 justification" in v for v in violations)
    assert any("states a line count" in v for v in violations)


def test_gate_is_green_on_the_committed_tree() -> None:
    """The baseline R98 requires: a gate may only be adopted over a clean baseline."""
    violations, over_cap, headers, scanned = GATE.scan()
    assert not violations, "gate 15 baseline is dirty:\n" + "\n\n".join(violations)
    assert over_cap >= GATE.MIN_OVER_CAP
    assert headers >= GATE.MIN_HEADERS
    for root, floor in GATE.MIN_FILES.items():
        assert scanned[root] >= floor, f"{root}/ scanned {scanned[root]}, floor {floor}"


def test_the_non_vacuity_floors_would_fire_on_an_empty_tree(tmp_path: Path) -> None:
    """`scanned nothing, found nothing` must never read as green (gate 16's own lesson)."""
    _, over_cap, headers, scanned = GATE.scan(tmp_path)
    assert over_cap < GATE.MIN_OVER_CAP and headers < GATE.MIN_HEADERS
    assert all(scanned[root] < floor for root, floor in GATE.MIN_FILES.items())


def test_the_floors_are_per_root_because_one_global_floor_was_measured_too_weak() -> None:
    """A single corpus-wide floor let a typo that dropped `src/` entirely report GREEN.

    Mutation-tested at adoption: renaming `src` in `ROOTS` left 109 over-cap files, which
    cleared the then-floor of 100. Each root now carries its own floor, so losing any one of
    them is fatal on its own. This test pins the SHAPE, not the numbers.
    """
    _violations, _over, _headers, scanned = GATE.scan()
    assert set(GATE.MIN_FILES) == set(GATE.ROOTS), (
        "every scanned root needs its own floor, or losing that root reads as green"
    )
    for root, floor in GATE.MIN_FILES.items():
        assert 0 < floor <= scanned[root], f"{root}/ floor {floor} vs {scanned[root]} scanned"


def test_the_gates_own_self_test_passes() -> None:
    """LAW-07: the trigger proves it can fire on every invocation, including this one."""
    assert GATE.self_test() == 0


def test_the_self_test_covers_both_halves() -> None:
    """A self-test that only ever asserts green is decoration."""
    assert any(must_fire for _n, _h, must_fire in GATE.SELF_TEST)
    assert any(not must_fire for _n, _h, must_fire in GATE.SELF_TEST)
