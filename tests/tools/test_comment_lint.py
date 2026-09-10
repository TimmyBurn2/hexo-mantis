"""The comment-length lint's producer test (LAW-07): the measures must BITE.

Every arm drives the SAME `measure_source` / `verdict` / `parse_floor` the gate itself calls,
spec-loaded by path because `tools/` is not an importable package and R5 bars `sys.path` writes.
The mutation arm is the load-bearing one: it plants a comment block and a banner into real
source and requires the measure to move, so a lint that silently measured nothing would fail
here rather than print a green over an empty scan.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "tools" / "ci_gates" / "comment_lint.py"
FLOOR_PATH = REPO_ROOT / "tools" / "ci_gates" / "comment_length_floor.txt"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_comment_lint_under_test", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _flat(value: int) -> dict[str, int]:
    return dict.fromkeys(GATE._FIELDS, value)


def test_the_gate_self_test_passes_so_its_trigger_is_live():
    assert GATE.self_test() == 0


@pytest.mark.parametrize(("src", "want"), [
    ("# a\n# b\n# c\n# d\n# e\nx = 1\n", 3),
    ("# a\n# b\nx = 1\n", 0),
    ("# a\n# b\n# c\nx = 1\n# d\n# e\n# f\n", 2),
    ("x = 1  # t\ny = 2  # t\nz = 3  # t\nw = 4  # t\n", 0),
    ("    # a\n    # b\n    # c\n    # d\n", 2),
])
def test_a_run_of_own_line_comments_is_measured_beyond_the_cap(src: str, want: int):
    assert GATE.measure_source("a.py", src).comment_excess_lines == want


def test_a_trailing_comment_does_not_join_the_block_above_it():
    joined = "# a\n# b\nx = 1  # c\n# d\n"
    assert GATE.measure_source("a.py", joined).comment_excess_lines == 0


@pytest.mark.parametrize("banner", [
    "# ────────────────────",
    "# ==================== section ====================",
    "# --------------------------------------------------",
    "# ~~~~~~~~~~~~~~~~~~~~",
])
def test_a_rule_of_repeated_characters_is_a_banner(banner: str):
    assert GATE.measure_source("a.py", banner + "\nx = 1\n").banner_comment_lines == 1


@pytest.mark.parametrize("ordinary", [
    "# a plain sentence about the invariant",
    "# the value is 50, measured at 240.2 s",
    "#!/usr/bin/env python3",
    "# -*- coding: utf-8 -*-",
    "# fmt: off",
])
def test_an_ordinary_or_functional_comment_is_not_a_banner(ordinary: str):
    assert GATE.measure_source("a.py", ordinary + "\nx = 1\n").banner_comment_lines == 0


@pytest.mark.parametrize(("src", "want"), [
    ('"""One line."""\n', 0),
    ('"""One\ntwo\nthree"""\n', 2),
    ('def f():\n    """One\n    two"""\n', 1),
    ('class C:\n    """One line."""\n', 0),
    ("x = 1\n", 0),
    ('x = """not a docstring\nbut a value"""\n', 0),
])
def test_docstring_excess_counts_lines_beyond_the_first(src: str, want: int):
    assert GATE.measure_source("a.py", src).docstring_excess_lines == want


def test_a_rust_comment_run_is_measured_and_a_slash_slash_in_a_string_is_not():
    src = '// a\n// b\n// c\n// d\nfn f() { let s = "// not a comment // at all"; }\n'
    got = GATE.measure_source("a.rs", src)
    assert got.comment_excess_lines == 2
    assert got.docstring_excess_lines == 0


def test_a_rust_raw_string_hides_a_comment_shaped_body():
    src = 'fn f() { let s = r#"// a\n// b\n// c\n// d"#; }\n'
    assert GATE.measure_source("a.rs", src).comment_excess_lines == 0


def test_a_rust_block_comment_counts_every_line_it_spans():
    src = "/* a\n b\n c\n d\n e */\nfn f() {}\n"
    assert GATE.measure_source("a.rs", src).comment_excess_lines == 3


def test_unparseable_python_measures_as_zero_rather_than_crashing_the_gate():
    assert GATE.measure_source("a.py", "def f(:\n").comment_excess_lines == 0


@pytest.mark.parametrize(("rel", "want"), [
    ("src/mantis/run.py", True),
    ("crates/mantis-core/src/lib.rs", True),
    ("tools/ci_gates/comment_lint.py", True),
    ("tests/tools/test_comment_lint.py", True),
    ("docs/design/repo_design.md", False),
    ("docs/x.py", False),
    ("src/mantis/py.typed", False),
    ("configs/run6.yaml", False),
])
def test_the_scope_predicate_names_the_four_directories_and_two_suffixes(rel: str, want: bool):
    assert GATE.in_scope(rel) is want


@pytest.mark.parametrize(("now", "tree", "ref", "want"), [
    (10, 10, 10, 0),
    (5, 10, 10, 0),
    (11, 10, 10, 1),
    (10, 12, 10, 1),
    (8, 8, 10, 0),
    (12, 12, 10, 1),
])
def test_the_verdict_refuses_growth_and_refuses_a_raised_floor(
        now: int, tree: int, ref: int, want: int):
    rc, _ = GATE.verdict({**_flat(0), **dict.fromkeys(GATE.GATED, now)},
                         {**_flat(0), **dict.fromkeys(GATE.GATED, tree)},
                         {**_flat(0), **dict.fromkeys(GATE.GATED, ref)}, "test")
    assert rc == want


def test_a_measure_below_the_floor_says_so_without_failing():
    rc, msgs = GATE.verdict({**_flat(0), **dict.fromkeys(GATE.GATED, 5)},
                            {**_flat(0), **dict.fromkeys(GATE.GATED, 10)},
                            {**_flat(0), **dict.fromkeys(GATE.GATED, 10)}, "test")
    assert rc == 0
    assert any("ratchet the floor down" in m for m in msgs)


@pytest.mark.parametrize("text", [
    "comment_excess_lines 3\n",
    "comment_excess_lines\n",
    "not_a_measure 3\nbanner_comment_lines 1\ndocstring_excess_lines 1\n",
    "comment_excess_lines 3 4\n",
])
def test_a_malformed_or_incomplete_floor_is_refused(text: str):
    with pytest.raises(ValueError):
        GATE.parse_floor(text)


def test_a_complete_floor_parses_and_comments_in_it_are_ignored():
    got = GATE.parse_floor(
        "# grounds\ncomment_excess_lines 3\nbanner_comment_lines 1  # trailing\n"
        "docstring_excess_lines 2\nruling_cite_comment_lines 9\n")
    assert got == {"comment_excess_lines": 3, "banner_comment_lines": 1,
                   "docstring_excess_lines": 2, "ruling_cite_comment_lines": 9}


def test_the_committed_floor_parses_and_covers_every_gated_measure():
    floor = GATE.parse_floor(FLOOR_PATH.read_text(encoding="utf-8"))
    assert set(GATE.GATED) <= set(floor)
    assert all(floor[name] >= 0 for name in GATE.GATED)


def test_the_tree_measures_below_its_own_committed_floor():
    now, seen = GATE.measure_tree(REPO_ROOT)
    assert seen > 100, "the scan measured almost nothing; a green here would be a green over air"
    floor = GATE.parse_floor(FLOOR_PATH.read_text(encoding="utf-8"))
    for name in GATE.GATED:
        assert getattr(now, name) <= floor[name], (
            f"{name} is {getattr(now, name)} against a floor of {floor[name]}")


def test_planting_comments_into_real_source_moves_every_gated_measure():
    """The mutation arm: a lint that measured nothing would pass every arm above but this one."""
    original = (REPO_ROOT / "tools" / "ci_gates" / "comment_lint.py").read_text(encoding="utf-8")
    before = GATE.measure_source("tools/ci_gates/comment_lint.py", original)
    planted = original + (
        '\n\ndef _planted() -> None:\n'
        '    """Line one.\n\n    Line three.\n    """\n'
        "    # a\n    # b\n    # c\n    # d\n    # ====================\n"
        "    return None\n")
    after = GATE.measure_source("tools/ci_gates/comment_lint.py", planted)
    assert after.comment_excess_lines > before.comment_excess_lines
    assert after.banner_comment_lines > before.banner_comment_lines
    assert after.docstring_excess_lines > before.docstring_excess_lines
