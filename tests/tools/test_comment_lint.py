"""The comment-length lint's producer test (LAW-07): the measures must BITE.

Every arm drives the SAME `measure_source` / `verdict` / `parse_floor` the gate itself calls,
spec-loaded by path because `tools/` is not an importable package and R5 bars `sys.path` writes.
The mutation arm is the load-bearing one: it plants a comment block and a banner into real
source and requires the measure to move, so a lint that silently measured nothing would fail
here rather than print a green over an empty scan.
"""
from __future__ import annotations

import importlib.util
import subprocess
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
        "docstring_excess_lines 2\nprivate_docstring_excess_lines 1\nrust_doc_excess_lines 4\n"
        "ruling_cite_lines 9\ntextfile_comment_excess_lines 6\n")
    assert got == {"comment_excess_lines": 3, "banner_comment_lines": 1,
                   "docstring_excess_lines": 2, "private_docstring_excess_lines": 1,
                   "rust_doc_excess_lines": 4, "ruling_cite_lines": 9,
                   "textfile_comment_excess_lines": 6}


def test_a_reference_floor_that_predates_a_gated_measure_parses_leniently_and_ratchets_the_rest():
    """Adding a gated measure must not disable the ratchet half for the measures the reference has."""
    old = "comment_excess_lines 2\nbanner_comment_lines 2\ndocstring_excess_lines 2\n"
    with pytest.raises(ValueError):
        GATE.parse_floor(old)
    ref = GATE.parse_floor(old, strict=False)
    assert "rust_doc_excess_lines" not in ref
    tree = {**_flat(2), "comment_excess_lines": 4}
    rc, msgs = GATE.verdict(_flat(2), tree, ref, "ref")
    assert rc == 1 and any("ratchet" in m for m in msgs)
    rc, _ = GATE.verdict(_flat(2), _flat(2), ref, "ref")
    assert rc == 0


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


@pytest.mark.parametrize(("src", "want"), [
    ('def _f():\n    """one\n    two\n    three"""\n', 2),
    ('def f():\n    """one\n    two"""\n', 0),
    ('def __init__(self):\n    """one\n    two"""\n', 0),
    ('def f():\n    def g():\n        """one\n        two"""\n    return g\n', 1),
    ('class _C:\n    def m(self):\n        """one\n        two"""\n', 1),
])
def test_private_docstring_excess_counts_private_and_nested_symbols_only(src: str, want: int):
    assert GATE.measure_source("a.py", src).private_docstring_excess_lines == want


@pytest.mark.parametrize(("src", "want"), [
    ("/// a\n/// b\n/// c\nfn f() {}\n", 2),
    ("//! a\n//! b\nfn f() {}\n", 1),
    ("/// a\nfn f() {}\n", 0),
    ("// a\n// b\n// c\nfn f() {}\n", 0),
    ("/// a\n/// b\nfn f() {}\n/// c\n/// d\n/// e\nfn g() {}\n", 3),
])
def test_rust_doc_excess_counts_lines_beyond_the_first_of_each_doc_run(src: str, want: int):
    assert GATE.measure_source("a.rs", src).rust_doc_excess_lines == want


def test_the_two_new_measures_are_gated_and_in_the_committed_floor():
    assert "private_docstring_excess_lines" in GATE.GATED and "rust_doc_excess_lines" in GATE.GATED
    assert "ruling_cite_lines" in GATE.GATED and "textfile_comment_excess_lines" in GATE.GATED
    floor = GATE.parse_floor(FLOOR_PATH.read_text(encoding="utf-8"))
    assert floor["private_docstring_excess_lines"] >= 0 and floor["rust_doc_excess_lines"] >= 0
    assert floor["ruling_cite_lines"] > 0 and floor["textfile_comment_excess_lines"] > 0


def test_planting_a_private_docstring_and_a_rust_doc_run_moves_the_new_measures():
    py = 'def _f():\n    """one"""\n'
    assert GATE.measure_source("a.py", py).private_docstring_excess_lines == 0
    assert GATE.measure_source("a.py", py.replace('"""one"""', '"""one\n    two"""')).private_docstring_excess_lines == 1
    rs = "/// a\nfn f() {}\n"
    assert GATE.measure_source("a.rs", rs).rust_doc_excess_lines == 0
    assert GATE.measure_source("a.rs", "/// a\n/// b\n" + rs[6:]).rust_doc_excess_lines == 1


def test_a_bare_ruling_cite_in_a_docstring_moves_ruling_cite_lines():
    """The docstring half of the cite measure: a neutered scanner would not move."""
    plain = 'def f():\n    """no cites here"""\n'
    cited = 'def f():\n    """grounded by R123 with no clause letter"""\n'
    assert GATE.measure_source("a.py", plain).ruling_cite_lines == 0
    assert GATE.measure_source("a.py", cited).ruling_cite_lines == 1
    assert GATE.measure_source("a.py", 'x = 1  # R8 stays out\n').ruling_cite_lines == 0
    assert GATE.measure_source("a.py", 'x = 1  # R45 counts\n').ruling_cite_lines == 1


def test_a_hash_run_in_a_scoped_textfile_moves_textfile_comment_excess_lines():
    """The text-format half: three # lines are excess, and its cites count in the cite measure."""
    quiet = "# a\n# b\nx = 1\n"
    loud = "# a\n# b\n# c\n# cites R99\nx = 1\n"
    assert GATE.measure_source("tools/gate_x.sh", quiet).textfile_comment_excess_lines == 0
    m = GATE.measure_source("tools/gate_x.sh", loud)
    assert m.textfile_comment_excess_lines == 2 and m.ruling_cite_lines == 1
    assert GATE.measure_source("tests/fixtures/x.toml", loud).textfile_comment_excess_lines == 0


def test_the_tree_measures_its_two_new_measures_on_real_files(tmp_path):
    """End to end over a scratch git tree: both classes in tracked files must move the totals."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text('def f():\n    """cites R123"""\n', encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "x.sh").write_text("# a\n# b\n# c\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/a.py", "tools/x.sh"], cwd=tmp_path, check=True)
    total, _seen = GATE.measure_tree(tmp_path)
    assert total.ruling_cite_lines == 1, "a tracked docstring cite was not counted"
    assert total.textfile_comment_excess_lines == 1, "a tracked # run was not counted"
