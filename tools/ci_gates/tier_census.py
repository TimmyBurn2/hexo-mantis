#!/usr/bin/env python3
"""Gate 3c's second arm — every DESELECTING marker in `tests/` is declared.

AUDIT-1 F-12. Gate 3c counts COLLECTED tests. `pytest --collect-only` with `-m ''` walks the
whole tree, so a test carrying `@pytest.mark.slow`, `@pytest.mark.skip`, or a host-true
`@pytest.mark.skipif` is COUNTED by the floor and EXECUTED by nothing: `make test` runs
`-m "not integration and not slow"` and `make test.integration` runs `-m integration`. One
`slow` line is therefore enough to remove a test from every tier the repo runs while the
collected-test floor, the suite's own green, and every other gate stay rc 0.

`tests/model/conformance/test_leaf_forward_throughput_harness.py` already refuses exactly this
for its own suite — `marker_census` + `require_declared_tier_placement`, scoped to
`_SLOW_TIER_MEMBERS` — after RED-TEAM planted one `slow` line above a cross-crate legal-set
claim and every instrument in the repo reported green. This is that guard's TREE-WIDE twin.
R5 bars importing the conformance helper from here (`tests` is not a package), so the census
is re-implemented rather than shared; the two are checked against each other by
`tests/tools/test_tier_census.py`.

BOTH DIRECTIONS ARE REFUSED, for the reason the conformance guard states: an UNDECLARED marker
is a test that left its tier without anyone saying so, and a STALE declaration is a standing
licence nobody is using, behind which the next real one is invisible.

WHY A DECLARATION FILE AND NOT A COUNT. A count answers "did the number change"; the question
is "which tests stopped running". The file names them, so a diff on it is the review.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: Markers that REMOVE a test from at least one tier the repo actually runs. `parametrize`,
#: `usefixtures` and friends are not here: they change how a test runs, never whether.
DESELECTING: frozenset[str] = frozenset({"slow", "skip", "skipif", "integration"})

#: The module-level `pytestmark` form's stand-in test name. It takes a whole file out of the
#: tier in one line and carries no decorator, which is the form a reader scanning the function
#: bodies would miss.
MODULE_SCOPE = "<module>"

REPO_ROOT = Path(__file__).resolve().parents[2]
DECLARATION = REPO_ROOT / "tools" / "ci_gates" / "tier_declaration.txt"
TESTS_ROOT = REPO_ROOT / "tests"

Row = tuple[str, str, str]


def _is_mark_root(node: ast.expr) -> bool:
    """True for the `pytest.mark` in `pytest.mark.slow` — the only spelling this repo uses."""
    return (isinstance(node, ast.Attribute) and node.attr == "mark"
            and isinstance(node.value, ast.Name) and node.value.id == "pytest")


def census(tests_root: Path) -> set[Row]:
    """`(path, test, marker)` for every deselecting marker under ``tests_root``.

    Raises:
        SyntaxError: a test module does not parse. Deliberately NOT caught — a census that
            skipped an unparseable file would report a clean tier over a file it could not
            read, which is the phantom-gate class this file exists to close.
    """
    rows: set[Row] = set()
    for path in sorted(tests_root.rglob("test_*.py")):
        rel = path.relative_to(tests_root.parent).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test_"):
                for decorator in node.decorator_list:
                    func = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if isinstance(func, ast.Attribute) and _is_mark_root(func.value) \
                            and func.attr in DESELECTING:
                        rows.add((rel, node.name, func.attr))
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            ):
                for inner in ast.walk(node.value):
                    if isinstance(inner, ast.Attribute) and _is_mark_root(inner.value) \
                            and inner.attr in DESELECTING:
                        rows.add((rel, MODULE_SCOPE, inner.attr))
    return rows


def load_declaration(path: Path) -> set[Row]:
    """The committed declaration. A file with no rows is a FAILURE, never a vacuous pass."""
    rows: set[Row] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}: malformed row {line!r} (want path<TAB>test<TAB>marker)")
        rows.add((parts[0], parts[1], parts[2]))
    if not rows:
        raise ValueError(
            f"{path} declares no deselected tests. An empty declaration would make this gate "
            "pass over any marker in the tree — the phantom-gate class (LAW-07)."
        )
    return rows


def compare(observed: set[Row], declared: set[Row]) -> tuple[list[Row], list[Row]]:
    """`(undeclared, stale)` — markers in the tree with no line, and lines with no marker."""
    return sorted(observed - declared), sorted(declared - observed)


def self_test() -> int:
    """Both refusals must FIRE. A check never shown to fail is indistinguishable from one that
    always passes — the same reasoning `ruling_census.py` and `sync_governance.py` carry."""
    base: set[Row] = {("tests/a/test_x.py", "test_one", "slow")}
    cases = [
        ("clean", base, base, ([], [])),
        ("undeclared marker", base | {("tests/a/test_x.py", "test_two", "slow")}, base,
         ([("tests/a/test_x.py", "test_two", "slow")], [])),
        ("stale declaration", set(), base, ([], [("tests/a/test_x.py", "test_one", "slow")])),
    ]
    bad = 0
    for name, observed, declared, want in cases:
        got = compare(observed, declared)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'OK' if ok else 'SELF-TEST FAILED'}] {name}: {got}")
    empty_refused = False
    try:
        load_declaration(_empty_declaration())
    except ValueError:
        empty_refused = True
    print(f"  [{'OK' if empty_refused else 'SELF-TEST FAILED'}] an empty declaration is refused")
    bad += 0 if empty_refused else 1
    print("self-test: all controls fire" if not bad else f"self-test: {bad} DID NOT FIRE")
    return bad


def _empty_declaration() -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write("# only a comment\n")
    handle.close()
    return Path(handle.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tests-root", default=str(TESTS_ROOT))
    parser.add_argument("--declaration", default=str(DECLARATION))
    parser.add_argument("--self-test", action="store_true", help="prove both refusals fire")
    parser.add_argument("--write", action="store_true",
                        help="rewrite the declaration from the tree (a REVIEWED act: the "
                             "diff is what a reader checks, so never run it to make a red "
                             "gate green without reading what moved)")
    args = parser.parse_args(argv)

    if args.self_test:
        return 1 if self_test() else 0

    observed = census(Path(args.tests_root))
    if args.write:
        path = Path(args.declaration)
        header = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip() or line.lstrip().startswith("#")
        ).rstrip("\n")
        path.write_text(header + "\n" + "".join(
            "\t".join(row) + "\n" for row in sorted(observed)), encoding="utf-8")
        print(f"tier census: wrote {len(observed)} row(s) to {path}")
        return 0

    declared = load_declaration(Path(args.declaration))
    undeclared, stale = compare(observed, declared)
    if not undeclared and not stale:
        print(f"tier census: {len(observed)} deselected test(s), all declared")
        return 0
    for row in undeclared:
        print(f"UNDECLARED  {row[0]}::{row[1]}  @pytest.mark.{row[2]}", file=sys.stderr)
    for row in stale:
        print(f"STALE       {row[0]}::{row[1]}  @pytest.mark.{row[2]}", file=sys.stderr)
    print(
        f"\ntier census FAIL: {len(undeclared)} undeclared, {len(stale)} stale.\n"
        "  A DESELECTED test is still COLLECTED, so the gate-3c floor, the suite's own green\n"
        "  and every other gate stay rc 0 while the test runs in no tier at all.\n"
        f"  Declare the change in {args.declaration} (or `--write` and READ THE DIFF).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
