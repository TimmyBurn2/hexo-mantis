"""The exit-code family table in `docs/design/repo_design.md` is DERIVED, not transcribed.

Collects every `*_EXIT_CODE` constant defined under `src/mantis/**` by AST and compares it with
the rc tables parsed out of the design doc, in BOTH directions, so a constant with no row and a
row with no constant are equally loud. Gate 13 reads only the run-config contract doc, so the
design doc's binding tables have no other coverage.

A hand-written list here would be a second authority that goes stale the first time someone
adds a code — the class that let an undocumented rc 71 through.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "mantis"
_DOC = _REPO / "docs" / "design" / "repo_design.md"

#: A table row in the exit-code family. The rc column must be a bare integer, which is what
#: distinguishes these tables from every other one in the document.
_ROW = re.compile(r"^\s*\|\s*(\d+)\s*\|(.+?)\|", re.MULTILINE)

#: EVERY constant-shaped name in the cell — `findall`, not one match: row 42's cell carries two
#: names, and a one-name-per-row parser would report the alias absent. Dotted prefixes strip.
_CONST = re.compile(r"([A-Z][A-Z0-9_]*_EXIT_CODE)\b")


def _defined_exit_codes(root: Path) -> dict[str, int]:
    """Every module-level `NAME_EXIT_CODE = <int literal>` under `root`, by AST."""
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            targets = (
                [node.target] if isinstance(node, ast.AnnAssign)
                else node.targets if isinstance(node, ast.Assign)
                else []
            )
            value = getattr(node, "value", None)
            if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id.endswith("_EXIT_CODE"):
                    found[target.id] = int(value.value)
    return found


def _documented_exit_codes(doc: Path) -> dict[str, int]:
    """Every constant named in an rc table row, mapped to that row's rc."""
    documented: dict[str, int] = {}
    for rc, cell in _ROW.findall(doc.read_text(encoding="utf-8")):
        for name in _CONST.findall(cell):
            documented[name] = int(rc)
    return documented


def test_every_exit_code_constant_is_declared_in_the_design_table() -> None:
    """BOTH directions: an undocumented code a supervisor may read, and a row for a constant
    that no longer exists, are equally wrong."""
    defined = _defined_exit_codes(_SRC)
    documented = _documented_exit_codes(_DOC)

    assert defined, "the AST collector found no exit-code constants at all — it is broken"
    missing = sorted(set(defined) - set(documented))
    assert not missing, (
        f"exit-code constants with NO row in {_DOC.name}: {missing}. A supervisor reads these "
        "numbers; a code the design does not name is a contract nobody agreed to"
    )
    stale = sorted(set(documented) - set(defined))
    assert not stale, (
        f"{_DOC.name} names exit-code constants that `src/mantis/**` no longer defines: "
        f"{stale}. The table is describing a contract the code has abandoned"
    )


def test_the_documented_rc_equals_the_constants_actual_value() -> None:
    """Names matching is not enough — the NUMBER is the contract, and a wrong one reads as
    verified."""
    defined = _defined_exit_codes(_SRC)
    documented = _documented_exit_codes(_DOC)
    wrong = {name: (documented[name], defined[name])
             for name in sorted(set(defined) & set(documented))
             if documented[name] != defined[name]}
    assert not wrong, f"documented rc != the constant's value (doc, code): {wrong}"


def test_the_row_42_cell_carrying_TWO_constants_yields_BOTH() -> None:
    """THE PARSER ROW: row 42's cell is the document's only one carrying TWO constant names,
    so the multi-name extraction is a claim rather than an accident of the regex."""
    documented = _documented_exit_codes(_DOC)
    assert documented.get("WATCHDOG_STALL_EXIT_CODE") == 42
    assert documented.get("SELFPLAY_STALL_EXIT_CODE") == 42, (
        "the alias inside row 42's parenthetical was not extracted; the census would report a "
        "live constant as undocumented"
    )
    cell = next(c for rc, c in _ROW.findall(_DOC.read_text(encoding="utf-8")) if rc == "42")
    assert len(_CONST.findall(cell)) == 2, (
        f"row 42 must still be the two-name cell this row exists for; got {cell!r}"
    )


def test_the_census_bites_a_planted_undocumented_constant(tmp_path) -> None:
    """MUTATION SELF-TEST, both halves, against synthetic inputs, so the census is proven to
    bite without anyone editing the real table."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text(
        "ALPHA_EXIT_CODE: int = 42\nBETA_EXIT_CODE = 99\nNOT_A_CODE = 7\n", encoding="utf-8",
    )
    doc = tmp_path / "doc.md"
    doc.write_text(
        "| rc | constant |\n|---|---|\n| 42 | `ALPHA_EXIT_CODE` |\n", encoding="utf-8",
    )

    defined = _defined_exit_codes(src)
    documented = _documented_exit_codes(doc)
    assert defined == {"ALPHA_EXIT_CODE": 42, "BETA_EXIT_CODE": 99}, (
        f"the AST collector must take int-valued *_EXIT_CODE names and nothing else; {defined}"
    )
    assert set(defined) - set(documented) == {"BETA_EXIT_CODE"}, (
        "the undocumented-constant direction did not bite a planted constant"
    )

    doc.write_text(
        "| rc | constant |\n|---|---|\n| 42 | `ALPHA_EXIT_CODE` |\n"
        "| 55 | `GHOST_EXIT_CODE` |\n", encoding="utf-8",
    )
    assert set(_documented_exit_codes(doc)) - set(defined) == {"GHOST_EXIT_CODE"}, (
        "the stale-row direction did not bite a planted row"
    )

    doc.write_text(
        "| rc | constant |\n|---|---|\n| 43 | `ALPHA_EXIT_CODE` |\n", encoding="utf-8",
    )
    assert _documented_exit_codes(doc)["ALPHA_EXIT_CODE"] == 43 != defined["ALPHA_EXIT_CODE"], (
        "the value check has nothing to compare — a wrong number would read as verified"
    )
