"""The exit-code family table in `docs/design/repo_design.md` is DERIVED, not transcribed.

Q3 red-team A4c: rc 71 became a code the SUPERVISOR reads the day F-816-19 put an
`os._exit(71)` inside `mantis.run.main`'s arming gate, and the design's exit-code table never
said so. Nothing caught it, and nothing could: `tools/ci_gates/contract_doc_gate.py`'s
`DEFAULT_DOC` is `docs/contracts/run_config_schema.md` and gate 13 reads no other document, so
the design doc's binding tables have NO gate coverage at all (queued informationally as RQ-12).

This file is the cheapest honest answer: a producer test with a live producer, in the default
tier, that does not mint a CI gate number (which would be its own scope decision). It collects
every `*_EXIT_CODE` constant DEFINED in `src/mantis/**` by AST and compares it with the rc
tables parsed out of the design doc — in BOTH directions, so a constant with no row and a row
with no constant are equally loud. DERIVED, NEVER TRANSCRIBED (R192(e)/G-DFIX-4, the rule gate
15 enforces on line counts): a hand-written list here would be a second authority that goes
stale the first time someone adds a code, which is precisely the class that let 71 through.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src" / "mantis"
_DOC = _REPO / "docs" / "design" / "repo_design.md"

#: A table row in the exit-code family: `| <rc> | <constant cell> | <authority> | <delivery> |`.
#: The rc column must be a bare integer, which is what distinguishes these tables from every
#: other table in the document without needing to know where they sit.
_ROW = re.compile(r"^\s*\|\s*(\d+)\s*\|(.+?)\|", re.MULTILINE)

#: EVERY constant-shaped name inside the constant cell — deliberately `findall` and not a
#: single match. Row 42's cell carries TWO names ("`WATCHDOG_STALL_EXIT_CODE`
#: (= `lifecycle.watchdog.SELFPLAY_STALL_EXIT_CODE`)"), and a one-name-per-row parser would
#: report `SELFPLAY_STALL_EXIT_CODE` absent on its very first run — a false red that a future
#: reader silences rather than fixes. The dotted prefix is stripped by the pattern itself.
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
    """BOTH directions. A constant with no row is an undocumented code a supervisor may read
    (that is exactly what rc 71 was); a row naming a constant that no longer exists is a table
    describing a contract the code has abandoned."""
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
    """Names matching is not enough — the NUMBER is the contract. A table row that says 71
    beside a constant that is 70 is worse than no row, because it reads as verified."""
    defined = _defined_exit_codes(_SRC)
    documented = _documented_exit_codes(_DOC)
    wrong = {name: (documented[name], defined[name])
             for name in sorted(set(defined) & set(documented))
             if documented[name] != defined[name]}
    assert not wrong, f"documented rc != the constant's value (doc, code): {wrong}"


def test_the_row_42_cell_carrying_TWO_constants_yields_BOTH() -> None:
    """THE PARSER ROW the fix-review required (`Q3_FIX_REVIEW.md` F7).

    Row 42's constant cell is `` `WATCHDOG_STALL_EXIT_CODE` (=
    `lifecycle.watchdog.SELFPLAY_STALL_EXIT_CODE`) `` — two constant-shaped names in ONE cell,
    the only such row in the document. A naive one-constant-per-row parser silently drops the
    alias and reports it absent on the very first run, which is a false red a future reader
    silences instead of fixing. This row pins the multi-name extraction directly, so the
    behaviour is a claim rather than an accident of the regex."""
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
    """LAW-07 MUTATION SELF-TEST, both halves, run against synthetic inputs so the checker is
    proven to bite without anyone editing the real table.

    A census that cannot fail is a green light nobody earned — and this one replaces a CI gate,
    so it carries the burden a gate would have."""
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
