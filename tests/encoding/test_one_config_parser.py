"""AUDIT-1 F-45 — every config-shaped YAML read goes through ONE parser.

THE DEFECT. Four readers of the same files disagreed. `config.loader.load_config` refused a
duplicate key; `encoding.audit_sections` §4 — the section whose stated job is to report on
"whatever `load_config` accepts" — used a bare `yaml.safe_load`, which is LAST-WINS, so it
could report clean on a file the loader refuses; `tools/mint_config.py` used a third call;
and the loader's own read passed no `encoding=`, so it decoded through the platform codepage
and would raise `UnicodeDecodeError` on any non-UTF-8 locale (gate 16's rule, whose ENFORCED
scope is `tools/` and `tests/` but whose RULE is the whole tree).

An auditor strictly more permissive than the thing it audits is not an auditor. These rows
pin the parser, its refusal, and the census that keeps a fourth one from appearing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mantis.config.loader import load_config
from mantis.util.yaml_io import DuplicateKeyError, parse_config_yaml

_REPO = Path(__file__).resolve().parents[2]
_DUP = "seed: 1\nseed: 2\n"


def test_the_one_parser_refuses_a_duplicate_key(tmp_path: Path) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text(_DUP, encoding="utf-8")
    with pytest.raises(DuplicateKeyError, match="duplicate key"):
        parse_config_yaml(p)


def test_the_encoding_audit_refuses_the_SAME_file_the_loader_refuses(tmp_path: Path) -> None:
    """The load-bearing row. `tests/config/test_loader_duplicate_key.py` proves the LOADER
    refuses this shape; before the repair the audit's §4 accepted it silently, last-wins.
    Both now go through the same parser, so one refusal implies the other."""
    variants = tmp_path / "variants"
    variants.mkdir()
    (variants / "dup.yaml").write_text(_DUP, encoding="utf-8")

    from mantis.encoding.audit import AuditReport
    from mantis.encoding.audit_sections import _section_variants

    report = AuditReport()
    _section_variants(report, variants)
    statuses = [row[-1] for row in report.sections["§4"].rows]
    assert "DUPLICATE-KEY" in statuses, (
        f"§4 accepted a duplicate-key config the loader refuses; rows: {report.sections['§4'].rows}"
    )
    with pytest.raises(DuplicateKeyError):
        load_config(variants / "dup.yaml")


def test_the_parser_decodes_utf8_regardless_of_locale(tmp_path: Path) -> None:
    """The read is explicitly UTF-8. A non-ASCII byte in a comment or a note field used to
    depend on the platform codepage."""
    p = tmp_path / "utf8.yaml"
    p.write_text("# nøtes — em-dash and ø\nseed: 1\n", encoding="utf-8")
    assert parse_config_yaml(p) == {"seed": 1}


def test_no_second_config_yaml_parser_in_src_or_tools() -> None:
    """Structure, not text: an AST census for `yaml.safe_load` / `yaml.load` outside the one
    owner. `mint_config.py` still calls `safe_load`/`safe_dump` on STRINGS it just rendered
    (the header round-trip), which is not a file read and not a second authority — the census
    is over calls whose argument is a file read, so those do not match."""
    owner = Path("src/mantis/util/yaml_io.py")
    offenders: dict[str, list[str]] = {}
    for root in ("src/mantis", "tools"):
        for path in sorted((_REPO / root).rglob("*.py")):
            rel = path.relative_to(_REPO)
            if rel == owner:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"safe_load", "load"}:
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "yaml"):
                    continue
                arg = node.args[0] if node.args else None
                reads_a_file = isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) \
                    and arg.func.attr in {"read_text", "read", "open"}
                if reads_a_file:
                    offenders.setdefault(str(rel), []).append(f"line {node.lineno}")
    assert not offenders, (
        f"a second config-YAML file parser: {offenders}. "
        "`mantis.util.yaml_io.parse_config_yaml` is the one owner (AUDIT-1 F-45)."
    )
