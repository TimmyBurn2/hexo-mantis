# R8 justify: one drift check over one document, every arm answered by importing `RunConfig`
# itself, so an arm in another file would need its own copy of that derivation — and a
# transcribed key list is how the doc this gate checks rotted through four schema versions.
"""CI gate 13: docs/contracts/run_config_schema.md may not cite a config key or a `mantis.*`
symbol the shipped schema does not have.

The doc drifted for four schema versions while every gate stayed green, because nothing read it.
Every check is answered by importing the LIVE authority — `RunConfig`, the module tree,
`mantis.config.schema.leaf_paths` — never a transcribed copy, four divergent copies of the leaf
walker having already been measured. The "deliberately absent" section is checked in REVERSE:
every citation under it must FAIL to resolve, so a retired key coming back reds the gate.

The bounds: the doc need not enumerate every leaf, prose is not checked for truth, and a bare
undotted name is checked only in the first two cells of the cross-field table, where doc-wide
checking would need a word-list exemption for history the doc legitimately cites.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

from mantis.config.schema import RunConfig, leaf_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC = REPO_ROOT / "docs" / "contracts" / "run_config_schema.md"

#: A backticked token that looks like a dotted config key: it starts at a real top-level
#: section name, so prose words and Python identifiers cannot be mistaken for one.
_KEY_RE = re.compile(r"`([a-z_]+(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
#: A citation SHAPED like a config key: an all-lowercase root and snake_case tails.
_KEY_SHAPED_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")

#: The dotted roots this doc may cite that are NOT `RunConfig` sections. A stale key and a
#: module path are structurally identical (`train.gone_away` vs `torch.dtype`), so the
#: legitimate non-config roots are DECLARED and any other unknown root is stale.
_NON_CONFIG_ROOTS: frozenset[str] = frozenset({
    "mantis",  # the package; symbol citations are separately resolved by `_SYMBOL_RE`
    "torch",   # `torch.dtype` in the amp-dtype rows
    "spec",    # a local name in prose about a resolved spec object
    "dict",    # `dict.get` and friends, describing Python behaviour
})
#: A backticked or bare dotted symbol rooted at the package.
_SYMBOL_RE = re.compile(r"(?<![\w.])(mantis(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
#: The doc's own statement of the leaf count, e.g. "**191 leaf key-paths**"; the live side of
#: the comparison is always derived from the schema.
_COUNT_RE = re.compile(r"\*\*(\d+) leaf key-paths\*\*")
#: The doc's header version line, e.g. "- version: v9".
_HEADER_VERSION_RE = re.compile(r"^- version:\s*v(\d+)\s*$", re.M)
#: A version-table row, e.g. "| v13 | one new OPTIONAL leaf … |".
_TABLE_VERSION_RE = re.compile(r"^\|\s*v(\d+)\s*\|", re.M)
#: The heading below which every citation is checked in REVERSE (see the module docstring).
ABSENT_HEADING = "## Deliberately absent"
#: The heading whose table's first two cells are LIVE-claim bare citations (the DSV2-2 arm).
_CROSS_FIELD_HEADING = "## Cross-field rules"
#: Where a region ends. Any later `## ` heading closes it.
_HEADING_RE = re.compile(r"^## ")
#: A backticked bare identifier (no dots) — only consulted inside the claim columns.
_BARE_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def _schema_defined_names() -> set[str]:
    """Every name DEFINED in the schema package, by static AST walk — no imports.

    Static on purpose: building the name set by import would execute the modules, and the
    universe must exist even while the package is broken enough to need checking.
    """
    import ast

    names: set[str] = set()
    for path in (REPO_ROOT / "src" / "mantis" / "config" / "schema").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _symbol_exists(dotted: str) -> bool:
    """True iff `dotted` names an importable module, or an attribute reachable from one.

    Walks the longest importable prefix, then resolves the remainder by `getattr`, so a retired
    module and a retired function on a live module both fail.
    """
    parts = dotted.split(".")
    module = None
    consumed = 0
    for stop in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:stop]))
        except ImportError:
            continue
        consumed = stop
        break
    if module is None:
        return False
    obj: object = module
    for attr in parts[consumed:]:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True


def check(doc_path: Path) -> list[str]:
    """Return one failure line per stale citation; empty list = clean."""
    text = doc_path.read_text(encoding="utf-8")
    failures: list[str] = []

    leaves = leaf_paths(RunConfig)
    sections = set(RunConfig.model_fields)
    # A doc may name an interior BLOCK as well as a leaf, so a cited key passes if it is a leaf
    # or a dotted prefix of one.
    valid = set(leaves)
    for leaf in leaves:
        parts = leaf.split(".")
        for stop in range(1, len(parts)):
            valid.add(".".join(parts[:stop]))

    schema_names = _schema_defined_names()
    in_absent = False
    saw_absent = False
    in_cross_field = False
    saw_cross_field_rows = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _HEADING_RE.match(line):
            in_absent = line.strip() == ABSENT_HEADING
            saw_absent = saw_absent or in_absent
            in_cross_field = line.strip().startswith(_CROSS_FIELD_HEADING)
        # The first two cells of a cross-field table row are live-claim citations — validator
        # name, model name — and must be DEFINED names.
        if in_cross_field and line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            claim_tokens = [
                tok for cell in cells[:2] for tok in _BARE_RE.findall(cell)
            ]
            if claim_tokens:
                saw_cross_field_rows += 1
            for tok in claim_tokens:
                if tok not in schema_names:
                    failures.append(
                        f"{doc_path}:{lineno}: cross-field claim column cites `{tok}`, "
                        f"which is not defined anywhere in mantis.config.schema"
                    )
        for match in _KEY_RE.finditer(line):
            key = match.group(1)
            root = key.split(".")[0]
            if root not in sections:
                # A stale citation and a module path are structurally identical, so shape
                # separates them: `a.b_c` with a snake_case tail is a config-key CITATION, and
                # an unknown root then means the section is gone.
                if _KEY_SHAPED_RE.fullmatch(key) and root not in _NON_CONFIG_ROOTS:
                    failures.append(
                        f"{doc_path}:{lineno}: cites `{key}`, whose root section `{root}` "
                        f"does not exist in RunConfig. Known sections: {sorted(sections)}. "
                        "A renamed or deleted section makes every citation under it read as "
                        "prose, which is exactly the drift this gate exists to catch."
                    )
                continue
            exists = key in valid
            if in_absent and exists:
                failures.append(
                    f"{doc_path}:{lineno}: lists config key `{key}` as deliberately absent, "
                    f"but RunConfig HAS it"
                )
            elif not in_absent and not exists:
                failures.append(
                    f"{doc_path}:{lineno}: cites config key `{key}`, which is not a key path "
                    f"of RunConfig"
                )
        for match in _SYMBOL_RE.finditer(line):
            symbol = match.group(1)
            exists = _symbol_exists(symbol)
            if in_absent and exists:
                failures.append(
                    f"{doc_path}:{lineno}: lists symbol `{symbol}` as deliberately absent, "
                    f"but it resolves"
                )
            elif not in_absent and not exists:
                failures.append(
                    f"{doc_path}:{lineno}: cites symbol `{symbol}`, which does not resolve"
                )

    if not saw_absent:
        failures.append(
            f'{doc_path}: has no "{ABSENT_HEADING}" section. The reversed-citation region is '
            f"part of this gate's reach; removing the heading would silently retire it"
        )
    if saw_cross_field_rows == 0:
        failures.append(
            f'{doc_path}: the "{_CROSS_FIELD_HEADING}" table has no claim-column citations. '
            f"The bare-symbol arm is part of this gate's reach (DSV2-2); a moved or emptied "
            f"table would silently retire it"
        )

    # The version table is the authority — a row lands when a version does — and nothing
    # checked the header line against it while it said v9 over a table ending at v12.
    header = _HEADER_VERSION_RE.search(text)
    rows = [int(m.group(1)) for m in _TABLE_VERSION_RE.finditer(text)]
    if not rows:
        failures.append(
            f"{doc_path}: no version-table rows parsed. This gate's version check would pass "
            "vacuously over a reshaped table, which is the phantom-gate class."
        )
    elif header is None:
        failures.append(
            f"{doc_path}: no `- version: vN` header line, so the stated version cannot be "
            f"checked against the table (whose last row is v{max(rows)})"
        )
    elif int(header.group(1)) != max(rows):
        failures.append(
            f"{doc_path}: header says v{header.group(1)} but the version table's last row is "
            f"v{max(rows)}. The table is the authority — a row lands when a version does."
        )

    stated = _COUNT_RE.search(text)
    if stated is None:
        failures.append(
            f"{doc_path}: does not state its leaf-key-path count; the gate needs a "
            f'"**N leaf key-paths**" claim to check against RunConfig (live count: '
            f"{len(leaves)})"
        )
    elif int(stated.group(1)) != len(leaves):
        failures.append(
            f"{doc_path}: states {stated.group(1)} leaf key-paths; RunConfig has "
            f"{len(leaves)}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args(argv)
    if not args.doc.is_file():
        print(f"contract-doc gate: {args.doc} does not exist")
        return 2
    failures = check(args.doc)
    for line in failures:
        print(line)
    if failures:
        print(f"contract-doc gate: {len(failures)} stale citation(s) in {args.doc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
