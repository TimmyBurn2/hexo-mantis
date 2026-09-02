# R8 justify: one drift check over one document, and every arm of it is answered by importing
# `RunConfig` itself — the leaf walk, the key citations, the reversed "deliberately absent"
# region, the symbol resolution and the stated leaf count all read the SAME derived key set.
# An arm living in another file would need its own copy of that derivation, and a transcribed
# key list is exactly how the doc this gate checks rotted through four schema versions.
"""CI gate 13: docs/contracts/run_config_schema.md may not cite a key or a symbol the
shipped schema does not have (WPMINT Phase W, R91's design question).

WHY THIS EXISTS. Contract #5's doc drifted for four schema versions while every gate stayed
green, because nothing in the repo read it — measured and recorded in repo_design §4's own v2
amendment ("no test, no tool, no Makefile target, no CI gate names the contract file"). The
three claims that had gone FALSE by WPMINT Phase W were all of one shape: a config key that no
longer exists (`selfplay.legal_move_radius_schedule`), a resolver symbol that was retired
(`mantis.config.resolve.radius.resolve_radius_from_schedule`), and a count that had moved.
This gate closes exactly that shape.

THE GATE-12 PATTERN, AND WHAT IT FORBIDS. Every check below is answered by importing the LIVE
authority — `RunConfig` for keys, the module tree for symbols — never by consulting a
transcribed copy of either. Nothing here is a list of key names that a future phase would have
to remember to update.

THE ONE COPY THIS FILE DOES CARRY, AND WHY IT CANNOT ROT SILENTLY. `_leaf_paths` is a third
transcription of the walker that `tests/config/test_every_key_has_consumer.py` and its `_p2`
twin already hold twice. That is deliberate and it is self-defending: those two files assert
`len(_leaf_paths(RunConfig)) == 175` against the same schema, and this gate asserts the doc's
stated count against ITS walker. A walker here that diverged from theirs would produce a
different number, disagree with the doc, and red this gate. The alternative — importing the
walker from a test module — is barred outright (no package named `tests`, no sys.path
mutation, R5/LAW-17).

THE "DELIBERATELY ABSENT" SECTION IS CHECKED IN REVERSE, not exempted. That section exists to
name keys and modules the schema does NOT have — a retired radius field, a dead gate knob, six
deleted coordinator fields — so an exemption would have made the doc's most load-bearing list
the one part of it nothing checks. Instead every citation under that heading must FAIL to
resolve. A retired key that quietly comes back reds this gate, which is the direction that
matters: `selfplay.legal_move_radius_schedule` returning is exactly the consumer-less-knob
regression the section was written to prevent.

WHAT THIS GATE DELIBERATELY DOES NOT DO. It does not require the doc to enumerate all 175
leaves, and it does not check prose for truth. It is a citation check: every config key and
every `mantis.*` symbol the doc NAMES must exist (or, under the absent heading, must not). A
claim the doc simply omits is invisible to it. That bound is stated rather than hidden, because
a gate whose real reach is narrower than its name is the class this repo keeps closing.

THE BARE-SYMBOL ARM AND ITS BOUND (WPCLEAN Phase RES, closing the DSV2-2 blind spot). The
WPMINT close-out measured that a doc naming a DELETED validator by bare name left this gate
at rc 0 — `_SYMBOL_RE` sees only `mantis.`-rooted dotted names. The closure is structural,
not doc-wide: in the `## Cross-field rules` table, the first two cells of every data row are
LIVE-claim citations (validator name, model name) and each backticked identifier there must
be a name DEFINED in the schema package (static AST walk over `src/mantis/config/schema` —
no imports, no side effects). Doc-wide bare-name checking is deliberately NOT done: the doc
legitimately cites retired names as history (`min_samples` in the version table), and a
word-list exemption for "retired"-flavored prose is exactly the teach-people-to-word-around-
the-gate failure the armed-abort census warns about. The bound: a stale bare name in PROSE
still passes; one in the claim columns reds.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import typing
from pathlib import Path

from pydantic import BaseModel

from mantis.config.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC = REPO_ROOT / "docs" / "contracts" / "run_config_schema.md"

#: A backticked token that looks like a dotted config key: it starts at a real top-level
#: section name, so prose words and Python identifiers cannot be mistaken for one.
_KEY_RE = re.compile(r"`([a-z_]+(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
#: A citation SHAPED like a config key: an all-lowercase root and snake_case tails.
_KEY_SHAPED_RE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")

#: The dotted roots this doc may cite that are NOT `RunConfig` sections. AUDIT-1 F-26: the
#: gate did `if key.split(".")[0] not in sections: continue`, which is how a dotted module
#: path escapes the key check — and equally how a citation whose SECTION was renamed or
#: deleted escapes it, since a stale key and a module path are structurally identical
#: (`train.gone_away` vs `torch.dtype`). Shape cannot separate them, so the legitimate
#: non-config roots are DECLARED and anything else with an unknown root is stale.
#: Adding a root here is the act of saying "this doc may talk about that namespace", and it
#: is four entries wide precisely so the diff is the review.
_NON_CONFIG_ROOTS: frozenset[str] = frozenset({
    "mantis",  # the package; symbol citations are separately resolved by `_SYMBOL_RE`
    "torch",   # `torch.dtype` in the amp-dtype rows
    "spec",    # a local name in prose about a resolved spec object
    "dict",    # `dict.get` and friends, describing Python behaviour
})
#: A backticked or bare dotted symbol rooted at the package.
_SYMBOL_RE = re.compile(r"(?<![\w.])(mantis(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
#: The doc's own statement of the leaf count, e.g. "**175 leaf key-paths**".
_COUNT_RE = re.compile(r"\*\*(\d+) leaf key-paths\*\*")
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

    The bare-symbol arm's universe. Static on purpose: importing arbitrary modules to
    build a name set would execute them, and the universe must exist even while the
    package is broken enough that the doc's claims are exactly what needs checking.
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


def _nested_block(ann: object) -> type[BaseModel] | None:
    """The single `BaseModel` a field annotation resolves to, or None.

    Mirrors the consumer-registry walker: an OPTIONAL block (`Block | None`) is descended
    into, and a `list[SubModel]` field stays ONE leaf.
    """
    if typing.get_origin(ann) is list:
        return None
    candidates = [
        arg for arg in (typing.get_args(ann) or (ann,))
        if isinstance(arg, type) and issubclass(arg, BaseModel)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _leaf_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        nested = _nested_block(field.annotation)
        if nested is not None:
            out.extend(_leaf_paths(nested, path + "."))
        else:
            out.append(path)
    return out


def _symbol_exists(dotted: str) -> bool:
    """True iff `dotted` names an importable module, or an attribute reachable from one.

    Walks the longest importable prefix, then resolves the remainder by `getattr`. A retired
    module (`mantis.config.resolve.radius`) and a retired function on a live module both fail.
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

    leaves = _leaf_paths(RunConfig)
    sections = set(RunConfig.model_fields)
    # A doc may legitimately name an interior BLOCK (`train.draw_rate_abort`,
    # `monitor.drain`) as well as a leaf, so a cited key passes if it is a leaf or a
    # dotted prefix of one.
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
        # The bare-symbol arm (DSV2-2): the first two cells of a cross-field table row are
        # live-claim citations — validator name, model name — and must be DEFINED names.
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
                # AUDIT-1 F-26. This `continue` is how a dotted module path, a filename or
                # ordinary prose escapes the key check — necessary, and it is also the hole:
                # a key whose SECTION was renamed or removed has an unknown root too, so a
                # STALE citation reads exactly like prose and the check that exists to catch
                # stale citations skips it. The two are separated by shape: `a.b_c` with a
                # snake_case tail and no file extension is a config-key CITATION, and an
                # unknown root then means the section is gone.
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
