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
`len(_leaf_paths(RunConfig)) == 170` against the same schema, and this gate asserts the doc's
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

WHAT THIS GATE DELIBERATELY DOES NOT DO. It does not require the doc to enumerate all 170
leaves, and it does not check prose for truth. It is a citation check: every config key and
every `mantis.*` symbol the doc NAMES must exist (or, under the absent heading, must not). A
claim the doc simply omits is invisible to it. That bound is stated rather than hidden, because
a gate whose real reach is narrower than its name is the class this repo keeps closing.
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
#: A backticked or bare dotted symbol rooted at the package.
_SYMBOL_RE = re.compile(r"(?<![\w.])(mantis(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
#: The doc's own statement of the leaf count, e.g. "**170 leaf key-paths**".
_COUNT_RE = re.compile(r"\*\*(\d+) leaf key-paths\*\*")
#: The heading below which every citation is checked in REVERSE (see the module docstring).
ABSENT_HEADING = "## Deliberately absent"
#: Where the reversed region ends. Any later `## ` heading closes it.
_HEADING_RE = re.compile(r"^## ")


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

    in_absent = False
    saw_absent = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _HEADING_RE.match(line):
            in_absent = line.strip() == ABSENT_HEADING
            saw_absent = saw_absent or in_absent
        for match in _KEY_RE.finditer(line):
            key = match.group(1)
            if key.split(".")[0] not in sections:
                continue  # not a config key at all (a dotted module, a filename, prose)
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
