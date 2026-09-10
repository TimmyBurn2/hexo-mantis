"""No module keeps its OWN table of arch-kind names.

A private hand copy of `mantis.model.ARCH_KINDS` in `eval/snapshot.py` once omitted a kind, so
that kind was selectable, buildable and trainable but could not survive one eval round. A
round-trip test proves the ONE table it exercises is complete; only a census can see a second
table elsewhere.

An AST walk over every dict literal in `src/mantis/`, judged on its string keys — so a renamed
constant, a reformatted literal or a different variable name cannot slip past it.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.model import ARCH_KINDS

_SRC = Path(__file__).resolve().parents[3] / "src" / "mantis"
#: A key set is judged an arch-kind table at this many known kinds. TWO, not one: a single
#: entry is a legitimate one-arm special case, two names is a module answering "which kinds exist".
_MIN_KEYS_TO_JUDGE = 2


def _dict_key_strings(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _arch_keyed_dict_literals() -> list[tuple[str, int, list[str]]]:
    """Return `(relpath, lineno, keys)` for every `src/mantis/` dict literal keyed by two or more arch kinds."""
    known = set(ARCH_KINDS)
    found: list[tuple[str, int, list[str]]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = _dict_key_strings(node)
            if len(set(keys) & known) >= _MIN_KEYS_TO_JUDGE:
                found.append((str(path.relative_to(_SRC.parents[1])), node.lineno, sorted(keys)))
    return found


def test_every_arch_kind_keyed_dict_in_src_is_the_whole_vocabulary() -> None:
    incomplete = [
        (rel, line, keys) for rel, line, keys in _arch_keyed_dict_literals()
        if set(keys) != set(ARCH_KINDS)
    ]
    assert not incomplete, (
        f"a dict literal keyed by arch-kind names that is NOT the whole vocabulary "
        f"{sorted(ARCH_KINDS)}: {incomplete}. This is F-16's shape — a private table that "
        "silently omits a kind, so the kind is selectable everywhere and unsupported here. "
        "Import `mantis.model.ARCH_KINDS` instead of re-typing its members."
    )


def test_the_census_can_actually_SEE_a_table(tmp_path: Path) -> None:
    """Mutation self-test: a census that reaches no dict literal would pass vacuously forever.

    THE PLANT IS A SUPERSET, not a subset: the vocabulary now has exactly two kinds, so any table
    naming both is complete. The census checks DISAGREEMENT in either direction, so a table naming
    both real kinds plus one this build does not have is the plant that still exercises it.
    """
    planted = tmp_path / "mut.py"
    kinds = sorted(ARCH_KINDS)
    entries = ", ".join(f'"{kind}": 1' for kind in kinds)
    planted.write_text(
        f'_TABLE = {{{entries}, "GnnArchFromTheFuture": 9}}\n', encoding="utf-8",
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    dicts = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert len(dicts) == 1
    keys = _dict_key_strings(dicts[0])
    assert len(set(keys) & set(ARCH_KINDS)) >= _MIN_KEYS_TO_JUDGE, "the finder's own predicate"
    assert set(keys) != set(ARCH_KINDS), (
        "the planted table must DISAGREE with the vocabulary for this self-test to mean "
        "anything"
    )


def test_the_snapshot_module_reads_the_shared_vocabulary_and_keeps_no_copy() -> None:
    """The assertion is IDENTITY, not equality: an alias cannot drift from the vocabulary, a copy can."""
    from mantis.eval import snapshot

    assert snapshot._ARCH_TYPES is ARCH_KINDS, (
        "eval.snapshot must ALIAS the shared vocabulary, not copy it — a copy is what made "
        "GnnArchV2 unsnapshottable while it was selectable, buildable and trainable"
    )
