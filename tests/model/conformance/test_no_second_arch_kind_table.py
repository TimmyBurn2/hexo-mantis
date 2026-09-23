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

import pytest

from mantis.model import ARCH_KINDS

_SRC = Path(__file__).resolve().parents[3] / "src" / "mantis"
#: A key set is judged an arch-kind table at this many known kinds. TWO, not one: a single
#: entry is a legitimate one-arm special case, two names is a module answering "which kinds exist".
_MIN_KEYS_TO_JUDGE = 2


def _dict_key_strings(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _arch_keyed_dict_literals(root: Path) -> list[tuple[str, int, list[str]]]:
    """Return `(relpath, lineno, keys)` for every dict literal under `root` keyed by two or more arch kinds."""
    known = set(ARCH_KINDS)
    found: list[tuple[str, int, list[str]]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = _dict_key_strings(node)
            if len(set(keys) & known) >= _MIN_KEYS_TO_JUDGE:
                found.append((str(path.relative_to(root.parents[1])), node.lineno, sorted(keys)))
    return found


def _incomplete_tables(root: Path) -> list[tuple[str, int, list[str]]]:
    """The census verdict: every arch-keyed dict literal under `root` whose keys are not the whole vocabulary."""
    return [(rel, line, keys) for rel, line, keys in _arch_keyed_dict_literals(root)
            if set(keys) != set(ARCH_KINDS)]


def test_every_arch_kind_keyed_dict_in_src_is_the_whole_vocabulary() -> None:
    incomplete = _incomplete_tables(_SRC)
    assert not incomplete, (
        f"a dict literal keyed by arch-kind names that is NOT the whole vocabulary "
        f"{sorted(ARCH_KINDS)}: {incomplete}. This is F-16's shape — a private table that "
        "silently omits a kind, so the kind is selectable everywhere and unsupported here. "
        "Import `mantis.model.ARCH_KINDS` instead of re-typing its members."
    )


@pytest.mark.parametrize("shape", ["superset", "subset"])
def test_the_census_can_actually_SEE_a_table(shape: str, tmp_path: Path) -> None:
    """Mutation self-test: the census over a planted module reports a superset and a short table alike."""
    kinds = sorted(ARCH_KINDS)
    names = [*kinds, "GnnArchFromTheFuture"] if shape == "superset" else kinds[:_MIN_KEYS_TO_JUDGE]
    assert set(names) != set(ARCH_KINDS), "premise: the plant must DISAGREE with the vocabulary"
    root = tmp_path / "planted"
    root.mkdir()
    entries = ", ".join(f'"{name}": 1' for name in names)
    (root / "mut.py").write_text(f"_TABLE = {{{entries}}}\n", encoding="utf-8")
    found = _incomplete_tables(root)
    assert [keys for _rel, _line, keys in found] == [sorted(names)], found


def test_the_snapshot_module_reads_the_shared_vocabulary_and_keeps_no_copy() -> None:
    """The assertion is IDENTITY, not equality: an alias cannot drift from the vocabulary, a copy can."""
    from mantis.eval import snapshot

    assert snapshot._ARCH_TYPES is ARCH_KINDS, (
        "eval.snapshot must ALIAS the shared vocabulary, not copy it — a copy is what made "
        "GnnArchV2 unsnapshottable while it was selectable, buildable and trainable"
    )
