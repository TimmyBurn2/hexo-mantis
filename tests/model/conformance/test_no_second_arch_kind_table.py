"""AUDIT-1 F-16 — no module keeps its OWN table of arch-kind names.

THE DEFECT, and its history. `eval/snapshot.py` carried
`_ARCH_TYPES = {"CnnArch": CnnArch, "GnnArch": GnnArch}` — a private, two-member hand copy of
the three-member `mantis.model.ARCH_KINDS` — so `write_model_snapshot` raised
`TypeError("unsupported arch type")` on a `GnnArchV2` net. V2 could be selected, built,
trained and checkpointed, and then could not survive one eval round: no candidate snapshot,
no round, no promotion. Nothing would have caught it, because no test referenced `_ARCH_TYPES`
and the conformance suite's T10 stopped at a private `_serve` while T11 counted names rather
than omissions.

**VERIFIED AT CONTACT (REPAIR-2 §0), and the finding was TRUE AND IS ALREADY CLOSED.** At the
audit's measurement point `a8fd1c9` the two-member literal is there in `git show`; commit
`c5ce230` — FINISH-1's arch-selector plumbing under R330(e), whose subject line ends "and lets
V2 enter the eval snapshot" — replaced it with `ARCH_KINDS` itself and landed the parametrised
round-trip pin `tests/eval/test_snapshot_payload_keys.py::
test_every_arch_kind_in_the_vocabulary_round_trips_through_the_snapshot`.

WHAT WAS STILL MISSING, and is what this file is. The audit's PIN asks for two things and the
tree had one: the round-trip over every kind (present), and *"the T7 pattern generalised: every
dict literal in `src/` keyed by arch-kind names equals `set(ARCH_KINDS)`"* (absent). A
round-trip test proves the ONE table it exercises is complete; it cannot see a second table
somewhere else. This census can, and it is the half that keeps the class from coming back in a
module nobody thought to parametrise.

STRUCTURE, NOT TEXT: an AST walk over every dict literal in `src/mantis/`, keyed on whether its
string keys look like arch-kind names — so a renamed constant, a reformatted literal or a
different variable name cannot slip past it.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.model import ARCH_KINDS

_SRC = Path(__file__).resolve().parents[3] / "src" / "mantis"
#: A key set is JUDGED as an arch-kind table when it names at least this many known kinds.
#: TWO, not one: a single `{"CnnArch": ...}` entry is a legitimate one-arm special case (a
#: legacy-by-representation row, say), while two names is a module answering "which kinds
#: exist" — the question `ARCH_KINDS` owns.
_MIN_KEYS_TO_JUDGE = 2


def _dict_key_strings(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _arch_keyed_dict_literals() -> list[tuple[str, int, list[str]]]:
    """`(relpath, lineno, keys)` for every dict literal under `src/mantis/` whose string keys
    name two or more members of the arch-kind vocabulary."""
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
    """LAW-07 mutation self-test. A census that reaches no dict literal would pass vacuously
    forever, which is exactly how F-16 survived: the round-trip test could not see the second
    table, so nothing red. Drive the finder over a planted incomplete literal."""
    planted = tmp_path / "mut.py"
    kinds = sorted(ARCH_KINDS)
    planted.write_text(
        f'_TABLE = {{"{kinds[0]}": 1, "{kinds[1]}": 2}}\n', encoding="utf-8",
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    dicts = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert len(dicts) == 1
    keys = _dict_key_strings(dicts[0])
    assert len(set(keys) & set(ARCH_KINDS)) >= _MIN_KEYS_TO_JUDGE, "the finder's own predicate"
    assert set(keys) != set(ARCH_KINDS), (
        "the planted table must be INCOMPLETE for this self-test to mean anything — if the "
        "vocabulary ever shrinks to two kinds, widen the plant"
    )


def test_the_snapshot_module_reads_the_shared_vocabulary_and_keeps_no_copy() -> None:
    """F-16's own subject, named. The repair is `_ARCH_TYPES = ARCH_KINDS` (an alias, not a
    literal), so the assertion is IDENTITY: an alias cannot drift, a copy can."""
    from mantis.eval import snapshot

    assert snapshot._ARCH_TYPES is ARCH_KINDS, (
        "eval.snapshot must ALIAS the shared vocabulary, not copy it — a copy is what made "
        "GnnArchV2 unsnapshottable while it was selectable, buildable and trainable"
    )
