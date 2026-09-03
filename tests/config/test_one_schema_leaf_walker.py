"""THE schema leaf walk has ONE implementation, and both of its modes are driven here.

AUDIT-1 F-44 counted FOUR hand-mirrored copies of this walk. The census at REPAIR-3's landing
found FIVE, and the fifth is the interesting one: the audit's census was scoped to the NAME
`_leaf_paths`, and the fifth copy is called `live_leaf_paths`. Measured against the live schema
at that moment, the five walked to THREE different answers — 191 (gate 13 and the consumer
bijection), 182 (`test_eval_config_remint.py`'s pre-DR-6 copy, which stopped at `Block | None`
— the exact blindness R93 fixed — while its docstring claimed to mirror the others), and 199
(the conformance partition's copy, which descends `list[SubModel]`). Nothing compared them.

TWO ARMS, AND THE SECOND IS THE ONE THAT WOULD HAVE CAUGHT THE FIFTH COPY. The first drives the
walker's predicate on a fixture carrying every shape the schema uses. The second is a STRUCTURAL
census — an AST walk for any function that recurses over `model_fields` — because a name-scoped
search is precisely what missed `live_leaf_paths`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import BaseModel

from mantis.config.schema import RunConfig
from mantis.config.schema.leaves import leaf_paths, nested_block

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "tools", "tests")

#: The one walker's own module, by the path the census reports.
THE_WALKER = "src/mantis/config/schema/leaves.py::leaf_paths"

#: EVERY OTHER walker-shaped function in the tree, each with the grounds that make it not a
#: duplicate leaf walk. The set is asserted for EQUALITY, so it reds two ways: on a new
#: recursive `model_fields` walk, and on a row here whose subject is gone. That two-way ratchet
#: is the whole mechanism — an allowlist that only reds one way becomes a graveyard.
DECLARED_NON_DUPLICATES: dict[str, str] = {
    "tests/config/test_schema.py::walk":
        "the MODEL census — it maps reachable block TYPES to the path they were first reached "
        "by, not key-paths to leaves. Its output type is different and its predicate is the "
        "authority's (`nested_block`, asked both of its questions).",
    "tests/config/test_one_schema_leaf_walker.py::naive":
        "the PLANTED BREAK below. It exists to be different from the authority; a census that "
        "excused it by name would be excusing exactly the shape it hunts.",
}


class _Inner(BaseModel):
    a: int
    b: int


class _Other(BaseModel):
    c: int


class _Fixture(BaseModel):
    required_block: _Inner
    optional_block: _Inner | None
    block_list: list[_Inner]
    block_map: dict[str, _Inner]
    either_block: _Inner | _Other
    scalar: int


def test_the_default_mode_hands_out_only_paths_a_config_can_write():
    """DR-6/R93 descends the OPTIONAL block; NIT-3 keeps every generic container one leaf."""
    assert leaf_paths(_Fixture) == (
        "required_block.a", "required_block.b",
        "optional_block.a", "optional_block.b",
        "block_list",
        "block_map",
        "either_block",
        "scalar",
    ), (
        "an OPTIONAL nested block is descended exactly like a required one — optionality, not "
        "nesting, was the cause of the LAW-08 hole (DR-6). A `list[SubModel]` and a "
        "`dict[str, Block]` stay ONE leaf each: their members are addressed by index or by a "
        "runtime key, so no config writes `block_list.a`. A union naming TWO blocks is a leaf "
        f"because there is no single key-path to hand out; got {leaf_paths(_Fixture)}"
    )


def test_the_container_mode_hands_out_every_field_name_the_schema_reaches():
    """The arch-vocabulary probe's question: what could a future key hide an arch inside?"""
    assert leaf_paths(_Fixture, descend_containers=True) == (
        "required_block.a", "required_block.b",
        "optional_block.a", "optional_block.b",
        "block_list.a", "block_list.b",
        "block_map.a", "block_map.b",
        "either_block",
        "scalar",
    ), (
        "descend_containers walks INTO a container whose args name exactly one block, so a "
        "rung field called `graph_depth` exists for the vocabulary probe to fire on. A union "
        "of two blocks is still a leaf — that rule is about ambiguity, not about containers"
    )


def test_the_two_modes_differ_by_exactly_the_container_expansion():
    """The divergence is an ARGUMENT, and this says what the argument buys, on the live schema."""
    writable = set(leaf_paths(RunConfig))
    reachable = set(leaf_paths(RunConfig, descend_containers=True))
    only_writable = writable - reachable
    only_reachable = reachable - writable
    assert only_writable == {"eval.ladder.rungs", "train.replay_capacity_schedule"}, (
        "the writable walk's extra leaves are exactly the container FIELDS themselves; if this "
        f"set moved, a new container block entered the schema: {sorted(only_writable)}"
    )
    assert all(leaf.split(".")[0:2] in (["eval", "ladder"], ["train", "replay_capacity_schedule"])
               or leaf.startswith(("eval.ladder.rungs.", "train.replay_capacity_schedule."))
               for leaf in only_reachable), sorted(only_reachable)
    assert len(reachable) > len(writable)


def test_a_descend_anything_that_mentions_a_block_implementation_is_refused():
    """PLANTED BREAK. The naive walker — descend any annotation whose args name a block — is
    what `live_leaf_paths` actually was, and it silently produced a different schema."""
    def naive(model: type[BaseModel], prefix: str = "") -> tuple[str, ...]:
        out: list[str] = []
        for name, field in model.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            arms = getattr(field.annotation, "__args__", (field.annotation,))
            block = next((a for a in arms
                          if isinstance(a, type) and issubclass(a, BaseModel)), None)
            out.extend(naive(block, path) if block is not None else [path])
        return tuple(out)

    assert naive(_Fixture) != leaf_paths(_Fixture), (
        "the planted naive walker must NOT agree with the authority — if it does, the default "
        "mode has silently become the container mode and the writable-path guarantee is gone"
    )
    assert "block_list.a" in naive(_Fixture)
    assert "either_block.a" in naive(_Fixture), (
        "the naive form also picks the FIRST arm of an ambiguous union, which is the second "
        "way it differs and the one no count would reveal"
    )


def test_nested_block_refuses_an_ambiguous_union_in_both_modes():
    assert nested_block(_Inner | _Other) is None
    assert nested_block(_Inner | _Other, descend_containers=True) is None
    assert nested_block(_Inner | None) is _Inner
    assert nested_block(list[_Inner]) is None
    assert nested_block(list[_Inner], descend_containers=True) is _Inner


# --------------------------------------------------------------------------------------- #
# The structural census — the arm that would have found the fifth copy
# --------------------------------------------------------------------------------------- #
def _walker_shaped_functions(root: Path) -> list[str]:
    """Every function that iterates `.model_fields` AND recurses into itself.

    Structure, not text: this is what a second copy IS, whatever it is called. The
    name-scoped search the audit ran could not see `live_leaf_paths`, and the transferable
    lesson (REPAIR-2 §6 item 2) is that a census over source TEXT misses the case it is for.
    """
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a file that does not parse is not a walker
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            reads_fields = any(
                isinstance(sub, ast.Attribute) and sub.attr == "model_fields"
                for sub in ast.walk(node)
            )
            recurses = any(
                isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                and sub.func.id == node.name
                for sub in ast.walk(node)
            )
            if reads_fields and recurses:
                rel = path.relative_to(root.parent).as_posix()
                hits.append(f"{rel}::{node.name}")
    return hits


def test_exactly_one_walker_shaped_function_exists_in_the_tree():
    hits: list[str] = []
    for name in SCAN_ROOTS:
        hits.extend(_walker_shaped_functions(REPO_ROOT / name))
    assert sorted(hits) == sorted([THE_WALKER, *DECLARED_NON_DUPLICATES]), (
        "the set of functions that walk the schema by recursing over `model_fields` moved. A "
        "NEW one is a duplicate walker whatever it is named — five of them once walked to "
        "three different answers about the same schema (AUDIT-1 F-44), and the audit's own "
        "census missed the fifth because it searched for a NAME. A row that VANISHED means a "
        f"declared exemption outlived its subject. Found: {sorted(hits)}"
    )


def test_the_census_finds_a_planted_copy_under_any_name(tmp_path: Path):
    """LAW-07 positive control. The census must fire on a copy that shares NO name with the
    authority — the exact case (`live_leaf_paths`) a name-scoped census missed."""
    (tmp_path / "some_module.py").write_text(
        "def totally_unrelated_name(model, prefix=''):\n"
        "    out = []\n"
        "    for name, field in model.model_fields.items():\n"
        "        out.extend(totally_unrelated_name(field.annotation, name))\n"
        "    return out\n",
        encoding="utf-8",
    )
    hits = _walker_shaped_functions(tmp_path)
    assert len(hits) == 1 and hits[0].endswith("::totally_unrelated_name"), hits


def test_the_census_refuses_to_report_clean_on_an_empty_scan(tmp_path: Path):
    """Vacuity control: an empty tree yields zero hits, so the assertion above would be
    satisfiable by scanning nothing. The real test asserts a NON-empty exact set, and this
    records why that shape was chosen."""
    assert _walker_shaped_functions(tmp_path) == []
    with pytest.raises(AssertionError):
        assert _walker_shaped_functions(tmp_path) == [THE_WALKER]
