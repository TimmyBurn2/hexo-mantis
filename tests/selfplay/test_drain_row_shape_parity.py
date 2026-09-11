"""The game-result row has ONE arity: Rust producer, stub, Python consumer and every fake agree.

DELETE-1 shrank the row 10 -> 8 and only the real engine drain noticed; the fakes kept 10.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RUST_RUNNER = _REPO / "crates" / "mantis-selfplay" / "src" / "runner" / "mod.rs"
_STUB = _REPO / "src" / "mantis" / "_engine.pyi"
_POOL_DRAIN = _REPO / "src" / "mantis" / "selfplay" / "pool_drain.py"
_GOLDEN = _REPO / "tests" / "fixtures" / "selfplay" / "drain" / "drain_goldens.json"
_FAKE_FILES = (
    _REPO / "tests" / "selfplay" / "test_game_complete_delivery.py",
    _REPO / "tests" / "selfplay" / "test_lifecycle_events.py",
)

_ROW_TYPE_RE = re.compile(r"pub type GameResultRow\s*=\s*\((?P<fields>.*?)\);", re.S)


def rust_row_arity(source: str) -> int:
    """Field count of `pub type GameResultRow = (...)` — the producer's own declaration."""
    match = _ROW_TYPE_RE.search(source)
    assert match, "no `pub type GameResultRow = (...)` in the runner source"
    fields = [f.strip() for f in _split_top_level(match.group("fields")) if f.strip()]
    return len(fields)


def _split_top_level(text: str) -> list[str]:
    """Split on commas outside `<...>` / `(...)` so `Vec<(i32, i32)>` stays one field."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def stub_row_arity(source: str) -> int:
    """Element count of the tuple inside `drain_game_results`'s `list[tuple[...]]` return."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "drain_game_results":
            ret = node.returns
            assert isinstance(ret, ast.Subscript), "return annotation is not `list[...]`"
            inner = ret.slice
            assert isinstance(inner, ast.Subscript), "return annotation is not `list[tuple[...]]`"
            elts = inner.slice
            assert isinstance(elts, ast.Tuple), "tuple annotation has no element list"
            return len(elts.elts)
    raise AssertionError("no `drain_game_results` in the stub")


def consumer_unpack_arity(source: str) -> int:
    """Target count of the tuple-unpack assignment whose value is the loop variable `entry`."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                and node.value.id == "entry" and isinstance(node.targets[0], ast.Tuple)):
            return len(node.targets[0].elts)
    raise AssertionError("no `(...) = entry` unpack in pool_drain.py")


def scripted_row_arities(source: str) -> set[int]:
    """Lengths of every tuple literal shaped like a game-result row: `(int, int, [...], ...)`."""
    found: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Tuple) and len(node.elts) >= 3
                and isinstance(node.elts[0], ast.Constant) and isinstance(node.elts[1], ast.Constant)
                and isinstance(node.elts[2], ast.List)):
            found.add(len(node.elts))
    return found


def golden_row_arities(golden: dict) -> set[int]:
    """Lengths of the golden's scripted `games_batch` rows."""
    return {len(row) for row in golden["_constants"]["games_batch"]}


def test_every_side_of_the_drain_seam_agrees_on_the_row_arity() -> None:
    """Producer declaration, stub, consumer unpack, golden rows and every fake: ONE number."""
    producer = rust_row_arity(_RUST_RUNNER.read_text(encoding="utf-8"))
    stub = stub_row_arity(_STUB.read_text(encoding="utf-8"))
    consumer = consumer_unpack_arity(_POOL_DRAIN.read_text(encoding="utf-8"))
    golden = golden_row_arities(json.loads(_GOLDEN.read_text(encoding="utf-8")))
    fakes = {path.name: scripted_row_arities(path.read_text(encoding="utf-8"))
             for path in _FAKE_FILES}

    assert producer == stub == consumer, (
        f"row arity drift across the FFI: Rust GameResultRow={producer}, "
        f"_engine.pyi={stub}, pool_drain unpack={consumer}"
    )
    assert golden == {producer}, f"golden games_batch rows are {golden}, producer is {producer}"
    for name, arities in fakes.items():
        assert arities, f"{name}: no scripted game-result row found — the fake moved or vanished"
        assert arities == {producer}, f"{name}: scripted rows are {arities}, producer is {producer}"


def test_the_checker_bites_on_a_consumer_that_unpacks_the_old_ten() -> None:
    """Mutation self-test: the pre-fix consumer shape is detected as a disagreement."""
    mutated = _POOL_DRAIN.read_text(encoding="utf-8").replace(
        "terminal_reason, mv_min, mv_max, mv_distinct) = entry",
        "terminal_reason, mv_min, mv_max, mv_distinct,\n             seeded, solver_fires) = entry",
    )
    assert consumer_unpack_arity(mutated) == 10
    assert consumer_unpack_arity(mutated) != rust_row_arity(_RUST_RUNNER.read_text(encoding="utf-8"))


def test_the_checker_bites_on_a_producer_that_grows_a_field() -> None:
    """Mutation self-test on the Rust side: an added field is counted, nested generics are not."""
    source = _RUST_RUNNER.read_text(encoding="utf-8")
    base = rust_row_arity(source)
    grown = _ROW_TYPE_RE.sub(
        lambda m: "pub type GameResultRow = (" + m.group("fields") + ", u8);", source, count=1)
    assert rust_row_arity(grown) == base + 1
    assert rust_row_arity("pub type GameResultRow = (usize, Vec<(i32, i32)>, u8);") == 3


@pytest.mark.parametrize("bad", ["", "pub type Other = (u8, u8);"])
def test_a_missing_declaration_refuses_rather_than_reporting_zero(bad: str) -> None:
    """Vacuity guard: no declaration is a refusal, never an arity of 0 that matches nothing."""
    with pytest.raises(AssertionError):
        rust_row_arity(bad)


_BRIDGE_RUNNER = _REPO / "crates" / "mantis-bridge" / "src" / "runner.rs"
_POOL_PUSH = _REPO / "src" / "mantis" / "selfplay" / "pool_push.py"
_ROW_ALIAS_RE = re.compile(r"type GraphRecordRow\s*=\s*\((?P<fields>.*?)\);", re.S)


def graph_row_arity(source: str) -> int:
    """Field count of the bridge's `type GraphRecordRow = (...)` — what `collect_graph_data` hands over."""
    match = _ROW_ALIAS_RE.search(source)
    assert match, "no `type GraphRecordRow = (...)` in the bridge runner source"
    return len([f for f in _split_top_level(match.group("fields")) if f.strip()])


def stub_graph_row_arity(source: str) -> int:
    """Element count of the tuple inside `collect_graph_data`'s `list[tuple[...]]` return."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "collect_graph_data":
            ret = node.returns
            assert isinstance(ret, ast.Subscript) and isinstance(ret.slice, ast.Subscript)
            elts = ret.slice.slice
            assert isinstance(elts, ast.Tuple)
            return len(elts.elts)
    raise AssertionError("no `collect_graph_data` in the stub")


def test_the_graph_row_carries_its_tail_mass_from_the_bridge_to_the_push() -> None:
    """GUMBEL-3's drain dropped the tail mass and the push defaulted it to 0, so the first sparse
    row with a real alpha was refused at insert: bridge alias, stub and push slots must agree."""
    bridge = graph_row_arity(_BRIDGE_RUNNER.read_text(encoding="utf-8"))
    stub = stub_graph_row_arity(_STUB.read_text(encoding="utf-8"))
    assert bridge == stub == 11, f"GraphRecordRow: bridge={bridge} stub={stub}"
    push = _POOL_PUSH.read_text(encoding="utf-8")
    assert "tail_mass = float(rec[-2])" in push and "runner_game_id = int(rec[-1])" in push
    assert "push_graph_position(*rec[:-2]" in push and "tail_mass=tail_mass" in push
    bridge_src = _BRIDGE_RUNNER.read_text(encoding="utf-8")
    body = bridge_src[bridge_src.index("fn collect_graph_data"):]
    assert body.index("r.tail_mass,") < body.index("r.game_id,"), "the tail rides before the id"
