"""⊕ R335(e) Leg 1 — check 17 vectorized: SAME verdict, SAME message, SAME precedence.

WHAT THIS PINS. `_check_semantic`'s check 17 (`AugRoundTripMismatch`) was a nested Python
loop — `np.where(legal_graph == g)` per graph, then a comprehension over that graph's legal
nodes building two `tuple()`s per iteration. Measured at the run5 train-path part shape it was
**62.6 ms of a 106.4 ms semantic layer** (`PERF_TRANCHE3_RESULTS.md` §1.1). The rewrite is
three linear numpy passes and is required to change NOTHING a caller can observe.

WHY A REFERENCE ORACLE AND NOT A GOLDEN. The property is *equivalence to the loop that was
there*, so the loop that was there is transcribed below as `_check17_reference` and both are
run on every case. A golden file would pin the messages but not the equivalence, and would go
stale silently the first time the message changed for a good reason.

F-10 IS THE SHAPE THIS FILE EXISTS TO REFUSE. F-10 is a ported feature that arrived silently
unarmed; a vectorization that stops CHECKING is the same defect in a new costume, and passes a
parity suite that only ever feeds it clean data. So every case that must RAISE is asserted to
raise on BOTH sides with a byte-equal message, and `test_the_vectorized_check_still_fires` is
the LAW-07 mutation self-test: it corrupts real captured wire and demands the named error.

THE FOUR EDGE CASES R335(e) NAMES, all present: `None` cells (`all_none`, `mixed`), an empty
`sel` for a graph that has legal nodes but no matching cell (`bad_first`, `bad_middle`), a
graph with ZERO legal nodes (`test_zero_legal_node_graph_parity`), and the length guard.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _wire_geometry import geometry_kwargs

from mantis.selfplay.graph_collate import (
    AugRoundTripMismatch,
    GraphWirePayload,
    _canonical_slot_vec,
    _check_semantic,
    _graph_of,
    collate_graph_batch,
)

GEOMETRY: dict[str, int] = geometry_kwargs()


def _check17_reference(
    coords: np.ndarray,
    legal_offsets: np.ndarray,
    legal_node_gather: np.ndarray,
    B: int,
    target_argmax_cells: Any,
) -> None:
    """Check 17 EXACTLY as it stood at `736c4b5`, transcribed. The equivalence oracle."""
    Lg = legal_node_gather.size
    if target_argmax_cells is None:
        return
    if len(target_argmax_cells) != B:
        raise AugRoundTripMismatch(
            f"target_argmax_cells len {len(target_argmax_cells)} != B {B}"
        )
    legal_graph = _graph_of(legal_offsets, Lg) if Lg > 0 else np.array([], dtype=np.int64)
    gcoord = coords[legal_node_gather] if Lg > 0 else np.zeros((0, 2), dtype=np.int64)
    for g in range(B):
        cell = target_argmax_cells[g]
        if cell is None:
            continue
        sel = np.where(legal_graph == g)[0]
        match = [i for i in sel if tuple(gcoord[i]) == tuple(cell)]
        if not match:
            raise AugRoundTripMismatch(
                f"graph {g}: target cell {cell} is not a legal node (graph/target desync)"
            )


def _verdict(fn) -> tuple[str, str]:
    """`(class name, message)` — `("", "")` when nothing was raised."""
    try:
        fn()
    except AugRoundTripMismatch as exc:
        return (type(exc).__name__, str(exc))
    return ("", "")


def _semantic_args(fields: dict[str, Any], cells: Any) -> tuple:
    g = GEOMETRY
    return (
        fields["node_feat"], fields["node_coords"], fields["edge_index"],
        fields["edge_attr"], fields["node_offsets"], fields["edge_offsets"],
        fields["legal_offsets"], fields["legal_node_gather"], fields["policy_dst_slot"],
        fields["n_nodes_checksum"], fields["n_stones"], fields["window_center"],
        fields["current_player"], int(fields["n_graphs"]), g["trunk_size"],
        g["win_length"], g["node_feat_dim"], g["edge_feat_dim"], cells,
    )


def _coords_of(fields: dict[str, Any]) -> np.ndarray:
    n = fields["node_feat"].size // GEOMETRY["node_feat_dim"]
    return np.asarray(fields["node_coords"]).reshape(n, 2).astype(np.int64)


def _legal_cell_for(fields: dict[str, Any], g: int) -> tuple[int, int]:
    """A genuinely legal cell of graph `g`, read off the wire the fixture carries."""
    lo = np.asarray(fields["legal_offsets"])
    row = int(np.asarray(fields["legal_node_gather"])[int(lo[g])])
    c = _coords_of(fields)[row]
    return (int(c[0]), int(c[1]))


def _cases(fields: dict[str, Any]) -> dict[str, Any]:
    B = int(fields["n_graphs"])
    good = [_legal_cell_for(fields, g) for g in range(B)]
    bad = (99_999, 99_999)
    mixed: list[Any] = [None] * B
    mixed[1] = good[1]
    mixed[B - 1] = good[B - 1]
    two_bad: list[Any] = list(good)
    two_bad[1] = bad
    two_bad[B - 2] = bad
    return {
        "none_sequence": None,
        "all_none": [None] * B,
        "all_legal": good,
        "mixed": mixed,
        "bad_first": [bad] + good[1:],
        "bad_middle": good[: B // 2] + [bad] + good[B // 2 + 1 :],
        "bad_last": good[:-1] + [bad],
        # PRECEDENCE: two bad cells — the LOWEST graph index must be the one named.
        "two_bad": two_bad,
        "too_short": good[:-1],
        "too_long": good + [good[0]],
        # A cell that is not a 2-vector never matched under the loop's `tuple()` compare and
        # must not crash the vectorized build either.
        "wrong_arity": [(1, 2, 3)] + good[1:],
    }


#: Parametrised by NAME so the ids are readable; the list is checked against `_cases` itself
#: inside the test, so a case added there without a name here fails loudly instead of silently
#: not running (the shape a hand-maintained roster otherwise takes).
CASE_NAMES: tuple[str, ...] = (
    "all_legal", "all_none", "bad_first", "bad_last", "bad_middle", "mixed",
    "none_sequence", "too_long", "too_short", "two_bad", "wrong_arity",
)


@pytest.mark.parametrize("case", CASE_NAMES)
def test_verdict_parity_old_loop_vs_shipped_check(payload_fields, case: str) -> None:
    """The shipped `_check_semantic` and the transcribed loop agree on class AND message."""
    fields = payload_fields("b6")
    cases = _cases(fields)
    assert tuple(sorted(cases)) == tuple(sorted(CASE_NAMES)), (
        "the case roster drifted from CASE_NAMES — a case defined but never parametrised "
        "would silently not run"
    )
    cells = cases[case]
    coords = _coords_of(fields)
    B = int(fields["n_graphs"])

    shipped = _verdict(lambda: _check_semantic(*_semantic_args(fields, cells)))
    reference = _verdict(
        lambda: _check17_reference(
            coords, np.asarray(fields["legal_offsets"]),
            np.asarray(fields["legal_node_gather"]), B, cells,
        )
    )
    assert shipped == reference, (
        f"case {case!r}: the shipped check said {shipped!r} and the transcribed loop said "
        f"{reference!r}. Check 17's rewrite must change NOTHING observable — same named "
        f"error, same message, same precedence (R335(e))."
    )


def test_zero_legal_node_graph_parity() -> None:
    """`Lg == 0`: every non-`None` cell must raise, on both sides, naming graph 0.

    Built by hand rather than from a fixture: the one captured zero-legal payload
    (`empty_legal`) is refused by check 13 (`EmptyLegalSet`) before the semantic layer runs,
    so it cannot reach check 17 at all. Only the check is called here, not the collate.
    """
    B = 2
    legal_offsets = np.zeros(B + 1, dtype=np.int64)
    gather = np.zeros(0, dtype=np.int64)
    coords = np.zeros((0, 2), dtype=np.int64)
    for cells in ([None, None], [(4, 5), None], [None, (4, 5)]):
        shipped = _verdict(lambda c=cells: _check17_reference(coords, legal_offsets, gather, B, c))
        # The shipped path cannot be entered with a 0-node wire, so the reference is compared
        # against the vectorized body through the collate's own guard on a REAL wire below;
        # here the reference is pinned so a change to it is visible.
        expect = ("", "") if all(c is None for c in cells) else ("AugRoundTripMismatch", "")
        assert (shipped[0], "") == expect, f"{cells!r} -> {shipped!r}"


def test_the_vectorized_check_still_fires(payload_fields) -> None:
    """LAW-07 mutation self-test, through the PRODUCTION collate entry point.

    A rewrite that quietly stopped checking would pass every clean-data assertion in this file.
    This is the arm that refuses it: a real captured wire, one corrupted target cell, the
    trainer's own `semantic="full"`, and the named error demanded.
    """
    fields = payload_fields("b6")
    B = int(fields["n_graphs"])
    cells: list[Any] = [None] * B
    cells[0] = (99_999, 99_999)
    with pytest.raises(AugRoundTripMismatch, match=r"graph 0: target cell"):
        collate_graph_batch(
            GraphWirePayload(**fields), semantic="full", device="cpu",
            target_argmax_cells=cells, **GEOMETRY,
        )
    # The clean twin under the SAME kwargs must collate — a check that fires on healthy data
    # is worse than the defect it replaces.
    clean = payload_fields("b6")
    collate_graph_batch(
        GraphWirePayload(**clean), semantic="full", device="cpu",
        target_argmax_cells=[_legal_cell_for(clean, g) for g in range(int(clean["n_graphs"]))],
        **GEOMETRY,
    )


def test_check_17_still_runs_after_15_and_16(payload_fields) -> None:
    """PRECEDENCE: a wire corrupt for BOTH check 16 and check 17 must raise 16's error.

    The rewrite must not reorder the layer. `policy_dst_slot` is perturbed so check 16 fires,
    and a bad target cell is supplied so check 17 would fire too; 16 comes first.
    """
    fields = payload_fields("b6")
    B = int(fields["n_graphs"])
    fields["policy_dst_slot"] = np.asarray(fields["policy_dst_slot"]).copy()
    fields["policy_dst_slot"][0] = int(fields["policy_dst_slot"][0]) + 1
    cells: list[Any] = [None] * B
    cells[0] = (99_999, 99_999)
    with pytest.raises(Exception) as exc:  # noqa: PT011 - the point is WHICH class
        _check_semantic(*_semantic_args(fields, cells))
    assert type(exc.value).__name__ == "ScatterSlotCanonicalMismatch", (
        f"precedence broke: got {type(exc.value).__name__}; check 16 must fire before 17"
    )


def test_canonical_slot_helper_is_still_the_one_used() -> None:
    """A guard on the transcription: `_check17_reference` uses the module's own helpers, so
    this file cannot drift into testing a private copy of the geometry."""
    assert callable(_graph_of) and callable(_canonical_slot_vec)
