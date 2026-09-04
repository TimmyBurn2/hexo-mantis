"""T2-1 (R334(e), AUDIT-1 F-51 HOT-04) — three collate checks rewritten, VERDICTS UNCHANGED.

The three are `_graph_of` (a repeat instead of `count` binary searches), checks 7+8 folded into
ONE segmented pass, and check 11's duplicate test (a bounded count instead of a sort). All three
are pure-cost changes: same inputs, same verdicts, same named errors, same precedence.

WHY THIS FILE EXISTS BESIDE THE SUITES THAT ALREADY DRIVE THOSE CHECKS.
`test_graph_collate_adv.py` and `test_graph_collate_edge_containment.py` prove the checks REFUSE
what they must refuse; they were written against the previous formulations and they still pass,
which is necessary and not sufficient. What a rewrite additionally has to show is that it did not
move a verdict on any input — including the ones no existing row constructs: an empty edge
segment, a payload violating BOTH bound checks at once, a slot alias that is also off-window.
Each of those is a case where the old and new forms could plausibly disagree, so each is driven.

THE PRECEDENCE ROW IS THE LOAD-BEARING ONE. Check 7 used to run before check 8, so a payload
that is out of `[0, N)` reported `EdgeIndexOutOfBounds` rather than `EdgeCrossesGraphBoundary`.
The fold computes the segment extrema first and could easily have inverted that; the operator
consequence is real — "this row is in no graph at all" is a different diagnosis from "this row is
in the wrong graph" — so the order is pinned, not left to the implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis.selfplay.graph_collate import (
    EdgeCrossesGraphBoundary,
    EdgeIndexOutOfBounds,
    GraphWirePayload,
    ScatterSlotAliasing,
    _graph_of,
    collate_graph_batch,
)

from _wire_geometry import COLLATE_FIXTURE_ENCODING, geometry_kwargs


def _searchsorted_graph_of(offsets: np.ndarray, count: int) -> np.ndarray:
    """The PREVIOUS formulation, kept here as the oracle the new one is judged against."""
    return np.searchsorted(offsets, np.arange(count), side="right") - 1


@pytest.mark.parametrize(
    "offsets",
    [
        [0, 3],
        [0, 1, 2, 3],
        [0, 0, 3],            # a leading EMPTY segment
        [0, 3, 3],            # a trailing EMPTY segment
        [0, 2, 2, 2, 5],      # two empties in the middle
        [0, 0, 0, 0],         # every segment empty, count 0
        [0, 7],
    ],
)
def test_graph_of_agrees_with_the_searchsorted_form_on_every_csr_shape(offsets) -> None:
    """Including the empty-segment shapes, which are where the two forms could differ: the
    search maps an index to the LAST graph sharing its boundary, and the repeat skips a
    zero-length graph entirely. They are the same answer, and that is asserted rather than
    reasoned about."""
    off = np.array(offsets, dtype=np.int64)
    count = int(off[-1])
    assert np.array_equal(_graph_of(off, count), _searchsorted_graph_of(off, count))


def _wire(**over) -> GraphWirePayload:
    """A two-graph payload that PASSES every check, as the base for each corruption.

    Geometry comes from the registry row (F-41), never from literals here.
    """
    node_feat = np.zeros(6 * 11, dtype=np.float32)
    node_coords = np.zeros(6 * 2, dtype=np.int32)
    edge_index = np.array([0, 1, 3, 4, 1, 0, 4, 3], dtype=np.int64)
    edge_attr = np.zeros(4 * 5, dtype=np.float32)
    edge_attr[0::5] = 1.0
    fields = {
        "contract_version": 1,
        "builder_impl": 1,
        "n_graphs": 2,
        "node_feat": node_feat,
        "node_coords": node_coords,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_offsets": np.array([0, 3, 6], dtype=np.int64),
        "edge_offsets": np.array([0, 2, 4], dtype=np.int64),
        "legal_offsets": np.array([0, 2, 4], dtype=np.int64),
        "legal_node_gather": np.array([1, 2, 4, 5], dtype=np.int64),
        "policy_dst_slot": np.array([10, 11, 12, 13], dtype=np.int32),
        "n_nodes_checksum": np.array([3, 3], dtype=np.uint32),
        "n_stones": np.array([1, 1], dtype=np.uint16),
        "window_center": np.zeros(4, dtype=np.int32),
        "current_player": np.array([1, -1], dtype=np.int8),
    }
    fields.update(over)
    return GraphWirePayload(**fields)


def _collate(wire: GraphWirePayload):
    return collate_graph_batch(
        wire, expected_version=1, device="cpu", semantic="off",
        **geometry_kwargs(COLLATE_FIXTURE_ENCODING),
    )


def test_the_clean_payload_still_collates() -> None:
    """The control. Without it every refusal row below is satisfied by a collate that refuses
    everything."""
    batch = _collate(_wire())
    assert batch.n_graphs == 2
    assert batch.edge_index.shape == (2, 4)


def test_a_row_outside_the_global_range_still_raises_EdgeIndexOutOfBounds() -> None:
    ei = np.array([0, 1, 3, 99, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_a_NEGATIVE_row_still_raises_EdgeIndexOutOfBounds() -> None:
    """The other side of the global bound, and the one a `reduceat` maximum cannot see."""
    ei = np.array([0, 1, 3, -1, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_a_row_in_range_but_in_the_WRONG_graph_still_raises_EdgeCrossesGraphBoundary() -> None:
    ei = np.array([0, 1, 3, 0, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(_wire(edge_index=ei))


def test_THE_PRECEDENCE_a_payload_violating_BOTH_reports_the_GLOBAL_error_first() -> None:
    """Check 7 ran before check 8, so the out-of-range diagnosis won. The fold computes the
    segment extrema first and must NOT invert that: "in no graph at all" is the stronger
    statement and the one an operator needs before "in the wrong graph"."""
    ei = np.array([0, 1, 3, 99, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_an_EMPTY_edge_segment_is_still_dropped_and_still_checked() -> None:
    """Graph 0 owns no edges. The segment partition must still cover every edge, so graph 1's
    cross-boundary row is still caught."""
    clean = _wire(edge_offsets=np.array([0, 0, 4], dtype=np.int64),
                  edge_index=np.array([3, 4, 3, 5, 4, 3, 5, 3], dtype=np.int64))
    assert _collate(clean).n_graphs == 2
    bad = _wire(edge_offsets=np.array([0, 0, 4], dtype=np.int64),
                edge_index=np.array([3, 4, 3, 0, 4, 3, 5, 3], dtype=np.int64))
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(bad)


def test_a_slot_alias_inside_ONE_graph_still_raises_ScatterSlotAliasing() -> None:
    slots = np.array([10, 10, 12, 13], dtype=np.int32)
    with pytest.raises(ScatterSlotAliasing):
        _collate(_wire(policy_dst_slot=slots))


def test_the_SAME_slot_in_DIFFERENT_graphs_is_still_legal() -> None:
    """The half a bincount over a flat key space could silently break: the key must keep the
    graph id, or two graphs reusing slot 10 would read as an alias."""
    slots = np.array([10, 11, 10, 13], dtype=np.int32)
    assert _collate(_wire(policy_dst_slot=slots)).n_graphs == 2


def test_repeated_OFF_WINDOW_slots_are_still_exempt_from_the_alias_check() -> None:
    """`-1` is the off-window sentinel and many legal nodes carry it; counting those as
    duplicates would refuse every wide position. The old form dropped them before `unique`
    and the new one must drop them before the count."""
    slots = np.array([-1, -1, -1, -1], dtype=np.int32)
    assert _collate(_wire(policy_dst_slot=slots)).n_graphs == 2


def test_the_alias_check_reaches_the_TOP_of_the_slot_range() -> None:
    """`361` is the largest legal slot and the key is `graph * 400 + slot`, so the count's
    length must cover `B * 400` — a `minlength` short by one graph would index out of range or,
    worse, silently miss an alias in the last graph."""
    slots = np.array([10, 11, 361, 361], dtype=np.int32)
    with pytest.raises(ScatterSlotAliasing):
        _collate(_wire(policy_dst_slot=slots))
