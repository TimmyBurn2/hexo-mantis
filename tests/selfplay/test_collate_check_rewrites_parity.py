"""Three collate checks rewritten, verdicts unchanged.

`_graph_of` uses a repeat instead of `count` binary searches, checks 7+8 fold into one
segmented pass, and check 11's duplicate test is a bounded count instead of a sort. All three
are pure-cost changes: same inputs, same verdicts, same named errors, same precedence.

The existing collate suites prove the checks refuse what they must, but were written against
the previous formulations. What a rewrite must additionally show is that no verdict moved on
inputs no existing row constructs: an empty edge segment, a payload violating both bound
checks at once, a slot alias that is also off-window.

The precedence row is load-bearing: check 7 ran before check 8, so a payload outside `[0, N)`
reports `EdgeIndexOutOfBounds`, not `EdgeCrossesGraphBoundary`. "In no graph at all" is a
different diagnosis from "in the wrong graph", so the order is pinned.
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
    """The previous formulation, kept as the oracle the new one is judged against."""
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
    """Prove `_graph_of` agrees with the searchsorted form on every CSR shape.

    The empty-segment shapes are where the two could differ: the search maps an index to the
    last graph sharing its boundary, and the repeat skips a zero-length graph entirely.
    """
    off = np.array(offsets, dtype=np.int64)
    count = int(off[-1])
    assert np.array_equal(_graph_of(off, count), _searchsorted_graph_of(off, count))


def _wire(**over) -> GraphWirePayload:
    """Build a two-graph payload that passes every check, as the base for each corruption.

    Geometry comes from the registry row, never from literals here.
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
    """Control: a clean payload still collates, so the refusal rows below are not vacuous."""
    batch = _collate(_wire())
    assert batch.n_graphs == 2
    assert batch.edge_index.shape == (2, 4)


def test_a_row_outside_the_global_range_still_raises_EdgeIndexOutOfBounds() -> None:
    ei = np.array([0, 1, 3, 99, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_a_NEGATIVE_row_still_raises_EdgeIndexOutOfBounds() -> None:
    """Prove a negative row still raises: the side of the bound a `reduceat` maximum cannot see."""
    ei = np.array([0, 1, 3, -1, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_a_row_in_range_but_in_the_WRONG_graph_still_raises_EdgeCrossesGraphBoundary() -> None:
    ei = np.array([0, 1, 3, 0, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(_wire(edge_index=ei))


def test_THE_PRECEDENCE_a_payload_violating_BOTH_reports_the_GLOBAL_error_first() -> None:
    """Prove a payload violating both checks reports the global error first, as before the fold."""
    ei = np.array([0, 1, 3, 99, 1, 0, 4, 3], dtype=np.int64)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(_wire(edge_index=ei))


def test_an_EMPTY_edge_segment_is_still_dropped_and_still_checked() -> None:
    """Prove an empty edge segment still leaves every edge covered by the partition."""
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
    """Prove the same slot in different graphs is legal: the count key must keep the graph id."""
    slots = np.array([10, 11, 10, 13], dtype=np.int32)
    assert _collate(_wire(policy_dst_slot=slots)).n_graphs == 2


def test_repeated_OFF_WINDOW_slots_are_still_exempt_from_the_alias_check() -> None:
    """Prove repeated off-window `-1` slots are exempt: counting them would refuse every wide position."""
    slots = np.array([-1, -1, -1, -1], dtype=np.int32)
    assert _collate(_wire(policy_dst_slot=slots)).n_graphs == 2


def test_the_alias_check_reaches_the_TOP_of_the_slot_range() -> None:
    """Prove the alias check reaches the top slot: the key is `graph * 400 + slot`, so the
    count must cover `B * 400` or it misses an alias in the last graph."""
    slots = np.array([10, 11, 361, 361], dtype=np.int32)
    with pytest.raises(ScatterSlotAliasing):
        _collate(_wire(policy_dst_slot=slots))
