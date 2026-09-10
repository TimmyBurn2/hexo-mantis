"""Check 8's segmented-reduction formulation — the flip-set for its OWN boundaries.

The check is a per-graph min/max over `edge_offsets` (`np.minimum.reduceat`) rather than a
per-EDGE graph id plus two E-long gathers: 13.48 ms -> 0.93 ms at the minted cap. It reasons
about SEGMENT EXTREMES and drops empty edge segments, so each conjunct gets its own flip.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis.encoding import lookup
from mantis.selfplay.graph_collate import (
    EdgeCrossesGraphBoundary,
    GraphWirePayload,
    collate_graph_batch,
)

_ENC = "gnn_axis_v1"


def _collate(fields: dict):
    spec = lookup(_ENC)
    return collate_graph_batch(
        GraphWirePayload(**fields), expected_version=1, trunk_size=spec.trunk_size,
        win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
        edge_feat_dim=spec.edge_feat_dim, device="cpu",
    )


def _shape(fields):
    eo = np.asarray(fields["edge_offsets"])
    return int(eo[-1]), eo


@pytest.mark.parametrize("endpoint", ["src", "dst"], ids=["src", "dst"])
@pytest.mark.parametrize("direction", ["below_lo", "at_or_above_hi"])
def test_each_conjunct_of_the_containment_predicate_flips(
    payload_fields, endpoint, direction
) -> None:
    """All four conjuncts: `src`/`dst` × `min < lo` / `max >= hi`, on a MIDDLE graph so both
    replacement values stay inside the global `[0, N)` where check 7 cannot see them. Aimed at
    the LAST graph, two rows fire through check 7 instead, which is not a flip of their
    conjunct."""
    fields = payload_fields("b6")
    E, eo = _shape(fields)
    no = np.asarray(fields["node_offsets"])
    assert len(eo) - 1 >= 3, "this row needs a graph with neighbours on both sides"
    g = 1                                 # a MIDDLE graph: 0 < lo, hi < N
    e = int(eo[g])                        # its first edge
    idx = e if endpoint == "src" else E + e
    fields["edge_index"][idx] = int(no[g]) - 1 if direction == "below_lo" else int(no[g + 1])
    assert 0 <= int(fields["edge_index"][idx]) < int(no[-1]), (
        "the corruption must stay inside the GLOBAL range or check 7 catches it instead"
    )
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields)


def test_an_edge_inside_the_GLOBAL_range_but_outside_ITS_graph_is_caught(payload_fields) -> None:
    """The defect the segmentation exists to catch, isolated: graph 0's first edge points at a
    node of the LAST graph — a valid index inside `[0, N)` belonging to another game, which only
    per-graph containment can see."""
    fields = payload_fields("b6")
    no = np.asarray(fields["node_offsets"])
    fields["edge_index"][0] = int(no[-2])   # first node of the last graph
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields)


def test_a_graph_with_ZERO_edges_still_makes_its_stolen_edges_cross(payload_fields) -> None:
    """Moving graph 0's edges into graph 1 at the OFFSET level: it keeps its nodes and legal set
    and owns no edges, so its former edges must still be caught for the containment reason. It
    does NOT guard the empty-segment drop — its LEADING empty segment is one `reduceat` accepts,
    and both mutants of that handling also raise here."""
    fields = payload_fields("b6")
    eo = np.asarray(fields["edge_offsets"]).copy()
    eo[1] = 0                              # graph 0: [0, 0) -> zero edges
    fields["edge_offsets"] = eo
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields)


@pytest.mark.parametrize("position", ["leading", "middle", "trailing"])
def test_a_CLEAN_payload_carrying_an_empty_edge_segment_still_collates(
    payload_fields, position
) -> None:
    """THE row that guards the empty-segment drop, and the only one that does.

    Two mutants are caught by nothing else — removing the `nonempty` filter, and truncating
    `seg_lo`/`seg_hi` positionally instead of by the same mask — because both are silent on every
    corrupted payload and fire on a CLEAN one. Only the TRAILING position reaches the mode where
    `reduceat` rejects a start index equal to the array length.
    """
    fields = payload_fields("b6")
    eo = np.asarray(fields["edge_offsets"]).copy()
    ei = np.asarray(fields["edge_index"])
    no = np.asarray(fields["node_offsets"])
    E = int(eo[-1])
    B = len(eo) - 1
    assert B >= 3, "this row needs three graphs to place an empty segment in the middle"
    # Give EVERY edge to one graph, with every endpoint inside that graph's node range, so the
    # payload stays legal and the other graphs own zero edges.
    g = {"leading": B - 1, "middle": 0, "trailing": 0}[position]
    if position == "middle":
        g = 1
    eo[:] = 0
    eo[g + 1:] = E
    lo, hi = int(no[g]), int(no[g + 1])
    ei2 = ei.reshape(2, E).copy()
    ei2[:] = lo + (ei2 % max(hi - lo, 1))
    fields["edge_offsets"] = eo
    fields["edge_index"] = ei2.reshape(-1)
    _collate(fields)   # must NOT raise


def test_the_clean_twin_still_collates(payload_fields) -> None:
    """The other half of the producer test: it catches a formulation that rejects everything,
    which every corruption row above would happily pass."""
    batch = _collate(payload_fields("b6"))
    assert batch.n_graphs >= 2


def test_a_b1_single_graph_payload_still_collates(payload_fields) -> None:
    """B = 1 is the degenerate segmentation, where a per-graph and a whole-array reduction
    coincide — the case that cannot distinguish a correct implementation from the incorrect one."""
    assert _collate(payload_fields("b1")).n_graphs == 1
