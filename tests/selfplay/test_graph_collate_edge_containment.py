"""Check 8's segmented-reduction formulation — the flip-set for its OWN boundaries (R71/R72).

R284's P-CHECKS work replaced check 8 (`EdgeCrossesGraphBoundary`) with a per-graph min/max over
`edge_offsets` (`np.minimum.reduceat`) instead of a per-EDGE graph id plus two E-long gathers and
two E-long comparisons. Measured at the minted cap (E = 1,942,920): **13.48 ms -> 0.93 ms**.

`test_graph_collate_adv.py::test_adv_3_edge_crosses_graph` is the parity row and still passes —
but it drives ONE corruption (edge 0's src into the next graph), and the new formulation has
failure modes the old one did not: it reasons about SEGMENT EXTREMES rather than per-edge
identity, and it drops empty edge segments to keep `reduceat` off a start index it rejects.

Every conjunct of the shipped predicate gets a flip (R72). There are four —
`min(src) < lo`, `max(src) >= hi`, `min(dst) < lo`, `max(dst) >= hi` — and a corruption that
only trips one of them is exactly what a partial rewrite would miss.
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
    """R72: all four conjuncts. `src`/`dst` × `min < lo` / `max >= hi`.

    The corruption targets a MIDDLE graph, and that is what makes the row bite. Both
    replacement values are then real node indices of NEIGHBOURING graphs — strictly inside the
    global `[0, N)` — so check 7 (`EdgeIndexOutOfBounds`) cannot see either, and a formulation
    that reduced over the whole array instead of per segment would call both clean.

    The first draft aimed at the LAST graph and two of the four rows failed for the wrong
    reason: for the last graph `hi == N`, so `at_or_above_hi` leaves the global range and check
    7 fires first. Recorded rather than silently re-aimed — a flip-set row that fires through a
    DIFFERENT check is not a flip of the conjunct it names."""
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
    """The defect the segmentation exists to catch, isolated. Graph 0's first edge is pointed at
    a node of the LAST graph — a perfectly valid node index, inside `[0, N)`, that belongs to
    another game. Check 7 (`EdgeIndexOutOfBounds`) cannot see it; only per-graph containment
    can, and a formulation that reduced over the whole array instead of per segment would call
    this clean."""
    fields = payload_fields("b6")
    no = np.asarray(fields["node_offsets"])
    fields["edge_index"][0] = int(no[-2])   # first node of the last graph
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields)


def test_a_graph_with_ZERO_edges_does_not_break_the_segmentation(payload_fields) -> None:
    """The dropped-empty-segment path. `reduceat` REJECTS a start index equal to the array
    length, so an empty trailing segment is not merely inefficient — it raises. Empty segments
    are dropped instead, and this row proves the drop preserves the partition rather than
    silently skipping a real graph's edges.

    Built by moving graph 0's edges into graph 1 at the OFFSET level: graph 0 keeps its nodes
    and legal set (so checks 4/5/6/12 stay satisfied) and simply owns no edges. Its former edges
    now belong to graph 1 — which makes them cross-graph — so the corruption must STILL be
    caught, and caught for the containment reason rather than by an index error."""
    fields = payload_fields("b6")
    eo = np.asarray(fields["edge_offsets"]).copy()
    eo[1] = 0                              # graph 0: [0, 0) -> zero edges
    fields["edge_offsets"] = eo
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields)


def test_the_clean_twin_still_collates(payload_fields) -> None:
    """LAW-07's other half — and the row that would catch a formulation which rejects
    everything, which every one of the corruption rows above would happily pass."""
    batch = _collate(payload_fields("b6"))
    assert batch.n_graphs >= 2


def test_a_b1_single_graph_payload_still_collates(payload_fields) -> None:
    """B = 1 is the degenerate segmentation: one segment covering every edge. Pinned because it
    is the case where a per-graph reduction and a whole-array reduction coincide, i.e. the case
    that cannot distinguish a correct implementation from the incorrect one."""
    assert _collate(payload_fields("b1")).n_graphs == 1
