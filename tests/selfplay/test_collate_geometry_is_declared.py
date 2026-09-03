"""`collate_graph_batch` has no geometry defaults, and the tests that call it read the registry.

AUDIT-1 F-41. The four geometry parameters (`trunk_size`, `win_length`, `node_feat_dim`,
`edge_feat_dim`) used to default to the `gnn_axis_v1` row's values typed as literals in
`graph_collate.py`, under comments saying "caller passes spec.*". All three PRODUCTION callers
did pass `spec.*` — which is exactly why the defaults were invisible: their only consumers were
test files that omitted the kwargs, and a payload re-captured at another registry row would have
collated under the stale numbers with nothing red.

R1 applied to a function surface: an expectation the caller does not state is not an expectation.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from _wire_geometry import GRAPH_ROWS, geometry_kwargs, spec_for

from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    NodeFeatDimMismatch,
    collate_graph_batch,
)

GEOMETRY_PARAMS = ("trunk_size", "win_length", "node_feat_dim", "edge_feat_dim")


def test_the_four_geometry_parameters_carry_no_default():
    params = inspect.signature(collate_graph_batch).parameters
    defaulted = [name for name in GEOMETRY_PARAMS
                 if params[name].default is not inspect.Parameter.empty]
    assert defaulted == [], (
        f"{defaulted} carry a default again. A default here is a silent EXPECTATION: the value "
        "is what the wire is checked against, so defaulting it means a payload built at another "
        "registry row validates against this one's numbers (AUDIT-1 F-41)"
    )
    assert all(params[name].kind is inspect.Parameter.KEYWORD_ONLY for name in GEOMETRY_PARAMS)


def test_a_wire_whose_node_feat_dim_disagrees_with_the_declared_row_is_refused(payload_fields,
                                                                              wire_geometry):
    """The planted break the audit named: declare 12 where the wire carries 11.

    Under the repair this raises the NAMED contract error. Under the old defaults the caller
    could omit the parameter entirely and silently get 11 — the same green, no matter the row.
    """
    fields = payload_fields("b6")
    with pytest.raises(NodeFeatDimMismatch):
        collate_graph_batch(GraphWirePayload(**fields), expected_version=1,
                            **{**wire_geometry, "node_feat_dim": wire_geometry["node_feat_dim"] + 1})

    collate_graph_batch(GraphWirePayload(**payload_fields("b6")), expected_version=1,
                        **wire_geometry)  # LAW-07 clean twin: same call, true geometry


@pytest.mark.parametrize("row", GRAPH_ROWS)
def test_every_graph_row_hands_out_a_complete_geometry(row: str):
    """Parametrised over the registry's graph rows, so a NEW row is covered with no test edit.

    `gnn_axis_r8` was added and the four suites that type this geometry could not see it — the
    reason F-41 is a class-6 finding and not a typo. The roster comes from `all_specs()`.
    """
    kwargs = geometry_kwargs(row)
    assert set(kwargs) == set(GEOMETRY_PARAMS)
    assert all(isinstance(v, int) and v > 0 for v in kwargs.values()), kwargs
    spec = spec_for(row)
    assert kwargs["node_feat_dim"] == spec.node_feat_dim
    assert kwargs["edge_feat_dim"] == spec.edge_feat_dim


def test_the_graph_row_roster_is_not_empty_and_includes_the_r8_row():
    """Vacuity control: a parametrised test over an empty roster passes for free."""
    assert len(GRAPH_ROWS) >= 2, GRAPH_ROWS
    assert "gnn_axis_r8" in GRAPH_ROWS, (
        "the roster is derived from `all_specs()`; if r8 has left the registry that is a "
        f"registry event, not a test edit. Rows: {GRAPH_ROWS}"
    )


def test_the_capture_association_is_derived_from_the_arrays_not_declared(payload_fields,
                                                                        wire_geometry):
    """`wire_geometry` divides the captured arrays by the row's dims. This drives the same
    arithmetic on a DIFFERENT payload so the fixture's proof is not a single-payload accident."""
    fields = payload_fields("b1")
    n_nodes = np.asarray(fields["node_coords"]).size // 2
    n_edges = np.asarray(fields["edge_index"]).size // 2
    assert np.asarray(fields["node_feat"]).size == n_nodes * wire_geometry["node_feat_dim"]
    assert np.asarray(fields["edge_attr"]).size == n_edges * wire_geometry["edge_feat_dim"]
