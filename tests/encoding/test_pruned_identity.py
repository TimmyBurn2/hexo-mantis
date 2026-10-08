"""`gnn_axis_r8_pruned` is `gnn_axis_r8` with no edge between two empty cells, and nothing else, through the spec surface Python sees."""
from __future__ import annotations

import mantis.encoding as encoding


def test_the_pruned_row_differs_from_r8_in_exactly_its_edge_set() -> None:
    """DERIVED over every field the spec surface exposes, never a typed list of fields that should match."""
    a, b = encoding.lookup("gnn_axis_r8"), encoding.lookup("gnn_axis_r8_pruned")
    fields = [f for f in dir(a) if not f.startswith("_") and not callable(getattr(a, f))]
    assert "empty_edges" in fields, "the spec surface does not expose the edge set"
    differing = {f for f in fields if getattr(a, f) != getattr(b, f)}
    assert differing == {"name", "notes", "empty_edges"}, sorted(differing)
    assert (a.empty_edges, b.empty_edges) == ("kept", "pruned")
