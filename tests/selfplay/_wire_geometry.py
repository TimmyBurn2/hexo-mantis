"""The graph-wire geometry the collate fixtures were captured at, READ OFF THE REGISTRY.

AUDIT-1 F-41. `collate_graph_batch`'s four geometry parameters used to default to the
`gnn_axis_v1` row's values, typed as literals in `graph_collate.py`; every production caller
passed `spec.*`, so the defaults' only consumers were tests that omitted them. The parameters
are required now, which means every test states its geometry — and this module is why "states"
does not mean "retypes". `NODE_FEAT_DIM = 11` and `EDGE_FEAT_DIM = 5` were typed in two suites
whose own subject is that the collate must refuse geometry it was not given.

THE ROW-TO-CAPTURE ASSOCIATION IS PROVEN, NOT DECLARED — see `conftest.py::wire_geometry`,
which divides the captured arrays by the row's dims. A re-capture at a row with different dims
reds that fixture instead of collating under the wrong numbers.
"""
from __future__ import annotations

from typing import Any

from mantis.encoding.registry import all_specs

#: The registry row the captured collate payloads were built at.
COLLATE_FIXTURE_ENCODING = "gnn_axis_v1"


def spec_for(name: str) -> Any:
    """The registry spec by name — the ONE authority for a row's geometry.

    Raises:
        LookupError: no registry row carries that name.
    """
    for spec in all_specs():
        if spec.name == name:
            return spec
    raise LookupError(f"registry has no row {name!r}; rows: {[s.name for s in all_specs()]}")


def geometry_kwargs(name: str = COLLATE_FIXTURE_ENCODING) -> dict[str, int]:
    """The four kwargs `collate_graph_batch` requires, for one registry row."""
    spec = spec_for(name)
    return {
        "trunk_size": spec.trunk_size,
        "win_length": spec.win_length,
        "node_feat_dim": spec.node_feat_dim,
        "edge_feat_dim": spec.edge_feat_dim,
    }


#: Every GRAPH row in the registry, by name — the roster a geometry test parametrises over so a
#: new row (r8 was one) is covered without a per-row edit.
GRAPH_ROWS: tuple[str, ...] = tuple(
    s.name for s in all_specs() if str(getattr(s, "representation", "")) == "graph"
)
