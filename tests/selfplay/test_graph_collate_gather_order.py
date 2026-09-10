"""Make the gather ORDER invariant fail loud.

`legal_node_gather` is the CONTRACT ORDER of every per-legal-node quantity, while the boolean-mask
formulation returns rows in ascending row index: the two coincide only while the gather ascends,
and no other structural check constrains order. Every corruption row carries its clean twin.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis.encoding import lookup
from mantis.selfplay.graph_collate import (
    GatherNotStrictlyIncreasing,
    GraphContractError,
    GraphWirePayload,
    collate_graph_batch,
)

_ENC = "gnn_axis_v1"


def _collate(fields: dict, **kw):
    spec = lookup(_ENC)
    return collate_graph_batch(
        GraphWirePayload(**fields), expected_version=1, trunk_size=spec.trunk_size,
        win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
        edge_feat_dim=spec.edge_feat_dim, device="cpu", **kw,
    )


def test_check_13_is_a_named_member_of_the_contract_error_family() -> None:
    """Prove the check is in the contract-error family, which every die-loud call site catches."""
    assert issubclass(GatherNotStrictlyIncreasing, GraphContractError)
    assert issubclass(GatherNotStrictlyIncreasing, ValueError)


def test_a_swapped_adjacent_pair_raises_named(payload_fields) -> None:
    """Prove a swapped adjacent pair raises by name: same set, same length, same graph, so only
    the ORDER moves."""
    fields = payload_fields("b6")
    g = fields["legal_node_gather"]
    g[0], g[1] = int(g[1]), int(g[0])
    with pytest.raises(GatherNotStrictlyIncreasing) as err:
        _collate(fields)
    assert "not strictly increasing" in str(err.value)


def test_a_duplicated_row_raises_named(payload_fields) -> None:
    """Prove a duplicated row raises by name — the non-strict half a `< 0` check waves through."""
    fields = payload_fields("b6")
    fields["legal_node_gather"][1] = int(fields["legal_node_gather"][0])
    with pytest.raises(GatherNotStrictlyIncreasing):
        _collate(fields)


def test_a_reversed_gather_raises_named(payload_fields) -> None:
    fields = payload_fields("b1")
    fields["legal_node_gather"] = np.ascontiguousarray(
        fields["legal_node_gather"][::-1], dtype=np.int64
    )
    with pytest.raises(GraphContractError):
        _collate(fields)


@pytest.mark.parametrize("stem", ["b0", "b1", "b6"])
def test_clean_twin_every_collatable_payload_still_collates(payload_fields, stem) -> None:
    """Prove every payload the bank expects to collate clean still does, so this check cannot
    break the production path. `empty_legal` is absent because it is a corruption fixture."""
    batch = _collate(payload_fields(stem))
    g = np.asarray(payload_fields(stem)["legal_node_gather"])
    assert g.size == 0 or bool(np.all(np.diff(g) > 0))
    assert int(batch.legal_offsets[-1]) == g.size


@pytest.mark.parametrize(
    "row,where",
    [(-1, "first"), (-100000, "first"), (10**9, "last"),
     (-1, "middle"), (10**9, "middle"), (None, "middle-at-N")],
    ids=["negative-one", "large-negative", "far-past-N",
         "negative-in-the-MIDDLE", "far-past-N-in-the-MIDDLE", "exactly-N-in-the-MIDDLE"],
)
def test_a_gather_row_outside_0_N_dies_NAMED_and_not_by_numpy(payload_fields, row, where) -> None:
    """Prove a gather row outside [0, N) dies NAMED, before numpy's fancy index sees it.

    Fancy indexing WRAPS silently on a negative row and raises a bare `IndexError` past N, outside
    the contract-error family. The `middle` rows are the load-bearing ones: an endpoints-only
    flip-set passes a guard that inspects only endpoints, and the order check runs last.
    """
    fields = payload_fields("b6")
    g = fields["legal_node_gather"]
    n_nodes = int(fields["node_offsets"][-1])
    idx = {"first": 0, "last": len(g) - 1, "middle": len(g) // 2, "middle-at-N": len(g) // 2}[where]
    g[idx] = n_nodes if row is None else row       # `None` means "exactly N", the boundary
    with pytest.raises(GraphContractError) as err:
        _collate(fields)
    assert not isinstance(err.value, GatherNotStrictlyIncreasing), (
        "the corruption must be reached as a RANGE failure, not short-circuited as an order one"
    )
    assert "outside [0," in str(err.value)


def test_the_Lg_le_1_boundary_cannot_raise(payload_fields) -> None:
    """Prove the `Lg <= 1` boundary cannot raise, measured at 0; the conjunct short-circuits
    identically at 1, which the committed bank cannot construct by truncation."""
    batch = _collate(payload_fields("b0"))
    assert int(batch.legal_offsets[-1]) == 0
