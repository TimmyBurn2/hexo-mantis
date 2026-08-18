"""Check 13 (`GatherNotStrictlyIncreasing`) — the gather ORDER invariant, made fail-loud.

NOT part of Suite A (`test_graph_collate_adv.py`), deliberately: that suite is a PARITY port
whose every row asserts the exception class an old-side capture produced, and this check has no
old-side row because it did not exist old-side. It is a NEW producer test for a NEW assertion
(LAW-07), filed under R284's P-MASK design §1.4.

WHY THE CHECK EXISTS. `legal_node_gather` is the CONTRACT ORDER of every per-legal-node
quantity — `policy_dst_slot[i]`, `segment_softmax`'s segment `i`, and the Rust-side
`assemble_ls_from_gnn_probs` all read position `i` as gather position `i`. The boolean-mask
formulation `emb[legal_mask]` instead returns rows in ASCENDING ROW INDEX. The two coincide
exactly while the gather ascends, and silently mispair priors to cells when it does not. That
made the invariant load-bearing for the code as it stood BEFORE P-MASK as well as after — it
was simply never checked. Nothing else in the 18 covers it: check 9 constrains which graph a
row points into, check 11 constrains slot aliasing, neither constrains order.

LAW-07: every corruption row carries its clean twin, following Suite A's stated discipline —
a resolver that rejected everything would otherwise pass this file.
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
    """Every die-loud call site catches `GraphContractError`; a check that raised outside the
    family would escape all of them (the F1 silent-corruption class wearing a new exception)."""
    assert issubclass(GatherNotStrictlyIncreasing, GraphContractError)
    assert issubclass(GatherNotStrictlyIncreasing, ValueError)


def test_a_swapped_adjacent_pair_raises_named(payload_fields) -> None:
    """The MINIMAL corruption: two adjacent gather rows swapped. Same set, same length, same
    graph — only the ORDER moves, which is precisely what every other structural check is blind
    to and what the byte-parity of the P-MASK gather rests on."""
    fields = payload_fields("b6")
    g = fields["legal_node_gather"]
    g[0], g[1] = int(g[1]), int(g[0])
    with pytest.raises(GatherNotStrictlyIncreasing) as err:
        _collate(fields)
    assert "not strictly increasing" in str(err.value)


def test_a_duplicated_row_raises_named(payload_fields) -> None:
    """A repeat is non-strict without being descending — the `<= 0` half of the predicate, which
    a `< 0` check would wave through (R72: every conjunct appears in some flip-set)."""
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
    """LAW-07's other half. Every payload the bank expects to collate CLEAN must still do so —
    if the native builder did not in fact emit ascending gathers, check 13 would break the
    production path, and that is exactly the failure this twin exists to make impossible to
    miss.

    `empty_legal` is deliberately NOT in this list and its absence is not an oversight: it is a
    CORRUPTION fixture, the hand-built 1-stone/0-legal payload whose whole purpose is to raise
    `EmptyLegalSet` (`test_graph_collate_adv.py::test_empty_legal_set`). Listing it as a clean
    twin would assert the opposite of what the bank says it is."""
    batch = _collate(payload_fields(stem))
    g = np.asarray(payload_fields(stem)["legal_node_gather"])
    assert g.size == 0 or bool(np.all(np.diff(g) > 0))
    assert int(batch.legal_offsets[-1]) == g.size


@pytest.mark.parametrize(
    "row,where",
    [(-1, "first"), (-100000, "first"), (10**9, "last")],
    ids=["negative-one", "large-negative", "far-past-N"],
)
def test_a_gather_row_outside_0_N_dies_NAMED_and_not_by_numpy(payload_fields, row, where) -> None:
    """The range hole check 13 exposed and did not itself close (isolated review, finding 2).

    `_check_structural`'s check 9 reads `node_graph[legal_node_gather]` — numpy FANCY INDEXING,
    which for a negative row WRAPS silently (−1 reads the last node of the last graph, and the
    old boolean-mask formulation then gathered that row and placed it LAST: a silent mispairing)
    and for a row >= N raises a bare `IndexError`, which is not a `GraphContractError` and so
    escapes every die-loud catch site in the tree.

    Both are now refused by name, BEFORE the fancy index. Ascending order is preserved by each
    corruption so the row is reached through check 13 rather than short-circuited by it — the
    negative rows go at the front, the huge one at the back."""
    fields = payload_fields("b6")
    g = fields["legal_node_gather"]
    g[0 if where == "first" else -1] = row
    with pytest.raises(GraphContractError) as err:
        _collate(fields)
    assert not isinstance(err.value, GatherNotStrictlyIncreasing), (
        "the corruption must be reached as a RANGE failure, not short-circuited as an order one"
    )
    assert "outside [0," in str(err.value)


def test_the_Lg_le_1_boundary_cannot_raise(payload_fields) -> None:
    """The guard's other conjunct: `Lg > 1`. `b0` carries `Lg == 0` and collates clean, which
    pins the short-circuit at zero.

    `Lg == 1` is NOT asserted here, and the reason is recorded rather than left as a silent gap:
    a valid one-legal-node payload is not constructible from the committed bank by truncation —
    `legal_node_gather`, `legal_offsets`, `policy_dst_slot`, `n_nodes_checksum`, `n_stones` and
    the node rows themselves are coupled by checks 4/6/15, and a hand-built one would be exactly
    the synthesized payload this suite's siblings refuse on principle. What covers it is the
    conjunct itself: `Lg > 1` short-circuits identically at 0 and 1, the 0 arm is measured here,
    and the `<=` half of the comparison is flipped by the duplicate-row row above (R72)."""
    batch = _collate(payload_fields("b0"))
    assert int(batch.legal_offsets[-1]) == 0
