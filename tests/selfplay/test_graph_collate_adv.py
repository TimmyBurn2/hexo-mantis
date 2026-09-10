"""The full 9-payload ADV suite plus the boundary and dtype sweeps.

>300 justify (R8): ONE contract — the rows gate a single producer, the structural and semantic
layers of `graph_collate`, and splitting them would break the "full suite green in one place"
exit condition.

All four batch-level ADV payloads RAISE old-side, so this is a PARITY port: every corruption row
asserts the exception CLASS the old side produced, never a message substring — one row fires the
*endpoint* arm of `OffsetsNonMonotonic`, so a substring pin would pass for the wrong reason.

LAW-07: every corruption row carries its clean twin, the same payload uncorrupted under the same
kwargs, asserted NOT to raise. A resolver that rejected everything would otherwise pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from _retired_batch_fields import RETIRED_BATCH_FIELDS
from _wire_geometry import geometry_kwargs
import pytest

from mantis.selfplay.graph_collate import (
    AugRoundTripMismatch,
    BatchCountMismatch,
    DtypeMismatch,
    EdgeAttrDimMismatch,
    EdgeAttrGeometryMismatch,
    EdgeCrossesGraphBoundary,
    EdgeIndexOutOfBounds,
    EmptyLegalSet,
    GatherNotLegalNode,
    GraphContractError,
    GraphContractVersionMismatch,
    GraphWirePayload,
    NodeCountChecksum,
    NodeFeatDimMismatch,
    NonNativeSampleBuilder,
    OffsetsNonMonotonic,
    ScatterGatherCrossesGraph,
    ScatterSlotAliasing,
    ScatterSlotCanonicalMismatch,
    ScatterSlotOutOfBounds,
    collate_graph_batch,
    reset_semantic_canary,
)

# Module-level golden load: pytest parametrization is evaluated at COLLECTION time, so the sweep
# tables cannot come from a session fixture.
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"
_EXPECT: dict[str, Any] = json.loads(
    (_FIXTURES / "collate" / "collate_expectations.json").read_text(encoding="utf-8")
)
_BOUNDARY_CELLS = sorted(_EXPECT["boundary_sweep"])
_DTYPE_CELLS = sorted(_EXPECT["dtype_sweep"])

# The capture stores CLASS NAMES; resolving them here is what lets `pytest.raises` bind the exact
# type rather than a message.
_ERROR_CLASSES = {
    cls.__name__: cls
    for cls in (
        AugRoundTripMismatch, BatchCountMismatch, DtypeMismatch, EdgeAttrDimMismatch,
        EdgeAttrGeometryMismatch, EdgeCrossesGraphBoundary, EdgeIndexOutOfBounds,
        EmptyLegalSet, GatherNotLegalNode, GraphContractVersionMismatch, NodeCountChecksum,
        NodeFeatDimMismatch, NonNativeSampleBuilder, OffsetsNonMonotonic,
        ScatterGatherCrossesGraph, ScatterSlotAliasing, ScatterSlotCanonicalMismatch,
        ScatterSlotOutOfBounds,
    )
}

#: The capture's geometry, READ OFF THE REGISTRY ROW it was built at. These were typed here as
#: literals, under a suite whose subject is that the collate refuses geometry it was not given —
#: while the suite itself asserted geometry nobody derived.
GEOMETRY: dict[str, int] = geometry_kwargs()
NODE_FEAT_DIM = GEOMETRY["node_feat_dim"]
EDGE_FEAT_DIM = GEOMETRY["edge_feat_dim"]


def _collate(fields: dict[str, Any], **kw: Any):
    """The geometry is stated on every call, never defaulted — a caller that omitted it used to
    get one encoding's numbers silently."""
    return collate_graph_batch(GraphWirePayload(**fields), **{**GEOMETRY, **kw})


def _clean_twin_ok(payload_fields, name: str = "b6", **kw: Any) -> None:
    """LAW-07 arm: the uncorrupted payload must collate under the SAME kwargs."""
    _collate(payload_fields(name), **kw)


def _expected_class(table: str, key: str):
    entry = _EXPECT[table][key]
    assert entry["outcome"] == "RAISE", f"capture says {key} did not raise old-side: {entry}"
    return _ERROR_CLASSES[entry["error_class"]]


def _n_nodes(fields: dict[str, Any]) -> int:
    return int(fields["node_feat"].size // NODE_FEAT_DIM)


def _n_edges(fields: dict[str, Any]) -> int:
    return int(fields["edge_attr"].size // EDGE_FEAT_DIM)


def _set_edge(fields: dict[str, Any], row: int, col: int, value: int) -> None:
    """`edge_index` travels FLAT on the wire; the old recipes reshape (2, E) to poke it."""
    e = _n_edges(fields)
    ei = fields["edge_index"].reshape(2, e).copy()
    ei[row, col] = value
    fields["edge_index"] = ei.reshape(-1)


def test_clean_capture_collates_full_semantic(payload_fields, collate_expectations):
    """A well-formed captured wire collates with its captured shapes, dtypes and legal-mask sum."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    golden = collate_expectations["collated"]["b6"]

    assert int(batch.n_graphs) == golden["n_graphs"] == 6
    assert str(batch.device) == golden["device"] == "cpu"
    for field, meta in golden["tensors"].items():
        if field in RETIRED_BATCH_FIELDS:
            # The expectations file still records this field and is NOT rewritten. Asserted
            # ABSENT rather than skipped — a silent continue over an unmatched golden key is a
            # check that passes by not checking.
            assert not hasattr(batch, field), f"{field}: retired, yet produced"
            continue
        tensor = getattr(batch, field)
        assert list(tensor.shape) == meta["shape"], f"{field}: shape drift"
        assert str(tensor.dtype) == meta["torch_dtype"], f"{field}: torch dtype drift"

    scalars = collate_expectations["b6_scalars"]
    # Re-expressed against the gather; the CAPTURED scalar is untouched. `legal_mask.sum()` counted
    # DISTINCT legal nodes, because the mask was a scatter and a repeated gather row would have
    # collapsed into one cell. `unique().numel()` is that same quantity named directly.
    assert int(batch.legal_node_gather.unique().numel()) == scalars["legal_mask_sum"] == scalars["Lg"], (
        "the gather must contain exactly one row per distinct legal node (captured 2088)"
    )


def test_off_window_sentinel_survives_collate(payload_fields, collate_expectations):
    """The off-window `-1` sentinels survive collate — nothing clamps or drops them."""
    expected = collate_expectations["b6_scalars"]["off_window_sentinel_count"]
    fields = payload_fields("b6")
    assert int((fields["policy_dst_slot"] == -1).sum()) == expected == 1800

    _collate(fields, expected_version=1, device="cpu", semantic="full")
    # The device tensor this once counted is retired — collate carried it and nothing read it — so
    # the claim moves to the path that consumes the sentinel: the WIRE array the bridge reads off
    # the queue. Collate VALIDATES that array and must not MUTATE it, and the arrays here are the
    # caller's own objects passed by reference, so this bites for real.
    assert int((fields["policy_dst_slot"] == -1).sum()) == expected, (
        "collate mutated the caller's policy_dst_slot — the off-window sentinels it is supposed "
        "to validate did not survive the call that validated them"
    )


def test_single_graph_batch_clean(payload_fields, collate_expectations):
    """The degenerate B=1 case, where local index == global index, collates clean."""
    fields = payload_fields("b1")
    assert int(fields["node_offsets"][0]) == 0
    batch = _collate(fields, expected_version=1, device="cpu", semantic="full")
    golden = collate_expectations["collated"]["b1"]
    assert int(batch.n_graphs) == golden["n_graphs"] == 1
    for field, meta in golden["tensors"].items():
        if field in RETIRED_BATCH_FIELDS:
            # The expectations file still records this field and is NOT rewritten. Asserted ABSENT
            # rather than skipped — a silent continue is a check that passes by not checking.
            assert not hasattr(batch, field), f"{field}: retired, yet produced"
            continue
        assert list(getattr(batch, field).shape) == meta["shape"], f"{field}: shape drift"


def test_empty_batch_pinned_to_old(payload_fields, collate_expectations):
    """B=0 SUCCEEDS with the captured empty shapes: every check is guarded by a positive count."""
    golden = collate_expectations["collated"]["b0_full"]
    assert golden["outcome"] == "ok", "capture pins B=0 as SUCCESS — do not weaken this"

    for semantic in ("full", "off"):
        batch = _collate(payload_fields("b0"), expected_version=1, device="cpu",
                         semantic=semantic)
        assert int(batch.n_graphs) == 0, f"semantic={semantic}: n_graphs must be 0"
        for field, meta in golden["tensors"].items():
            if field in RETIRED_BATCH_FIELDS:
                # The expectations file still records this field and is NOT rewritten. Asserted
                # ABSENT rather than skipped — a silent continue passes by not checking.
                assert not hasattr(batch, field), f"{field}: retired, yet produced"
                continue
            assert list(getattr(batch, field).shape) == meta["shape"], (
                f"semantic={semantic}, {field}: empty-batch shape drift"
            )


def test_contract_version_mismatch(payload_fields):
    """A wire built by a different contract revision raises rather than reaching the NN."""
    cls = _expected_class("suite_a_remaining", "A-03_contract_version_mismatch")
    assert cls is GraphContractVersionMismatch

    fields = payload_fields("b6")
    fields["contract_version"] = 2
    with pytest.raises(GraphContractVersionMismatch):
        _collate(fields, expected_version=1, device="cpu")
    _clean_twin_ok(payload_fields, expected_version=1, device="cpu")


def test_non_native_builder_handshake(payload_fields, monkeypatch):
    """A non-native builder raises, and BOTH escape hatches — the kwarg and the env var — accept it."""
    cls = _expected_class("suite_a_remaining", "A-04_non_native_builder")
    assert cls is NonNativeSampleBuilder

    fields = payload_fields("b6")
    fields["builder_impl"] = 2
    with pytest.raises(NonNativeSampleBuilder):
        _collate(fields, device="cpu")

    # arm 2: explicit kwarg — captured NO_RAISE
    fields = payload_fields("b6")
    fields["builder_impl"] = 2
    _collate(fields, device="cpu", allow_oracle_builder=True, semantic="off")

    # arm 3: env flag — captured NO_RAISE (old token renamed per DV-3)
    monkeypatch.setenv("MANTIS_ALLOW_ORACLE_BUILDER", "1")
    fields = payload_fields("b6")
    fields["builder_impl"] = 2
    _collate(fields, device="cpu", semantic="off")

    monkeypatch.delenv("MANTIS_ALLOW_ORACLE_BUILDER")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_1a_offsets_non_monotonic(payload_fields):
    """A node span disagreeing with N raises, asserted on the CLASS: the old side fires the
    *endpoint* arm, so a message substring would pin the wrong thing."""
    cls = _expected_class("adv_batch_level", "ADV-1a_OffsetsNonMonotonic")
    assert cls is OffsetsNonMonotonic

    fields = payload_fields("b6")
    fields["node_offsets"][-1] = _n_nodes(fields) - 1
    with pytest.raises(OffsetsNonMonotonic):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_2a_gather_crosses_graph(payload_fields):
    """A gather row pointing into the next graph raises, never gathering across games."""
    cls = _expected_class("adv_batch_level", "ADV-2a_ScatterGatherCrossesGraph")
    assert cls is ScatterGatherCrossesGraph

    fields = payload_fields("b6")
    fields["legal_node_gather"][0] = int(fields["node_offsets"][1])
    with pytest.raises(ScatterGatherCrossesGraph):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_3_edge_crosses_graph(payload_fields):
    """An edge endpoint pointing into the next graph raises, so message passing cannot leak."""
    cls = _expected_class("adv_batch_level", "ADV-3_EdgeCrossesGraphBoundary")
    assert cls is EdgeCrossesGraphBoundary

    fields = payload_fields("b6")
    _set_edge(fields, 1, 0, int(fields["node_offsets"][1]))
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_4_edge_index_wrong_dtype(payload_fields):
    """A uint16 `edge_index` raises: u16 silently WRAPS node indices above 65535, which is the
    quietest possible corruption."""
    cls = _expected_class("adv_batch_level", "ADV-4_DtypeMismatch")
    assert cls is DtypeMismatch

    fields = payload_fields("b6")
    fields["edge_index"] = fields["edge_index"].astype(np.uint16)
    with pytest.raises(DtypeMismatch):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_1b_interior_off_by_one(payload_fields):
    """An interior off-by-one leaves the array monotonic, so only the per-graph checksum sees it."""
    cls = _expected_class("suite_a_remaining", "A-06_ADV-1b_interior_off_by_one")
    assert cls is NodeCountChecksum

    fields = payload_fields("b6")
    fields["node_offsets"][1] = int(fields["node_offsets"][1]) + 1
    with pytest.raises(NodeCountChecksum):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_2b_slot_aliasing(payload_fields):
    """Two legal nodes mapped to the SAME policy slot raise, never colliding into one logit."""
    cls = _expected_class("suite_a_remaining", "A-08_ADV-2b_slot_aliasing")
    assert cls is ScatterSlotAliasing

    fields = payload_fields("b6")
    g0_end = int(fields["legal_offsets"][1])
    in_window = [i for i in range(g0_end) if fields["policy_dst_slot"][i] != -1]
    assert len(in_window) >= 2, "capture recipe needs ≥2 in-window slots in graph 0"
    fields["policy_dst_slot"][in_window[1]] = fields["policy_dst_slot"][in_window[0]]
    with pytest.raises(ScatterSlotAliasing):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_7_slot_map_unrotated(payload_fields):
    """A shifted `window_center` raises: an unrotated slot map mis-scatters the policy."""
    cls = _expected_class("suite_a_remaining", "A-11_ADV-7_slot_map_unrotated")
    assert cls is ScatterSlotCanonicalMismatch

    fields = payload_fields("b6")
    fields["window_center"][0] = fields["window_center"][0] + 1
    with pytest.raises(ScatterSlotCanonicalMismatch):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


def test_adv_8_edge_attr_permuted(payload_fields):
    """A flipped `signed_dist` raises out of the engine's own edge-geometry verifier."""
    cls = _expected_class("suite_a_remaining", "A-12_ADV-8_edge_attr_permuted")
    assert cls is EdgeAttrGeometryMismatch

    fields = payload_fields("b6")
    fields["edge_attr"][3] = -fields["edge_attr"][3]
    with pytest.raises(EdgeAttrGeometryMismatch):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


def test_adv_9_gather_at_stone_node(payload_fields):
    """A gather row pointing at a stone or dummy node raises, never reading an occupied cell."""
    cls = _expected_class("suite_a_remaining", "A-13_ADV-9_gather_at_stone_node")
    assert cls is GatherNotLegalNode

    fields = payload_fields("b6")
    fields["legal_node_gather"][0] = int(fields["node_offsets"][0])
    with pytest.raises(GatherNotLegalNode):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


def test_empty_legal_set(payload_fields):
    """A position with no legal moves raises rather than being fed forward as a training row."""
    cls = _expected_class("suite_a_remaining", "A-14_empty_legal_set")
    assert cls is EmptyLegalSet

    with pytest.raises(EmptyLegalSet):
        _collate(payload_fields("empty_legal"), device="cpu")
    _clean_twin_ok(payload_fields, "b1", device="cpu")


def test_batch_count_mismatch(payload_fields):
    """A per-graph array shorter than B raises, never broadcast or truncated."""
    cls = _expected_class("suite_a_remaining", "A-15_batch_count_mismatch")
    assert cls is BatchCountMismatch

    fields = payload_fields("b6")
    fields["current_player"] = fields["current_player"][:-1].copy()
    with pytest.raises(BatchCountMismatch):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_aug_round_trip_mismatch(payload_fields):
    """A target argmax cell that is not legal in its graph raises on the trainer path."""
    cls = _expected_class("suite_a_remaining", "A-16_aug_round_trip_mismatch")
    assert cls is AugRoundTripMismatch

    fields = payload_fields("b6")
    targets: list[tuple[int, int] | None] = [None] * int(fields["n_graphs"])
    targets[0] = (99999, 99999)
    with pytest.raises(AugRoundTripMismatch):
        _collate(fields, device="cpu", semantic="full", target_argmax_cells=targets)

    # LAW-07 twin: all-None targets on the SAME trainer path must collate clean.
    clean = payload_fields("b6")
    _collate(clean, device="cpu", semantic="full",
             target_argmax_cells=[None] * int(clean["n_graphs"]))


@pytest.mark.parametrize("slot", [400, -2], ids=["above_window", "below_sentinel"])
def test_scatter_slot_out_of_bounds(payload_fields, slot):
    """A slot outside the window or below the −1 sentinel raises, never indexing a neighbour."""
    key = "A-17a_scatter_slot_400" if slot == 400 else "A-17b_scatter_slot_minus2"
    assert _expected_class("suite_a_remaining", key) is ScatterSlotOutOfBounds

    fields = payload_fields("b6")
    fields["policy_dst_slot"][0] = slot
    with pytest.raises(ScatterSlotOutOfBounds):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_edge_index_out_of_bounds(payload_fields):
    """An `edge_index` entry beyond N raises rather than reading past the node tensor."""
    cls = _expected_class("suite_a_remaining", "A-18_edge_index_out_of_bounds")
    assert cls is EdgeIndexOutOfBounds

    fields = payload_fields("b6")
    _set_edge(fields, 0, 0, _n_nodes(fields) + 5)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_dim_mismatches(payload_fields):
    """A flat buffer that does not divide by its declared feature dim raises, never reshaped."""
    assert _expected_class("suite_a_remaining",
                           "A-19a_node_feat_dim_mismatch") is NodeFeatDimMismatch
    assert _expected_class("suite_a_remaining",
                           "A-19b_edge_attr_dim_mismatch") is EdgeAttrDimMismatch

    fields = payload_fields("b6")
    fields["node_feat"] = fields["node_feat"][:-1].copy()
    with pytest.raises(NodeFeatDimMismatch):
        _collate(fields, device="cpu")

    fields = payload_fields("b6")
    fields["edge_attr"] = fields["edge_attr"][:-1].copy()
    with pytest.raises(EdgeAttrDimMismatch):
        _collate(fields, device="cpu")

    _clean_twin_ok(payload_fields, device="cpu")


@pytest.mark.parametrize("cell", _BOUNDARY_CELLS)
def test_offsets_boundary_sweep(payload_fields, cell):
    """Each ±1 poke at every offset boundary raises the class the old side raised for that cell."""
    spec = _EXPECT["boundary_sweep"][cell]
    assert spec["outcome"] == "RAISE", f"{cell}: capture says old side did not raise"
    expected = _ERROR_CLASSES[spec["error_class"]]

    fields = payload_fields("b6")
    array, index, delta = spec["array"], int(spec["index"]), int(spec["delta"])
    fields[array][index] = int(fields[array][index]) + delta
    with pytest.raises(expected):
        _collate(fields, device="cpu")


def test_offsets_boundary_sweep_clean_twin(payload_fields):
    """LAW-07 arm: the unpoked payload collates, so the 24 cells are not measured against a
    resolver that rejects every batch."""
    _clean_twin_ok(payload_fields, device="cpu")


@pytest.mark.parametrize("field", _DTYPE_CELLS)
def test_dtype_sweep(payload_fields, field):
    """Every one of the 13 arrays given a wrong dtype raises AND the message NAMES the field."""
    spec = _EXPECT["dtype_sweep"][field]
    assert spec["error_class"] == "DtypeMismatch"

    fields = payload_fields("b6")
    fields[field] = fields[field].astype(np.dtype(spec["wrong_dtype"]))
    with pytest.raises(DtypeMismatch) as excinfo:
        _collate(fields, device="cpu")
    assert field in str(excinfo.value), (
        f"DtypeMismatch must name the offending field {field!r} — an unnamed dtype error "
        "makes a 13-array wire undebuggable"
    )


def test_dtype_sweep_clean_twin(payload_fields):
    """LAW-07 arm: the untouched payload collates, so the sweep is not vacuous."""
    _clean_twin_ok(payload_fields, device="cpu")


def _canary_trace(payload_fields, semantic: str, period: int, n_calls: int = 8) -> list[bool]:
    """Drive the cadence with the corruption visible ONLY to the semantic layer, reporting which
    call indices raised."""
    reset_semantic_canary()
    raised: list[bool] = []
    for _ in range(n_calls):
        fields = payload_fields("b6")
        fields["window_center"][0] = fields["window_center"][0] + 1
        try:
            _collate(fields, device="cpu", semantic=semantic, canary_period=period)
        except GraphContractError:
            raised.append(True)
        else:
            raised.append(False)
    return raised


@pytest.mark.parametrize(
    "trace_key,semantic,period",
    [
        ("semantic_off", "off", 64),
        ("semantic_full", "full", 64),
        ("semantic_canary_period_3", "canary", 3),
        ("semantic_canary_period_64", "canary", 64),
    ],
)
def test_semantic_canary_cadence(payload_fields, trace_key, semantic, period):
    """The per-call raise pattern equals the captured cadence: 'off' never runs the semantic layer,
    'full' always does, 'canary' fires on call 0 and every period-th call."""
    captured = [c["outcome"] == "RAISE" for c in _EXPECT["canary_cadence"][trace_key]]
    observed = _canary_trace(payload_fields, semantic, period, n_calls=len(captured))
    assert observed == captured, (
        f"{trace_key}: canary cadence drifted — captured {captured}, got {observed}"
    )
    reset_semantic_canary()


def test_semantic_canary_cadence_clean_twin(payload_fields):
    """LAW-07 arm: an UNCORRUPTED payload raises on NO call in any mode."""
    reset_semantic_canary()
    for _ in range(8):
        _collate(payload_fields("b6"), device="cpu", semantic="canary", canary_period=3)
    reset_semantic_canary()
