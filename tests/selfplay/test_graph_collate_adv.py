"""⊕ Suite A — the FULL 9-payload ADV suite + boundary/dtype sweeps (WP-SP, A-01 … A-24).

>300 justify: Suite A is ONE contract — the 24 rows of DESIGN §b Suite A gate a single
producer (the ported `_check_structural`/semantic layer of `graph_collate`), and splitting
them across files would break the "full 9-payload suite green in one place" exit condition.

Written oracle-first against the dispatcher's old-side capture (#C1/#C2/#C2b/#C2c,
wp/WPSP/CAPTURE_LOG.md) BEFORE any port code. RED at import until IMPL writes
`mantis.selfplay.graph_collate` — that is the correct pre-port state (PREREG §3).

CAPTURE headline: all four batch-level ADV payloads RAISE old-side, so PREREG's AM-2
hardening clause is DORMANT — this is a PARITY port. Every corruption row therefore
asserts the exception CLASS the old side produced, never a message substring: ADV-1a fires
the *endpoint* arm of `OffsetsNonMonotonic` (`node_offsets[B] != total`), so a test keyed
on a "not non-decreasing" substring would pass for the wrong reason.

LAW-07: every corruption row carries its clean twin — the same payload, uncorrupted, under
the same collate kwargs, asserted NOT to raise. A resolver that rejected everything would
otherwise pass this suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from _retired_batch_fields import RETIRED_BATCH_FIELDS
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

# Module-level golden load: pytest parametrization is evaluated at COLLECTION time, so the
# sweep tables (A-20 24 cells, A-23 13 cells) cannot come from a session fixture.
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"
_EXPECT: dict[str, Any] = json.loads(
    (_FIXTURES / "collate" / "collate_expectations.json").read_text(encoding="utf-8")
)
_BOUNDARY_CELLS = sorted(_EXPECT["boundary_sweep"])
_DTYPE_CELLS = sorted(_EXPECT["dtype_sweep"])

# Named-error lookup: the capture stores CLASS NAMES (strings); the oracle resolves them to
# the real classes so `pytest.raises` binds the exact type, never a message.
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

NODE_FEAT_DIM = 11
EDGE_FEAT_DIM = 5


# ── helpers ───────────────────────────────────────────────────────────────────────────
def _collate(fields: dict[str, Any], **kw: Any):
    return collate_graph_batch(GraphWirePayload(**fields), **kw)


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


# ═══ A-01 / A-02 / A-21 — clean captures collate (the LAW-07 self-tests) ═══════════════
def test_clean_capture_collates_full_semantic(payload_fields, collate_expectations):
    """A-01 — PASS iff the captured B=6 wire collates under semantic='full' with the
    captured shapes/dtypes/legal-mask-sum. FAIL = the resolver rejects or reshapes a
    well-formed wire (the false-positive half of the contract)."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    golden = collate_expectations["collated"]["b6"]

    assert int(batch.n_graphs) == golden["n_graphs"] == 6
    assert str(batch.device) == golden["device"] == "cpu"
    for field, meta in golden["tensors"].items():
        if field in RETIRED_BATCH_FIELDS:
            # RQ-16 / R297(c): the expectations file still records this field and is
            # NOT rewritten. Asserted ABSENT rather than skipped — a silent continue
            # over an unmatched golden key is a check that passes by not checking.
            assert not hasattr(batch, field), f"{field}: retired, yet produced"
            continue
        tensor = getattr(batch, field)
        assert list(tensor.shape) == meta["shape"], f"{field}: shape drift"
        assert str(tensor.dtype) == meta["torch_dtype"], f"{field}: torch dtype drift"

    scalars = collate_expectations["b6_scalars"]
    # Re-expressed against the gather (RQ-16 / R297(c)); the CAPTURED scalar is untouched.
    # `legal_mask.sum()` counted DISTINCT legal nodes, because the mask was a scatter and a
    # repeated gather row would have collapsed into one cell. `unique().numel()` is that same
    # quantity named directly, so the captured 2088 still means what it meant.
    assert int(batch.legal_node_gather.unique().numel()) == scalars["legal_mask_sum"] == scalars["Lg"], (
        "the gather must contain exactly one row per distinct legal node (captured 2088)"
    )


def test_off_window_sentinel_survives_collate(payload_fields, collate_expectations):
    """A-02 — PASS iff the 1800 off-window `-1` sentinels in `policy_dst_slot` survive
    collate unchanged. FAIL = the resolver silently clamps/drops off-window legal moves."""
    expected = collate_expectations["b6_scalars"]["off_window_sentinel_count"]
    fields = payload_fields("b6")
    assert int((fields["policy_dst_slot"] == -1).sum()) == expected == 1800

    _collate(fields, expected_version=1, device="cpu", semantic="full")
    # (was: the same count on `batch.policy_dst_slot`.) That device tensor is retired by RQ-16 /
    # R297(c) — collate carried it and nothing read it. The sentinel still matters, but on the
    # path that actually consumes it: the WIRE array, which the bridge reads as
    # `meta.policy_dst_slot` off the queue. So the claim becomes the one still worth making —
    # collate VALIDATES that array (`_require_dtype`) and must not MUTATE it. The arrays here are
    # the caller's own objects, passed by reference into the payload, so this bites for real.
    assert int((fields["policy_dst_slot"] == -1).sum()) == expected, (
        "collate mutated the caller's policy_dst_slot — the off-window sentinels it is supposed "
        "to validate did not survive the call that validated them"
    )


def test_single_graph_batch_clean(payload_fields, collate_expectations):
    """A-21 — PASS iff the captured B=1 wire collates clean (the degenerate case where
    node_offsets[0]==0 ⇒ local index == global index). FAIL = the single-graph path
    diverged from the multi-graph one."""
    fields = payload_fields("b1")
    assert int(fields["node_offsets"][0]) == 0
    batch = _collate(fields, expected_version=1, device="cpu", semantic="full")
    golden = collate_expectations["collated"]["b1"]
    assert int(batch.n_graphs) == golden["n_graphs"] == 1
    for field, meta in golden["tensors"].items():
        if field in RETIRED_BATCH_FIELDS:
            # RQ-16 / R297(c): the expectations file still records this field and is
            # NOT rewritten. Asserted ABSENT rather than skipped — a silent continue
            # over an unmatched golden key is a check that passes by not checking.
            assert not hasattr(batch, field), f"{field}: retired, yet produced"
            continue
        assert list(getattr(batch, field).shape) == meta["shape"], f"{field}: shape drift"


def test_empty_batch_pinned_to_old(payload_fields, collate_expectations):
    """A-22 — PASS iff B=0 SUCCEEDS under both semantic='full' and semantic='off', with the
    captured empty shapes. The old side does NOT raise on B=0 (capture #C1(iii): np.diff of
    a length-1 offsets array is empty ⇒ EmptyLegalSet cannot fire, and every other check is
    guarded by E>0 / Lg>0). FAIL = the port turned a clean empty batch into an error."""
    golden = collate_expectations["collated"]["b0_full"]
    assert golden["outcome"] == "ok", "capture pins B=0 as SUCCESS — do not weaken this"

    for semantic in ("full", "off"):
        batch = _collate(payload_fields("b0"), expected_version=1, device="cpu",
                         semantic=semantic)
        assert int(batch.n_graphs) == 0, f"semantic={semantic}: n_graphs must be 0"
        for field, meta in golden["tensors"].items():
            if field in RETIRED_BATCH_FIELDS:
                # RQ-16 / R297(c): the expectations file still records this field and is
                # NOT rewritten. Asserted ABSENT rather than skipped — a silent continue
                # over an unmatched golden key is a check that passes by not checking.
                assert not hasattr(batch, field), f"{field}: retired, yet produced"
                continue
            assert list(getattr(batch, field).shape) == meta["shape"], (
                f"semantic={semantic}, {field}: empty-batch shape drift"
            )


# ═══ A-03 / A-04 — the handshake rows ═════════════════════════════════════════════════
def test_contract_version_mismatch(payload_fields):
    """A-03 — PASS iff contract_version=2 against expected_version=1 raises
    GraphContractVersionMismatch. FAIL = a wire built by a different contract revision
    reaches the NN."""
    cls = _expected_class("suite_a_remaining", "A-03_contract_version_mismatch")
    assert cls is GraphContractVersionMismatch

    fields = payload_fields("b6")
    fields["contract_version"] = 2
    with pytest.raises(GraphContractVersionMismatch):
        _collate(fields, expected_version=1, device="cpu")
    _clean_twin_ok(payload_fields, expected_version=1, device="cpu")


def test_non_native_builder_handshake(payload_fields, monkeypatch):
    """A-04 — PASS iff builder_impl=2 raises NonNativeSampleBuilder, AND both escape hatches
    accept it: the `allow_oracle_builder=True` kwarg and the MANTIS_ALLOW_ORACLE_BUILDER=1
    env var (DV-3 renames the old env token; the new tree must never carry the old spelling).
    FAIL = the Python-builder sample-path trap is reachable, or an escape hatch is missing."""
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


# ═══ A-05 / A-07 / A-09 / A-10 — THE FOUR BATCH-LEVEL ADV PAYLOADS (WP exit condition) ══
def test_adv_1a_offsets_non_monotonic(payload_fields):
    """A-05 (ADV-1a) — PASS iff `node_offsets[-1] = N-1` (the drop-last-node recipe) raises
    OffsetsNonMonotonic. Asserted on the CLASS: the old side fires the *endpoint* arm
    (`node_offsets[B] != total`), not the np.diff arm, so a message substring would be a
    false pin. FAIL = a fused batch whose node span disagrees with N reaches the NN."""
    cls = _expected_class("adv_batch_level", "ADV-1a_OffsetsNonMonotonic")
    assert cls is OffsetsNonMonotonic

    fields = payload_fields("b6")
    fields["node_offsets"][-1] = _n_nodes(fields) - 1
    with pytest.raises(OffsetsNonMonotonic):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_2a_gather_crosses_graph(payload_fields):
    """A-07 (ADV-2a) — PASS iff `legal_node_gather[0] = node_offsets[1]` (graph 0's first
    gather pointing into graph 1) raises ScatterGatherCrossesGraph. FAIL = one game's policy
    is gathered from another game's nodes — silent cross-game contamination."""
    cls = _expected_class("adv_batch_level", "ADV-2a_ScatterGatherCrossesGraph")
    assert cls is ScatterGatherCrossesGraph

    fields = payload_fields("b6")
    fields["legal_node_gather"][0] = int(fields["node_offsets"][1])
    with pytest.raises(ScatterGatherCrossesGraph):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_3_edge_crosses_graph(payload_fields):
    """A-09 (ADV-3) — PASS iff an edge endpoint pointing into the next graph raises
    EdgeCrossesGraphBoundary. FAIL = message passing leaks across game boundaries in the
    fused batch."""
    cls = _expected_class("adv_batch_level", "ADV-3_EdgeCrossesGraphBoundary")
    assert cls is EdgeCrossesGraphBoundary

    fields = payload_fields("b6")
    _set_edge(fields, 1, 0, int(fields["node_offsets"][1]))
    with pytest.raises(EdgeCrossesGraphBoundary):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_4_edge_index_wrong_dtype(payload_fields):
    """A-10 (ADV-4) — PASS iff a uint16 `edge_index` raises DtypeMismatch. This is the i64
    wire law: u16 silently WRAPS node indices above 65535, so a narrowed index array is the
    quietest possible corruption. FAIL = the wire law is unenforced."""
    cls = _expected_class("adv_batch_level", "ADV-4_DtypeMismatch")
    assert cls is DtypeMismatch

    fields = payload_fields("b6")
    fields["edge_index"] = fields["edge_index"].astype(np.uint16)
    with pytest.raises(DtypeMismatch):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


# ═══ A-06 / A-08 / A-11 / A-12 / A-13 — the WP7-achievable five, re-fired at collate ════
def test_adv_1b_interior_off_by_one(payload_fields):
    """A-06 (ADV-1b) — PASS iff `node_offsets[1] += 1` raises NodeCountChecksum (the interior
    boundary still leaves the array monotonic, so only the per-graph checksum catches it).
    FAIL = one node migrates between graphs undetected."""
    cls = _expected_class("suite_a_remaining", "A-06_ADV-1b_interior_off_by_one")
    assert cls is NodeCountChecksum

    fields = payload_fields("b6")
    fields["node_offsets"][1] = int(fields["node_offsets"][1]) + 1
    with pytest.raises(NodeCountChecksum):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_adv_2b_slot_aliasing(payload_fields):
    """A-08 (ADV-2b) — PASS iff two in-window legal nodes of graph 0 mapped to the SAME
    policy slot raise ScatterSlotAliasing. FAIL = two moves collide into one logit and the
    policy target is silently wrong."""
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
    """A-11 (ADV-7) — PASS iff a shifted `window_center` raises ScatterSlotCanonicalMismatch
    under semantic='full' (the slot map no longer matches the canonical window slot of the
    rotated coord). FAIL = an unrotated slot map silently mis-scatters the policy."""
    cls = _expected_class("suite_a_remaining", "A-11_ADV-7_slot_map_unrotated")
    assert cls is ScatterSlotCanonicalMismatch

    fields = payload_fields("b6")
    fields["window_center"][0] = fields["window_center"][0] + 1
    with pytest.raises(ScatterSlotCanonicalMismatch):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


def test_adv_8_edge_attr_permuted(payload_fields):
    """A-12 (ADV-8) — PASS iff flipping one `signed_dist` raises EdgeAttrGeometryMismatch
    (re-raised out of the engine's edge-geometry verifier). FAIL = scrambled/misaligned edge
    attributes reach the GNN as if they were geometry."""
    cls = _expected_class("suite_a_remaining", "A-12_ADV-8_edge_attr_permuted")
    assert cls is EdgeAttrGeometryMismatch

    fields = payload_fields("b6")
    fields["edge_attr"][3] = -fields["edge_attr"][3]
    with pytest.raises(EdgeAttrGeometryMismatch):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


def test_adv_9_gather_at_stone_node(payload_fields):
    """A-13 (ADV-9) — PASS iff a gather row pointing at a stone/dummy node raises
    GatherNotLegalNode under semantic='full'. FAIL = the policy is read off an occupied
    cell's node."""
    cls = _expected_class("suite_a_remaining", "A-13_ADV-9_gather_at_stone_node")
    assert cls is GatherNotLegalNode

    fields = payload_fields("b6")
    fields["legal_node_gather"][0] = int(fields["node_offsets"][0])
    with pytest.raises(GatherNotLegalNode):
        _collate(fields, device="cpu", semantic="full")
    _clean_twin_ok(payload_fields, device="cpu", semantic="full")


# ═══ A-14 … A-19 — the remaining named contract errors ════════════════════════════════
def test_empty_legal_set(payload_fields):
    """A-14 — PASS iff the hand-built 1-stone/0-legal single-graph payload raises
    EmptyLegalSet. LAW-07 twin: the B=1 capture (348 legal rows) must still collate clean.
    FAIL = a position with no legal moves is fed forward as a valid training row."""
    cls = _expected_class("suite_a_remaining", "A-14_empty_legal_set")
    assert cls is EmptyLegalSet

    with pytest.raises(EmptyLegalSet):
        _collate(payload_fields("empty_legal"), device="cpu")
    _clean_twin_ok(payload_fields, "b1", device="cpu")


def test_batch_count_mismatch(payload_fields):
    """A-15 — PASS iff dropping one `current_player` entry raises BatchCountMismatch.
    FAIL = a per-graph array shorter than B is broadcast/truncated silently."""
    cls = _expected_class("suite_a_remaining", "A-15_batch_count_mismatch")
    assert cls is BatchCountMismatch

    fields = payload_fields("b6")
    fields["current_player"] = fields["current_player"][:-1].copy()
    with pytest.raises(BatchCountMismatch):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_aug_round_trip_mismatch(payload_fields):
    """A-16 — PASS iff a target argmax cell that is not a legal cell of its graph raises
    AugRoundTripMismatch on the trainer path. FAIL = an augmented target silently fails to
    round-trip and the policy label lands on the wrong cell."""
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
    """A-17 — PASS iff a `policy_dst_slot` outside the window (400) or below the −1 sentinel
    (−2) raises ScatterSlotOutOfBounds. FAIL = an out-of-range slot indexes into a neighbour
    logit (or wraps negative) during scatter."""
    key = "A-17a_scatter_slot_400" if slot == 400 else "A-17b_scatter_slot_minus2"
    assert _expected_class("suite_a_remaining", key) is ScatterSlotOutOfBounds

    fields = payload_fields("b6")
    fields["policy_dst_slot"][0] = slot
    with pytest.raises(ScatterSlotOutOfBounds):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_edge_index_out_of_bounds(payload_fields):
    """A-18 — PASS iff an `edge_index` entry beyond N raises EdgeIndexOutOfBounds.
    FAIL = gather/scatter reads past the node tensor."""
    cls = _expected_class("suite_a_remaining", "A-18_edge_index_out_of_bounds")
    assert cls is EdgeIndexOutOfBounds

    fields = payload_fields("b6")
    _set_edge(fields, 0, 0, _n_nodes(fields) + 5)
    with pytest.raises(EdgeIndexOutOfBounds):
        _collate(fields, device="cpu")
    _clean_twin_ok(payload_fields, device="cpu")


def test_dim_mismatches(payload_fields):
    """A-19 — PASS iff a truncated `node_feat` raises NodeFeatDimMismatch and a truncated
    `edge_attr` raises EdgeAttrDimMismatch. FAIL = a flat buffer that does not divide by the
    declared feature dim is reshaped anyway (the classic silent re-interpretation)."""
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


# ═══ A-20 — the off-by-one boundary sweep (24 captured cells; ≥ the 12 DESIGN asks) ════
@pytest.mark.parametrize("cell", _BOUNDARY_CELLS)
def test_offsets_boundary_sweep(payload_fields, cell):
    """A-20 — PASS iff each ±1 poke at every boundary of all three offset arrays raises the
    class the old side raised for that exact cell. The capture found 24/24 cells raising
    (nothing corrupts silently), so PREREG's AM-2 clause stays dormant here too.
    FAIL = an off-by-one at some boundary now slips through — the red-team lens pre-paid."""
    spec = _EXPECT["boundary_sweep"][cell]
    assert spec["outcome"] == "RAISE", f"{cell}: capture says old side did not raise"
    expected = _ERROR_CLASSES[spec["error_class"]]

    fields = payload_fields("b6")
    array, index, delta = spec["array"], int(spec["index"]), int(spec["delta"])
    fields[array][index] = int(fields[array][index]) + delta
    with pytest.raises(expected):
        _collate(fields, device="cpu")


def test_offsets_boundary_sweep_clean_twin(payload_fields):
    """A-20 (LAW-07 arm) — the unpoked payload must collate under the sweep's kwargs; without
    this the 24 sweep cells would pass against a resolver that rejects every batch."""
    _clean_twin_ok(payload_fields, device="cpu")


# ═══ A-23 — the 13-way dtype sweep ════════════════════════════════════════════════════
@pytest.mark.parametrize("field", _DTYPE_CELLS)
def test_dtype_sweep(payload_fields, field):
    """A-23 — PASS iff every one of the 13 payload arrays given a wrong dtype raises
    DtypeMismatch AND the message NAMES the offending field (the capture proved the old
    message carries `<field> dtype <got> != <want>`; the field name is asserted, the rest of
    the string is not). FAIL = a silently re-interpreted buffer reaches the NN."""
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
    """A-23 (LAW-07 arm) — the untouched payload collates; the sweep is not vacuous."""
    _clean_twin_ok(payload_fields, device="cpu")


# ═══ A-24 — semantic canary cadence ═══════════════════════════════════════════════════
def _canary_trace(payload_fields, semantic: str, period: int, n_calls: int = 8) -> list[bool]:
    """Drive the cadence with the ADV-7 corruption (visible ONLY to the semantic layer) and
    report which call indices raised — the capture harness's own probe, verbatim."""
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
    """A-24 — PASS iff the per-call raise pattern equals the captured cadence: 'off' never
    runs the semantic layer (0/8), 'full' always does (8/8), and 'canary' fires on call 0 and
    every `period`-th call thereafter — i.e. `(n == 0) or (period > 0 and n % period == 0)`.
    FAIL = the expensive semantic layer runs always (a silent throughput regression) or never
    (a silent loss of the ADV-7/8/9 checks in production)."""
    captured = [c["outcome"] == "RAISE" for c in _EXPECT["canary_cadence"][trace_key]]
    observed = _canary_trace(payload_fields, semantic, period, n_calls=len(captured))
    assert observed == captured, (
        f"{trace_key}: canary cadence drifted — captured {captured}, got {observed}"
    )
    reset_semantic_canary()


def test_semantic_canary_cadence_clean_twin(payload_fields):
    """A-24 (LAW-07 arm) — an UNCORRUPTED payload must raise on NO call in any mode; without
    this the cadence table would also be satisfied by a resolver that raises unconditionally."""
    reset_semantic_canary()
    for _ in range(8):
        _collate(payload_fields("b6"), device="cpu", semantic="canary", canary_period=3)
    reset_semantic_canary()
