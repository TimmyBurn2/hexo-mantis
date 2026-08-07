"""ADV named-error suite (O1) — ACHIEVABLE subset + bridge single-graph guards.

WP7 gates the 5 achievable payloads through a bridge-reachable producer, each
with a LAW-07 clean-input self-test:

  ACHIEVABLE from Python (externally injectable):
    * ADV-8  EdgeAttrGeometryMismatch  -> verify_edge_geometry (permuted edge_attr)
             — the guaranteed floor; the headline semantic check.
    * WireAlreadyConsumed  -> GraphWire.take() called twice
    * builder_impl handshake -> GraphWire.builder_impl == native code (1)

  ACHIEVABLE only via a buggy BUILDER (NOT external input) — the live producer is
  the EXISTING mantis-graph Rust `#[should_panic]` test, cited here, LAW-07
  satisfied at the producer layer (crates/mantis-graph/src/lib.rs):
    * ADV-1b NodeCountChecksum            -> verify_contract leaf (lib.rs:767)
    * ADV-2b ScatterSlotAliasing          -> verify_contract_dies_loud_on_slot_aliasing
                                             (lib.rs:1039, #[should_panic])
    * ADV-7  ScatterSlotCanonicalMismatch -> verify_contract leaf (lib.rs:874/885)
    * ADV-9  GatherNotLegalNode           -> verify_contract leaf (lib.rs:853/862)
  These fire on a malformed builder, not on any Python-injectable payload, so they
  are bridge-unreachable-by-external-input and rely on their Rust producers.

DEFERRED to the collate-resolver WP (tracked-not-silent; NO producer in WP7 scope,
mantis-graph/src/lib.rs:754): the 4 batch/wire-context payloads —
  ADV-1a OffsetsNonMonotonic, ADV-2a ScatterGatherCrossesGraph,
  ADV-3  EdgeCrossesGraphBoundary, ADV-4 DtypeMismatch.
Their absence from O1 is not a WP7 FAIL (PREREG §Deferred).
"""
import numpy as np
import pytest

from mantis import _engine

# --- verify_edge_geometry clean fixture (ports graph_contract.rs clean_fixture) --
NODE_FEAT_DIM = 11
EDGE_FEAT_DIM = 5
WIN_LENGTH = 6


def _clean_fixture():
    node_feat = np.zeros(4 * NODE_FEAT_DIM, dtype=np.float32)
    node_feat[0] = 1.0  # stone 0: own=1
    node_feat[NODE_FEAT_DIM + 1] = 1.0  # stone 1: opp=1
    node_coords = np.array([0, 0, 1, 0, 2, 0, 0, 0], dtype=np.int32)
    edge_index = np.array([0, 2], dtype=np.int64)  # src=[0], dst=[2]
    edge_attr = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    edge_attr[0] = 1.0  # axis-0 one-hot
    edge_attr[3] = 2.0  # signed_dist
    edge_attr[4] = 1.0  # src_player = (1-0)*cp(+1)
    node_offsets = np.array([0, 4], dtype=np.int64)
    current_player = np.array([1], dtype=np.int8)
    return node_feat, node_coords, edge_index, edge_attr, node_offsets, current_player


def _call(fix):
    nf, nc, ei, ea, no, cp = fix
    return _engine.verify_edge_geometry(
        nf, nc, ei, ea, no, cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
    )


def test_adv8_clean_input_passes():
    """LAW-07 self-test: a clean edge payload returns without raising."""
    assert _call(_clean_fixture()) is None


def test_adv8_permuted_edge_attr_raises_geometry_mismatch():
    """ADV-8 bites: flipping the signed_dist column (the EdgeAttrGeometryMismatch
    corruption) raises ValueError from verify_edge_geometry."""
    nf, nc, ei, ea, no, cp = _clean_fixture()
    ea = ea.copy()
    ea[3] = -ea[3]  # permute edge geometry
    with pytest.raises(ValueError, match="edge delta"):
        _engine.verify_edge_geometry(
            nf, nc, ei, ea, no, cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
        )


def test_adv8_dirty_onehot_raises():
    nf, nc, ei, ea, no, cp = _clean_fixture()
    ea = ea.copy()
    ea[1] = 1.0  # two axes set -> not a clean one-hot
    with pytest.raises(ValueError, match="one-hot"):
        _engine.verify_edge_geometry(
            nf, nc, ei, ea, no, cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
        )


def test_verify_edge_geometry_hostile_input_raises_not_panics():
    """The never-panic contract: out-of-range endpoints raise ValueError, not a
    process abort / PanicException."""
    nf, nc, _ei, ea, no, cp = _clean_fixture()
    bad_edge = np.array([0, 99], dtype=np.int64)  # dst outside [0, N)
    with pytest.raises(ValueError, match="out of"):
        _engine.verify_edge_geometry(
            nf, nc, bad_edge, ea, no, cp, NODE_FEAT_DIM, EDGE_FEAT_DIM, WIN_LENGTH
        )


# --- bridge single-graph structural guards ------------------------------------
def _one_graph_wire():
    hb = _engine.HexgBuffer(8, "gnn_axis_v1", 128)
    hb.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 1.0)], 1, 100, 2, True, 0.0, True, 1)
    wire, _targets = hb.sample_graph_batch(1)
    return wire


def test_wire_already_consumed_second_take_raises():
    wire = _one_graph_wire()
    first = wire.take()
    assert isinstance(first, dict) and len(first) == 16  # 3 scalars + 13 arrays
    with pytest.raises(_engine.WireAlreadyConsumed):
        wire.take()


def test_wire_getters_repeatable_after_take():
    """The per-array COPY getters stay freely-repeatable (frozen behaviour) even
    after the single-read take() latch fires."""
    wire = _one_graph_wire()
    _ = wire.take()
    a = np.asarray(wire.node_feat)
    b = np.asarray(wire.node_feat)
    assert np.array_equal(a, b) and a.size > 0


def test_builder_impl_native_handshake():
    """The native-builder handshake: builder_impl is the native code (1)."""
    wire = _one_graph_wire()
    assert wire.builder_impl == 1
    assert wire.n_graphs == 1
