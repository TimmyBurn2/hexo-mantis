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
from mantis.encoding.registry import lookup

# --- verify_edge_geometry clean fixture (ports graph_contract.rs clean_fixture) --
#: AUDIT-1 F-41: these were `11`, `5` and `6` typed here — the geometry of the row the fixture
#: is built for, restated by hand in a suite whose subject is that the bridge REFUSES wrong
#: geometry. `node_feat_dim`/`edge_feat_dim` come off the registry row; `win_length` comes off
#: the ENGINE, which is where it is owned (`mantis_core::board::WIN_LENGTH`, exported through
#: the bridge by REPAIR-2's F-42) — the registry's `win_length` is checked against it at parse.
_SPEC = lookup("gnn_axis_v1")
NODE_FEAT_DIM = _SPEC.node_feat_dim
EDGE_FEAT_DIM = _SPEC.edge_feat_dim
WIN_LENGTH = _engine.WIN_LENGTH


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


def test_wire_getters_repeatable_until_take():
    """The per-array COPY getters are freely repeatable BEFORE the single-read latch."""
    wire = _one_graph_wire()
    a = np.asarray(wire.node_feat)
    b = np.asarray(wire.node_feat)
    assert np.array_equal(a, b) and a.size > 0


def test_wire_getters_refuse_after_take():
    """PERF-TRANCHE-1 A2 contract change: `take()` MOVES the buffers into numpy, so after
    it there are none left to copy and every getter raises the NAMED error.

    The old contract kept the getters readable after `take()` because `take()` copied.
    Moving is the whole of A2 (ledger §10.1 #4, `wire_copyout` 12.43 ms/pop), and the
    alternative to raising here is a getter that hands back an EMPTY array — a silent zero
    a caller would read as a measurement.
    """
    wire = _one_graph_wire()
    taken = wire.take()
    moved = np.asarray(taken["node_feat"])
    assert moved.size > 0
    for name in ("node_feat", "edge_index", "edge_attr", "legal_offsets", "n_graphs"):
        with pytest.raises(_engine.WireAlreadyConsumed):
            getattr(wire, name)


def test_take_moves_rather_than_copies():
    """The moved array must carry the wire's own bytes — the move is not a truncation.

    Compares the pre-take getter copy against the post-take moved array, on a wire built
    twice from the same deterministic push, so a move that silently produced a fresh empty
    or a differently-ordered buffer cannot pass.
    """
    copied = _one_graph_wire().node_feat
    moved = _one_graph_wire().take()["node_feat"]
    assert np.array_equal(np.asarray(copied), np.asarray(moved))
    # The MECHANISM, not a proxy for it: `from_slice` makes numpy allocate and own the
    # buffer (`base is None`); `into_pyarray` hands numpy Rust's own allocation behind a
    # container base object. A regression to copying would flip both of these.
    assert copied.base is None and copied.flags["OWNDATA"], (
        "the getter still COPIES into a numpy-owned buffer")
    assert moved.base is not None and not moved.flags["OWNDATA"], (
        "take() must hand numpy the Rust allocation, not a copy of it")


def test_builder_impl_native_handshake():
    """The native-builder handshake: builder_impl is the native code (1)."""
    wire = _one_graph_wire()
    assert wire.builder_impl == 1
    assert wire.n_graphs == 1


# ── AUDIT-1 F-22(d): the shape guard must cover every offset the body reads ───────────
#
# `verify_edge_geometry_impl` guarded `node_feat_dim == 0`, then read
# `node_feat[s * node_feat_dim + 1]` — channel 1, the opponent-stone plane — for every node.
# A dim of ONE therefore passed the guard and indexed one past the end of the last node's row.
# This function is the ADV-8 producer and its own docstring promises it "never indexes out of
# range on a corrupt input", so an out-of-bounds read here is the guard failing at its stated
# job. `panic = "unwind"` is what kept it from being process-fatal — a property of the worst
# case, not a design.

@pytest.mark.parametrize("dim", [1, 0], ids=["one-channel", "zero-channel"])
def test_adv8_a_node_feat_dim_the_body_cannot_index_is_REFUSED(dim: int) -> None:
    """THE PIN. `dim == 1` used to pass the guard and then read out of range."""
    n_nodes = 4
    node_feat = np.zeros(max(n_nodes * dim, 1), dtype=np.float32)
    _nf, node_coords, edge_index, edge_attr, node_offsets, current_player = _clean_fixture()
    with pytest.raises(ValueError, match="degenerate dims"):
        _engine.verify_edge_geometry(
            node_feat, node_coords, edge_index, edge_attr, node_offsets, current_player,
            dim, EDGE_FEAT_DIM, WIN_LENGTH,
        )


def test_adv8_the_refusal_says_WHICH_channel_it_needs() -> None:
    """A reader who hits this needs to know the bound is about channel 1, not a round number."""
    node_feat = np.zeros(4, dtype=np.float32)
    _nf, nc, ei, ea, no, cp = _clean_fixture()
    with pytest.raises(ValueError) as excinfo:
        _engine.verify_edge_geometry(node_feat, nc, ei, ea, no, cp, 1, EDGE_FEAT_DIM,
                                     WIN_LENGTH)
    assert "channel 1" in str(excinfo.value), str(excinfo.value)


def test_adv8_the_real_registry_dim_still_passes() -> None:
    """The control: the shipped `node_feat_dim` is 11 and must be unaffected."""
    assert NODE_FEAT_DIM >= 2
    assert _call(_clean_fixture()) is None
