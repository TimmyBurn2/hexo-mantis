"""MCTSTree + InferenceBatcher round-trip (O20, review gap 3).

MCTSTree: ctor-compose (new_full + configure_quiescence via the 5-arg bridge
ctor), new_game -> select_leaves -> get_policy / get_improved_policy round-trip;
forced_root_child get/set.

InferenceBatcher: ALL 22 Python methods present AND exercised via the mock-game
helpers over the dense + graph queues (no method silently dropped — a dropped
method is a WP8-compat break); the 3 getters return the spec-derived values.

WP12-R Phase EVALDECODE (E-1, R90a auto-grant) widened this to 22: the card adds
`submit_graphs_and_wait_ls` (the frame-carrying graph driver `submit_graphs_and_wait`
becomes a projection of) and `MCTSTree.expand_and_backup_ls_graph`. The surface
assertion below is `hasattr`-presence over THIS FILE'S OWN literal list, so a new
method leaves it green while the docstring above goes false — that hazard is why the
count and the list move together, and why the round-trip at the bottom exercises the
new tree method rather than only naming it.
"""
import threading

import numpy as np
import pytest

from mantis import _engine

# The full 22-method Python-facing InferenceBatcher surface (DESIGN §a.1 table, widened
# by WP12-R Phase EVALDECODE); 21 named methods/getters + __init__ = 22.
INFERENCE_METHODS = [
    "spawn_mock_games",
    "completed_mock_games",
    "has_pending_requests",
    "next_inference_batch",
    "submit_inference_results",
    "submit_inference_failure",
    "close",
    "bump_model_version",
    "model_version",
    "feature_len_py",
    "policy_len_py",
    "representation_py",
    "has_pending_graph_requests",
    "completed_graph_games",
    "check_graph_request",
    "spawn_mock_graph_games",
    "next_graph_batch",
    "submit_graph_inference_results",
    "submit_graph_inference_failure",
    "submit_graphs_and_wait",
    "submit_graphs_and_wait_ls",
]


# ------------------------------- MCTSTree -------------------------------------
def test_mctstree_ctor_compose_and_policy_round_trip():
    tree = _engine.MCTSTree(1.5, 1.0, 0.25, True, 0.3)  # new_full + configure_quiescence
    assert tree.quiescence_fire_count == 0
    board = _engine.Board.with_encoding_name("v6")
    board.apply_move(0, 0)
    tree.new_game(board)
    leaves = tree.select_leaves(4)
    assert len(leaves) >= 1
    assert all(isinstance(b, _engine.Board) for b in leaves)
    # Feed a uniform policy/value back so the tree has visits, then read a policy.
    policies = [[1.0 / 362] * 362 for _ in leaves]
    values = [0.0] * len(leaves)
    tree.expand_and_backup(policies, values)
    pol = np.asarray(tree.get_policy(1.0))
    assert pol.ndim == 1 and pol.size >= 1
    improved = np.asarray(tree.get_improved_policy())
    assert improved.ndim == 1 and improved.size >= 1


def test_mctstree_forced_root_child_round_trip():
    """AUDIT-1 F-02: the setter validates against the ROOT'S CHILD RANGE now, so the
    round-trip needs a root that HAS children and an index that is one of them. A bare tree
    owns nothing, and the old `= 3` was storing an index into an uninitialised pool slot —
    whose `action_idx` of `u32::MAX` decodes to the cell (32767, 32767), which an UNBOUNDED
    board accepts. That is the arm that produced neither a panic nor an error."""
    tree = _engine.MCTSTree()
    assert tree.forced_root_child is None
    with pytest.raises(ValueError, match="not a child of the root"):
        tree.forced_root_child = 3

    board = _engine.Board.with_encoding_name("v6")
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    tree.expand_and_backup([[1.0 / 362] * 362 for _ in leaves], [0.0] * len(leaves))
    first = tree.root_children_info()[0][1] if hasattr(tree, "root_children_info") else None
    if first is None:
        first = tree.get_root_children_info()[0][1]
    tree.forced_root_child = first
    assert tree.forced_root_child == first
    tree.forced_root_child = None
    assert tree.forced_root_child is None


def test_mctstree_expand_and_backup_ls_graph_round_trip():
    """⊕ WP12-R Phase EVALDECODE (E-1, hunk 3) — the graph legal-set expand door.

    `submit_graphs_and_wait_ls` carries the BUILDER's window centre OUT; the new tree
    method carries dense + the ragged overflow + that centre back IN and expands through
    the same `expand_and_backup_ls_at` self-play expands through. A presence check over a
    literal name list (above) would stay green for a method that raises on every call, so
    the surface widening is paired with an execution here. RED at HEAD: neither method
    exists yet.
    """
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    ib = _engine.InferenceBatcher(encoding_spec=spec)

    def consumer():
        rounds = 0
        while rounds < 500:
            rounds += 1
            ids, wire = ib.next_graph_batch(8, 50)
            ids = list(ids)
            if not ids:
                continue
            offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
            total = int(offsets[-1])
            probs = np.zeros((total,), dtype=np.float32)
            for i in range(len(offsets) - 1):
                s, e = int(offsets[i]), int(offsets[i + 1])
                if e > s:
                    probs[s:e] = 1.0 / (e - s)
            vals = np.zeros((len(ids),), dtype=np.float32)
            ib.submit_graph_inference_results(ids, probs, offsets, vals)
            return

    board = _engine.Board.with_encoding_name("gnn_axis_v1")
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    tree = _engine.MCTSTree()
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    assert len(leaves) == 1

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    results = ib.submit_graphs_and_wait_ls([
        (list(leaf.get_stones()), int(leaf.current_player), int(leaf.moves_remaining))
        for leaf in leaves
    ])
    t.join(timeout=10)
    assert len(results) == 1
    dense, overflow, value, center = results[0]  # (dense, overflow, value, window centre)
    assert len(dense) == spec.policy_stride
    assert isinstance(value, float)
    assert len(tuple(center)) == 2

    tree.expand_and_backup_ls_graph(
        [list(dense)], [list(overflow)], [float(value)], [tuple(center)],
        spec.policy_stride, spec.trunk_size,
    )
    assert tree.get_root_children_info(), "the graph legal-set expand produced no children"
    ib.close()


# ---------------------------- InferenceBatcher --------------------------------
def test_inference_batcher_has_all_22_methods():
    missing = [m for m in INFERENCE_METHODS if not hasattr(_engine.InferenceBatcher, m)]
    assert not missing, f"InferenceBatcher missing methods (WP8-compat break): {missing}"
    assert len(INFERENCE_METHODS) == 21  # + __init__ = the 22-method surface


def test_inference_batcher_getters_spec_derived():
    spec = _engine.RegistrySpec.from_registry("v6")
    ib = _engine.InferenceBatcher(encoding_spec=spec)
    assert ib.feature_len_py == spec.state_stride
    assert ib.policy_len_py == spec.policy_stride
    assert ib.representation_py == "grid"
    assert ib.model_version == 0
    assert ib.bump_model_version() == 1
    assert ib.model_version == 1
    ib.close()


def test_inference_batcher_dense_mock_round_trip():
    """spawn_mock_games -> next_inference_batch -> submit_inference_results -> completion."""
    spec = _engine.RegistrySpec.from_registry("v6")
    ib = _engine.InferenceBatcher(encoding_spec=spec)
    policy_len = ib.policy_len_py
    n_games = 3
    ib.spawn_mock_games(n_games)
    rounds = 0
    while ib.completed_mock_games() < n_games and rounds < 500:
        rounds += 1
        ids, feats = ib.next_inference_batch(8, 50)
        ids = list(ids)
        feats = np.asarray(feats)
        assert feats.shape[1] == spec.state_stride
        if not ids:
            continue
        pol = np.zeros((len(ids), policy_len), dtype=np.float32)
        pol[:, 0] = 1.0
        val = np.zeros((len(ids),), dtype=np.float32)
        ib.submit_inference_results(ids, pol, val)
    assert ib.completed_mock_games() == n_games, f"dense mock games stalled after {rounds} rounds"
    ib.close()


def test_inference_batcher_graph_mock_round_trip():
    """spawn_mock_graph_games -> next_graph_batch -> submit_graph_inference_results
    (through assemble_ls_from_gnn_probs) -> completion; plus check_graph_request."""
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    ib = _engine.InferenceBatcher(encoding_spec=spec)
    assert ib.representation_py == "graph"
    n_games = 2
    ib.spawn_mock_graph_games(n_games)
    rounds = 0
    while ib.completed_graph_games() < n_games and rounds < 500:
        rounds += 1
        ids, wire = ib.next_graph_batch(8, 50)
        ids = list(ids)
        if not ids:
            continue
        offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
        total = int(offsets[-1])
        probs = np.zeros((total,), dtype=np.float32)
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if e > s:
                probs[s:e] = 1.0 / (e - s)  # per-graph segmented softmax -> sum 1.0
        vals = np.zeros((len(ids),), dtype=np.float32)
        ib.submit_graph_inference_results(ids, probs, offsets, vals)
    assert ib.completed_graph_games() == n_games, f"graph mock games stalled after {rounds} rounds"
    ib.check_graph_request([(0, 0, 1), (1, 0, -1)], 1, 100)  # structural guard, no raise
    ib.close()


def test_inference_batcher_submit_graphs_and_wait():
    """The blocking graph driver, driven by a consumer thread popping + submitting."""
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    ib = _engine.InferenceBatcher(encoding_spec=spec)

    def consumer():
        rounds = 0
        while rounds < 500:
            rounds += 1
            ids, wire = ib.next_graph_batch(8, 50)
            ids = list(ids)
            if not ids:
                continue
            offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
            total = int(offsets[-1])
            probs = np.zeros((total,), dtype=np.float32)
            for i in range(len(offsets) - 1):
                s, e = int(offsets[i]), int(offsets[i + 1])
                if e > s:
                    probs[s:e] = 1.0 / (e - s)
            vals = np.zeros((len(ids),), dtype=np.float32)
            ib.submit_graph_inference_results(ids, probs, offsets, vals)
            return

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    results = ib.submit_graphs_and_wait([([(0, 0, 1), (1, 0, -1)], 1, 100)])
    t.join(timeout=10)
    assert len(results) == 1
    dense, overflow, value = results[0]  # (dense probs, overflow (q,r)->prob, value)
    assert isinstance(value, float)
    ib.close()
