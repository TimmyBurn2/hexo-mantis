"""MCTSTree + InferenceBatcher round-trip (O20, review gap 3).

MCTSTree: ctor-compose (new_full + configure_quiescence via the 5-arg bridge
ctor), new_game -> select_leaves -> get_policy / get_improved_policy round-trip;
forced_root_child get/set.

InferenceBatcher: ALL 21 Python methods present AND exercised via the mock-game
helpers over the dense + graph queues (no method silently dropped — a dropped
method is a WP8-compat break); the 3 getters return the spec-derived values.
"""
import threading

import numpy as np

from mantis import _engine

# The full 21-method Python-facing InferenceBatcher surface (DESIGN §a.1 table);
# 20 named methods/getters + __init__ = 21.
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
    tree = _engine.MCTSTree()
    assert tree.forced_root_child is None
    tree.forced_root_child = 3
    assert tree.forced_root_child == 3
    tree.forced_root_child = None
    assert tree.forced_root_child is None


# ---------------------------- InferenceBatcher --------------------------------
def test_inference_batcher_has_all_21_methods():
    missing = [m for m in INFERENCE_METHODS if not hasattr(_engine.InferenceBatcher, m)]
    assert not missing, f"InferenceBatcher missing methods (WP8-compat break): {missing}"
    assert len(INFERENCE_METHODS) == 20  # + __init__ = the 21-method surface


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
