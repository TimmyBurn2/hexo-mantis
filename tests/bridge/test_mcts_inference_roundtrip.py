"""MCTSTree + InferenceBatcher round-trip.

MCTSTree: ctor-compose, new_game -> select_leaves -> get_policy / get_improved_policy, and
forced_root_child get/set. InferenceBatcher: the compat surface is present AND exercised over
the graph queue — a method that only exists would pass a presence check while raising on every
call, so each is driven for real.
"""
import threading

import numpy as np
import pytest

from mantis import _engine

# The Python-facing InferenceBatcher compat surface. A LIST, not a count: it is compared to
# `dir()` in BOTH directions, which is what a count could never do.
INFERENCE_METHODS = [
    "close",
    "bump_model_version",
    "model_version",
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


def test_mctstree_ctor_compose_and_policy_round_trip():
    tree = _engine.MCTSTree(1.5, 1.0, 0.25, True, 0.3)  # new_full + configure_quiescence
    assert tree.quiescence_fire_count == 0
    board = _engine.Board.with_encoding_name("gnn_axis_v1")
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
    """The setter validates against the ROOT'S CHILD RANGE, so the round-trip needs a root
    that HAS children: on a bare tree an arbitrary index decodes to a cell an UNBOUNDED board
    accepts, producing neither a panic nor an error."""
    tree = _engine.MCTSTree()
    assert tree.forced_root_child is None
    with pytest.raises(ValueError, match="not a child of the root"):
        tree.forced_root_child = 3

    board = _engine.Board.with_encoding_name("gnn_axis_v1")
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
    """The graph legal-set expand door, executed rather than only named.

    `submit_graphs_and_wait_ls` carries the BUILDER's window centre OUT; the tree method
    carries dense + ragged overflow + that centre back IN, through the same
    `expand_and_backup_ls_at` self-play expands through.
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
    dense, overflow, value, center = results[0]
    assert len(dense) == spec.policy_stride
    assert isinstance(value, float)
    assert len(tuple(center)) == 2

    tree.expand_and_backup_ls_graph(
        [list(dense)], [list(overflow)], [float(value)], [tuple(center)],
        spec.policy_stride, spec.trunk_size,
    )
    assert tree.get_root_children_info(), "the graph legal-set expand produced no children"
    ib.close()


#: Public methods on the live class the compat list deliberately does NOT carry, each with its
#: ground. Asserted for EQUALITY, so a row that stops being true reds as loudly as a new method.
NOT_IN_THE_COMPAT_SURFACE: dict[str, str] = {
    "graph_max_in_flight": "a CONSTRUCTION parameter read back, not a WP8 call-surface method; "
                           "it post-dates the compat list and belongs to the fused-graph caps",
    "lock_recoveries": "an instrument counter (poisoned-mutex recoveries), added by the "
                       "lock-recovery repair; nothing in the WP8 surface calls it",
}


def test_the_declared_inference_surface_equals_the_live_class():
    """Set equality, both directions: a count form could only notice an edit to the literal,
    while this reds on a new `#[pymethods]` fn nobody declared."""
    live = {name for name in dir(_engine.InferenceBatcher) if not name.startswith("_")}
    declared = set(INFERENCE_METHODS)
    assert declared - live == set(), (
        f"the compat list names methods the class does not carry: {sorted(declared - live)}"
    )
    assert live - declared == set(NOT_IN_THE_COMPAT_SURFACE), (
        "the live class and the declared compat surface disagree. A NEW public method must "
        "either join INFERENCE_METHODS or be declared in NOT_IN_THE_COMPAT_SURFACE with its "
        f"ground; a declared exclusion that is gone must be removed. Live-only: "
        f"{sorted(live - declared)}; declared exclusions: {sorted(NOT_IN_THE_COMPAT_SURFACE)}"
    )
    assert len(declared) > 10, "vacuity floor: an emptied compat list would pass the diffs above"


def test_inference_batcher_getters_spec_derived():
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    ib = _engine.InferenceBatcher(encoding_spec=spec)
    assert ib.policy_len_py == spec.policy_stride
    assert ib.representation_py == "graph"
    assert ib.model_version == 0
    assert ib.bump_model_version() == 1
    assert ib.model_version == 1
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
    dense, overflow, value = results[0]  # overflow maps (q,r) -> prob
    assert isinstance(value, float)
    ib.close()
