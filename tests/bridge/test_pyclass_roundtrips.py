"""Consumer-less pyclass round-trips (O10, LOCKED #1).

Every class with no Python consumer until WP8+ gets a pymethod round-trip here (the interim
live consumer): HexgBuffer, GraphTargets, TacticalSolver, SelfPlayRunner(Config), MCTSTree,
InferenceBatcher. `ReplayBuffer` was the seventh and went with the dense path (R346(f)).
MCTSTree + InferenceBatcher get their DEPTH coverage in test_mcts_inference_roundtrip.py
(O20); here they get a construction/round-trip smoke so no consumer-less class is
registered-but-unexercised.
"""
import numpy as np

from mantis import _engine


def test_hexg_buffer_and_graph_targets_round_trip():
    hb = _engine.HexgBuffer(16, "gnn_axis_v1", 128)
    hb.push_graph_position([(0, 0, 1), (1, 0, -1)], [(2, 0, 1.0)], 1, 100, 2, True, 0.0, True, 1)
    assert hb.size == 1
    assert hb.encoding_name == "gnn_axis_v1"
    wire, targets = hb.sample_graph_batch(1)
    # GraphTargets: the 4 COPY getters + target_argmax_cells.
    assert np.asarray(targets.policy_target).dtype == np.float32
    assert np.asarray(targets.outcomes).size >= 1
    assert np.asarray(targets.value_valid).dtype == np.uint8
    assert np.asarray(targets.is_full_search).dtype == np.uint8
    cells = targets.target_argmax_cells
    assert isinstance(cells, list) and len(cells) >= 1
    # GraphWire round-trips its scalar getters.
    assert wire.n_graphs == 1
    assert isinstance(wire.contract_version, int)


def test_tactical_solver_prove_round_trip():
    ts = _engine.TacticalSolver()
    board = _engine.Board.with_encoding_name("gnn_axis_v1")
    for q, r in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]:
        board.apply_move(q, r)
    result, moves, nodes = ts.prove(board, 3, 10_000)
    assert isinstance(result, int)
    assert isinstance(moves, list)
    assert isinstance(nodes, int) and nodes >= 1


def test_selfplay_runner_config_field_round_trip():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=2, encoding_name="gnn_axis_v1")
    # The post-ctor get/set knobs round-trip. Nine of the ten went with the solver, forced-win
    # and seed-corpus levers the dense path carried (R346(f)); `search_kind` is what is left.
    cfg.search_kind = "gumbel"
    assert cfg.search_kind == "gumbel"


def test_selfplay_runner_construct_and_counters():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=1, encoding_name="gnn_axis_v1")
    runner = _engine.SelfPlayRunner(cfg)
    assert runner.is_running() is False
    assert runner.model_version == 0
    runner.set_model_version(5)
    assert runner.model_version == 5
    assert isinstance(runner.batcher, _engine.InferenceBatcher)


def test_mcts_and_inference_batcher_construct():
    """Smoke: both remain constructible (depth coverage in O20)."""
    assert _engine.MCTSTree().root_visits() == 0
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    ib = _engine.InferenceBatcher(feature_len=0, policy_len=spec.policy_stride)
    assert ib.policy_len_py == spec.policy_stride
    assert ib.representation_py == "graph"
    ib.close()
