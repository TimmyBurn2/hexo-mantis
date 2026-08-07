"""Consumer-less pyclass round-trips (O10, LOCKED #1).

All 7 classes with no Python consumer until WP8+ get a pymethod round-trip here
(the interim live consumer): ReplayBuffer, HexgBuffer, GraphTargets,
TacticalSolver, SelfPlayRunner(Config), MCTSTree, InferenceBatcher. MCTSTree +
InferenceBatcher get their DEPTH coverage in test_mcts_inference_roundtrip.py
(O20); here they get a construction/round-trip smoke so no consumer-less class is
registered-but-unexercised.
"""
import numpy as np

from mantis import _engine


def test_replay_buffer_push_sample_round_trip():
    spec = _engine.RegistrySpec.from_registry("v6")
    s = spec.board_size
    n_cells = spec.n_cells
    state = np.zeros((8, s, s), dtype=np.float16)
    chain = np.zeros((6, s, s), dtype=np.float16)
    policy = np.zeros(spec.policy_stride, dtype=np.float32)
    policy[0] = 1.0
    own = np.ones(n_cells, dtype=np.uint8)
    wl = np.zeros(n_cells, dtype=np.uint8)
    rb = _engine.ReplayBuffer(16, "v6")
    rb.push(state, chain, policy, 0.0, own, wl)
    assert rb.size == 1 and rb.capacity == 16
    batch = rb.sample_batch(1, False)
    assert len(batch) == 8
    assert np.asarray(batch[0]).shape == (1, 8, s, s)
    assert np.asarray(batch[0]).dtype == np.float16
    assert len(rb.sample_batch_with_pos(1, False)) == 9
    assert rb.encoding.name == "v6"
    stats = rb.get_buffer_stats()
    assert stats[0] == 1 and stats[1] == 16


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
    board = _engine.Board.with_encoding_name("v6")
    for q, r in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]:
        board.apply_move(q, r)
    result, moves, nodes = ts.prove(board, 3, 10_000)
    assert isinstance(result, int)
    assert isinstance(moves, list)
    assert isinstance(nodes, int) and nodes >= 1


def test_selfplay_runner_config_field_round_trip():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=2, encoding_name="v6")
    # The 10 post-ctor get/set knobs round-trip.
    cfg.solver_enabled = True
    cfg.solver_depth = 7
    cfg.seed_fraction = 0.25
    cfg.forced_win_policy_enabled = True
    assert cfg.solver_enabled is True
    assert cfg.solver_depth == 7
    assert cfg.seed_fraction == 0.25
    assert cfg.forced_win_policy_enabled is True


def test_selfplay_runner_construct_and_counters():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=1, encoding_name="v6")
    runner = _engine.SelfPlayRunner(cfg)
    assert runner.is_running() is False
    assert runner.feature_len() == _engine.RegistrySpec.from_registry("v6").state_stride
    assert runner.model_version == 0
    runner.set_model_version(5)
    assert runner.model_version == 5
    assert isinstance(runner.batcher, _engine.InferenceBatcher)


def test_mcts_and_inference_batcher_construct():
    """Smoke: both remain constructible (depth coverage in O20)."""
    assert _engine.MCTSTree().root_visits() == 0
    ib = _engine.InferenceBatcher(feature_len=2888, policy_len=362)
    assert ib.feature_len_py == 2888
    ib.close()
