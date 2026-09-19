"""`tools/ladder/backends.py` (LADDER-1 §1.2): the two backends behind one seam — a compound turn out, the head's own sims count with it, the same game id replaying to the same moves."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board
from mantis.encoding import lookup
from mantis.model import GnnArchV2, build_net
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import save_checkpoint

_REPO = Path(__file__).resolve().parents[2]
_ENC = "gnn_axis_v1"
_TINY = dict(hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)


def _minimal_config() -> dict[str, Any]:
    path = _REPO / "tests" / "train" / "_warmstart_config.py"
    spec = importlib.util.spec_from_file_location("_warmstart_config_for_ladder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.minimal_config()


def _tiny_checkpoint(tmp_path: Path, encoding: str = _ENC) -> tuple[Path, str]:
    """A stamped tiny net; the stamp's config must agree with `encoding` (the one loader refuses a disagreeing pair)."""
    from mantis.config.loader import load_config

    spec = lookup(encoding)
    arch = GnnArchV2(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), **_TINY)
    net = build_net(arch)
    config = _minimal_config() if encoding == _ENC else load_config(_REPO / "configs" / "run8.yaml").model_dump()
    assert config["identity"]["encoding"] == encoding
    path = save_checkpoint(model=net, optimizer=None, scaler=None, scheduler=None, step=0,
                           config=config, kind="weights",
                           metadata_kwargs={"encoding_name": encoding, "run_id": "tiny", "arch": arch},
                           checkpoint_dir=tmp_path)
    return path, net_param_hash(net)


def _mid_game() -> Board:
    board = Board.with_encoding_name(_ENC)
    for q, r in [(0, 0), (1, 0), (0, 1), (-1, 0), (-1, 1), (2, 0), (2, -1)]:
        board.apply_move(q, r)
    return board


@pytest.fixture
def mantis_backend(ladder, smoke_run_config, tmp_path: Path):
    checkpoint, net_hash = _tiny_checkpoint(tmp_path)
    config = smoke_run_config("dev_example.yaml", eval={"gate": {"deploy_sims": 12}, "worker_device": "cpu"})
    backend = ladder.backends.open_mantis(config, checkpoint, threads=2)
    try:
        yield backend, net_hash, checkpoint
    finally:
        backend.close()


def test_the_mantis_backend_is_named_by_its_net_hash_and_plays_at_the_configs_deploy_sims(mantis_backend) -> None:
    backend, net_hash, _checkpoint = mantis_backend
    assert backend.backend == "mantis"
    assert backend.net_hash == net_hash
    assert backend.name == f"mantis:{net_hash[:8]}"
    assert backend.sims == 12
    assert backend.encoding == _ENC
    assert backend.search["kind"] == "puct"


def test_a_turn_is_two_distinct_legal_cells_and_the_heads_own_sims_count(ladder, mantis_backend) -> None:
    backend, _net_hash, _checkpoint = mantis_backend
    backend.new_game("g_1")
    board = _mid_game()
    turn = backend.select_turn(board)
    first, second = turn.placements
    assert first != second
    assert board.is_legal(*first)
    board.apply_move(*first)
    assert board.is_legal(*second)
    assert turn.sims == 24  # 12 leaves per stone, two stones, read off `DeployHeadPlayer.last_sims`
    assert turn.ms > 0


def test_the_same_game_id_replays_to_the_same_turn(mantis_backend) -> None:
    backend, _net_hash, _checkpoint = mantis_backend
    backend.new_game("g_7Qm2Kx")
    first = backend.select_turn(_mid_game()).placements
    backend.new_game("g_7Qm2Kx")
    assert backend.select_turn(_mid_game()).placements == first


def test_the_game_id_seeds_the_head(ladder, mantis_backend) -> None:
    backend, _net_hash, _checkpoint = mantis_backend
    backend.new_game("g_a")
    seed_a = backend.seed
    backend.new_game("g_b")
    assert backend.seed != seed_a
    backend.new_game("g_a")
    assert backend.seed == seed_a


def test_a_checkpoint_stamped_for_another_encoding_is_refused_by_name(ladder, smoke_run_config, tmp_path: Path) -> None:
    checkpoint, _net_hash = _tiny_checkpoint(tmp_path, encoding="gnn_axis_r8")
    config = smoke_run_config("dev_example.yaml", eval={"worker_device": "cpu"})
    with pytest.raises(ladder.backends.BackendError, match="gnn_axis_r8"):
        ladder.backends.open_mantis(config, checkpoint, threads=1)


def test_the_backend_seam_never_imports_a_ring_writer() -> None:
    """Posture A (packet §0.3): ladder games are EVAL; the absence of any ring writer is the pin."""
    forbidden = ("mantis.data", "HexgBuffer", "push_graph_position", "replay_buffer", "mantis.selfplay.runner")
    for path in [*sorted((_REPO / "tools" / "ladder").glob("*.py")), _REPO / "tools" / "ladder_bot.py"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        assert hits == [], f"{path.name} names a ring writer: {hits}"


@pytest.mark.integration
def test_the_strix_backend_plays_at_the_pin_with_the_drivers_own_sims(ladder) -> None:
    from mantis.bots.strix import strix_availability

    available, why = strix_availability()
    if not available:
        pytest.skip(f"LOUD SKIP — strix is not vendored here: {why}")
    backend = ladder.backends.open_strix(sims=8, threads=2)
    try:
        assert backend.backend == "strix"
        assert backend.name == f"strix:{backend.net_hash[:8]}"
        assert len(backend.net_hash) == 64
        assert backend.sims == 8 and backend.search["solver"] == "on"
        backend.new_game("g_1")
        board = Board.with_encoding_name("gnn_axis_r8")
        for q, r in [(0, 0), (1, 0), (0, 1)]:
            board.apply_move(q, r)
        turn = backend.select_turn(board)
        first, second = turn.placements
        assert first != second and board.is_legal(*first)
        board.apply_move(*first)
        assert board.is_legal(*second)
        assert turn.sims > 0
        backend.new_game("g_1")
        board2 = Board.with_encoding_name("gnn_axis_r8")
        for q, r in [(0, 0), (1, 0), (0, 1)]:
            board2.apply_move(q, r)
        assert backend.select_turn(board2).placements == turn.placements
    finally:
        backend.close()
