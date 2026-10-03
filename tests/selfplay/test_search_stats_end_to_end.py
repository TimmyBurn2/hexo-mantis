"""The search-stats wire end to end: a real pool at `search_stats_every: 1` writes visits/q/prior per ply."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from _fused_caps import CAPS_DICT
from mantis._engine import HexgBuffer
from mantis.diagnostics.ring_reader import load_ring
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.monitor.game_record import iter_run_games
from mantis.monitor.game_recorder import GameRecorder
from mantis.selfplay.pool import WorkerPool

_ENCODING = "gnn_axis_v1"
_TIMEOUT_S = 60.0
_PLY_CAP = 6


def _cfg() -> dict[str, Any]:
    selfplay: dict[str, Any] = {
        "search": {"kind": "gumbel", "tactics": None}, "n_workers": 1, "leaf_batch_size": 8,
        "max_game_moves": _PLY_CAP, "c_visit": 50.0, "c_scale": 1.0, "q_rescale": True,
        "gumbel_m": 4, "gumbel_explore_moves": 10, "search_stats_every": 1,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 8,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    inference = {
        "inference_batch_size": 4, "inference_max_wait_ms": 10,
        "edge_geometry_check": "inline", "compile_trunk": False,
        "fused_graph_caps": CAPS_DICT,
    }
    return {"encoding": _ENCODING, "deploy": {"search": {"kind": "puct", "tactics": None}}, "selfplay": selfplay,
            "inference": inference, "train": {}}


@pytest.mark.integration
def test_a_sampled_self_play_record_reaches_the_shard(tmp_path: Path) -> None:
    spec = lookup(_ENCODING)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=16, num_layers=1)
    recorder = GameRecorder(record_dir=tmp_path, run_id="e2e", seed=1)
    pool = WorkerPool(
        build_net(arch), _cfg(), torch.device("cpu"),
        HexgBuffer(capacity=256, encoding=_ENCODING, visit_capacity=128), arch=arch,
        recorder=recorder,
    )
    pool.start()
    try:
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and recorder.games_written == 0:
            pool.check_producer_health()
            time.sleep(0.2)
    finally:
        pool.stop()
    pool.check_producer_health()

    games = list(iter_run_games(tmp_path, "e2e"))
    assert games, "no self-play record reached the shard inside the budget"
    sampled = [g for g in games if "search_stats" in g]
    assert len(sampled) == len(games), "at search_stats_every=1 every game is sampled"
    record = sampled[0]
    searched = sum(1 for sims in record["move_sims"] if sims > 0)
    assert len(record["search_stats"]) == searched, "one entry per SEARCHED ply"
    for entry in record["search_stats"]:
        assert set(entry) >= {"ply", "root_value", "root_raw", "search_value", "visits", "q", "prior"}
        assert entry["visits"], "a searched root has visited children"
        assert len(entry["q"]) == len(entry["visits"]) == len(entry["prior"])
        assert all(n >= 1 for _q, _r, n in entry["visits"])
        assert -1.0 <= entry["root_value"] <= 1.0 and -1.0 <= entry["search_value"] <= 1.0


@pytest.mark.integration
def test_every_row_the_pool_pushes_carries_its_searchs_root_value(tmp_path: Path) -> None:
    """Through the production drain every pushed row is flagged, and each (ply, search value) the shard recorded is a ring row's."""
    spec = lookup(_ENCODING)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=16, num_layers=1)
    recorder = GameRecorder(record_dir=tmp_path, run_id="e2e", seed=1)
    ring = HexgBuffer(capacity=256, encoding=_ENCODING, visit_capacity=128)
    pool = WorkerPool(build_net(arch), _cfg(), torch.device("cpu"), ring, arch=arch, recorder=recorder)
    pool.start()
    try:
        deadline = time.monotonic() + _TIMEOUT_S
        while time.monotonic() < deadline and recorder.games_written < 2:
            pool.check_producer_health()
            time.sleep(0.2)
    finally:
        pool.stop()
    pool.check_producer_health()
    path = tmp_path / "ring.hexg"
    ring.save_to_path(str(path))
    rows = load_ring(path)
    assert rows.header.size > 0, "the pool pushed no row inside the budget"
    assert rows.root_value_valid.tolist() == [1] * rows.header.size, "a searched self-play row carries its root value"
    in_ring = set(zip(rows.ply_index.tolist(), rows.root_value.tolist(), strict=True))
    recorded = [(e["ply"], e["search_value"]) for g in iter_run_games(tmp_path, "e2e") for e in g["search_stats"]]
    assert recorded and all(pair in in_ring for pair in recorded), "a recorded search value is no ring row's"
