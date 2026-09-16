"""The search-stats wire end to end (R355(d)): a real pool at `search_stats_every: 1` writes visits/q/prior per ply."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis._engine import HexgBuffer
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
        "search": {"kind": "gumbel"}, "n_workers": 1, "leaf_batch_size": 8,
        "max_game_moves": _PLY_CAP, "c_visit": 50.0, "c_scale": 1.0, "q_rescale": True,
        "gumbel_m": 4, "gumbel_explore_moves": 10, "search_stats_every": 1,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 8, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 8, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    inference = {
        "inference_batch_size": 4, "inference_max_wait_ms": 10,
        "edge_geometry_check": "inline", "compile_trunk": False,
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    return {"encoding": _ENCODING, "deploy": {"search": {"kind": "puct"}}, "selfplay": selfplay,
            "inference": inference, "train": {"draw_reward": -0.5, "ply_cap_value": -0.5}}


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
        assert set(entry) >= {"ply", "root_value", "root_raw", "visits", "q", "prior"}
        assert entry["visits"], "a searched root has visited children"
        assert len(entry["q"]) == len(entry["visits"]) == len(entry["prior"])
        assert all(n >= 1 for _q, _r, n in entry["visits"])
        assert -1.0 <= entry["root_value"] <= 1.0
