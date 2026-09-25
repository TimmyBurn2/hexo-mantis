"""`tools/bench_server.py` (R360(b) step 2): the real batcher and `InferenceServer`, positions replayed from a run's games, one row per B."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
#: One production member: the witness is the server's leaf accounting, and each member builds a full net.
_CONFIG = production_configs(_REPO)[0]
_GAME = ["(-2,2)", "(1,2)", "(3,1)", "(-2,1)", "(-2,0)", "(-2,-2)", "(-2,-1)", "(-1,1)", "(-1,0)",
         "(0,0)", "(-3,2)", "(0,1)", "(-3,1)", "(-4,1)", "(1,1)", "(-4,2)", "(-1,-1)", "(0,-2)"]


@pytest.fixture(scope="module")
def bench():
    path = _REPO / "tools" / "bench_server.py"
    return load_module_by_path("bench_server_under_test", path)


def _events(tmp_path: Path, games: list[list[str]]) -> Path:
    path = tmp_path / "events_runx_seg0001.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "runner_started", "n_workers": 2}) + "\n")
        for moves in games:
            fh.write(json.dumps({"event": "game_complete", "moves": len(moves),
                                 "moves_list": moves, "winner": 0}) + "\n")
    return path


def test_positions_are_every_prefix_of_every_recorded_game(bench, tmp_path: Path) -> None:
    """A game of n moves yields n `(stones, current_player, moves_remaining)` from ply 0; `limit` keeps the LAST ones."""
    path = _events(tmp_path, [_GAME[:5], _GAME[:3]])
    positions = bench.positions_from_events(path, "gnn_axis_r8", limit=None)
    assert len(positions) == 8
    stones, player, remaining = positions[0]
    assert stones == [] and player == 1 and remaining == 1
    stones, player, remaining = positions[2]
    assert len(stones) == 2 and player == -1 and remaining == 1
    assert all(p in (1, -1) and r in (1, 2) for _s, p, r in positions)
    last3 = bench.positions_from_events(path, "gnn_axis_r8", limit=3)
    assert [len(s) for s, _p, _r in last3] == [0, 1, 2], "the last three are the second game's"


def test_summary_derives_the_step1_columns_from_two_snapshots(bench) -> None:
    """PERF3's columns — pops/s, B, served leaves/s, cycle, the stage means per pop — as differences of cumulative totals."""
    def snap(pops: int, leaves: int, hist: dict[str, int], ms: float) -> dict:
        agg = {"count": pops, "total_ms": ms * pops, "mean_ms": ms, "min_ms": ms, "max_ms": ms}
        return {"batch_size": 64, "max_wait_ms": 10,
                "queue_wait": dict(agg), "collate": dict(agg),
                "occupancy": {"count": pops, "total": leaves, "mean": leaves / max(pops, 1),
                              "min": 1, "max": 64, "fill_pct_mean": 0.0, "histogram": hist},
                "empty_polls": 0,
                "fusion": {"fusion_parts": pops, "fused_batch_edges": {"count": pops, "total": leaves * 1000,
                           "mean": 1000.0, "min": 1, "max": 1, "histogram": {}}},
                "pipeline": {"depth": 2, "launch": dict(agg), "gpu_wait": dict(agg)}}
    before = snap(100, 4000, {"32": 80, "64": 20}, 10.0)
    after = snap(200, 8600, {"32": 165, "64": 35}, 10.0)
    row = bench.summarize(before, after, wall_s=1.444)
    assert row["pops"] == 100 and row["leaves"] == 4600
    assert row["b_mean"] == pytest.approx(46.0)
    assert row["pops_per_s"] == pytest.approx(100 / 1.444)
    assert row["leaves_per_s"] == pytest.approx(4600 / 1.444)
    assert row["cycle_ms"] == pytest.approx(14.44)
    assert row["full_share"] == pytest.approx(0.15) and row["sat_share"] == pytest.approx(0.85)
    assert row["deadline_share"] == pytest.approx(0.0)
    assert row["launch_ms"] == pytest.approx(10.0) and row["gpu_wait_ms"] == pytest.approx(10.0)
    assert row["edges_per_graph"] == pytest.approx(1000.0)


def test_a_cell_serves_every_leaf_its_workers_submitted(bench, tmp_path: Path) -> None:
    """The budget witness (R360(b)): served leaves == Σ positions the workers submitted, on the real server."""
    config = load_config(_CONFIG)
    spec = lookup(config.identity.encoding)
    net = build_net(arch_from_spec_and_config(spec, config.model_dump()))
    positions = bench.positions_from_events(_events(tmp_path, [_GAME, _GAME[:7]]), config.identity.encoding, limit=None)
    row = bench.run_cell(net, torch.device("cpu"), config.model_dump(), positions,
                         batch_size=8, workers=2, leaf_batch=4, seconds=1.0, compile_trunk=False)
    assert row["batch_size"] == 8 and row["workers"] == 2 and row["leaf_batch"] == 4
    assert row["submitted"] > 0 and row["served"] == row["submitted"]
    assert 1 <= row["b_mean"] <= 8 and row["pops"] >= 1 and row["leaves"] <= row["served"]
    assert len(row["probe_values"]) == 4
    assert row["device"] == "cpu" and row["wall_s"] >= 1.0
