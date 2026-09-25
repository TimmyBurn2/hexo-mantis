"""`tools/bench_server.py`: the real batcher and `InferenceServer`, positions replayed from a run's games, one row per B."""
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


def _cell(bench, tmp_path: Path, *, plant_noise: bool = False) -> dict:
    config = load_config(_CONFIG)
    spec = lookup(config.identity.encoding)
    net = build_net(arch_from_spec_and_config(spec, config.model_dump()))
    if plant_noise:
        forward = net.forward_batch

        def noisy(*args, **kwargs):
            logits, value, bins = forward(*args, **kwargs)
            return logits, value + 1e-3 * torch.rand_like(value), bins

        net.forward_batch = noisy
    positions = bench.positions_from_events(_events(tmp_path, [_GAME, _GAME[:7]]), config.identity.encoding, limit=None)
    return bench.run_cell(net, torch.device("cpu"), config.model_dump(), positions, batch_size=8,
                          workers=2, leaf_batch=4, seconds=1.0, compile_trunk=False, probe=6, windows=2)


def test_a_cell_serves_every_leaf_its_workers_submitted(bench, tmp_path: Path) -> None:
    """The budget witness: served leaves == Σ positions the workers submitted, on the real server."""
    row = _cell(bench, tmp_path)
    assert row["batch_size"] == 8 and row["workers"] == 2 and row["leaf_batch"] == 4
    assert row["submitted"] > 0 and row["served"] == row["submitted"]
    assert 1 <= row["b_mean"] <= 8 and row["pops"] >= 1 and row["leaves"] <= row["served"]
    assert len(row["probe_values"]) == 6 and len(row["window_leaves_per_s"]) == 2
    assert row["leaves_per_s_q1"] <= row["leaves_per_s_median"] <= row["leaves_per_s_q3"]
    assert row["device"] == "cpu" and row["wall_s"] >= 1.0


def test_the_repeat_probe_reads_exact_on_a_deterministic_server(bench, tmp_path: Path) -> None:
    """The CPU fp32 path is deterministic, so the same distinct positions served twice compare equal."""
    rep = _cell(bench, tmp_path)["probe_repeat"]
    assert rep == {"n": 6, "exact": True, "max_abs_value": 0.0, "max_abs_policy": 0.0}


def test_the_repeat_probe_reds_on_a_planted_nondeterministic_net(bench, tmp_path: Path) -> None:
    """PLANTED BREAK: a net whose value carries fresh noise per forward must read DIFFERS, or the probe is blind."""
    rep = _cell(bench, tmp_path, plant_noise=True)["probe_repeat"]
    assert rep["exact"] is False and rep["max_abs_value"] > 0.0


def test_the_probe_positions_are_pairwise_distinct(bench) -> None:
    """Duplicates collapse before the spread is taken; a pool short of `n` distinct positions is refused."""
    a, b, c = ([(0, 0, 1)], -1, 1), ([(0, 0, 1), (1, 0, -1)], 1, 2), ([(1, 0, -1), (0, 0, 1)], 1, 2)
    assert bench.distinct_positions([a, a, b, c, a], 2) == [a, b]
    with pytest.raises(ValueError, match="2 distinct positions, the probe needs 3"):
        bench.distinct_positions([a, b, c], 3)


def _row(*windows: float, **over) -> dict:
    row = {"batch_size": 64, "workers": 32, "leaf_batch": 8, "device": "cuda", "compile_trunk": True,
           "max_wait_ms": 10, "edge_geometry_check": "checker_thread", "window_leaves_per_s": list(windows)}
    return {**row, **over}


def test_the_iqr_reading_needs_separated_quartiles_and_the_run_spread(bench) -> None:
    """Faster only when q1 clears the baseline's q3 AND the median clears RUN_SPREAD; one run's windows understate the noise."""
    assert bench.quartiles([1.0, 2.0, 3.0, 4.0, 5.0]) == (2.0, 3.0, 4.0)
    base = _row(100.0, 101.0, 102.0, 103.0, 104.0)
    fast = _row(110.0, 111.0, 112.0, 113.0, 114.0)
    separated_but_small = _row(104.0, 105.0, 106.0, 107.0, 108.0)
    got = bench.compare_rows(base, fast)
    assert got["faster_beyond_iqr"] and not got["slower_beyond_iqr"]
    assert got["delta_pct"] == pytest.approx(100.0 * 10 / 102)
    assert not bench.compare_rows(base, separated_but_small)["faster_beyond_iqr"]
    assert bench.compare_rows(fast, base)["slower_beyond_iqr"]


def test_a_baseline_from_another_cell_or_before_l0_is_refused(bench) -> None:
    """A different load shape or a record without sub-windows is not a baseline, and says why."""
    with pytest.raises(bench.BaselineMismatch, match="workers"):
        bench.compare_rows(_row(100.0, workers=48), _row(110.0))
    with pytest.raises(bench.BaselineMismatch, match="before the repeat probe"):
        bench.compare_rows({"batch_size": 64, "leaves_per_s": 100.0}, _row(110.0))
