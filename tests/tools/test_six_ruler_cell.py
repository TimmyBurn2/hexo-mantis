"""The Six ruler cell: gen 30 at 16 nodes, threaded from the follower's unit through the frontier to the receipt and the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _toolpath import load_module_by_path

from mantis.bots.six import FINDING_LOG_MARKER, PROVIDER_LOG_MARKER
from mantis.config.census import production_configs

_REPO = Path(__file__).resolve().parents[2]
_CELL = {"label": "six30_16", "candidate": "x", "search_kind": "puct", "sims": 256, "opponent": "six",
         "six_net": "gen0030", "six_nodes": 16, "games": 288, "concurrency": 8}


@pytest.fixture(scope="module")
def follower():
    return load_module_by_path("strix_follower_six", _REPO / "tools/strix_follower.py")


@pytest.fixture(scope="module")
def frontier():
    return load_module_by_path("strength_frontier_six", _REPO / "tools/strength_frontier.py")


@pytest.mark.parametrize("config_path", production_configs(_REPO), ids=lambda p: p.name)
def test_a_six_cell_composes_the_ruler_at_its_nodes_and_the_candidate_at_its_sims(
        frontier, tmp_path: Path, config_path: Path) -> None:
    from mantis.config.loader import load_config

    config = load_config(config_path)
    base = frontier.base_round_spec(config, work_dir=tmp_path / "w")
    spec = frontier.cell_spec(_CELL, base, cell_dir=tmp_path / "c", config=config)
    (job,) = spec.rung_jobs
    assert (job.bot, job.variant, job.opponent_sims, job.opening_book) == ("six", "gen0030", 16,
                                                                          config.eval.gate.opening_book)
    assert spec.rung_model_sims == 256 and spec.search_kind == "puct" and not spec.gate.run_gate
    assert frontier.cell_channel(_CELL) == "external"
    for key in ("six_net", "six_nodes"):
        with pytest.raises(frontier.FrontierCellError, match=key):
            frontier.cell_spec({k: v for k, v in _CELL.items() if k != key}, base, cell_dir=tmp_path / "d",
                               config=config)


def test_the_row_names_the_network_and_the_nodes(frontier) -> None:
    readout = {"wr": 0.35, "wr_ci_lower": 0.3, "wr_ci_upper": 0.4, "wins": 100, "losses": 186, "draws": 2,
               "games": 288, "eff_n": 288, "sec_per_game": 3.1, "median_plies": 40.0}
    row = frontier.format_row({"label": "l", "cell": _CELL, "rc": 0, "wall_sec": 1.0, "readout": readout,
                               "six_findings": {"count": 2, "first": []}})
    assert "six_net=gen0030" in row and "six_nodes=16" in row and "six forfeits 2" in row


def test_the_child_log_yields_the_forfeits_and_the_provider(frontier) -> None:
    log = "\n".join([
        f"2026 INFO mantis.bots.six {PROVIDER_LOG_MARKER} provider=cuda variant=gen0030 nodes=16 engine_sha256=e net_sha256=n",
        f"2026 INFO mantis.bots.six {PROVIDER_LOG_MARKER} provider=cuda variant=gen0030 nodes=16 engine_sha256=e net_sha256=n",
        f"2026 WARNING mantis.bots.six {FINDING_LOG_MARKER} failed ply 40: bestmove none",
        "2026 INFO unrelated line provider=cpu",
    ])
    record = frontier.six_log_record(log)
    assert record["six_engine"] == {"starts": 2, "provider": "cuda", "engine_sha256": ["e"], "net_sha256": ["n"]}
    assert record["six_findings"]["count"] == 1 and "bestmove none" in record["six_findings"]["first"][0]
    assert frontier.six_log_record("")["six_engine"] == {"starts": 0, "provider": None, "engine_sha256": [],
                                                         "net_sha256": []}


def test_the_follower_unit_composes_the_cell_and_names_its_receipt(follower) -> None:
    assert follower.UNITS["six30_16"] == (256, 16, "six30_16") and follower.SIX_UNITS == {"six30_16": ("gen0030", 30)}
    cell = follower.compose_cell(Path("/x/run8_00045000_deadbeef.ckpt"), unit="six30_16", step=45000, games=288,
                                 concurrency=8, label="six30_16_run8_45000")
    assert {k: cell[k] for k in ("opponent", "six_net", "six_nodes", "sims", "search_kind")} == {
        "opponent": "six", "six_net": "gen0030", "six_nodes": 16, "sims": 256, "search_kind": "puct"}
    assert "strix_sims" not in cell
    assert follower.sidecar_path(Path("/x/a.ckpt"), "six30_16").name == "a.ckpt.six30_16.json"


def _record(**over) -> dict:
    record = {"label": "l", "cell": {"step": 45000, "concurrency": 8}, "rc": 0, "wall_sec": 1.0,
              "provenance": {"candidate": {"net_hash": "n"}},
              "readout": {"wr": 0.35, "wr_ci_lower": 0.3, "wr_ci_upper": 0.4, "games": 288, "eff_n": 288},
              "six_engine": {"starts": 8, "provider": "cuda", "engine_sha256": ["e"], "net_sha256": ["n"]},
              "six_findings": {"count": 0, "first": []}}
    record.update(over)
    return record


def test_the_receipt_is_labelled_with_the_generation_the_nodes_and_the_provider(follower, tmp_path: Path) -> None:
    ckpt = tmp_path / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    pin = {"commit": "f2b5ec2", "engine_sha256": "e", "net": "gen-0030.onnx", "net_sha256": "n"}
    body = follower.sidecar_record(ckpt, unit="six30_16", trigger="once", record=_record(), regime_name="IDLE",
                                   regime_evidence={}, run_id="run8", started=0.0, finished=1.0, pin=pin)
    assert "strix" not in body and body["unit"] == "six30_16"
    assert body["six"] == {**pin, "generation": 30, "nodes": 16, "cache_entries": 0, "provider": "cuda",
                           "engine_starts": 8}
    assert body["six_findings"] == {"count": 0, "first": []} and body["wr"] == 0.35


def test_follow_reads_the_ruler_units_only(follower) -> None:
    assert set(follower.FOLLOW_UNITS) == {"equal_work", "six30_16"}
    with pytest.raises(SystemExit):
        follower.main(["--config", "c", "--run-dir", "d", "--run-id", "r", "--work-dir", "w", "--follow",
                       "--unit", "net_only"])


def test_the_six_pin_is_read_off_the_pin_file(follower) -> None:
    pin = follower.opponent_pin("six30_16")
    assert pin["commit"] == "f2b5ec2d4d7ec42e8b655f739e65821808e03698" and pin["net"] == "gen-0030.onnx"
    assert pin["net_sha256"].startswith("4afcbb11") and pin["engine_sha256"].startswith("7d06548b")
    assert follower.opponent_pin("equal_work")["checkpoint"] == "checkpoint_00237000.pt"


def test_the_dashboard_draws_the_six_rung_as_its_own_series(follower, external, tmp_path: Path) -> None:
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    ckpt = ck / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    pin = {"commit": "f2b5ec2", "engine_sha256": "e", "net": "gen-0030.onnx", "net_sha256": "n"}
    six = follower.sidecar_record(ckpt, unit="six30_16", trigger="once", record=_record(), regime_name="IDLE",
                                  regime_evidence={}, run_id="run8", started=0.0, finished=1.0, pin=pin)
    strix = follower.sidecar_record(ckpt, unit="equal_work", trigger="once", record=_record(wall_sec=2.0),
                                    regime_name="IDLE", regime_evidence={}, run_id="run8", started=0.0,
                                    finished=1.0, pin={})
    (ck / f"{ckpt.name}.six30_16.json").write_text(json.dumps(six), encoding="utf-8")
    (ck / f"{ckpt.name}.strix256.json").write_text(json.dumps(strix), encoding="utf-8")
    (ck / f"{ckpt.name}.six30_16.failed.json").write_text("{}", encoding="utf-8")
    points, note = external.load_external_points([ck])
    labels = sorted(external.series_by_unit(points))
    assert labels == ["run8 · equal_work: ours PUCT-256 vs strix 256 sims",
                      "run8 · six30_16: ours PUCT-256 vs Six gen 30 @ 16 nodes"]
    assert "2 sidecar(s) read" in note and "failed cell, not a receipt" in note
    six_point = next(p for p in points if p.unit == "six30_16")
    assert six_point.opponent == "Six gen 30" and "vs Six gen 30" in external.gap_statement(six_point)
