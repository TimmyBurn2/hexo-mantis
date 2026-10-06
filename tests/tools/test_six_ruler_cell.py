"""The Six ruler cell: gen 30 at 16 nodes, threaded from the follower's unit through the frontier to the receipt and the dashboard."""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
from pathlib import Path

import pytest
from _six_fake import fake_vendor
from _toolpath import load_module_by_path

from mantis.arena.books import book_sha256

from mantis._engine import Board
from mantis.bots.six import FINDING_LOG_MARKER, PROVIDER_LOG_MARKER, resolve_six
from mantis.monitor.logging_setup import _FORMAT
from mantis.util.hashing import sha256_file
from mantis.config.census import production_configs

_REPO = Path(__file__).resolve().parents[2]
_CELL = {"label": "six30_16", "candidate": "x", "search_kind": "puct", "sims": 256, "opponent": "six",
         "six_net": "gen0030", "six_nodes": 16, "games": 288, "concurrency": 8}


def _board(*moves: tuple[int, int]) -> Board:
    board = Board.with_encoding_name("gnn_axis_r8")
    for q, r in moves:
        board.apply_move(q, r)
    return board


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
        f"2026 INFO mantis.bots.six {PROVIDER_LOG_MARKER} provider=cuda variant=gen0030 nodes=16 engine_sha256=e net_sha256=n runtime_sha256=r",
        f"2026 INFO mantis.bots.six {PROVIDER_LOG_MARKER} provider=cuda variant=gen0030 nodes=16 engine_sha256=e net_sha256=n runtime_sha256=r",
        f"2026 WARNING mantis.bots.six {FINDING_LOG_MARKER} failed ply 40: bestmove none",
        "2026 INFO unrelated line provider=cpu",
    ])
    record = frontier.six_log_record(log)
    assert record["six_engine"] == {"starts": 2, "provider": "cuda", "engine_sha256": ["e"], "net_sha256": ["n"],
                                    "runtime_sha256": ["r"], "searches": 0, "stale_pending": 0}
    assert record["six_findings"]["count"] == 1 and "bestmove none" in record["six_findings"]["first"][0]
    assert frontier.six_log_record("")["six_engine"] == {"starts": 0, "provider": None, "engine_sha256": [],
                                                         "net_sha256": [], "runtime_sha256": [], "searches": 0,
                                                         "stale_pending": 0}


def test_the_follower_unit_composes_the_cell_and_names_its_receipt(follower) -> None:
    assert follower.UNITS["six30_16"] == (256, 16, "six30_16") and follower.SIX_UNITS["six30_16"] == ("gen0030", 30)
    cell = follower.compose_cell(Path("/x/run8_00045000_deadbeef.ckpt"), unit="six30_16", step=45000, games=288,
                                 concurrency=8, label="six30_16_run8_45000")
    assert {k: cell[k] for k in ("opponent", "six_net", "six_nodes", "sims", "search_kind")} == {
        "opponent": "six", "six_net": "gen0030", "six_nodes": 16, "sims": 256, "search_kind": "puct"}
    assert "strix_sims" not in cell and "opening_book" not in cell, "the config's gate book, as every receipt on record"
    assert follower.sidecar_path(Path("/x/a.ckpt"), "six30_16").name == "a.ckpt.six30_16.json"


@pytest.mark.parametrize(("unit", "net", "sha8"), [("six150_eq", "gen0150", "21672eeb"), ("six200_eq", "gen0200", "17328c43"),
                                                   ("six250_eq", "gen0250", "250451cc"), ("six300_eq", "gen0300", "d9cc22c4")])
def test_an_equal_playout_unit_plays_its_generation_at_256_per_turn_against_128_per_stone_on_the_arena_book(
        follower, unit: str, net: str, sha8: str, tmp_path: Path) -> None:
    assert follower.UNITS[unit] == (128, 256, unit) and follower.SIX_UNITS[unit] == (net, int(net[3:]))
    cell = follower.compose_cell(Path("/x/r_00156000_deadbeef.ckpt"), unit=unit, step=156000, games=128,
                                 concurrency=8, label=unit)
    assert {k: cell[k] for k in ("sims", "six_net", "six_nodes", "opening_book")} == {
        "sims": 128, "six_net": net, "six_nodes": 256, "opening_book": "arena_s20261006_p5"}
    assert follower.opponent_pin(unit)["net_sha256"].startswith(sha8)
    ckpt = tmp_path / "r_00156000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    body = follower.sidecar_record(ckpt, unit=unit, trigger="once", record=_record(cell={"step": 1, "concurrency": 8,
                                   **cell}), regime_name="IDLE", regime_evidence={}, run_id="r", started=0.0,
                                   finished=1.0, pin={})
    assert body["opening_book"] == "arena_s20261006_p5" and body["ours"]["sims"] == 128 and body["six"]["nodes"] == 256
    assert body["opening_book_sha256"] == book_sha256("arena_s20261006_p5")
    assert len(body["opening_book_sha256"]) == 64, "the book's manifest pin, verified against its file"


def test_equal_work_on_the_arena_book_is_s_at_its_own_budget(follower) -> None:
    cell = follower.compose_cell(Path("/x/a.ckpt"), unit="equal_work_arena", step=1, games=576, concurrency=8, label="s")
    assert (cell["sims"], cell["strix_sims"], cell["opening_book"]) == (256, 256, "arena_s20261006_p5")
    assert "strix_radius" not in cell and "strix_solver" not in cell


def _record(**over) -> dict:
    record = {"label": "l", "cell": {"step": 45000, "concurrency": 8}, "rc": 0, "wall_sec": 1.0,
              "provenance": {"candidate": {"net_hash": "n"}},
              "readout": {"wr": 0.35, "wr_ci_lower": 0.3, "wr_ci_upper": 0.4, "games": 288, "eff_n": 288},
              "six_engine": {"starts": 8, "provider": "cuda", "engine_sha256": ["e"], "net_sha256": ["n"],
                             "runtime_sha256": ["r"], "searches": 100, "stale_pending": 0},
              "six_findings": {"count": 0, "first": []}}
    record.update(over)
    return record


def test_the_receipt_is_labelled_with_the_generation_the_nodes_and_the_provider(follower, tmp_path: Path) -> None:
    ckpt = tmp_path / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    pin = {"commit": "f2b5ec2", "engine_sha256": "e", "net": "gen-0030.onnx", "net_sha256": "n", "runtime_sha256": "r"}
    body = follower.sidecar_record(ckpt, unit="six30_16", trigger="once", record=_record(), regime_name="IDLE",
                                   regime_evidence={}, run_id="run8", started=0.0, finished=1.0, pin=pin)
    assert "strix" not in body and body["unit"] == "six30_16"
    assert body["six"] == {**pin, "generation": 30, "nodes": 16, "cache_entries": 0, "provider": "cuda",
                           "engine_starts": 8, "searches": 100, "stale_pending": 0}
    assert body["six_findings"] == {"count": 0, "first": []} and body["wr"] == 0.35


def test_follow_reads_the_ruler_units_only(follower) -> None:
    assert set(follower.FOLLOW_UNITS) == {"equal_work", "six30_16", "equal_work_arena", "six150_eq", "six200_eq",
                                          "six250_eq", "six300_eq"}
    with pytest.raises(SystemExit):
        follower.main(["--config", "c", "--run-dir", "d", "--run-id", "r", "--work-dir", "w", "--follow",
                       "--unit", "net_only"])


def test_the_six_pin_is_read_off_the_pin_file(follower) -> None:
    pin = follower.opponent_pin("six30_16")
    assert pin["commit"] == "f2b5ec2d4d7ec42e8b655f739e65821808e03698" and pin["net"] == "gen-0030.onnx"
    assert pin["net_sha256"].startswith("4afcbb11") and pin["engine_sha256"].startswith("7d06548b")
    assert [h[:8] for h in pin["runtime_sha256"].split(",")] == ["1aacefdf", "1defa2f8", "c6a12593"]
    assert follower.opponent_pin("equal_work")["checkpoint"] == "checkpoint_00237000.pt"


@pytest.fixture(scope="module")
def sidecars(dash):
    return importlib.import_module("dash.readers.sidecars")


def test_the_dash_reads_the_six_rung_as_its_own_series(follower, sidecars, tmp_path: Path) -> None:
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    ckpt = ck / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    pin = {"commit": "f2b5ec2", "engine_sha256": "e", "net": "gen-0030.onnx", "net_sha256": "n", "runtime_sha256": "r"}
    six = follower.sidecar_record(ckpt, unit="six30_16", trigger="once", record=_record(), regime_name="IDLE",
                                  regime_evidence={}, run_id="run8", started=0.0, finished=1.0, pin=pin)
    strix = follower.sidecar_record(ckpt, unit="equal_work", trigger="once", record=_record(wall_sec=2.0),
                                    regime_name="IDLE", regime_evidence={}, run_id="run8", started=0.0,
                                    finished=1.0, pin={})
    (ck / f"{ckpt.name}.six30_16.json").write_text(json.dumps(six), encoding="utf-8")
    (ck / f"{ckpt.name}.strix256.json").write_text(json.dumps(strix), encoding="utf-8")
    (ck / f"{ckpt.name}.six30_16.failed.json").write_text("{}", encoding="utf-8")
    cells, skipped = sidecars.load([ck])
    assert sorted(c.family for c in cells) == ["six", "strix"] and len({c.unit for c in cells}) == 2
    assert any("a failed cell" in s for s in skipped)
    six_cell = next(c for c in cells if c.family == "six")
    assert six_cell.label.endswith("no tactics. Six: gen 30, 16 nodes") and six_cell.regime == "IDLE"


def test_the_real_producers_lines_read_back_through_the_frontier(frontier, tmp_path: Path,
                                                                   caplog: pytest.LogCaptureFixture) -> None:
    """The producer test: the lines `mantis.bots.six` logs, in the child's own format, are what `six_log_record` reads."""
    caplog.set_level(logging.INFO)
    runtime = hashlib.sha256(b"a runtime").hexdigest()
    root = fake_vendor(tmp_path, runtime_sha=runtime)
    bot = resolve_six(opponent_sims=16, variant="gen0030", device="cuda", vendor_root=root)()
    bot.new_game()
    for q, r in ((0, 0), (1, 0), (0, 1)):
        bot.observe_move(q, r)
    bot.select_move(_board((0, 0), (1, 0), (0, 1)))  # the fake answers `bestmove none`: one search, one forfeit
    bot.close()
    formatter = logging.Formatter(_FORMAT)
    record = frontier.six_log_record("\n".join(formatter.format(r) for r in caplog.records))
    engine = sha256_file(root / "external" / "six-assets" / "release" / "engine" / "sixengine")
    assert record["six_engine"] == {"starts": 1, "provider": "cuda", "engine_sha256": [engine],
                                    "net_sha256": [sha256_file(root / "external" / "six-assets" / "gen-0030.onnx")],
                                    "runtime_sha256": [runtime], "searches": 1, "stale_pending": 0}
    assert record["six_findings"]["count"] == 1


def test_follow_accepts_the_six_unit(follower, monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(follower, "_real_run_cell", lambda _config, _work: (lambda cell: {}))
    monkeypatch.setattr(follower.Follower, "follow", lambda self, _poll: seen.append(self.unit))
    assert follower.main(["--config", "c", "--run-dir", str(tmp_path), "--run-id", "r", "--work-dir",
                          str(tmp_path / "w"), "--follow", "--unit", "six30_16"]) == 0
    assert seen == ["six30_16"]


@pytest.mark.parametrize("nets, status", [(["n"], "written"), (["n", "m"], "failed"), (["m"], "failed"), ([], "failed")])
def test_the_receipt_names_the_bytes_played_and_refuses_anything_but_the_pins(follower, tmp_path: Path,
                                                                              nets: list[str], status: str) -> None:
    ckpt = tmp_path / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    pin = {"commit": "f2b5ec2", "engine_sha256": "e", "net": "gen-0030.onnx", "net_sha256": "n", "runtime_sha256": "r"}
    record = _record(six_engine={"starts": 9, "provider": "cuda", "engine_sha256": ["e"], "net_sha256": nets,
                                 "runtime_sha256": ["r"], "searches": 10, "stale_pending": 0})
    run = follower.Follower(run_dir=tmp_path, run_id="run8", run_cell=lambda cell: record, unit="six30_16", pin=pin,
                            clock=lambda: 0.0, log=lambda _s: None,
                            host_load=lambda: follower.HostLoad(load_1m=0.0, cpu_count=16, gpu_util_pct=(0,)))
    got, path = run.read_one(ckpt, trigger="once")
    assert got == status
    if status == "written":
        assert json.loads(path.read_text(encoding="utf-8"))["six"]["net_sha256"] == "n"
    else:
        assert "sha256" in json.loads(path.read_text(encoding="utf-8"))["error"]


def test_a_cell_that_played_another_runtime_is_a_failed_cell(follower) -> None:
    pin = {"engine_sha256": "e", "net_sha256": "n", "runtime_sha256": "r"}
    played = {"engine_sha256": ["e"], "net_sha256": ["n"], "runtime_sha256": ["other"]}
    assert "runtime_sha256" in follower.played_bytes_error({"six_engine": played}, pin)
    assert follower.played_bytes_error({"six_engine": {**played, "runtime_sha256": ["r"]}}, pin) is None


def test_a_cell_that_exits_without_an_error_names_its_rc(follower, tmp_path: Path) -> None:
    ckpt = tmp_path / "run8_00045000_deadbeef.ckpt"
    ckpt.write_bytes(b"w")
    run = follower.Follower(run_dir=tmp_path, run_id="run8", run_cell=lambda cell: {"label": "l", "rc": 1, "wall_sec": 1.0},
                            unit="six30_16", pin={}, clock=lambda: 0.0, log=lambda _s: None,
                            host_load=lambda: follower.HostLoad(load_1m=0.0, cpu_count=16, gpu_util_pct=(0,)))
    status, path = run.read_one(ckpt, trigger="once")
    assert status == "failed" and "rc 1" in json.loads(path.read_text(encoding="utf-8"))["error"]
