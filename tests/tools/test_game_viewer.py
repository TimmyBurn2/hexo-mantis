"""The game viewer (VIEWER-1, R352(g)): index, derived hex facts, the page, per-shard data, the CLI."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

from _toolpath import load_module_by_path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIM = REPO_ROOT / "tools" / "game_viewer.py"


@pytest.fixture(scope="module")
def hexlogic(viewer):
    return importlib.import_module("viewer.hexlogic")


@pytest.fixture(scope="module")
def reader(viewer):
    return importlib.import_module("viewer.reader")


@pytest.fixture(scope="module")
def html(viewer):
    return importlib.import_module("viewer.html")


# The hex facts.

def test_the_owner_of_a_ply_follows_the_engines_turn_mapping(hexlogic):
    """Ply 0 is p1's single; plies 1-2 are p2's turn, 3-4 p1's, 5-6 p2's (`Ply::turn`)."""
    assert [hexlogic.owner(p) for p in range(7)] == [0, 1, 1, 0, 0, 1, 1]


def _six_for_p1() -> list[list[int]]:
    """p1 lays (0,0)…(5,0) along the E axis; p2 answers far away, never six in a row. p1 plies: 0, 3, 4, 7, 8, 11."""
    p1 = [[k, 0] for k in range(6)]
    p2 = [[2 * k, 5 + k % 2] for k in range(6)]
    moves: list[list[int]] = [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4],
                              p2[4], p2[5], p1[5]]
    return moves


def test_the_axes_and_win_length_are_the_engines_own(hexlogic):
    """No hex fact is transcribed: the constants are the bridge's exports, value and order."""
    from mantis import _engine

    assert hexlogic.HEX_AXES == tuple((int(dq), int(dr)) for dq, dr in _engine.HEX_AXES)
    assert hexlogic.WIN_LENGTH == _engine.WIN_LENGTH


def test_the_six_in_a_row_is_found_through_the_completing_stone(hexlogic):
    moves = _six_for_p1()
    line = hexlogic.win_line(moves)
    assert line == [[k, 0] for k in range(6)]
    assert hexlogic.owner(len(moves) - 1) == 0, "the completing stone is p1's"


def test_a_longer_run_is_returned_whole_and_a_diagonal_axis_is_found(hexlogic):
    # p1 along (1,-1): (0,0),(1,-1),(2,-2),(3,-3),(4,-4),(5,-5); p2 elsewhere.
    p1 = [[k, -k] for k in range(6)]
    p2 = [[2 * k, 7 + k % 2] for k in range(6)]
    moves = [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4], p2[4], p2[5], p1[5]]
    assert hexlogic.win_line(moves) == p1


def test_no_line_on_a_board_without_six(hexlogic):
    assert hexlogic.win_line(_six_for_p1()[:-1]) is None
    assert hexlogic.win_line([]) is None


# The index and the shard data.

def _record(game_id: str, moves: list[list[int]], **over) -> dict:
    row = {"contract": "game-record-v1", "game_id": game_id, "run_id": "t", "channel": "selfplay",
           "step": 7, "step_kind": "actor", "seed": 1, "served_sims": 64, "plies": len(moves),
           "result": "p1", "termination": "six_in_a_row", "moves": moves, "worker_id": 3,
           "game_id_byte_hash": "abc"}
    row.update(over)
    return row


def _games_dir(tmp_path: Path, run_id: str = "t") -> Path:
    d = tmp_path / "games"
    d.mkdir()
    shard = d / f"games_{run_id}_seg0001_2026091400.jsonl"
    rows = [{"contract": "game-record-v1", "record": "shard_opened", "run_id": run_id, "segment": 1,
             "hour": "2026091400"},
            _record("g1", _six_for_p1()),
            _record("g2", [[0, 0], [1, 0], [0, 1]], result="draw", termination="ply_cap"),
            _record("r1_promotion_00001", [[0, 0], [1, 0], [0, 1], [2, 0]], channel="promotion",
                    step_kind="round", result="p2", termination="six_in_a_row", rung=None,
                    phase="screen", game_index=1, colors={"candidate": 1, "opponent": -1},
                    search_stats=[{"ply": 0, "by": "candidate", "root_value": 0.1,
                                   "visits": [[1, 0, 30], [0, 1, 10]]}])]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / f"games_{run_id}_index.jsonl").write_text(json.dumps(
        {"record": "shard_closed", "shard": shard.name, "games": 3, "bytes": shard.stat().st_size,
         "run_id": run_id, "segment": 1, "hour": "2026091400", "contract": "game-record-v1"}) + "\n",
        encoding="utf-8")
    return d


def test_the_index_carries_the_facts_the_list_filters_on(reader, tmp_path):
    built = reader.build_run(_games_dir(tmp_path), "t")
    assert [g["id"] for g in built.index] == ["g1", "g2", "r1_promotion_00001"]
    g1 = built.index[0]
    assert (g1["run"], g1["ch"], g1["res"], g1["pl"], g1["term"], g1["step"]) == (
        "t", "selfplay", "p1", 12, "six_in_a_row", 7)
    assert g1["shard"] == 0 and built.shards[0].name == "games_t_seg0001_2026091400.jsonl"
    assert built.index[2]["stats"] is True and g1["stats"] is False
    assert built.index[2]["cand"] == 1 and built.index[2]["sims"] == 64, "eval rows carry the seat and sims for the move label"
    assert "cand" not in g1 and "sims" not in g1


def test_the_shard_data_carries_moves_the_win_line_and_stats_where_present(reader, tmp_path):
    built = reader.build_run(_games_dir(tmp_path), "t")
    data = built.shards[0].games
    assert data["g1"]["m"] == _six_for_p1() and data["g1"]["win"] == [[k, 0] for k in range(6)]
    assert data["g2"]["win"] is None and "s" not in data["g2"], "no stats on self-play: absent, not []"
    assert data["r1_promotion_00001"]["s"] == [{"ply": 0, "by": "candidate", "root_value": 0.1,
                                                "visits": [[1, 0, 30], [0, 1, 10]]}]


def test_the_shard_data_carries_each_moves_arm_and_the_arms_sims(reader, tmp_path):
    """R353(d): one arm character per ply plus the sims per arm; a pre-producer record carries none."""
    d = tmp_path / "games"
    d.mkdir()
    shard = d / "games_t_seg0001_2026091400.jsonl"
    rows = [{"contract": "game-record-v1", "record": "shard_opened", "run_id": "t", "segment": 1,
             "hour": "2026091400"},
            _record("armed", [[0, 0], [1, 0], [0, 1], [2, 0]], result="draw", termination="ply_cap",
                    move_arms=["opening", "full", "fast", "fast"], move_sims=[0, 320, 64, 64]),
            _record("legacy", [[0, 0], [1, 0], [0, 1]], result="draw", termination="ply_cap")]
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    built = reader.build_run(d, "t")
    data = built.shards[0].games
    assert data["armed"]["a"] == "ofqq"
    assert data["armed"]["sims"] == {"f": 320, "q": 64}
    assert "a" not in data["legacy"] and "sims" not in data["legacy"]


def test_the_page_labels_every_stone_with_its_arm_and_states_an_unrecorded_arm(reader, html, tmp_path):
    page = html.render([reader.build_run(_games_dir(tmp_path), "t")], "t")
    assert "arm not recorded" in page, "a pre-producer record must say the arm is absent"
    assert "fast" in page and "full" in page and "opening" in page
    assert ".stone.fast" in page, "fast-arm stones are drawn apart from full-search ones"
    assert "cur.a" in page and "cur.sims" in page, "the label reads the shard's arm string and sims"


def test_a_six_in_a_row_record_whose_line_belongs_to_the_other_side_is_a_finding(reader, tmp_path):
    """The owner formula and the record's `result` must agree; a disagreement is reported, not hidden."""
    d = _games_dir(tmp_path)
    shard = next(d.glob("games_t_seg*.jsonl"))
    rows = [json.loads(l) for l in shard.read_text(encoding="utf-8").splitlines()]
    rows[1]["result"] = "p2"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    built = reader.build_run(d, "t")
    assert built.findings and "g1" in built.findings[0] and "p2" in built.findings[0]


def test_an_empty_games_dir_refuses(reader, tmp_path):
    d = tmp_path / "games"
    d.mkdir()
    with pytest.raises(reader.EmptyGameRecord):
        reader.build_run(d, "t")


# The page.

def test_the_page_is_self_contained_and_references_only_its_own_data_files(reader, html, tmp_path):
    built = reader.build_run(_games_dir(tmp_path), "t")
    page = html.render([built], "t")
    for forbidden in ("http://", "https://", "<iframe", "@import", "<link"):
        assert forbidden not in page
    srcs = re.findall(r'<script src="([^"]+)"', page)
    assert srcs == [], "shard data is loaded on demand, never eagerly"
    assert 'data-shard="data/t/games_t_seg0001_2026091400.js"' in page or "data/t/" in page
    assert "MANTIS_INDEX" in page and '"g1"' in page
    assert "ArrowLeft" in page and "ArrowRight" in page and "touchstart" in page, "keyboard and swipe"
    assert "writeText" in page, "c copies the position for the analyzer"
    assert "<svg" in page and "six" in page.lower()


def test_the_page_states_the_self_play_stats_gap_rather_than_drawing_an_empty_heatmap(reader, html, tmp_path):
    page = html.render([reader.build_run(_games_dir(tmp_path), "t")], "t")
    assert "no per-position search stats" in page and "MANTIS_STATS_GAP" in page
    assert "searchParams" in page and "replaceState" in page, "a position is addressable as ?g=run/id&ply=N"


def test_the_cli_writes_the_page_and_one_data_file_per_shard(tmp_path):
    module = load_module_by_path("game_viewer", _SHIM)
    d = _games_dir(tmp_path)
    out = tmp_path / "viewer"
    rc = module.main(["--run", f"t={d}", "--out", str(out), "--title", "t"])
    assert rc == 0
    assert (out / "index.html").is_file()
    data = out / "data" / "t" / "games_t_seg0001_2026091400.js"
    assert data.is_file()
    body = data.read_text(encoding="utf-8")
    assert body.startswith("window.MANTIS_SHARD(") and '"g1"' in body
