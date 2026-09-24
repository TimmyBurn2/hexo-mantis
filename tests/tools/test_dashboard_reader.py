"""The record reader: refuses nothing-records, reads the ladder file in the shape the run WRITES,
and says what it dropped."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from _dashboard_rows import write_events


def test_an_eventless_record_REFUSES_rather_than_rendering_a_clean_page(reader, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n   \nnot json at all\n", encoding="utf-8")
    with pytest.raises(reader.EmptyRunRecord):
        reader.load_record(empty)


def test_heavy_per_game_payloads_are_dropped_and_the_drop_is_counted(reader, tmp_path):
    rows = [{"event": "game_complete", "moves": 11, "winner": 0, "moves_list": ["(0,0)"] * 11,
             "moves_detail": None, "value_trace": None, "terminal_reason": "six_in_a_row"}]
    rec = reader.load_record(write_events(tmp_path, rows))
    game = rec.rows("game_complete")[0]
    assert "moves_list" not in game and "moves_detail" not in game
    assert game["moves"] == 11 and game["terminal_reason"] == "six_in_a_row"
    assert rec.dropped_fields == {"game_complete": 3}


def test_the_ladder_file_is_read_in_the_shape_the_run_writes(reader, tmp_path):
    ladder = tmp_path / "eval_ladder_state.json"
    ladder.write_text(json.dumps({
        "sealbot_d5": {"name": "sealbot_d5", "status": "active", "consec": 0,
                       "history": [{"round_idx": 1, "games": 32, "wr": 0.1875, "ci_lo": 0.0625}]},
    }), encoding="utf-8")
    rec = reader.load_record(write_events(tmp_path, [{"event": "run_boot_identity"}]), ladder)
    assert [name for name, _ in rec.rungs()] == ["sealbot_d5"]
    assert rec.rungs()[0][1][0]["games"] == 32


def test_an_unparseable_ladder_file_is_an_absence_with_its_reason(reader, tmp_path):
    ladder = tmp_path / "eval_ladder_state.json"
    ladder.write_text("{not json", encoding="utf-8")
    rec = reader.load_record(write_events(tmp_path, [{"event": "run_boot_identity"}]), ladder)
    assert rec.ladder is None and rec.rungs() == []
    assert "did not parse" in rec.ladder_note


def test_a_series_keeps_only_finite_numeric_pairs(reader, tmp_path):
    rows = [{"event": "trainer_step", "step": 1, "loss": 2.0},
            {"event": "trainer_step", "step": 2, "loss": None},
            {"event": "trainer_step", "step": "x", "loss": 1.0},
            {"event": "trainer_step", "step": 4, "loss": float("nan")},
            {"event": "trainer_step", "step": 5, "loss": 0.5}]
    rec = reader.load_record(write_events(tmp_path, rows))
    assert rec.series("trainer_step", "step", "loss") == [(1.0, 2.0), (5.0, 0.5)]


def test_wall_hours_run_from_the_first_row_to_the_last(reader, tmp_path):
    rows = [{"event": "run_boot_identity", "ts": 1000.0},
            {"event": "trainer_step", "ts": 1000.0 + 7200.0}]
    rec = reader.load_record(write_events(tmp_path, rows))
    assert rec.wall_hours() == pytest.approx(2.0)
    assert reader.load_record(write_events(tmp_path, [{"event": "x"}])).wall_hours() is None
