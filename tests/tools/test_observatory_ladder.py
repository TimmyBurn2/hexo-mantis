"""The ladder join over a Snapshot gives the dashboard's points, readings and slope."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ladder(observatory):
    return importlib.import_module("observatory.readers.ladder")


@pytest.fixture(scope="module")
def series(observatory):
    return importlib.import_module("observatory.readers.series")


def _round(idx: int, wr, games_total=56, promoted=None, lo=None, hi=None) -> dict:
    return {"event": "eval_round_complete", "round_id": f"r{idx:06d}_{idx * 1000}", "step": idx * 1000,
            "games_total": games_total, "promoted": promoted, "wr_sealbot": wr,
            "wr_sealbot_ci_lower": lo, "wr_sealbot_ci_upper": hi, "ts": float(idx)}


def _snap(series, rows):
    red = series.Reducers()
    for r in rows:
        red.feed(r)
    return red.snapshot()


def _ladder_file(tmp_path: Path, history) -> Path:
    path = tmp_path / "eval_ladder_state.json"
    path.write_text(json.dumps({"sealbot_d5": {
        "name": "sealbot_d5", "status": "active", "consec": 0,
        "history": [{"round_idx": i, "games": g, "wr": w} for i, g, w in history]}}), encoding="utf-8")
    return path


def test_ladder_rows_join_the_round_rows_by_index_and_carry_wilson(ladder, series, tmp_path):
    snap = _snap(series, [_round(1, 0.1875, lo=0.06, hi=0.34), _round(2, 0.25, promoted=True)])
    state, note = ladder.load_ladder(_ladder_file(tmp_path, [(1, 32, 0.1875), (2, 32, 0.25)]))
    assert note == "loaded"
    pts = ladder.rung_series(snap, state)["sealbot_d5"]
    assert [(p.step, p.wr, p.games, p.promoted, p.broken) for p in pts] == [
        (1000, 0.1875, 32, None, False), (2000, 0.25, 32, True, False)]
    assert pts[0].ci == (0.06, 0.34) and pts[0].wilson is not None and pts[1].ci is None


def test_without_a_ladder_the_primary_rung_reads_the_round_rows(ladder, series):
    snap = _snap(series, [_round(1, 0.2), _round(2, None, games_total=None)])
    state, note = ladder.load_ladder(None)
    assert state is None and "no eval_ladder_state.json" in note
    pts = ladder.rung_series(snap, state)[ladder.primary_rung(snap, state)]
    assert [(p.step, p.wr, p.broken) for p in pts] == [(1000, 0.2, False), (2000, None, True)]


def test_the_promotion_reading_has_three_states(ladder, series):
    snap = _snap(series, [_round(1, 0.1, promoted=False), _round(2, 0.2, promoted=None),
                          _round(3, None, games_total=None)])
    decided, latest = ladder.promotion_reading(snap)
    assert decided is not None and decided.state == "not promoted" and decided.step == 1000
    assert latest is not None and latest.state == "broken round — no decision taken" and latest.step == 3000
    assert ladder.promotion_reading(_snap(series, [])) == (None, None)


def test_the_trend_matches_the_dashboards_over_the_same_points(ladder, series, dashboard, tmp_path):
    wrs = (0.1, 0.15, 0.2, 0.3)
    rows = [_round(i, w) for i, w in enumerate(wrs, start=1)]
    snap = _snap(series, rows)
    state, _ = ladder.load_ladder(_ladder_file(tmp_path, [(i, 32, w) for i, w in enumerate(wrs, start=1)]))
    fit = ladder.trend(ladder.rung_series(snap, state)["sealbot_d5"])
    reader = importlib.import_module("dashboard.reader")
    strength = importlib.import_module("dashboard.strength")
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rec = reader.load_record(path, tmp_path / "eval_ladder_state.json")
    old = strength.trend(strength.rung_series(rec)["sealbot_d5"])
    assert fit is not None and old is not None
    assert (fit.slope, fit.lo, fit.hi, fit.n) == (old.slope, old.lo, old.hi, old.n)


def test_an_unparseable_ladder_file_is_an_absence_with_its_reason(ladder, tmp_path):
    path = tmp_path / "eval_ladder_state.json"
    path.write_text("{not json", encoding="utf-8")
    state, note = ladder.load_ladder(path)
    assert state is None and "did not parse" in note
