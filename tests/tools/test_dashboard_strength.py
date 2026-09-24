"""Strength readings: the ladder's per-round rows joined to the round events by round index, the
three-state promotion reading, and the register's Elo-slope witness as the trend."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from _dashboard_rows import ladder_state


@pytest.fixture(scope="module")
def strength(dashboard):
    return importlib.import_module("dashboard.strength")


def _round(idx: int, wr, games_total=56, promoted=None, lo=None, hi=None) -> dict:
    return {"event": "eval_round_complete", "round_id": f"r{idx:06d}_{idx * 1000}",
            "step": idx * 1000, "wall_sec": 600.0, "games_total": games_total,
            "promoted": promoted, "wr_sealbot": wr, "wr_sealbot_ci_lower": lo,
            "wr_sealbot_ci_upper": hi}


def _record(reader, tmp_path: Path, rows: list[dict], ladder: dict | None = None):
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ladder_path = None
    if ladder is not None:
        ladder_path = tmp_path / "eval_ladder_state.json"
        ladder_path.write_text(json.dumps(ladder), encoding="utf-8")
    return reader.load_record(events, ladder_path)


def test_ladder_rows_join_the_round_events_by_round_index(strength, reader, tmp_path):
    rec = _record(reader, tmp_path,
                  [_round(1, 0.1875, lo=0.0625, hi=0.34375), _round(2, 0.25, promoted=True)],
                  ladder_state([(1, 32, 0.1875), (2, 32, 0.25)]))
    series = strength.rung_series(rec)
    assert list(series) == ["sealbot_d5"]
    first, second = series["sealbot_d5"]
    assert (first.step, first.wr, first.games) == (1000, 0.1875, 32)
    assert first.ci == (0.0625, 0.34375) and first.wilson is not None
    assert first.promoted is None and second.promoted is True
    assert second.step == 2000 and not second.broken


def test_without_a_ladder_the_primary_rung_is_read_from_the_round_events_alone(
        strength, reader, tmp_path):
    rec = _record(reader, tmp_path, [_round(1, 0.1875, lo=0.0625, hi=0.34375)])
    series = strength.rung_series(rec)
    (point,) = series[strength.primary_rung(rec)]
    assert point.wr == 0.1875 and point.games is None and point.wilson is None
    assert point.ci == (0.0625, 0.34375)


def test_a_broken_round_is_a_broken_point_at_its_step_not_a_zero(strength, reader, tmp_path):
    rec = _record(reader, tmp_path,
                  [_round(1, 0.2), _round(2, None, games_total=None, promoted=False)],
                  ladder_state([(1, 32, 0.2)]))
    points = strength.rung_series(rec)["sealbot_d5"]
    assert [p.step for p in points] == [1000, 2000]
    assert points[1].broken and points[1].wr is None
    assert points[1].promoted is None, "a broken round took no decision, whatever it emitted"


def test_the_promotion_reading_has_three_states(strength, reader, tmp_path):
    rec = _record(reader, tmp_path, [_round(1, 0.2), _round(2, 0.3, promoted=True),
                                     _round(3, 0.1, promoted=False), _round(4, 0.2)])
    decided, latest = strength.promotion_reading(rec)
    assert decided is not None and (decided.state, decided.step) == ("not promoted", 3000)
    assert latest is not None and (latest.state, latest.step) == ("no decision taken", 4000)
    assert strength.promotion_reading(_record(reader, tmp_path, [{"event": "x"}])) == (None, None)


def test_the_trend_is_the_ols_elo_slope_per_thousand_steps_over_completed_rounds(
        strength, reader, tmp_path):
    rounds = [_round(i, wr) for i, wr in enumerate([0.2, 0.25, 0.3, 0.35, 0.4], start=1)]
    rounds.append(_round(6, None, games_total=None))
    rec = _record(reader, tmp_path, rounds, ladder_state([(i, 32, wr) for i, wr in
                                                     enumerate([0.2, 0.25, 0.3, 0.35, 0.4], 1)]))
    fit = strength.trend(strength.rung_series(rec)["sealbot_d5"])
    assert fit is not None and fit.n == 5, "the broken round cannot be fitted and is excluded"
    assert fit.slope > 0 and fit.excludes_zero


def test_a_zero_win_rate_round_enters_the_trend_as_a_finite_deficit(strength, reader, tmp_path):
    rounds = [_round(i, wr) for i, wr in enumerate([0.0, 0.1, 0.2, 0.0], start=1)]
    rec = _record(reader, tmp_path, rounds, ladder_state([(1, 32, 0.0), (2, 32, 0.1),
                                                     (3, 32, 0.2), (4, 32, 0.0)]))
    fit = strength.trend(strength.rung_series(rec)["sealbot_d5"])
    assert fit is not None and fit.n == 4
