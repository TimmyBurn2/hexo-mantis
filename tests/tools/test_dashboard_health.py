"""The health badge: bad beats warn beats unmeasured beats ok, and an absent input can never
leave the badge green."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def health(dashboard):
    return importlib.import_module("dashboard.health")


def _record(reader, tmp_path: Path, rows: list[dict], record_dir: Path | None = None):
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return reader.load_record(events, None, record_dir)


def _clean_rows() -> list[dict]:
    return [
        {"event": "run_segment_started", "segment": 1, "run_id": "t", "ts": 1.0},
        {"event": "heartbeat_watchdog_armed", "enabled": True},
        {"event": "selfplay_stall_watchdog_armed", "enabled": True},
        # The ply-cap terms armed at window 4 ride the one monitor_gates row.
        {"event": "monitor_gates", "step": 1000, "ply_cap_abort_rate": 0.5, "ply_cap_window_games": 4,
         "gates": {"grad_norm_hard_abort": {"checks": 1000, "fires": 0, "skips": 0, "warns": 0}}},
        {"event": "resume_state_persisted", "step": 1000, "unreceipted_bundles": [1000]},
        {"event": "disk_free", "disk_free_gb": 88.0, "ts": 2.0},
        {"event": "trainer_step", "step": 0, "policy_loss": 2.30},
        {"event": "trainer_step", "step": 1000, "policy_loss": 2.28},
        {"event": "trainer_step", "step": 2000, "policy_loss": 2.25},
        # Four decisive games, so the ply-cap window above is measured.
        *({"event": "game_complete", "winner": 0, "moves": 30, "terminal_reason": "six_in_a_row"}
          for _ in range(4)),
    ]


def _states(reading) -> dict[str, str]:
    return {i.name: i.state for i in reading.inputs}


def test_a_fully_measured_clean_record_is_ok(health, reader, tmp_path):
    rec = _record(reader, tmp_path, _clean_rows(), record_dir=tmp_path)
    reading = health.assess(rec)
    assert reading.state == "ok", _states(reading)
    assert set(_states(reading).values()) == {"ok"}


def test_an_absent_input_makes_the_badge_unmeasured_never_green(health, reader, tmp_path):
    rows = [r for r in _clean_rows() if r["event"] != "monitor_gates"]
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert reading.state == "unmeasured"
    assert _states(reading)["gate fires"] == "unmeasured"
    assert any("monitor_gates" in i.reason for i in reading.inputs)


def test_no_record_dir_leaves_the_firings_input_unmeasured(health, reader, tmp_path):
    reading = health.assess(_record(reader, tmp_path, _clean_rows()))
    assert _states(reading)["F-816-37 firings"] == "unmeasured"
    assert reading.state == "unmeasured"


def test_a_gate_fire_or_a_hard_abort_is_bad(health, reader, tmp_path):
    rows = _clean_rows()
    rows[3]["gates"]["grad_norm_hard_abort"]["fires"] = 1
    assert health.assess(_record(reader, tmp_path, rows, tmp_path)).state == "bad"
    rows = _clean_rows() + [{"event": "hard_abort", "rule": "x", "step": 5, "message": "m"}]
    reading = health.assess(_record(reader, tmp_path, rows, tmp_path))
    assert reading.state == "bad" and _states(reading)["hard aborts"] == "bad"


def test_a_watchdog_fire_is_bad_and_a_disarmed_watchdog_is_unmeasured(health, reader, tmp_path):
    rows = _clean_rows() + [{"event": "heartbeat_watchdog_fired"}]
    assert _states(health.assess(_record(reader, tmp_path, rows, tmp_path)))["watchdogs"] == "bad"
    rows = [r for r in _clean_rows() if not r["event"].endswith("_armed")]
    assert _states(health.assess(_record(reader, tmp_path, rows, tmp_path)))["watchdogs"] == "unmeasured"


def test_training_alerts_and_mirror_lag_are_warnings(health, reader, tmp_path):
    rows = _clean_rows() + [{"event": "training_alert", "rule": "loss_increase_window", "step": 4}]
    reading = health.assess(_record(reader, tmp_path, rows, tmp_path))
    assert reading.state == "warn" and _states(reading)["training alerts"] == "warn"
    rows = _clean_rows()
    rows[4]["unreceipted_bundles"] = [1000, 2000]
    reading = health.assess(_record(reader, tmp_path, rows, tmp_path))
    assert reading.state == "warn" and _states(reading)["mirror receipts"] == "warn"


def test_bad_outranks_warn_outranks_unmeasured(health, reader, tmp_path):
    rows = _clean_rows() + [{"event": "training_alert", "rule": "r", "step": 4},
                            {"event": "hard_abort", "rule": "x", "step": 5, "message": "m"}]
    assert health.assess(_record(reader, tmp_path, rows)).state == "bad"
    rows = _clean_rows() + [{"event": "training_alert", "rule": "r", "step": 4}]
    assert health.assess(_record(reader, tmp_path, rows)).state == "warn"


def _with_policy_loss(rows: list[dict], series: list[tuple[int, float]]) -> list[dict]:
    kept = [r for r in rows if r["event"] != "trainer_step"]
    return kept + [{"event": "trainer_step", "step": s, "policy_loss": v} for s, v in series]


def test_the_policy_loss_trough_is_a_warning_not_a_halt(health, reader, tmp_path):
    """R351(d): the prereg's trough signature reads `warn`, never `bad` — the halt is demoted."""
    rows = _with_policy_loss(_clean_rows(), [(0, 2.28), (1000, 2.50), (2000, 2.60), (3000, 2.70)])
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["policy-loss trough"] == "warn"
    assert reading.state == "warn"
    detail = next(i.reason for i in reading.inputs if i.name == "policy-loss trough")
    assert "2.28" in detail and "0.2" in detail and "R351(d)" in detail


def test_a_rise_after_the_trough_window_or_a_short_rise_is_not_the_signature(health, reader, tmp_path):
    late = _with_policy_loss(_clean_rows(), [(0, 2.28), (6000, 2.60), (7000, 2.70), (8000, 2.80)])
    assert _states(health.assess(_record(reader, tmp_path, late, record_dir=tmp_path)))[
        "policy-loss trough"] == "ok"
    short = _with_policy_loss(_clean_rows(), [(0, 2.28), (1000, 2.60), (2000, 2.70), (3000, 2.30)])
    assert _states(health.assess(_record(reader, tmp_path, short, record_dir=tmp_path)))[
        "policy-loss trough"] == "ok"


def test_fewer_than_two_policy_loss_rows_leave_the_trough_unmeasured(health, reader, tmp_path):
    rows = _with_policy_loss(_clean_rows(), [(0, 2.28)])
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["policy-loss trough"] == "unmeasured"


def _games(*reasons: str) -> list[dict]:
    return [{"event": "game_complete", "winner": -1 if r == "ply_cap" else 0, "moves": 256 if r == "ply_cap" else 30,
             "terminal_reason": r} for r in reasons]


def test_a_rolling_window_over_the_rate_reads_bad_the_attractor_signature(health, reader, tmp_path):
    """R352(c): three caps in the last four games (0.75 > 0.5), read whether or not the halt was armed."""
    rows = _clean_rows() + _games("six_in_a_row", "ply_cap", "ply_cap", "ply_cap")
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["ply-cap attractor"] == "bad"
    detail = next(i.reason for i in reading.inputs if i.name == "ply-cap attractor")
    assert "0.75" in detail and "4" in detail and "R352(c)" in detail


def test_a_window_at_or_below_the_rate_is_ok_with_the_peak_stated(health, reader, tmp_path):
    rows = _clean_rows() + _games("ply_cap", "six_in_a_row", "ply_cap", "six_in_a_row")  # peak 0.5
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["ply-cap attractor"] == "ok"
    detail = next(i.reason for i in reading.inputs if i.name == "ply-cap attractor")
    assert "0.50" in detail


def test_fewer_games_than_the_window_leave_the_ply_cap_input_unmeasured(health, reader, tmp_path):
    rows = [r for r in _clean_rows() if r["event"] != "game_complete"] + _games("ply_cap", "ply_cap", "ply_cap")
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["ply-cap attractor"] == "unmeasured"
    assert reading.state == "unmeasured"


def test_an_unarmed_record_is_read_at_the_minted_terms_and_says_so(health, reader, tmp_path):
    """No ply-cap terms on `monitor_gates`: read at R352(c)'s minted {0.5, 600}."""
    rows = [r for r in _clean_rows() if r["event"] != "monitor_gates"] + [
        {"event": "monitor_gates", "step": 2000,
         "gates": {"grad_norm_hard_abort": {"checks": 2000, "fires": 0, "skips": 0, "warns": 0}}}]
    rows += _games(*(["six_in_a_row"] * 600 + ["ply_cap"] * 400))
    reading = health.assess(_record(reader, tmp_path, rows, record_dir=tmp_path))
    assert _states(reading)["ply-cap attractor"] == "bad"
    detail = next(i.reason for i in reading.inputs if i.name == "ply-cap attractor")
    assert "600" in detail and "not armed" in detail
