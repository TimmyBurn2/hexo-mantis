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


@pytest.fixture(scope="module")
def reader(dashboard):
    return importlib.import_module("dashboard.reader")


def _record(reader, tmp_path: Path, rows: list[dict], record_dir: Path | None = None):
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return reader.load_record(events, None, record_dir)


def _clean_rows() -> list[dict]:
    return [
        {"event": "run_segment_started", "segment": 1, "run_id": "t", "ts": 1.0},
        {"event": "heartbeat_watchdog_armed", "enabled": True},
        {"event": "selfplay_stall_watchdog_armed", "enabled": True},
        {"event": "monitor_gates", "step": 1000,
         "gates": {"grad_norm_hard_abort": {"checks": 1000, "fires": 0, "skips": 0, "warns": 0}}},
        {"event": "resume_state_persisted", "step": 1000, "unreceipted_bundles": [1000]},
        {"event": "disk_free", "disk_free_gb": 88.0, "ts": 2.0},
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
