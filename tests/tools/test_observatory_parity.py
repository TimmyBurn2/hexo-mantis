"""The readers give the dashboard's numbers: on a synthetic full-shaped record, and on the run6 mirror record when MANTIS_DASH_FIXTURE_EVENTS names it (never a fixture in git, R7)."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

_SERIES = ([("trainer_step", "step", k) for k in ("value_loss", "policy_loss", "loss", "grad_norm", "lr")]
           + [("iteration_complete", "step", k)
              for k in ("games_per_hour", "positions_per_hour", "steps_per_hour", "sims_per_sec")]
           + [("disk_free", "ts", "disk_free_gb")])
_WHOLE = ("eval_round_complete", "eval_round_started", "monitor_gates", "resume_state_persisted",
          "training_alert", "eval_channel_health", "run_segment_started")


@pytest.fixture(scope="module")
def mods(observatory, dashboard):
    return {name: importlib.import_module(f"observatory.readers.{name}") for name in ("events", "ladder")} | {
        "reader": importlib.import_module("dashboard.reader"),
        "health": importlib.import_module("dashboard.health"),
        "strength": importlib.import_module("dashboard.strength")}


def _synthetic(logs: Path) -> Path:
    rows: list[dict] = [{"event": "run_segment_started", "run_id": "t", "segment": 1, "ts": 0.0},
                        {"event": "run_boot_identity", "run_id": "t", "config_sha256": "ab" * 32, "ts": 0.5}]
    for s in range(1, 41):
        rows.append({"event": "trainer_step", "step": s, "value_loss": 1 / s, "policy_loss": 2.0 + (s % 3) / 10,
                     "loss": 2.0 + 1 / s, "grad_norm": 3.0 + (s % 5), "lr": 1e-3, "ts": float(s),
                     "edges": 100 * s, "caps_max_edges": 8000, "nodes": 30 * s, "caps_max_nodes": 4000,
                     "microbatches": 2})
        if s % 2 == 0:
            rows.append({"event": "iteration_complete", "step": s, "games_per_hour": 100.0 + s,
                         "positions_per_hour": 5000.0, "steps_per_hour": 90.0, "sims_per_sec": 1.5,
                         "buffer_size": 10 * s, "buffer_capacity": 1000, "games_total": s,
                         "gumbel_alpha_full": {"rows": s, "graph_rows": 100 * s, "per_1000": 10.0}, "ts": float(s)})
            rows.append({"event": "game_complete", "moves": 20 + s, "winner": s % 3 - 1, "ts": float(s),
                         "terminal_reason": "ply_cap" if s % 10 == 0 else "six_in_a_row", "moves_list": ["x"],
                         "game_id_byte_hash": f"h{s % 7}"})
        if s % 10 == 0:
            rid = f"r{s // 10:06d}_{s}"
            rows.append({"event": "eval_round_started", "round_id": rid, "step": s, "ts": float(s)})
            rows.append({"event": "eval_round_complete", "round_id": rid, "step": s, "wall_sec": 30.0,
                         "games_total": 56 if s != 30 else None, "promoted": (s == 20) or None,
                         "wr_sealbot": (s / 100) if s != 30 else None, "wr_sealbot_ci_lower": 0.0,
                         "wr_sealbot_ci_upper": 0.3, "ts": float(s) + 0.5})
            rows.append({"event": "monitor_gates", "step": s, "gates": {"g": {"checks": 1, "fires": 0}},
                         "ts": float(s)})
    rows.append({"event": "disk_free", "ts": 41.0, "disk_free_gb": 80.0})
    logs.mkdir(exist_ok=True)
    path = logs / "events_t_seg0001.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (logs / "eval_ladder_state.json").write_text(json.dumps({"sealbot_d5": {
        "name": "sealbot_d5", "status": "active", "consec": 0,
        "history": [{"round_idx": i, "games": 32, "wr": i / 10} for i in (1, 2, 4)]}}), encoding="utf-8")
    return path


def _assert_parity(mods, events_path: Path, run_id: str) -> None:
    ladder_path = events_path.with_name("eval_ladder_state.json")
    ladder_arg = ladder_path if ladder_path.is_file() else None
    rec = mods["reader"].load_record(events_path, ladder_arg)
    snap = mods["events"].EventTail(events_path.parent, run_id).poll()
    for name, x, y in _SERIES:
        assert snap.series(name, x, y).pairs() == rec.series(name, x, y), (name, y)
    assert snap.wall_hours() == rec.wall_hours()
    assert dict(snap.dropped_fields) == rec.dropped_fields
    for name in _WHOLE:
        assert snap.rows(name) == rec.rows(name), name
    for name in ("iteration_complete", "trainer_step"):
        assert (snap.last(name) or {}).get("step") == (rec.last(name) or {}).get("step"), name
    games = rec.rows("game_complete")
    assert snap.games.count == len(games)
    window = mods["health"].ply_cap_terms(rec)[1]
    assert snap.games.cap_windowed(window) == mods["health"].ply_cap_windowed(games, window)
    assert len(snap.games.hashes) == len([g for g in games if g.get("game_id_byte_hash")])
    state, _ = mods["ladder"].load_ladder(ladder_arg)
    new, old = mods["ladder"].rung_series(snap, state), mods["strength"].rung_series(rec)
    assert set(new) == set(old)
    for rung in old:
        assert ([(p.step, p.wr, p.games, p.ci, p.wilson, p.promoted, p.broken) for p in new[rung]]
                == [(p.step, p.wr, p.games, p.ci, p.wilson, p.promoted, p.broken) for p in old[rung]]), rung
    def _fields(reading):
        return None if reading is None else (reading.state, reading.step, reading.round_id)
    assert (tuple(_fields(r) for r in mods["ladder"].promotion_reading(snap))
            == tuple(_fields(r) for r in mods["strength"].promotion_reading(rec)))


def test_the_readers_give_the_dashboards_numbers_on_a_full_shaped_record(mods, tmp_path):
    _assert_parity(mods, _synthetic(tmp_path / "logs"), "t")


def test_the_readers_give_the_dashboards_numbers_on_the_run6_mirror_record(mods):
    events = os.environ.get("MANTIS_DASH_FIXTURE_EVENTS")
    if not events or not Path(events).is_file():
        pytest.skip("MANTIS_DASH_FIXTURE_EVENTS is unset or absent — the run6 record is never a fixture in git (R7)")
    path = Path(events)
    _assert_parity(mods, path, path.name.split("_seg")[0].removeprefix("events_"))
