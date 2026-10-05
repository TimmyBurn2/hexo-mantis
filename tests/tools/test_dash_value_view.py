"""The Run view's value section: the lagged read's word past its interval, gaps at their saves, halts, the gap rule, the reach."""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

from _dash_record import game, segment_start, six_in_a_row_for_p1, trainer_rows, write_config, write_heartbeat, write_segment, write_shard


def _record(tmp_path: Path) -> Path:
    run = tmp_path / "r1"
    write_segment(run / "logs", "r1", 1, [segment_start("r1", 1), *trainer_rows(range(1, 40))])
    write_config(run, "r1", None)
    write_heartbeat(run / "logs", "r1", time.time())
    write_shard(run / "logs" / "games", "r1", 1, "2026100510",
                [game("a", six_in_a_row_for_p1(), stats=[{"ply": 0, "root_value": 0.2, "visits": []}])])
    return run


def _snap(dash, run: Path, *, records: Path | None = None):
    record = importlib.import_module("dash.readers.record")
    return record.RunRecord("r1", run, records).poll()


def _save(step: int, **over) -> dict:
    """A save record in the monitor's shape, with every pinned key."""
    raw = {"step": step, "saved_ts": float(step), "final": False,
           "gen": {"cf_ce": 0.56, "temperature": 1.2, "auc": 0.77, "policy_ce": 2.2,
                   "bands": {b: {"cf_ce": 0.6} for b in ("plies_0_10", "plies_11_40", "plies_41_up")}},
           "exams": {"T4_V": {"calibrated_mean": 0.3, "floor": 0.154, "holds": True}},
           "ring_bands": {"rows": {}, "misses": []}, "rates": {}, "halting_rows": [], "reported_rows": [],
           "armed_floors": [], "floors_live": True, "gap_rule": {"gap": None, "line": 0.05, "over": [], "fired": False}}
    for key, value in over.items():
        raw[key] = value
    return raw


def _records(tmp_path: Path, saves: list[dict], **files) -> Path:
    import json as _json
    root = tmp_path / "records"
    (root / "saves").mkdir(parents=True)
    for raw in saves:
        (root / "saves" / f"{raw['step']:08d}.json").write_text(_json.dumps(raw), encoding="utf-8")
    for name, body in files.items():
        (root / f"{name}.json").write_text(_json.dumps(body), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def value(dash):
    return importlib.import_module("dash.views.run_value")


def _lag(step: int, diff: float, lo: float, hi: float, gap: float = 0.01) -> dict:
    return {"step": step, "diff": diff, "ci": [lo, hi], "games": 300, "worse": diff > 0,
            "current": {"gap": {"cf_ce": gap}}}


@pytest.mark.parametrize(("diff", "lo", "hi", "word"), [(-0.011, -0.017, -0.005, "beats its lagged net by 0.011 nats on the games after its save (95 % interval 0.005 to 0.017)"),
                                                        (0.011, 0.005, 0.017, "trails its lagged net by 0.011"),
                                                        (-0.002, -0.006, 0.003, "shows no clear difference from its lagged net")])
def test_the_lagged_reads_word_appears_only_past_its_interval(dash, value, tmp_path, diff, lo, hi, word):
    root = _records(tmp_path, [_save(3000), _save(6000, lagged_of=_lag(3000, diff, lo, hi))])
    _, aside, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert word in aside


def test_the_cross_entropy_is_stated_without_a_direction_word(dash, value, tmp_path):
    root = _records(tmp_path, [_save(3000), _save(36000)])
    sentence, _, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert "down from" not in sentence and "up from" not in sentence and "(0.560 at 3k)" in sentence


def test_the_gap_is_drawn_at_the_save_it_belongs_to(dash, value, tmp_path):
    root = _records(tmp_path, [_save(3000), _save(6000, lagged_of=_lag(3000, -0.01, -0.02, -0.001, gap=0.012)),
                               _save(9000, lagged_of=_lag(6000, -0.01, -0.02, -0.001, gap=0.02))])
    snap = _snap(dash, _record(tmp_path), records=root)
    assert value.gaps(snap.records.saves) == [(3000.0, 0.012), (6000.0, 0.02)]


def test_no_temperature_and_no_exam_are_sentences_never_empty_axes(dash, value, tmp_path):
    unread = {"T4_V": {"calibrated_mean": None, "floor": 0.154, "holds": None, "not_measured": "no temperature"}}
    gen = {"cf_ce": 0.69, "temperature": None, "auc": 0.5, "policy_ce": 2.3, "bands": {}}
    root = _records(tmp_path, [_save(3000, gen=gen, exams=unread), _save(6000, gen=gen, exams=unread)])
    _, _, panels = value.section([_snap(dash, _record(tmp_path), records=root)])
    temp = panels[panels.index(">Temperature</h3>"):]
    exam = panels[panels.index(">T4_V exam</h3>"):]
    assert "Not measured at any save yet." in temp[:900] and "Not measured at any save yet." in exam[:900] and "<svg" not in exam[:900]


def test_records_given_with_no_save_read_say_so(dash, value, tmp_path):
    root = _records(tmp_path, [])
    sentence, aside, panels = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert sentence == "No save read yet." and "No save read yet." in panels and "--records" not in panels


@pytest.mark.parametrize(("halt", "phrase"), [
    ({"step": 6000, "halting_rows": ["T4_V twice"], "armed": True, "final_save": False, "signal": {"sent": True}},
     "The monitor signalled the run"),
    ({"step": 6000, "halting_rows": ["T4_V twice"], "armed": False, "final_save": False}, "was not armed to signal")])
def test_a_halt_on_record_leads_and_says_whether_the_run_was_signalled(dash, value, tmp_path, halt, phrase):
    root = _records(tmp_path, [_save(6000)], HALT=halt)
    sentence, _, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert sentence.startswith("<strong>Halt at 6\u202f000</strong>: T4_V twice.") and phrase in sentence


def test_a_gap_rule_that_fired_stays_on_the_page_after_the_gap_falls(dash, value, tmp_path):
    root = _records(tmp_path, [_save(9000)], GAP_RULE={"step": 9000, "gap": 0.06, "line": 0.05, "over": [6000, 9000],
                                                       "fired": True})
    sentence, aside, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert "Gap rule fired at 9k.</strong>" in sentence and "6k and 9k above the gap line" in aside


def test_the_reach_names_no_turn_instead_of_none(dash, value):
    horizon = importlib.import_module("dash.readers.horizon")
    curve = horizon.Curve(share=(None,) * 31, n=(0,) * 31)
    h = horizon.Horizon(curve, curve, curve, curve, None, 3, 10, 2, (0, 1), (8, 9))
    text = value._reach(h)
    assert "None" not in text and "3 turns out at 90\u202f% in the last fifth" in text and "and never in the first" in text


def test_an_armed_halt_whose_signal_was_not_sent_says_why(dash, value, tmp_path):
    halt = {"step": 6000, "halting_rows": ["T4_V twice"], "armed": True, "final_save": False,
            "signal": {"sent": False, "reason": "the run was already gone"}}
    root = _records(tmp_path, [_save(6000)], HALT=halt)
    sentence, _, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert "armed but sent no signal (the run was already gone)" in sentence
