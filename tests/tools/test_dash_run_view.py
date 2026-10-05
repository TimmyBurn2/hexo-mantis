"""The Run view: verdict words only past their interval, gaps as sentences never zeros, losses unscored, a byte-identical freeze."""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _dash_record import (
    game,
    game_rows,
    iteration_rows,
    segment_start,
    sidecar,
    six_in_a_row_for_p1,
    trainer_rows,
    write_config,
    write_heartbeat,
    write_segment,
    write_shard,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def view(dash):
    return importlib.import_module("dash.views.run")


@pytest.fixture(scope="module")
def strength(dash):
    return importlib.import_module("dash.views.run_strength")


@pytest.fixture(scope="module")
def series(dash):
    return importlib.import_module("dash.views.run_series")


def _record(tmp_path: Path, run_id: str = "r1", *, games: int = 2500, first_wins_every: int = 2, entropy: bool = True,
            parent: str | None = "p0_00045000_abcd1234", label: str | None = None) -> Path:
    run = tmp_path / (label or run_id)
    rows = [segment_start(run_id, 1), *trainer_rows(range(1, 400), entropy=entropy), *iteration_rows(range(1, 400, 5)),
            *game_rows(games, first_wins_every=first_wins_every, cap_every=50),
            {"event": "training_alert", "rule": "grad_norm_spike", "message": "grad norm 10.3", "step": 300, "ts": 1300.0},
            {"event": "periodic_checkpoint_save", "step": 300, "path": "x", "ts": 1300.0}]
    write_segment(run / "logs", run_id, 1, rows)
    write_config(run, run_id, parent)
    write_heartbeat(run / "logs", run_id, time.time())
    write_shard(run / "logs" / "games", run_id, 1, "2026100510",
                [game("a", six_in_a_row_for_p1(), stats=[{"ply": 0, "root_value": 0.2, "visits": []}])])
    return run


def _snap(dash, run: Path, *, cells: tuple[Path, ...] = (), records: Path | None = None, label: str = "r1"):
    record = importlib.import_module("dash.readers.record")
    return record.RunRecord(label, run, records, cells).poll()


def _ruled(dash, tmp_path: Path, mine: list[tuple[int, float]], parent: float | None, n: int = 576):
    cells = tmp_path / "cells"
    for step, wr in mine:
        sidecar(cells, "r1", step, wr, n=n)
    if parent is not None:
        sidecar(cells, "p0", 45000, parent, n=n)
    record = importlib.import_module("dash.readers.record")
    return record.RunRecord("r1", _record(tmp_path), None, (cells,), rule="six30_16").poll()


@pytest.mark.parametrize(("latest", "parent", "word"), [
    (0.76, 0.59, "stronger than"), (0.45, 0.59, "weaker than"), (0.61, 0.59, "not separable from")])
def test_a_strength_word_appears_only_when_its_interval_excludes_zero(dash, strength, tmp_path, latest, parent, word):
    sentence, aside = strength.verdict(_ruled(dash, tmp_path, [(3000, 0.60), (6000, latest)], parent))
    assert f"<strong>{word}</strong> its parent" in sentence
    assert ("yet" in sentence) == (word == "not separable from")
    assert "exclude zero" in aside and "(the rule)" in aside


def test_the_going_forward_read_states_its_cells_and_side_of_the_line(dash, strength, tmp_path):
    _, aside = strength.verdict(_ruled(dash, tmp_path, [(32201, 0.759)], 0.594))
    assert "mean of the last 1 cell is +0.77 logit" in aside and "above the line at +0.17" in aside
    assert "(1 of 4 cells so far)" in aside


def test_no_cell_is_a_stated_gap_never_an_empty_axis(dash, strength, tmp_path):
    _, _, html = strength.section([_snap(dash, _record(tmp_path))])
    assert "No ruler reading in this run" in html and "<svg" not in html


def test_report_only_rulers_read_in_logit_over_their_own_parent(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "p0", 45000, 0.59)
    extra = {"unit": "six455_128", "six": {"commit": "c0ffee", "net_sha256": "g455", "generation": 455, "nodes": 128}}
    sidecar(cells, "r1", 300, 0.45, suffix="six455_128.full", **extra)
    sidecar(cells, "p0", 45000, 0.30, suffix="six455_128.full", **extra)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), rule="six30_16").poll()
    _, aside, html = strength.section([snap])
    assert "Report-only: six455_128.full +0.65 logit over its parent" in aside
    assert "Every ruler, over its parent" in html and "six455_128.full, report-only" in html and "six30_16.full, the rule" in html


@pytest.mark.parametrize(("every", "expected"), [(2, "balanced sides"), (3, "the second player favoured"),
                                                 (1, "the first player favoured")])
def test_the_side_verdict_reads_balanced_only_when_half_is_inside_the_interval(dash, series, tmp_path, every, expected):
    snap = _snap(dash, _record(tmp_path, first_wins_every=every))
    sentence, _, _ = series.selfplay([snap])
    assert expected in sentence


def test_too_few_games_for_the_window_is_a_gap_not_a_share(dash, series, tmp_path):
    snap = _snap(dash, _record(tmp_path, games=300))
    sentence, aside, panels = series.selfplay([snap])
    assert "Fewer than 2" in panels and "favoured" not in sentence and "balanced" not in sentence
    assert "need" in aside


def test_losses_carry_no_verdict(dash, series, tmp_path):
    sentence, aside, panels = series.training([_snap(dash, _record(tmp_path))])
    assert not re.search(r"improv|better|worse|lower|higher", sentence)
    assert "never scored" in aside and "1 warning" in sentence and "300" in sentence


def test_an_unrecorded_series_is_a_sentence_in_place_of_a_chart(dash, series, tmp_path):
    _, _, panels = series.training([_snap(dash, _record(tmp_path, entropy=False))])
    assert "Not recorded in this run." in panels and "trainer_step.policy_entropy" in panels


def test_a_compared_run_without_a_series_shows_a_disabled_legend_entry(dash, series, tmp_path):
    a = _snap(dash, _record(tmp_path, label="a"), label="a")
    b = _snap(dash, _record(tmp_path, label="b", entropy=False), label="b")
    _, _, panels = series.training([a, b])
    entropy = panels[panels.index("Policy entropy"):]
    assert 'class="c2 off"' in entropy[:2000] and "not recorded" in entropy[:2000]


def test_no_monitor_records_is_a_stated_gap(dash, view, tmp_path):
    html = view.page([_snap(dash, _record(tmp_path))], runs=("r1",), now=time.time())
    assert "No monitor records for this run." in html


def test_the_status_line_reads_live_from_a_fresh_heartbeat_and_frozen_without_a_clock(dash, view, tmp_path):
    snap = _snap(dash, _record(tmp_path))
    assert 'class="state live"' in view.status(snap, time.time())
    assert 'class="state stopped"' in view.status(snap, time.time() + 7200)
    assert 'class="state frozen"' in view.status(snap, None)


def test_a_hostile_label_is_escaped_everywhere(dash, view, tmp_path):
    label = '<script>alert(1)</script>'
    snap = _snap(dash, _record(tmp_path), label=label)
    html = view.page([snap], runs=(label,), now=time.time())
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def _freeze(run: Path, cells: Path, out: Path, seed: str) -> bytes:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "dash.py"), "freeze", "--run", f"r1={run}",
                    "--cells", str(cells), "--out", str(out)], check=True, env=env, capture_output=True, timeout=300)
    return out.read_bytes()


def test_the_same_record_freezes_byte_identically(dash, tmp_path):
    run = _record(tmp_path)
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.70)
    sidecar(cells, "p0", 45000, 0.59)
    one, two = _freeze(run, cells, tmp_path / "a.html", "1"), _freeze(run, cells, tmp_path / "b.html", "999")
    assert one == two
    text = one.decode("utf-8")
    assert '<link rel="stylesheet"' not in text and "<style>" in text and 'class="state frozen"' in text
    assert len(one) < 300_000


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


@pytest.mark.parametrize(("diff", "lo", "hi", "word"), [(-0.011, -0.017, -0.005, "better than"),
                                                        (0.011, 0.005, 0.017, "worse than"),
                                                        (-0.002, -0.006, 0.003, "not separable from")])
def test_the_lagged_reads_word_appears_only_past_its_interval(dash, value, tmp_path, diff, lo, hi, word):
    root = _records(tmp_path, [_save(3000), _save(6000, lagged_of=_lag(3000, diff, lo, hi))])
    _, aside, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert f"reads {word} its lagged net" in aside


def test_the_cross_entropy_is_stated_without_a_direction_word(dash, value, tmp_path):
    root = _records(tmp_path, [_save(3000), _save(36000)])
    sentence, _, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert "down from" not in sentence and "up from" not in sentence and "it read 0.560 at 3k" in sentence


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
    temp = panels[panels.index("<h3>Temperature</h3>"):]
    assert "Not measured at any save yet." in temp[:600] and "No exam measured at any save yet." in panels


def test_records_given_with_no_save_read_say_so(dash, value, tmp_path):
    root = _records(tmp_path, [])
    sentence, aside, panels = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert sentence == "No save has been read yet." and "No save read yet." in panels and "--records" not in panels


@pytest.mark.parametrize(("halt", "phrase"), [
    ({"step": 6000, "halting_rows": ["T4_V twice"], "armed": True, "final_save": False, "signal": {"sent": True}},
     "the monitor signalled the run"),
    ({"step": 6000, "halting_rows": ["T4_V twice"], "armed": False, "final_save": False}, "was not armed to signal")])
def test_a_halt_on_record_leads_and_says_whether_the_run_was_signalled(dash, value, tmp_path, halt, phrase):
    root = _records(tmp_path, [_save(6000)], HALT=halt)
    sentence, _, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert sentence.startswith("<strong>A halting row fired</strong> at 6 000") and phrase in sentence


def test_a_gap_rule_that_fired_stays_on_the_page_after_the_gap_falls(dash, value, tmp_path):
    root = _records(tmp_path, [_save(9000)], GAP_RULE={"step": 9000, "gap": 0.06, "line": 0.05, "over": [6000, 9000],
                                                       "fired": True})
    _, aside, _ = value.section([_snap(dash, _record(tmp_path), records=root)])
    assert "The gap rule fired at 9 000 (saves 6k, 9k above the line)" in aside


def test_the_reach_names_no_turn_instead_of_none(dash, value):
    horizon = importlib.import_module("dash.readers.horizon")
    curve = horizon.Curve(share=(None,) * 31, n=(0,) * 31)
    h = horizon.Horizon(curve, curve, curve, curve, None, 3, 10, 2, (0, 1), (8, 9))
    text = value._reach(h)
    assert "None" not in text and "3 turns out" in text and "first fifth, no turn" in text
