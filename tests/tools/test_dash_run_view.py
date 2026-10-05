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


def _strength(sc, cells_dir: Path, mine: list[tuple[int, float]], parent: float | None, n: int = 576):
    for step, wr in mine:
        sidecar(cells_dir, "r1", step, wr, n=n)
    if parent is not None:
        sidecar(cells_dir, "p0", 45000, parent, n=n)
    cells, _ = sc.load([cells_dir])
    return sc.strength(cells, "six", "r1", "p0_00045000_abcd1234")


@pytest.mark.parametrize(("latest", "parent", "word"), [
    (0.76, 0.59, "stronger than"), (0.45, 0.59, "weaker than"), (0.61, 0.59, "not separable from")])
def test_a_strength_word_appears_only_when_its_interval_excludes_zero(dash, strength, tmp_path, latest, parent, word):
    sc = importlib.import_module("dash.readers.sidecars")
    s = _strength(sc, tmp_path, [(3000, 0.60), (6000, latest)], parent)
    sentence, aside = strength.verdict(s)
    assert f"<strong>{word}</strong> its parent" in sentence
    assert ("yet" in sentence) == (word == "not separable from")
    assert "exclude zero" in aside


def test_the_going_forward_read_states_its_cells_and_side_of_the_line(dash, strength, tmp_path):
    sc = importlib.import_module("dash.readers.sidecars")
    _, aside = strength.verdict(_strength(sc, tmp_path, [(32201, 0.759)], 0.594))
    assert "mean of the last 1 cell is +0.77 logit" in aside and "above the line at +0.17" in aside
    assert "(1 of 4 cells so far)" in aside


def test_no_cell_is_a_stated_gap_never_an_empty_axis(dash, strength, tmp_path):
    snap = _snap(dash, _record(tmp_path))
    html = strength.panel([snap], "six", primary=True)
    assert "No Six reading in this run" in html and "<svg" not in html


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
