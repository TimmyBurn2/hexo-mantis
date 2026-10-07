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
    (0.76, 0.59, "beats"), (0.45, 0.59, "trails"), (0.61, 0.59, "shows no clear difference from")])
def test_a_strength_word_appears_only_when_its_interval_excludes_zero(dash, strength, tmp_path, latest, parent, word):
    sentence, aside = strength.verdict(_ruled(dash, tmp_path, [(3000, 0.60), (6000, latest)], parent))
    assert f"<strong>{word}</strong> its parent" in sentence and "(the rule)" in sentence
    assert "beats" not in aside and "trails" not in aside


def test_the_going_forward_read_states_its_cells_and_side_of_the_line(dash, strength, tmp_path):
    _, aside = strength.verdict(_ruled(dash, tmp_path, [(32201, 0.759)], 0.594))
    assert "Mean of the last 1 cell: +0.77 logit over the parent" in aside and "above the +0.17 bar" in aside
    assert "The rule needs 4 cells; 1 is read." in aside


def test_no_cell_is_a_stated_gap_never_an_empty_axis(dash, strength, tmp_path):
    _, _, html = strength.section([_snap(dash, _record(tmp_path))])
    assert "No ruler reading for this run" in html and "<svg" not in html


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
    assert "six455_128.full at 300: 45.0\u202f%, parent 30.0\u202f%, +0.65 logit over the parent." in aside
    assert "Win rate against six30_16.full" in html and "Win rate against six455_128.full" in html and "Report-only. " in html
    assert "Every ruler against its parent" in html and "six455_128.full, report-only" in html and "six30_16.full, the rule" in html


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
    assert "not strength" in aside and "1 gradient-norm spike" in sentence and "300" in sentence


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


def test_an_ambiguous_rule_is_stated_and_its_going_forward_read_withheld(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "r1", 350, 0.50, six={"commit": "newpin", "net_sha256": "beef", "generation": 30, "nodes": 16})
    sidecar(cells, "p0", 45000, 0.59)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), rule="six30_16").poll()
    _, aside = strength.verdict(snap)
    assert "matches 2 series" in aside and "withheld" in aside and "Last " not in aside


def test_a_declared_rule_with_no_cell_keeps_the_others_report_only(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    extra = {"unit": "six455_128", "six": {"commit": "c0ffee", "net_sha256": "g455", "generation": 455, "nodes": 128}}
    sidecar(cells, "r1", 300, 0.45, suffix="six455_128.full", **extra)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), rule="six30_16").poll()
    sentence, aside = strength.verdict(snap)
    assert "(report-only)" in sentence and "reads six30_16, which has no cell yet" in aside


def test_the_logit_whisker_carries_the_parents_interval_too(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.16, n=288, wr_ci_lower=0.12, wr_ci_upper=0.20)
    sidecar(cells, "p0", 45000, 0.111, n=288, wr_ci_lower=0.08, wr_ci_upper=0.145)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), rule="six30_16").poll()
    blob = re.findall(r'<script type="application/json" class="xh">(.*?)</script>', strength._logit_panel(snap, 400.0, ""))[0]
    import json as _json
    (pt,) = _json.loads(blob)["series"][0]["pts"]
    assert pt[2] < 0 < pt[3], "the difference's interval contains zero, as separation() says"


def test_no_script_derives_a_turn_or_an_owner():
    web = REPO_ROOT / "tools" / "dash" / "web"
    for name in ("board.js", "games.js", "analyzer.js"):
        code = (web / name).read_text(encoding="utf-8")
        assert "turnOf" not in code and ">> 1" not in code and "% 2" not in code, name


_G455 = {"commit": "c0ffee", "net_sha256": "g455", "generation": 455}


def _rulers_snap(dash, tmp_path, *, rule: str | None = "six30_16", compare: bool = False, screens: int = 0):
    """The rule's ruler and a report-only six455_128 (each with a parent), a parentless two-cell six455_256, a lone screen cell."""
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "p0", 45000, 0.59)
    for step, wr in ((300, 0.45), (600, 0.47)):
        sidecar(cells, "r1", step, wr, suffix="six455_128.full", unit="six455_128", six={**_G455, "nodes": 128},
                six_findings={"count": 4} if step == 600 else None)
    sidecar(cells, "p0", 45000, 0.30, suffix="six455_128.full", unit="six455_128", six={**_G455, "nodes": 128})
    for step in (300, 600):
        sidecar(cells, "r1", step, 0.20, suffix="six455_256.full", unit="six455_256", six={**_G455, "nodes": 256})
    sidecar(cells, "r1", 300, 0.10, n=128, suffix="six455_512.full", unit="six455_512", six={**_G455, "nodes": 512})
    for k in range(screens):
        sidecar(cells, "r1", 300, 0.1, n=128, suffix=f"six{k}_8.full", unit=f"six{k}_8", six={**_G455, "nodes": 8 + k})
    record = importlib.import_module("dash.readers.record")
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), rule=rule).poll()]
    if compare:
        sidecar(tmp_path / "other_cells", "r2", 300, 0.5)
        snaps.append(record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other_cells",)).poll())
    return snaps


def _figure(html: str, title: str) -> str:
    start = html.index(f">{title}</h3>")
    return html[start:html.find("<figure", start)]


def test_each_ruler_with_a_parent_or_a_line_gets_its_own_win_rate_chart_and_a_lone_cell_does_not(dash, strength, tmp_path):
    sentence, aside, html = strength.section(_rulers_snap(dash, tmp_path))
    assert html.count(">Win rate against ") == 3 and ">Win rate against six455_512.full</h3>" not in html
    report = _figure(html, "Win rate against six455_128.full")
    assert "Report-only. " in report and "bar, parent" not in report and "Beats or trails" not in report
    assert "bar, parent" in _figure(html, "Win rate against six30_16.full")
    assert "No parent cell on six455_256.full, six455_512.full, so not drawn here." in html
    assert "six455_128.full at 600: 46.6\u202f% (4 Six forfeits left out), parent 30.0\u202f%" in aside


def test_a_compared_run_with_no_cell_on_a_ruler_is_named_so_in_its_legend(dash, strength, tmp_path):
    _, _, html = strength.section(_rulers_snap(dash, tmp_path, compare=True))
    report = _figure(html, "Win rate against six455_128.full")
    assert "r2 (not read on this ruler)" in report and "not read on this ruler" not in _figure(html, "Win rate against six30_16.full")


def test_many_parentless_rulers_are_counted_and_no_rule_means_no_report_only_mark(dash, strength, tmp_path):
    _, _, html = strength.section(_rulers_snap(dash, tmp_path, screens=3))
    assert "5 rulers have no parent cell, so are not drawn here." in html
    _, _, bare = strength.section(_rulers_snap(dash, tmp_path / "bare", rule=None))
    assert "Report-only. " not in bare



def test_a_rule_switch_is_marked_its_former_rule_named_and_their_bridge_stated(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 150, 0.75)
    sidecar(cells, "r1", 300, 0.78)
    for step, wr in ((300, 0.30), (600, 0.33)):
        sidecar(cells, "r1", step, wr, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    record, sc = importlib.import_module("dash.readers.record"), importlib.import_module("dash.readers.sidecars")
    rule, switches = sc.parse_rule("six30_16,ladder455_n16@450")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), rule=rule, switches=switches).poll()
    sentence, aside, html = strength.section([snap])
    assert "on ladder455_n16.full (the rule)" in sentence
    assert "The rule moved from six30_16 to ladder455_n16 at 450; bridge 300: 78\u202f% on six30_16, 30\u202f% on ladder455_n16." in aside
    assert "rule six30_16 → ladder455_n16 at 450" in html and "Win rate against six30_16.full" in html
    assert "The rule before 450. " in _figure(html, "Win rate against six30_16.full") and "the rule before 450" in html
