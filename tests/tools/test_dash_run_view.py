# >300 justify (R8): every Run-view section is tested over one synthetic run builder (_record, _ruled, _rulers_snap);
# a split copies that builder, a second authority on what a run record looks like.
"""The Run view: verdict words only past their interval, gaps as sentences never zeros, losses unscored, a byte-identical freeze."""
from __future__ import annotations

import importlib
import json
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
            parent: str | None = "p0_00045000_abcd1234", label: str | None = None, extra: tuple[dict, ...] = ()) -> Path:
    run = tmp_path / (label or run_id)
    rows = [segment_start(run_id, 1), *trainer_rows(range(1, 400), entropy=entropy), *iteration_rows(range(1, 400, 5)),
            *game_rows(games, first_wins_every=first_wins_every, cap_every=50),
            {"event": "training_alert", "rule": "grad_norm_spike", "message": "grad norm 10.3", "step": 300, "ts": 1300.0},
            {"event": "periodic_checkpoint_save", "step": 300, "path": "x", "ts": 1300.0}, *extra]
    write_segment(run / "logs", run_id, 1, rows)
    write_config(run, run_id, parent)
    write_heartbeat(run / "logs", run_id, time.time())
    write_shard(run / "logs" / "games", run_id, 1, "2026100510",
                [game("a", six_in_a_row_for_p1(), stats=[{"ply": 0, "root_value": 0.2, "visits": []}])])
    return run


def _units(directory: Path, rule: str | None, **more: object) -> Path | None:
    """The records' units file naming `rule` (None: no file, so no rule declared)."""
    if rule is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "units.json"
    path.write_text(json.dumps({"rule": rule, **more}), encoding="utf-8")
    return path


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
    return record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll()


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


def test_a_report_only_ruler_is_a_reading_with_its_logit_over_its_parent(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "p0", 45000, 0.59)
    extra = {"unit": "six455_128", "six": {"commit": "c0ffee", "net_sha256": "g455", "generation": 455, "nodes": 128}}
    sidecar(cells, "r1", 300, 0.45, suffix="six455_128.full", **extra)
    sidecar(cells, "p0", 45000, 0.30, suffix="six455_128.full", **extra)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll()
    _, aside, html = strength.section([snap])
    assert "Six gen 455, 128 nodes" not in aside and "report-only, beats its parent, +0.65 logit" in html
    assert '<a class="reading" href="#sp-r-six455-128-full" data-key="r-six455-128-full">' in html and 'id="sp-r-six455-128-full"' in html
    assert "Report-only. " in _figure(html, "Six gen 455, 128 nodes") and "bar, parent" in _figure(html, "Six gen 30, 16 nodes")


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
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll()
    _, aside = strength.verdict(snap)
    assert "matches 2 series" in aside and "withheld" in aside and "Last " not in aside


def test_a_declared_rule_with_no_cell_keeps_the_others_report_only(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    extra = {"unit": "six455_128", "six": {"commit": "c0ffee", "net_sha256": "g455", "generation": 455, "nodes": 128}}
    sidecar(cells, "r1", 300, 0.45, suffix="six455_128.full", **extra)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll()
    sentence, aside = strength.verdict(snap)
    assert "(report-only)" in sentence and "reads six30_16, which has no cell yet" in aside


def test_a_head_to_head_gates_its_word_and_combines_both_logit_intervals(dash, strength, tmp_path):
    _, aside, html = strength.section(_rulers_snap(dash, tmp_path, compare=True))
    assert "Against r2, at the same save, 300: 76.0\u202f% against 50.0\u202f%, +1.15 logit (+0.88 to +1.43), ahead; unpaired." in aside
    assert "Head to head with r2, 1 save (unpaired)" in _figure(html, "Six gen 30, 16 nodes")


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
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, rule)).poll()]
    if compare:
        sidecar(tmp_path / "other_cells", "r2", 300, 0.5)
        snaps.append(record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other_cells",)).poll())
    return snaps


def _figure(html: str, title: str) -> str:
    start = html.index(f">{title}</h3>")
    return html[start:html.index("</figure>", start)]


def test_a_ruler_with_a_parent_or_a_second_cell_gets_a_chart_and_a_lone_cell_a_table_row(dash, strength, tmp_path):
    _, aside, html = strength.section(_rulers_snap(dash, tmp_path))
    assert html.count('<figure class="chart">') == 3 and 'id="sp-r-six455-512-full"' not in html
    oneoffs = html[html.index('id="sp-oneoffs"'):]
    assert "One-off reads" in oneoffs and "<td>Six gen 455, 512 nodes</td>" in oneoffs and "<svg" not in oneoffs
    report = _figure(html, "Six gen 455, 128 nodes")
    assert "Report-only. " in report and "bar, parent" not in report and "4 forfeits left out" in report
    assert "report-only, beats its parent, +0.71 logit" in html and "Six gen 455, 128 nodes" not in aside
    assert '<span class="v">47\u202f% at 600</span>' in html, "the row's value is the save its parent read is from"


def test_a_compared_run_with_no_cell_on_a_ruler_is_named_so_in_its_legend(dash, strength, tmp_path):
    _, _, html = strength.section(_rulers_snap(dash, tmp_path, compare=True))
    report = _figure(html, "Six gen 455, 128 nodes")
    assert "r2 (not read on this ruler)" in report and "not read on this ruler" not in _figure(html, "Six gen 30, 16 nodes")


def test_lone_cells_are_counted_in_one_reading_and_no_rule_means_no_report_only_mark(dash, strength, tmp_path):
    _, _, html = strength.section(_rulers_snap(dash, tmp_path, screens=3))
    assert '<span class="t">One-off reads</span><span class="v">4 cells</span>' in html
    _, _, bare = strength.section(_rulers_snap(dash, tmp_path / "bare", rule=None))
    assert "Report-only. " not in bare and "report-only" not in bare


def test_a_rule_switch_is_marked_its_former_rule_named_and_their_bridge_stated(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 150, 0.75)
    sidecar(cells, "r1", 300, 0.78)
    for step, wr in ((300, 0.30), (600, 0.33)):
        sidecar(cells, "r1", step, wr, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16,ladder455_n16@450")).poll()
    sentence, aside, html = strength.section([snap])
    assert "against Six gen 455, 16 nodes (the rule)" in sentence
    assert ("The rule moved here from Six gen 30, 16 nodes at 450; the 300 save reads 78\u202f% on the old ruler and "
            "30\u202f% on the new.") in aside
    assert '<line class="switch"' in html and "rule switch at 450" in html
    assert "The rule before 450. " in _figure(html, "Six gen 30, 16 nodes") and "the rule before 450" in html


def test_right_after_a_switch_the_former_rule_leads_and_the_new_one_is_named_unread_once(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 150, 0.75)
    sidecar(cells, "r1", 300, 0.78)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16,ladder455_n16@450")).poll()
    sentence, aside = strength.verdict(snap)
    assert "against Six gen 30, 16 nodes (the rule before 450)" in sentence and "Mean of the last" not in aside
    assert "The rule moved from Six gen 30, 16 nodes to ladder455_n16 at 450; the run has no cell on it yet." in aside
    assert aside.count("no cell") == 1


def test_a_rule_back_on_a_former_unit_reads_as_the_rule_with_no_former_lead_in(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    for step in (150, 300):
        sidecar(cells, "r1", step, 0.75)
    record = importlib.import_module("dash.readers.record")
    snap = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16,ladder455_n16@100,six30_16@200")).poll()
    _, _, html = strength.section([snap])
    assert "The rule. " not in html and "(the rule)" in strength.verdict(snap)[0]


def test_the_verdict_sets_the_latest_cell_against_the_first_of_the_going_forward_window(dash, strength, tmp_path):
    sentence, _ = strength.verdict(_ruled(dash, tmp_path, [(s, 0.6) for s in (3000, 6000, 9000, 12000, 15000)], 0.59))
    assert "and shows no clear difference from 6k against" in sentence


def test_a_compared_run_off_the_lead_ruler_is_read_on_the_latest_ruler_both_share(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "r1", 600, 0.31, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    sidecar(tmp_path / "other", "r2", 300, 0.50)
    record = importlib.import_module("dash.readers.record")
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16,ladder455_n16@450")).poll(),
             record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other",)).poll()]
    _, aside, html = strength.section(snaps)
    assert "Against r2 on Six gen 30, 16 nodes (it has no cell on the rule's ruler yet), at the same save, 300: 76.0\u202f%" in aside
    assert '<b class="c1">76\u202f% at 300</b><b class="c2">50\u202f% at 300</b>' in _figure(html, "Six gen 30, 16 nodes")


def test_a_lone_cell_a_compared_run_also_read_is_charted_not_listed(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "r1", 300, 0.17, suffix="six455_128.full", unit="six455_128", six={**_G455, "nodes": 128})
    sidecar(tmp_path / "other", "r2", 300, 0.26, suffix="six455_128.full", unit="six455_128", six={**_G455, "nodes": 128})
    record = importlib.import_module("dash.readers.record")
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll(),
             record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other",)).poll()]
    _, _, html = strength.section(snaps)
    assert 'id="sp-r-six455-128-full"' in html and 'id="sp-oneoffs"' not in html


def test_a_frozen_page_is_unscripted_so_every_reading_shows(dash):
    page = importlib.import_module("dash.views.page")
    served = page.render("t", page.Shell("run", ("r1",), "r1"), "")
    frozen = page.render("t", page.Shell("run", ("r1",), "r1", frozen=True), "")
    assert "classList.add('js')" in served and "classList.add('js')" not in frozen


def test_a_compared_run_read_at_several_saves_gives_every_gap_of_the_last_four(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    for step, wr in ((300, 0.76), (600, 0.78)):
        sidecar(cells, "r1", step, wr)
        sidecar(tmp_path / "other", "r2", step, wr - 0.20)
    record = importlib.import_module("dash.readers.record")
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll(),
             record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other",)).poll()]
    _, aside, _ = strength.section(snaps)
    assert "Against r2, at the 2 saves both read: +0.91, +0.94 logit, 300 to 600 (the latest +0." in aside and "ahead; unpaired." in aside


def _pair(tmp_path, mine: list[tuple[int, float]], theirs: list[tuple[int, float]], n: int = 576):
    record = importlib.import_module("dash.readers.record")
    for step, wr in mine:
        sidecar(tmp_path / "cells", "r1", step, wr, n=n)
    for step, wr in theirs:
        sidecar(tmp_path / "other", "r2", step, wr, n=n)
    return [record.RunRecord("r1", _record(tmp_path), None, (tmp_path / "cells",), ladder_file=_units(tmp_path, "six30_16")).poll(),
            record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other",)).poll()]


@pytest.mark.parametrize(("theirs", "word"), [(0.90, "behind"), (0.75, "level"), (0.50, "ahead")])
def test_the_gap_word_is_gated_by_the_interval_it_prints(dash, strength, tmp_path, theirs, word):
    _, aside, _ = strength.section(_pair(tmp_path, [(300, 0.76)], [(300, theirs)]))
    assert f"), {word}; unpaired." in aside


def test_more_than_four_shared_saves_list_the_last_four_and_say_so(dash, strength, tmp_path):
    steps = [(s, 0.70) for s in (100, 200, 300, 400, 500, 600)]
    _, aside, _ = strength.section(_pair(tmp_path, steps, [(s, 0.50) for s, _ in steps]))
    assert "at the last 4 of the 6 saves both read: +0.85, +0.85, +0.85, +0.85 logit, 300 to 600 (the latest" in aside


def test_a_compared_run_on_the_lead_ruler_at_other_saves_is_told_apart_from_none(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    sidecar(cells, "r1", 300, 0.31, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    sidecar(tmp_path / "other", "r2", 300, 0.50)
    sidecar(tmp_path / "other", "r2", 600, 0.70, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    record = importlib.import_module("dash.readers.record")
    snaps = [record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16,ladder455_n16@250")).poll(),
             record.RunRecord("r2", _record(tmp_path, "r2"), None, (tmp_path / "other",)).poll()]
    _, aside, html = strength.section(snaps)
    assert "(no save both read on the rule's ruler yet)" in aside
    assert '<b class="c2">70\u202f% at 600</b>' in _figure(html, "Six gen 455, 16 nodes")


def test_runs_that_share_no_save_say_so(dash, strength, tmp_path):
    _, aside, _ = strength.section(_pair(tmp_path, [(300, 0.76)], [(600, 0.50)]))
    assert "r2 shares no save with this run on any ruler yet." in aside


def test_the_head_to_head_table_rows_carry_both_readings_and_the_difference(dash, strength, tmp_path):
    _, _, html = strength.section(_pair(tmp_path, [(300, 0.76)], [(300, 0.50)]))
    assert "<td>300</td><td>76.0\u202f%</td><td>50.0\u202f%</td><td>+26.0 pts</td>" in html and "<td>+1.15</td>" in html


def test_two_rulers_slugging_alike_get_distinct_keys_and_titles(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.76)
    for unit in ("six455_128", "six455-128"):
        for step in (300, 600):
            sidecar(cells / unit, "r1", step, 0.4, suffix=f"{unit}.full", unit=unit, six={**_G455, "nodes": 128})
    record = importlib.import_module("dash.readers.record")
    _, _, html = strength.section([record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=_units(tmp_path, "six30_16")).poll()])
    assert 'id="sp-r-six455-128-full"' in html and 'id="sp-r-six455-128-full-2"' in html
    assert "Six gen 455, 128 nodes (six455_128.full)" in html and "Six gen 455, 128 nodes (six455-128.full)" in html


def test_a_rule_the_records_write_after_the_server_starts_is_read_at_the_next_poll(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.78)
    record = importlib.import_module("dash.readers.record")
    run = record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=tmp_path / "units" / "units.json")
    assert run.poll().rule is None
    _units(tmp_path / "units", "six30_16")
    snap = run.poll()
    assert snap.rule == "six30_16" and snap.rulers[0].rule


def test_a_unit_the_records_tag_legacy_reads_legacy_beside_the_rule(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    for step, wr in ((300, 0.30), (600, 0.33)):
        sidecar(cells, "r1", step, wr, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
        sidecar(cells, "r1", step, wr - 0.1, suffix="six455_128.full", unit="six455_128", six={**_G455, "nodes": 128})
    record = importlib.import_module("dash.readers.record")
    units = _units(tmp_path, "ladder455_n16", legacy=["six455_128"])
    _, _, html = strength.section([record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=units).poll()])
    assert "Legacy. " in _figure(html, "Six gen 455, 128 nodes") and '<span class="n">legacy</span>' in html


def test_each_save_reads_its_ruler_of_record_the_second_ruler_the_monitor_rows_its_rate_and_its_gate_round(dash, view,
                                                                                                         tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.31, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    sidecar(cells, "r1", 300, 0.57, family="strix", suffix="strix256_arena.full", unit="equal_work_arena")
    sidecar(cells, "r1", 300, 0.80)
    rounds = ({"event": "eval_round_complete", "round_id": "g3", "step": 300, "promoted": False, "wall_sec": 900,
               "ts": 1301.0},)
    base = {"final": False, "gen": {"cf_ce": 0.5401, "temperature": 1.176, "auc": 0.78, "policy_ce": 2.2},
            "exams": {"T4_V": {"calibrated_mean": 0.37, "floor": 0.154, "holds": True},
                      "DEF_V_att": {"calibrated_mean": 0.05, "floor": 0.1, "holds": False}},
            "gap_rule": {"gap": -0.0062, "line": 0.05, "over": [], "fired": False}}
    records = tmp_path / "records" / "saves"
    records.mkdir(parents=True)
    (records / "00000300.json").write_text(json.dumps({**base, "step": 300, "saved_ts": 1300.0,
                                                       "rates": {"positions_per_h": 200384.9}}), encoding="utf-8")
    (records / "00000600.json").write_text(json.dumps({**base, "step": 600, "saved_ts": 1600.0,
                                                       "rates": {"note": "NOT MEASURED: fewer than two iteration rows"}}),
                                           encoding="utf-8")
    record = importlib.import_module("dash.readers.record")
    units = _units(tmp_path, "six30_16,ladder455_n16@250", second="equal_work_arena")
    snap = record.RunRecord("r1", _record(tmp_path, extra=rounds), tmp_path / "records", (cells,), ladder_file=units).poll()
    html = view.saves(snap)
    rows = re.findall(r"<tr>(.*?)</tr>", html)
    assert re.findall(r"<th>(.*?)</th>", html) == ["save", "ruler of record", "second ruler", "GEN cf CE", "T", "gap",
                                                    "exams held", "positions/h save to save", "gate round"]
    assert rows[2] == "".join(f"<td>{c}</td>" for c in (
        "300", "31 % [27–35 %]", "57 % [53–61 %]", "0.540", "1.18", "−0.006", "1 of 2",
        "200.4k", "g3 not promoted"))
    assert rows[1] == "".join(f"<td>{c}</td>" for c in ("600", "not read", "not read", "0.540", "1.18", "−0.006",
                                                        "1 of 2", "not measured", "—"))
    assert "Six gen 455, 16 nodes" in html and "Strix 256 sims" in html, "the caption names both rulers"


def test_a_unit_naming_two_series_reads_ambiguous_in_the_saves_table_never_one_of_them(dash, view, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.31, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    sidecar(cells, "r1", 600, 0.35, suffix="ladder455_n16.full", unit="ladder455_n16",
            six={**_G455, "nodes": 16, "commit": "newpin"})
    records = tmp_path / "records" / "saves"
    records.mkdir(parents=True)
    for step in (300, 600):
        (records / f"{step:08d}.json").write_text(json.dumps({"step": step, "gen": {}, "exams": {}, "rates": {}}),
                                                   encoding="utf-8")
    record = importlib.import_module("dash.readers.record")
    units = _units(tmp_path, "ladder455_n16")
    snap = record.RunRecord("r1", _record(tmp_path), tmp_path / "records", (cells,), ladder_file=units).poll()
    rows = re.findall(r"<tr>(.*?)</tr>", view.saves(snap))
    assert all("<td>ambiguous (2 series)</td>" in row for row in rows[1:])
    by_name = _units(tmp_path, "ladder455_n16.full")
    snap = record.RunRecord("r1", _record(tmp_path), tmp_path / "records", (cells,), ladder_file=by_name).poll()
    rows = re.findall(r"<tr>(.*?)</tr>", view.saves(snap))
    assert all("<td>ambiguous (2 series)</td>" in row for row in rows[1:]), "a unit.arm name is matched past its hash tag"


def test_a_declared_second_ruler_with_no_cell_is_named_in_the_saves_caption(dash, view, tmp_path):
    cells = tmp_path / "cells"
    sidecar(cells, "r1", 300, 0.31, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
    records = tmp_path / "records" / "saves"
    records.mkdir(parents=True)
    (records / "00000300.json").write_text(json.dumps({"step": 300, "gen": {}, "exams": {}, "rates": {}}), encoding="utf-8")
    record = importlib.import_module("dash.readers.record")
    units = _units(tmp_path, "ladder455_n16", second="equal_work_arena")
    snap = record.RunRecord("r1", _record(tmp_path), tmp_path / "records", (cells,), ladder_file=units).poll()
    assert "the second ruler equal_work_arena." in view.saves(snap)



def test_the_declared_second_ruler_reads_as_such_never_report_only(dash, strength, tmp_path):
    cells = tmp_path / "cells"
    for step, wr in ((300, 0.30), (600, 0.33)):
        sidecar(cells, "r1", step, wr, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, "nodes": 16})
        sidecar(cells, "r1", step, wr + 0.2, family="strix", suffix="strix256_arena.full", unit="equal_work_arena")
    record = importlib.import_module("dash.readers.record")
    units = _units(tmp_path, "ladder455_n16", second="equal_work_arena")
    _, _, html = strength.section([record.RunRecord("r1", _record(tmp_path), None, (cells,), ladder_file=units).poll()])
    assert '<span class="n">the second ruler</span>' in html and "The second ruler. " in _figure(html, "Strix 256 sims")
