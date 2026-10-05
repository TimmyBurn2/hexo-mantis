"""The copy pass's mechanisms: axis ticks, signed zero, the warnings sentence, each net's turn marks, nets by run, compared values."""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

from _dash_record import (game, game_rows, iteration_rows, segment_start, six_in_a_row_for_p1, trainer_rows, write_config,
                          write_heartbeat, write_segment, write_shard)


@pytest.fixture(scope="module")
def svg(dash):
    return importlib.import_module("dash.views.svg")


@pytest.mark.parametrize(("lo", "hi"), [(0.74, 0.79), (-0.012, 0.05), (0.47, 0.53), (39.0, 41.0), (-3.0, -1.0), (0.0, 1e-3)])
def test_an_axis_has_at_least_two_ticks_each_printed_exactly(svg, lo, hi):
    ticks = svg.ticks(lo, hi, 4)
    labels = svg.shared_decimals(ticks)
    assert len(ticks) >= 2 and len(set(labels)) == len(labels)
    assert all(float(label.replace("−", "-").replace(" ", "")) == pytest.approx(v) for v, label in zip(ticks, labels, strict=True))


def test_a_difference_that_rounds_to_zero_carries_no_sign(dash):
    fmt = importlib.import_module("dash.views.fmt")
    assert fmt.signed(-0.001) == "0.00" and fmt.signed(0.004, 3) == "+0.004" and fmt.signed(-0.2) == "−0.20"


def test_the_warnings_sentence_names_one_kind_and_counts_several(dash):
    series = importlib.import_module("dash.views.run_series")
    spike = [{"rule": "grad_norm_spike", "step": s * 1000} for s in (19, 40)]
    assert series._alerts_said([], spike, bold=False)[0] == "no aborts. 2 gradient-norm spikes: 19k and 40k."
    many = [{"rule": "grad_norm_spike", "step": s * 1000} for s in range(1, 10)]
    said, _ = series._alerts_said([], many, bold=True)
    assert said.endswith("1k, 2k, 3k, 4k, 5k, 6k and 3 more.") and said.count(" and ") == 1
    mixed, notes = series._alerts_said([{"event": "hard_abort"}], spike + [{"rule": "loss_increase_window", "step": 30000}], bold=True)
    assert mixed.startswith("1 abort. <strong>3 warnings</strong>") and notes == "gradient-norm spike ×2, loss rise"
    assert series._alerts_said([], [], bold=False)[0] == "no aborts, no warnings."


def test_each_nets_turn_marks_a_shared_cell_split_and_never_a_dangling_slash(dash):
    text = importlib.import_module("dash.views.analyzer_text")
    two, one = text.Turn([(0, 1), (1, 1)]), text.Turn([(0, 1)], unread="refused")
    assert text.ghosts(two, one) == [[0, 1, "1", "gab"], [1, 1, "2", "ga"]]
    assert text.ghosts(one, two) == [[0, 1, "1", "gab"], [1, 1, "2", "gb"]]
    assert text.ghosts(two, text.Turn([(1, 1), (0, 1)])) == [[0, 1, "1/2", "gab"], [1, 1, "2/1", "gab"]]


def test_the_default_nets_come_from_the_games_own_run_and_are_named_by_its_label(dash):
    desk = importlib.import_module("dash.desk")
    rows = [{"id": f"{run}_{step:08d}_abcd1234", "kind": "mantis", "run_id": run, "step": step}
            for run, step in (("run_a", 3000), ("run_a", 6000), ("run_b", 3000), ("run_b", 6000))]
    fake = object.__new__(desk.Desk)
    fake.rows = lambda: rows  # type: ignore[method-assign]
    assert fake.default_pair("run_a") == ("run_a_00006000_abcd1234", "run_a_00003000_abcd1234")
    assert fake.default_pair("nobody") == ("run_b_00006000_abcd1234", "run_b_00003000_abcd1234")
    text = importlib.import_module("dash.views.analyzer_text")
    record = {"engine": {"run_id": "run_a", "step": 6000}}
    assert text.name_of(record, {"run_a": "label"}) == "label at 6k" and text.name_of(record) == "run_a at 6k"


def _run(root: Path, run_id: str, games_per_hour: float) -> Path:
    run = root / run_id
    rows = [segment_start(run_id, 1), *trainer_rows(range(1, 200)),
            *[{**r, "games_per_hour": games_per_hour} for r in iteration_rows(range(1, 200, 5))], *game_rows(2100)]
    write_segment(run / "logs", run_id, 1, rows)
    write_config(run, run_id, None)
    write_heartbeat(run / "logs", run_id, time.time())
    write_shard(run / "logs" / "games", run_id, 1, "2026100510", [game("g", six_in_a_row_for_p1())])
    return run


def test_a_compared_runs_header_carries_each_runs_latest_value_in_its_colour(dash, tmp_path):
    record = importlib.import_module("dash.readers.record")
    series = importlib.import_module("dash.views.run_series")
    snaps = [record.RunRecord(label, _run(tmp_path, label, gph)).poll() for label, gph in (("ra", 800.0), ("rb", 2300.0))]
    sentence, aside, panels = series.selfplay(snaps)
    head = panels[panels.index(">Games per hour</h3>"):][:400]
    assert '<b class="c1">800</b><b class="c2">2 300</b>' in head
    assert sentence.startswith("<strong>800 games an hour</strong>") and '<span class="c2">rb: 2 300 games an hour' in aside
