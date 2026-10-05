"""The Analyzer view: the verdict from tactics and each net's first choice, A vs B sums to zero, a variation drops the game's search."""
from __future__ import annotations

import importlib
import time

import pytest

from _dash_record import game, segment_start, six_in_a_row_for_p1, trainer_rows, write_config, write_heartbeat, write_segment, write_shard


@pytest.fixture(scope="module")
def text(dash):
    return importlib.import_module("dash.views.analyzer_text")


def _record(policy, value=0.2, cls="block", cells=((-6, -1), (-2, -5)), to_move="p2", run_id="r", step=45000, search=None):
    return {"engine": {"id": f"{run_id}_{step}", "run_id": run_id, "step": step},
            "position": {"to_move": to_move}, "raw": {"value": value, "policy": [[q, r, p] for (q, r), p in policy.items()]},
            "search": search or {"absent": "sims=0 (raw only)"},
            "tactics": {"class": cls, "cells": [list(c) for c in cells]}}


A = {(-6, -1): 0.6, (-2, -5): 0.1, (0, 0): 0.3}
B = {(-6, -1): 0.2, (-4, 2): 0.5, (0, 0): 0.3}


def test_a_vs_b_sums_to_zero_over_the_union_of_cells(text):
    panel = text.compose(_record(A), _record(B, step=3000), None, None, None)
    cells = {(d[0], d[1]) for d in panel["lens"]["diff"]}
    assert cells == set(A) | set(B) and sum(d[2] for d in panel["lens"]["diff"]) == pytest.approx(0.0)


def test_the_verdict_names_the_forced_move_and_what_each_net_does(text):
    both = text.compose(_record(A), _record({**B, (-6, -1): 0.6}, step=3000), None, None, None)
    assert both["verdict"].startswith("<strong>Dark must block</strong> Light's four this turn, at (−6, −1) or (−2, −5).")
    assert "Both nets block with their first choice, (−6, −1)." in both["verdict"]
    split = text.compose(_record(A), _record(B, step=3000), None, None, None)
    assert "r at 45k does block" in split["verdict"] and "r at 3k does not block with its first choice, (−4, 2)" in split["verdict"]


def test_the_games_search_rides_only_on_the_games_line(text):
    entry = {"v": 0.5, "top": [[-6, -1, 30], [0, 0, 10]], "n": 40}
    on = text.compose(_record(A), None, entry, (-6, -4), (-6, -1))
    assert on["search_source"] == "the game's recorded search" and on["chances"][-1]["label"] == "game"
    assert on["game_line"] == "In the game Dark played (−6, −4) and then (−6, −1) (a block)."
    off = text.compose(_record(A), None, None, None, None)
    assert off["lens"]["search"] == [] and off["search_source"] == "" and len(off["chances"]) == 1


def test_the_analyzers_own_search_counts_only_visited_cells(text):
    search = {"sims": 64, "children": [[-6, -1, 0.5, 40, 0.1], [0, 0, 0.3, 24, 0.0], [5, 5, 0.01, 0, 0.0]]}
    panel = text.compose(_record(A, search=search), None, None, None, None)
    assert {(c[0], c[1]) for c in panel["lens"]["search"]} == {(-6, -1), (0, 0)}
    assert panel["search_source"] == "the analyzer's search, 64 sims"


def test_light_chance_is_read_from_the_movers_value(text):
    panel = text.compose(_record(A, value=0.6, to_move="p2"), None, None, None, None)
    assert panel["chances"][0]["light"] == pytest.approx(0.2)


def test_a_terminal_and_a_quiet_position_say_so(text):
    assert "has six in a row" in text.compose({**_record(A, cls="terminal", cells=()), "position": {"to_move": "p1", "winner": "p1"}},
                                              None, None, None, None)["verdict"]
    assert text.compose(_record(A, cls="quiet", cells=()), None, None, None, None)["verdict"] == "Nothing is forced this turn."


def _hub(tmp_path):
    serve = importlib.import_module("dash.serve")
    record = importlib.import_module("dash.readers.record")
    run = tmp_path / "r1"
    write_segment(run / "logs", "r1", 1, [segment_start("r1", 1), *trainer_rows(range(1, 5))])
    write_config(run, "r1", None)
    write_heartbeat(run / "logs", "r1", time.time())
    stats = [{"ply": 3, "root_value": 0.4, "visits": [[1, 0, 9]]}]
    write_shard(run / "logs" / "games", "r1", 1, "2026100510", [game("g1", six_in_a_row_for_p1(), stats=stats)])
    hub = serve.Hub([record.RunRecord("r1", run)])
    hub.poll_once()
    return hub


def test_a_variation_off_the_line_drops_the_games_search(dash, tmp_path):
    desk = importlib.import_module("dash.desk")
    hub = _hub(tmp_path)
    line = [tuple(m) for m in six_in_a_row_for_p1()]
    on = desk.context(hub, "r1", "g1", line[:3])
    assert on.on_line and on.off == 0 and on.entry["top"] == [[1, 0, 9]] and on.nxt == line[3] and on.second == line[4]
    off = desk.context(hub, "r1", "g1", line[:2] + [(9, 9)])
    assert not off.on_line and off.off == 1 and off.entry is None and off.nxt is None


def test_without_engines_the_page_says_how_to_start_and_the_api_refuses(dash, tmp_path):
    desk = importlib.import_module("dash.desk")
    hub = _hub(tmp_path)
    page = importlib.import_module("dash.analyzer_routes").page(None)(hub, ["analyzer"], {"run": ["r1"], "g": ["g1"], "ply": ["4"]})
    html = page.body.decode("utf-8")
    assert page.status == 200 and "No engines loaded." in html and "--checkpoints" in html
    assert len(__import__("re").findall(r'class="s[12]"', html.split('<svg class="board"')[1].split("</svg>")[0])) == 4
    out = importlib.import_module("dash.analyzer_routes").post(hub, None)("/api/read", b"{}")
    assert out.status == 503


def test_a_real_engine_reads_through_the_desk_path(dash, dash_mantis_engine, positions):
    analysis = importlib.import_module("dash.engine.analysis")
    rec = analysis.analyze(dash_mantis_engine, positions["CHECK"], 0)
    panel = importlib.import_module("dash.views.analyzer_text").compose(rec, None, None, None, None)
    assert panel["tactics"]["cls"] == "block" and panel["verdict"].startswith("<strong>Dark must block</strong>")
    assert panel["rows"] and all(r[4] is None for r in panel["rows"])


def test_the_engine_layer_loads_only_on_the_first_analyzer_request(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = ("import importlib, sys\nfrom pathlib import Path\nfrom mantis.util.loadpkg import load_tools_package\n"
              "load_tools_package('dash', repo_root=Path(sys.argv[1]))\n"
              "desk = importlib.import_module('dash.desk')\nimportlib.import_module('dash.cli')\n"
              "lazy = desk.LazyDesk([], strix=False, device='cpu', threads=1)\nbefore = 'torch' in sys.modules\n"
              "lazy(); after = 'torch' in sys.modules\nlazy.close()\nprint(before, after)\n")
    out = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.split() == ["False", "True"], out.stderr[-2000:]


def test_a_read_with_a_list_field_is_refused_by_name(dash, tmp_path):
    desk = importlib.import_module("dash.desk")
    hub = _hub(tmp_path)

    class Fake:
        analyst = None

    for body in (b'{"a": "x", "run": ["r1"]}', b'{"a": "x", "b": [1]}', b'{"b": "y"}', b'{"a": "x", "sims": -1}'):
        out = importlib.import_module("dash.analyzer_routes").post(hub, lambda: Fake())("/api/read", body)
        assert out.status == 400, body


def test_two_nets_with_one_name_are_told_apart(text):
    a = _record(A)
    b = _record(B)
    a["engine"]["sha8"], b["engine"]["sha8"] = "aaaa1111", "bbbb2222"
    panel = text.compose(a, b, None, None, None)
    assert panel["a"] != panel["b"] and "aaaa1111" in panel["a"] and "does not block" in panel["verdict"]
