"""The Games view: the position at ?ply=N without script, sentences from the record, a cell outside the visits is a dash, the routes."""
from __future__ import annotations

import importlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _dash_record import game, segment_start, six_in_a_row_for_p1, trainer_rows, write_config, write_heartbeat, write_segment, write_shard

_STATS = [{"ply": 0, "root_value": 0.0, "visits": [[0, 0, 5], [1, 1, 2]]},
          {"ply": 1, "root_value": 0.2, "visits": [[0, 5, 9], [9, 9, 1]]},
          {"ply": 3, "root_value": 0.9, "visits": [[7, 7, 4]]},
          {"ply": 4, "root_value": -0.5, "visits": [[8, 8, 3], [2, 0, 1]]}]


@pytest.fixture(scope="module")
def gv(dash):
    return importlib.import_module("dash.views.games")


@pytest.fixture(scope="module")
def text(dash):
    return importlib.import_module("dash.views.games_text")


@pytest.fixture(scope="module")
def GameView(dash):
    return importlib.import_module("dash.readers.games").GameView


def _game(**over):
    arms = ["opening", "full", "opening"] + ["full", "fast"] * 4 + ["full"]
    return game("g1", six_in_a_row_for_p1(), stats=_STATS, move_arms=arms, move_sims=[0] + [320, 64] * 5 + [320], **over)


def test_every_position_has_its_sentences_and_the_final_one_closes_the_list(gv, GameView):
    body = gv.payload(GameView.from_record(_game()), "r1", None)
    assert len(body["pos"]) == len(body["moves"]) + 1 and body["pos"][-1]["think"] is None
    assert body["pos"][-1]["where"].startswith("<b>Final position</b>")
    assert "Light wins</strong> with six in a row on turn 7" in body["head"]
    assert body["pos"][11]["threat"].startswith("<strong>Light can win now</strong> at ") and "(5, 0)" in body["pos"][11]["threat"]


def test_a_played_cell_outside_the_recorded_visits_reads_a_dash_never_zero(text, GameView):
    view = GameView.from_record(_game())
    out = text.thought(view, 3, set(), False)
    assert "outside the recorded visits (—)" in out["text"] and "0 %" not in out["text"]


def test_the_played_cell_and_its_share_and_the_second_stone_flag(text, GameView):
    view = GameView.from_record(_game())
    first = text.thought(view, 1, {(0, 5)}, False)
    assert first["text"] == "Full search, 10 visits. It played (0, 5), its most-visited move."
    assert first["cands"][0] == [0, 5, 0.9, ["played", "blocks"]] and first["light"] == pytest.approx(0.4)
    assert first["second"] is False
    second = text.thought(view, 4, set(), False)
    assert second["second"] is True and "with 25\u202f% of the visits" in second["text"]


def test_an_opening_stone_and_an_unsampled_game_say_why_there_is_no_search(text, GameView):
    view = GameView.from_record(_game())
    assert text.thought(view, 2, set(), False)["text"].startswith("An opening stone")
    unsampled = GameView.from_record(game("g2", six_in_a_row_for_p1()))
    assert "not sampled" in text.thought(unsampled, 4, set(), False)["text"]


@pytest.mark.parametrize(("step", "kind", "net"), [(-1, "actor", "before the actor's first sync"),
                                                   (36000, "actor", "r1 at 36k, the actor's copy"),
                                                   (32201, "round", "r1 at 32.2k")])
def test_the_net_that_played_is_named_as_the_record_says(text, GameView, step, kind, net):
    view = GameView.from_record(game("g", [[0, 0]], step=step, step_kind=kind))
    assert dict(text.facts(view, "r1", None))["Net"] == net


def test_the_shard_hour_reads_in_central_european_time(text):
    assert text.hour_of("games_r1_seg0003_2026100510.jsonl") == "5 Oct, the hour from 12:00 CEST"
    assert text.hour_of("nonsense.jsonl") is None


@pytest.mark.parametrize(("result", "term", "phrase"), [("p2", "six_in_a_row", "Dark wins</strong> with six"),
                                                        ("draw", "ply_cap", "No winner</strong>: the game hit the cap"),
                                                        ("unknown", "unknown", "result is unknown")])
def test_the_headline_follows_the_records_result_and_termination(text, GameView, result, term, phrase):
    view = GameView.from_record(game("g", [[0, 0], [1, 1], [2, 2]], result=result, termination=term))
    assert phrase in text.headline(view)


def _run(root: Path) -> Path:
    run = root / "r1"
    write_segment(run / "logs", "r1", 1, [segment_start("r1", 1), *trainer_rows(range(1, 30))])
    write_config(run, "r1", None)
    write_heartbeat(run / "logs", "r1", time.time())
    games = [_game(), game("e1", six_in_a_row_for_p1(), channel="promotion", colors={"candidate": 1, "opponent": -1}),
             game('x"<b>', six_in_a_row_for_p1())]
    write_shard(run / "logs" / "games", "r1", 1, "2026100510", games)
    return run


@pytest.fixture()
def server(dash, tmp_path):
    serve = importlib.import_module("dash.serve")
    routes = importlib.import_module("dash.routes")
    record = importlib.import_module("dash.readers.record")
    hub = serve.Hub([record.RunRecord("r1", _run(tmp_path))])
    hub.poll_once()
    httpd = serve.make_server("127.0.0.1", 0, hub, get_extra=routes.GET)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def test_the_position_at_a_ply_carries_exactly_that_many_stones_without_script(server):
    status, html = _get(f"{server}/run/r1/games?g=g1&ply=5")
    board = re.search(r'<svg class="board".*?</svg>', html, re.S).group(0)
    assert status == 200 and len(re.findall(r'class="s[12]"', board)) == 5
    assert "<b>Turn 4</b> of 7, Dark to place two stones" in html


def test_the_list_offers_only_the_channels_present_and_the_inline_game_parses(server):
    status, html = _get(f"{server}/run/r1/games")
    assert status == 200 and ">Gate</a>" in html and ">External</a>" not in html
    state = json.loads(re.search(r'<script type="application/json" id="state">(.*?)</script>', html, re.S).group(1))
    assert state["game"]["id"] == 'x"<b>' and 'x"<b>' not in html.replace('x\\"<b>', "")


def test_an_unknown_game_is_a_404_and_the_api_names_it(server):
    assert _get(f"{server}/run/r1/games?g=nope")[0] == 404
    status, body = _get(f"{server}/api/run/r1/game/nope")
    assert status == 404 and json.loads(body)["refused"] == "no such game"


def test_the_api_window_is_capped_and_filters_by_channel(server):
    status, body = _get(f"{server}/api/run/r1/games?n=99999&kind=promotion")
    page = json.loads(body)
    assert status == 200 and [r["id"] for r in page["rows"]] == ["e1"] and page["games"] == 3


def test_a_query_value_outside_its_set_is_dropped_never_echoed(server):
    status, html = _get(f"{server}/run/r1/games?kind=%3Cscript%3E&winner=zz&sort=sideways")
    assert status == 200 and "<script>" not in html.split("<body")[1].split('<script type="application/json"')[0]


def test_all_clears_the_kind_filter_and_a_unicode_digit_is_never_a_ply(server):
    status, html = _get(f"{server}/run/r1/games?kind=promotion")
    all_link = re.search(r'<a href="\?([^"]*)" aria-pressed="false">All</a>', html).group(1)
    assert status == 200 and "kind" not in all_link
    assert _get(f"{server}/run/r1/games?g=g1&ply=%C2%B2")[0] == 200
    assert _get(f"{server}/api/run/r1/games?n=%C2%B2")[0] == 200


def test_load_more_rows_are_the_servers_own_rows(server):
    page = json.loads(_get(f"{server}/api/run/r1/games?n=1")[1])
    assert len(page["html"]) == 1 and page["html"][0].startswith('<a class="lrow"')
