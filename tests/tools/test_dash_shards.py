"""The dash shard index: a closed shard read once, the open one re-tailed, a game by one seek, bounded pages with a stable cursor."""
from __future__ import annotations

import importlib
import json

import pytest

from _dash_record import game, write_shard


@pytest.fixture(scope="module")
def shards(dash):
    return importlib.import_module("dash.readers.shards")


def _games(n: int, prefix: str = "g", **over) -> list[dict]:
    return [game(f"{prefix}{i}", [[0, 0]] * (5 + i), **over) for i in range(n)]


def test_a_closed_shard_is_read_once_and_marked_closed(shards, tmp_path):
    games = tmp_path / "games"
    path = write_shard(games, "r1", 1, "2026100510", _games(3))
    index = shards.ShardIndex(games, "r1")
    index.poll()
    assert index.shards[0].closed and index.total == 3
    path.write_bytes(path.read_bytes() + b'{"game_id": "late"}\n')
    index.poll()
    assert index.total == 3, "a closed shard is never re-read"


def test_the_open_shard_is_re_tailed_and_a_torn_line_is_held(shards, tmp_path):
    games = tmp_path / "games"
    path = write_shard(games, "r1", 1, "2026100510", _games(2), closed=False)
    index = shards.ShardIndex(games, "r1")
    index.poll()
    line = json.dumps(game("late", [[1, 1]]))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line[:15])
    index.poll()
    assert index.total == 2 and not index.shards[0].closed
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line[15:] + "\n")
    index.poll()
    assert index.total == 3 and index.fetch("late")["moves"] == [[1, 1]]


def test_a_game_is_fetched_whole_by_one_seek(shards, tmp_path):
    games = tmp_path / "games"
    write_shard(games, "r1", 1, "2026100510", _games(3))
    write_shard(games, "r1", 2, "2026100511", _games(2, prefix="h"))
    index = shards.ShardIndex(games, "r1")
    index.poll()
    assert index.fetch("h1")["plies"] == 6 and index.fetch("g2")["game_id"] == "g2"
    assert index.fetch("nope") is None


def test_a_page_is_a_window_with_the_filters_total_newest_first(shards, tmp_path):
    games = tmp_path / "games"
    write_shard(games, "r1", 1, "2026100510", _games(5))
    index = shards.ShardIndex(games, "r1")
    index.poll()
    page = index.page(None, 2)
    assert [r["id"] for r in page.rows] == ["g4", "g3"] and page.total == 5 and page.next_cursor is not None
    rest = index.page(page.next_cursor, 10)
    assert [r["id"] for r in rest.rows] == ["g2", "g1", "g0"] and rest.next_cursor is None


def test_the_cursor_is_stable_while_the_open_shard_grows(shards, tmp_path):
    games = tmp_path / "games"
    path = write_shard(games, "r1", 1, "2026100510", _games(4), closed=False)
    index = shards.ShardIndex(games, "r1")
    index.poll()
    first = index.page(None, 2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(g) + "\n" for g in _games(3, prefix="new")))
    index.poll()
    second = index.page(first.next_cursor, 2)
    assert [r["id"] for r in second.rows] == ["g1", "g0"], "the window continues where it was, not shifted by new games"
    assert second.total == 7


@pytest.mark.parametrize(("where", "expected"), [
    ({"channel": "promotion"}, ["e1", "e0"]),
    ({"with_search": True}, ["e1", "s0"]),
    ({"winner": "p2"}, ["e1"]),
    ({"winner": "cap"}, ["c0"]),
])
def test_the_filters_admit_only_their_games(shards, tmp_path, where, expected):
    games = tmp_path / "games"
    rows = [game("s0", [[0, 0]], stats=[{"ply": 0, "root_value": 0.1, "visits": []}]),
            game("c0", [[0, 0]], result="draw", termination="ply_cap"),
            game("e0", [[0, 0]], channel="promotion", stats=None),
            game("e1", [[0, 0]], channel="promotion", result="p2", stats=[{"ply": 0, "by": "candidate"}])]
    write_shard(games, "r1", 1, "2026100510", rows)
    index = shards.ShardIndex(games, "r1")
    index.poll()
    page = index.page(None, 50, where=shards.Filter(**where))
    assert [r["id"] for r in page.rows] == expected and page.total == len(expected)


def test_longest_and_shortest_sort_by_stones(shards, tmp_path):
    games = tmp_path / "games"
    write_shard(games, "r1", 1, "2026100510", [game("a", [[0, 0]] * 9), game("b", [[0, 0]] * 3),
                                               game("c", [[0, 0]] * 20)])
    index = shards.ShardIndex(games, "r1")
    index.poll()
    assert [r["id"] for r in index.page(None, 5, sort="longest").rows] == ["c", "a", "b"]
    assert [r["id"] for r in index.page(None, 5, sort="shortest").rows] == ["b", "a", "c"]


def test_an_unknown_cursor_restarts_the_window(shards, tmp_path):
    games = tmp_path / "games"
    write_shard(games, "r1", 1, "2026100510", _games(3))
    index = shards.ShardIndex(games, "r1")
    index.poll()
    assert index.page("x.y", 1).rows[0]["id"] == "g2"


def test_every_new_game_reaches_the_observers_once(shards, tmp_path):
    games = tmp_path / "games"
    path = write_shard(games, "r1", 1, "2026100510", _games(2), closed=False)
    seen: list[str] = []
    index = shards.ShardIndex(games, "r1", observers=(lambda g: seen.append(g["game_id"]),))
    index.poll()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(game("g2", [[0, 0]])) + "\n")
    index.poll()
    index.poll()
    assert seen == ["g0", "g1", "g2"]


def test_a_light_row_keeps_absent_facts_absent_and_the_records_minus_one_step(shards, tmp_path):
    games = tmp_path / "games"
    write_shard(games, "r1", 1, "2026100510", [game("a", [[0, 0]], step=-1, worker_id=3)])
    index = shards.ShardIndex(games, "r1")
    index.poll()
    row = index.page(None, 1).rows[0]
    assert row["step"] == -1 and row["w"] == 3 and "rung" not in row and row["stats"] is False
    assert index.channels() == ["selfplay"]


def test_a_malformed_game_is_skipped_once_and_never_doubles_the_index(shards, tmp_path):
    games = tmp_path / "games"
    path = write_shard(games, "r1", 1, "2026100510", _games(2), closed=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"game_id": "bad", "moves": 7, "plies": "x"}) + "\n")
    seen: list[str] = []
    index = shards.ShardIndex(games, "r1", observers=(lambda g: seen.append(g["game_id"]),))
    for _ in range(3):
        index.poll()
    assert index.total == 2 and seen == ["g0", "g1"] and index.shards[0].skipped == 1
