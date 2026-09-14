"""The shard index: closed shards indexed once, the open shard re-tailed, a game by one seek, a bounded page."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def shards(observatory):
    return importlib.import_module("observatory.readers.shards")


def _record(game_id: str, moves: list[list[int]], **over) -> dict:
    row = {"contract": "game-record-v1", "game_id": game_id, "run_id": "t", "channel": "selfplay",
           "step": 7, "step_kind": "actor", "seed": 1, "served_sims": 64, "plies": len(moves),
           "result": "p1", "termination": "six_in_a_row", "moves": moves, "worker_id": 3,
           "game_id_byte_hash": "abc"}
    row.update(over)
    return row


def _header(run_id: str, seg: int, hour: str) -> dict:
    return {"contract": "game-record-v1", "record": "shard_opened", "run_id": run_id, "segment": seg, "hour": hour}


def _shard(d: Path, seg: int, hour: str, games: list[dict], run_id: str = "t") -> Path:
    path = d / f"games_{run_id}_seg{seg:04d}_{hour}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [_header(run_id, seg, hour)] + games), encoding="utf-8")
    return path


def _close(d: Path, path: Path, games: int, run_id: str = "t", extra_bytes: int = 0) -> None:
    with (d / f"games_{run_id}_index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"record": "shard_closed", "shard": path.name, "games": games,
                             "bytes": path.stat().st_size + extra_bytes, "run_id": run_id,
                             "contract": "game-record-v1"}) + "\n")


def _dir(tmp_path: Path) -> Path:
    d = tmp_path / "games"
    d.mkdir()
    return d


def test_a_closed_shard_is_indexed_once_and_an_open_one_is_re_tailed(shards, tmp_path):
    d = _dir(tmp_path)
    closed = _shard(d, 1, "2026091400", [_record("a", [[0, 0]]), _record("b", [[0, 0], [1, 0]])])
    _close(d, closed, 2)
    opened = _shard(d, 1, "2026091401", [_record("c", [[0, 0]])])
    index = shards.ShardIndex(d, "t")
    index.poll()
    assert [s.closed for s in index.shards] == [True, False] and index.total == 3
    with opened.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_record("d", [[0, 0], [2, 0], [3, 0]])) + "\n")
    index.poll()
    assert index.total == 4 and [len(s.rows) for s in index.shards] == [2, 2]


def test_a_shard_named_closed_but_short_on_disk_is_treated_as_open(shards, tmp_path):
    d = _dir(tmp_path)
    path = _shard(d, 1, "2026091400", [_record("a", [[0, 0]])])
    _close(d, path, 5, extra_bytes=500)
    index = shards.ShardIndex(d, "t")
    index.poll()
    assert index.shards[0].closed is False


def test_fetch_reads_one_game_by_its_offset_and_equals_the_records_own_reader(shards, tmp_path):
    from mantis.monitor.game_record import read_shard
    d = _dir(tmp_path)
    games = [_record("a", [[0, 0]]),
             _record("r1_promotion_00001", [[0, 0], [1, 0]], channel="promotion", step_kind="round",
                     rung=None, phase="screen", game_index=1, colors={"candidate": 1, "opponent": -1},
                     search_stats=[{"ply": 0, "by": "candidate", "root_value": 0.1, "visits": [[1, 0, 30]]}])]
    path = _shard(d, 1, "2026091400", games)
    index = shards.ShardIndex(d, "t")
    index.poll()
    fetched = index.fetch("r1_promotion_00001")
    assert fetched is not None and fetched == read_shard(path)[0][1]
    assert index.fetch("nope") is None
    rows = index.shards[0].rows
    assert rows.ids == ["a", "r1_promotion_00001"] and list(rows.has_stats) == [0, 1]
    assert list(rows.candidate) == [0, 1] and rows.phase == [None, "screen"]


def test_a_torn_tail_in_the_open_shard_is_held_and_counted_only_when_final(shards, tmp_path):
    d = _dir(tmp_path)
    path = _shard(d, 1, "2026091400", [_record("a", [[0, 0]])])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"contract": "game-record-v1", "game_id": "b"')
    index = shards.ShardIndex(d, "t")
    index.poll()
    assert index.total == 1 and index.shards[0].held.startswith(b'{"contract"') and index.shards[0].skipped == 0
    with path.open("a", encoding="utf-8") as fh:
        fh.write(', "channel": "selfplay", "plies": 1, "result": "p2", "termination": "six_in_a_row",'
                 ' "moves": [[0, 0]]}\n')
    index.poll()
    fetched = index.fetch("b")
    assert index.total == 2 and fetched is not None and fetched["result"] == "p2"


def test_page_is_a_bounded_window_newest_first_with_a_stable_cursor_and_the_filtered_total(shards, tmp_path):
    d = _dir(tmp_path)
    first = _shard(d, 1, "2026091400",
                   [_record(f"g{i}", [[0, 0]] * (i + 1), result="p1" if i % 2 else "p2") for i in range(5)])
    _close(d, first, 5)
    _shard(d, 1, "2026091401", [_record(f"h{i}", [[0, 0]] * (i + 1)) for i in range(3)])
    index = shards.ShardIndex(d, "t")
    index.poll()
    page = index.page(None, 4)
    assert [r["id"] for r in page.rows] == ["h2", "h1", "h0", "g4"] and page.total == 8
    assert page.next_cursor == (0, 4)
    page2 = index.page(page.next_cursor, 4)
    assert [r["id"] for r in page2.rows] == ["g3", "g2", "g1", "g0"] and page2.next_cursor is None
    only_p1 = index.page(None, 10, result="p1")
    assert [r["id"] for r in only_p1.rows] == ["h2", "h1", "h0", "g3", "g1"] and only_p1.total == 5
    assert index.page(None, 10, min_plies=4).total == 2
    assert index.page(None, 10, newest_first=False).rows[0]["id"] == "g0"
    assert page.rows[0]["shard"] == 1 and page.rows[0]["pl"] == 3 and page.rows[0]["ch"] == "selfplay"
    assert "rung" not in page.rows[0] and "sims" not in page.rows[0], "absent facts stay absent"


def test_an_empty_games_dir_refuses_and_another_runs_shards_are_ignored(shards, tmp_path):
    d = _dir(tmp_path)
    _shard(d, 1, "2026091400", [_record("x", [[0, 0]], run_id="u")], run_id="u")
    with pytest.raises(shards.EmptyGameRecord):
        shards.ShardIndex(d, "t").poll()
