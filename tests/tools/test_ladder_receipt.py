"""`tools/ladder/receipt.py` (LADDER-1): one receipt per game, keyed by net hash, refused before it is written when it cannot be read back — the producer test with its planted break and the mutation self-test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_NET = "a9a46c55bd1ceadb38145fef6527254d77d37900aebea4332def55ca75f56bfc"
_STRIX = "351ed562065bed55528bf4a4bcc7da7bf48f36aec47c60ab108014d6a256447e"
_OPENING = {"book": "book_v1_s20260625_p4", "index": 3, "opening_id": "3", "relative": [[0, 0], [1, 0], [0, 1], [1, -1]]}


def _body(ladder, **overrides: Any) -> dict[str, Any]:
    r = ladder.receipt.GameReceipt(
        server="https://example.invalid", game_id="g_7Qm2Kx",
        bot={"name": f"mantis:{_NET[:8]}", "backend": "mantis", "net_hash": _NET, "display_name": "Mantis",
             "profile_id": "p_me"},
        opponent={"display_name": "Strix", "profile_id": "p_them", "elo": 1000},
        side="x", time_control={"mode": "unlimited"}, rated=False,
        sims_configured=256, search={"kind": "puct", "c_visit": 50.0}, started=1_789_800_000.0, opening=_OPENING)
    r.add_move(request_id=1, stones=3, time_limit=None, placements=((1, -1), (2, 0)), sims=256, ms=812.5,
               server_date="Sat, 19 Sep 2026 12:00:00 GMT", book_stones=1)
    r.add_move(request_id=2, stones=7, time_limit=45.0, placements=((-1, 0), (-1, 1)), sims=256, ms=790.0,
               server_date="Sat, 19 Sep 2026 12:00:03 GMT", book_stones=0)
    body = r.finish(winner="o", reason="six-in-a-row", finished=1_789_800_009.5, finished_game={
        "startedAt": 1789800000100, "finishedAt": 1789800009400, "moveCount": 11,
        "gameResult": {"reason": "six-in-a-row", "winningPlayerId": "b2", "abortedByPlayerId": None},
        "players": [{"playerId": "a1", "profileId": "p_me", "displayName": "Mantis", "elo": 1000, "eloChange": None, "isBot": True},
                    {"playerId": "b2", "profileId": "p_them", "displayName": "Strix", "elo": 1000, "eloChange": None, "isBot": True}],
        "moves": [{"moveNumber": i + 1, "playerId": "a1" if i in (0, 3, 4, 7, 8) else "b2",
                   "x": i, "y": -1, "timestamp": 1789800000100 + 800 * i} for i in range(11)]})
    body.update(overrides)
    return body


def test_the_receipt_carries_the_packets_fields_and_the_full_move_list_in_wire_coordinates(ladder) -> None:
    body = _body(ladder)
    assert body["schema_version"] == 2
    assert body["opening"] == {**_OPENING, "off_book_at": None}
    assert [m["book_stones"] for m in body["moves"]] == [1, 0]
    assert body["bot"]["name"] == "mantis:a9a46c55" and body["bot"]["net_hash"] == _NET
    assert body["opponent"]["display_name"] == "Strix"
    assert body["sims_configured"] == 256
    assert [m["placements"] for m in body["moves"]] == [[[1, -1], [2, 0]], [[-1, 0], [-1, 1]]]
    assert [m["sims"] for m in body["moves"]] == [256, 256]
    assert body["result"] == {"winner": "o", "reason": "six-in-a-row", "outcome": "loss"}
    assert body["plies"] == 11
    # HeXO (x, y) -> htttx (q, r) is q = x + y, r = -y: the record's (3, -1) is our (2, 1).
    assert body["moves_full"][3] == [2, 1, "x"]
    assert body["moves_full"][1] == [0, 1, "o"]
    assert body["wall_sec"] == 9.5
    assert body["started_utc"] == "2026-09-19T06:40:00Z"
    assert body["server_clock"] == {"started_at_ms": 1789800000100, "finished_at_ms": 1789800009400,
                                    "first_move_date": "Sat, 19 Sep 2026 12:00:00 GMT",
                                    "last_move_date": "Sat, 19 Sep 2026 12:00:03 GMT"}
    assert body["think_ms_total"] == 1602.5


def test_a_win_and_an_abort_are_named_from_our_side(ladder) -> None:
    assert _body(ladder)["result"]["outcome"] == "loss"
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g", bot={"name": "n", "backend": "mantis", "net_hash": _NET, "display_name": "M", "profile_id": "p"},
        opponent={"display_name": "S", "profile_id": "q", "elo": None}, side="o", time_control={"mode": "unlimited"},
        rated=False, sims_configured=8, search={}, started=0.0, opening=_OPENING)
    assert r.finish(winner="o", reason="surrender", finished=1.0, finished_game=None)["result"]["outcome"] == "win"
    assert r.finish(winner=None, reason="aborted", finished=1.0, finished_game=None)["result"]["outcome"] == "aborted"


def test_without_a_finished_game_record_the_plies_are_the_stones_we_saw_plus_ours(ladder) -> None:
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g", bot={"name": "n", "backend": "strix", "net_hash": _STRIX, "display_name": "S", "profile_id": "p"},
        opponent={"display_name": "M", "profile_id": "q", "elo": 1000}, side="o", time_control={"mode": "unlimited"},
        rated=False, sims_configured=256, search={}, started=0.0, opening=_OPENING)
    r.add_move(request_id=1, stones=1, time_limit=None, placements=((1, 0), (0, 1)), sims=0, ms=5.0, server_date=None,
               book_stones=2)
    body = r.finish(winner="x", reason="six-in-a-row", finished=1.0, finished_game=None)
    assert body["plies"] is None and body["plies_seen"] == 3 and body["moves_full"] is None


def test_the_receipt_is_written_under_the_net_hash_and_reads_back(ladder, tmp_path: Path) -> None:
    body = _body(ladder)
    path = ladder.receipt.write_receipt(tmp_path, body)
    assert path == tmp_path / "receipts" / "a9a46c55" / "g_7Qm2Kx.json"
    assert ladder.receipt.read_receipt(path) == body
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("plant", [
    {"bot": {"name": "mantis:a9a46c55", "backend": "mantis", "net_hash": "not-a-hash", "display_name": "M", "profile_id": "p"}},
    {"moves": [{"request_id": 1, "stones": 3, "time_limit": None, "placements": [[1, -1]], "sims": 256, "ms": 1.0,
                "server_date": None}]},
    {"moves": [{"request_id": 1, "stones": 3, "time_limit": None, "placements": [[1, -1], [2, 0]], "sims": None,
                "ms": 1.0, "server_date": None}]},
    {"result": {"winner": "o", "reason": "six-in-a-row", "outcome": "draw"}},
    {"schema_version": 3},
    {"opening": None},
    {"opening": {"book": "book_v1_s20260625_p4", "index": 3, "opening_id": "3"}},
    {"moves": [{"request_id": 1, "stones": 3, "time_limit": None, "placements": [[1, -1], [2, 0]], "sims": 256,
                "ms": 1.0, "server_date": None, "book_stones": 3}]},
    {"moves": [{"request_id": 1, "stones": 3, "time_limit": None, "placements": [[1, -1], [2, 0]], "sims": 256,
                "ms": 1.0, "server_date": None}]},
], ids=["net_hash", "one_placement", "no_sims", "draw_outcome", "schema", "no_opening", "opening_without_relative",
        "three_book_stones", "no_book_stones"])
def test_a_planted_break_is_refused_before_anything_reaches_disk(ladder, tmp_path: Path, plant: dict[str, Any]) -> None:
    with pytest.raises(ladder.receipt.ReceiptError):
        ladder.receipt.write_receipt(tmp_path, _body(ladder, **plant))
    assert not (tmp_path / "receipts").exists()


def test_the_validator_is_what_stops_the_planted_break(ladder, tmp_path: Path, monkeypatch) -> None:
    """Mutation self-test: with the validator disarmed the same break lands on disk, so the refusal above is the validator's."""
    monkeypatch.setattr(ladder.receipt, "validate_receipt", lambda _body: None)
    path = ladder.receipt.write_receipt(tmp_path, _body(ladder, schema_version=3))
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 3


def test_a_receipt_that_no_longer_reads_back_is_refused_by_the_reader(ladder, tmp_path: Path) -> None:
    path = ladder.receipt.write_receipt(tmp_path, _body(ladder))
    path.write_text(path.read_text(encoding="utf-8").replace('"outcome": "loss"', '"outcome": "lost"'), encoding="utf-8")
    with pytest.raises(ladder.receipt.ReceiptError, match="outcome"):
        ladder.receipt.read_receipt(path)


def test_a_rejected_move_is_recorded_with_the_servers_code_and_what_we_sent(ladder) -> None:
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g", bot={"name": "n", "backend": "mantis", "net_hash": _NET, "display_name": "M", "profile_id": "p"},
        opponent={"display_name": "S", "profile_id": "q", "elo": 1000}, side="x", time_control={"mode": "unlimited"},
        rated=False, sims_configured=8, search={}, started=0.0, opening=_OPENING)
    r.add_rejection(request_id=3, placements=((0, 0), (1, 0)), code="occupied", message="That cell is already occupied.")
    body = r.finish(winner="o", reason="surrender", finished=1.0, finished_game=None)
    assert body["rejections"] == [{"request_id": 3, "placements": [[0, 0], [1, 0]], "code": "occupied",
                                   "message": "That cell is already occupied."}]
    assert _body(ladder)["rejections"] == []


def test_the_servers_racy_move_array_is_put_back_in_move_number_order(ladder) -> None:
    """Measured live 2026-09-19: the record's `moves[]` carried moveNumber 4 before 3 (async appends); `moveNumber` is right."""
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g", bot={"name": "n", "backend": "mantis", "net_hash": _NET, "display_name": "M", "profile_id": "p"},
        opponent={"display_name": "S", "profile_id": "q", "elo": 1000}, side="x", time_control={"mode": "unlimited"},
        rated=False, sims_configured=8, search={}, started=0.0, opening=_OPENING)
    body = r.finish(winner="o", reason="six-in-a-row", finished=1.0, finished_game={
        "startedAt": 0, "finishedAt": 1, "moveCount": 3,
        "gameResult": {"reason": "six-in-a-row", "winningPlayerId": "b", "abortedByPlayerId": None}, "players": [],
        "moves": [{"moveNumber": 2, "playerId": "a", "x": 0, "y": 0, "timestamp": 0},
                  {"moveNumber": 4, "playerId": "b", "x": 1, "y": -5, "timestamp": 1},
                  {"moveNumber": 3, "playerId": "b", "x": 0, "y": -2, "timestamp": 1}]})
    assert body["moves_full"] == [[0, 0, "x"], [-2, 2, "o"], [-4, 5, "o"]]


def test_the_first_off_book_stone_count_is_kept_and_later_ones_do_not_move_it(ladder) -> None:
    r = ladder.receipt.GameReceipt(
        server="s", game_id="g", bot={"name": "n", "backend": "mantis", "net_hash": _NET, "display_name": "M", "profile_id": "p"},
        opponent={"display_name": "S", "profile_id": "q", "elo": 1000}, side="x", time_control={"mode": "unlimited"},
        rated=False, sims_configured=8, search={}, started=0.0, opening=_OPENING)
    r.mark_off_book(stones=3)
    r.mark_off_book(stones=7)
    assert r.finish(winner="o", reason="surrender", finished=1.0, finished_game=None)["opening"]["off_book_at"] == 3
