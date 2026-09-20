"""`tools/ladder/session.py` (LADDER-1 §1.2): register (hold the stream) -> play every move request through the backend, the book's stones first (R363(c)) -> report one receipt per game; the challenger arm issues the paired challenges."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from _ladder_stub import TOKEN, base_url, serve

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "tests" / "fixtures" / "ladder"
_NET = "a9a46c55bd1ceadb38145fef6527254d77d37900aebea4332def55ca75f56bfc"
#: `book_v1_s20260625_p4` opening 0 is [[2, 2], [4, 1], [2, 0], [2, -1]]; relative to its first stone:
_OPENING_0 = [[0, 0], [2, -1], [0, -2], [0, -3]]


class _FakeBackend:
    """A `Backend` with scripted turns keyed by the stone count it is shown; records every call."""

    backend = "mantis"
    name = f"mantis:{_NET[:8]}"
    net_hash = _NET
    sims = 4
    encoding = "gnn_axis_v1"
    search = {"kind": "puct", "sims": 4}
    seed = None

    def __init__(self, turns: dict[int, tuple[tuple[int, int], tuple[int, int]]]) -> None:
        self.turns = turns
        self.games: list[str] = []
        self.seen: list[int] = []

    def new_game(self, game_id: str) -> None:
        self.games.append(game_id)

    def select_turn(self, board: Any, forced=()):
        from ladder.backends import TurnResult  # noqa: PLC0415 — the package is loaded by the `ladder` fixture

        stones = len(board.get_stones())
        self.seen.append(stones)
        scripted = self.turns.get(stones, ((9, 9), (9, 8)))
        placements = tuple(forced) + tuple(scripted[len(forced):])
        return TurnResult(placements=(placements[0], placements[1]), sims=8 * (2 - len(forced)) // 2, ms=1.5,
                          book_stones=len(forced))

    def close(self) -> None:
        return None


@pytest.fixture
def stub():
    yield from serve(_FIXTURES / "recorded_two_games.ndjson")


def _session(ladder, stub, backend, tmp_path: Path, **options: Any):
    client = ladder.client.LadderClient(base_url(stub), TOKEN, timeout_sec=5.0)
    opts = ladder.session.SessionOptions(server=base_url(stub), work_dir=tmp_path, open_for_challenges=True,
                                         reconnect=False, **options)
    return ladder.session.Session(client, backend, options=opts, log=lambda _line: None)


def _posts(stub, suffix: str) -> list[tuple[str, Any]]:
    return [(p, b) for m, p, _h, b in stub.seen if m == "POST" and p.endswith(suffix)]


def test_every_move_request_is_answered_with_the_backends_turn_and_the_request_id(ladder, stub, tmp_path: Path) -> None:
    """Game 1 (x): the recorded opponent's stones are not opening 0's, so the head goes off-book and searches; game 2 (o, the pair's second game, opening 0 again): plies 2–3 are the book's, unsearched."""
    backend = _FakeBackend({3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    summary = _session(ladder, stub, backend, tmp_path).run()
    assert backend.games == ["g_7Qm2Kx", "g_second"]
    assert _posts(stub, "/move") == [
        ("/api/bot/game/g_7Qm2Kx/move", {"move": {"pieces": [{"q": 2, "r": 0}, {"q": 3, "r": 0}]}, "request_id": 1}),
        ("/api/bot/game/g_7Qm2Kx/move", {"move": {"pieces": [{"q": -1, "r": 0}, {"q": -2, "r": 0}]}, "request_id": 2}),
        ("/api/bot/game/g_second/move", {"move": {"pieces": [{"q": 2, "r": -1}, {"q": 0, "r": -2}]}, "request_id": 1}),
    ]
    assert summary.games == 2 and summary.wins == 1 and summary.losses == 1 and summary.aborted == 0


def test_a_finished_game_leaves_one_receipt_under_the_net_hash_with_its_moves_and_result(ladder, stub, tmp_path: Path) -> None:
    backend = _FakeBackend({3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    _session(ladder, stub, backend, tmp_path).run()
    receipt = ladder.receipt.read_receipt(tmp_path / "receipts" / "a9a46c55" / "g_7Qm2Kx.json")
    assert receipt["bot"]["name"] == "mantis:a9a46c55" and receipt["bot"]["display_name"] == "Mantis"
    assert receipt["opponent"] == {"display_name": "Strix", "profile_id": "p_them", "elo": 1000}
    assert receipt["side"] == "x" and receipt["result"] == {"winner": "o", "reason": "six-in-a-row", "outcome": "loss"}
    assert [m["placements"] for m in receipt["moves"]] == [[[2, 0], [3, 0]], [[-1, 0], [-2, 0]]]
    assert [m["sims"] for m in receipt["moves"]] == [8, 8] and receipt["sims_configured"] == 4
    assert receipt["moves"][1]["time_limit"] == 45 and receipt["moves"][0]["server_date"] == "Sat, 19 Sep 2026 12:00:00 GMT"
    assert receipt["plies"] == 3 and receipt["moves_full"] == [[0, 0, "x"], [1, 0, "o"], [0, 1, "o"]]
    assert receipt["opening"] == {"book": "book_v1_s20260625_p4", "index": 0, "opening_id": "0", "relative": _OPENING_0,
                                  "off_book_at": 3}
    assert [m["book_stones"] for m in receipt["moves"]] == [0, 0]
    second = ladder.receipt.read_receipt(tmp_path / "receipts" / "a9a46c55" / "g_second.json")
    assert second["result"]["outcome"] == "win" and second["plies"] is None and second["plies_seen"] == 3
    assert second["opening"]["index"] == 0 and second["opening"]["off_book_at"] is None
    assert [(m["book_stones"], m["sims"], m["placements"]) for m in second["moves"]] == [(2, 0, [[2, -1], [0, -2]])]


def test_a_rejected_move_is_recorded_and_the_game_resigned_rather_than_left_on_the_clock(ladder, stub, tmp_path: Path) -> None:
    backend = _FakeBackend({3: ((0, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    summary = _session(ladder, stub, backend, tmp_path).run()
    assert _posts(stub, "/resign") == [("/api/bot/game/g_7Qm2Kx/resign", None)]
    receipt = ladder.receipt.read_receipt(tmp_path / "receipts" / "a9a46c55" / "g_7Qm2Kx.json")
    assert receipt["rejections"] == [{"request_id": 1, "placements": [[0, 0], [3, 0]], "code": "occupied",
                                      "message": "That cell is already occupied."}]
    assert summary.rejections == 1


def test_a_challenge_from_an_allowed_bot_is_accepted_and_any_other_declined(ladder, tmp_path: Path) -> None:
    challenge = {"challengeId": "c_1", "challenger": {"profileId": "p_them", "displayName": "Strix", "elo": 1000},
                 "destUser": {"profileId": "p_me", "displayName": "Mantis", "elo": 1000},
                 "timeControl": {"mode": "unlimited"}, "status": "created"}
    other = {**challenge, "challengeId": "c_2", "challenger": {"profileId": "p_stranger", "displayName": "X", "elo": None}}
    path = tmp_path / "stream.ndjson"
    path.write_text(json.dumps({"type": "challenge", "challenge": challenge}) + "\n"
                    + json.dumps({"type": "challenge", "challenge": other}) + "\n", encoding="utf-8")
    for stub in serve(path):
        _session(ladder, stub, _FakeBackend({}), tmp_path, accept_from=frozenset({"p_them"})).run()
        assert [p for p, _b in _posts(stub, "/accept") + _posts(stub, "/decline")] == [
            "/api/bot/challenge/c_1/accept", "/api/bot/challenge/c_2/decline"]


def test_the_challenger_issues_paired_challenges_alternating_first_player_and_stops_at_its_count(ladder, stub, tmp_path: Path) -> None:
    backend = _FakeBackend({3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    plan = ladder.session.ChallengePlan(profile_id="p_them", games=2, time_control={"mode": "unlimited"})
    summary = _session(ladder, stub, backend, tmp_path, challenge=plan).run()
    assert [b for _p, b in _posts(stub, "/api/bot/challenge/p_them")] == [
        {"timeControl": {"mode": "unlimited"}, "firstPlayer": "challenger"},
        {"timeControl": {"mode": "unlimited"}, "firstPlayer": "challenged"},
    ]
    assert summary.games == 2 and summary.challenges_issued == 2


def test_a_target_that_is_not_open_yet_is_retried_until_it_is(ladder, stub, tmp_path: Path) -> None:
    stub.challenge_refusal = {"error": "That bot is not taking challenges right now.", "code": "not-open"}
    threading.Timer(0.15, lambda: setattr(stub, "challenge_refusal", None)).start()
    backend = _FakeBackend({3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    plan = ladder.session.ChallengePlan(profile_id="p_them", games=1, time_control={"mode": "unlimited"})
    session = _session(ladder, stub, backend, tmp_path, challenge=plan, challenge_retry_sec=0.05,
                       challenge_timeout_sec=5.0)
    session.run()
    assert len(_posts(stub, "/api/bot/challenge/p_them")) >= 2


def test_a_target_that_never_opens_is_a_named_failure_not_a_hang(ladder, stub, tmp_path: Path) -> None:
    stub.challenge_refusal = {"error": "That bot is not taking challenges right now.", "code": "not-open"}
    plan = ladder.session.ChallengePlan(profile_id="p_them", games=1, time_control={"mode": "unlimited"})
    session = _session(ladder, stub, _FakeBackend({}), tmp_path, challenge=plan, challenge_retry_sec=0.02,
                       challenge_timeout_sec=0.1)
    with pytest.raises(ladder.session.SessionError, match="not-open"):
        session.run()


def _series(path: Path, opponents: list[str]) -> None:
    """One o-side game per entry: the origin on the wire, one move request, a finish — the forced plies 2–3 land in the POST."""
    lines = []
    for i, who in enumerate(opponents):
        gid = f"g_{i}"
        lines.append({"type": "gameStart", "gameId": gid, "side": "o", "opponent": {"profileId": who, "displayName": who, "elo": 1000},
                      "timeControl": {"mode": "unlimited"}, "rated": False})
        lines.append({"type": "moveRequest", "gameId": gid, "request": {"board": {"to_move": "o", "cells": [{"q": 0, "r": 0, "p": "x"}]},
                                                                          "request_id": 1}})
        lines.append({"type": "gameFinish", "gameId": gid, "winner": "x", "reason": "surrender"})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def test_pair_m_against_one_opponent_plays_opening_m_counted_per_opponent_from_the_offset(ladder, tmp_path: Path) -> None:
    """R363(c): opening index = match (pair) index — games 2m and 2m+1 share opening m; another opponent's game advances nothing; `match_offset` names where a resumed series left off."""
    from mantis.arena.books import book_openings

    book = book_openings("book_v1_s20260625_p4")
    path = tmp_path / "stream.ndjson"
    _series(path, ["p_them", "p_them", "p_other", "p_them", "p_them", "p_them"])
    for stub in serve(path):
        session = _session(ladder, stub, _FakeBackend({}), tmp_path, match_offset=5)
        session.run()
        posted = [b["move"]["pieces"] for _p, b in _posts(stub, "/move")]
        expect_index = [5, 5, 5, 6, 6, 7]
        for pieces, index in zip(posted, expect_index, strict=True):
            q0, r0 = book[index].moves[0]
            assert pieces == [{"q": q - q0, "r": r - r0} for q, r in book[index].moves[1:3]], index
        receipts = sorted((tmp_path / "receipts" / "a9a46c55").glob("g_*.json"), key=lambda p: int(p.stem[2:]))
        assert [ladder.receipt.read_receipt(r)["opening"]["index"] for r in receipts] == expect_index
