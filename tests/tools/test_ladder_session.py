"""`tools/ladder/session.py` (LADDER-1 §1.2): register (hold the stream) -> play every move request through the backend -> report one receipt per game; the challenger arm issues the paired challenges."""
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

    def select_turn(self, board: Any):
        from ladder.backends import TurnResult  # noqa: PLC0415 — the package is loaded by the `ladder` fixture

        stones = len(board.get_stones())
        self.seen.append(stones)
        return TurnResult(placements=self.turns[stones], sims=8, ms=1.5)

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
    backend = _FakeBackend({3: ((2, 0), (3, 0)), 7: ((-1, 0), (-2, 0)), 1: ((1, 0), (0, 1))})
    summary = _session(ladder, stub, backend, tmp_path).run()
    assert backend.games == ["g_7Qm2Kx", "g_second"]
    assert _posts(stub, "/move") == [
        ("/api/bot/game/g_7Qm2Kx/move", {"move": {"pieces": [{"q": 2, "r": 0}, {"q": 3, "r": 0}]}, "request_id": 1}),
        ("/api/bot/game/g_7Qm2Kx/move", {"move": {"pieces": [{"q": -1, "r": 0}, {"q": -2, "r": 0}]}, "request_id": 2}),
        ("/api/bot/game/g_second/move", {"move": {"pieces": [{"q": 1, "r": 0}, {"q": 0, "r": 1}]}, "request_id": 1}),
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
    second = ladder.receipt.read_receipt(tmp_path / "receipts" / "a9a46c55" / "g_second.json")
    assert second["result"]["outcome"] == "win" and second["plies"] is None and second["plies_seen"] == 3


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
