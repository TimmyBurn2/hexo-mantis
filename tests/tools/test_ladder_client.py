"""`tools/ladder/client.py` (LADDER-1): the ONE module with endpoint strings, driven against a recorded stream on a loopback stub that answers as the DEPLOYED server does (TimmyBurn2/HeXO@8166053, verified live 2026-09-19)."""
from __future__ import annotations

from pathlib import Path

import pytest

from _ladder_stub import TOKEN as _TOKEN
from _ladder_stub import base_url, serve

_REPO = Path(__file__).resolve().parents[2]
_STREAM = _REPO / "tests" / "fixtures" / "ladder" / "recorded_stream.ndjson"


@pytest.fixture
def stub():
    yield from serve(_STREAM)


def _client(ladder, stub, token: str = _TOKEN):
    return ladder.client.LadderClient(base_url(stub), token, timeout_sec=5.0)


def test_the_stream_yields_the_recorded_events_in_order_and_skips_keepalives(ladder, stub) -> None:
    with _client(ladder, stub).stream(open_for_challenges=True) as stream:
        assert stub.seen, "the connection is up on enter, before the first event is read"
        events = list(stream)
    assert [e["type"] for e in events] == ["gameStart", "moveRequest", "moveRequest", "gameFinish",
                                           "challenge", "challengeDeclined", "challengeCanceled"]
    assert events[2]["request"]["time_limit"] == 45
    method, path, headers, _body = stub.seen[0]
    assert (method, path) == ("GET", "/api/bot/stream?open=1")
    assert headers["Authorization"] == f"Bearer {_TOKEN}"


def test_a_stream_opened_without_open_takes_no_challenges(ladder, stub) -> None:
    with _client(ladder, stub).stream(open_for_challenges=False) as stream:
        list(stream)
    assert stub.seen[0][1] == "/api/bot/stream"


def test_a_move_posts_the_body_and_returns_the_server_clock(ladder, stub) -> None:
    result = _client(ladder, stub).move("g_7Qm2Kx", {"move": {"pieces": [{"q": 1, "r": -1}, {"q": 2, "r": 0}]}, "request_id": 1})
    assert result.server_date == "Sat, 19 Sep 2026 12:00:00 GMT"
    method, path, headers, body = stub.seen[0]
    assert (method, path) == ("POST", "/api/bot/game/g_7Qm2Kx/move")
    assert headers["Content-Type"] == "application/json"
    assert body == {"move": {"pieces": [{"q": 1, "r": -1}, {"q": 2, "r": 0}]}, "request_id": 1}


def test_a_rejected_move_raises_with_the_servers_code(ladder, stub) -> None:
    with pytest.raises(ladder.client.MoveRejected) as caught:
        _client(ladder, stub).move("g_7Qm2Kx", {"move": {"pieces": [{"q": 0, "r": 0}, {"q": 2, "r": 0}]}})
    assert caught.value.code == "occupied"
    assert caught.value.status == 400


def test_a_challenge_posts_first_player_and_time_control_and_no_rated_key(ladder, stub) -> None:
    challenge = _client(ladder, stub).challenge("p_them", time_control={"mode": "unlimited"}, first_player="challenged")
    assert challenge["challengeId"] == "c_new1"
    _method, path, _headers, body = stub.seen[0]
    assert path == "/api/bot/challenge/p_them"
    assert body == {"timeControl": {"mode": "unlimited"}, "firstPlayer": "challenged"}


def test_accept_decline_and_resign_hit_their_paths(ladder, stub) -> None:
    client = _client(ladder, stub)
    client.accept("c_1")
    client.decline("c_2")
    client.resign("g_1")
    assert [(m, p) for m, p, _h, _b in stub.seen] == [
        ("POST", "/api/bot/challenge/c_1/accept"), ("POST", "/api/bot/challenge/c_2/decline"),
        ("POST", "/api/bot/game/g_1/resign")]


def test_the_account_read_returns_the_bot_and_its_active_games(ladder, stub) -> None:
    account = _client(ladder, stub).account()
    assert account["bot"]["displayName"] == "Mantis"
    assert account["activeGames"] == []


def test_a_finished_game_comes_back_with_its_moves_or_none_when_unknown(ladder, stub) -> None:
    client = _client(ladder, stub)
    record = client.finished_game("g_7Qm2Kx")
    assert record is not None and record["moveCount"] == 3 and len(record["moves"]) == 3
    assert client.finished_game("g_unknown") is None
    assert [p for _m, p, _h, _b in stub.seen] == ["/api/finished-games/g_7Qm2Kx", "/api/finished-games/g_unknown"]


def test_a_rotated_token_raises_401_by_name(ladder, stub) -> None:
    with pytest.raises(ladder.client.ApiError) as caught:
        _client(ladder, stub, token="hxo_rotated").account()
    assert caught.value.status == 401


def test_endpoint_strings_live_only_in_the_client_module() -> None:
    """Packet §1.1: no hand-written endpoint outside `client.py`."""
    offenders = []
    for path in [*sorted((_REPO / "tools" / "ladder").glob("*.py")), _REPO / "tools" / "ladder_bot.py"]:
        if path.name == "client.py" or not path.exists():
            continue
        if "/api/" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []
