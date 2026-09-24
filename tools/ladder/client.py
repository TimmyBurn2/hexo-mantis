"""The HeXO bot-API client (LADDER-1), the ONE module with endpoint strings, stdlib only; verified against the DEPLOYED server (TimmyBurn2/HeXO@8166053, live 2026-09-19), which is ahead of the spec file where CARD-LADDER-RUNG says."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

#: A keepalive newline arrives every 10 s; a read that sees nothing for this long is a dead stream.
STREAM_READ_TIMEOUT_SEC = 60.0
FIRST_PLAYER = ("challenger", "challenged", "random")


class ApiError(RuntimeError):
    """A non-2xx answer: `status`, the server's `error` text and its machine `code` when it sent one."""

    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}" + (f" [{code}]" if code else ""))
        self.status, self.message, self.code = status, message, code


class MoveRejected(ApiError):
    """A 400 on a move: the clock keeps running and the same turn may be answered again."""


@dataclass(frozen=True)
class MoveResult:
    """One accepted move: the server's `Date` header is the only server clock a bot sees."""

    server_date: str | None


class Stream:
    """One held-open NDJSON connection; see `LadderClient.stream`."""

    def __init__(self, connect: Any) -> None:
        self._connect = connect
        self._resp: Any = None

    def __enter__(self) -> Stream:
        self._resp = self._connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._resp is None:
            raise RuntimeError("iterate a Stream inside its `with` block")
        try:
            for raw in self._resp:
                line = raw.strip()
                if line:
                    yield json.loads(line)
        except TimeoutError as exc:
            raise TimeoutError(f"no stream byte for {STREAM_READ_TIMEOUT_SEC:.0f} s") from exc


class LadderClient:
    """Bearer-token HTTP over `base_url`; every method raises `ApiError` on a non-2xx answer."""

    def __init__(self, base_url: str, token: str, *, timeout_sec: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_sec)

    def _request(self, method: str, path: str, body: Any = None, *, timeout: float | None = None,
                 stream: bool = False) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self._base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/json, application/x-ndjson")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout if timeout is None else timeout)
        except urllib.error.HTTPError as exc:
            raise _api_error(exc) from None
        return resp

    def _json(self, method: str, path: str, body: Any = None) -> tuple[Any, dict[str, str]]:
        with self._request(method, path, body) as resp:
            headers = {k: v for k, v in resp.headers.items()}
            raw = resp.read()
        return (json.loads(raw) if raw else None), headers

    def account(self) -> dict[str, Any]:
        """`GET /api/bot/account`: the bot, its owner and its active games."""
        return self._json("GET", "/api/bot/account")[0]

    def stream(self, *, open_for_challenges: bool) -> Stream:
        """`GET /api/bot/stream` as a context: up on enter (presence starts there), each event line parsed on iteration, keepalive blanks skipped; it ends when the server closes, or raises TimeoutError after `STREAM_READ_TIMEOUT_SEC` of silence — a drop, answered by reconnecting."""
        path = "/api/bot/stream" + ("?open=1" if open_for_challenges else "")
        return Stream(lambda: self._request("GET", path, timeout=STREAM_READ_TIMEOUT_SEC))

    def move(self, game_id: str, body: dict[str, Any]) -> MoveResult:
        """`POST /api/bot/game/{gameId}/move` with an htttx `MoveResponse`. Raises: MoveRejected on a 400 (carrying the server's `code`: not-your-turn, occupied, out-of-range, game-over, stale-request); ApiError on any other non-2xx."""
        try:
            _body, headers = self._json("POST", f"/api/bot/game/{game_id}/move", body)
        except ApiError as exc:
            if exc.status == 400:
                raise MoveRejected(exc.status, exc.message, exc.code) from None
            raise
        return MoveResult(server_date=headers.get("Date"))

    def resign(self, game_id: str) -> None:
        """`POST /api/bot/game/{gameId}/resign`."""
        self._json("POST", f"/api/bot/game/{game_id}/resign")

    def challenge(self, profile_id: str, *, time_control: dict[str, Any], first_player: str) -> dict[str, Any]:
        """`POST /api/bot/challenge/{profileId}`: the created challenge. `first_player` is one of `FIRST_PLAYER`."""
        if first_player not in FIRST_PLAYER:
            raise ValueError(f"first_player {first_player!r} is not one of {FIRST_PLAYER}")
        return self._json("POST", f"/api/bot/challenge/{profile_id}",
                          {"timeControl": time_control, "firstPlayer": first_player})[0]

    def accept(self, challenge_id: str) -> None:
        """`POST /api/bot/challenge/{challengeId}/accept`."""
        self._json("POST", f"/api/bot/challenge/{challenge_id}/accept")

    def decline(self, challenge_id: str) -> None:
        """`POST /api/bot/challenge/{challengeId}/decline`."""
        self._json("POST", f"/api/bot/challenge/{challenge_id}/decline")

    def finished_game(self, game_id: str) -> dict[str, Any] | None:
        """`GET /api/finished-games/{id}` (the website's public record, keyed by the stream's `gameId`): the full move list in HeXO `x,y` with server timestamps, or None when the server keeps no history."""
        try:
            return self._json("GET", f"/api/finished-games/{game_id}")[0]
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise


def _api_error(exc: urllib.error.HTTPError) -> ApiError:
    message, code = exc.reason, None
    try:
        payload = json.loads(exc.read())
        message = str(payload.get("error", message))
        code = payload.get("code")
    except (ValueError, AttributeError, OSError):
        pass
    return ApiError(exc.code, str(message), None if code is None else str(code))


__all__ = ["ApiError", "FIRST_PLAYER", "LadderClient", "MoveRejected", "MoveResult", "STREAM_READ_TIMEOUT_SEC",
           "Stream"]
