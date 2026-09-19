"""A loopback stand-in for the DEPLOYED HeXO bot API (TimmyBurn2/HeXO@8166053, verified live 2026-09-19) for the LADDER-1 client and session tests: it serves a recorded NDJSON stream and records every request it sees."""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOKEN = "hxo_test_token"
DATE = "Sat, 19 Sep 2026 12:00:00 GMT"
FINISHED_GAME = {
    "id": "g_7Qm2Kx", "startedAt": 1789819744122, "finishedAt": 1789819758808,
    "players": [{"playerId": "a1", "displayName": "Mantis", "profileId": "p_me", "elo": 1000, "eloChange": None, "isBot": True},
                {"playerId": "b2", "displayName": "Strix", "profileId": "p_them", "elo": 1000, "eloChange": None, "isBot": True}],
    "moveCount": 3, "gameOptions": {"timeControl": {"mode": "unlimited"}, "rated": False},
    "gameResult": {"abortedByPlayerId": None, "winningPlayerId": "b2", "durationMs": 14686, "reason": "six-in-a-row"},
    "moves": [{"moveNumber": 1, "playerId": "a1", "x": 0, "y": 0, "timestamp": 1789819744945},
              {"moveNumber": 2, "playerId": "b2", "x": 1, "y": 0, "timestamp": 1789819745252},
              {"moveNumber": 3, "playerId": "b2", "x": 1, "y": -1, "timestamp": 1789819745252}],
}


class Handler(BaseHTTPRequestHandler):
    """Each request lands in `server.seen` as `(method, path, headers, body)`; the stream is `server.stream_bytes`."""

    def log_message(self, *_args: Any) -> None:
        return

    def _reply(self, status: int, body: Any) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Date", DATE)
        self.end_headers()
        self.wfile.write(raw)

    def _record(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        self.server.seen.append((self.command, self.path, dict(self.headers), body))  # type: ignore[attr-defined]
        return body

    def do_GET(self) -> None:
        self._record()
        server: Any = self.server
        if self.path.startswith("/api/finished-games/"):
            game_id = self.path.rsplit("/", 1)[1]
            if game_id == FINISHED_GAME["id"]:
                self._reply(200, FINISHED_GAME)
            else:
                self._reply(404, {"error": "Finished game not found"})
            return
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._reply(401, {"error": "A bot account token is required."})
            return
        if self.path.startswith("/api/bot/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(server.stream_bytes)
            self.wfile.flush()
            return
        if self.path == "/api/bot/account":
            self._reply(200, {"bot": {"profileId": "p_me", "displayName": "Mantis", "elo": 1000},
                              "owner": {"profileId": "p_owner", "displayName": "owner", "elo": 1000},
                              "activeGames": []})
            return
        self._reply(404, {"error": "no such path"})

    def do_POST(self) -> None:
        body = self._record()
        server: Any = self.server
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._reply(401, {"error": "A bot account token is required."})
            return
        if re.fullmatch(r"/api/bot/game/[^/]+/move", self.path):
            pieces = (body or {}).get("move", {}).get("pieces", [])
            if pieces and pieces[0] == {"q": 0, "r": 0}:
                self._reply(400, {"error": "That cell is already occupied.", "code": "occupied"})
            else:
                self._reply(200, {"ok": True})
            return
        if re.fullmatch(r"/api/bot/challenge/[^/]+", self.path):
            if server.challenge_refusal is not None:
                self._reply(400, server.challenge_refusal)
                return
            server.challenges_created += 1
            self._reply(200, {"challengeId": f"c_new{server.challenges_created}",
                              "challenger": {"profileId": "p_me", "displayName": "Mantis", "elo": 1000},
                              "destUser": {"profileId": "p_them", "displayName": "Strix", "elo": 1000},
                              "timeControl": body["timeControl"], "status": "created"})
            return
        if re.fullmatch(r"/api/bot/challenge/[^/]+/(accept|decline|cancel)", self.path) or \
                re.fullmatch(r"/api/bot/game/[^/]+/resign", self.path):
            self._reply(200, {"ok": True})
            return
        self._reply(404, {"error": "no such path"})


def serve(stream_path: Path) -> Iterator[Any]:
    """Yield a running stub serving `stream_path`'s bytes; `server.seen` is what it saw."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen = []  # type: ignore[attr-defined]
    server.stream_bytes = stream_path.read_bytes()  # type: ignore[attr-defined]
    server.challenge_refusal = None  # type: ignore[attr-defined]
    server.challenges_created = 0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def base_url(server: Any) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}"
