"""One receipt per ladder game (LADDER-1 §1.3), keyed by net hash like the follower's sidecars — what was played, by which net, at what budget, what the server recorded; one that does not validate is REFUSED before it is written."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: v2: the game's `opening` (book, index, id, relative stones, where it went off-book) and `book_stones` per move.
RECEIPT_SCHEMA_VERSION = 2
OUTCOMES = ("win", "loss", "aborted")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("schema_version", "server", "game_id", "bot", "opponent", "side", "time_control", "rated",
             "sims_configured", "search", "opening", "moves", "rejections", "result", "plies", "plies_seen", "moves_full",
             "wall_sec", "think_ms_total", "started_utc", "finished_utc", "server_clock")
_BOT_REQUIRED = ("name", "backend", "net_hash", "display_name", "profile_id")
_OPENING_REQUIRED = ("book", "index", "opening_id", "relative", "off_book_at")
_MOVE_REQUIRED = ("request_id", "stones", "time_limit", "placements", "sims", "ms", "server_date", "book_stones")


class ReceiptError(ValueError):
    """A receipt body that cannot be read back; named field first."""


def _utc(ts: float) -> str:
    return time.strftime("%FT%TZ", time.gmtime(ts))


def _wire_moves(finished_game: Mapping[str, Any]) -> list[list[Any]]:
    """The record's HeXO `x,y` moves as wire `(q, r, side)` in `moveNumber` order (the array's order is racy within a turn); `q = x + y`, `r = -y`; `x` is whoever placed first."""
    moves = sorted(finished_game["moves"], key=lambda m: int(m["moveNumber"]))
    first = moves[0]["playerId"] if moves else None
    return [[int(m["x"]) + int(m["y"]), 0 - int(m["y"]), "x" if m["playerId"] == first else "o"] for m in moves]


class GameReceipt:
    """Accumulates one game from `gameStart` to `gameFinish`; `finish` returns the validated body."""

    def __init__(self, *, server: str, game_id: str, bot: Mapping[str, Any], opponent: Mapping[str, Any],
                 side: str, time_control: Mapping[str, Any], rated: bool, sims_configured: int,
                 search: Mapping[str, Any], started: float, opening: Mapping[str, Any]) -> None:
        self.server, self.game_id, self.side = server, game_id, side
        self.bot, self.opponent = dict(bot), dict(opponent)
        self.time_control, self.rated = dict(time_control), bool(rated)
        self.sims_configured, self.search, self.started = int(sims_configured), dict(search), float(started)
        self.opening = dict(opening)
        self.off_book_at: int | None = None
        self.moves: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []

    def mark_off_book(self, *, stones: int) -> None:
        """The FIRST stone count at which the board no longer matched the opening's prefix; later calls keep it."""
        if self.off_book_at is None:
            self.off_book_at = int(stones)

    def add_rejection(self, *, request_id: int | None, placements: tuple[tuple[int, int], tuple[int, int]],
                      code: str | None, message: str) -> None:
        """A move the server refused (a 400): what we sent and why it said no — a bug hunt, recorded not hidden."""
        self.rejections.append({"request_id": request_id, "placements": [[int(q), int(r)] for q, r in placements],
                                "code": code, "message": message})

    def add_move(self, *, request_id: int | None, stones: int, time_limit: float | None,
                 placements: tuple[tuple[int, int], tuple[int, int]], sims: int, ms: float,
                 server_date: str | None, book_stones: int) -> None:
        self.moves.append({"request_id": request_id, "stones": int(stones), "time_limit": time_limit,
                           "placements": [[int(q), int(r)] for q, r in placements], "sims": int(sims),
                           "ms": float(ms), "server_date": server_date, "book_stones": int(book_stones)})

    def finish(self, *, winner: str | None, reason: str, finished: float,
               finished_game: Mapping[str, Any] | None) -> dict[str, Any]:
        """The body: `outcome` from OUR side, `plies` and `moves_full` from the server's record when it has one. Raises: ReceiptError when it does not validate (the writer's own check)."""
        outcome = "aborted" if winner is None else ("win" if winner == self.side else "loss")
        last = self.moves[-1] if self.moves else None
        body = {
            "schema_version": RECEIPT_SCHEMA_VERSION, "server": self.server, "game_id": self.game_id,
            "bot": self.bot, "opponent": self.opponent, "side": self.side,
            "time_control": self.time_control, "rated": self.rated,
            "sims_configured": self.sims_configured, "search": self.search,
            "opening": {**self.opening, "off_book_at": self.off_book_at}, "moves": list(self.moves),
            "rejections": list(self.rejections),
            "result": {"winner": winner, "reason": reason, "outcome": outcome},
            "plies": None if finished_game is None else int(finished_game["moveCount"]),
            "plies_seen": None if last is None else int(last["stones"]) + 2,
            "moves_full": None if finished_game is None else _wire_moves(finished_game),
            "wall_sec": round(float(finished) - self.started, 3),
            "think_ms_total": round(sum(float(m["ms"]) for m in self.moves), 3),
            "started_utc": _utc(self.started), "finished_utc": _utc(finished),
            "server_clock": {
                "started_at_ms": None if finished_game is None else finished_game.get("startedAt"),
                "finished_at_ms": None if finished_game is None else finished_game.get("finishedAt"),
                "first_move_date": self.moves[0]["server_date"] if self.moves else None,
                "last_move_date": last["server_date"] if last is not None else None,
            },
        }
        validate_receipt(body)
        return body


def validate_receipt(body: Mapping[str, Any]) -> None:
    """Refuse a body a reader could not trust (every field present, the net hash a sha256, the opening's five fields, two placements, an integer `sims` and a `book_stones` in 0..2 per move, an outcome from OUTCOMES). Raises: ReceiptError naming the first field that fails."""
    if body.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ReceiptError(f"schema_version {body.get('schema_version')!r} is not {RECEIPT_SCHEMA_VERSION}")
    for key in _REQUIRED:
        if key not in body:
            raise ReceiptError(f"missing field {key!r}")
    for key in _BOT_REQUIRED:
        if key not in body["bot"]:
            raise ReceiptError(f"bot is missing {key!r}")
    if not _HEX64.match(str(body["bot"]["net_hash"])):
        raise ReceiptError(f"bot.net_hash {body['bot']['net_hash']!r} is not a sha256 hex digest")
    opening = body["opening"]
    if not isinstance(opening, Mapping):
        raise ReceiptError(f"opening {opening!r} is not the game's opening record")
    for key in _OPENING_REQUIRED:
        if key not in opening:
            raise ReceiptError(f"opening is missing {key!r}")
    for i, move in enumerate(body["moves"]):
        for key in _MOVE_REQUIRED:
            if key not in move:
                raise ReceiptError(f"moves[{i}] is missing {key!r}")
        if len(move["placements"]) != 2 or any(len(p) != 2 for p in move["placements"]):
            raise ReceiptError(f"moves[{i}].placements {move['placements']!r} is not two (q, r) pairs")
        if not isinstance(move["sims"], int) or isinstance(move["sims"], bool):
            raise ReceiptError(f"moves[{i}].sims {move['sims']!r} is not an integer")
        if move["book_stones"] not in (0, 1, 2):
            raise ReceiptError(f"moves[{i}].book_stones {move['book_stones']!r} is not 0, 1 or 2")
    if body["result"].get("outcome") not in OUTCOMES:
        raise ReceiptError(f"result.outcome {body['result'].get('outcome')!r} is not one of {OUTCOMES}")


def receipt_path(work_dir: Path, net_hash: str, game_id: str) -> Path:
    return Path(work_dir) / "receipts" / net_hash[:8] / f"{game_id}.json"


def write_receipt(work_dir: Path, body: Mapping[str, Any]) -> Path:
    """Validate, then write atomically under `receipts/<net_hash[:8]>/<game_id>.json`; return the path. Raises: ReceiptError when the body does not validate — nothing is written."""
    validate_receipt(body)
    path = receipt_path(work_dir, str(body["bot"]["net_hash"]), str(body["game_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def read_receipt(path: Path) -> dict[str, Any]:
    """The receipt at `path`, validated. Raises: ReceiptError when the file is not a receipt this reader trusts."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReceiptError(f"{path}: not JSON ({exc})") from exc
    validate_receipt(body)
    return body


__all__ = ["GameReceipt", "OUTCOMES", "RECEIPT_SCHEMA_VERSION", "ReceiptError", "read_receipt", "receipt_path",
           "validate_receipt", "write_receipt"]
