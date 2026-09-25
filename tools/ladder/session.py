"""The bot's loop (LADDER-1 §1.2): hold the stream (that IS registration and presence), answer every move request through the backend in arrival order — the book's stones first (pair `m` against one opponent plays opening `m`, both bots counting the same `gameStart`s) — accept challenges, write one receipt per finished game; the challenger arm issues paired challenges, first player alternating, and stops at its count."""
from __future__ import annotations

import http.client
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ladder.client import ApiError, LadderClient, MoveRejected
from ladder.openings import LadderOpening, forced_stones, ladder_opening
from ladder.receipt import GameReceipt, write_receipt
from ladder.wire import WireError, board_from_wire, move_response

#: Rejections that mean the turn already moved on; anything else is a move WE got wrong, and the game is resigned.
_STALE_CODES = frozenset({"stale-request", "game-over", "not-your-turn"})
#: The challenge answers worth waiting on: the target is offline, not open, at its cap, or a challenge is pending.
_RETRY_CODES = frozenset({"not-open"})


class SessionError(RuntimeError):
    """The loop cannot continue as asked: a wire position that does not replay, or a challenge that never lands."""


@dataclass(frozen=True)
class ChallengePlan:
    """Issue `games` challenges to `profile_id`, first player alternating challenger/challenged, one at a time."""

    profile_id: str
    games: int
    time_control: dict[str, Any]


@dataclass
class SessionOptions:
    server: str
    work_dir: Path
    open_for_challenges: bool
    accept_from: frozenset[str] | None = None
    challenge: ChallengePlan | None = None
    reconnect: bool = True
    reconnect_delay_sec: float = 5.0
    challenge_retry_sec: float = 10.0
    challenge_timeout_sec: float = 900.0
    #: The pair index the FIRST pair against an opponent plays (a resumed series names where it left off).
    match_offset: int = 0


@dataclass
class SessionSummary:
    games: int = 0
    wins: int = 0
    losses: int = 0
    aborted: int = 0
    rejections: int = 0
    challenges_issued: int = 0
    receipts: list[Path] = field(default_factory=list)


class Session:
    """One registered bot: one client, one backend, one stream at a time."""

    def __init__(self, client: LadderClient, backend: Any, *, options: SessionOptions,
                 log: Callable[[str], None] = print, clock: Callable[[], float] = time.time) -> None:
        self.client, self.backend, self.options = client, backend, options
        self.log, self.clock = log, clock
        self.summary = SessionSummary()
        self._games: dict[str, GameReceipt] = {}
        self._openings: dict[str, LadderOpening] = {}
        self._ordinals: dict[str, int] = {}
        self._account: dict[str, Any] = {}
        self._outgoing: str | None = None
        self._games_finished_for_plan = 0

    def run(self) -> SessionSummary:
        """Hold the stream until the plan is done (challenger) or the stream ends without `reconnect`. Raises: SessionError on a wire position that does not replay, or a challenge target that never opens."""
        self._account = self.client.account()
        bot = self._account["bot"]
        self.log(f"ladder: {self.backend.name} is {bot['displayName']} ({bot['profileId']}) on {self.options.server}, "
                 f"{len(self._account.get('activeGames', []))} active game(s) at start")
        while True:
            try:
                with self.client.stream(open_for_challenges=self.options.open_for_challenges) as stream:
                    self.log("ladder: stream open" + (", taking challenges" if self.options.open_for_challenges else ""))
                    self._on_open()
                    for event in stream:
                        self.handle(event)
                        if self._plan_done():
                            return self.summary
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                self.log(f"ladder: stream dropped ({type(exc).__name__}: {exc})")
            if self._plan_done() or not self.options.reconnect:
                return self.summary
            time.sleep(self.options.reconnect_delay_sec)
            self.log("ladder: reconnecting")

    def handle(self, event: dict[str, Any]) -> None:
        """Dispatch one stream line; unknown types are logged and skipped, never fatal."""
        kind = event.get("type")
        if kind == "gameStart":
            self._start(event)
        elif kind == "moveRequest":
            self._move(event)
        elif kind == "gameFinish":
            self._finish(event)
        elif kind == "challenge":
            self._challenge(event["challenge"])
        elif kind in ("challengeDeclined", "challengeCanceled"):
            self._challenge_ended(event)
        else:
            self.log(f"ladder: unknown event type {kind!r} skipped")

    # --- games ---

    def _start(self, event: dict[str, Any]) -> None:
        game_id = str(event["gameId"])
        if game_id in self._games:
            self.log(f"ladder: {game_id} replayed on reconnect; keeping its receipt")
            return
        opponent = dict(event.get("opponent") or {})
        opening = self._opening_for(str(opponent.get("profileId")))
        self._openings[game_id] = opening
        self._games[game_id] = GameReceipt(
            server=self.options.server, game_id=game_id,
            bot={"name": self.backend.name, "backend": self.backend.backend, "net_hash": self.backend.net_hash,
                 "display_name": self._account["bot"]["displayName"], "profile_id": self._account["bot"]["profileId"]},
            opponent={"display_name": opponent.get("displayName"), "profile_id": opponent.get("profileId"),
                      "elo": opponent.get("elo")},
            side=str(event["side"]), time_control=dict(event.get("timeControl") or {}), rated=bool(event.get("rated")),
            sims_configured=int(self.backend.sims), search=dict(self.backend.search), started=self.clock(),
            opening=opening.to_record())
        self.backend.new_game(game_id)
        self.log(f"ladder: {game_id} started, playing {event['side']} vs {opponent.get('displayName')}, "
                 f"opening {opening.index} of {opening.book}")

    def _opening_for(self, opponent_id: str) -> LadderOpening:
        """Pair `match_offset + n // 2` for the n-th game (0-based) against this opponent in this session: both bots count the same `gameStart`s, so the pair's two games share the opening without a word on the wire."""
        ordinal = self._ordinals.get(opponent_id, 0)
        self._ordinals[opponent_id] = ordinal + 1
        return ladder_opening(self.options.match_offset + ordinal // 2)

    def _move(self, event: dict[str, Any]) -> None:
        game_id = str(event["gameId"])
        request = event["request"]
        request_id = request.get("request_id")
        receipt = self._games.get(game_id)
        if receipt is None:
            # Our turn in a game whose start we never saw: refuse to guess a side; the server will re-ask on reconnect.
            self.log(f"ladder: move request for unknown game {game_id}; skipped")
            return
        try:
            board = board_from_wire(request["board"], encoding=self.backend.encoding)
        except WireError as exc:
            raise SessionError(f"{game_id} request {request_id}: the wire position does not replay: {exc}") from exc
        cells = [(int(c["q"]), int(c["r"])) for c in request["board"]["cells"]]
        forced = forced_stones(self._openings[game_id], cells)
        if forced is None:
            if receipt.off_book_at is None:
                self.log(f"ladder: {game_id} request {request_id}: off-book after {len(cells)} stones; searching")
            receipt.mark_off_book(stones=len(cells))
            forced = ()
        turn = self.backend.select_turn(board, forced)
        limit = request.get("time_limit")
        if limit is not None and turn.ms / 1000.0 > float(limit):
            self.log(f"ladder: {game_id} request {request_id}: thought {turn.ms / 1000.0:.1f} s against a "
                     f"{float(limit):.1f} s limit (the budget is fixed; not adapted)")
        try:
            result = self.client.move(game_id, move_response(turn.placements, request_id=request_id))
        except MoveRejected as exc:
            self.summary.rejections += 1
            receipt.add_rejection(request_id=request_id, placements=turn.placements, code=exc.code, message=exc.message)
            self.log(f"ladder: {game_id} request {request_id}: move {turn.placements} REJECTED "
                     f"[{exc.code}] {exc.message}")
            if exc.code not in _STALE_CODES:
                self.log(f"ladder: {game_id}: resigning rather than leaving the clock to run on our illegal move")
                try:
                    self.client.resign(game_id)
                except ApiError as resign_exc:
                    self.log(f"ladder: {game_id}: resign refused: {resign_exc}")
            return
        receipt.add_move(request_id=request_id, stones=len(cells), time_limit=limit, placements=turn.placements,
                         sims=turn.sims, ms=turn.ms, server_date=result.server_date, book_stones=turn.book_stones)

    def _finish(self, event: dict[str, Any]) -> None:
        game_id = str(event["gameId"])
        receipt = self._games.pop(game_id, None)
        self._openings.pop(game_id, None)
        winner, reason = event.get("winner"), str(event.get("reason"))
        if receipt is None:
            self.log(f"ladder: {game_id} finished ({reason}, winner {winner}) but its start was never seen; no receipt")
        else:
            try:
                record = self.client.finished_game(game_id)
            except ApiError as exc:
                self.log(f"ladder: {game_id}: finished-game record unavailable ({exc}); plies from what we saw")
                record = None
            body = receipt.finish(winner=winner, reason=reason, finished=self.clock(), finished_game=record)
            path = write_receipt(self.options.work_dir, body)
            self.summary.receipts.append(path)
            outcome = body["result"]["outcome"]
            self.summary.games += 1
            self.summary.wins += outcome == "win"
            self.summary.losses += outcome == "loss"
            self.summary.aborted += outcome == "aborted"
            self.log(f"ladder: {game_id} over: {outcome} ({reason}), {body['plies']} plies, receipt {path.name}")
        if self.options.challenge is not None:
            self._games_finished_for_plan += 1
            self._outgoing = None
            if not self._plan_done():
                self._issue_challenge()

    # --- challenges ---

    def _challenge(self, challenge: dict[str, Any]) -> None:
        challenger = dict(challenge.get("challenger") or {})
        challenge_id = str(challenge["challengeId"])
        allowed = self.options.accept_from
        if allowed is None or challenger.get("profileId") in allowed:
            self.client.accept(challenge_id)
            self.log(f"ladder: accepted challenge {challenge_id} from {challenger.get('displayName')}")
        else:
            self.client.decline(challenge_id)
            self.log(f"ladder: declined challenge {challenge_id} from {challenger.get('displayName')} (not allowed)")

    def _challenge_ended(self, event: dict[str, Any]) -> None:
        challenge_id = str(event["challenge"]["challengeId"])
        self.log(f"ladder: challenge {challenge_id} {event['type']} ({event.get('reason', 'declined')})")
        if self.options.challenge is not None and challenge_id == self._outgoing:
            self._outgoing = None
            if not self._plan_done():
                self._issue_challenge()

    def _on_open(self) -> None:
        if self.options.challenge is not None and self._outgoing is None and not self._plan_done():
            self._issue_challenge()

    def _plan_done(self) -> bool:
        plan = self.options.challenge
        return plan is not None and self._games_finished_for_plan >= plan.games

    def _issue_challenge(self) -> None:
        """One challenge, retried while the target is not open, until `challenge_timeout_sec`. Raises: SessionError when the target never opened, or refused for a reason waiting cannot fix."""
        plan = self.options.challenge
        assert plan is not None
        first_player = "challenger" if self.summary.challenges_issued % 2 == 0 else "challenged"
        deadline = self.clock() + self.options.challenge_timeout_sec
        while True:
            try:
                created = self.client.challenge(plan.profile_id, time_control=plan.time_control,
                                                first_player=first_player)
            except ApiError as exc:
                retry = exc.status == 400 and (exc.code in _RETRY_CODES or "pending" in exc.message)
                if not retry:
                    raise SessionError(f"challenge to {plan.profile_id} refused: {exc}") from exc
                if self.clock() >= deadline:
                    raise SessionError(f"challenge to {plan.profile_id} not accepted within "
                                       f"{self.options.challenge_timeout_sec:.0f} s: last answer {exc}") from exc
                self.log(f"ladder: challenge to {plan.profile_id} refused ({exc.code or exc.message}); retrying in "
                         f"{self.options.challenge_retry_sec:.0f} s")
                time.sleep(self.options.challenge_retry_sec)
                continue
            self._outgoing = str(created["challengeId"])
            self.summary.challenges_issued += 1
            self.log(f"ladder: challenge {self._outgoing} issued to {plan.profile_id} "
                     f"({self.summary.challenges_issued}/{plan.games}, first player {first_player})")
            return


__all__ = ["ChallengePlan", "Session", "SessionError", "SessionOptions", "SessionSummary"]
