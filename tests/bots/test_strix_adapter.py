"""The strix rung adapter (RUNG-2): the position sent, the fence counted, the pin verified, 20 live games."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board
from mantis.bots.protocol import BotProtocol, RungUnresolvable
from mantis.bots.random_bot import RandomBot
from mantis.bots.strix import (
    CHECKPOINT_ABSENT_MARKER,
    SHA_MISMATCH_MARKER,
    VENDOR_ABSENT_MARKER,
    StrixBot,
    strix_availability,
    verify_checkpoint_sha,
)

_ENCODING = "gnn_axis_r8"


class _FakeTransport:
    """Replies with a scripted move and a legal set derived from the board the test holds."""

    def __init__(self, reply_move: tuple[int, int], legal: list[tuple[int, int]] | None = None,
                 legal_delta: tuple[int, int] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply_move = reply_move
        self.legal = legal
        self.legal_delta = legal_delta
        self.closed = False

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        legal = list(self.legal or [])
        if self.legal_delta is not None:
            legal = [c for c in legal if c != self.legal_delta]
        return {"move": list(self.reply_move), "legal": [list(c) for c in legal], "ms": 1.0, "sims": 8}

    def close(self) -> None:
        self.closed = True


def _board_after(moves: list[tuple[int, int]]) -> Any:
    board = Board.with_encoding_name(_ENCODING)
    for q, r in moves:
        board.apply_move(q, r)
    return board


def test_the_adapter_satisfies_the_bot_protocol() -> None:
    bot = StrixBot(transport=_FakeTransport((1, 0)), sims=8, name="strix_test")
    assert isinstance(bot, BotProtocol) and bot.name() == "strix_test"


def test_the_request_carries_stones_side_to_move_and_placements_left() -> None:
    board = _board_after([(0, 0), (1, 0), (0, 1)])  # p1's single, then p2's two: p1 to move, 2 left
    legal = [tuple(m) for m in board.legal_moves()]
    transport = _FakeTransport(legal[0], legal=legal)
    bot = StrixBot(transport=transport, sims=8, name="s")
    bot.new_game()
    assert bot.select_move(board) == legal[0]
    req = transport.requests[-1]
    assert req["op"] == "select"
    assert sorted(req["stones"]) == sorted([[0, 0, 1], [1, 0, -1], [0, 1, -1]])
    assert req["to_move"] == int(board.current_player) == 1
    assert req["moves_remaining"] == int(board.moves_remaining) == 2
    assert bot.fence_disagreements == 0 and bot.out_of_fence == 0 and bot.moves == 1


def test_the_opening_single_is_the_origin_and_consults_no_driver() -> None:
    """hexo_rs seats p1 at (0, 0) by rule; every opening cell is the same position up to translation."""
    transport = _FakeTransport((5, 5))
    bot = StrixBot(transport=transport, sims=8, name="s")
    assert bot.select_move(Board.with_encoding_name(_ENCODING)) == (0, 0)
    assert transport.requests == [] and bot.moves == 1


def test_the_drivers_own_sims_count_rides_the_reply_into_last_sims() -> None:
    """LADDER-1's budget witness reads `last_sims`; the opening single, which consults no driver, reads None."""
    board = _board_after([(0, 0), (1, 0), (0, 1)])
    bot = StrixBot(transport=_FakeTransport((2, 0), legal=board.legal_moves()), sims=8, name="strix_test")
    assert bot.last_sims is None
    bot.select_move(board)
    assert bot.last_sims == 8
    bot.select_move(_board_after([]))
    assert bot.last_sims is None


def test_a_fence_disagreement_is_a_counted_finding_and_the_move_is_still_returned() -> None:
    board = _board_after([(0, 0), (1, 0), (0, 1)])
    legal = [tuple(m) for m in board.legal_moves()]
    transport = _FakeTransport(legal[1], legal=legal, legal_delta=legal[0])
    bot = StrixBot(transport=transport, sims=8, name="s")
    assert bot.select_move(board) == legal[1]
    assert bot.fence_disagreements == 1
    assert bot.findings and str(legal[0]) in bot.findings[0]


def test_an_out_of_fence_reply_is_returned_unchanged_for_the_arena_to_forfeit() -> None:
    """The arena forfeits an illegal move (REPAIR-A2); the adapter never substitutes one, it counts it."""
    board = _board_after([(0, 0), (1, 0), (0, 1)])
    legal = [tuple(m) for m in board.legal_moves()]
    transport = _FakeTransport((40, 40), legal=legal)
    bot = StrixBot(transport=transport, sims=8, name="s")
    assert bot.select_move(board) == (40, 40)
    assert not board.is_legal(40, 40)
    assert bot.out_of_fence == 1 and any("out of" in f for f in bot.findings)


def test_a_driver_error_line_is_a_named_exception_not_a_silent_move() -> None:
    class _Broken(_FakeTransport):
        def ask(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"error": "RuntimeError: boom"}

    bot = StrixBot(transport=_Broken((0, 0)), sims=8, name="s")
    with pytest.raises(RuntimeError, match="boom"):
        bot.select_move(_board_after([(0, 0), (1, 0), (0, 1)]))


def test_the_pinned_sha_is_verified_and_a_mismatch_is_a_named_refusal(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_00237000.pt"
    ckpt.write_bytes(b"not the pinned bytes")
    good = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    verify_checkpoint_sha(ckpt, good)
    with pytest.raises(RungUnresolvable) as exc:
        verify_checkpoint_sha(ckpt, "0" * 64)
    assert SHA_MISMATCH_MARKER in exc.value.reason and good[:12] in exc.value.reason


def test_the_resolver_names_the_missing_step_or_resolves() -> None:
    """Environment-robust: with the vendor tree it RESOLVES; without, the refusal names the missing step."""
    from mantis.bots.resolve import resolve_bot

    try:
        factory = resolve_bot("strix", depth=None, opponent_sims=8)
    except RungUnresolvable as exc:
        assert any(m in exc.reason for m in (VENDOR_ABSENT_MARKER, CHECKPOINT_ABSENT_MARKER,
                                            "vendor_build_strix", SHA_MISMATCH_MARKER)), exc.reason
        assert "make vendor" in exc.reason or "vendor_build_strix" in exc.reason or \
            "strix_models" in exc.reason
        return
    bot = factory()
    assert isinstance(bot, BotProtocol)
    bot.close()


@pytest.mark.integration
def test_the_live_driver_plays_twenty_legal_games_end_to_end() -> None:
    """RUNG-2's witness: 20 games through the arena's own loop, zero fence disagreements, zero forfeits."""
    from mantis.arena.match import _play_one_game
    from mantis.bots.resolve import resolve_bot

    available, why = strix_availability()
    if not available:
        pytest.skip(f"LOUD SKIP — strix is not vendored here: {why}")
    factory = resolve_bot("strix", depth=None, opponent_sims=8)
    strix = factory()
    random_bot = RandomBot(seed=1)
    terminals: list[str] = []
    try:
        for game in range(20):
            candidate_color = 1 if game % 2 == 0 else -1
            winner, plies, _moves, terminal, *_rest = _play_one_game(
                random_bot, strix, [], candidate_color=candidate_color,
                board_factory=lambda: Board.with_encoding_name(_ENCODING), max_plies=120,
                opening_id=f"w{game}", adjudicator=None,
            )
            terminals.append(terminal)
    finally:
        strix.close()
    assert "forfeit" not in terminals, terminals
    assert strix.moves >= 20 and strix.fence_disagreements == 0 and strix.out_of_fence == 0
    assert strix.findings == []
