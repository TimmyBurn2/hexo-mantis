"""R345(b)(2) — the arena must not score a move it never checked against the legal set.

WHAT WAS MISSING, AND IT IS MISSING IN TWO LAYERS. `_play_one_game` called
`board.apply_move(q, r)` for every replayed opening move and every `select_move` result,
consulting `legal_move_count()` only as a loop-termination test. The board underneath did not
close the gap: `Board::apply_move` rejects an OCCUPIED cell and nothing else, so a coordinate
outside the encoding's `legal_move_radius` ball was accepted, played on, and scored. The
bridge's own docstring said *"Raises ValueError if the move is illegal"*, which is true of
occupancy and false of the legal set — so a reader checking whether the arena was guarded
would have found a sentence saying it was.

WHY IT MATTERS ON THE PROMOTION CHANNEL SPECIFICALLY. Both players there are deploy heads
taking argmax over a policy the search produced; a head whose legal projection drifts from the
board's — different radius, a stale window centre, an off-window cell reached through the
overflow map — plays a move the search believes in and the rules do not. The game continues,
the result feeds `aggregate_gate`, and the promotion bar reads a win that was never legal.
Nothing in the record would say so.

THE TWO DISPOSITIONS ARE DIFFERENT ON PURPOSE. A PLAYER's illegal move is a forfeit — it is
that player's defect, the game has a correct result (they lose), and the run continues so a
single bad head cannot take down a block. A BOOK's illegal move is fatal: the corpus is shared
by every game in the round, and a book that does not replay is not a bad game, it is a bad
instrument.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board
from mantis.arena.adjudicate import TERMINAL_FORFEIT, TERMINAL_REASONS
from mantis.arena.match import IllegalOpeningError, play_paired_match
from mantis.arena.regime import RegimeKey

_REPO = Path(__file__).resolve().parents[2]
_ENCODING = "v6_live2_ls"

#: Far outside any `legal_move_radius` ball reachable from an opening near the origin — the
#: legal set is the union of radius-R balls around PLACED STONES, so distance from the stones
#: is what makes a coordinate illegal, and this one is 100 cells away from all of them.
_FAR_AWAY = (100, 100)


@dataclass(frozen=True)
class _Opening:
    opening_id: str
    moves: list


def _board_factory() -> Any:
    return Board.with_encoding_name(_ENCODING)


class _FirstLegalBot:
    """Plays the first legal move. The control arm."""

    def new_game(self) -> None:
        return None

    def select_move(self, board: Any) -> tuple[int, int]:
        return board.legal_moves()[0]

    def name(self) -> str:
        return "first_legal"


class _IllegalOnNthMoveBot:
    """Plays legally until its `n`-th move, then returns a coordinate outside the legal set.

    The defect this stands for is a head whose legal projection has drifted from the board's,
    which is a real and undramatic way for a bot to emit an unplayable coordinate — not a
    contrived one. Nothing about the return path is faked: it is an ordinary
    `select_move` returning an ordinary tuple.
    """

    def __init__(self, n: int = 1, coord: tuple[int, int] = _FAR_AWAY) -> None:
        self._n = n
        self._coord = coord
        self._i = 0

    def new_game(self) -> None:
        self._i = 0

    def select_move(self, board: Any) -> tuple[int, int]:
        self._i += 1
        if self._i == self._n:
            return self._coord
        return board.legal_moves()[0]

    def name(self) -> str:
        return "illegal_on_nth"


def _regime() -> RegimeKey:
    return RegimeKey(
        bot="first_legal", variant="uniform", model_sims=1,
        opponent_spec="first_legal:uniform", opening_book="inline",
        deploy_matched=False, encoding=_ENCODING,
    )


def _play(candidate: Any, opponent: Any, openings: list[_Opening], **kw: Any) -> list:
    return play_paired_match(
        candidate, opponent, openings, regime_key=_regime(),
        board_factory=_board_factory, max_plies=16, **kw,
    )


# ── a player's illegal move is a forfeit ────────────────────────────────────────────────
def test_a_candidate_playing_off_the_legal_set_forfeits_and_the_move_is_not_applied() -> None:
    records = _play(_IllegalOnNthMoveBot(n=2), _FirstLegalBot(),
                    [_Opening("op0", [(0, 0), (1, 0)])])

    forfeits = [r for r in records if r.terminal == TERMINAL_FORFEIT]
    assert forfeits, (
        f"no game ended in a forfeit; terminals were {[r.terminal for r in records]}. The "
        "illegal coordinate was applied to the board and the game was scored on it."
    )
    for record in forfeits:
        assert record.winner == "opponent", (
            "the forfeiting side was not scored a loss — a player that cannot produce a "
            "legal move must not be able to draw or win by producing an illegal one"
        )
        assert _FAR_AWAY not in record.moves, (
            "the illegal coordinate is in the move list, so it WAS applied; the forfeit was "
            "recorded after the fact rather than instead of the move"
        )


def test_an_opponent_playing_off_the_legal_set_forfeits_to_the_candidate() -> None:
    """The boundary is seat-symmetric — it is a property of the move, not of who made it."""
    records = _play(_FirstLegalBot(), _IllegalOnNthMoveBot(n=1),
                    [_Opening("op0", [(0, 0), (1, 0)])])
    forfeits = [r for r in records if r.terminal == TERMINAL_FORFEIT]
    assert forfeits, "an illegal OPPONENT move was played and scored"
    assert all(r.winner == "candidate" for r in forfeits)


def test_the_forfeit_is_in_the_closed_terminal_vocabulary() -> None:
    """A record's `terminal` is a CLOSED set; a fourth value that is not in it makes every
    consumer's exhaustive branch fall through silently."""
    assert TERMINAL_FORFEIT in TERMINAL_REASONS


def test_an_occupied_cell_forfeits_too_rather_than_raising_out_of_the_bridge() -> None:
    """Occupancy is the ONE illegality the board already caught, and it caught it by raising
    a bare `ValueError` out of the FFI — which halts the whole match rather than the game.
    Routing it through the same boundary makes one disposition for one class of defect."""
    records = _play(_IllegalOnNthMoveBot(n=1, coord=(0, 0)), _FirstLegalBot(),
                    [_Opening("op0", [(0, 0), (1, 0)])])
    assert any(r.terminal == TERMINAL_FORFEIT for r in records), (
        "replaying an occupied cell did not forfeit — it either raised out of the bridge or "
        "was somehow accepted"
    )


# ── a book's illegal move is fatal ──────────────────────────────────────────────────────
def test_an_opening_that_does_not_replay_is_a_fatal_corpus_error() -> None:
    """Not a forfeit: the book is shared by every game in the round.

    A forfeit here would score each affected game a loss for whichever side happened to be
    the candidate, and the colour-swap law means that is BOTH sides in turn — so a broken
    book would produce a perfectly balanced 50% win rate and read as a healthy instrument.
    """
    with pytest.raises(IllegalOpeningError) as excinfo:
        _play(_FirstLegalBot(), _FirstLegalBot(),
              [_Opening("op_broken", [(0, 0), _FAR_AWAY])])
    message = str(excinfo.value)
    assert "op_broken" in message, "the error does not name the opening that failed to replay"
    assert "100" in message, "the error does not name the move that was rejected"


def test_an_opening_replaying_onto_its_own_stone_is_the_same_fatal_error() -> None:
    with pytest.raises(IllegalOpeningError):
        _play(_FirstLegalBot(), _FirstLegalBot(),
              [_Opening("op_dupe", [(0, 0), (0, 0)])])


# ── mutation half: the boundary must not fire on a legal game ───────────────────────────
def test_a_wholly_legal_match_is_untouched_by_the_boundary() -> None:
    """Without this, `forfeit everything` and `raise on every opening` pass every test above.

    Both arms are asserted: no game ends in a forfeit, and every game still ends in one of
    the three pre-existing terminals with a real move list behind it.
    """
    records = _play(_FirstLegalBot(), _FirstLegalBot(),
                    [_Opening("op0", [(0, 0), (1, 0)]), _Opening("op1", [(0, 0), (0, 1)])])

    assert len(records) == 4, "the paired-colour law did not produce two games per opening"
    for record in records:
        assert record.terminal != TERMINAL_FORFEIT, (
            "a legal game was forfeited — the boundary is refusing moves the rules allow"
        )
        assert record.terminal in TERMINAL_REASONS
        assert record.plies >= 2, "the opening was not replayed"


def test_the_legal_set_query_agrees_with_the_engine_move_list() -> None:
    """The boundary's authority is the engine's own legal set, not a re-derivation of it.

    `is_legal` is an O(1) membership test on the same `FxHashSet` `legal_moves()` collects
    from; a second implementation of the radius arithmetic here would be a second authority
    over the rules, which is the class R1 exists to close.
    """
    board = _board_factory()
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    listed = set(board.legal_moves())
    assert listed, "the engine reported no legal moves after two stones"
    for move in list(listed)[:25]:
        assert board.is_legal(*move), f"{move} is in legal_moves() but is_legal says no"
    assert not board.is_legal(*_FAR_AWAY), "a far-off cell is reported legal"
    assert not board.is_legal(0, 0), "an occupied cell is reported legal"
