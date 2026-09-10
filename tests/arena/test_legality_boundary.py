"""The arena must not score a move it never checked against the legal set.

The two dispositions differ on purpose: a PLAYER's illegal move is a forfeit (that player's
defect, the game has a correct result, the run continues), while a BOOK's illegal move is
fatal because the corpus is shared by every game in the round.
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
_ENCODING = "gnn_axis_v1"

#: The legal set is the union of radius-R balls around placed stones, so this is 100 cells
#: outside every one of them.
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
    """Play legally until the `n`-th move, then return a coordinate outside the legal set."""

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
    """`terminal` is a closed set, so a value outside it makes exhaustive branches fall through."""
    assert TERMINAL_FORFEIT in TERMINAL_REASONS


def test_an_occupied_cell_forfeits_too_rather_than_raising_out_of_the_bridge() -> None:
    """Occupancy forfeits like any other illegality, rather than raising out of the FFI and
    halting the whole match."""
    records = _play(_IllegalOnNthMoveBot(n=1, coord=(0, 0)), _FirstLegalBot(),
                    [_Opening("op0", [(0, 0), (1, 0)])])
    assert any(r.terminal == TERMINAL_FORFEIT for r in records), (
        "replaying an occupied cell did not forfeit — it either raised out of the bridge or "
        "was somehow accepted"
    )


def test_an_opening_that_does_not_replay_is_a_fatal_corpus_error() -> None:
    """Not a forfeit: under the colour-swap law a broken book would forfeit both sides in turn
    and read as a healthy balanced 50% win rate."""
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


def test_a_wholly_legal_match_is_untouched_by_the_boundary() -> None:
    """Mutation half: without it, `forfeit everything` passes every test above."""
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
    """The boundary's authority is the engine's own legal set, not a re-derivation of it."""
    board = _board_factory()
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    listed = set(board.legal_moves())
    assert listed, "the engine reported no legal moves after two stones"
    for move in list(listed)[:25]:
        assert board.is_legal(*move), f"{move} is in legal_moves() but is_legal says no"
    assert not board.is_legal(*_FAR_AWAY), "a far-off cell is reported legal"
    assert not board.is_legal(0, 0), "an occupied cell is reported legal"
