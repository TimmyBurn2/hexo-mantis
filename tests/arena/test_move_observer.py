"""The arena reports every stone it applies, in order, to a player that reads the move ORDER (the Board keeps none)."""
from __future__ import annotations

from typing import Any

from mantis._engine import Board
from mantis.arena.adjudicate import TERMINAL_FORFEIT
from mantis.arena.match import play_paired_match
from mantis.arena.regime import RegimeKey
from _arena_stubs import Opening

_ENCODING = "gnn_axis_r8"
_OPENING = Opening("o1", [(0, 0), (1, 0), (0, 1)])
_KEY = RegimeKey(bot="random", variant="t", model_sims=1, opponent_spec="t", opening_book="b",
                 deploy_matched=False, encoding=_ENCODING)


class _Observer:
    """Plays the first legal move (or `illegal_at`'s far cell on that call) and keeps what the arena reports."""

    def __init__(self, illegal_at: int | None = None) -> None:
        self.games: list[list[tuple[int, int]]] = []
        self.returned: list[tuple[int, int]] = []
        self._illegal_at = illegal_at

    def name(self) -> str:
        return "observer"

    def new_game(self) -> None:
        self.games.append([])

    def observe_move(self, q: int, r: int) -> None:
        self.games[-1].append((q, r))

    def select_move(self, board: Any) -> tuple[int, int]:
        move = tuple(board.legal_moves()[0])
        if self._illegal_at is not None and len(self.returned) == self._illegal_at:
            move = (100, 100)
        self.returned.append(move)
        return move


class _Plain:
    def name(self) -> str:
        return "plain"

    def new_game(self) -> None:
        return None

    def select_move(self, board: Any) -> tuple[int, int]:
        return tuple(board.legal_moves()[0])


def _play(candidate: Any, opponent: Any, max_plies: int = 20) -> list[Any]:
    return play_paired_match(candidate, opponent, [_OPENING], regime_key=_KEY,
                             board_factory=lambda: Board.with_encoding_name(_ENCODING),
                             max_plies=max_plies)


def test_an_observer_sees_the_opening_then_every_stone_of_both_seats_in_order() -> None:
    watcher = _Observer()
    records = _play(_Plain(), watcher)
    assert len(watcher.games) == 2
    for record, seen in zip(records, watcher.games, strict=True):
        assert seen == list(record.moves), "the observed order is the game's own move list"
        assert seen[:3] == _OPENING.moves


def test_both_seats_observe_when_both_can() -> None:
    a, b = _Observer(), _Observer()
    records = _play(a, b)
    assert a.games == b.games == [list(r.moves) for r in records]


def test_a_forfeited_stone_is_never_reported_as_played() -> None:
    watcher = _Observer(illegal_at=1)
    records = _play(_Plain(), watcher)
    forfeit = records[0]
    assert forfeit.terminal == TERMINAL_FORFEIT and forfeit.winner == "candidate"
    assert (100, 100) not in watcher.games[0] and watcher.games[0] == list(forfeit.moves)
