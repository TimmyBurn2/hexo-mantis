"""The candidate's tactics rows reach its GameRecord summed per game, and the eval record only when the head ran them."""
from __future__ import annotations

from typing import Any

from mantis._engine import Board
from mantis.arena.match import _play_one_game
from mantis.monitor.game_record import eval_record


class _Line:
    """Plays its listed cells in order; `rows` set exposes one tactics row a move (a proof on its odd moves)."""

    def __init__(self, cells: list[tuple[int, int]], *, rows: bool) -> None:
        self._cells, self._rows, self._n = cells, rows, 0
        self.last_tactics: dict[str, int] | None = None

    def new_game(self) -> None:
        self._n = 0

    def select_move(self, board: Any) -> tuple[int, int]:
        cell = self._cells[self._n]
        self._n += 1
        if self._rows:
            self.last_tactics = {"descents": 4, "root_proofs_found": self._n % 2}
        return cell


def _game(rows: bool) -> tuple:
    candidate = _Line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], rows=rows)
    # The opponent exposes rows too: only the candidate's may be summed.
    opponent = _Line([(0, 3), (1, 3), (2, 3), (3, 3), (-3, 3), (-4, 3)], rows=True)
    return _play_one_game(candidate, opponent, [], candidate_color=1, board_factory=Board, max_plies=64,
                          opening_id="t", adjudicator=None)


def test_the_candidates_rows_are_summed_over_its_stones_and_never_the_opponents() -> None:
    winner, plies, *_rest, tactics = _game(rows=True)
    assert winner == "candidate" and plies == 12
    # Six candidate stones, the sixth making six, each a row of 4 descents with a proof on the odd ones.
    assert tactics == {"descents": 24, "root_proofs_found": 3, "stones": 6}


def test_a_head_that_ran_no_tactics_leaves_no_rows_and_no_record_key() -> None:
    candidate = _Line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], rows=False)
    opponent = _Line([(0, 3), (1, 3), (2, 3), (3, 3), (-3, 3), (-4, 3)], rows=True)
    *_rest, tactics = _play_one_game(candidate, opponent, [], candidate_color=1, board_factory=Board, max_plies=64,
                                     opening_id="t", adjudicator=None)
    assert tactics is None, "an opponent's rows never stand in for a candidate that ran none"
    kw: dict[str, Any] = dict(game_id="g", run_id="r", step=1, channel="external", rung="six", phase="rung",
                              game_index=1, moves=[(0, 0)], result="p1", plies=1, termination="win",
                              candidate_color=1, seed=1, served_sims=256)
    assert "candidate_tactics" not in eval_record(**kw)
    assert eval_record(**kw, candidate_tactics={"stones": 1})["candidate_tactics"] == {"stones": 1}
