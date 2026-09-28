"""The candidate's tactics rows reach its GameRecord summed per game, and the eval record only when the head ran them."""
from __future__ import annotations

from typing import Any

from mantis._engine import Board
from mantis.arena.match import _play_one_game
from mantis.monitor.game_record import eval_record


class _Line:
    """Plays its listed cells in order; `rows` set exposes one tactics row a move, `last_sims` 0 on odd moves."""

    def __init__(self, cells: list[tuple[int, int]], *, rows: bool) -> None:
        self._cells, self._rows, self._n = cells, rows, 0
        self.last_tactics: dict[str, int] | None = None
        self.last_sims: int | None = None

    def new_game(self) -> None:
        self._n = 0

    def select_move(self, board: Any) -> tuple[int, int]:
        cell = self._cells[self._n]
        self._n += 1
        if self._rows:
            self.last_tactics = {"descents": 4, "root_proofs_found": self._n % 2}
            self.last_sims = 0 if self._n % 2 else 4
        return cell


def _game(rows: bool) -> tuple:
    candidate = _Line([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], rows=rows)
    opponent = _Line([(0, 3), (1, 3), (2, 3), (3, 3), (-3, 3), (-4, 3)], rows=False)
    return _play_one_game(candidate, opponent, [], candidate_color=1, board_factory=Board, max_plies=64,
                          opening_id="t", adjudicator=None)


def test_the_candidates_rows_are_summed_over_its_moves_and_its_unsearched_moves_counted() -> None:
    winner, plies, *_rest, tactics = _game(rows=True)
    assert winner == "candidate" and plies == 12
    # Six candidate stones, the sixth making six: three decided (the odd moves), every row summed.
    assert tactics == {"descents": 24, "root_proofs_found": 3, "moves": 6, "decided_moves": 3}


def test_a_head_that_ran_no_tactics_leaves_no_rows_and_no_record_key() -> None:
    *_rest, tactics = _game(rows=False)
    assert tactics is None
    kw: dict[str, Any] = dict(game_id="g", run_id="r", step=1, channel="external", rung="six", phase="rung",
                              game_index=1, moves=[(0, 0)], result="p1", plies=1, termination="win",
                              candidate_color=1, seed=1, served_sims=256)
    assert "candidate_tactics" not in eval_record(**kw)
    assert eval_record(**kw, candidate_tactics={"moves": 1})["candidate_tactics"] == {"moves": 1}
