"""The forced-move tactics oracle: the FORCED_MOVE_CENSUS's hand-built positions, as rows."""
from __future__ import annotations

import numpy as np
import pytest

from mantis.diagnostics import tactics as T

# Axial coordinates, three axes, 6-in-a-row, 2-stone turns; -1 is the opponent, 1 the mover.


def _pos(cells: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(cells, dtype=np.int64).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


_FOUR = [(0, 0, -1), (1, 0, -1), (2, 0, -1), (3, 0, -1), (0, 5, 1)]
_FIVE = [(i, 0, -1) for i in range(5)] + [(0, 5, 1)]
_CAPPED_FIVE = [(i, 0, -1) for i in range(5)] + [(-1, 0, 1)]
_MOVER_FIVE = [(i, 0, 1) for i in range(5)] + [(0, 5, -1)]
_MOVER_FOUR = [(i, 0, 1) for i in range(4)] + [(0, 5, -1)]
_TWO_FIVES = _CAPPED_FIVE + [(i, 10, -1) for i in range(5)] + [(-1, 10, 1)]
_OVERLAP = _CAPPED_FIVE + [(5, j, -1) for j in range(1, 6)] + [(5, 6, 1)]
_WIN_BEATS_BLOCK = _CAPPED_FIVE + [(i, 3, 1) for i in range(5)] + [(-1, 3, -1)]


def test_an_opponent_four_is_check_with_no_mover_win() -> None:
    t = T.analyze(*_pos(_FOUR), 1, 1)
    assert t.check and not t.w1


def test_a_four_with_three_windows_is_lost_at_k1_because_no_single_cell_hits_all() -> None:
    t = T.analyze(*_pos(_FOUR), 1, 1)
    assert len(t.fours) == 3 and t.block == set()
    assert t.forced(1)[0] == "lost1"


def test_a_four_at_k2_has_every_end_cell_as_a_strict_first_stone_and_no_any_safe() -> None:
    t = T.analyze(*_pos(_FOUR), 1, 2)
    assert t.block_strict == {(-2, 0), (-1, 0), (4, 0), (5, 0)}
    assert t.block_any is False


def test_an_open_five_names_both_completing_cells() -> None:
    t = T.analyze(*_pos(_FIVE), 1, 1)
    assert t.opp_fives == {(-1, 0), (5, 0)}


def test_an_open_five_is_lost_at_k1_and_blocked_by_both_cells_at_k2() -> None:
    assert T.analyze(*_pos(_FIVE), 1, 1).forced(1)[0] == "lost1"
    t = T.analyze(*_pos(_FIVE), 1, 2)
    assert t.block_strict == {(-1, 0), (5, 0)} and t.block == t.block_strict


def test_a_capped_five_is_a_forced_block_on_its_one_cell() -> None:
    assert T.analyze(*_pos(_CAPPED_FIVE), 1, 1).forced(1) == ("block", {(5, 0)})


def test_a_mover_five_is_a_win_on_either_end() -> None:
    t = T.analyze(*_pos(_MOVER_FIVE), 1, 1)
    assert t.w1 == {(-1, 0), (5, 0)} and t.forced(1)[0] == "win"


def test_a_mover_four_is_a_two_stone_win_at_k2_and_quiet_at_k1() -> None:
    t = T.analyze(*_pos(_MOVER_FOUR), 1, 2)
    assert t.w2 == {(-2, 0), (-1, 0), (4, 0), (5, 0)} and not t.w1
    assert T.analyze(*_pos(_MOVER_FOUR), 1, 1).forced(1)[0] == "quiet"


def test_two_disjoint_capped_fives_are_lost_at_k1_and_blocked_at_k2() -> None:
    assert T.analyze(*_pos(_TWO_FIVES), 1, 1).forced(1)[0] == "lost1"
    t = T.analyze(*_pos(_TWO_FIVES), 1, 2)
    assert t.block == {(5, 0), (5, 10)} and t.block_strict == t.block


def test_two_fives_sharing_a_cell_are_blocked_there_at_k1() -> None:
    assert T.analyze(*_pos(_OVERLAP), 1, 1).forced(1) == ("block", {(5, 0)})


def test_two_fives_sharing_a_cell_make_any_first_stone_safe_at_k2() -> None:
    t = T.analyze(*_pos(_OVERLAP), 1, 2)
    assert t.block_any is True
    assert t.block_strict == {(5, 0), (6, 0), (5, -1)}


def test_a_win_beats_a_block() -> None:
    assert T.analyze(*_pos(_WIN_BEATS_BLOCK), 1, 1).forced(1) == ("win", {(5, 3)})


@pytest.mark.parametrize(("q", "r", "want"), [
    (np.array([0]), np.array([0]), 216),
    (np.zeros(0, int), np.zeros(0, int), 25),
])
def test_legal_count_matches_the_r8_ball_and_the_empty_board(q, r, want) -> None:
    assert T.legal_count(q, r) == want


def test_an_empty_position_is_quiet() -> None:
    t = T.analyze(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), 1, 2)
    assert t.forced(2) == ("quiet", set()) and t.n_legal == 25


def test_hitting_sets_with_no_fours_is_empty() -> None:
    assert T.hitting_sets([], 2) == (set(), set(), False)
