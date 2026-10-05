"""The dash's hex facts: the owner and turn of a ply against the engine, the engine's axes and win length, the six through the last stone."""
from __future__ import annotations

import importlib

import pytest

from mantis import _engine

from _dash_record import six_in_a_row_for_p1


@pytest.fixture(scope="module")
def hexlogic(dash):
    return importlib.import_module("dash.readers.hexlogic")


def test_the_owner_of_a_ply_follows_the_engines_turn_mapping(hexlogic):
    assert [hexlogic.owner(p) for p in range(7)] == [0, 1, 1, 0, 0, 1, 1]


def test_owner_turn_and_turn_start_agree_with_the_engines_board_at_every_position(dash, hexlogic):
    games = importlib.import_module("dash.views.games")
    moves = six_in_a_row_for_p1()[:-1]
    facts = games.turn_facts(len(moves))
    board = _engine.Board()
    for ply, (q, r) in enumerate(moves):
        assert hexlogic.owner(ply) == (0 if board.current_player == 1 else 1), ply
        assert hexlogic.first_of_turn(ply) == (ply == 0 or board.moves_remaining == 2), ply
        board.apply_move(q, r)
    assert facts["owners"] == [hexlogic.owner(p) for p in range(len(moves))]
    assert facts["turn_of"] == [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    assert facts["turn_starts"] == [0, 1, 3, 5, 7, 9]


def test_the_axes_and_win_length_are_the_engines_own(hexlogic):
    assert hexlogic.HEX_AXES == tuple((int(dq), int(dr)) for dq, dr in _engine.HEX_AXES)
    assert hexlogic.WIN_LENGTH == _engine.WIN_LENGTH


def test_the_six_in_a_row_is_found_through_the_completing_stone(hexlogic):
    moves = six_in_a_row_for_p1()
    assert hexlogic.win_line(moves) == [[k, 0] for k in range(6)]
    assert hexlogic.owner(len(moves) - 1) == 0, "the completing stone is p1's"


def test_a_longer_run_is_returned_whole_and_a_diagonal_axis_is_found(hexlogic):
    p1 = [[k, -k] for k in range(6)]
    p2 = [[2 * k, 7 + k % 2] for k in range(6)]
    moves = [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4], p2[4], p2[5], p1[5]]
    assert hexlogic.win_line(moves) == p1


def test_no_line_on_a_board_without_six(hexlogic):
    assert hexlogic.win_line(six_in_a_row_for_p1()[:-1]) is None
    assert hexlogic.win_line([]) is None
