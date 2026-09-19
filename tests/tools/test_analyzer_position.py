"""The position vocabulary: a move list is refused by name or becomes exactly the engine's Board."""
from __future__ import annotations

import importlib
import random

import pytest

from mantis._engine import Board

ENC = "gnn_axis_v1"


@pytest.fixture(scope="module")
def position(analyzer):
    return importlib.import_module("analyzer.position")


def test_the_compact_text_form_round_trips_and_tolerates_spaces_and_json(position):
    moves = [(0, 0), (1, 0), (-2, 3)]
    assert position.parse_moves(position.format_moves(moves)) == moves
    assert position.parse_moves(" 0,0 ; 1, 0;-2,3 ") == moves
    assert position.parse_moves("[[0, 0], [1, 0], [-2, 3]]") == moves
    assert position.parse_moves([[0, 0], [1, 0]]) == [(0, 0), (1, 0)]
    assert position.parse_moves("") == []


@pytest.mark.parametrize("text", ["0,0;x,1", "0,0;1", "[[0, 0], [1]]", "[1, 2]", "[[0, true]]", None, {"a": 1}])
def test_malformed_moves_are_refused_by_name(position, text):
    with pytest.raises(position.PositionRefused, match="move"):
        position.parse_moves(text)


def test_the_owner_cadence_agrees_with_the_engine_over_random_legal_plies(position):
    rng = random.Random(7)
    board = Board.with_encoding_name(ENC)
    for ply in range(40):
        assert position.owner_of_ply(ply) == position.SIDE_OF_INT[int(board.current_player)]
        q, r = rng.choice(board.legal_moves())
        board.apply_move(q, r)
        if board.check_win():
            break


def test_build_board_replays_the_moves_and_reports_the_facts(position):
    pos = position.build_board([(0, 0), (1, 0), (0, 1)], ENC)
    rec = position.position_record(pos)
    assert rec["moves"] == [[0, 0], [1, 0], [0, 1]] and rec["ply"] == 3
    assert rec["to_move"] == "p1" and rec["moves_remaining"] == 2 and rec["winner"] is None
    assert rec["legal"] == len(pos.board.legal_moves()) == len(rec["legal_window"])
    assert rec["win_line"] is None


def test_an_occupied_cell_and_a_cell_outside_the_radius_are_refused_at_their_ply(position):
    with pytest.raises(position.PositionRefused, match=r"illegal move \(0, 0\) at ply 1"):
        position.build_board([(0, 0), (0, 0)], ENC)
    with pytest.raises(position.PositionRefused, match=r"illegal move \(40, 40\) at ply 1"):
        position.build_board([(0, 0), (40, 40)], ENC)


def test_a_six_sets_the_winner_and_a_further_move_is_refused_even_though_apply_move_accepts_it(position, positions):
    pos = position.build_board(positions["SIX"], ENC)
    assert pos.winner == "p1"
    rec = position.position_record(pos)
    assert rec["winner"] == "p1" and sorted(rec["win_line"]) == [[i, 0] for i in range(6)]
    with pytest.raises(position.PositionRefused, match="after the game ended at ply 12"):
        position.build_board(positions["SIX"] + [(6, 1)], ENC)


def test_the_side_table_is_the_engines_own_ints(position):
    assert position.SIDE_OF_INT == {1: "p1", -1: "p2"}


def test_the_pixel_inverse_round_trips_a_ring_of_cells(position):
    for q in range(-6, 7):
        for r in range(-6, 7):
            x, y = position.hex_to_pixel(q, r)
            assert position.pixel_to_hex(x + 0.3, y - 0.2) == (q, r)
