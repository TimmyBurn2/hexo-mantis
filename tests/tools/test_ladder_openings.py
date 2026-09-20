"""`tools/ladder/openings.py` (R363(c)): the ladder's unit is `book_v1_s20260625_p4` PAIRED, opening index = match index, both bots playing the same book prefix from the server's auto-placed origin — a convention between OUR bots, since the server's challenge carries no opening field."""
from __future__ import annotations

import pytest

from mantis.arena.books import book_openings


def test_the_unit_is_book_v1_in_file_order_with_no_seed_and_no_permutation(ladder) -> None:
    book = book_openings("book_v1_s20260625_p4")
    assert ladder.openings.BOOK_ID == "book_v1_s20260625_p4"
    for index in (0, 1, 143, 511):
        opening = ladder.openings.ladder_opening(index)
        assert opening.index == index and opening.opening_id == book[index].opening_id
        q0, r0 = book[index].moves[0]
        assert opening.relative == tuple((q - q0, r - r0) for q, r in book[index].moves)
        assert opening.relative[0] == (0, 0)


def test_a_match_index_past_the_book_is_refused_by_name_not_wrapped(ladder) -> None:
    with pytest.raises(IndexError, match="512"):
        ladder.openings.ladder_opening(512)


def test_the_opening_is_translated_onto_the_servers_origin_stone(ladder) -> None:
    opening = ladder.openings.LadderOpening(book="b", index=0, opening_id="0", relative=((0, 0), (2, -1), (0, -2), (0, 1)))
    assert opening.on_origin((0, 0)) == ((0, 0), (2, -1), (0, -2), (0, 1))
    assert opening.on_origin((5, 3)) == ((5, 3), (7, 2), (5, 1), (5, 4))


def test_the_second_player_plays_plies_two_and_three_and_the_first_player_ply_four_plus_one_searched(ladder) -> None:
    opening = ladder.openings.LadderOpening(book="b", index=0, opening_id="0", relative=((0, 0), (2, -1), (0, -2), (0, 1)))
    assert ladder.openings.forced_stones(opening, [(0, 0)]) == ((2, -1), (0, -2))
    assert ladder.openings.forced_stones(opening, [(0, 0), (2, -1), (0, -2)]) == ((0, 1),)
    assert ladder.openings.forced_stones(opening, [(0, 0), (2, -1), (0, -2), (0, 1), (4, 4)]) == ()
    assert ladder.openings.forced_stones(opening, [(3, 3)]) == ((5, 2), (3, 1))


def test_a_board_that_left_the_book_is_off_book_and_stays_off_book(ladder) -> None:
    opening = ladder.openings.LadderOpening(book="b", index=0, opening_id="0", relative=((0, 0), (2, -1), (0, -2), (0, 1)))
    assert ladder.openings.forced_stones(opening, [(0, 0), (2, -1), (9, 9)]) is None
    assert ladder.openings.forced_stones(opening, [(0, 0), (2, -1), (9, 9), (0, 1), (4, 4)]) is None
    assert ladder.openings.forced_stones(opening, []) is None


def test_every_opening_of_the_book_replays_from_the_origin_under_the_servers_fence(ladder) -> None:
    from mantis._engine import Board

    for index in range(512):
        opening = ladder.openings.ladder_opening(index)
        board = Board.with_encoding_name("gnn_axis_r8")
        for q, r in opening.on_origin((0, 0)):
            assert board.is_legal(q, r), (index, (q, r))
            board.apply_move(q, r)
        assert board.moves_remaining == 1 and board.current_player == 1


def test_the_record_names_the_book_the_index_and_the_relative_stones(ladder) -> None:
    opening = ladder.openings.ladder_opening(7)
    record = opening.to_record()
    assert record["book"] == "book_v1_s20260625_p4" and record["index"] == 7
    assert record["opening_id"] == opening.opening_id and record["relative"] == [list(c) for c in opening.relative]
    assert ladder.openings.LadderOpening.from_record(record) == opening
