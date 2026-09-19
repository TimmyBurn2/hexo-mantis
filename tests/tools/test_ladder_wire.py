"""`tools/ladder/wire.py` (LADDER-1): the server's htttx board replays into the engine `Board` 1:1, cadence-checked."""
from __future__ import annotations

import pytest

_ENC = "gnn_axis_r8"


def _cells(*stones: tuple[int, int, str]) -> list[dict[str, object]]:
    return [{"q": q, "r": r, "p": p} for q, r, p in stones]


def test_a_wire_board_replays_into_the_engine_board_stone_for_stone(ladder) -> None:
    board = ladder.wire.board_from_wire(
        {"to_move": "x", "cells": _cells((0, 0, "x"), (1, 0, "o"), (0, 1, "o"))}, encoding=_ENC)
    assert sorted(board.get_stones()) == [(0, 0, 1), (0, 1, -1), (1, 0, -1)]
    assert board.current_player == 1
    assert board.moves_remaining == 2
    assert board.legal_move_radius() == 8


def test_a_full_compound_turn_lands_on_o_with_two_to_place(ladder) -> None:
    board = ladder.wire.board_from_wire(
        {"to_move": "o", "cells": _cells((0, 0, "x"), (1, 0, "o"), (0, 1, "o"), (-1, 0, "x"), (2, 0, "x"))},
        encoding=_ENC)
    assert board.current_player == -1
    assert board.moves_remaining == 2
    assert len(board.get_stones()) == 5


def test_a_stone_on_the_wrong_side_of_the_cadence_is_refused_by_ply(ladder) -> None:
    with pytest.raises(ladder.wire.WireError, match="ply 1"):
        ladder.wire.board_from_wire(
            {"to_move": "o", "cells": _cells((0, 0, "x"), (1, 0, "x"), (0, 1, "o"))}, encoding=_ENC)


def test_a_to_move_that_disagrees_with_the_replayed_board_is_refused(ladder) -> None:
    with pytest.raises(ladder.wire.WireError, match="to_move"):
        ladder.wire.board_from_wire(
            {"to_move": "o", "cells": _cells((0, 0, "x"), (1, 0, "o"), (0, 1, "o"))}, encoding=_ENC)


def test_a_board_without_the_origin_stone_is_refused(ladder) -> None:
    with pytest.raises(ladder.wire.WireError, match="origin"):
        ladder.wire.board_from_wire({"to_move": "x", "cells": []}, encoding=_ENC)


def test_a_far_opponent_stone_inside_the_server_radius_replays(ladder) -> None:
    # The server allows radius 8 from any stone; gnn_axis_r8 shares that radius, so (8, 0) replays.
    board = ladder.wire.board_from_wire(
        {"to_move": "x", "cells": _cells((0, 0, "x"), (8, 0, "o"), (0, 1, "o"))}, encoding=_ENC)
    assert (8, 0, -1) in board.get_stones()


def test_an_occupied_cell_on_the_wire_is_refused(ladder) -> None:
    with pytest.raises(ladder.wire.WireError, match="occupied"):
        ladder.wire.board_from_wire(
            {"to_move": "x", "cells": _cells((0, 0, "x"), (0, 0, "o"), (0, 1, "o"))}, encoding=_ENC)


def test_two_placements_become_the_htttx_move_response_with_the_request_id_echoed(ladder) -> None:
    body = ladder.wire.move_response(((1, -1), (-2, 3)), request_id=7)
    assert body == {"move": {"pieces": [{"q": 1, "r": -1}, {"q": -2, "r": 3}]}, "request_id": 7}


def test_a_request_without_an_id_gets_a_response_without_one(ladder) -> None:
    body = ladder.wire.move_response(((1, -1), (-2, 3)), request_id=None)
    assert body == {"move": {"pieces": [{"q": 1, "r": -1}, {"q": -2, "r": 3}]}}


def test_a_mid_turn_position_is_refused_because_the_server_wants_exactly_two(ladder) -> None:
    with pytest.raises(ladder.wire.WireError, match="mid-turn"):
        ladder.wire.board_from_wire({"to_move": "o", "cells": _cells((0, 0, "x"), (1, 0, "o"))}, encoding=_ENC)
