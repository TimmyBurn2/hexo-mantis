"""`tools/openings/arena_draw.py` is the arena's opening draw, case for case with `packages/rules/test/opening.test.ts`
at cf28a07, and draw for draw with the arena's own `drawOpening` (the fixture is its output, its generator beside it)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from _toolpath import load_module_by_path

from mantis._engine import Board
from mantis.arena.books import book_openings

_REPO = Path(__file__).resolve().parents[2]
A = load_module_by_path("arena_draw_t", _REPO / "tools" / "openings" / "arena_draw.py")
_FIXTURE = _REPO / "tests" / "fixtures" / "arena_draw" / "arena_draws_cf28a07.json"
ORIGIN = (0, 0, 0)


def line(start: tuple[int, int], axis: tuple[int, int], players: list[int | None]) -> list[tuple[int, int, int]]:
    return [(start[0] + k * axis[0], start[1] + k * axis[1], p) for k, p in enumerate(players) if p is not None]


def scripted(cells: list[tuple[int, int]]) -> tuple[Any, list[int]]:
    """Places `cells` in order, resolving each to its index among the region cells still empty."""
    bounds: list[int] = []
    taken: set = set()
    it = iter(cells)

    def index(bound: int) -> int:
        nonlocal taken
        if bound == len(A.OPENING_REGION):
            taken = set()
        bounds.append(bound)
        cell = next(it)
        empty = [c for c in A.OPENING_REGION if c not in taken]
        taken.add(cell)
        return empty.index(cell)
    return index, bounds


def test_the_region_holds_the_18_distinct_cells_at_distance_1_or_2_in_x_then_y_order():
    assert len(A.OPENING_REGION) == 18 == len(set(A.OPENING_REGION))
    assert all(A.hex_distance(c, (0, 0)) in (1, 2) for c in A.OPENING_REGION)
    assert list(A.OPENING_REGION) == sorted(A.OPENING_REGION)


@pytest.mark.parametrize("axis", A.LINE_AXES)
def test_balance_rejects_four_or_five_of_one_player_in_six_cells(axis):
    for player in (0, 1):
        assert not A.is_balanced_opening(line((10, 10), axis, [player, None, player, player, None, player]))
    assert not A.is_balanced_opening(line((-3, 2), axis, [1, 1, 1, None, 1, 1]))


@pytest.mark.parametrize("axis", A.LINE_AXES)
def test_balance_accepts_a_touched_window_a_seven_cell_spread_and_three_in_six(axis):
    assert A.is_balanced_opening(line((0, 0), axis, [1, 1, 0, 1, 1, None]))
    assert A.is_balanced_opening(line((0, 0), axis, [1, 1, None, None, None, 1, 1]))
    assert A.is_balanced_opening(line((0, 0), axis, [0, 0, None, 0, None, None]))


def test_balance_accepts_bent_and_parallel_fours_the_empty_board_and_the_origin():
    assert A.is_balanced_opening(line((0, 0), (1, 0), [1, 1]) + line((1, 1), (0, 1), [1, 1]))
    assert A.is_balanced_opening(line((0, 0), (1, 0), [1, 1]) + line((0, 1), (1, 0), [1, 1]))
    assert A.is_balanced_opening([]) and A.is_balanced_opening([ORIGIN])


def _fault(opening: list[tuple[int, int, int]], plies: int) -> str | None:
    if len(opening) != plies:
        return f"{len(opening)} stones"
    if opening[0] != ORIGIN:
        return "no origin first"
    for ply, (x, y, player) in enumerate(opening):
        if player != A.owner_of_ply(ply):
            return f"ply {ply} owned by {player}"
        if A.hex_distance((x, y), (0, 0)) > 2:
            return f"ply {ply} past distance 2"
    if len({(x, y) for x, y, _ in opening}) != plies:
        return "a cell taken twice"
    return None if A.is_balanced_opening(opening) else "unbalanced"


def test_every_length_draws_that_many_stones_owned_by_ply_within_distance_2_and_balanced():
    rng = A.Mulberry32(0x0BEE)
    faults = [f"{p}/{d}: {f}" for p in (1, 3, 5, 7, 9) for d in range(2000)
              if (f := _fault(A.draw_opening(p, rng.int), p)) is not None]
    assert faults == []


def test_one_ply_is_the_origin_alone_and_never_consults_the_source():
    def refuse(_bound):
        raise AssertionError("consulted")
    assert A.draw_opening(1, refuse) == [ORIGIN]


def test_the_source_is_asked_once_per_ply_over_the_cells_still_empty():
    index, bounds = scripted(A.OPENING_REGION[:4])
    A.draw_opening(5, index)
    assert bounds == [18, 17, 16, 15]


def test_a_rejected_draw_is_discarded_whole_and_redrawn_from_ply_1():
    threat = [(-2, 1), (-1, 1), (1, -1), (2, -1), (0, 1), (1, 1)]
    clean = [(-2, 1), (-1, 1), (0, 1), (2, -1), (1, -1), (1, 1)]
    index, bounds = scripted(threat + clean)
    opening = A.draw_opening(7, index)
    assert bounds == [18, 17, 16, 15, 14, 13] * 2
    assert [(x, y) for x, y, _ in opening[1:]] == clean
    assert A.is_balanced_opening(opening)


@pytest.mark.parametrize("plies", [0, 2, 4, 6, 8, 10, 11, -1, 2.5])
def test_even_lengths_zero_and_lengths_past_nine_are_refused(plies):
    with pytest.raises(ValueError):
        A.draw_opening(plies, A.Mulberry32(1).int)


def _count_rejected(plies: int) -> tuple[int, int]:
    ones_n = ((plies - 1) // 2 + 1) // 2 * 2
    zeros_n = plies - 1 - ones_n
    positions = rejected = 0
    for ones in itertools.combinations(range(18), ones_n):
        rest = [i for i in range(18) if i not in ones]
        for zeros in itertools.combinations(rest, zeros_n):
            stones = [ORIGIN] + [(*A.OPENING_REGION[i], 1) for i in ones] + [(*A.OPENING_REGION[i], 0) for i in zeros]
            positions += 1
            rejected += not A.is_balanced_opening(stones)
    return positions, rejected


def test_no_opening_of_five_plies_or_fewer_is_rejected():
    assert _count_rejected(3) == (153, 0)
    assert _count_rejected(5) == (18_360, 0)


def test_546_of_the_278_460_seven_ply_openings_are_rejected():
    assert _count_rejected(7) == (278_460, 546)


def test_every_draw_equals_the_arenas_own_draw_off_the_same_seed():
    ref = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert [list(c) for c in A.OPENING_REGION] == ref["region"]
    rng = A.Mulberry32(0x0BEE)
    assert [rng.int(18) for _ in range(10)] == ref["ints"]
    checked = 0
    for key, draws in ref.items():
        if key in ("region", "ints"):
            continue
        seed, plies = map(int, key.split("_"))
        rng = A.Mulberry32(seed)
        assert [[list(s) for s in A.draw_opening(plies, rng.int)] for _ in draws] == draws, key
        checked += len(draws)
    assert checked == 941


_BOOK_ID = "arena_s20261006_p5"
_BOOK = _REPO / "src" / "mantis" / "arena" / "books" / f"{_BOOK_ID}.json"


def test_the_cell_book_is_the_minters_output_byte_for_byte(tmp_path):
    out = tmp_path / "book.json"
    assert A.main(["--seed", "20261006", "--plies", "5", "--n", "288", "--out", str(out)]) == 0
    assert out.read_bytes() == _BOOK.read_bytes()


@pytest.mark.parametrize("encoding", ["gnn_axis_r8", "gnn_axis_v1"])
def test_every_book_opening_replays_owned_by_ply_and_hands_player_two_a_whole_turn(encoding: str) -> None:
    openings = book_openings(_BOOK_ID)
    assert len(openings) == 288
    for opening in openings:
        board = Board.with_encoding_name(encoding)
        assert opening.moves[0] == (0, 0)
        for ply, (q, r) in enumerate(opening.moves):
            assert board.is_legal(q, r), (opening.opening_id, ply)
            assert board.current_player == (1 if A.owner_of_ply(ply) == 0 else -1)
            board.apply_move(q, r)
        assert (board.current_player, board.moves_remaining) == (-1, 2)
        assert not board.check_win()
