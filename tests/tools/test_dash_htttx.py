"""The htttx notation: the operator's 74-turn game replays from the origin and ends in Light's six; writing it back is the same text."""
from __future__ import annotations

import importlib

import pytest

#: A game as the operator pasted it from the HeXO site, 2026-10-05: 74 turns after the origin, Light's six on the last.
GAME = (
    "version[1];\n"
    "1. [1,-2][-1,1];\n"
    "2. [0,1][-1,0];\n"
    "3. [-2,0][0,-2];\n"
    "4. [2,-1][1,-3];\n"
    "5. [4,0][4,-3];\n"
    "6. [3,-1][4,-2];\n"
    "7. [-2,3][2,0];\n"
    "8. [2,-5][2,-6];\n"
    "9. [0,3][-1,3];\n"
    "10. [-2,4][1,3];\n"
    "11. [-5,2][-5,3];\n"
    "12. [-4,3][-4,2];\n"
    "13. [-5,1][-5,4];\n"
    "14. [-5,5][-5,0];\n"
    "15. [-8,4][-4,1];\n"
    "16. [-6,1][-6,3];\n"
    "17. [-7,3][-6,2];\n"
    "18. [-3,-1][-9,5];\n"
    "19. [-6,4][-7,4];\n"
    "20. [-3,4][-9,4];\n"
    "21. [-2,-2][-2,-1];\n"
    "22. [-2,1][-1,-2];\n"
    "23. [-7,2][-7,5];\n"
    "24. [-7,0][-7,6];\n"
    "25. [5,-1][3,0];\n"
    "26. [5,0][6,-2];\n"
    "27. [2,-4][-1,-4];\n"
    "28. [-2,-4][-3,-3];\n"
    "29. [-6,0][-8,6];\n"
    "30. [-9,7][-8,2];\n"
    "31. [0,-6][-9,3];\n"
    "32. [-4,-3][-4,7];\n"
    "33. [-3,-4][-4,6];\n"
    "34. [-8,-1][-3,7];\n"
    "35. [-6,-3][-3,8];\n"
    "36. [-6,9][-9,-1];\n"
    "37. [-10,-1][-6,7];\n"
    "38. [-8,-2][-5,8];\n"
    "39. [-3,6][-8,0];\n"
    "40. [-5,10][-5,9];\n"
    "41. [-5,7][-6,10];\n"
    "42. [-5,11][-7,9];\n"
    "43. [-8,9][-5,12];\n"
    "44. [-2,6][-8,8];\n"
    "45. [-7,8][-1,5];\n"
    "46. [-1,4][-9,9];\n"
    "47. [1,4][-9,8];\n"
    "48. [-9,10][-9,-2];\n"
    "49. [-10,-2][-7,7];\n"
    "50. [-6,-1][-12,12];\n"
    "51. [-10,0][-9,12];\n"
    "52. [-10,1][-14,12];\n"
    "53. [-13,12][-9,-3];\n"
    "54. [-12,10][-5,-2];\n"
    "55. [-9,2][-10,-3];\n"
    "56. [-10,-4][-7,-1];\n"
    "57. [-7,-3][-5,-1];\n"
    "58. [-8,-3][0,-3];\n"
    "59. [-11,11][-1,-3];\n"
    "60. [-14,10][-8,-4];\n"
    "61. [-14,11][-8,-6];\n"
    "62. [-15,11][-7,-2];\n"
    "63. [-4,-2][-11,9];\n"
    "64. [-15,10][-4,9];\n"
    "65. [-13,10][-2,9];\n"
    "66. [-13,9][-12,8];\n"
    "67. [-16,12][-11,7];\n"
    "68. [-12,7][-12,9];\n"
    "69. [-12,11][-12,6];\n"
    "70. [-15,8][-15,9];\n"
    "71. [-15,7][-15,12];\n"
    "72. [-14,9][-13,8];\n"
    "73. [-16,9][-17,12];\n"
    "74. [-16,11][-11,6];\n"
)


@pytest.fixture(scope="module")
def htttx(dash):
    return importlib.import_module("dash.readers.htttx")


def test_the_game_replays_from_the_origin_and_ends_in_lights_six(htttx):
    from mantis._engine import Board

    moves = htttx.parse(GAME)
    assert len(moves) == 149 and moves[:3] == [(0, 0), (1, -2), (-1, 1)]
    board = Board.with_encoding_name(htttx.SITE_RULES)
    for q, r in moves:
        board.apply_move(q, r)
    assert board.winner() == 1


def test_htttx_cells_are_the_wire_cells_the_ladder_reads_off_the_sites_records(htttx, ladder):
    # The site's own htttx parser places [q,r] at its board's (q + r, -r); the receipt maps that record back to the wire.
    moves = htttx.parse(GAME)
    record = {"moves": [{"moveNumber": i, "x": q + r, "y": -r, "playerId": "p1" if i == 0 else "p2"}
                        for i, (q, r) in enumerate(moves)]}
    assert [tuple(m[:2]) for m in ladder.receipt._wire_moves(record)] == moves


def test_writing_a_parsed_game_gives_back_the_same_text(htttx):
    assert htttx.write(htttx.parse(GAME)) == GAME


def test_a_position_ending_mid_turn_writes_a_last_turn_of_one_stone_and_reads_back(htttx):
    text = htttx.write([(0, 0), (1, 0), (2, 0), (3, 0)])
    assert text == "version[1];\n1. [1,0][2,0];\n2. [3,0];\n"
    assert htttx.parse(text) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    with pytest.raises(htttx.NotationRefused, match="a turn after the first stone"):
        htttx.write([(0, 0)])


def test_a_game_that_does_not_open_on_the_origin_is_moved_there_whole(htttx):
    game = [(-1, -1), (0, -1), (2, 0), (-1, 0)]
    assert htttx.moved(game) and not htttx.moved([(0, 0), (1, 0)])
    assert htttx.write(game) == "version[1];\n1. [1,0][3,1];\n2. [0,1];\n"
    assert htttx.parse(htttx.write(game)) == [(q + 1, r + 1) for q, r in game]


def test_each_turns_line_carries_its_plies_whole_and_first_stone_alone(htttx):
    assert htttx.turns([(0, 0), (1, 0), (2, 0), (3, 0)]) == [(1, 3, "1. [1,0][2,0];", "1. [1,0];"), (3, 4, "2. [3,0];", "2. [3,0];")]


@pytest.mark.parametrize("text", [
    "version[1]; 1. [1,0][2,0]; 2. [3,0][4,0];",
    "version[1];\r\n1. [1,0][2,0];\r\n\r\n2. [3,0][4,0];\r\n",
    "```\nversion[1];\n1. [1,0][2,0];\n2. [3,0][4,0];\n```",
    "\ufeffVersion [1];\n1.[1,0] [2,0];\n2. [ 3 , 0 ][4,0]",
    "1. [1,0][2,0]; 2. [3,0][4,0];",
])
def test_the_sites_statement_form_reads_however_it_was_pasted(htttx, text):
    assert htttx.parse(text) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


@pytest.mark.parametrize(("text", "why"), [
    ("version[2];\n1. [1,0][2,0];", "only htttx version 1"),
    ("version[1];", "no turn"),
    ("", "no turn"),
    ("1. [1,0][2,0];\n3. [4,0][5,0];", "numbered 3"),
    ("1. [1,0][2,0];\n2. hello;", "is not"),
    ("1. [1,0][1,0];", "already occupied"),
    ("1. [0,0][1,0];", "already occupied"),
    ("1. [1,0];\n2. [2,0][3,0];", "only the last turn"),
    ("1. [1,0][2,0][3,0];", "places 3 stones"),
    ("1. [9,0][1,0];", "more than 8 cells from every stone"),
    ("1. [4,4][-5,-4];", r"\(-5, -4\) is more than 8"),
    ("1. [99999999999,0][1,0];", "is not"),
])
def test_a_text_that_does_not_replay_is_refused_by_turn_and_rule(htttx, text, why):
    with pytest.raises(htttx.NotationRefused, match=why):
        htttx.parse(text)


def test_a_stone_after_the_six_is_refused(htttx):
    with pytest.raises(htttx.NotationRefused, match="already won"):
        htttx.parse(GAME + "75. [20,20][21,21];\n")


def test_a_game_past_the_sites_stone_cap_is_refused(htttx):
    line = "".join(f"{n}. [{2 * n - 1},0][{2 * n},0];" for n in range(1, htttx.MAX_STONES // 2 + 1))
    with pytest.raises(htttx.NotationRefused, match="more than 2000 stones"):
        htttx.parse(line)
