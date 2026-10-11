"""The ring audit's future-field rows: a field read against its game's next row in the same ring."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mantis._engine import Board, HexgBuffer, future_horizon
from mantis.diagnostics import ring_audit as A
from mantis.diagnostics.ring_reader import load_ring

_ENC = "gnn_axis_v1"
#: Thirty placements spiralling out from the origin, every one legal where it is played.
_MOVES = [(q, r) for k in range(1, 4) for q, r in ((k, 0), (0, k), (-k, k), (-k, 0), (0, -k), (k, -k))] + [
    (2, 1), (1, 2), (-1, 3), (-3, 1), (-2, -1), (1, -3), (3, -2), (2, -3), (-1, -2), (-2, 3), (3, -1), (1, 1)]


def _game() -> list[dict[str, Any]]:
    """Each ply's row of the scripted game, its field its next placements, read off a replayed board."""
    board, placed = Board(), []
    for q, r in _MOVES:
        placed.append((q, r, board.current_player))
        board.apply_move(q, r)
    turns = (len(_MOVES) + 1) // 2
    rows, board = [], Board()
    for ply, (q, r) in enumerate(_MOVES):
        rows.append({"stones": placed[:ply], "player": board.current_player, "remaining": board.moves_remaining,
                     "ply": ply, "turns": turns, "future": placed[ply:ply + future_horizon()]})
        board.apply_move(q, r)
    return rows


def _ring(tmp_path: Path, games: dict[int, list[dict[str, Any]]]) -> Any:
    buf = HexgBuffer(256, _ENC, 16)
    for gid, rows in games.items():
        for row in rows:
            buf.push_graph_position(row["stones"], [(9, 9, 1.0)], row["player"], row["remaining"], row["ply"], True,
                                    1.0, True, row["turns"], game_id=gid, future_stones=row["future"])
    path = tmp_path / f"ring_{len(games)}.hexg"
    buf.save_to_path(str(path))
    return load_ring(path)


def test_a_games_own_fields_read_no_disagreement(tmp_path: Path) -> None:
    rows = _game()
    ring = _ring(tmp_path, {7: rows[::2], 8: rows[1::3]})
    checked, bad = A.future_disagreements(ring)
    assert checked == (len(rows[::2]) - 1) + (len(rows[1::3]) - 1) and bad.size == 0
    got = {row.key: row for row in A.future_rows(ring)}
    assert got["future_field_share"].value == 1.0 and got["future_disagree"].value == 0.0


@pytest.mark.parametrize("plant", ["swap_order", "swap_turns"])
def test_a_row_whose_field_disagrees_with_its_game_is_named(tmp_path: Path, plant: str) -> None:
    """PLANTED BREAK: a field the push guard admits (owners in turn) but its game contradicts reds both pairs it is in."""
    rows = _game()
    bad = dict(rows[11])  # the mover holds both stones of its turn, so a swap within it keeps the owners in turn
    f = list(bad["future"])
    if plant == "swap_order":
        f[0], f[1] = f[1], f[0]
    else:
        f[0], f[2] = (f[2][0], f[2][1], f[0][2]), (f[0][0], f[0][1], f[2][2])
    bad["future"] = f
    ring = _ring(tmp_path, {3: rows[:11] + [bad] + rows[12:]})
    _checked, pairs = A.future_disagreements(ring)
    assert [[int(ring.ply_index[i]) for i in pair] for pair in pairs] == [[10, 11], [11, 12]]
    assert {row.key: row for row in A.future_rows(ring)}["future_disagree"].value == 2.0


def test_a_ring_without_the_field_reads_none_and_says_why(tmp_path: Path) -> None:
    buf = HexgBuffer(16, _ENC, 16)
    for row in _game()[:4]:
        buf.push_graph_position(row["stones"], [(9, 9, 1.0)], row["player"], row["remaining"], row["ply"], True,
                                1.0, True, row["turns"], game_id=1)
    path = tmp_path / "bare.hexg"
    buf.save_to_path(str(path))
    got = {row.key: row for row in A.future_rows(load_ring(path))}
    assert got["future_field_share"].value == 0.0
    assert got["future_disagree"].value is None and "no game" in got["future_disagree"].note
