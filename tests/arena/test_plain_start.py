"""A bookless game starts from the arena's 1-ply draw, the origin stone alone, never from an empty board."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from test_legality_boundary import _FirstLegalBot, _Opening, _play

from mantis.arena.books import PLAIN_START

_REPO = Path(__file__).resolve().parents[2]


class _AwayFirstBot(_FirstLegalBot):
    """Opens off the origin when handed an empty board, so a game that starts empty shows it."""

    def select_move(self, board: Any) -> tuple[int, int]:
        return (2, -1) if not board.get_stones() else super().select_move(board)


def test_a_game_with_no_opening_stones_starts_from_the_origin_alone() -> None:
    records = _play(_AwayFirstBot(), _AwayFirstBot(), [_Opening("plain", [])])
    assert records and all(tuple(r.moves[0]) == (0, 0) for r in records)


def test_the_plain_start_is_the_arena_protocols_one_ply_draw() -> None:
    spec = importlib.util.spec_from_file_location("arena_draw_plain", _REPO / "tools" / "openings" / "arena_draw.py")
    assert spec is not None and spec.loader is not None
    draw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draw)
    one_ply = draw.draw_opening(1, lambda bound: 0)
    assert tuple((x, y) for x, y, _ in one_ply) == PLAIN_START == ((0, 0),)
