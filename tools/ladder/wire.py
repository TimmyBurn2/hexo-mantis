"""The htttx wire <-> engine `Board` seam: axial `q,r` on the wire IS mantis-core's `(q, r)`, so the map is the identity."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mantis._engine import Board

SIDE_TO_PLAYER = {"x": 1, "o": -1}
PLAYER_TO_SIDE = {1: "x", -1: "o"}

Cell = tuple[int, int]


class WireError(ValueError):
    """A wire board this adapter refuses to play from: it does not replay under the rules."""


def board_from_wire(board: Mapping[str, Any], *, encoding: str) -> Board:
    """Replay the wire's `cells` (placement order, as the server appends) into a fresh engine board. Raises: WireError on no origin stone, a stone on the wrong side for its ply, an occupied cell, a `to_move` that disagrees with the replay, or a mid-turn position."""
    cells = list(board["cells"])
    if not cells:
        raise WireError("the wire board carries no origin stone; the server places it before a bot moves")
    engine = Board.with_encoding_name(encoding)
    for ply, cell in enumerate(cells):
        q, r, side = int(cell["q"]), int(cell["r"]), str(cell["p"])
        expected = PLAYER_TO_SIDE[engine.current_player]
        if side != expected:
            raise WireError(f"ply {ply}: the wire places {side} at ({q}, {r}) but the cadence has {expected} to move")
        if engine.get(q, r) != 0:
            raise WireError(f"ply {ply}: ({q}, {r}) is already occupied on the wire board")
        engine.apply_move(q, r)
    to_move = str(board["to_move"])
    if SIDE_TO_PLAYER.get(to_move) != engine.current_player:
        raise WireError(f"to_move {to_move!r} disagrees with the replayed position "
                        f"({PLAYER_TO_SIDE[engine.current_player]} to move after {len(cells)} stones)")
    if engine.moves_remaining != 2:
        raise WireError(f"mid-turn position after {len(cells)} stones ({engine.moves_remaining} to place); "
                        "every move request wants exactly two placements")
    return engine


def move_response(placements: tuple[Cell, Cell], *, request_id: int | None) -> dict[str, Any]:
    """The htttx `MoveResponse` for two placements, echoing `request_id` when the request carried one."""
    body: dict[str, Any] = {"move": {"pieces": [{"q": int(q), "r": int(r)} for q, r in placements]}}
    if request_id is not None:
        body["request_id"] = int(request_id)
    return body


__all__ = ["Cell", "PLAYER_TO_SIDE", "SIDE_TO_PLAYER", "WireError", "board_from_wire", "move_response"]
