"""The hex facts the viewer derives from a move list: the owner of a ply and the winning line."""
from __future__ import annotations

from mantis._engine import HEX_AXES as _ENGINE_HEX_AXES
from mantis._engine import WIN_LENGTH as _ENGINE_WIN_LENGTH

#: The three line axes in axial coordinates, read from the engine's own export.
HEX_AXES: tuple[tuple[int, int], ...] = tuple((int(dq), int(dr)) for dq, dr in _ENGINE_HEX_AXES)
#: Stones in a row that win, the engine's own constant.
WIN_LENGTH: int = int(_ENGINE_WIN_LENGTH)


def owner(ply: int) -> int:
    """0 = p1, 1 = p2. Ply 0 is p1's single; plies 2k-1, 2k are turn k (`Ply::turn`), turns alternate."""
    return ((ply + 1) // 2) % 2


def win_line(moves: list[list[int]]) -> list[list[int]] | None:
    """The whole run of >= WIN_LENGTH stones through the LAST (completing) stone on one axis, or None; the scan stays local because `Board.find_winning_line` falls back to any stone's line, which would mask exactly the disagreement this function exists to catch."""
    if not moves:
        return None
    last = len(moves) - 1
    side = owner(last)
    mine = {(q, r) for ply, (q, r) in enumerate(moves) if owner(ply) == side}
    q0, r0 = moves[last]
    for dq, dr in HEX_AXES:
        run = [(q0, r0)]
        q, r = q0 + dq, r0 + dr
        while (q, r) in mine:
            run.append((q, r))
            q, r = q + dq, r + dr
        q, r = q0 - dq, r0 - dr
        while (q, r) in mine:
            run.insert(0, (q, r))
            q, r = q - dq, r - dr
        if len(run) >= WIN_LENGTH:
            return [[q, r] for q, r in run]
    return None
