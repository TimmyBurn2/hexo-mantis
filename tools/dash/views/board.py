"""The board as SVG, drawn server-side with the marks `web/board.js` draws: stones by luminance, one channel per mark."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..readers.hexlogic import owner, turn_of
from .fmt import esc

_SQ3 = math.sqrt(3.0)
Cell = tuple[int, int]
#: The smallest area a board shows, so a few stones are never drawn huge: about 15 cells across, 13 rows down.
MIN_W, MIN_H = _SQ3 * 15, 1.5 * 13


def xy(c: Cell) -> tuple[float, float]:
    """The pointy-top centre of an axial cell."""
    return _SQ3 * (c[0] + c[1] / 2), 1.5 * c[1]


def hex_points(c: Cell, rad: float) -> str:
    """The polygon points of a hexagon of radius `rad` around a cell."""
    cx, cy = xy(c)
    return " ".join(f"{cx + rad * math.cos(math.radians(60 * i - 30)):.3f},{cy + rad * math.sin(math.radians(60 * i - 30)):.3f}"
                    for i in range(6))


@dataclass(frozen=True)
class Scene:
    """What to draw: the moves, the stones placed, the frame that fixes the view, and each optional mark."""

    moves: Sequence[Cell]
    ply: int
    frame: Sequence[Cell] = ()
    numbers: bool = False
    heat: Sequence[tuple[Cell, float, str]] = ()
    win_cells: Sequence[Cell] = ()
    block_cells: Sequence[Cell] = ()
    ghosts: Sequence[tuple[Cell, str]] = ()
    win_line: Sequence[Cell] | None = None


def _hex(cls: str, c: Cell, rad: float, extra: str = "") -> str:
    return f'<polygon class="{cls}" points="{hex_points(c, rad)}"{extra}/>'


def render(scene: Scene, label: str = "Board position") -> str:
    """The board for the position after `ply` stones; the frame is the whole game's extent, so it never jumps."""
    frame = list(scene.frame) or list(scene.moves) or [(0, 0)]
    xs, ys = [xy(c)[0] for c in frame], [xy(c)[1] for c in frame]
    pad = 2.6
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    if x1 - x0 < MIN_W:
        x0, x1 = (x0 + x1 - MIN_W) / 2, (x0 + x1 + MIN_W) / 2
    if y1 - y0 < MIN_H:
        y0, y1 = (y0 + y1 - MIN_H) / 2, (y0 + y1 + MIN_H) / 2
    placed = {tuple(m): i for i, m in enumerate(scene.moves[:scene.ply])}
    out = [f'<svg class="board" viewBox="{x0:.3f} {y0:.3f} {x1 - x0:.3f} {y1 - y0:.3f}" preserveAspectRatio="xMidYMid meet" '
           f'role="img" aria-label="{esc(label)}">']
    for r in range(math.floor(y0 / 1.5) - 1, math.ceil(y1 / 1.5) + 2):
        for q in range(math.floor(x0 / _SQ3 - r / 2) - 1, math.ceil(x1 / _SQ3 - r / 2) + 2):
            x, y = xy((q, r))
            if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1:
                out.append(_hex("cell", (q, r), 0.95, f' data-c="{q},{r}"'))
    for c, rel, _text in scene.heat:
        if c not in placed:
            out.append(_hex("heat", c, 0.26 + 0.56 * math.sqrt(max(0.0, min(1.0, rel)))))
    for c in scene.block_cells:
        out.append(_hex("block", c, 0.8))
    for c in scene.win_cells:
        out.append(_hex("wincell", c, 0.8) + _hex("windot", c, 0.17))
    for c, _label in scene.ghosts:
        if c not in placed:
            out.append(_hex("ghost", c, 0.86))
    for i in range(scene.ply):
        out.append(_hex(f"s{owner(i) + 1}", scene.moves[i], 0.84))
    if scene.win_line:
        out.extend(_hex("winrim", c, 0.84) for c in scene.win_line)
        pts = " ".join(f"{xy(c)[0]:.3f},{xy(c)[1]:.3f}" for c in scene.win_line)
        out.append(f'<polyline class="winline" points="{pts}"/>')
    last_turn = turn_of(scene.ply - 1) if scene.ply else 0
    for i in range(scene.ply):
        c, side = scene.moves[i], owner(i) + 1
        is_last = turn_of(i) == last_turn
        if scene.numbers:
            out.append(f'<text class="n{side}" x="{xy(c)[0]:.3f}" y="{xy(c)[1]:.3f}">{turn_of(i)}</text>')
            if is_last:
                out.append(_hex(f"ltr{side}", c, 0.62))
        elif is_last:
            out.append(_hex(f"lt{side}", c, 0.2))
    for c, label in scene.ghosts:
        if label and c not in placed:
            out.append(f'<text class="ghostnum" x="{xy(c)[0]:.3f}" y="{xy(c)[1]:.3f}">{esc(label)}</text>')
    numbered = {c for c, label in scene.ghosts if label}
    for c, _rel, text in scene.heat:
        if text and c not in placed and c not in numbered:
            out.append(f'<text class="heatnum" x="{xy(c)[0]:.3f}" y="{xy(c)[1]:.3f}">{esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def heat_of(cands: Sequence[Sequence[Any]]) -> list[tuple[Cell, float, str]]:
    """Inner hexagons from candidate rows `[q, r, share, tags]`: size by share of the top cell, the top three labelled in %."""
    if not cands:
        return []
    top = max(float(c[2]) for c in cands) or 1.0
    return [((int(c[0]), int(c[1])), float(c[2]) / top, str(round(float(c[2]) * 100)) if i < 3 and float(c[2]) >= 0.03 else "")
            for i, c in enumerate(cands)]
