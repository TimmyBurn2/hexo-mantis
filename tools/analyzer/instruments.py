"""The 12 hex symmetries about the first stone (+ one translation) and the sweep; the value trace in P1's view."""
from __future__ import annotations

from typing import Any

from .engines import raw_argmax
from .position import SIDE_OF_INT, PositionRefused, build_board

#: (name, rotations by 60°, reflect first); the identity is row 0.
MAPS: list[tuple[str, int, bool]] = [
    (("identity" if k == 0 and not refl else f"rot{60 * k}" + ("·refl" if refl else "")), k, refl)
    for k in range(6) for refl in (False, True)
]
TRANSLATION = (1, -1)


def _rot(q: int, r: int, times: int) -> tuple[int, int]:
    for _ in range(times % 6):
        q, r = -r, q + r
    return q, r


def transform(cell: tuple[int, int], centre: tuple[int, int], k: int, reflect: bool, *,
              inverse: bool = False) -> tuple[int, int]:
    """`R^k(S(x - c)) + c` (reflect first, then rotate) or its inverse `S(R^-k(x - c)) + c`."""
    q, r = cell[0] - centre[0], cell[1] - centre[1]
    if inverse:
        q, r = _rot(q, r, 6 - k)
        if reflect:
            q, r = r, q
    else:
        if reflect:
            q, r = r, q
        q, r = _rot(q, r, k)
    return q + centre[0], r + centre[1]


def _read(engine: Any, moves: list[tuple[int, int]]) -> tuple[float, tuple[int, int]]:
    raw = engine.raw_read(build_board(moves, engine.encoding).board)
    return float(raw.value), raw_argmax(raw.children)


def sweep(engine: Any, moves: list[tuple[int, int]]) -> dict[str, Any]:
    """Raw reads over the 12 maps about the first stone (+ the translation when it stays in the opening window)."""
    if not moves:
        return {"absent": "an empty board has no first stone to centre on"}
    centre = moves[0]
    rows: list[dict[str, Any]] = []
    for name, k, refl in MAPS:
        value, best = _read(engine, [transform(m, centre, k, refl) for m in moves])
        back = transform(best, centre, k, refl, inverse=True)
        rows.append({"map": name, "value": round(value, 4), "argmax": [back[0], back[1]]})
    shifted = [(q + TRANSLATION[0], r + TRANSLATION[1]) for q, r in moves]
    try:
        value, best = _read(engine, shifted)
        translation: dict[str, Any] = {"map": f"translate{TRANSLATION}", "value": round(value, 4),
                                       "argmax": [best[0] - TRANSLATION[0], best[1] - TRANSLATION[1]]}
    except PositionRefused as exc:
        translation = {"absent": f"the first stone would leave the opening window ({exc})"}
    values = [row["value"] for row in rows]
    identity = rows[0]
    worst = max(rows[1:], key=lambda row: abs(row["value"] - identity["value"]))
    agree = sum(row["argmax"] == identity["argmax"] for row in rows)
    return {"n": len(rows), "centre": [centre[0], centre[1]], "value_min": min(values), "value_max": max(values),
            "spread": round(max(values) - min(values), 4), "argmax_agreement": f"{agree}/{len(rows)}",
            "worst": worst, "translation": translation, "rows": rows}


def p1_view(value: float, to_move: str) -> float:
    """A mover's value in P1's fixed perspective."""
    return value if to_move == "p1" else -value


def trace(engine: Any, moves: list[tuple[int, int]], sims: int) -> list[dict[str, Any]]:
    """One row per ply 0..n: the net's value (and the head's at `sims` ≥ 1) in P1's view; terminal plies are gaps."""
    rows: list[dict[str, Any]] = []
    for ply in range(len(moves) + 1):
        pos = build_board(moves[:ply], engine.encoding)
        if pos.winner is not None:
            rows.append({"ply": ply, "to_move": None, "raw": None, "root": None, "terminal": pos.winner})
            continue
        to_move = SIDE_OF_INT[int(pos.board.current_player)]
        raw = engine.raw_read(pos.board)
        row: dict[str, Any] = {"ply": ply, "to_move": to_move, "raw": round(p1_view(raw.value, to_move), 4),
                               "root": None}
        if sims >= 1:
            s = engine.search(pos.board, sims)
            row["root"] = round(p1_view(s.root_value, to_move), 4)
            row["ms"] = round(s.ms, 1)
        rows.append(row)
    return rows


__all__ = ["MAPS", "TRANSLATION", "p1_view", "sweep", "trace", "transform"]
