"""Hex coordinate helpers shared across Python callers, on WINDOW-LOCAL axial coordinates.

Kept pure — no numpy, no `mantis` imports — so any script or test can use them; they mirror the
Rust `from_flat`/`to_flat` scatter tables and mantis-core's `hex_distance`. For cluster-centred
global coordinates, compose with the centre offset at the call site.
"""
from __future__ import annotations


def flat_to_axial(flat_idx: int, board_size: int) -> tuple[int, int]:
    """Window-local flat index → axial `(q, r)`, byte-exact with the Rust `from_flat` table.

    Args:
        flat_idx: integer in `[0, board_size * board_size)`.
        board_size: odd window side length (19 for the standard board).

    Returns:
        `(q, r)` with `q, r` in `[-half, half]`.
    """
    half = (board_size - 1) // 2
    q = flat_idx // board_size - half
    r = flat_idx % board_size - half
    return q, r


def axial_to_flat(q: int, r: int, board_size: int) -> int | None:
    """Axial `(q, r)` → window-local flat index, or `None` if outside the `[-half, half]`
    window. Inverse of `flat_to_axial`, matching the Rust `to_flat` contract.
    """
    half = (board_size - 1) // 2
    wq = q + half
    wr = r + half
    if 0 <= wq < board_size and 0 <= wr < board_size:
        return wq * board_size + wr
    return None


def cell_to_flat(cell_str: str, board_size: int) -> int:
    """Parse a ``"q,r"`` cell string (optionally parenthesised) into a window-local flat index.

    Raises `ValueError` rather than returning `None` on an out-of-window cell, because a string
    literal represents caller intent that the cell exists.
    """
    tok = cell_str.strip().strip("()")
    parts = tok.split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'q,r', got {cell_str!r}")
    q = int(parts[0].strip())
    r = int(parts[1].strip())
    flat = axial_to_flat(q, r, board_size)
    if flat is None:
        raise ValueError(
            f"cell ({q}, {r}) outside window of size {board_size}"
        )
    return flat


def axial_distance(a: tuple[float, float], b: tuple[float, float]):
    """Hex Manhattan distance between two axial points, as `max(|dq|, |dr|, |dq + dr|)`.

    Accepts `int` or `float` tuples and returns the input's type, so a float centroid gets the
    exact sub-unit distance without flooring.
    """
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return max(dq, dr, ds)
