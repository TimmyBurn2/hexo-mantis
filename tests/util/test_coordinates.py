"""Unit tests for mantis.util.coordinates — pure hex coord helpers.

LOCKED #12 re-anchoring: `BOARD` (the window side / board_size) is derived from
the v6 registry spec rather than baked as 19, and every hardcoded centre-offset
(9) / out-of-window index (10) is expressed via `HALF = (BOARD-1)//2`. The
known-triple and known-distance tables are pure axial relationships stated in
terms of HALF/BOARD, so they re-anchor automatically.
"""
from __future__ import annotations

import pytest

from mantis.encoding import lookup as _lookup_encoding
from mantis.util.coordinates import (
    flat_to_axial,
    axial_to_flat,
    cell_to_flat,
    axial_distance,
)

BOARD: int = _lookup_encoding("v6").board_size   # 19 for v6 — registry-derived
HALF: int = (BOARD - 1) // 2                      # 9


# ── Round trip ───────────────────────────────────────────────────────────────

def test_flat_to_axial_round_trip_all_cells():
    for flat in range(BOARD * BOARD):
        q, r = flat_to_axial(flat, BOARD)
        assert -HALF <= q <= HALF, f"flat {flat}: q={q}"
        assert -HALF <= r <= HALF, f"flat {flat}: r={r}"
        back = axial_to_flat(q, r, BOARD)
        assert back == flat, f"flat {flat} round trip to {back} via ({q},{r})"


def test_axial_to_flat_out_of_window_returns_none():
    assert axial_to_flat(HALF + 1, 0, BOARD) is None
    assert axial_to_flat(-(HALF + 1), 0, BOARD) is None
    assert axial_to_flat(0, HALF + 1, BOARD) is None
    assert axial_to_flat(0, -(HALF + 1), BOARD) is None
    assert axial_to_flat(HALF, HALF, BOARD) is not None
    assert axial_to_flat(-HALF, -HALF, BOARD) is not None


# ── Known values ─────────────────────────────────────────────────────────────

KNOWN_TRIPLES = [
    (0,                       -HALF, -HALF),
    (HALF,                    -HALF,  0),
    (HALF * 2,                -HALF,  HALF),
    (HALF * BOARD,             0,    -HALF),
    (HALF * BOARD + HALF,      0,     0),
    (HALF * BOARD + HALF * 2,  0,     HALF),
    (BOARD * BOARD - 1,        HALF,  HALF),
    ((HALF + 1) * BOARD + HALF, 1,    0),
    (HALF * BOARD + HALF + 1,  0,     1),
    ((HALF - 1) * BOARD + HALF - 1, -1, -1),
]


@pytest.mark.parametrize("flat,q,r", KNOWN_TRIPLES)
def test_flat_to_axial_known_values(flat: int, q: int, r: int):
    assert flat_to_axial(flat, BOARD) == (q, r)
    assert axial_to_flat(q, r, BOARD) == flat


# ── cell_to_flat ─────────────────────────────────────────────────────────────

def test_cell_to_flat_origin():
    assert cell_to_flat("0,0", BOARD) == HALF * BOARD + HALF


def test_cell_to_flat_corners():
    assert cell_to_flat(f"{-HALF},{-HALF}", BOARD) == 0
    assert cell_to_flat(f"{HALF},{HALF}", BOARD) == BOARD * BOARD - 1
    assert cell_to_flat(f"{-HALF},{HALF}", BOARD) == BOARD - 1                 # row 0, last col
    assert cell_to_flat(f"{HALF},{-HALF}", BOARD) == (BOARD - 1) * BOARD       # last row, col 0


def test_cell_to_flat_whitespace_and_parens():
    ref = cell_to_flat("3,-4", BOARD)
    assert cell_to_flat(" 3, -4 ", BOARD) == ref
    assert cell_to_flat("(3,-4)", BOARD) == ref
    assert cell_to_flat("(3, -4)", BOARD) == ref


def test_cell_to_flat_invalid_format_raises():
    with pytest.raises(ValueError):
        cell_to_flat("bad", BOARD)
    with pytest.raises(ValueError):
        cell_to_flat("1,2,3", BOARD)


def test_cell_to_flat_out_of_window_raises():
    with pytest.raises(ValueError):
        cell_to_flat(f"{HALF + 1},0", BOARD)
    with pytest.raises(ValueError):
        cell_to_flat(f"0,{-(HALF + 1)}", BOARD)


# ── axial_distance ──────────────────────────────────────────────────────────

KNOWN_DISTANCES = [
    ((0, 0), (0, 0), 0),
    ((0, 0), (1, 0), 1),
    ((0, 0), (0, 1), 1),
    ((0, 0), (1, -1), 1),
    ((0, 0), (3, 0), 3),
    ((0, 0), (3, -3), 3),          # axis 2
    ((0, 0), (2, 3), 5),           # dq=2, dr=3, ds=5 → 5
    ((-4, 2), (4, -2), 8),         # dq=8, dr=4, ds=8 → 8
    ((-3, -3), (3, 3), 12),        # (1,1) is NOT a hex unit direction — need 12 steps
    ((1, 0), (-1, 0), 2),
]


@pytest.mark.parametrize("a,b,expected", KNOWN_DISTANCES)
def test_axial_distance_known_values(a, b, expected):
    assert axial_distance(a, b) == expected
    assert axial_distance(b, a) == expected  # symmetric


def test_axial_distance_accepts_float_centroids():
    assert axial_distance((0.5, 0.5), (3.5, 0.5)) == 3
    assert axial_distance((-2.0, 1.0), (2.0, -1.0)) == 4


def test_axial_distance_float_inputs_return_float():
    """Float inputs must produce a float result, not an int (no int() cast)."""
    result = axial_distance((0.0, 0.0), (1.5, 0.0))
    assert isinstance(result, float), f"expected float, got {type(result)}"

    result2 = axial_distance((0.5, 0.5), (3.5, 0.5))
    assert isinstance(result2, float), f"expected float, got {type(result2)}"

    result3 = axial_distance((-2.0, 1.0), (2.0, -1.0))
    assert isinstance(result3, float), f"expected float, got {type(result3)}"


def test_axial_distance_float_non_integer_values():
    """Non-integer float distances are returned exactly, not floored."""
    d = axial_distance((0.0, 0.0), (3.0, 2.9))
    assert d == pytest.approx(5.9), f"expected 5.9, got {d}"
    assert d != 5, "distance must not be floored to 5 (old int() cast behaviour)"

    d2 = axial_distance((1.0, 0.0), (1.0, 4.5))
    assert d2 == pytest.approx(4.5), f"expected 4.5, got {d2}"


def test_axial_distance_float_threshold_boundary():
    """Two pairs that differ below and above a threshold of 6.0."""
    d_just_below = axial_distance((0.0, 0.0), (3.0, 2.999))
    assert d_just_below == pytest.approx(5.999, abs=1e-9)
    assert d_just_below < 6.0, "5.999 must be below threshold 6.0"
    assert d_just_below != 5, "5.999 must NOT be floored to 5 (old bug)"

    d_just_above = axial_distance((0.0, 0.0), (3.0, 3.001))
    assert d_just_above == pytest.approx(6.001, abs=1e-9)
    assert d_just_above > 6.0, "6.001 must be above threshold 6.0"
