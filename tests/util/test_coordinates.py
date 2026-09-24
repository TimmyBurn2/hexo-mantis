"""Unit tests for mantis.util.coordinates.axial_distance, the hex Manhattan distance."""
from __future__ import annotations

import pytest

from mantis.util.coordinates import axial_distance


# axial_distance

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
