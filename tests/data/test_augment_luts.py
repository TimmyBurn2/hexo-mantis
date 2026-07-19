"""O4c — augment policy-scatter LUT parity vs the committed old-side npz.

``get_policy_scatters(board_size, has_pass=True)`` regenerated in-test must byte-match the
committed ``value_probes/augment/scatters_bs{19,25}.npz`` — the 12 int64 scatter arrays
(keys ``sym_00``..``sym_11``), sym 0 == identity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mantis.data.augment import get_policy_scatters

_AUG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "augment"


@pytest.mark.parametrize(("board_size", "length"), [(19, 362), (25, 626)])
def test_policy_scatters_byte_match(board_size: int, length: int) -> None:
    ref = np.load(_AUG_DIR / f"scatters_bs{board_size}.npz")
    scatters = get_policy_scatters(board_size, has_pass=True)
    assert len(scatters) == 12
    for sym in range(12):
        arr = scatters[sym]
        expected = ref[f"sym_{sym:02d}"]
        assert arr.dtype == np.int64
        assert arr.shape == (length,)
        assert np.array_equal(arr, expected), f"bs{board_size} sym_{sym:02d} byte mismatch"


def test_sym0_is_identity() -> None:
    for board_size, length in [(19, 362), (25, 626)]:
        sym0 = get_policy_scatters(board_size, has_pass=True)[0]
        assert np.array_equal(sym0, np.arange(length, dtype=np.int64))


def test_pass_slot_invariant_and_no_pass_slot_variant() -> None:
    # has_pass=True: the pass index (n_cells) maps to itself under every symmetry.
    n_cells = 19 * 19
    for sym in get_policy_scatters(19, has_pass=True):
        assert sym[n_cells] == n_cells
        assert sym.shape == (n_cells + 1,)
    # has_pass=False: the pass row is omitted entirely (generic knob; v8 name never passed).
    for sym in get_policy_scatters(19, has_pass=False):
        assert sym.shape == (n_cells,)
