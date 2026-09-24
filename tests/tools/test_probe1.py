"""PROBE-1's instrument (R365(b)): the decomposer's exclusion and mr split, the gap table's pairing, the KL/calibration reductions."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mantis import _engine
from mantis.diagnostics import ring_reader as R
from mantis.util.loadpkg import load_tools_package

_ENCODING = "gnn_axis_r8"
_K1_SEQ = [(0, 0), (1, 0), (0, 1), (3, 3), (-3, 3), (2, -4)]  # six plies: p2's second stone, mr = 1
_K2_SEQ = [(0, 0), (1, 0), (0, 1), (3, 3), (-3, 3), (2, -4), (-2, 6)]  # seven: p1 to move, mr = 2


@pytest.fixture(scope="module")
def probe1() -> Any:
    load_tools_package("analyzer")
    load_tools_package("probe1")
    return importlib.import_module("probe1.rings"), importlib.import_module("probe1.readings")


def _board(seq: list[tuple[int, int]]) -> Any:
    board = _engine.Board.with_encoding_name(_ENCODING)
    for q, r in seq:
        board.apply_move(q, r)
    return board


def _row(board: Any, visits: list[tuple[int, int, float]], *, full: bool = True, tail: float = 0.0) -> tuple[Any, ...]:
    return (list(board.get_stones()), visits, int(board.current_player), int(board.moves_remaining), 12, full, 1.0, True, 20,
            0, tail)


@pytest.fixture
def ring(tmp_path: Path) -> R.Ring:
    """mr = 1: a one-hot, a soft, a TAIL-ONLY row (α = 1.0 with one zero-mass explicit entry — the ring refuses an empty set); mr = 2: a one-hot, a soft; a quick row."""
    k1, k2 = _board(_K1_SEQ), _board(_K2_SEQ)
    assert k1.moves_remaining == 1 and k2.moves_remaining == 2
    rows = [_row(k1, [(5, 5, 1.0)]), _row(k1, [(5, 5, 0.6), (4, 4, 0.4)]), _row(k1, [(5, 5, 0.0)], tail=1.0),
            _row(k2, [(5, 5, 1.0)]), _row(k2, [(5, 5, 0.5), (4, 4, 0.5)]), _row(k2, [(4, 4, 1.0)], full=False)]
    buf = _engine.HexgBuffer(64, _ENCODING, 16)
    for row in rows:
        buf.push_graph_position(*row)
    path = tmp_path / "planted.ring.bin"
    buf.save_to_path(str(path))
    return R.load_ring(path)


def test_the_decomposer_reads_the_audit_number_then_drops_tail_only_rows_and_splits_by_mr(probe1, ring: R.Ring) -> None:
    rings, _readings = probe1
    d = rings.decompose(ring)
    # The audit counts the tail-only row as a one-hot (H = 0): 3 of 5 full rows; excluded, 2 of 4.
    assert d["full_arm"] == {"n": 5, "one_hot_share": pytest.approx(3 / 5), "tail_only": 1}
    assert d["one_hot_share_full_as_audited"] == pytest.approx(3 / 5)
    assert d["one_hot_share_full_excl_tail_only"] == pytest.approx(2 / 4)
    assert d["by_moves_remaining"]["1"]["as_audited"] == {"n": 3, "one_hot_share": pytest.approx(2 / 3)}
    assert d["by_moves_remaining"]["1"]["excl_tail_only"] == {"n": 2, "one_hot_share": pytest.approx(1 / 2)}
    assert d["by_moves_remaining"]["1"]["tail_only"] == 1
    assert d["by_moves_remaining"]["2"]["excl_tail_only"] == {"n": 2, "one_hot_share": pytest.approx(1 / 2)}


def test_row_selectors_draw_full_arm_rows_and_rebuild_them(probe1, ring: R.Ring) -> None:
    rings, _readings = probe1
    picked = rings.full_arm_rows(ring, seed=1, n=10)
    assert picked.tolist() == [0, 1, 2, 3, 4], "the quick row is never a root"
    assert rings.full_arm_rows(ring, seed=1, n=10, moves_remaining=2).tolist() == [3, 4]
    rebuilt = rings.reconstructed(ring, picked)
    assert [i for i, _b, _m in rebuilt] == [0, 1, 2, 3, 4]
    assert all(b.moves_remaining == ring.moves_remaining[i] for i, b, _m in rebuilt)
    assert rings.target_argmax(ring, 1) == (5, 5) and rings.target_argmax(ring, 2) == (5, 5)
    positions = rings.spread_positions(ring, seed=1, n=4)
    assert {p["row"] for p in positions} == {3, 4} and all(len(p["moves"]) == 7 for p in positions)


def test_the_gap_table_pairs_each_net_with_its_own_and_the_next_ring(probe1) -> None:
    _rings, readings = probe1
    per_ring = [
        {"ring_step": 3000, "losses": {"a": {"step": 3000, "policy": 2.0, "value": 0.6}}},
        {"ring_step": 6000, "losses": {"b": {"step": 6000, "policy": 1.9, "value": 0.55},
                                       "a": {"step": 3000, "policy": 2.3, "value": 0.7}}},
    ]
    table = readings.gap_table(per_ring)
    assert table[0]["net_step"] == 3000 and table[0]["heldout_ring"] == 6000
    assert table[0]["gap_policy"] == pytest.approx(0.3) and table[0]["gap_value"] == pytest.approx(0.1)
    assert table[1]["heldout_ring"] is None and "gap_policy" not in table[1]


def test_calibration_and_kl_reductions_bucket_by_mr(probe1) -> None:
    _rings, readings = probe1
    h = "net"
    rows = {f"{h}:ev": np.array([0.5, -0.5, 0.9, 0.0]), f"{h}:z": np.array([1.0, -1.0, 1.0, 1.0]),
            f"{h}:valid": np.array([True, True, True, False]), f"{h}:mr": np.array([1, 1, 2, 2]),
            f"{h}:value_ce": np.array([0.4, 0.4, 0.1, 9.0]), f"{h}:full": np.array([True, True, True, False]),
            f"{h}:alpha": np.array([0.1, 0.0, 0.2, 0.0]), f"{h}:kl_prior_target": np.array([1.0, 3.0, 0.5, 9.0]),
            f"{h}:kl_target_prior": np.array([0.1, 0.2, 0.3, 9.0]), f"{h}:kl_target_soft": np.array([0.5, 0.5, 0.5, 9.0]),
            f"{h}:policy_ce": np.array([2.0, 2.0, 2.0, 9.0]), f"{h}:unsupported_mass": np.array([0.0, 0.5, 0.0, 0.0])}
    cal = readings.calibration(rows, h, seed=1)
    assert cal["n_valid"] == 3 and cal["by_moves_remaining"]["1"]["mae"] == pytest.approx(0.5)
    assert cal["by_moves_remaining"]["2"]["mae"] == pytest.approx(0.1)
    assert cal["mae_ratio_mr1_over_mr2"] == pytest.approx(5.0)
    kl = readings.kl_summary(rows, h)
    assert kl["n_full_arm"] == 3 and kl["kl_prior_target"]["all"]["median"] == pytest.approx(1.0)
    assert kl["kl_prior_target"]["mr1"]["median"] == pytest.approx(2.0)
    assert kl["unsupported_prior_mass"]["share_rows_gt_1e-3"] == pytest.approx(1 / 3)
    assert kl["kl_prior_target_supported_rows_only"]["n"] == 2 and kl["alpha_zero_share_full_arm"] == pytest.approx(1 / 3)
