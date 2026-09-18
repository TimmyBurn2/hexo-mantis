"""`mantis.diagnostics.ring_audit` (R357(b)): a planted ring, one poisoned forced-block row, exit 1 naming it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mantis.diagnostics import ring_audit as A
from mantis.diagnostics import ring_reader as R

_engine = pytest.importorskip("mantis._engine")

_ENCODING = "gnn_axis_r8"
# P2 holds a capped five on the q axis; P1 (the mover, k = 1) must block (5, 0): the tactics
# oracle reads ('block', {(5, 0)}) and the rest of the stones make no line of their own.
_BLOCK_SEQ = [(-1, 0), (0, 0), (1, 0), (3, 4), (-3, 3), (2, 0), (3, 0), (2, -4), (-2, 6), (4, 0),
              (6, 6), (5, -5)]
_BLOCK, _ELSEWHERE = (5, 0), (0, 1)
_QUIET_SEQ = [(0, 0), (1, 0), (0, 1), (3, 3), (-3, 3), (2, -4), (-2, 6)]


def _board(seq: list[tuple[int, int]]):
    board = _engine.Board.with_encoding_name(_ENCODING)
    for q, r in seq:
        board.apply_move(q, r)
    return board


def _row(board, visits, *, full=True, outcome=1.0, valid=True, length=20, gid=0, tail=0.0):
    return (list(board.get_stones()), visits, int(board.current_player), int(board.moves_remaining),
            12, full, outcome, valid, length, gid, tail)


def _write(path: Path, rows) -> None:
    buf = _engine.HexgBuffer(64, _ENCODING, 16)
    for row in rows:
        buf.push_graph_position(*row)
    buf.save_to_path(str(path))


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """Row 0 is the POISONED forced-block row (one-hot elsewhere); row 1 blocks; rows 2–5 are quiet."""
    block, quiet = _board(_BLOCK_SEQ), _board(_QUIET_SEQ)
    rows = [
        _row(block, [(_ELSEWHERE[0], _ELSEWHERE[1], 0.98), (_BLOCK[0], _BLOCK[1], 0.02)], gid=0),
        _row(block, [(_BLOCK[0], _BLOCK[1], 1.0)], gid=1),
        _row(quiet, [(5, 5, 0.6), (4, 4, 0.4)], gid=2),
        _row(quiet, [(5, 5, 1.0)], gid=3, outcome=-0.1),
        _row(quiet, [(4, 4, 0.5), (5, 5, 0.5)], full=False, gid=4, outcome=-0.1, valid=False),
        _row(quiet, [(4, 4, 1.0)], full=False, gid=4, outcome=-0.1, valid=False),
    ]
    path = tmp_path / "planted.ring.bin"
    _write(path, rows)
    return path


@pytest.fixture
def clean(tmp_path: Path) -> Path:
    block, quiet = _board(_BLOCK_SEQ), _board(_QUIET_SEQ)
    rows = [
        _row(block, [(_BLOCK[0], _BLOCK[1], 0.7), (4, 4, 0.3)], gid=0),
        _row(quiet, [(5, 5, 0.6), (4, 4, 0.4)], gid=1),
        _row(quiet, [(5, 5, 0.55), (4, 4, 0.45)], gid=2),
        _row(quiet, [(4, 4, 0.5), (5, 5, 0.5)], full=False, gid=3),
    ]
    path = tmp_path / "clean.ring.bin"
    _write(path, rows)
    return path


def _bands(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bands.toml"
    path.write_text("[ring_audit.bands]\n" + body, encoding="utf-8")
    return path


def test_the_forced_block_stats_name_the_poisoned_row(planted: Path) -> None:
    stats = A.forced_block_stats(R.load_ring(planted))
    assert stats.n == 2
    assert stats.mass_median == pytest.approx(0.51)
    assert stats.share_ge_half == pytest.approx(0.5)
    assert stats.counter_threat_share == pytest.approx(0.5)
    assert stats.counter_threat_rows == [0]


def test_the_residue_oracle_reads_the_a2_veto_on_the_poisoned_row(planted: Path) -> None:
    """Through the real backup the counter-threat child reads −1 from the root, the block child ≈ 0."""
    ring = R.load_ring(planted)
    board = A.reconstruct(ring, 0)
    assert board is not None
    assert A.child_q(board, _ELSEWHERE) == pytest.approx(-1.0)
    assert A.child_q(board, _BLOCK) == pytest.approx(0.0, abs=1e-6)
    residue = A.quiescence_residue(ring, A.forced_block_stats(ring))
    assert (residue.n_through, residue.count, residue.rows, residue.n_unreached) == (1, 0, [], 0)


def test_the_residue_counts_and_names_a_row_the_oracle_ranks_wrong(planted: Path) -> None:
    ring = R.load_ring(planted)
    residue = A.quiescence_residue(ring, A.forced_block_stats(ring), child_q=lambda _b, _c: 0.0)
    assert (residue.count, residue.rows) == (1, [0])


def test_h_rows_are_the_reader_entropy_by_arm_with_the_one_hot_share(planted: Path) -> None:
    ring = R.load_ring(planted)
    got = {row.key: row for row in A.entropy_rows(ring)}
    h = R.explicit_entropy(ring)
    full, quick = h[ring.is_full_search != 0], h[ring.is_full_search == 0]
    assert got["h_full_median"].value == pytest.approx(float(np.median(full)))
    assert got["h_full_mean"].value == pytest.approx(float(full.mean()))
    assert got["one_hot_share_full"].value == pytest.approx(2 / 4)
    assert got["h_quick_median"].value == pytest.approx(float(np.median(quick)))
    assert got["one_hot_share_quick"].value == pytest.approx(1 / 2)
    assert got["h_pooled_median"].value == pytest.approx(float(np.median(h)))


def test_cap_rate_draw_share_and_mean_abs_z_are_over_distinct_games(planted: Path) -> None:
    got = {row.key: row for row in A.outcome_rows(R.load_ring(planted))}
    assert got["cap_rate"].value == pytest.approx(1 / 5)
    assert got["cap_rate"].n == 5
    assert got["draw_share"].value == pytest.approx(1 / 4)
    assert got["mean_abs_z"].value == pytest.approx((1.0 + 1.0 + 1.0 + 0.1) / 4)


def test_sample_age_and_replay_ratio_are_not_measured_and_say_why(clean: Path) -> None:
    got = {row.key: row for row in A.audit(R.load_ring(clean))}
    assert got["sample_age"].value is None and "no step field" in got["sample_age"].note
    assert got["replay_ratio"].value is None and "iteration_complete" in got["replay_ratio"].note


def test_the_planted_ring_exits_1_naming_the_poisoned_row(planted: Path, tmp_path: Path, capsys) -> None:
    """LAW-07: the producer test — one poisoned forced-block row reds the audit and is named."""
    bands = _bands(tmp_path, "counter_threat_share = { lt = 0.005 }\nquiescence_residue = { le = 0 }\n")
    assert A.main([str(planted), "--bands", str(bands)]) == 1
    out = capsys.readouterr().out
    assert "MISS" in out and "counter_threat_share" in out and "rows [0]" in out


def test_a_clean_ring_passes_the_run8_bands(clean: Path, capsys) -> None:
    prereg = Path("docs/design/measurements/RUN8_PREREG_2026-09-17.md")
    assert A.main([str(clean), "--bands", str(prereg)]) == 0
    assert "ring_audit: PASS" in capsys.readouterr().out


def test_the_prereg_carries_the_five_run8_bands() -> None:
    bands = A.load_bands(Path("docs/design/measurements/RUN8_PREREG_2026-09-17.md"))
    assert bands == {
        "counter_threat_share": ("lt", 0.005), "quiescence_residue": ("le", 0.0),
        "h_full_median": ("gt", 0.02), "one_hot_share_full": ("lt", 0.25), "cap_rate": ("lt", 0.05),
    }


def test_a_band_on_an_unmeasured_row_is_a_miss(clean: Path, tmp_path: Path, capsys) -> None:
    bands = _bands(tmp_path, "sample_age = { lt = 1000 }\n")
    assert A.main([str(clean), "--bands", str(bands)]) == 1
    assert "NOT MEASURED" in capsys.readouterr().out


def test_a_band_naming_no_row_is_refused(clean: Path, tmp_path: Path) -> None:
    bands = _bands(tmp_path, "no_such_row = { lt = 1.0 }\n")
    assert A.main([str(clean), "--bands", str(bands)]) == 2


def test_a_band_with_an_unknown_operator_is_refused(clean: Path, tmp_path: Path) -> None:
    bands = _bands(tmp_path, "cap_rate = { near = 0.05 }\n")
    assert A.main([str(clean), "--bands", str(bands)]) == 2


def test_without_bands_the_table_prints_and_nothing_gates(planted: Path, capsys) -> None:
    assert A.main([str(planted)]) == 0
    out = capsys.readouterr().out
    assert "counter_threat_share" in out and "no bands" in out
