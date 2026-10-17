"""The pure-numpy HEXG reader agrees with the engine that wrote the ring, field by field, on every layout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mantis.diagnostics import ring_reader as R

_engine = pytest.importorskip("mantis._engine")

_ENCODING = "gnn_axis_v1"
# stones (q, r, p); visits (q, r, prob) summing to 1 - tail_mass; player, remaining, ply,
# full search, outcome, value valid, game length, game id, tail_mass
_ROWS = [
    ([(0, 0, 1)], [(1, 0, 0.75), (0, 1, 0.25)], -1, 2, 1, True, 1.0, True, 20, 7, 0.0),
    ([(0, 0, 1), (1, 0, -1), (2, 0, -1)], [(3, 0, 0.5), (-1, 0, 0.3)], 1, 1, 3, False, -1.0,
     True, 20, 7, 0.2),
    ([(0, 0, 1), (1, 0, -1), (2, 0, -1), (0, 1, 1)], [(5, 5, 1.0)], -1, 2, 4, True, 0.0, False,
     21, 8, 0.0),
]
# (root_value, root_value_valid) per row of `_ROWS`.
_ROOTS = [(0.5, True), (0.0, False), (-1.0, True)]
# The future-occupancy field per row of `_ROWS`: a full horizon, none, a full horizon.
_FUTURES = [[(i - 10, 2, -1 if i % 4 < 2 else 1) for i in range(20)], None,
            [(i - 10, -3, -1 if i % 4 < 2 else 1) for i in range(20)]]
# The proven-tail flag per row of `_ROWS`: only the quick row, as a full-search row cannot carry it.
_PROVEN = [False, True, False]
_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "replay"


def _write_ring(path: Path) -> _engine.HexgBuffer:
    buf = _engine.HexgBuffer(16, _ENCODING, 64)
    for (stones, visits, player, remaining, ply, full, outcome, valid, length, gid, tail), (root, root_ok), future, \
            proven in zip(_ROWS, _ROOTS, _FUTURES, _PROVEN, strict=True):
        buf.push_graph_position(stones, visits, player, remaining, ply, full, outcome, valid,
                                length, gid, tail, root_value=root, root_value_valid=root_ok,
                                future_stones=future, proven_tail=proven)
    buf.save_to_path(str(path))
    return buf


def test_header_and_count_match_the_writer(tmp_path: Path) -> None:
    path = tmp_path / "ring.hexg"
    buf = _write_ring(path)
    ring = R.load_ring(path)
    assert ring.header.size == buf.size == len(_ROWS)
    assert ring.header.encoding == _ENCODING
    assert ring.header.max_visits == buf.visit_capacity
    assert ring.header.capacity == buf.capacity


def test_every_field_reads_back_as_pushed(tmp_path: Path) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    ring = R.load_ring(path)
    for i, (stones, visits, player, remaining, ply, full, outcome, valid, length, gid, tail) \
            in enumerate(_ROWS):
        assert ring.current_player[i] == player
        assert ring.moves_remaining[i] == remaining
        assert ring.ply_index[i] == ply
        assert bool(ring.is_full_search[i]) is full
        assert bool(ring.value_valid[i]) is valid
        assert ring.outcome[i] == pytest.approx(outcome)
        assert ring.game_length[i] == length
        assert ring.game_id[i] == gid
        assert ring.tail_mass[i] == pytest.approx(tail)
        assert ring.root_value[i] == _ROOTS[i][0]
        assert bool(ring.root_value_valid[i]) is _ROOTS[i][1]
        got_stones = ring.row_stones(i)
        assert [(int(s["q"]), int(s["r"]), int(s["p"])) for s in got_stones] == stones
        got_visits = ring.row_visits(i)
        assert [(int(v["q"]), int(v["r"])) for v in got_visits] == [(q, r) for q, r, _ in visits]
        assert np.allclose([float(v["prob"]) for v in got_visits], [p for _, _, p in visits])
        assert bool(ring.future_valid[i]) is (_FUTURES[i] is not None)
        assert [(int(f["q"]), int(f["r"]), int(f["p"])) for f in ring.row_future(i)] == (_FUTURES[i] or [])
        assert bool(ring.proven_tail[i]) is _PROVEN[i]


def test_the_engine_loads_the_same_count_the_reader_parsed(tmp_path: Path) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    ring = R.load_ring(path)
    other = _engine.HexgBuffer(16, _ENCODING, 64)
    assert other.load_from_path(str(path)) == ring.header.size


def test_a_wrong_magic_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="magic"):
        R.load_ring(path)


def test_a_truncated_payload_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    data = path.read_bytes()
    path.write_bytes(data[:-3])
    with pytest.raises(ValueError):
        R.load_ring(path)


def test_the_cli_prints_the_header_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    assert R.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert f"encoding={_ENCODING}" in out and f"size={len(_ROWS)}" in out
    assert f"proven_tail rows: 1 of {len(_ROWS)}" in out


def test_explicit_entropy_is_the_census_h_over_the_renormalised_explicit_masses(tmp_path: Path) -> None:
    """(0.75, 0.25) → H; (0.5, 0.3) under α 0.2 renormalises to (0.625, 0.375); a one-hot reads 0."""
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    h = R.explicit_entropy(R.load_ring(path))
    expected = [
        -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)),
        -(0.625 * np.log(0.625) + 0.375 * np.log(0.375)),
        0.0,
    ]
    assert h.shape == (len(_ROWS),)
    assert np.allclose(h, expected, atol=1e-6)


def test_explicit_entropy_reads_zero_on_a_row_with_no_explicit_mass(tmp_path: Path) -> None:
    """An α = 1 row has no mass to renormalise: 0.0, not NaN (planted: the engine refuses to write one)."""
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    ring = R.load_ring(path)
    probs = ring.visits["prob"].copy()
    probs[ring.visit_off[1]: ring.visit_off[1] + ring.n_visits[1]] = 0.0
    ring.visits["prob"] = probs
    ring.tail_mass[1] = 1.0
    h = R.explicit_entropy(ring)
    assert h[1] == 0.0 and np.isfinite(h).all()
    assert h[0] == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))


def test_the_cli_prints_the_entropy_line_by_arm(tmp_path: Path, capsys) -> None:
    path = tmp_path / "ring.hexg"
    _write_ring(path)
    assert R.main([str(path)]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("H(explicit) nats:"))
    assert "full n=2 median" in line and "quick n=1 median" in line


def test_the_v2_golden_reads_with_no_root_value_on_any_row() -> None:
    """A v2 ring still reads: its fields as written, every row default-filled to no root value."""
    ring = R.load_ring(_REPLAY / "hexg_v2_golden.hexg")
    assert ring.header.size == 2
    assert ring.outcome.tolist() == [1.0, -1.0]
    assert ring.tail_mass.tolist() == [0.0, 0.25]
    assert ring.game_id.tolist() == [42, 43]
    assert ring.root_value_valid.tolist() == [0, 0]
    assert ring.root_value.view(np.uint32).tolist() == [0, 0]
    assert ring.future_valid.tolist() == [0, 0] and ring.future.size == 0
    assert ring.proven_tail.tolist() == [0, 0]


def test_the_v3_golden_reads_its_root_values() -> None:
    ring = R.load_ring(_REPLAY / "hexg_v3_golden.hexg")
    assert ring.root_value_valid.tolist() == [1, 0]
    assert ring.root_value.tolist() == [-0.625, 0.0]
    assert ring.tail_mass.tolist() == [0.0, 0.25]
    assert ring.game_id.tolist() == [42, 43]
    assert ring.future_valid.tolist() == [0, 0] and ring.future.size == 0
    assert ring.proven_tail.tolist() == [0, 0]


def test_the_v4_golden_reads_its_future_stones() -> None:
    """Record 0 carries its game's eleven remaining stones in placement order; record 1 carries none."""
    ring = R.load_ring(_REPLAY / "hexg_v4_golden.hexg")
    assert ring.root_value.tolist() == [-0.625, 0.0]
    assert ring.future_valid.tolist() == [1, 0]
    assert ring.n_future.tolist() == [11, 0]
    assert [(int(f["q"]), int(f["r"]), int(f["p"])) for f in ring.row_future(0)][:3] == [(2, 0, 1), (0, 2, 1),
                                                                                         (-1, 0, -1)]
    assert ring.proven_tail.tolist() == [0, 0]


def test_the_v5_golden_reads_its_proven_tail_flags() -> None:
    """Record 1, the quick row, is a decided tail's; record 0 keeps its future stones and no flag."""
    ring = R.load_ring(_REPLAY / "hexg_v5_golden.hexg")
    assert ring.header.version == 5
    assert ring.proven_tail.tolist() == [0, 1]
    assert ring.is_full_search.tolist() == [1, 0]
    assert ring.future_valid.tolist() == [1, 0] and ring.n_future.tolist() == [11, 0]
    assert ring.root_value.tolist() == [-0.625, 0.0]
    assert ring.game_id.tolist() == [42, 43]


@pytest.mark.parametrize(("fixture", "stamp"), [("hexg_v3_golden.hexg", 2), ("hexg_v2_golden.hexg", 3),
                                                ("hexg_v4_golden.hexg", 3), ("hexg_v3_golden.hexg", 4),
                                                ("hexg_v5_golden.hexg", 4), ("hexg_v4_golden.hexg", 5)])
def test_a_payload_read_through_the_other_versions_layout_is_refused(tmp_path: Path, fixture: str,
                                                                      stamp: int) -> None:
    """PLANTED BREAK: one layout for every version takes root-value, future or flag bytes for stones; the reader must refuse."""
    data = bytearray((_REPLAY / fixture).read_bytes())
    data[4:8] = stamp.to_bytes(4, "little")
    path = tmp_path / "relabelled.hexg"
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        R.load_ring(path)


def test_a_version_this_reader_has_no_layout_for_is_refused() -> None:
    with pytest.raises(ValueError, match="version 1"):
        R.load_ring(_REPLAY / "hexg_v1_golden.hexg")


@pytest.mark.parametrize("cut", [40, 60])
def test_a_payload_cut_inside_a_header_or_a_fixed_head_is_a_value_error(tmp_path: Path, cut: int) -> None:
    path = tmp_path / "cut.hexg"
    path.write_bytes((_REPLAY / "hexg_v3_golden.hexg").read_bytes()[:cut])
    with pytest.raises(ValueError):
        R.load_ring(path)
