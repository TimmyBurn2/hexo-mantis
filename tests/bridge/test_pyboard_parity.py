"""PyBoard value-parity (O18, review gap 2).

`Board.to_tensor` byte-exact vs the WP3 encode goldens
(`tests/fixtures/encode_parity/`) for the bound single-window v6 to_planes cases;
`get_cluster_views` count/shape/dims; `size` pinned for the bound cases (v6 -> 19,
v6w25 -> 25) AND the unbound `Board()` -> BOARD_SIZE(19); the radius/cluster guards
raise ValueError when an encoding is bound.

The goldens are the WP3 MEPB blobs (magic `MEPB`, version 1) — a to_planes case
carries its move sequence plus the golden 18x361 f32 output; we rebuild the board
from the moves and compare `to_tensor()` byte-for-byte, the same construction path
`common/mod.rs::board_with_moves` uses.
"""
import pathlib
import struct

import numpy as np
import pytest

from mantis import _engine

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "encode_parity"
# manifest.tsv header `encodings = v6,v6w25,v6_live2_ls` -> enc_name(id) index order.
ENC_NAMES = ["v6", "v6w25", "v6_live2_ls"]
K_TO_PLANES = 4
TO_PLANES_CASES = [21, 22]  # the two to_planes C-rows (both v6, 18x361)


class _Cur:
    def __init__(self, b):
        self.b = b
        self.o = 0

    def take(self, n):
        s = self.b[self.o : self.o + n]
        assert len(s) == n, "truncated blob"
        self.o += n
        return s

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]


def _read_to_planes_blob(path):
    """Return (encoding_name, moves, golden_bytes) for a to_planes MEPB blob."""
    c = _Cur(pathlib.Path(path).read_bytes())
    assert c.take(4) == b"MEPB", "bad magic"
    assert c.u32() == 1, "bad version"
    header_case_id = c.u32()
    case_id = c.u32()
    kernel = c.u8()
    enc_id = c.u8()
    _class = c.u8()
    assert kernel == K_TO_PLANES, f"case {case_id} kernel {kernel} != to_planes"
    assert case_id == header_case_id
    n_moves = c.u32()
    moves = [(c.i32(), c.i32()) for _ in range(n_moves)]
    n_fields = c.u8()
    assert n_fields != 0, "to_planes case must carry a golden payload"
    name_len = c.u8()
    c.take(name_len)  # field name
    c.u8()  # dtype
    ndim = c.u8()
    for _ in range(ndim):
        c.u32()  # dims
    nbytes = c.u64()
    payload = c.take(nbytes)
    return ENC_NAMES[enc_id], moves, payload


@pytest.mark.parametrize("case_id", TO_PLANES_CASES)
def test_to_tensor_byte_exact_vs_golden(case_id):
    enc, moves, golden_bytes = _read_to_planes_blob(FIXTURES / "raw" / f"case_{case_id:05d}.bin")
    golden = np.frombuffer(golden_bytes, dtype="<f4")
    board = _engine.Board.with_encoding_name(enc)  # BOUND path
    for q, r in moves:
        board.apply_move(q, r)
    tensor = np.asarray(board.to_tensor())
    assert tensor.dtype == np.float32
    assert tensor.size == golden.size
    assert np.array_equal(tensor, golden), f"to_tensor byte mismatch for case {case_id} ({enc})"


def test_size_pinned_bound_and_unbound():
    assert _engine.Board.with_encoding_name("v6").size == 19
    assert _engine.Board.with_encoding_name("v6w25").size == 25
    assert _engine.Board().size == 19  # unbound -> BOARD_SIZE, a raw geometry default


def test_get_cluster_views_count_shape_dims():
    board = _engine.Board.with_encoding_name("v6w25")  # S = 25
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    views, centers = board.get_cluster_views()
    assert len(views) == len(centers)
    assert len(views) >= 1
    for v in views:
        arr = np.asarray(v)
        assert arr.shape == (2, 25, 25)  # (plane, S, S), not transposed
        assert arr.dtype == np.float32
    for qr in centers:
        assert len(qr) == 2


def test_radius_and_cluster_guards_raise_when_bound():
    """The radius/cluster setters raise ValueError on an encoding-bound board
    (identity is owned by the registry entry, not a caller override)."""
    board = _engine.Board.with_encoding_name("v6")
    with pytest.raises(ValueError):
        board.set_legal_move_radius(5)
    with pytest.raises(ValueError):
        board.set_cluster_threshold(8)
    with pytest.raises(ValueError):
        board.set_cluster_window_size(25)
