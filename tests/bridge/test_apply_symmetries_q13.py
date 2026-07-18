"""apply_symmetries Q13 parity (O4).

Ports the pure-Python 12-hex-symmetry coordinate oracle from the frozen OLD
`tests/test_chain_plane_augmentation.py` (`_apply_sym_to_coord` / `_transform_stones`
— the INDEPENDENT re-derivation of `sym_tables.rs`, deliberately NOT calling into
the engine) and asserts `_engine.apply_symmetries_batch` scatters state planes
byte-exact against it over the 4 canonical near-origin positions x all 12 syms.

Scope note: the OLD test also gated the `ReplayBuffer.sample_batch(augment=True)`
CHAIN-plane scatter via `_compute_chain_planes`. That leg lands with P2's
`mantis.env.game_state` (which owns `_compute_chain_planes`); here we gate the
load-bearing spatial-scatter primitive `apply_symmetries_batch` directly, which is
self-contained. The chain-scatter leg is tracked to P2 (tests/encoding/), not
silently dropped.
"""
import numpy as np
import pytest

from mantis import _engine

BOARD_SIZE = 19
HALF = (BOARD_SIZE - 1) // 2  # 9


def _flat_to_axial(idx):
    qi, ri = divmod(idx, BOARD_SIZE)
    return qi - HALF, ri - HALF


def _axial_to_flat(q, r):
    qi, ri = q + HALF, r + HALF
    if 0 <= qi < BOARD_SIZE and 0 <= ri < BOARD_SIZE:
        return qi * BOARD_SIZE + ri
    return None


def _apply_sym_to_coord(q, r, sym_idx):
    """Replicate sym_tables.rs: reflection (q,r)->(r,q) first, then n_rot x 60 deg
    via (q,r)->(-r, q+r)."""
    reflect = sym_idx >= 6
    n_rot = sym_idx % 6
    if reflect:
        q, r = r, q
    for _ in range(n_rot):
        q, r = -r, q + r
    return q, r


def _transform_plane(plane, sym_idx):
    """Scatter a (19,19) plane by one hex symmetry; cells mapping out of window
    are dropped (matches the Rust scatter-drop kernel)."""
    out = np.zeros_like(plane)
    flat = plane.reshape(-1)
    for i in range(BOARD_SIZE * BOARD_SIZE):
        if flat[i] == 0:
            continue
        sq, sr = _flat_to_axial(i)
        dq, dr = _apply_sym_to_coord(sq, sr, sym_idx)
        d = _axial_to_flat(dq, dr)
        if d is not None:
            out.reshape(-1)[d] = flat[i]
    return out


def _pos_isolated():
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), np.float32)
    opp = np.zeros_like(cur)
    cur[HALF, HALF] = 1.0
    return cur, opp


def _pos_open_three():
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), np.float32)
    opp = np.zeros_like(cur)
    for q in (-1, 0, 1):
        cur[HALF + q, HALF] = 1.0
    return cur, opp


def _pos_mixed():
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), np.float32)
    opp = np.zeros_like(cur)
    for q in (0, 1, 2):
        cur[HALF + q, HALF] = 1.0
    for r in (1, 2):
        cur[HALF, HALF + r] = 1.0
    opp[HALF - 2, HALF] = 1.0
    opp[HALF, HALF - 2] = 1.0
    return cur, opp


def _pos_asymmetric():
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), np.float32)
    opp = np.zeros_like(cur)
    for r in (0, 1, 2, 3):
        cur[HALF, HALF + r] = 1.0
    return cur, opp


POSITIONS = [
    ("isolated_stone", _pos_isolated),
    ("open_three_axis0", _pos_open_three),
    ("mixed_crosses", _pos_mixed),
    ("asymmetric_axis1", _pos_asymmetric),
]


@pytest.mark.parametrize("pos_name,pos_fn", POSITIONS)
def test_apply_symmetries_batch_byte_exact_vs_oracle(pos_name, pos_fn):
    cur, opp = pos_fn()
    state = np.stack([cur, opp]).astype(np.float32)  # (2, 19, 19)
    batch = np.repeat(state[None], 12, axis=0)  # (12, 2, 19, 19)
    sym_indices = list(range(12))
    out = np.asarray(_engine.apply_symmetries_batch(batch, sym_indices))
    assert out.shape == (12, 2, BOARD_SIZE, BOARD_SIZE)
    for s in range(12):
        exp_cur = _transform_plane(cur, s)
        exp_opp = _transform_plane(opp, s)
        assert np.array_equal(out[s, 0], exp_cur), f"[{pos_name}] sym {s} cur-plane mismatch"
        assert np.array_equal(out[s, 1], exp_opp), f"[{pos_name}] sym {s} opp-plane mismatch"


def test_positions_span_multiple_sym_outputs():
    """Discriminatory power: the non-isolated positions each yield >= 2 distinct
    sym outputs, so a no-op scatter could not pass silently."""
    for name, pos_fn in POSITIONS:
        if name == "isolated_stone":
            continue
        cur, _opp = pos_fn()
        keys = {(_transform_plane(cur, s)).tobytes() for s in range(12)}
        assert len(keys) >= 2, f"[{name}] only {len(keys)} unique sym outputs"


def test_apply_symmetries_batch_rejects_bad_sym_index():
    state = np.zeros((1, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    with pytest.raises(ValueError):
        _engine.apply_symmetries_batch(state, [12])  # out of range 0..12
    with pytest.raises(ValueError):
        _engine.apply_symmetries_batch(state, [0, 1])  # length != batch
