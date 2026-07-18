"""GameState oracle — from_board init/after-moves, 2-plane view semantics,
apply_move immutability, history cap, zobrist parity vs Rust, to_tensor shape +
plane semantics, eq/hash.

LOCKED #12 re-anchoring: the hardcoded (1,18,19,19) / (2,19,19) / plane-index /
centre-offset literals are DERIVED from the v6 registry spec + the named
source-plane offsets in `mantis.encoding.resolvers`, never baked. Explicit
deltas: `S` = v6 board_size (19), `NSRC` = v6 n_source_planes (18), `HALF` =
window-centre offset ((S-1)//2), `OPP` = opponent source plane (== HISTORY_LEN),
`MR`/`PLY` = the two scalar source planes, `STONE_PLANES` = 2*HISTORY_LEN.
"""
import numpy as np
import pytest

from mantis._engine import Board
from mantis.encoding import lookup as _lookup_encoding
from mantis.encoding import resolvers as _res
from mantis.env.game_state import GameState, HISTORY_LEN

# ── Registry-derived anchors (LOCKED #12: no baked 19/18/8) ───────────────────
_SPEC = _lookup_encoding("v6")
S: int = _SPEC.board_size                       # window side (19 for v6)
NSRC: int = _SPEC.n_source_planes               # full source-plane count (18)
HALF: int = (S - 1) // 2                         # window-centre offset (9)
CUR: int = _res._CUR_STONE_SRC_PLANE            # current-player t0 plane (0)
OPP: int = _res._OPP_STONE_SRC_PLANE            # opponent t0 plane (8 == HISTORY_LEN)
MR: int = _res._MOVES_REMAINING_SRC_PLANE       # moves-remaining scalar plane (16)
PLY: int = _res._PLY_PARITY_SRC_PLANE           # ply-parity scalar plane (17)
STONE_PLANES: int = 2 * HISTORY_LEN             # stone planes 0..2*H-1 (16)


def test_anchor_coupling_sane() -> None:
    """The named source planes must be internally consistent (OPP == HISTORY_LEN)."""
    assert OPP == HISTORY_LEN
    assert NSRC == STONE_PLANES + 2  # stone planes + the 2 scalar planes


def test_from_board_initial():
    b = Board()
    s = GameState.from_board(b)
    assert s.ply == 0
    assert s.current_player == 1
    assert s.moves_remaining == 1
    assert len(s.views) == 1
    assert s.views[0].shape == (2, S, S)
    assert len(s.centers) == 1
    assert s.centers[0] == (0, 0)


def test_from_board_after_moves():
    b = Board()
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    s = GameState.from_board(b)
    assert s.ply == 2
    assert s.current_player == -1
    assert s.moves_remaining == 1


def test_board_array_p1_stone():
    b = Board()
    b.apply_move(0, 0)   # P1 stone at (0,0)
    # After ply0 it's now P2's turn. P1 is opponent → plane 1 of the 2-plane view.
    s = GameState.from_board(b)
    q_idx, r_idx = HALF, HALF
    assert s.views[0][1, q_idx, r_idx] == 1.0   # P1 stone in opponent plane (plane 1)


def test_board_array_p2_stone():
    b = Board()
    b.apply_move(0, 0)    # P1 ply0
    b.apply_move(-1, 0)   # P2 first stone
    b.apply_move(1, 0)    # P2 second stone
    # Now P1's turn again. P2 is opponent → plane 1 of the 2-plane view.
    s = GameState.from_board(b)
    assert s.views[0][1, -1 + HALF, 0 + HALF] == 1.0  # P2 stone at (-1,0) in opp plane


def test_board_array_empty_cells_are_zero():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.views[0][0, 1 + HALF, 0 + HALF] == 0.0


def test_apply_move_returns_new_state():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    assert s1.ply == 1
    assert s1.current_player == -1
    assert s1.moves_remaining == 2
    # It's now P2's turn. P1's stone at (0,0) is in opponent plane 1 (2-plane view).
    assert s1.views[0][1, HALF, HALF] == 1.0


def test_apply_move_does_not_mutate_history_board():
    b = Board()
    s0 = GameState.from_board(b)
    _ = s0.apply_move(b, 0, 0)
    # The snapshot stored in s0 should still be all zeros (no stones yet).
    assert np.all(s0.views[0] == 0)


def test_move_history_grows():
    b = Board()
    s = GameState.from_board(b)
    assert len(s.move_history) == 0
    s1 = s.apply_move(b, 0, 0)
    assert len(s1.move_history) == 1


def test_move_history_capped_at_history_len():
    b = Board()
    s = GameState.from_board(b)
    for i in range(HISTORY_LEN + 2):
        s = s.apply_move(b, i, 0)
    assert len(s.move_history) == HISTORY_LEN


def test_zobrist_hash_matches_rust():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    assert s.zobrist_hash == b.zobrist_hash()


def test_zobrist_hash_used_for_python_hash():
    b = Board()
    b.apply_move(0, 0)
    s = GameState.from_board(b)
    # zobrist_hash is u128; Python's hash() reduces large ints to Py_hash_t width.
    assert hash(s) == hash(s.zobrist_hash)


def test_to_tensor_shape():
    b = Board()
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    assert t.shape == (1, NSRC, S, S)
    assert t.dtype == np.float16


def test_to_tensor_empty_board_has_zero_stone_planes():
    b = Board()
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    assert np.all(t[0, :STONE_PLANES] == 0.0)


def test_to_tensor_stone_planes_after_moves():
    b = Board()
    b.apply_move(0, 0)   # P1 at (0,0); now P2's turn
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    # current player is -1 (P2); opponent is P1
    # opponent plane (OPP == HISTORY_LEN) should show opponent stone at (0,0)
    assert t[0, OPP, HALF, HALF] == 1.0


def test_to_tensor_moves_remaining_channel():
    b = Board()
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    assert np.all(t[0, MR] == 0.0)


def test_to_tensor_turn_parity_channel():
    b = Board()
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    assert np.all(t[0, PLY] == 0.0)


def test_equal_states_have_equal_hash():
    b1 = Board()
    b1.apply_move(0, 0)
    s1 = GameState.from_board(b1)

    b2 = Board()
    b2.apply_move(0, 0)
    s2 = GameState.from_board(b2)

    assert s1 == s2
    assert hash(s1) == hash(s2)


def test_different_states_are_not_equal():
    b1 = Board()
    b1.apply_move(0, 0)
    s1 = GameState.from_board(b1)

    b2 = Board()
    b2.apply_move(1, 1)
    s2 = GameState.from_board(b2)

    assert s1 != s2
    assert hash(s1) != hash(s2)


def test_eq_rejects_non_gamestate():
    b = Board()
    s = GameState.from_board(b)
    assert (s == object()) is False


# ── History plane tests ───────────────────────────────────────────────────────

def test_history_planes_graceful_missing():
    """At ply 0 (no history), all my/opp history planes must be zeros."""
    b = Board()
    s = GameState.from_board(b)
    t, _ = s.to_tensor()
    assert t.shape == (1, NSRC, S, S)
    # my-stone history planes 1..HISTORY_LEN-1
    assert np.all(t[0, 1:HISTORY_LEN] == 0.0), "my-stone history should be zero at start"
    # opp-stone history planes HISTORY_LEN+1..2*HISTORY_LEN-1
    assert np.all(
        t[0, HISTORY_LEN + 1 : STONE_PLANES] == 0.0
    ), "opp-stone history should be zero at start"


def test_history_planes_are_filled():
    """After 3 moves the t-1 history planes must be non-zero (prior positions present)."""
    b = Board()
    s = GameState.from_board(b)
    # ply 0: P1 at (0,0)
    s1 = s.apply_move(b, 0, 0)
    # ply 1+2: P2 at (1,0) and (2,0) — completes P2's turn, passes back to P1
    s2 = s1.apply_move(b, 1, 0)
    s3 = s2.apply_move(b, 2, 0)

    t, centers = s3.to_tensor()
    assert t.shape[1] == NSRC

    # plane CUR = current player's (P1) stones at t.
    cq, cr = centers[0]
    p1_wq = 0 - cq + HALF
    p1_wr = 0 - cr + HALF
    assert t[0, CUR, p1_wq, p1_wr] == 1.0, "P1 stone at (0,0) should be in current my-stones plane"

    # plane 1 = prior my-stones at t-1 (= s2.views[k][0]); s2 was P2's turn with a stone → non-zero.
    assert t[0, 1].any(), "t-1 my-stones plane should be non-zero (prior position recorded)"


def test_to_tensor_uses_cached_views():
    """to_tensor() must produce (K, NSRC, S, S) with correct stone planes from cached views."""
    b = Board()
    b.apply_move(0, 0)   # P1 at (0,0); now P2's turn
    s = GameState.from_board(b)
    t, _c = s.to_tensor()
    assert t.shape == (1, NSRC, S, S)
    # Current player is P2; opponent (P1) stone at (0,0) is in the opp t0 plane.
    assert t[0, OPP, HALF, HALF] == 1.0, "opponent stone must appear in the opp t0 plane"


# ── Split-responsibility boundary tests ───────────────────────────────────────
# Python's to_tensor() reads the cached 2-plane views (Rust supplies 2 planes,
# Python assembles the full source-plane stack).

def test_plane_cur_my_stones_after_p1_move():
    b = Board()
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)   # P1 ply 0 → P2's turn
    s = s.apply_move(b, 1, 0)   # P2 ply 1
    s = s.apply_move(b, 2, 0)   # P2 ply 2 → P1's turn

    t, centers = s.to_tensor()
    cq, cr = centers[0]
    wq = 0 - cq + HALF
    wr = 0 - cr + HALF
    assert t[0, CUR, wq, wr] == 1.0, "plane CUR (current my-stones) must contain P1's stone at (0,0)"


def test_plane_opp_stones_after_p1_move():
    b = Board()
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)   # P1 ply 0 → P2's turn
    s = s.apply_move(b, 1, 0)   # P2 ply 1
    s = s.apply_move(b, 2, 0)   # P2 ply 2 → P1's turn

    t, centers = s.to_tensor()
    cq, cr = centers[0]
    wq1 = 1 - cq + HALF
    wr1 = 0 - cr + HALF
    wq2 = 2 - cq + HALF
    wr2 = 0 - cr + HALF
    assert t[0, OPP, wq1, wr1] == 1.0, "opp t0 plane must contain P2's first stone at (1,0)"
    assert t[0, OPP, wq2, wr2] == 1.0, "opp t0 plane must contain P2's second stone at (2,0)"
