"""GameState oracle — from_board init/after-moves, apply_move immutability, history cap,
zobrist parity vs Rust, eq/hash.

The `to_tensor` half of this file — the (1, 18, S, S) shape, the 2-plane cluster views, the
plane-index and centre-offset rows and the history-plane fill — is RETIRED with the dense
encode kernels (R346(f)). `GameState` now carries position IDENTITY and its bounded move
history and nothing else, so what is pinned here is exactly that surface: the snapshot reads
the engine board, `apply_move` returns a NEW state without moving the old one, the history
deque is bounded by `HISTORY_LEN`, and identity is the engine's zobrist rather than anything
Python re-derives.
"""
from mantis._engine import Board
from mantis.env.game_state import HISTORY_LEN, GameState


def test_from_board_initial():
    b = Board()
    s = GameState.from_board(b)
    assert s.ply == 0
    assert s.current_player == 1
    assert s.moves_remaining == 1
    assert len(s.move_history) == 0


def test_from_board_after_moves():
    b = Board()
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    s = GameState.from_board(b)
    assert s.ply == 2
    assert s.current_player == -1
    assert s.moves_remaining == 1


def test_apply_move_returns_new_state():
    b = Board()
    s0 = GameState.from_board(b)
    s1 = s0.apply_move(b, 0, 0)
    assert s1.ply == 1
    assert s1.current_player == -1
    assert s1.moves_remaining == 2


def test_apply_move_does_not_move_the_prior_snapshot():
    """The prior state is a SNAPSHOT: playing on the shared board must not change it."""
    b = Board()
    s0 = GameState.from_board(b)
    before = (s0.ply, s0.current_player, s0.moves_remaining, s0.zobrist_hash)
    _ = s0.apply_move(b, 0, 0)
    assert (s0.ply, s0.current_player, s0.moves_remaining, s0.zobrist_hash) == before


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
