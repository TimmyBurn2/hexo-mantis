"""HISTORY_LEN single source of truth.

HISTORY_LEN lives in `mantis.util.constants` as the single SoT; `env.game_state` re-exports it
(no second independent definition). Its old second half — the coupling to the 18-plane source
layout, where HISTORY_LEN was ALSO the opponent block's source-plane offset — is RETIRED with
the dense encode kernels (R346(f)). What the constant bounds now is the `GameState`
move-history deque depth, and this file pins the one property that is left.
"""
from mantis.env.game_state import HISTORY_LEN as GS_HISTORY_LEN
from mantis.util.constants import HISTORY_LEN as CONST_HISTORY_LEN


def test_history_len_single_sot():
    """game_state re-exports the constants SoT, not a second definition."""
    assert GS_HISTORY_LEN == CONST_HISTORY_LEN


def test_history_len_bounds_the_game_state_history_deque():
    """The property that replaced the plane-offset coupling: the deque `GameState.from_board`
    builds is bounded by the SoT, so a depth change reaches the only consumer that is left."""
    from collections import deque

    from mantis._engine import Board
    from mantis.env.game_state import GameState

    board = Board.with_encoding_name("gnn_axis_r8")
    state = GameState.from_board(board)
    assert isinstance(state.move_history, deque)
    assert state.move_history.maxlen == CONST_HISTORY_LEN
