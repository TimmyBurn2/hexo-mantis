"""HISTORY_LEN single source of truth + opp-plane coupling pin.

HISTORY_LEN lives in `mantis.util.constants` as the single SoT; `env.game_state`
re-exports it (no second independent definition). It is load-bearing-coupled to
the plane layout: the opponent-stone block begins at source plane HISTORY_LEN
(== _OPP_STONE_SRC_PLANE == 8). A future history-depth change editing one
definition but not the other (or not the opp-plane offset) would silently
corrupt the corpus tensor build. These pins guard the coupling.
"""
from mantis.util.constants import HISTORY_LEN as CONST_HISTORY_LEN
from mantis.env.game_state import HISTORY_LEN as GS_HISTORY_LEN
from mantis.encoding.resolvers import _OPP_STONE_SRC_PLANE


def test_history_len_single_sot():
    """game_state re-exports the constants SoT, not a second definition."""
    assert GS_HISTORY_LEN == CONST_HISTORY_LEN


def test_history_len_couples_to_opp_stone_plane():
    """The opponent block begins at source plane HISTORY_LEN; if HISTORY_LEN
    ever moves, the opp-stone source plane MUST move with it (else the
    `tensor[k, HISTORY_LEN]` opp write in game_state corrupts silently)."""
    assert CONST_HISTORY_LEN == _OPP_STONE_SRC_PLANE
