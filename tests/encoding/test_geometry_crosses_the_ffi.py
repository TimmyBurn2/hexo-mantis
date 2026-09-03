"""AUDIT-1 F-42 — the geometry constants are ONE set on both sides of the FFI.

THE DEFECT. The v6 source-plane indices, the three hex axes and the win length were each
typed on BOTH sides with nothing pinning across: `resolvers.py` carried
`_CUR_STONE_SRC_PLANE = 0 … _PLY_PARITY_SRC_PLANE = 17` beside
`mantis_encoding::encode::{MY_STONE_PLANE … PLY_PARITY_PLANE}`; `env/game_state.py`,
`selfplay/graph_collate.py` and `selfplay/instrumentation.py` each typed the axis table
beside `mantis_core::board::HEX_AXES` / `mantis_graph::WIN_AXES`; `util/constants.HISTORY_LEN`
pinned Python to Python. The audit's phrase for it: Python pins Python, Rust pins a literal,
nothing pins across the FFI.

WHAT IS PINNED HERE, and why each row is not the others:

- the four plane indices agree with the engine's — the Python names are now derived from the
  engine, so this row is the guard against a future re-typing rather than a live disagreement;
- `OPP_STONE_PLANE == HISTORY_LEN`, the RELATION the wire format is built on (the opponent's
  t0 plane sits immediately after the current player's history block);
- `v6_live2_ls.kept_plane_indices` is the SAME four indices, read off the shipped registry —
  the audit's `registry_census.rs::mechanical_delta_pins` pins those consts to literals and
  never to this row, so the registry could move underneath them;
- the three Python axis modules and the engine are one ordered table.
"""
from __future__ import annotations

import mantis._engine as engine
from mantis.encoding import lookup
from mantis.encoding.resolvers import (
    _CUR_STONE_SRC_PLANE,
    _MOVES_REMAINING_SRC_PLANE,
    _OPP_STONE_SRC_PLANE,
    _PLY_PARITY_SRC_PLANE,
)
from mantis.env.game_state import _CHAIN_CAP, _HEX_AXES
from mantis.selfplay.graph_collate import WIN_AXES
from mantis.selfplay.instrumentation import _HEX_AXES as _INSTR_HEX_AXES
from mantis.selfplay.instrumentation import _WIN_LENGTH as _INSTR_WIN_LENGTH
from mantis.util.constants import HISTORY_LEN


def test_the_four_source_plane_indices_are_the_engines() -> None:
    assert (
        _CUR_STONE_SRC_PLANE,
        _OPP_STONE_SRC_PLANE,
        _MOVES_REMAINING_SRC_PLANE,
        _PLY_PARITY_SRC_PLANE,
    ) == (
        engine.MY_STONE_PLANE,
        engine.OPP_STONE_PLANE,
        engine.MOVES_REMAINING_PLANE,
        engine.PLY_PARITY_PLANE,
    ), "the wire format's plane indices are the encode kernels', not a Python transcription"


def test_the_opponent_plane_sits_immediately_after_the_history_block() -> None:
    """The RELATION, not two equal numbers: `HISTORY_LEN` slots of current-player history
    are followed by the opponent's t0 plane. Move one and the other must move."""
    assert engine.OPP_STONE_PLANE == HISTORY_LEN == _OPP_STONE_SRC_PLANE


def test_the_shipped_registry_row_carries_those_same_four_indices() -> None:
    """`v6_live2_ls` keeps exactly the four planes the constants name, in that order. This is
    the cross-check the audit found missing: the Rust census pins the consts to LITERALS and
    never to this registry row, so the row could be re-minted underneath them."""
    kept = tuple(lookup("v6_live2_ls").kept_plane_indices)
    assert kept == (
        engine.MY_STONE_PLANE,
        engine.OPP_STONE_PLANE,
        engine.MOVES_REMAINING_PLANE,
        engine.PLY_PARITY_PLANE,
    ), f"v6_live2_ls.kept_plane_indices {kept} no longer names the four plane constants"


def test_every_python_axis_table_is_the_engines() -> None:
    engine_axes = tuple((int(dq), int(dr)) for dq, dr in engine.HEX_AXES)
    assert _HEX_AXES == engine_axes, "env.game_state (chain-plane order)"
    assert WIN_AXES == engine_axes, "selfplay.graph_collate (edge one-hot order)"
    assert tuple(_INSTR_HEX_AXES) == engine_axes, "selfplay.instrumentation (line scan)"


def test_every_python_win_length_is_the_engines() -> None:
    assert _CHAIN_CAP == engine.WIN_LENGTH, "env.game_state chain saturation cap"
    assert _INSTR_WIN_LENGTH == engine.WIN_LENGTH, "instrumentation longest_line cap"


def test_the_axis_order_itself_is_pinned() -> None:
    """Agreement alone stays green if every copy moves together — and every frozen fixture
    and every trained net was built against THIS order."""
    assert tuple((int(dq), int(dr)) for dq, dr in engine.HEX_AXES) == (
        (1, 0), (0, 1), (1, -1),
    )
