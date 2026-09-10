"""AUDIT-1 F-42 — the geometry constants are ONE set on both sides of the FFI.

THE DEFECT. The three hex axes and the win length were each typed on BOTH sides with nothing
pinning across: `env/game_state.py`, `selfplay/graph_collate.py` and
`selfplay/instrumentation.py` each typed the axis table beside `mantis_core::board::HEX_AXES`
/ `mantis_graph::WIN_AXES`. The audit's phrase for it: Python pins Python, Rust pins a
literal, nothing pins across the FFI.

The four dense SOURCE-PLANE rows this file also carried are RETIRED with the grid path
(R346(f)): `_CUR_STONE_SRC_PLANE` .. `_PLY_PARITY_SRC_PLANE`, the
`OPP_STONE_PLANE == HISTORY_LEN` relation and the `v6_live2_ls.kept_plane_indices` row all
described an 18-plane wire format that no registered encoding produces. What remains is the
axis table and the win length, which the GRAPH builder reads.

WHAT IS PINNED HERE, and why each row is not the others:

- the two Python axis modules and the engine are one ordered table;
- the win length is the engine's on both Python copies;
- the ORDER itself is pinned to literals, because agreement alone stays green if every copy
  moves together — and every frozen fixture and every trained net was built against THIS
  order.
"""
from __future__ import annotations

import mantis._engine as engine
from mantis.selfplay.graph_collate import WIN_AXES
from mantis.selfplay.instrumentation import _HEX_AXES as _INSTR_HEX_AXES
from mantis.selfplay.instrumentation import _WIN_LENGTH as _INSTR_WIN_LENGTH


def test_every_python_axis_table_is_the_engines() -> None:
    engine_axes = tuple((int(dq), int(dr)) for dq, dr in engine.HEX_AXES)
    assert WIN_AXES == engine_axes, "selfplay.graph_collate (edge one-hot order)"
    assert tuple(_INSTR_HEX_AXES) == engine_axes, "selfplay.instrumentation (line scan)"


def test_every_python_win_length_is_the_engines() -> None:
    assert _INSTR_WIN_LENGTH == engine.WIN_LENGTH, "instrumentation longest_line cap"


def test_the_axis_order_itself_is_pinned() -> None:
    """Agreement alone stays green if every copy moves together — and every frozen fixture
    and every trained net was built against THIS order."""
    assert tuple((int(dq), int(dr)) for dq, dr in engine.HEX_AXES) == (
        (1, 0), (0, 1), (1, -1),
    )
