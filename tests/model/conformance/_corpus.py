"""Shared constructors, refusals and engine readers for the architecture conformance suite.

Helper module (leading `_`, the convention `tests/model/_bf16_parity.py` already sets): it is
not collected. Every tier imports its board construction, its named refusals and its three
frame readers from here so that a break planted against one tier exercises the same
construction code the gate runs (R-O1, mechanism-not-proxy).

NO TUNABLE LITERAL LIVES HERE (R1/R26). The window side `S`, the cluster threshold and the
legal-move radius are read off a CONSTRUCTED board at the point of use —
`Board.with_encoding_name(enc).cluster_window_size()` / `.cluster_threshold()` /
`.legal_move_radius()` — never from `spec.cluster_window_size`, which is the string `"none"`
on `v6` and `gnn_axis_v1` (`crates/mantis-encoding/src/registry.toml:57`, `:164`) and would
force either a raise or a planted `19`.

THE BOARD FRAME IS NOT ON THE PYTHON SURFACE. `Board::window_center()`
(`crates/mantis-core/src/board/state/core.rs:377`) has no `PyBoard` getter; the frame is
observable only through `to_flat` (`crates/mantis-bridge/src/board.rs:359` →
`Board::window_flat_idx`, `core.rs:393-398`, `usize::MAX` off-window). `board_frame_centre`
inverts the engine's own reported index rather than recomputing the midpoint in Python — a
Python midpoint here would be a second authority over the rule the suite exists to census.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import mantis.encoding as encoding
from mantis._engine import Board, HexgBuffer
from mantis.selfplay.graph_collate import graph_wire_from_rust


class ConformanceRefusal(AssertionError):
    """Base of every refusal in this suite. A tier that cannot construct its subject FAILS."""


class DegenerateCorpusMember(ConformanceRefusal):
    """A corpus member is stoneless or centre-less.

    `get_cluster_views` pushes `(0, 0)` with no clusters (`cluster.rs:48-49`) and
    `Board::window_center` returns `(0, 0)` with no stones (`core.rs:378-380`); either is a
    translation-invariant constant that DISAGREES with the signed rule, so such a member is a
    red-at-HEAD risk rather than a vacuity. It is refused at the point of use, by name.
    """


class BoardFrameUnreadable(ConformanceRefusal):
    """`to_flat` reported the probe cell off-window, so the board frame cannot be recovered."""


class GraphArmUnavailable(ConformanceRefusal):
    """The graph wire could not be built for an encoding whose spec says it is a graph."""


class RosterCollapsed(ConformanceRefusal):
    """The parametrisation roster is empty or has shrunk against the live registry surface."""


# The six unit axial directions. The set is closed under negation, so both crossing
# directions are constructed by pairing it with base shapes of both signs (below).
UNIT_AXIAL: tuple[tuple[int, int], ...] = (
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
)

#: Sign-class names. A class whose executed-assertion counter is zero fails the tier.
CLASS_EVEN = "even"
CLASS_ODD_NO_CROSS = "odd_no_cross"
CLASS_ODD_CROSS_TO_NEGATIVE = "odd_cross_to_negative"
CLASS_ODD_CROSS_TO_NON_NEGATIVE = "odd_cross_to_non_negative"
SIGN_CLASSES: tuple[str, ...] = (
    CLASS_EVEN, CLASS_ODD_NO_CROSS, CLASS_ODD_CROSS_TO_NEGATIVE, CLASS_ODD_CROSS_TO_NON_NEGATIVE,
)

FRAME_DENSE_CLUSTER = "dense_cluster"
FRAME_DENSE_BOARD = "dense_board"
FRAME_GRAPH = "graph"


def roster() -> tuple[Any, ...]:
    """The parametrisation roster: the public registry surface, never a hand-typed tuple."""
    return tuple(encoding.all_specs())


def roster_names(specs: tuple[Any, ...] | None = None) -> tuple[str, ...]:
    return tuple(sorted(s.name for s in (roster() if specs is None else specs)))


def check_roster(observed: tuple[str, ...], live: tuple[str, ...]) -> int:
    """Refuse an empty or shrunken roster; return the cardinality as a derived output.

    Separated from the test that calls it so the planted breaks can drive it with a stand-in
    roster (PC-1). pytest's default `empty_parameter_set_mark` is `skip` and
    `pyproject.toml` sets none, so an empty roster would otherwise be a SILENT SKIP.
    """
    if not observed:
        raise RosterCollapsed(
            "the conformance parametrisation roster is EMPTY. Under pytest's default "
            "empty_parameter_set_mark every tier in this suite would collect one SKIPPED item "
            "and CI would show skips, not failures — the vacuous pass this suite exists to "
            "prevent, arriving through the parametrisation machinery."
        )
    if set(observed) != set(live):
        raise RosterCollapsed(
            f"the roster {observed} differs from the live registry surface {live} — "
            "mantis.encoding.all_specs() reads a process-global cache this suite does not own, "
            "so a session-mate that patches it silently shrinks every tier's subject."
        )
    return len(observed)


def build_board(enc: str, moves: list[tuple[int, int]]) -> Board:
    """A board constructed for `enc` with `moves` applied in order. Path A of T3."""
    board = Board.with_encoding_name(enc)
    for q, r in moves:
        board.apply_move(q, r)
    return board


def translate(moves: list[tuple[int, int]], t: tuple[int, int]) -> list[tuple[int, int]]:
    """The same relative geometry placed at an origin offset by `t`."""
    return [(q + t[0], r + t[1]) for q, r in moves]


def bbox_sums(moves: list[tuple[int, int]]) -> tuple[int, int]:
    """`(min + max)` per axis over the placed cells — the quantity `a` of the signed rule."""
    qs = [q for q, _ in moves]
    rs = [r for _, r in moves]
    return (min(qs) + max(qs), min(rs) + max(rs))


def signed_delta(a: int, t: int) -> int:
    """The exact delta of the engine's truncating centre under a translation by `t`.

    `c(a) = trunc(a/2)` (Rust i32 `/`), `c(a) = floor(a/2) + [a < 0 and a odd]`, and `2t`
    preserves the parity of `a`, so

        c(a + 2t) - c(a) = t + odd(a) * ( [a + 2t < 0] - [a < 0] ).

    Computed here from `a` and `t`; never typed as a constant, and carrying no tolerance.
    """
    odd = 1 if (a % 2) != 0 else 0
    return t + odd * ((1 if a + 2 * t < 0 else 0) - (1 if a < 0 else 0))


def sign_class(a: int, t: int) -> str:
    if (a % 2) == 0:
        return CLASS_EVEN
    if (a < 0) == (a + 2 * t < 0):
        return CLASS_ODD_NO_CROSS
    if a >= 0:
        return CLASS_ODD_CROSS_TO_NEGATIVE
    return CLASS_ODD_CROSS_TO_NON_NEGATIVE


def require_corpus_member(board: Board, ctx: str) -> None:
    """Every corpus member is stone-bearing AND centre-bearing, asserted at the point of use."""
    if not board.get_stones():
        raise DegenerateCorpusMember(
            f"{ctx}: the position carries no stones. Board::window_center returns a constant "
            "(0, 0) when !has_stones (core.rs:378-380), which is translation-invariant and "
            "disagrees with the signed rule — refused rather than compared."
        )
    if len(board.get_cluster_views()[1]) < 1:
        raise DegenerateCorpusMember(
            f"{ctx}: the position has no cluster centre. get_cluster_views pushes (0, 0) with "
            "no clusters (cluster.rs:48-49) — refused rather than compared."
        )


def cluster_frame_centre(board: Board) -> tuple[int, int]:
    """The dense CLUSTER frame origin: the engine's own first reported centre."""
    centres = board.get_cluster_views()[1]
    if not centres:
        raise DegenerateCorpusMember("cluster frame: the engine reported no centre")
    return (int(centres[0][0]), int(centres[0][1]))


def board_frame_centre(board: Board) -> tuple[int, int]:
    """The dense BOARD frame origin, inverted out of the engine's own `to_flat` index.

    `window_flat_idx_at_geom` (`core.rs:422-431`) maps `(q, r)` to
    `(q - cq + half) * S + (r - cr + half)`, or `usize::MAX` off-window. Probing a stone the
    board itself reports and inverting that index reads the frame off the engine; recomputing
    the midpoint in Python would make this a second authority over the rule under census.
    """
    stones = board.get_stones()
    if not stones:
        raise DegenerateCorpusMember("board frame: the position carries no stones")
    side = board.cluster_window_size()
    half = (side - 1) // 2
    q, r, _ = stones[0]
    flat = board.to_flat(q, r)
    if flat >= side * side:
        raise BoardFrameUnreadable(
            f"to_flat({q}, {r}) = {flat} is the off-window sentinel; the board frame cannot be "
            "recovered from a probe the engine places outside its own window."
        )
    wq, wr = divmod(flat, side)
    return (q - wq + half, r - wr + half)


def graph_wire_for(enc: str, board: Board) -> Any:
    """The graph wire for one position, through the production surface.

    `HexgBuffer::new` refuses a grid encoding by construction
    (`crates/mantis-selfplay/src/replay/hexg/mod.rs:246-252`), which is why the graph frame
    exists for exactly one registered encoding.
    """
    require_corpus_member(board, f"graph wire for {enc}")
    legal = board.legal_moves()
    if not legal:
        raise GraphArmUnavailable(f"{enc}: the position has no legal move to carry a visit row")
    buffer = HexgBuffer(2, enc, 8)
    buffer.push_graph_position(
        board.get_stones(), [(legal[0][0], legal[0][1], 1.0)],
        board.current_player, board.moves_remaining, board.ply, True, 0.0, True, 8,
    )
    wire, _targets = buffer.sample_graph_batch(1, augment=False)
    return graph_wire_from_rust(wire)


def graph_frame_centre(enc: str, board: Board) -> tuple[int, int]:
    """The graph arm's window origin, read off the wire `mantis-graph` produced."""
    payload = graph_wire_for(enc, board)
    centre = np.asarray(payload.window_center).reshape(-1, 2)
    if centre.shape[0] != 1:
        raise GraphArmUnavailable(
            f"{enc}: the wire carried {centre.shape[0]} window centres for one position"
        )
    return (int(centre[0][0]), int(centre[0][1]))
