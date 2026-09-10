"""Shared constructors, refusals and engine readers for the architecture conformance suite.

A helper module (leading `_`), so it is not collected; every tier imports from here, so a break
planted against one tier exercises the code the gate runs. NO TUNABLE LITERAL LIVES HERE: the
window side and legal-move radius are read off a CONSTRUCTED board at the point of use.

The board frame is not on the Python surface — it is observable only by inverting `to_flat`, and
recomputing the midpoint in Python would be a second authority over the rule under census.
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
    """A corpus member is stoneless, so the reported centre is a constant."""


class BoardFrameUnreadable(ConformanceRefusal):
    """`to_flat` reported the probe cell off-window, so the board frame cannot be recovered."""


class GraphArmUnavailable(ConformanceRefusal):
    """The graph wire could not be built for an encoding whose spec says it is a graph."""


class RosterCollapsed(ConformanceRefusal):
    """The parametrisation roster is empty or has shrunk against the live registry surface."""


# The six unit axial directions, closed under negation.
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

FRAME_DENSE_BOARD = "dense_board"
FRAME_GRAPH = "graph"


def roster() -> tuple[Any, ...]:
    """Return the parametrisation roster — the public registry surface, never hand-typed."""
    return tuple(encoding.all_specs())


def roster_names(specs: tuple[Any, ...] | None = None) -> tuple[str, ...]:
    return tuple(sorted(s.name for s in (roster() if specs is None else specs)))


def check_roster(observed: tuple[str, ...], live: tuple[str, ...]) -> int:
    """Refuse an empty or shrunken roster; return the cardinality as a derived output.
    `empty_parameter_set_mark` defaults to `skip`, so an empty roster is a SILENT SKIP."""
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
    """Build a board for `enc` with `moves` applied in order."""
    board = Board.with_encoding_name(enc)
    for q, r in moves:
        board.apply_move(q, r)
    return board


def translate(moves: list[tuple[int, int]], t: tuple[int, int]) -> list[tuple[int, int]]:
    """Place the same relative geometry at an origin offset by `t`."""
    return [(q + t[0], r + t[1]) for q, r in moves]


def bbox_sums(moves: list[tuple[int, int]]) -> tuple[int, int]:
    """Return `(min + max)` per axis — the quantity `a` of the signed rule."""
    qs = [q for q, _ in moves]
    rs = [r for _, r in moves]
    return (min(qs) + max(qs), min(rs) + max(rs))


def signed_delta(a: int, t: int) -> int:
    """The exact delta of the engine's truncating centre under a translation by `t`:
    `c(a) = trunc(a/2)` and `2t` preserves the parity of `a`, so the delta is
    `t + odd(a) * ([a + 2t < 0] - [a < 0])`. Computed, never typed."""
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
    """Assert at the point of use that a corpus member is stone-bearing."""
    if not board.get_stones():
        raise DegenerateCorpusMember(
            f"{ctx}: the position carries no stones. Board::window_center returns a constant "
            "(0, 0) when !has_stones (core.rs:378-380), which is translation-invariant and "
            "disagrees with the signed rule — refused rather than compared."
        )



def board_frame_centre(board: Board) -> tuple[int, int]:
    """Return the dense BOARD frame origin, inverted out of the engine's own `to_flat` index:
    it maps `(q, r)` to `(q - cq + half) * S + (r - cr + half)`, so probing a stone the board
    reports and inverting reads the frame off the engine rather than recomputing it."""
    stones = board.get_stones()
    if not stones:
        raise DegenerateCorpusMember("board frame: the position carries no stones")
    # `Board.size` is the window side: `window_flat_idx` indexes at the encoding's
    # `trunk_size`, and every registered row mints `trunk_size == board_size`.
    side = board.size
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
    """Build the graph wire for one position, through the production surface."""
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
    """Return the graph arm's window origin, read off the wire the builder produced."""
    payload = graph_wire_for(enc, board)
    centre = np.asarray(payload.window_center).reshape(-1, 2)
    if centre.shape[0] != 1:
        raise GraphArmUnavailable(
            f"{enc}: the wire carried {centre.shape[0]} window centres for one position"
        )
    return (int(centre[0][0]), int(centre[0][1]))
