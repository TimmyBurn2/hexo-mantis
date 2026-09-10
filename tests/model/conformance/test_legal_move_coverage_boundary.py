# >300 justify (R8): the arm and every control that shows it can refuse are ONE unit — the
# real-path control, the length-preserving substitution and the radius refusal only mean
# anything beside the arm they control, and a workaround that silently removes the second
# producer has to go red in the same file as the arm it removed it from.
"""T4 — the action space's legal-move coverage, exact on the graph arm.

GRAPH ARM — TWO PRODUCERS, and they cannot collude. `mantis-graph` computes the legal set from
STONES ALONE (`legal_moves_from_stones`, `lib.rs:393`, called at `:483` inside
`build_axis_graph`, whose only position input is the stone list) while `mantis-core` computes it
independently (`Board::legal_moves`, `moves.rs:173`, over the hex ball at `:118-122`).
`mantis-graph` is dep-free by the repo's own DAG, so it cannot call `mantis-core`. What is NEW
here is the CROSS-CRATE half only: `verify_contract` already asserts
`legal_node_gather.len() == n_legal` on every build (`lib.rs:852`), but `n_legal` is
`mantis-graph`'s OWN count, so that assert is self-consistency inside one crate and cannot see
the two crates disagreeing. This tier adds the coordinate SET against `mantis-core` and the
count against `Board.legal_move_count()`. The wire's own named errors (`EmptyLegalSet`,
`GatherNotLegalNode`, …) are NOT re-implemented here.

THE EMPTY-BOARD CASE IS SINGLE-PRODUCER AND IS LABELLED AS ONE. Both crates hard-code the same
25-cell fallback, radius-independent (`mantis-graph/src/lib.rs:397-405`,
`mantis-core/src/board/moves.rs:109-115`), so on an empty board the "two producers" are two
transcriptions of one literal and the arm is green for any radius. The stone-bearing partition
is therefore asserted non-empty and pinned.

THE SHARED RADIUS IS AN ASSERTION, NOT AN ASSUMPTION. The wire's radius comes from
`spec.graph_radius` (`crates/mantis-selfplay/src/replay/hexg/mod.rs:263`); the Board's from
`spec.legal_move_radius`. They coincide only where the registry says so, and a Board built
under a mismatched encoding must produce a NAMED REFUSAL rather than a red set-comparison that
a reader would take for a completeness bug.

THE DENSE ARM IS GONE. It was one producer and a derived boundary — the engine's coverage
boundary against the arithmetic implied by `Board.cluster_window_size()` and
`.legal_move_radius()` — and the K-cluster window it rested on went with the grid path
(R346(f)). What is left is the two-producer graph arm, which is the half that could never
collude.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis._engine import Board, HexgBuffer
from mantis.selfplay.graph_collate import GraphWirePayload

from _corpus import (
    ConformanceRefusal,
    build_board,
    graph_wire_for,
    require_corpus_member,
    roster,
)

ARM_GRAPH = "graph"
ARM_DENSE = "dense"


class LegalMoveNotRepresentable(ConformanceRefusal):
    """A legal move the action space cannot represent."""


class WireSurfaceMismatch(ConformanceRefusal):
    """The tier was handed something other than the WIRE payload."""


class RadiusDisagreement(ConformanceRefusal):
    """The graph wire's radius and the Board's legal-move radius are not the same parameter."""


class ArmMatrixDisagreement(ConformanceRefusal):
    """The arm matrix the specs declare differs from the one the engine admits."""


class EmptyCoveragePartition(ConformanceRefusal):
    """A partition this tier quantifies over is empty."""


class NoWitnessConstructed(ConformanceRefusal):
    """No position outside the boundary was constructed, so its property was never asserted."""


class BoundaryNotWhereDerived(ConformanceRefusal):
    """The first position OUTSIDE the derived boundary covers every legal move after all."""


class DensePartitionRefused(ConformanceRefusal):
    """A multi-centre or stoneless position was injected into the single-cluster dense arm."""


def derived_arm_matrix(specs) -> frozenset[tuple[str, str]]:
    """`(encoding, arm)` derived from `spec.is_graph`, pinned as a set — never a name list."""
    return frozenset(
        (spec.name, ARM_GRAPH if spec.is_graph else ARM_DENSE) for spec in specs
    )


# --------------------------------------------------------------------------------------- #
# GRAPH ARM
# --------------------------------------------------------------------------------------- #
def require_wire_payload(payload) -> None:
    """PB-31. `node_coords` is TWO things with one name: the WIRE array is live, the
    same-named DEVICE tensor on `GraphBatch` was RETIRED by R297(c). Reading the batch-side name
    would raise — or worse, motivate re-adding a dead host-to-device transfer."""
    if not isinstance(payload, GraphWirePayload):
        raise WireSurfaceMismatch(
            f"expected the wire payload GraphWirePayload, got {type(payload).__name__}. The "
            "batch-side `node_coords` was retired; this tier reads the wire."
        )


def require_radius_agreement(board: Board, spec) -> int:
    if spec.graph_radius is None or board.legal_move_radius() != spec.graph_radius:
        raise RadiusDisagreement(
            f"{spec.name}: the Board reports legal_move_radius={board.legal_move_radius()} "
            f"while the wire is built at graph_radius={spec.graph_radius}. The two crates are "
            "comparable only when they consume the SAME radius; this is an assertion, not an "
            "assumption."
        )
    return spec.graph_radius


def gathered_coords(payload) -> set[tuple[int, int]]:
    require_wire_payload(payload)
    coords = np.asarray(payload.node_coords).reshape(-1, 2)
    gather = np.asarray(payload.legal_node_gather)
    return {(int(q), int(r)) for q, r in coords[gather]}


def require_completeness(payload, board: Board, ctx: str) -> int:
    """The cross-crate half: the coordinate SET and the count, neither statable in-crate."""
    require_wire_payload(payload)
    wire_set = gathered_coords(payload)
    core_set = {(int(q), int(r)) for q, r in board.legal_moves()}
    if wire_set != core_set:
        raise LegalMoveNotRepresentable(
            f"{ctx}: the graph wire's gathered legal nodes differ from mantis-core's legal set. "
            f"only-on-wire={sorted(wire_set - core_set)}; "
            f"only-in-core={sorted(core_set - wire_set)}"
        )
    gathered = int(np.asarray(payload.legal_node_gather).size)
    if gathered != board.legal_move_count():
        raise LegalMoveNotRepresentable(
            f"{ctx}: legal_node_gather carries {gathered} rows against "
            f"Board.legal_move_count()={board.legal_move_count()}"
        )
    return gathered


def require_stone_bearing_partition(count: int, enc: str) -> int:
    """PB-35. The empty-board case may stay a case; it may not be counted as two
    producers, so the stone-bearing partition carries its own non-empty refusal."""
    if count <= 0:
        raise EmptyCoveragePartition(
            f"{enc}: the stone-bearing partition is EMPTY. On an empty board both crates "
            "hard-code the same fallback region, so this arm would be green for any radius "
            "and its two producers are two transcriptions of one literal."
        )
    return count


def stone_bearing_graph_corpus() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
    ]


@pytest.mark.parametrize(
    "spec", [s for s in roster() if s.is_graph], ids=lambda s: s.name
)
def test_the_graph_wire_carries_exactly_mantis_cores_legal_set(spec, derived):
    checked = 0
    for moves in stone_bearing_graph_corpus():
        board = build_board(spec.name, moves)
        require_corpus_member(board, f"{spec.name} graph corpus")
        require_radius_agreement(board, spec)
        payload = graph_wire_for(spec.name, board)
        require_completeness(payload, board, f"{spec.name}/{moves}")
        checked += 1
    derived(
        f"t4.graph.stone_bearing_positions.{spec.name}",
        require_stone_bearing_partition(checked, spec.name),
    )


def engine_admitted_arm_matrix(specs) -> frozenset[tuple[str, str]]:
    """`(encoding, arm)` OBSERVED FROM THE ENGINE, which is the side `derived_arm_matrix` lacks.

    `HexgBuffer::new` refuses a grid encoding by construction
    (`crates/mantis-selfplay/src/replay/hexg/mod.rs:246-252`), so which encodings the graph
    replay surface admits is a fact the engine holds, and `spec.is_graph` is a CLAIM about that
    fact. Constructing the buffer is how the claim gets a second producer.
    """
    observed: set[tuple[str, str]] = set()
    for spec in specs:
        try:
            HexgBuffer(2, spec.name, 8)
        except (ValueError, TypeError):
            observed.add((spec.name, ARM_DENSE))
        else:
            observed.add((spec.name, ARM_GRAPH))
    return frozenset(observed)


def require_arm_matrix_agreement(declared: frozenset, admitted: frozenset) -> int:
    if declared != admitted:
        raise ArmMatrixDisagreement(
            f"the arm matrix the SPECS declare and the one the ENGINE admits disagree: "
            f"declared-only={sorted(declared - admitted)}; engine-only={sorted(admitted - declared)}"
        )
    return len(declared)


def test_the_arm_matrix_the_SPECS_declare_equals_the_one_the_ENGINE_admits(derived):
    """PB-30. Both sides used to come from ONE source: `expected` was `derived_arm_matrix(specs)`
    and `executed` was a re-typing of that function's own generator expression, so
    `executed == expected` could not fail for ANY input — measured over four stand-in rosters,
    including one that lost the graph flag on every spec and one that was empty, all four True.
    That is the F2 defect class, found and fixed in T1's frame matrix and left standing here;
    its practical cost was measured too, when one `slow` marker removed the graph arm from the
    CI tier and this test still reported the arm as present.

    The second side is the engine's own refusal, which no spec field produces.
    """
    specs = roster()
    declared = derived_arm_matrix(specs)
    admitted = engine_admitted_arm_matrix(specs)
    derived("t4.arm_matrix.declared", sorted(declared))
    derived("t4.arm_matrix.engine_admitted", sorted(admitted))
    derived("t4.arm_matrix.cardinality", require_arm_matrix_agreement(declared, admitted))
    assert any(arm == ARM_GRAPH for _, arm in declared), "no graph arm — the tier has no subject"
    assert not any(arm == ARM_DENSE for _, arm in declared), (
        "a registered encoding claims the DENSE arm, which R346(f) deleted — the engine would "
        "have to refuse its own registry row for the matrix to still agree"
    )


class _ArmClaim:
    """A spec stand-in carrying only what the arm matrix reads: a registered name and a claim."""

    def __init__(self, name: str, is_graph: bool) -> None:
        self.name = name
        self.is_graph = is_graph


def test_a_FLIPPED_arm_claim_is_refused_by_the_engine_side():
    """The break the one-source comparison structurally cannot see. The same input is fed to
    both comparisons: the old shape is still satisfied, the engine side names the encoding."""
    specs = [_ArmClaim(spec.name, not spec.is_graph) for spec in roster()]
    one_source = frozenset(
        (spec.name, ARM_GRAPH if spec.is_graph else ARM_DENSE) for spec in specs
    )
    assert one_source == derived_arm_matrix(specs), (
        "the comparison this replaces is SATISFIED by a roster with every arm claim flipped"
    )
    with pytest.raises(ArmMatrixDisagreement, match="declared-only"):
        require_arm_matrix_agreement(derived_arm_matrix(specs), engine_admitted_arm_matrix(specs))


def test_an_EMPTY_roster_cannot_pass_the_arm_matrix():
    """The old comparison was True on the empty roster; the subject asserts survive it, and
    this states which assertion is doing that work rather than leaving it to be inferred."""
    assert require_arm_matrix_agreement(derived_arm_matrix([]), engine_admitted_arm_matrix([])) == 0
    assert not any(arm == ARM_GRAPH for _, arm in derived_arm_matrix([]))


def test_a_LENGTH_PRESERVING_coordinate_substitution_fails_the_SET_half():
    """PB-32. A length-only drop is already caught in-crate by `verify_contract:852`, so the
    break must be length-preserving or it exercises a relation the engine already asserts."""
    spec = next(s for s in roster() if s.is_graph)
    board = build_board(spec.name, [(0, 0), (1, 0)])
    payload = graph_wire_for(spec.name, board)
    require_completeness(payload, board, "control")
    coords = np.array(payload.node_coords).reshape(-1, 2)
    gather = np.asarray(payload.legal_node_gather)
    coords[gather[0]] = coords[gather[0]] + np.asarray([100, 100])
    mutated = GraphWirePayload(
        **{**payload.__dict__, "node_coords": coords.reshape(-1)}
    )
    assert np.asarray(mutated.legal_node_gather).size == gather.size
    with pytest.raises(LegalMoveNotRepresentable, match="only-on-wire"):
        require_completeness(mutated, board, "planted")


def test_the_REAL_PATH_control_REDS_on_a_wrong_position():
    """PB-33. Proving the comparator rejects a hand-edited array says nothing about the tier
    catching a real regression; both sides here are production surfaces."""
    spec = next(s for s in roster() if s.is_graph)
    moves = [(0, 0), (1, 0)]
    payload = graph_wire_for(spec.name, build_board(spec.name, moves))
    shifted_board = build_board(spec.name, [*moves, (2, 0)])
    with pytest.raises(LegalMoveNotRepresentable):
        require_completeness(payload, shifted_board, "real-path control")


def test_a_MISMATCHED_radius_is_refused_by_name():
    """PB-34. A Board built under one encoding, checked against a spec at a DIFFERENT radius,
    must refuse — not produce a red set comparison that reads as a completeness bug.

    THE PAIR IS SEARCHED FOR, NOT TAKEN AS "THE FIRST TWO", and the difference is not cosmetic.
    `roster()` iterates the registry in an UNORDERED way, so "the first" and "the second" are
    whichever the iteration happened to yield, and a pair that happens to share a radius makes
    the control fail its own precondition — an ORDER-DEPENDENT red that passes when the file is
    run alone. Searching for a pair whose radii differ asks for what the control needs and is
    order-independent. The pair used to be one grid row and one graph row; since R346(f) both
    members are graph rows, which is why the search is over the whole roster both ways.
    """
    pair = next(
        ((other, graph) for graph in roster() if graph.is_graph
         for other in roster()
         if build_board(other.name, [(0, 0), (1, 0)]).legal_move_radius() != graph.graph_radius),
        None,
    )
    if pair is None:
        pytest.fail(
            "no pair in the registry has differing radii, so a radius MISMATCH is "
            "unconstructible and this control has no subject"
        )
    grid_spec, graph_spec = pair
    board = build_board(grid_spec.name, [(0, 0), (1, 0)])
    assert board.legal_move_radius() != graph_spec.graph_radius, "the search returned a match"
    with pytest.raises(RadiusDisagreement, match="SAME radius"):
        require_radius_agreement(board, graph_spec)


def test_a_BATCH_SIDE_stand_in_is_refused_rather_than_silently_accepted():
    """PB-31's break: the retired device-side field must not become an alternative path."""
    class NotTheWire:
        node_coords = np.zeros(4, dtype=np.int32)
        legal_node_gather = np.zeros(2, dtype=np.int64)

    with pytest.raises(WireSurfaceMismatch, match="GraphWirePayload"):
        require_wire_payload(NotTheWire())


def test_an_EMPTY_stone_bearing_graph_partition_is_refused():
    """PB-35's break, run through the SAME helper the gate calls — a control that raises the
    exception itself proves nothing about the guard."""
    spec = next(s for s in roster() if s.is_graph)
    assert require_stone_bearing_partition(len(stone_bearing_graph_corpus()), spec.name) > 0
    with pytest.raises(EmptyCoveragePartition, match="EMPTY"):
        require_stone_bearing_partition(0, spec.name)
