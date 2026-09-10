# >300 justify (R8): the arm and every control that shows it can refuse are ONE unit — the
# real-path control, the length-preserving substitution and the radius refusal only mean anything
# beside the arm they control, and a workaround that removes the second producer has to go red in
# the same file as the arm it removed it from.
"""Check the action space's legal-move coverage against two producers that cannot collude.

The graph crate computes the legal set from stones alone and is dep-free by the repo's DAG, so it
cannot call the core crate that computes it independently; the wire's own contract check only
compares a count against the graph crate's OWN count. On an EMPTY board both crates hard-code the
same fallback, hence the stone-bearing partition's non-empty refusal.
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
    """Derive `(encoding, arm)` from `spec.is_graph`, as a set rather than a name list."""
    return frozenset(
        (spec.name, ARM_GRAPH if spec.is_graph else ARM_DENSE) for spec in specs
    )


def require_wire_payload(payload) -> None:
    """Refuse anything but the wire payload; the same-named device tensor was retired."""
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
    """Check the coordinate SET and the count across crates, neither statable in-crate."""
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
    """Refuse an empty stone-bearing partition, which leaves only a single-producer case."""
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
    """Observe `(encoding, arm)` from the engine, which admits or refuses an encoding by
    construction — a fact the engine holds, against which `spec.is_graph` is only a claim."""
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
    """Prove the arm matrix the specs declare equals the one the engine admits; both sides once
    came from one source and could not fail for ANY input, measured over four stand-in rosters."""
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
    """Stand in for a spec, carrying only what the arm matrix reads."""

    def __init__(self, name: str, is_graph: bool) -> None:
        self.name = name
        self.is_graph = is_graph


def test_a_FLIPPED_arm_claim_is_refused_by_the_engine_side():
    """Prove a flipped arm claim is refused by the engine side, which one source cannot see."""
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
    """Prove an empty roster cannot pass, and name which assertion does that work."""
    assert require_arm_matrix_agreement(derived_arm_matrix([]), engine_admitted_arm_matrix([])) == 0
    assert not any(arm == ARM_GRAPH for _, arm in derived_arm_matrix([]))


def test_a_LENGTH_PRESERVING_coordinate_substitution_fails_the_SET_half():
    """Prove a length-preserving coordinate substitution fails the SET half."""
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
    """Prove the comparator reds on a wrong position with both sides production surfaces."""
    spec = next(s for s in roster() if s.is_graph)
    moves = [(0, 0), (1, 0)]
    payload = graph_wire_for(spec.name, build_board(spec.name, moves))
    shifted_board = build_board(spec.name, [*moves, (2, 0)])
    with pytest.raises(LegalMoveNotRepresentable):
        require_completeness(payload, shifted_board, "real-path control")


def test_a_MISMATCHED_radius_is_refused_by_name():
    """Prove a mismatched radius is refused by name. The pair is SEARCHED for, not taken as the
    first two: `roster()` is unordered, so a pair sharing a radius would fail the precondition."""
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
    """Prove a batch-side stand-in is refused, so the retired field is not an alternative path."""
    class NotTheWire:
        node_coords = np.zeros(4, dtype=np.int32)
        legal_node_gather = np.zeros(2, dtype=np.int64)

    with pytest.raises(WireSurfaceMismatch, match="GraphWirePayload"):
        require_wire_payload(NotTheWire())


def test_an_EMPTY_stone_bearing_graph_partition_is_refused():
    """Prove an empty stone-bearing partition is refused, through the helper the gate calls."""
    spec = next(s for s in roster() if s.is_graph)
    assert require_stone_bearing_partition(len(stone_bearing_graph_corpus()), spec.name) > 0
    with pytest.raises(EmptyCoveragePartition, match="EMPTY"):
        require_stone_bearing_partition(0, spec.name)
