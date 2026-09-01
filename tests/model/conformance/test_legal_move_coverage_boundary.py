# >300 justify (R8): the two arms answer the same question about different producers, and each
# arm's controls only mean anything beside the arm they control — the graph arm's real-path
# control and the dense arm's engine-read coverage fact are the two places where a natural
# workaround silently removes the second producer, and both must go red in the same file.
"""T4 — the action space's legal-move coverage: exact on the graph arm, a derived BOUNDARY on
the dense arm.

TWO GENUINELY DIFFERENT ARMS, and the names say which is which.

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

DENSE ARM — ONE PRODUCER AND A DERIVED BOUNDARY, which is why this module is named `_boundary`:
a green means "the boundary is unchanged", never "coverage is correct". The tier says the
engine's coverage boundary equals the arithmetic implied by the engine-RESOLVED window size and
legal-move radius. A FIX to CNN-3 changes that arithmetic and REDS this tier, which is correct.

THE BOUNDARY IS TWO-SIDED AND ASYMMETRIC, because the centre TRUNCATES. Coverage on an axis
needs BOTH `(max - c) + R <= half` and `(c - min) + R <= half`, with `half = (S - 1) / 2` and
`S`, `R` read off the CONSTRUCTED BOARD (`Board.cluster_window_size()`, `.legal_move_radius()`)
— never from `spec.cluster_window_size`, which is the string `"none"` on `v6`, the encoding
this arm covers. On an ODD span the two sides differ by one and WHICH SIDE IS FARTHER FLIPS
with the sign of the bbox sum, so a single-sided floor half-span `k` is one cell too permissive;
the control below shows `k` predicting coverage where the engine has none.

THE COVERAGE FACT COMES FROM THE ENGINE, NOT FROM A PYTHON RE-IMPLEMENTATION. If the fact were
also this tier's arithmetic, the gate would prove the tier self-consistent and nothing else —
ZERO producers. It is read from `Board.to_flat` (`board.rs:359-361` → `window_flat_idx`,
`core.rs:393-398`), whose off-window sentinel is `usize::MAX`.

DENSE SCOPE, both halves required: the witness is constructed SINGLE-CLUSTER, because the
boundary is derived for one cluster and one window; and EMPTY boards are excluded, because
`moves.rs:109-115` emits a fixed region regardless of `R` and the ball geometry the boundary
rests on does not apply.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantis._engine import Board, HexgBuffer
from mantis.selfplay.graph_collate import GraphWirePayload

from _corpus import (
    ConformanceRefusal,
    build_board,
    cluster_frame_centre,
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
    assert any(arm == ARM_DENSE for _, arm in declared), "no dense arm — the tier has no subject"


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
    """PB-34. A grid encoding's Board against a graph spec must refuse, not produce a red set
    comparison that reads as a completeness bug.

    THE PAIR IS SEARCHED FOR, NOT TAKEN AS "THE FIRST OF EACH", and the difference is not
    cosmetic. `roster()` iterates the registry in an UNORDERED way — observed as
    `v6w25, v6, gnn_axis_r8, v6_live2_ls, gnn_axis_v1` in one run and differently in another —
    so "the first grid" and "the first graph" are whichever the iteration happened to yield.
    While `gnn_axis_v1` (radius 6) was the only graph row, no ordering could collide with a
    grid row, because none is at 6. R328(b) registered `gnn_axis_r8` at radius 8 and `v6w25` is
    also at 8, so an ordering that yields that pair made this control fail its own precondition
    — an ORDER-DEPENDENT red that passes when the file is run alone. Searching for a pair whose
    radii differ asks for what the control actually needs and is order-independent.
    """
    pair = next(
        ((grid, graph) for graph in roster() if graph.is_graph
         for grid in roster() if not grid.is_graph
         and build_board(grid.name, [(0, 0), (1, 0)]).legal_move_radius() != graph.graph_radius),
        None,
    )
    if pair is None:
        pytest.fail(
            "no grid/graph pair in the registry has differing radii, so a radius MISMATCH is "
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


# --------------------------------------------------------------------------------------- #
# DENSE ARM — the two-sided derived boundary
# --------------------------------------------------------------------------------------- #
def require_inside_partition(count: int, enc: str) -> int:
    """PB-37. If the strictly-inside partition is empty the positive half of the dense gate
    asserts nothing and only the witness runs — a gate with nothing inside it."""
    if count <= 0:
        raise EmptyCoveragePartition(
            f"{enc}: the strictly-inside partition is EMPTY, so the positive half of this gate "
            "asserts nothing and only the witness would run."
        )
    return count


def require_witness(outside: list[int], enc: str) -> int:
    """PB-38. A witness that was never constructed is an assertion that never ran, so "no
    witness" is a FAILURE with its own message rather than a quietly skipped clause."""
    if not outside:
        raise NoWitnessConstructed(
            f"{enc}: no position outside the derived boundary was constructed, so the "
            "boundary's other side was never asserted."
        )
    return outside[0]


def require_uncovered_at_witness(holes: list[tuple[int, int]], enc: str, span: int) -> int:
    """The witness's own refusal, behind a helper so a planted break can DRIVE it (R-O1).

    It was an inline `assert` in the gate, which left PB-36 — the permissive coverage stand-in —
    able to show only that the stand-in reports no holes. "And therefore the tier reds" was an
    inference about code the break never executed. It executes now.
    """
    if not holes:
        raise BoundaryNotWhereDerived(
            f"{enc}: the first position outside the boundary (span={span}) covers every legal "
            "move, so the boundary is not where the tier says it is — or the coverage fact came "
            "from something more permissive than the engine's own to_flat."
        )
    return len(holes)


def require_dense_partition(board: Board, ctx: str) -> tuple[int, int]:
    """PB-40. Single-cluster and stone-bearing, verified FROM THE ENGINE; anything else is
    refused rather than silently compared."""
    require_corpus_member(board, ctx)
    centres = board.get_cluster_views()[1]
    if len(centres) != 1:
        raise DensePartitionRefused(
            f"{ctx}: the engine reports {len(centres)} centres. The boundary below is derived "
            "for ONE cluster and ONE window; a multi-cluster position's coverage is a union "
            "over windows and is not what this boundary states."
        )
    return cluster_frame_centre(board)


def boundary_holds(board: Board, centre: tuple[int, int]) -> bool:
    """The two-sided boundary, every term read off the engine, with no `k` shortcut."""
    side = board.cluster_window_size()
    half = (side - 1) // 2
    radius = board.legal_move_radius()
    stones = board.get_stones()
    qs = [q for q, _, _ in stones]
    rs = [r for _, r, _ in stones]
    return (
        (max(qs) - centre[0]) + radius <= half
        and (centre[0] - min(qs)) + radius <= half
        and (max(rs) - centre[1]) + radius <= half
        and (centre[1] - min(rs)) + radius <= half
    )


def uncovered_legal_moves(board: Board) -> list[tuple[int, int]]:
    """The coverage FACT, taken from the engine's own `to_flat`, never re-derived in Python."""
    side = board.cluster_window_size()
    return [(q, r) for q, r in board.legal_moves() if board.to_flat(q, r) >= side * side]


def dense_line_position(span: int) -> list[tuple[int, int]]:
    """A single compact cluster of a chosen bbox span — one stone per cell, so span is exact."""
    return [(i, 0) for i in range(span + 1)]


def dense_scan(enc: str) -> tuple[list[int], list[int]]:
    """`(inside_spans, outside_spans)` for a growing single-cluster line, boundary derived."""
    probe = Board.with_encoding_name(enc)
    reach = probe.cluster_window_size() // 2 + probe.legal_move_radius() + 2
    inside: list[int] = []
    outside: list[int] = []
    for span in range(1, reach):
        board = build_board(enc, dense_line_position(span))
        centre = require_dense_partition(board, f"{enc} span={span}")
        (inside if boundary_holds(board, centre) else outside).append(span)
    return inside, outside


@pytest.mark.parametrize(
    "spec", [s for s in roster() if not s.is_graph], ids=lambda s: s.name
)
def test_dense_coverage_holds_strictly_inside_the_derived_boundary(spec, derived):
    inside, outside = dense_scan(spec.name)
    derived(f"t4.dense.inside_spans.{spec.name}", inside)
    derived(f"t4.dense.outside_spans.{spec.name}", outside)
    require_inside_partition(len(inside), spec.name)
    for span in inside:
        board = build_board(spec.name, dense_line_position(span))
        require_dense_partition(board, f"{spec.name} span={span}")
        holes = uncovered_legal_moves(board)
        if holes:
            raise LegalMoveNotRepresentable(
                f"{spec.name} span={span}: {len(holes)} legal moves fall in no window while the "
                f"derived boundary says coverage holds; first={holes[0]}"
            )
    derived(f"t4.dense.inside_assertions.{spec.name}", len(inside))


@pytest.mark.parametrize(
    "spec", [s for s in roster() if not s.is_graph], ids=lambda s: s.name
)
def test_the_FIRST_position_outside_the_boundary_is_a_named_WITNESS(spec, derived):
    """PB-38. "No witness constructed" is a FAILURE with its own message — a witness that was
    never built is an assertion that never ran."""
    _inside, outside = dense_scan(spec.name)
    span = require_witness(outside, spec.name)
    board = build_board(spec.name, dense_line_position(span))
    require_dense_partition(board, f"{spec.name} witness span={span}")
    holes = uncovered_legal_moves(board)
    derived(f"t4.dense.witness_span.{spec.name}", span)
    derived(f"t4.dense.witness_uncovered.{spec.name}", len(holes))
    require_uncovered_at_witness(holes, spec.name, span)


def test_an_EMPTY_inside_partition_is_refused():
    """PB-37's break, through the same helper the dense gate calls."""
    spec = next(s for s in roster() if not s.is_graph)
    inside, _outside = dense_scan(spec.name)
    assert require_inside_partition(len(inside), spec.name) > 0
    with pytest.raises(EmptyCoveragePartition, match="strictly-inside"):
        require_inside_partition(0, spec.name)


def test_a_MISSING_witness_is_refused():
    """PB-38's break: "no witness constructed" must be a named failure, not a silent pass."""
    spec = next(s for s in roster() if not s.is_graph)
    _inside, outside = dense_scan(spec.name)
    assert require_witness(outside, spec.name) == outside[0]
    with pytest.raises(NoWitnessConstructed, match="never asserted"):
        require_witness([], spec.name)


def test_the_boundary_is_a_function_of_the_ENGINE_reported_S_and_R():
    """PB-41. A stand-in board reporting different geometry must MOVE the computed boundary; a
    boundary computed from a transcribed `S` or `R` would not notice."""
    spec = next(s for s in roster() if not s.is_graph)
    real = build_board(spec.name, dense_line_position(8))
    centre = cluster_frame_centre(real)

    class ShrunkenGeometry:
        def __init__(self, board):
            self._board = board

        def cluster_window_size(self):
            return self._board.cluster_window_size() - 4

        def legal_move_radius(self):
            return self._board.legal_move_radius()

        def get_stones(self):
            return self._board.get_stones()

    assert boundary_holds(real, centre)
    assert not boundary_holds(ShrunkenGeometry(real), centre)


def test_a_ONE_CELL_centre_shift_is_reported_in_BOTH_directions():
    """PB-39. The boundary is two-sided and asymmetric on odd spans, so a break that only
    shifts one way cannot distinguish the `ceil(span/2)` form from the discarded floor `k`."""
    spec = next(s for s in roster() if not s.is_graph)
    inside, _outside = dense_scan(spec.name)
    span = inside[-1]
    board = build_board(spec.name, dense_line_position(span))
    centre = cluster_frame_centre(board)
    assert boundary_holds(board, centre)
    assert not boundary_holds(board, (centre[0] + 1, centre[1]))
    assert not boundary_holds(board, (centre[0] - 1, centre[1]))


def test_the_SINGLE_SIDED_half_span_form_is_one_cell_too_permissive():
    """The `k` form the design discarded, shown wrong against the engine rather than argued
    away: at the first outside-the-boundary span it predicts coverage the engine does not have."""
    spec = next(s for s in roster() if not s.is_graph)
    _inside, outside = dense_scan(spec.name)
    span = outside[0]
    board = build_board(spec.name, dense_line_position(span))
    side = board.cluster_window_size()
    half = (side - 1) // 2
    radius = board.legal_move_radius()
    single_sided_says_covered = (span // 2) + radius <= half
    assert single_sided_says_covered, "this control needs an odd-span first-outside position"
    assert uncovered_legal_moves(board), "the engine covers it after all"
    assert not boundary_holds(board, cluster_frame_centre(board))


def test_a_MULTI_CENTRE_or_STONELESS_position_is_refused_by_the_dense_partition():
    """PB-40's break, both halves."""
    spec = next(s for s in roster() if not s.is_graph)
    probe = Board.with_encoding_name(spec.name)
    separation = 2 * probe.cluster_threshold() + 2
    two_blobs = build_board(
        spec.name, [(0, 0), (1, 0), (separation, 0), (separation, 1)]
    )
    with pytest.raises(DensePartitionRefused, match="centres"):
        require_dense_partition(two_blobs, "planted multi-centre")
    from _corpus import DegenerateCorpusMember

    with pytest.raises(DegenerateCorpusMember):
        require_dense_partition(Board.with_encoding_name(spec.name), "planted stoneless")


def test_a_PERMISSIVE_coverage_stand_in_REDS_the_tier():
    """PB-36. If the coverage FACT were a Python copy of `window_flat_idx_at_geom` the gate
    would prove the tier self-consistent and nothing else — zero producers."""
    spec = next(s for s in roster() if not s.is_graph)
    _inside, outside = dense_scan(spec.name)
    board = build_board(spec.name, dense_line_position(outside[0]))
    assert uncovered_legal_moves(board), "the engine reports full coverage at the witness"

    class EverythingInWindow:
        def __init__(self, board):
            self._board = board

        def cluster_window_size(self):
            return self._board.cluster_window_size()

        def legal_moves(self):
            return self._board.legal_moves()

        def to_flat(self, q, r):
            del q, r
            return 0

    assert uncovered_legal_moves(EverythingInWindow(board)) == []
    with pytest.raises(BoundaryNotWhereDerived, match="more permissive"):
        require_uncovered_at_witness(
            uncovered_legal_moves(EverythingInWindow(board)), spec.name, outside[0]
        )


# --------------------------------------------------------------------------------------- #
# REPORT (`slow`)
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_report_the_uncovered_legal_move_distribution_per_grid_encoding(derived):
    """The distribution of uncovered-legal-move counts per position, per registered grid
    encoding. NO THRESHOLD. PLAN-E's prescribed `uncovered_forced_win` counter is an ENGINE
    change on the self-play path and is out of scope; this report is the measurement that would
    justify dispatching it, not the counter itself."""
    distribution: dict[str, list[tuple[int, int]]] = {}
    for spec in roster():
        if spec.is_graph:
            continue
        rows: list[tuple[int, int]] = []
        inside, outside = dense_scan(spec.name)
        for span in inside + outside:
            board = build_board(spec.name, dense_line_position(span))
            rows.append((span, len(uncovered_legal_moves(board))))
        distribution[spec.name] = rows
    derived("t4.report.uncovered_by_span", distribution)
    assert distribution
