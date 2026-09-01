# >300 justify (R8): the two engine paths, the provenance record that keeps them two, the three
# comparators, the partition keys and every control that shows the pipeline can REJECT are one
# unit. The U-1 trap this tier is built around — path B silently becoming path A — is defeated
# by provenance travelling with the board through the same functions the gate calls; split the
# harness from the gate and the repair that removes the second producer stops being visible.
"""T3 — the model input is a function of the POSITION, not of the board's mutation history.

TWO ENGINE PATHS. (A) a fresh `Board.with_encoding_name(enc)` with the move sequence applied
directly. (B) the same position handed back by `MCTSTree.select_leaves` after an apply/undo
descent — `Board::clone` per leaf, so the `FxHashMap` table layout and therefore
`get_clusters`' iteration order travel with the clone. That is leaf #1 vs leaf #7 of a batch,
which is CNN-5's own mechanism. (C) a cross-check: the same position reached by permuting the
two placements WITHIN one compound turn, which preserves ply, side to move and
`moves_remaining` (LAW-03's turn-vs-ply discipline is why the permutation is turn-internal).

`Board.to_tensor()` IS NEVER CALLED ON A PATH-B BOARD, and the reason is structural rather than
stylistic: `select_leaves` builds its boards with `PyBoard::from_inner`, which sets
`encoding: None` (`crates/mantis-bridge/src/board.rs:452-453`), and `to_tensor` PANICS on an
encoding-less board (`:272-282`). What survives is that `Board::clone` copies the GEOMETRY
(`core.rs:624-660`: `legal_move_radius`, `cluster_threshold`, `cluster_window_size`), and every
accessor this tier needs reads `self.inner`, so a leaf board carries the ROOT's resolved
geometry. `GameState.to_tensor()` (`src/mantis/env/game_state.py:169`) is pure Python over the
views and centres the board handed out, and it is the ONLY reachable surface for path B.

THE TRAP, MECHANISED RATHER THAN WRITTEN DOWN. The obvious repair when `to_tensor` panics is
"rebuild a fresh encoded board and replay the leaf's stones into it" — which silently converts
path B into path A, leaves the tier green forever, and looks in the diff exactly like a bug fix.
Every compared board therefore carries its PROVENANCE, a path-B board must have come from
`select_leaves`, and the harness self-test below feeds the comparison a reconstructed board and
requires the named refusal.

WHAT A GREEN MEANS. The two paths agree on this corpus, the pipeline that would detect a
disagreement HAS been shown to reject a real one, and the descent that produces path B HAS been
shown to run. It does NOT mean the two paths CAN differ: whether `FxHashMap::clone` preserves
bucket layout is unmeasured, and if the layout always travels the REPORT's order-instability
count is 0 — that zero is itself the finding, not a pass.

THE ORDER HALF IS SCOPED TO SINGLE-CENTRE POSITIONS, and that scoping is load-bearing. Centre
order follows cluster enumeration order, which seeds from `self.cells.keys()`
(`crates/mantis-core/src/board/moves.rs:660`) with no sort anywhere on the path — the repo says
so in its own oracle (`inv18b_cluster_center_negative_bbox.rs:36-39`). So on a MULTI-centre
position centre order is path-dependent BY THE VERY MECHANISM CNN-5 names: asserting it green
would assert the defect fixed. The multi-centre gate therefore asserts the centre MULTISET and
the planes KEYED BY CENTRE, and the order observation lives in the unthresholded REPORT half.

SCOPE RESIDUE, STATED. This tier compares the PYTHON `GameState` path. The Rust self-play
loop's dense model input goes through the encode kernel (`mantis_encoding::to_planes`,
`board.rs:281`), which is unreachable on a path-B board at all. This tier's green does not
claim the Rust encoder and its name does not imply it.

COMPACTNESS IS ESTABLISHED BY CONSTRUCTION AND CERTIFIED FROM ENGINE-READ NUMBERS. A cluster is
compact when its span is at most `S - offset` (`cluster.rs:43`). `S`, `R` and the observed span
come off the CONSTRUCTED BOARD (`cluster_window_size`, `legal_move_radius`, `get_stones`); the
`offset` has no `PyBoard` getter at all, so it is read off `cluster.rs`'s own definition line by
`span_threshold_offset` below — the one term of this bound that is not on the Python surface,
named here rather than typed as a `4`. The tier certifies `stone_span(target) <= S - offset` —
the target's own bbox, read off `get_stones()` — rather than re-deriving cluster membership in
Python, which would be a second authority over the clustering rule. It formerly certified the
a-priori bound `s + k*R` instead; that bound IMPLIES this one and is strictly looser, and at
radius 8 it certifies nothing at all while real targets sit well inside the threshold. See
`certify_compact`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mantis._engine import Board, MCTSTree
from mantis.env.game_state import GameState

from _corpus import ConformanceRefusal, build_board, require_corpus_member, roster

#: The engine source that DEFINES the small-cluster span threshold. There is no `PyBoard`
#: getter for it — `crates/mantis-bridge/src/board.rs` exposes `legal_move_radius`,
#: `cluster_threshold`, `cluster_window_size` and `to_flat` and nothing else relevant — so the
#: offset in `S - offset` cannot be asked of a board. It is READ OFF THE ENGINE'S OWN SOURCE
#: instead of typed here: a typed `4` is a second authority over a rule this tier certifies
#: against, and it would go on certifying the wrong rule silently if `cluster.rs` changed.
SPAN_THRESHOLD_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "crates" / "mantis-core" / "src" / "board" / "state" / "cluster.rs"
)
_SPAN_THRESHOLD_DEFINITION = re.compile(
    r"let\s+span_threshold\s*:\s*i32\s*=\s*window_size\s+as\s+i32\s*-\s*(\d+)\s*;"
)

PATH_A = "A:direct-construction"
PATH_B = "B:select_leaves-descent"
PATH_C = "C:turn-internal-permutation"
PARTITION_SINGLE = "single_centre_compact"
PARTITION_MULTI = "multi_centre_compact"
PARTITION_SPREAD = "spread_or_massive"
PARTITIONS = (PARTITION_SINGLE, PARTITION_MULTI, PARTITION_SPREAD)


class ConstructionPathDivergence(ConformanceRefusal):
    """Two construction paths produced different model inputs for one position."""


class ReconstructedPathBBoard(ConformanceRefusal):
    """A board claiming path B was rebuilt rather than descended — path B collapsed into A."""


class DescentDidNotRun(ConformanceRefusal):
    """The tree handed back the root unexpanded, so path B is a clone of path A."""


class NoMatchingLeaf(ConformanceRefusal):
    """No returned leaf equals the target, so the comparison loop would run zero times."""


class EmptyPartition(ConformanceRefusal):
    """A partition the tier quantifies over is empty."""


class CompactnessNotCertified(ConformanceRefusal):
    """The engine-read bound does not certify this target's clusters as compact."""


class SpanThresholdUnreadable(ConformanceRefusal):
    """The span-threshold definition could not be read off `cluster.rs` in its expected form."""


def span_threshold_offset(source: Path = SPAN_THRESHOLD_SOURCE) -> int:
    """The `offset` of the engine's `span_threshold = S - offset`, read off `cluster.rs`.

    A cluster is small when its span is at most `window_size - offset` (`cluster.rs:43`). The
    window side `S` is on the Python surface (`Board.cluster_window_size`); the offset is NOT,
    and transcribing it as a literal `4` would make this tier a second authority over the rule
    it certifies against — the certificate would survive a change to `cluster.rs` and go on
    admitting or refusing the wrong targets in silence. Reading the definition instead means a
    changed VALUE moves this tier with the engine, and a changed FORM fails by name rather than
    falling back to a default.
    """
    found = _SPAN_THRESHOLD_DEFINITION.findall(source.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise SpanThresholdUnreadable(
            f"{source} carries {len(found)} definitions matching "
            f"{_SPAN_THRESHOLD_DEFINITION.pattern!r}; exactly one is expected. The small-cluster "
            "span threshold has moved, been renamed or been duplicated, and every compactness "
            "certificate in this tier is a certificate over the old rule until this is read again."
        )
    return int(found[0])


@dataclass(frozen=True)
class PathBoard:
    """A board plus WHICH PATH produced it and HOW. Provenance travels with the board.

    `descended_from` is the leaf list `select_leaves` actually returned, and a path-B board
    must be one of those OBJECTS — not a board that merely compares equal to one. A label
    alone would be forgeable by the very repair this tier exists to refuse: rebuilding the
    position and calling the result path B. Object identity is not.
    """

    board: Board
    path: str
    provenance: str
    descended_from: tuple[Board, ...] = ()


def require_path_provenance(item: PathBoard) -> None:
    if item.path != PATH_B:
        return
    if item.provenance != "select_leaves":
        raise ReconstructedPathBBoard(
            f"a board declared path B carries provenance {item.provenance!r}. A path-B board "
            "must have come from MCTSTree.select_leaves and must not have been re-inserted, "
            "replayed or reconstructed — a reconstruction turns this tier into a comparison of "
            "a position with itself, green forever, measuring nothing."
        )
    if not any(item.board is leaf for leaf in item.descended_from):
        raise ReconstructedPathBBoard(
            "a board declared path B is not one of the objects select_leaves returned. It was "
            "rebuilt or replayed, which silently converts path B into path A while looking, in "
            "the diff, exactly like a bug fix."
        )


def stone_span(board: Board) -> int:
    stones = board.get_stones()
    qs = [q for q, _, _ in stones]
    rs = [r for _, r, _ in stones]
    return max(max(qs) - min(qs), max(rs) - min(rs))


def certify_compact(root: Board, added_plies: int, target: Board) -> None:
    """`stone_span(target) <= S - offset`, MEASURED off the target, with the a-priori bound
    `s + k*R` carried in the refusal beside it.

    Nothing here is a typed span threshold, and the offset in particular is not: it has no
    `PyBoard` getter, so it is read out of the engine's own source by `span_threshold_offset`
    rather than transcribed.

    MEASURED RATHER THAN BOUNDED, AND THE MEASURED FORM IS STRICTLY STRONGER. The predecessor
    certified `s + k*R <= S - offset`, an a-priori bound taken before the target existed. That
    bound IMPLIES this check — the target's stones are the root's plus `k` stones each within
    `R` of an existing one, so the target's bbox span is at most `s + k*R` — and is strictly
    looser. The looseness stopped being free the day a radius-8 encoding was registered
    (R328(b)): at `R = 8`, `s + 2R >= 16 > 15` for ANY root, so NO 2-ply descent target could be
    certified and the compact partition went empty for `gnn_axis_r8` — while its real targets
    measure a span of 13 against a threshold of 15. `added_plies` is not the knob that would
    have fixed it: `2` is the compound-turn structure the corpus is built on
    (`leaf.ply == root.ply + 2`), not a tuning constant.

    THIS IS NOT A LOOSENED CRITERION. The criterion is unchanged — the target's clusters lie
    inside the engine's small-cluster span threshold — and the EVIDENCE for it got stronger: a
    bbox over `get_stones()` is an engine read, and it over-approximates any single cluster's
    span, so it is still not the Python re-derivation of cluster membership this tier refuses to
    become a second authority for.
    """
    side = target.cluster_window_size()
    radius = target.legal_move_radius()
    threshold = side - span_threshold_offset()
    measured = stone_span(target)
    a_priori = stone_span(root) + added_plies * radius
    if measured > threshold:
        raise CompactnessNotCertified(
            f"target span {measured} exceeds the engine's small-cluster span threshold "
            f"{threshold} (S={side}); the a-priori bound was root span {stone_span(root)} + "
            f"{added_plies} plies x radius {radius} = {a_priori}. This target is not certified "
            "compact and may not enter a compact partition."
        )


def centres_of(item: PathBoard) -> list[tuple[int, int]]:
    require_path_provenance(item)
    return [(int(q), int(r)) for q, r in item.board.get_cluster_views()[1]]


def planes_of(item: PathBoard) -> list[np.ndarray]:
    require_path_provenance(item)
    return [np.asarray(v) for v in item.board.get_cluster_views()[0]]


def game_state_tensor(item: PathBoard) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The Python model input. `Board.to_tensor()` is deliberately NOT used — see the docstring."""
    require_path_provenance(item)
    tensor, centres = GameState.from_board(item.board).to_tensor()
    return np.asarray(tensor), [(int(q), int(r)) for q, r in centres]


# --------------------------------------------------------------------------------------- #
# The three comparators. Their difference is DEMONSTRATED below, never asserted.
# --------------------------------------------------------------------------------------- #
def ordered_centres_agree(left: list, right: list) -> bool:
    return list(left) == list(right)


def centre_multiset_agrees(left: list, right: list) -> bool:
    return sorted(left) == sorted(right)


def planes_by_centre_agree(
    left_c: list, left_p: list, right_c: list, right_p: list
) -> bool:
    if sorted(left_c) != sorted(right_c):
        return False
    lut = {c: p.tobytes() for c, p in zip(right_c, right_p, strict=True)}
    return all(p.tobytes() == lut[c] for c, p in zip(left_c, left_p, strict=True))


def compare_paths(a: PathBoard, b: PathBoard, partition: str) -> None:
    """The gate's comparison, used unchanged by the end-to-end differential (R-O1)."""
    ac, bc = centres_of(a), centres_of(b)
    ap, bp = planes_of(a), planes_of(b)
    if partition == PARTITION_SINGLE:
        if not ordered_centres_agree(ac, bc):
            raise ConstructionPathDivergence(
                f"centre lists differ between {a.path} ({ac}) and {b.path} ({bc})"
            )
        if len(ap) != len(bp) or any(x.tobytes() != y.tobytes() for x, y in zip(ap, bp, strict=False)):
            raise ConstructionPathDivergence(
                f"cluster planes differ between {a.path} and {b.path} at centres {ac}"
            )
        at, ac2 = game_state_tensor(a)
        bt, bc2 = game_state_tensor(b)
        if ac2 != bc2 or at.tobytes() != bt.tobytes():
            raise ConstructionPathDivergence(
                f"GameState model inputs differ between {a.path} and {b.path}"
            )
    elif partition == PARTITION_MULTI:
        if not centre_multiset_agrees(ac, bc):
            raise ConstructionPathDivergence(
                f"centre multisets differ between {a.path} ({sorted(ac)}) and "
                f"{b.path} ({sorted(bc)})"
            )
        if not planes_by_centre_agree(ac, ap, bc, bp):
            raise ConstructionPathDivergence(
                f"centre-keyed planes differ between {a.path} and {b.path}"
            )
    else:
        raise ConformanceRefusal(f"the GATE does not run on partition {partition}")


# --------------------------------------------------------------------------------------- #
# Corpus: one compact root and one spread root per encoding, plus a 2-ply descent
# --------------------------------------------------------------------------------------- #
def compact_root_moves() -> list[tuple[int, int]]:
    return [(0, 0), (1, 0), (0, 1), (1, 1)]


def spread_root_moves(enc: str) -> list[tuple[int, int]]:
    """Stones strung out at the engine's own connectivity threshold until the board bbox
    exceeds its own small-cluster span threshold. Every number is read off a board."""
    probe = Board.with_encoding_name(enc)
    step = min(probe.cluster_threshold(), probe.legal_move_radius())
    span_threshold = probe.cluster_window_size() - span_threshold_offset()
    count = span_threshold // step + 2
    # An EVEN placement count leaves the root one placement short of completing a compound
    # turn, which is the structure the path-A/path-C reconstruction below relies on.
    count += count % 2
    return [(i * step, 0) for i in range(count)]


@dataclass(frozen=True)
class Target:
    enc: str
    root_moves: list[tuple[int, int]]
    added: list[tuple[int, int]]
    partition: str
    root_ply: int


def descend(enc: str, root_moves: list[tuple[int, int]], sims: int, n_leaves: int):
    root = build_board(enc, root_moves)
    tree = MCTSTree()
    tree.new_game(root)
    tree.run_simulations_cpu_only(sims)
    leaves = tree.select_leaves(n_leaves)
    return root, tree, leaves


def require_descent_ran(tree: MCTSTree, root: Board, leaf: Board) -> None:
    """PB-26. An unexpanded root makes path B a clone of path A and the tier green for the
    wrong reason."""
    if tree.root_visits() <= 0:
        raise DescentDidNotRun(f"root_visits() = {tree.root_visits()}; the descent never ran")
    if leaf.ply <= root.ply:
        raise DescentDidNotRun(
            f"the returned leaf is at ply {leaf.ply}, not beyond the root's {root.ply}; "
            "path B did not descend"
        )


def added_moves_in_turn_order(root: Board, leaf: Board) -> list[tuple[int, int]]:
    """The descent's added placements, ordered by the colour the engine reports for each.

    The root sits one placement short of completing a turn (`moves_remaining == 1`, asserted by
    the corpus builder), so the first added placement carries the root's current player and the
    second carries the other. The ONLY ordering freedom is WITHIN one compound turn — which is
    exactly what path C exercises, by construction rather than by coincidence.
    """
    before = {(q, r) for q, r, _ in root.get_stones()}
    added = [(q, r, c) for q, r, c in leaf.get_stones() if (q, r) not in before]
    first = [(q, r) for q, r, c in added if c == root.current_player]
    second = [(q, r) for q, r, c in added if c != root.current_player]
    return first + second


def build_corpus(enc: str) -> list[tuple[Target, PathBoard, PathBoard, PathBoard]]:
    """Targets with their three paths. Path B boards keep `select_leaves` provenance."""
    out: list[tuple[Target, PathBoard, PathBoard, PathBoard]] = []
    for root_moves, intent in (
        (compact_root_moves(), "compact"),
        (spread_root_moves(enc), "spread"),
    ):
        root, tree, leaves = descend(enc, root_moves, sims=20, n_leaves=6)
        assert root.moves_remaining == 1, (
            "the compound-turn structure this corpus relies on has changed: the root must sit "
            "one placement short of completing a turn"
        )
        for leaf in leaves:
            if leaf.ply != root.ply + 2:
                continue
            require_descent_ran(tree, root, leaf)
            require_corpus_member(leaf, f"{enc} path-B leaf")
            added = added_moves_in_turn_order(root, leaf)
            if len(added) != 2:
                continue
            path_a = PathBoard(build_board(enc, root_moves + added), PATH_A, "direct")
            swapped = root_moves[:-1] + [added[0], root_moves[-1], added[1]]
            path_c = PathBoard(build_board(enc, swapped), PATH_C, "direct")
            path_b = PathBoard(leaf, PATH_B, "select_leaves", tuple(leaves))
            n_centres = len(centres_of(path_a))
            if intent == "compact":
                certify_compact(root, 2, path_a.board)
                partition = PARTITION_SINGLE if n_centres == 1 else PARTITION_MULTI
            else:
                partition = PARTITION_SPREAD
            out.append(
                (Target(enc, root_moves, added, partition, root.ply), path_a, path_b, path_c)
            )
    return out


def require_partitions_non_empty(counts: dict[str, int]) -> None:
    empty = sorted(p for p in PARTITIONS if counts.get(p, 0) <= 0)
    if empty:
        raise EmptyPartition(
            f"partitions {empty} are EMPTY across the corpus; a gate with nothing in its "
            "partition passes for the wrong reason."
        )


def require_comparisons_executed(comparisons: int, enc: str, counts: dict[str, int]) -> int:
    """The per-encoding gate counts COMPARISONS, not corpus members.

    The assertion this replaces was `sum(counts.values()) > 0`, which counts every corpus
    member INCLUDING the spread/massive ones the loop `continue`s past without comparing
    anything. An encoding whose corpus produced only spread targets therefore satisfied it
    with zero comparisons executed — the vacuous pass this tier exists to refuse, in the
    tier's own gate. `require_partitions_non_empty` covers the corpus ACROSS encodings; it
    says nothing about any one of them.
    """
    if comparisons <= 0:
        raise EmptyPartition(
            f"{enc}: the gate executed ZERO path comparisons over partition counts {counts}. "
            "Every member fell in the spread/massive partition, which the loop skips, so the "
            "three construction paths were never compared and pytest reports PASSED."
        )
    return comparisons


def require_target_matched(matches: int, target: Target) -> None:
    if matches != 1:
        raise NoMatchingLeaf(
            f"{target.enc}: {matches} leaves matched the target position instead of exactly 1. "
            "A zero-match target runs the comparison loop zero times and PASSES."
        )


def position_key(board: Board) -> tuple:
    return (tuple(sorted(board.get_stones())), board.current_player, board.moves_remaining)


# --------------------------------------------------------------------------------------- #
# The GATE
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", roster(), ids=lambda s: s.name)
def test_the_three_construction_paths_agree_on_the_compact_partitions(spec, derived):
    corpus = build_corpus(spec.name)
    assert corpus, f"{spec.name}: the corpus builder produced no target"
    counts: dict[str, int] = dict.fromkeys(PARTITIONS, 0)
    comparisons = 0
    for target, path_a, path_b, path_c in corpus:
        counts[target.partition] += 1
        if target.partition == PARTITION_SPREAD:
            continue
        compare_paths(path_a, path_b, target.partition)
        compare_paths(path_a, path_c, target.partition)
        comparisons += 2
    derived(f"t3.partition_counts.{spec.name}", counts)
    derived(
        f"t3.comparisons.{spec.name}",
        require_comparisons_executed(comparisons, spec.name, counts),
    )


def test_a_corpus_of_ONLY_SPREAD_targets_is_refused_by_the_per_encoding_gate():
    """The gate ran its loop zero times and passed. `sum(counts.values()) > 0` is satisfied by
    a corpus of nothing but spread targets — every one of which the loop skips — so the
    counter it asserted was the wrong counter, not a missing one."""
    spread_only = dict.fromkeys(PARTITIONS, 0)
    spread_only[PARTITION_SPREAD] = 7
    assert sum(spread_only.values()) > 0, "the assertion this replaces is SATISFIED here"
    with pytest.raises(EmptyPartition, match="ZERO path comparisons"):
        require_comparisons_executed(0, "stand-in", spread_only)


def test_the_comparison_counter_does_NOT_fire_when_comparisons_RAN():
    """Negative control: a counter that fires on a real count is not a counter."""
    counts = dict.fromkeys(PARTITIONS, 1)
    assert require_comparisons_executed(4, "stand-in", counts) == 4


def test_all_three_partitions_are_non_empty_across_the_corpus(derived):
    """PB-30. A corpus change that empties any partition fails here, by name."""
    counts: dict[str, int] = dict.fromkeys(PARTITIONS, 0)
    for spec in roster():
        for target, *_ in build_corpus(spec.name):
            counts[target.partition] += 1
    derived("t3.partition_counts.total", counts)
    require_partitions_non_empty(counts)


@pytest.mark.parametrize("spec", roster(), ids=lambda s: s.name)
def test_the_descent_that_produces_path_B_actually_RAN(spec, derived):
    """PB-26. An unexpanded root would make path B a clone of path A, and the tier green for
    exactly the reason it exists to refuse."""
    root, tree, leaves = descend(spec.name, compact_root_moves(), sims=20, n_leaves=6)
    derived(f"t3.root_visits.{spec.name}", tree.root_visits())
    deeper = [leaf for leaf in leaves if leaf.ply > root.ply]
    derived(f"t3.leaves_beyond_root.{spec.name}", len(deeper))
    assert deeper, "no returned leaf descended past the root"
    for leaf in deeper:
        require_descent_ran(tree, root, leaf)


@pytest.mark.parametrize("spec", roster(), ids=lambda s: s.name)
def test_every_target_is_matched_by_EXACTLY_ONE_leaf(spec, derived):
    """PB-25. The descent is policy-driven: if no returned leaf equals the target, a loop over
    matches runs zero times and the tier PASSES. The counter is asserted, not assumed."""
    root_a, _tree_a, leaves_a = descend(spec.name, compact_root_moves(), sims=20, n_leaves=6)
    _root_b, _tree_b, leaves_b = descend(spec.name, compact_root_moves(), sims=20, n_leaves=6)
    targets = [position_key(leaf) for leaf in leaves_a if leaf.ply == root_a.ply + 2]
    assert targets, "the descent produced no two-ply target"
    observed = [position_key(leaf) for leaf in leaves_b]
    matched = 0
    for key, target in zip(
        targets,
        [Target(spec.name, compact_root_moves(), [], PARTITION_SINGLE, root_a.ply)] * len(targets),
        strict=True,
    ):
        require_target_matched(observed.count(key), target)
        matched += 1
    derived(f"t3.targets_matched.{spec.name}", matched)
    assert matched > 0


# --------------------------------------------------------------------------------------- #
# The U-1 additions — whether this tier can REJECT at all
# --------------------------------------------------------------------------------------- #
def test_the_END_TO_END_differential_REDS_on_two_genuinely_different_positions(derived):
    """PB-24, in its strong form: BOTH SIDES REAL. A real `select_leaves` leaf at position P
    against a real path-A board at P' = P + one stone, through the SAME comparator, partition
    and provenance code the gate uses (R-O1) — no doctored array anywhere. Without this the
    tier's only demonstrated capability is that two comparator objects are distinguishable,
    which says nothing about the A-vs-B pipeline."""
    reds = 0
    for spec in roster():
        for target, path_a, path_b, _path_c in build_corpus(spec.name):
            if target.partition == PARTITION_SPREAD:
                continue
            extra = path_a.board.legal_moves()[0]
            shifted_board = build_board(
                spec.name, target.root_moves + target.added + [extra]
            )
            shifted = PathBoard(shifted_board, PATH_A, "direct")
            with pytest.raises(ConstructionPathDivergence) as caught:
                compare_paths(shifted, path_b, target.partition)
            assert PATH_A in str(caught.value) and PATH_B in str(caught.value), (
                "the failure message must name BOTH paths"
            )
            reds += 1
            break
    derived("t3.end_to_end_differential.reds", reds)
    assert reds > 0


def test_a_RECONSTRUCTED_path_B_board_is_REFUSED(derived):
    """PB-23, the single highest-value control here. The natural repair when `Board.to_tensor`
    panics on an encoding-less leaf is to replay the leaf's stones into a fresh encoded board —
    which silently converts path B into path A while looking, in the diff, like a bug fix."""
    spec = roster()[0]
    corpus = build_corpus(spec.name)
    target, path_a, path_b, _ = corpus[0]
    require_path_provenance(path_b)  # the genuine one passes
    rebuilt = PathBoard(
        build_board(spec.name, target.root_moves + target.added), PATH_B, "replayed-stones"
    )
    with pytest.raises(ReconstructedPathBBoard, match="replayed-stones"):
        compare_paths(path_a, rebuilt, target.partition)
    # And the harder half: the SAME reconstruction wearing the honest label. A string is
    # forgeable by exactly the repair this refuses; the identity of the object select_leaves
    # returned is not.
    disguised = PathBoard(
        build_board(spec.name, target.root_moves + target.added),
        PATH_B,
        "select_leaves",
        path_b.descended_from,
    )
    with pytest.raises(ReconstructedPathBBoard, match="not one of the objects"):
        compare_paths(path_a, disguised, target.partition)
    derived("t3.provenance_refusal", "fired-on-both-forms")


def test_a_tree_with_ZERO_simulations_is_refused():
    """PB-26's break: the descent-ran assertion must reject a stand-in that never descended."""
    spec = roster()[0]
    root = build_board(spec.name, compact_root_moves())
    tree = MCTSTree()
    tree.new_game(root)
    assert tree.root_visits() == 0
    with pytest.raises(DescentDidNotRun, match="never ran"):
        require_descent_ran(tree, root, root)


def test_an_EMPTY_partition_is_refused():
    counts = dict.fromkeys(PARTITIONS, 1)
    counts[PARTITION_MULTI] = 0
    with pytest.raises(EmptyPartition, match=PARTITION_MULTI):
        require_partitions_non_empty(counts)


def test_a_target_matched_by_ZERO_leaves_is_refused():
    with pytest.raises(NoMatchingLeaf, match="0 leaves"):
        require_target_matched(0, Target("stand-in", [], [], PARTITION_SINGLE, 0))


def test_an_UNCERTIFIED_compactness_bound_is_refused():
    """The compact partitions may not admit a target the engine-read bound does not certify."""
    spec = roster()[0]
    root = build_board(spec.name, spread_root_moves(spec.name))
    with pytest.raises(CompactnessNotCertified, match="span threshold"):
        certify_compact(root, 2, root)


def test_the_certification_reads_the_TARGET_and_not_the_ROOT():
    """PB-2. A compact ROOT may not certify a SPREAD TARGET, and nothing pinned that.

    The row above passes the same board as root and target, so it cannot tell the two apart:
    a `certify_compact` that measured the ROOT's span would satisfy it, and would then wave
    through every target whose own span exceeded the threshold — the precise failure the
    certification exists to prevent. Measured, not argued: replacing `stone_span(target)` with
    `stone_span(root)` leaves the whole tier green without this row and reds it with it.
    """
    spec = roster()[0]
    compact = build_board(spec.name, compact_root_moves())
    spread = build_board(spec.name, spread_root_moves(spec.name))
    assert stone_span(compact) < stone_span(spread), (
        "this row needs a compact root and a genuinely wider target; the two corpus "
        "constructors no longer differ in span and the control is vacuous"
    )
    with pytest.raises(CompactnessNotCertified, match="span threshold"):
        certify_compact(compact, 2, spread)


def test_the_span_threshold_OFFSET_is_read_off_the_engine_source_and_not_typed_here(derived):
    """The one term of the compactness bound with no `PyBoard` getter, and the controls that
    make reading it a mechanism rather than a decorated literal.

    A reader that silently returns a default when the engine source changes shape is worse than
    the literal it replaced, because the certificate would then be over a rule nobody has read.
    So the missing and the duplicated forms both FAIL by name, driven through the same function
    the gate calls.
    """
    offset = span_threshold_offset()
    derived("t3.span_threshold_offset", offset)
    derived("t3.span_threshold_source", SPAN_THRESHOLD_SOURCE.name)
    assert offset > 0

    probe = Board.with_encoding_name(roster()[0].name)
    assert probe.cluster_window_size() - offset > 0, (
        "the read offset leaves no positive span threshold on a constructed board"
    )


def test_an_UNREADABLE_span_threshold_definition_is_refused(tmp_path):
    """Both failure shapes of the reader: no definition, and more than one."""
    absent = tmp_path / "no_definition.rs"
    absent.write_text("let half: i32 = (window_size as i32 - 1) / 2;\n", encoding="utf-8")
    with pytest.raises(SpanThresholdUnreadable, match="0 definitions"):
        span_threshold_offset(absent)

    # DELIBERATELY not the engine's own offset: a reader that returned a constant instead of
    # reading the source would pass a fixture that agreed with the engine, and fail here.
    stand_in = 7
    line = f"    let span_threshold: i32 = window_size as i32 - {stand_in};\n"
    duplicated = tmp_path / "two_definitions.rs"
    duplicated.write_text(line * 2, encoding="utf-8")
    with pytest.raises(SpanThresholdUnreadable, match="2 definitions"):
        span_threshold_offset(duplicated)

    single = tmp_path / "one_definition.rs"
    single.write_text(line, encoding="utf-8")
    assert span_threshold_offset(single) == stand_in
    assert span_threshold_offset(single) != span_threshold_offset(), (
        "the stand-in source and the engine source read the same offset, so this control "
        "cannot distinguish a reader from a constant"
    )


# --------------------------------------------------------------------------------------- #
# The three comparator producers — their difference is DEMONSTRATED, not asserted
# --------------------------------------------------------------------------------------- #
def test_the_ORDERED_comparator_rejects_a_permutation():
    """PB-27."""
    left = [(0, 0), (5, 5)]
    assert not ordered_centres_agree(left, list(reversed(left)))
    assert ordered_centres_agree(left, list(left))


def test_the_MULTISET_comparator_ACCEPTS_the_same_permutation():
    """PB-28, a NEGATIVE control: if the multiset comparator rejected it, the two comparators
    would be indistinguishable and the single-centre scoping would be decorative."""
    left = [(0, 0), (5, 5)]
    assert centre_multiset_agrees(left, list(reversed(left)))


def test_the_MULTISET_comparator_REJECTS_a_one_cell_shift():
    """PB-29. Order-insensitive must not mean value-insensitive."""
    left = [(0, 0), (5, 5)]
    assert not centre_multiset_agrees(left, [(0, 0), (5, 6)])


def test_the_module_never_calls_to_tensor_on_anything_but_a_GameState():
    """Asserted STRUCTURALLY over this module's own AST, because `PyBoard.encoding is None` is
    not observable from Python and the panic only fires once the wrong surface is reached — by
    which time a reviewer reads the crash as a bug rather than as the trap it is. A text search
    for the call would match this assertion itself, which is why the check is an `ast` walk."""
    import ast as _ast
    from pathlib import Path as _Path

    tree = _ast.parse(_Path(__file__).read_text(encoding="utf-8"))
    receivers = [
        node.func.value
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Attribute)
        and node.func.attr == "to_tensor"
    ]
    assert receivers, "no to_tensor call found — this control has lost its subject"
    for receiver in receivers:
        assert (
            isinstance(receiver, _ast.Call)
            and isinstance(receiver.func, _ast.Attribute)
            and receiver.func.attr == "from_board"
        ), _ast.dump(receiver)


# --------------------------------------------------------------------------------------- #
# REPORT (`slow`) — the CNN-5 measurement, unthresholded
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_report_construction_path_disagreement_over_the_spread_partition(derived):
    """Over the spread/massive partition, the count and identity of positions whose paths
    disagree, plus the centre-ORDER instability count on the multi-centre compact partition.
    NO THRESHOLD, NO ASSERTION OF THE COUNT: order instability on a compact multi-centre
    position IS the CNN-5 defect, so the tier records it rather than asserting it away — and a
    zero is itself the finding, not a pass."""
    disagreements: list[str] = []
    order_unstable: list[str] = []
    examined = 0
    for spec in roster():
        for target, path_a, path_b, path_c in build_corpus(spec.name):
            examined += 1
            if target.partition == PARTITION_SPREAD:
                for other, label in ((path_b, PATH_B), (path_c, PATH_C)):
                    if not centre_multiset_agrees(centres_of(path_a), centres_of(other)):
                        disagreements.append(f"{spec.name}:{target.added} vs {label}")
            elif target.partition == PARTITION_MULTI:
                for other, label in ((path_b, PATH_B), (path_c, PATH_C)):
                    if not ordered_centres_agree(centres_of(path_a), centres_of(other)):
                        order_unstable.append(f"{spec.name}:{target.added} vs {label}")
    derived("t3.report.positions_examined", examined)
    derived("t3.report.spread_disagreements", disagreements)
    derived("t3.report.multi_centre_order_instability", order_unstable)
    assert examined > 0
