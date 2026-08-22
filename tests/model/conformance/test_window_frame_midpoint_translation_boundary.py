# >300 justify (R8): the two gate halves, their controls and their planted breaks are ONE unit
# because a control that lives apart from the check it controls can be deleted without the
# check going red — which is precisely the failure mode this tier exists to refuse. The census
# lexer sits beside the pin it produces for the same reason: a matcher and the set it certifies
# must move together or the certificate outlives the instrument.
"""T1 — the window-frame translation-equivariance CHARACTERIZATION, on all three frames.

WHAT A GREEN MEANS, AND IT IS NOT "THE FRAME IS TRANSLATION-EQUIVARIANT". On a constructed
third of these cases the frame is asserted to be NOT equivariant: the engine computes
`(min + max) / 2` with i32 truncate-toward-zero, deliberately, for checkpoint anchor
calibration and cross-arm byte-parity (`crates/mantis-core/src/board/state/cluster.rs:30-33`,
`core.rs:369-376`, `crates/mantis-graph/src/lib.rs:230-238`). A green means "the truncating
midpoint rule is unchanged on all three frames, and the set of midpoint constructions under
`crates/*/src` is unchanged" — that, and not one word more. A FIX to the truncation REDS this
tier, which is correct and is why the tier is named `_boundary` rather than `_equivariance`.

TWO HALVES COVERING DISJOINT FAILURE MODES, AND NEITHER SUFFICES ALONE.
  * The CROSS-ARM half: `mantis-core` and `mantis-graph` compute the origin independently
    (`crates/mantis-graph/Cargo.toml` has an empty `[dependencies]` table by contract, so it
    cannot call `mantis-core`), and this tier makes them meet. It catches one arm drifting.
  * The DERIVED SIGNED PREDICATE: one workspace-wide `cargo clippy --fix` for `manual_midpoint`
    rewrites every site in both crates CONSISTENTLY and sails through the cross-arm half. That
    is not hypothetical — it is the incident this repo's own oracles were written about
    (`crates/mantis-core/tests/inv18_window_center_negative_bbox.rs:4`,
    `inv18b_cluster_center_negative_bbox.rs:9`). `floor` gives a flat `t`; the truncating rule
    gives `t + odd(a)·([a+2t<0] − [a<0])`, and only the predicate can tell them apart.

NOT THIS TIER'S SUBJECT: the single-site truncation VALUE rule and the negative-odd fixed
cases, which are owned by `inv18_window_center_negative_bbox.rs` and
`inv18b_cluster_center_negative_bbox.rs`. This tier does not restate them (R304(a)).

NO TOLERANCE APPEARS ANYWHERE IN THIS TIER, by construction: every assertion is exact-integer,
and a tolerance here would be an armed value.

CENSUS SCOPE AND CASE POSTURE (R297(b)), stated because a census is only as honest as its
stated scope. Roots walked: `crates/*/src/**/*.rs`; how many files that is is a derived output
of the run (`t1.census.files_walked`), not a number written here. Case-SENSITIVE. Comments
(`//`, `///`, `//!`, nested `/* */`) and string/char literals are stripped by a lexer BEFORE
any matching, so a decoy in either is invisible; both decoys are planted as controls below.
The matcher is a token SHAPE — a balanced `(` group containing a `+`, divided by an integer
`2` — with NO name requirement of any kind (no `min`/`max`, no `midpoint`, no `window`) and NO
OPERAND GRAMMAR AT ALL. Both omissions are load-bearing and both were earned: a name
requirement is a blind spot because the dangerous case is a new origin under different names,
and an operand grammar is a blind spot because five spellings were measured walking past one —
`2i32` and `2_i32` (the divisor lexes as a single token), `(*a + *b) / 2`,
`((min_q + max_q) as i32) / 2` and `(v[0] + v[1]) / 2`, the last being an ordinary refactor
for a bbox held as `[i32; 2]` in a crate that already works in coordinate arrays. What the
matcher renders for the pin is therefore normalised token text, since it can render an operand
shape it cannot parse. `crates/*/tests/**` and
`crates/*/benches/**` are excluded by design (they hold the inv18 pins, which are assertions
ABOUT the rule, not authorities OVER it) and the exclusion is inert: the same census over those
roots returns zero constructions, asserted below rather than claimed.

THE COUNTING UNIT IS THE CONSTRUCTION, and stating it is not pedantry: the same census has four
defensible cardinalities — constructions, distinct `(file, line)` pairs, window-origin sites,
and `#[allow]` markers — and they are four different numbers. Which unit this tier counts in is
fixed here; WHAT it counts to is a derived output of the run (`t1.census.constructions`,
`t1.census.distinct_lines`) and is deliberately absent from this sentence. A transcribed tally
inside the very argument that a count needs its unit fixed is that same defect one level up: it
must be re-edited on every edit, will eventually be wrong, and is then read as evidence
(R192(e), SF-7, derive-or-delete). It already had been — this paragraph carried a line tally the
instrument itself contradicted.

THE `#[allow]` MARKER IS NOT THE CENSUS KEY, and the backstop that was supposed to excuse that
does not exist. A marker-keyed census sees only sites that already DECLARE their own
deliberateness — it is blind to exactly the dangerous case. The standing answer was to lean on
clippy; measured on the pinned toolchain, `clippy::manual-midpoint` is `allow` by default and
sits in `clippy::pedantic`, while CI gate 2 denies only `clippy::all`. An unmarked new origin
produces a warning nobody denies. So the census catches the unmarked case ITSELF, and the
unmarked planted break below is the load-bearing one.

WHAT THE CENSUS DOES NOT CLAIM. It censuses midpoint CONSTRUCTIONS, not window ORIGINS: it
cannot tell that `core.rs:381` is an origin and an unrelated average is not. That judgement is
the pinned table's `frame` column, which is prose and is NOT what the assertion compares — the
assertion compares triples. RESIDUE, named and unguarded, and MEASURED rather than imagined —
every form named here was planted against the real crates tree and confirmed to walk past: an
origin spelled `>> 1`, via `i32::midpoint`, through a `const`, with the sum more than one
bracket deep, or re-implemented in Python, is outside this census, and there is no second
instrument that would catch it. Three forms this paragraph USED to name are now inside, each
because it was measured rather than imagined: the zero-argument accessor
`(self.lo_q() + self.hi_q()) / 2`, which is what a field-to-getter refactor produces; the
argument-bearing helper call, which the operand grammar excluded and the token shape does not;
and the five spellings above. A residue that shrinks when it is measured is the only evidence
that the paragraph was ever a scope statement rather than a hedge.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from _corpus import (
    CLASS_EVEN,
    FRAME_DENSE_BOARD,
    FRAME_DENSE_CLUSTER,
    FRAME_GRAPH,
    SIGN_CLASSES,
    UNIT_AXIAL,
    ConformanceRefusal,
    DegenerateCorpusMember,
    bbox_sums,
    board_frame_centre,
    build_board,
    cluster_frame_centre,
    graph_frame_centre,
    graph_wire_for,
    require_corpus_member,
    roster,
    sign_class,
    signed_delta,
    translate,
)

CRATES = Path(__file__).resolve().parents[3] / "crates"


class WindowFrameDisagreement(ConformanceRefusal):
    """A frame's reported centre disagrees with the computed signed correction."""


class CrossArmDisagreement(ConformanceRefusal):
    """The dense board frame and the graph frame report different origins for one position."""


class FrameMatrixMismatch(ConformanceRefusal):
    """The executed `(encoding, frame)` matrix differs from the one derived at run time."""


class EmptySignClass(ConformanceRefusal):
    """A sign-class ran its loop body zero times; pytest would report that case as PASSED."""


class SecondWindowOriginAuthority(ConformanceRefusal):
    """The census returned a set of midpoint constructions different from the pinned one."""


# --------------------------------------------------------------------------------------- #
# GATE half 1 — the cross-arm translation-delta characterisation
# --------------------------------------------------------------------------------------- #
#: Base geometries, chosen so that every sign-class is REALISED under unit axial translation.
#: A crossing needs `a = min+max` odd AND the translation to carry `a` across zero, which with
#: unit vectors means `|a| = 1` on the moved axis — hence one shape per axis per sign.
SHAPES: dict[str, list[tuple[int, int]]] = {
    "aq_odd_positive": [(0, 0), (1, 0)],
    "aq_odd_negative": [(0, 0), (-1, 0)],
    "ar_odd_positive": [(0, 0), (0, 1)],
    "ar_odd_negative": [(0, 0), (0, -1)],
    "both_axes_even": [(0, 0), (2, 0), (0, 2)],
}


def frames_for(spec) -> tuple[str, ...]:
    """The frames that EXIST for an encoding, derived from the spec, never from a name list.

    `HexgBuffer::new` refuses a grid encoding by construction
    (`crates/mantis-selfplay/src/replay/hexg/mod.rs:246-252`), so the graph frame exists for
    exactly the encodings whose spec says `is_graph`. Read as a branch, a branch that silently
    reduces three frames to two is invisible; the derived matrix below is what makes it neither
    invisible nor red.
    """
    dense = (FRAME_DENSE_CLUSTER, FRAME_DENSE_BOARD)
    return dense + (FRAME_GRAPH,) if spec.is_graph else dense


def derived_frame_matrix(specs) -> frozenset[tuple[str, str]]:
    return frozenset((s.name, f) for s in specs for f in frames_for(s))


def read_frame(enc: str, frame: str, board) -> tuple[int, int]:
    """One frame's origin, read off the engine. Refuses a degenerate member at the point of use."""
    require_corpus_member(board, f"{enc}/{frame}")
    if frame == FRAME_DENSE_CLUSTER:
        return cluster_frame_centre(board)
    if frame == FRAME_DENSE_BOARD:
        return board_frame_centre(board)
    if frame == FRAME_GRAPH:
        return graph_frame_centre(enc, board)
    raise ConformanceRefusal(f"unknown frame {frame!r}")


def require_all_classes_seen(counters: dict[str, int], enc: str) -> None:
    empty = sorted(c for c in SIGN_CLASSES if counters.get(c, 0) <= 0)
    if empty:
        raise EmptySignClass(
            f"{enc}: sign-classes {empty} executed ZERO assertions. A class whose loop body "
            "never runs is reported by pytest as PASSED, which is the vacuous pass this tier "
            "exists to prevent."
        )


def require_matrix(executed: frozenset, expected: frozenset) -> None:
    if executed != expected:
        raise FrameMatrixMismatch(
            f"executed frame matrix {sorted(executed)} != derived {sorted(expected)}; "
            f"missing={sorted(expected - executed)} extra={sorted(executed - expected)}"
        )


def require_every_declared_frame_executed(specs, executed: frozenset) -> int:
    """The matrix's SECOND side, and the only one `frames_for` does not produce.

    `require_matrix` compares `derived_frame_matrix` against the pairs `run_frame_assertions`
    executed, and BOTH enumerate `frames_for`: a branch there that drops a frame shrinks the
    two sides together and the comparison stays green while the dropped frame's every
    assertion silently stops running. That is a comparison whose two sides come from one
    source — the failure class this suite exists to refuse, arriving inside it.

    This side does not call `frames_for`. It states the claim where it is asserted: the two
    dense frames exist for every registered encoding, and the graph frame exists exactly for
    the encodings whose spec says `is_graph` (`HexgBuffer::new` refuses a grid encoding by
    construction, `crates/mantis-selfplay/src/replay/hexg/mod.rs:246-252`).
    """
    missing: list[tuple[str, str]] = []
    for spec in specs:
        for frame in (FRAME_DENSE_CLUSTER, FRAME_DENSE_BOARD):
            if (spec.name, frame) not in executed:
                missing.append((spec.name, frame))
        if spec.is_graph and (spec.name, FRAME_GRAPH) not in executed:
            missing.append((spec.name, FRAME_GRAPH))
    if missing:
        raise FrameMatrixMismatch(
            f"frames declared by the specs but never executed: {sorted(missing)}. Every "
            "registered encoding carries both dense frames and every is_graph encoding carries "
            "the graph frame; a frame that no assertion reached is reported by pytest as PASSED."
        )
    return len(executed)


def require_cross_arm(enc: str, dense: tuple[int, int], graph: tuple[int, int]) -> None:
    if dense != graph:
        raise CrossArmDisagreement(
            f"{enc}: the dense board frame reports {dense} while the graph arm reports {graph}. "
            "Two crates that cannot call each other have drifted on one rule."
        )


def run_frame_assertions(spec) -> tuple[frozenset[tuple[str, str]], dict[str, int]]:
    """Every frame of one encoding against the COMPUTED signed correction, on q and r.

    Returns the executed `(encoding, frame)` pairs and the per-sign-class assertion counters,
    both as derived outputs. Pure: no module-level state, so it is shard-safe and the aggregate
    tests below can re-run it rather than depend on collection order.
    """
    enc = spec.name
    executed: set[tuple[str, str]] = set()
    counters: dict[str, int] = dict.fromkeys(SIGN_CLASSES, 0)
    for frame in frames_for(spec):
        for shape_name, moves in SHAPES.items():
            a_q, a_r = bbox_sums(moves)
            base_origin = read_frame(enc, frame, build_board(enc, moves))
            for t in UNIT_AXIAL:
                moved = read_frame(enc, frame, build_board(enc, translate(moves, t)))
                for axis, a, t_axis, base_c, moved_c in (
                    ("q", a_q, t[0], base_origin[0], moved[0]),
                    ("r", a_r, t[1], base_origin[1], moved[1]),
                ):
                    expected = signed_delta(a, t_axis)
                    if moved_c - base_c != expected:
                        raise WindowFrameDisagreement(
                            f"{enc}/{frame}/{shape_name} axis {axis}: bbox sum a={a}, "
                            f"translation t={t_axis}; the engine moved the origin by "
                            f"{moved_c - base_c}, the computed signed correction is {expected}"
                        )
                    counters[sign_class(a, t_axis)] += 1
                    executed.add((enc, frame))
    require_all_classes_seen(counters, enc)
    return frozenset(executed), counters


@pytest.mark.parametrize("spec", roster(), ids=lambda s: s.name)
def test_every_frame_moves_by_the_computed_signed_correction(spec, derived):
    executed, counters = run_frame_assertions(spec)
    derived(f"t1.frames.{spec.name}", sorted(f for _, f in executed))
    derived(f"t1.class_counters.{spec.name}", counters)
    assert min(counters.values()) > 0


def test_the_executed_frame_matrix_equals_the_matrix_derived_from_the_specs(derived):
    """PC-3/X3. The matrix is NOT `encodings × 3`: the graph frame exists for the encodings
    whose spec says so, and a branch that silently drops one is otherwise invisible."""
    specs = roster()
    expected = derived_frame_matrix(specs)
    executed: frozenset[tuple[str, str]] = frozenset()
    for spec in specs:
        pairs, _ = run_frame_assertions(spec)
        executed |= pairs
    derived("t1.frame_matrix", sorted(executed))
    derived("t1.frame_matrix.cardinality", len(executed))
    require_matrix(executed, expected)
    require_every_declared_frame_executed(specs, executed)
    assert len(executed) > 0


def test_a_DECLARED_frame_that_never_executes_is_refused():
    """PB-X3. The break `require_matrix` structurally cannot see, driven through the gate's own
    helper: with the graph pairs removed, the one-source comparison is still satisfiable while
    the declared-frame side names exactly what stopped running."""
    specs = roster()
    assert [s for s in specs if s.is_graph], "no graph encoding — this control has no subject"
    full = derived_frame_matrix(specs)
    assert require_every_declared_frame_executed(specs, full) == len(full)
    dropped = frozenset(pair for pair in full if pair[1] != FRAME_GRAPH)
    require_matrix(dropped, dropped)  # a `frames_for` that drops the frame shrinks both sides
    with pytest.raises(FrameMatrixMismatch, match="never executed"):
        require_every_declared_frame_executed(specs, dropped)


def test_the_dense_board_and_graph_arms_report_the_SAME_origin(derived):
    """The direct cross-arm equality. Asserting each frame against the shared predicate is NOT
    the same as asserting the frames against each other: if one arm's read silently stops
    executing, the survivors still agree with the predicate and the two-producer claim has
    quietly evaporated."""
    graph_specs = [s for s in roster() if s.is_graph]
    assert graph_specs, "no registered graph encoding — the cross-arm claim has no subject"
    checked = 0
    for spec in graph_specs:
        for moves in SHAPES.values():
            board = build_board(spec.name, moves)
            require_cross_arm(
                spec.name, board_frame_centre(board), graph_frame_centre(spec.name, board)
            )
            checked += 1
    derived("t1.cross_arm.assertions", checked)
    assert checked > 0


# --------------------------------------------------------------------------------------- #
# GATE half 1 — planted breaks
# --------------------------------------------------------------------------------------- #
def test_the_comparator_distinguishes_floor_from_truncation_in_BOTH_directions():
    """PB-3. A stand-in whose midpoint uses `floor` must be reported as DISAGREEING — and in
    both sign directions, since a comparator that only ever sees `+1` is the very defect this
    tier's own first draft carried, in comparator form."""
    def floor_rule_delta(a: int, t: int) -> int:
        return t

    to_negative = (1, -1)
    to_non_negative = (-1, 1)
    assert signed_delta(*to_negative) != floor_rule_delta(*to_negative)
    assert signed_delta(*to_non_negative) != floor_rule_delta(*to_non_negative)
    assert signed_delta(*to_negative) - floor_rule_delta(*to_negative) == 1
    assert signed_delta(*to_non_negative) - floor_rule_delta(*to_non_negative) == -1
    assert signed_delta(2, 1) == floor_rule_delta(2, 1)


def test_a_STONELESS_member_is_refused_rather_than_compared():
    """PB-2. `Board::window_center` returns a constant (0, 0) with no stones and
    `get_cluster_views` pushes (0, 0) with no clusters — both translation-invariant, both
    disagreeing with the predicate. The exclusion is deliberate and named, not incidental."""
    spec = roster()[0]
    from mantis._engine import Board

    with pytest.raises(DegenerateCorpusMember):
        read_frame(spec.name, FRAME_DENSE_CLUSTER, Board.with_encoding_name(spec.name))


def test_an_EMPTY_sign_class_is_refused():
    """PB-1. Driving one class's counter to zero must FAIL by name — not pass, not skip."""
    counters = dict.fromkeys(SIGN_CLASSES, 1)
    counters[CLASS_EVEN] = 0
    with pytest.raises(EmptySignClass, match=CLASS_EVEN):
        require_all_classes_seen(counters, "stand-in")


def test_a_MISSING_pair_in_the_frame_matrix_is_caught():
    """PB-5. Removing one `(encoding, frame)` pair must fail the set comparison."""
    expected = derived_frame_matrix(roster())
    shrunk = frozenset(sorted(expected)[1:])
    with pytest.raises(FrameMatrixMismatch, match="missing"):
        require_matrix(shrunk, expected)


def test_a_SINGLE_ARM_perturbation_is_caught_by_the_cross_arm_equality():
    """PB-4. One arm's origin shifted by +1 and the other's not: caught by THIS assertion, not
    merely by each arm's own agreement with the shared predicate."""
    graph_specs = [s for s in roster() if s.is_graph]
    assert graph_specs
    spec = graph_specs[0]
    board = build_board(spec.name, SHAPES["both_axes_even"])
    dense = board_frame_centre(board)
    graph = graph_frame_centre(spec.name, board)
    require_cross_arm(spec.name, dense, graph)
    with pytest.raises(CrossArmDisagreement):
        require_cross_arm(spec.name, (dense[0] + 1, dense[1]), graph)


# --------------------------------------------------------------------------------------- #
# GATE half 2 — the no-second-authority CENSUS
# --------------------------------------------------------------------------------------- #
_IDENT = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def lex_rust(src: str) -> list[tuple[str, int]]:
    """`(token, line)` pairs with comments and string/char literals STRIPPED.

    Lexing before matching is what defeats the failure the repo's own AST precedent names in
    its docstring (`tests/config/test_monitor_config_single_authority.py:25-28`): a text search
    hits comments and docstrings, and this repo has been bitten by both directions.
    """
    tokens: list[tuple[str, int]] = []
    i, n, line = 0, len(src), 1
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1
            i += 1
        elif ch.isspace():
            i += 1
        elif src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end < 0 else end
        elif src.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    line += src[i] == "\n"
                    i += 1
        elif ch == "r" and i + 1 < n and src[i + 1] in '"#':
            j, hashes = i + 1, 0
            while j < n and src[j] == "#":
                hashes, j = hashes + 1, j + 1
            if j < n and src[j] == '"':
                close = '"' + "#" * hashes
                end = src.find(close, j + 1)
                end = n if end < 0 else end
                line += src.count("\n", i, end)
                i = end + len(close)
                tokens.append(("<str>", line))
            else:
                j = i
                while j < n and src[j] in _IDENT:
                    j += 1
                tokens.append((src[i:j], line))
                i = j
        elif ch == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    line += 1
                j += 1
            i = j + 1
            tokens.append(("<str>", line))
        elif ch == "'":
            j = i + 1
            if j < n and src[j] in _IDENT and (j + 1 >= n or src[j + 1] != "'"):
                while j < n and src[j] in _IDENT:
                    j += 1
                tokens.append(("<lifetime>", line))
                i = j
            else:
                while j < n and src[j] != "'":
                    j += 2 if src[j] == "\\" else 1
                i = j + 1
                tokens.append(("<char>", line))
        elif ch in _IDENT:
            j = i
            while j < n and src[j] in _IDENT:
                j += 1
            tokens.append((src[i:j], line))
            i = j
        else:
            tokens.append((ch, line))
            i += 1
    return tokens


#: The divisor: an integer `2` with an optional Rust type suffix. `2i32` and `2_i32` lex as ONE
#: token, so a matcher comparing the token to the string "2" walks past both.
_DIVISOR_TWO = re.compile(r"^2(_?[iu](?:8|16|32|64|128|size))?$")
#: How deep inside the parenthesised group the `+` may sit. Depth 0 is `(a + b) / 2`; depth 1
#: admits the cast form `((min_q + max_q) as i32) / 2`, which is a second authority spelled
#: through a conversion and was measured to walk past a depth-0 matcher.
_MAX_SUM_DEPTH = 1


def _matching_open(tokens: list[tuple[str, int]], close: int) -> int | None:
    """The `(` matching the `)` at `close`, or `None` if the bracket that closes is not one."""
    depth = 0
    for i in range(close, -1, -1):
        token = tokens[i][0]
        if token in (")", "]"):
            depth += 1
        elif token in ("(", "["):
            depth -= 1
            if depth == 0:
                return i if token == "(" else None
    return None


def _has_sum(tokens: list[tuple[str, int]], open_idx: int, close_idx: int) -> bool:
    """A binary `+` inside the group, at most `_MAX_SUM_DEPTH` brackets deep."""
    depth = 0
    for i in range(open_idx + 1, close_idx):
        token = tokens[i][0]
        if token in ("(", "["):
            depth += 1
        elif token in (")", "]"):
            depth -= 1
        elif token == "+" and depth <= _MAX_SUM_DEPTH:
            return True
    return False


def _render(tokens: list[tuple[str, int]]) -> str:
    """The matched span as NORMALISED token text — one space between tokens, none around `.`.

    Rendering from the tokens rather than from a reassembled operand grammar is what lets the
    matcher accept any operand shape: `*a`, `v[0]`, `self.min_q()` and a cast all render without
    the renderer knowing what any of them are.
    """
    return (
        " ".join(token for token, _line in tokens).replace(" . ", ".").replace(" ( )", "()")
    )


def midpoint_constructions(root: Path, pattern: str = "*/src/**/*.rs") -> list[tuple[str, int, str]]:
    """Every `( … + … ) / 2` token SHAPE under `root`, as `(file, line, expression)`.

    THE OPERANDS ARE NOT PARSED, WHICH IS THE POINT. The matcher used to require each operand
    to be a dotted identifier run with an optional EMPTY argument list, so five spellings of a
    new origin walked past it — measured, one plant per row, against a temp crates tree:
    `2i32` and `2_i32` (the divisor lexes as one token), `(*a + *b) / 2`,
    `((min_q + max_q) as i32) / 2`, and `(v[0] + v[1]) / 2`. The last is the one that matters:
    a bbox held as `[i32; 2]` is an ordinary refactor and the graph crate already works in
    coordinate arrays. Requiring only a balanced group containing a `+`, divided by a 2-literal,
    removes the operand grammar as a place a new origin can hide.

    `>> 1` remains outside, with the rest of the residue in the module docstring: it is a
    different token shape, not a different operand shape.

    Root-parameterised on purpose: a break that cannot be constructed because the walk hard-codes
    its root is a failed obligation, and every temp-tree control below depends on this signature.
    """
    found: list[tuple[str, int, str]] = []
    for path in sorted(root.glob(pattern)):
        tokens = lex_rust(path.read_text(encoding="utf-8"))
        for i, (tok, _line) in enumerate(tokens):
            if tok != "/" or i + 1 >= len(tokens):
                continue
            if not _DIVISOR_TWO.match(tokens[i + 1][0]):
                continue
            if i == 0 or tokens[i - 1][0] != ")":
                continue
            opened = _matching_open(tokens, i - 1)
            if opened is None or not _has_sum(tokens, opened, i - 1):
                continue
            found.append(
                (
                    path.relative_to(root).as_posix(),
                    tokens[opened][1],
                    _render(tokens[opened : i + 2]),
                )
            )
    return sorted(found)


#: THE PIN. A source-level literal of `(file, line, expression)` triples, compared for
#: set-equality in BOTH directions. It is never a regenerated golden and never a bare count:
#: a pin the tier rewrites when it differs certifies its own breakage, and a count pins none of
#: the four ambiguous units this census could be measured in. The `frame` note beside each row
#: is prose for the reader; the assertion compares the triples only.
#:
#: THE EXPRESSION COLUMN IS NORMALISED TOKEN TEXT, which is what the matcher can render for an
#: operand shape it does not parse. The rows were re-pinned once, when the operand grammar was
#: removed; the `(file, line)` multiset of the census was IDENTICAL across that change except
#: for `core.rs:48` below, which the old matcher could not see at all — that is the check that
#: distinguishes a renderer change from a tree change, and it was made before re-pinning.
_THE_MIDPOINT_CONSTRUCTIONS: tuple[tuple[str, int, str], ...] = (
    # dense, cluster frame — small-cluster centroid branch
    ("mantis-core/src/board/state/cluster.rs", 71, "( min_q + max_q ) / 2"),
    ("mantis-core/src/board/state/cluster.rs", 71, "( min_r + max_r ) / 2"),
    # dense, cluster frame — massive-cluster no-anchor fallback
    ("mantis-core/src/board/state/cluster.rs", 92, "( min_q + max_q ) / 2"),
    ("mantis-core/src/board/state/cluster.rs", 92, "( min_r + max_r ) / 2"),
    # NOT A WINDOW ORIGIN — `hex_distance`'s axial halving. It is in the pin because the census
    # counts midpoint CONSTRUCTIONS and cannot tell an origin from an unrelated halving; that
    # judgement is this prose column, which the assertion does not read. It was invisible to
    # the operand grammar the matcher used to carry, so it arrived with the wider matcher.
    (
        "mantis-core/src/board/state/core.rs",
        48,
        "( ( q1 - q2 ).abs() + ( q1 + r1 - q2 - r2 ).abs() + ( r1 - r2 ).abs() ) / 2",
    ),
    # dense, BOARD frame — Board::window_center()
    ("mantis-core/src/board/state/core.rs", 381, "( self.min_q + self.max_q ) / 2"),
    ("mantis-core/src/board/state/core.rs", 382, "( self.min_r + self.max_r ) / 2"),
    # graph arm — mantis-graph's own fn window_center
    ("mantis-graph/src/lib.rs", 252, "( min_q + max_q ) / 2"),
    ("mantis-graph/src/lib.rs", 252, "( min_r + max_r ) / 2"),
)


def require_census(observed: list[tuple[str, int, str]], pinned: tuple) -> int:
    """Set-equality in BOTH directions, plus the empty refusal. Returns the cardinality."""
    if not observed:
        raise SecondWindowOriginAuthority(
            "the midpoint census returned EMPTY. Under a naive reading that says 'no second "
            "authority found', i.e. it reads as the property holding — the single most "
            "dangerous green in this tier. An empty census means the instrument stopped "
            "seeing, not that the tree stopped constructing."
        )
    got, want = set(observed), set(pinned)
    if got != want:
        raise SecondWindowOriginAuthority(
            f"midpoint constructions changed. NEW (a fifth origin?): {sorted(got - want)}; "
            f"GONE (a pinned site deleted?): {sorted(want - got)}"
        )
    return len(observed)


def test_the_census_returns_exactly_the_pinned_midpoint_constructions(derived):
    observed = midpoint_constructions(CRATES)
    cardinality = require_census(observed, _THE_MIDPOINT_CONSTRUCTIONS)
    derived("t1.census.constructions", cardinality)
    derived("t1.census.distinct_lines", len({(f, ln) for f, ln, _ in observed}))
    derived("t1.census.files_walked", len(list(CRATES.glob("*/src/**/*.rs"))))
    assert cardinality > 0


def test_the_EXCLUDED_tests_and_benches_roots_carry_no_construction(derived):
    """The exclusion's cost, measured rather than argued. `crates/*/tests/**` holds the inv18
    pins — assertions ABOUT the rule, not authorities OVER it — and excluding them is inert."""
    residue = midpoint_constructions(CRATES, "*/tests/**/*.rs")
    residue += midpoint_constructions(CRATES, "*/benches/**/*.rs")
    derived("t1.census.excluded_root_hits", len(residue))
    assert residue == [], f"the excluded roots are no longer inert: {residue}"


def test_an_EMPTY_census_is_REFUSED(tmp_path):
    """PB-6, the anti-self-pinning control. Without it an under-scoped census is
    indistinguishable from a clean one, and self-pinning then certifies the breakage."""
    (tmp_path / "somecrate" / "src").mkdir(parents=True)
    (tmp_path / "somecrate" / "src" / "lib.rs").write_text(
        "pub fn nothing() -> i32 { 7 }\n", encoding="utf-8"
    )
    assert midpoint_constructions(tmp_path) == []
    with pytest.raises(SecondWindowOriginAuthority, match="EMPTY"):
        require_census(midpoint_constructions(tmp_path), _THE_MIDPOINT_CONSTRUCTIONS)


def _plant(tmp_path: Path, body: str) -> Path:
    (tmp_path / "planted" / "src").mkdir(parents=True)
    target = tmp_path / "planted" / "src" / "lib.rs"
    target.write_text(body, encoding="utf-8")
    return target


def test_a_FIFTH_site_carrying_the_allow_marker_is_named(tmp_path):
    """PB-8. The declared variant."""
    _plant(
        tmp_path,
        "/// Truncate-toward-zero semantics preserves anchor calibration.\n"
        "#[allow(clippy::manual_midpoint)]\n"
        "pub fn extra_center(min_q: i32, max_q: i32) -> i32 {\n"
        "    (min_q + max_q) / 2\n"
        "}\n",
    )
    found = midpoint_constructions(tmp_path)
    assert found == [("planted/src/lib.rs", 4, "( min_q + max_q ) / 2")], found
    with pytest.raises(SecondWindowOriginAuthority, match="NEW"):
        require_census(found, _THE_MIDPOINT_CONSTRUCTIONS)


def test_a_FIFTH_site_with_NO_marker_and_DIFFERENT_names_is_named(tmp_path):
    """PB-9, the load-bearing control now that the clippy backstop is measured out of existence.

    The planted site carries no `#[allow]`, uses names the matcher has never heard of, is
    `self.`-prefixed — and sits beside a decoy in a comment and a decoy inside a string
    literal, so the control proves the LEXER and not merely the matcher.
    """
    _plant(
        tmp_path,
        "impl Thing {\n"
        "    // decoy in a comment: (self.lo_r + self.hi_r) / 2\n"
        "    pub fn origin(&self) -> i32 {\n"
        "        let _msg = \"decoy in a string: (self.lo_r + self.hi_r) / 2\";\n"
        "        (self.lo_q + self.hi_q) / 2\n"
        "    }\n"
        "}\n",
    )
    found = midpoint_constructions(tmp_path)
    assert found == [("planted/src/lib.rs", 5, "( self.lo_q + self.hi_q ) / 2")], found
    with pytest.raises(SecondWindowOriginAuthority, match="NEW"):
        require_census(found, _THE_MIDPOINT_CONSTRUCTIONS)


def test_a_fifth_site_whose_operands_are_ACCESSOR_or_HELPER_CALLS_is_named(tmp_path):
    """PB-9b. The field-to-getter spelling, measured to walk past the identifier-only matcher —
    and the ARGUMENT-BEARING helper call beside it, which used to be a stated residue and is
    now inside, because removing the operand grammar removed the distinction that kept it out.
    """
    _plant(
        tmp_path,
        "impl Thing {\n"
        "    pub fn origin(&self) -> i32 {\n"
        "        (self.lo_q() + self.hi_q()) / 2\n"
        "    }\n"
        "    pub fn helper_form(&self, a: i32, b: i32) -> i32 {\n"
        "        (mid_of(a, b) + mid_of(b, a)) / 2\n"
        "    }\n"
        "}\n",
    )
    found = midpoint_constructions(tmp_path)
    assert found == [
        ("planted/src/lib.rs", 3, "( self.lo_q() + self.hi_q() ) / 2"),
        ("planted/src/lib.rs", 6, "( mid_of ( a , b ) + mid_of ( b , a ) ) / 2"),
    ], found
    with pytest.raises(SecondWindowOriginAuthority, match="NEW"):
        require_census(found, _THE_MIDPOINT_CONSTRUCTIONS)


#: One plant per spelling RED-TEAM measured walking past the operand grammar. Rendering is not
#: asserted here — only that the site is SEEN, which is the property that failed.
_EVASION_SPELLINGS: tuple[tuple[str, str], ...] = (
    ("typed-suffix", "(min_q + max_q) / 2i32"),
    ("underscore-suffix", "(min_q + max_q) / 2_i32"),
    ("deref-operands", "(*a + *b) / 2"),
    ("cast", "((min_q + max_q) as i32) / 2"),
    ("indexed-operands", "(v[0] + v[1]) / 2"),
    ("mixed", "(*v[0] + self.hi.q()) / 2usize"),
)


@pytest.mark.parametrize(
    "label,expr", _EVASION_SPELLINGS, ids=[row[0] for row in _EVASION_SPELLINGS]
)
def test_each_MEASURED_evasion_spelling_is_SEEN_by_the_census(label, expr, tmp_path):
    """Every row here was planted against the real crates tree and confirmed to walk past the
    identifier-run operand grammar. `(v[0] + v[1]) / 2` is the one that matters: a bbox held as
    `[i32; 2]` is an ordinary refactor and the graph crate already works in coordinate arrays.
    """
    _plant(tmp_path / label, f"pub fn origin(a: i32, b: i32) -> i32 {{\n    {expr}\n}}\n")
    found = midpoint_constructions(tmp_path / label)
    assert len(found) == 1, f"{label}: {expr} was not seen — {found}"
    with pytest.raises(SecondWindowOriginAuthority, match="NEW"):
        require_census(found, _THE_MIDPOINT_CONSTRUCTIONS)


def test_the_SHIFT_spelling_remains_OUTSIDE_the_census(tmp_path):
    """The residue, asserted rather than claimed. `>> 1` is a different token shape, not a
    different operand shape, and no widening of the operand side reaches it — so a reader of
    the residue paragraph can check the paragraph against this."""
    _plant(tmp_path, "pub fn origin(a: i32, b: i32) -> i32 {\n    (a + b) >> 1\n}\n")
    assert midpoint_constructions(tmp_path) == []


def test_a_DELETED_pinned_site_is_named():
    """PB-10. Asserted as a set difference in BOTH directions, separately, because a
    one-directional comparison passes a deletion in silence."""
    observed = list(_THE_MIDPOINT_CONSTRUCTIONS[1:])
    with pytest.raises(SecondWindowOriginAuthority, match="GONE"):
        require_census(observed, _THE_MIDPOINT_CONSTRUCTIONS)


# --------------------------------------------------------------------------------------- #
# REPORT (`slow`) — the float residual the integer gate cannot see. No threshold, no verdict.
# --------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_report_the_graph_node_feature_translation_residual(derived):
    """`norm_q`/`norm_r` (`crates/mantis-graph/src/lib.rs:547-548`) are CENTROID-relative and
    computed in f64 before narrowing to f32, so their translation invariance is exact only up
    to rounding. This half reports the max |Δ| over the corpus and asserts NOTHING about its
    magnitude — a threshold here would be an armed value, and this is the one equivariance fact
    the orbit probe does not cover because it is a property of the ENCODER, not of the weights.

    Rows compared are the stone rows and the rows `legal_node_gather` names. The wire also
    carries one ungathered node whose coordinate is a fixed (0, 0) in every position; it is
    excluded by construction and its count is reported rather than silently dropped.
    """
    import numpy as np

    graph_specs = [s for s in roster() if s.is_graph]
    assert graph_specs, "no registered graph encoding — this report has no subject"
    worst = 0.0
    compared = 0
    excluded = 0
    for spec in graph_specs:
        for moves in SHAPES.values():
            base = graph_wire_for(spec.name, build_board(spec.name, moves))
            n_stones = int(np.asarray(base.n_stones)[0])
            rows = list(range(n_stones)) + np.asarray(base.legal_node_gather).tolist()
            total_nodes = np.asarray(base.node_coords).reshape(-1, 2).shape[0]
            excluded += total_nodes - len(rows)
            for t in UNIT_AXIAL:
                moved = graph_wire_for(spec.name, build_board(spec.name, translate(moves, t)))
                coords_b = np.asarray(base.node_coords).reshape(-1, 2)[rows]
                coords_t = np.asarray(moved.node_coords).reshape(-1, 2)[rows]
                assert np.array_equal(coords_t, coords_b + np.asarray(t)), (
                    "the compared rows are not the translated image of each other; the residual "
                    "below would be measuring node correspondence, not float rounding"
                )
                feat_b = np.asarray(base.node_feat).reshape(total_nodes, -1)[rows]
                feat_t = np.asarray(moved.node_feat).reshape(total_nodes, -1)[rows]
                worst = max(worst, float(np.max(np.abs(feat_t - feat_b))))
                compared += 1
    derived("t1.report.max_abs_node_feat_residual", worst)
    derived("t1.report.position_pairs_compared", compared)
    derived("t1.report.ungathered_nodes_excluded", excluded)
    assert compared > 0
