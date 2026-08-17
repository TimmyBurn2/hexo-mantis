# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence.
# The rows here are ONE claim — "`plan_fused_forwards` is the SAME greedy partition under a
# DIFFERENT name authority" — and they share one arithmetic rig (`_offsets`, the reference
# transcription, the property bundle). Splitting the partition properties from the refusal's
# name-truth rows would put the algorithm in one file and the only thing that proves it names
# the right config key in another, and D-2's whole content is that those two travel together.
"""⊕ F-816-10 F1/F2 — the fused-forward planner, its bound, and its typed refusal.

Written by ORACLE-WRITE **before** the feature exists (packet `plan/F816_10_PACKET.md`,
design verdict V-A, rulings D-2/D-6). Every row that imports `plan_fused_forwards` /
`FusedGraphOverCap` is RED at authorship — that is the correct pre-IMPL state, not a defect.

The defect each row is the ONLY witness to:

- **FG1-01/02/03** — a partition that drops, duplicates, reorders or splits early. A property
  is binary, so the randomised row is a bank of >=200 inputs and a counter-example HALTS.
- **FG1-04** — a trailing EMPTY part on `B == 0`. A naive reading of the greedy loop appends
  one unconditionally, and an empty part would collate a zero-graph batch on the inference
  arm, where there is no trainer to raise on it.
- **FG1-05** — a plan that depends on host state (RNG, dict order, a memoised accumulator).
  No behavioural row can see a nondeterministic partition that happens to be legal each time.
- **FG1-06** — a refusal that names the WRONG config key, the R73 name-truth class. This is
  the reason D-2 exists at all: the shared planner bakes `train.microbatch_caps` into its
  message, and an inference-side refusal carrying that string is a FALSE PROVENANCE RECORD
  that sends an operator to edit a key that had nothing to do with the failure.
- **FG1-07** — `FusedGraphOverCap` made a SUBCLASS of `GraphMicroBatchOverCap`. Every
  trainer-side `except GraphMicroBatchOverCap` would then silently swallow an inference-side
  refusal; the two seams would stop being diagnosable apart and no message assertion would
  notice, because the message would still be right.
- **FG1-08** — a required `key=` kwarg (the design's original shape, OVERRULED by D-2). A
  required parameter churns 13+ call sites and moves
  `tests/train/test_graph_microbatch_authority.py`, which the design itself declares MUST NOT
  MOVE. The default's ONE job is that every existing caller keeps its exact current message,
  so the message is frozen here verbatim rather than described.
- **FG2-01/02** — an off-by-one greedy that admits one over-bound part (`>=` for `>`), and an
  edges-only implementation that passes every row a node-blind bank can produce.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.selfplay.graph_wire_split import (
    FusedGraphOverCap,
    GraphMicroBatchOverCap,
    plan_fused_forwards,
    plan_microbatches,
)

SEED = 20260817

#: The refusal text `plan_microbatches` produces at HEAD for an over-cap graph, VERBATIM.
#: FG1-08 asserts the DEFAULT `key=` reproduces it byte for byte. Frozen as data rather than
#: described in prose, because "behaviour-preserving" is exactly the claim D-2 rests on and a
#: paraphrase cannot hold it.
_HEAD_EDGES_MESSAGE = (
    "graph 0 needs 7 edges and 4 nodes on its own, which exceeds max_edges=6 "
    "(train.microbatch_caps.max_edges). Micro-batching partitions at GRAPH boundaries, so a "
    "single graph is the atom and no split reduces it — this is out of the domain the caps "
    "were sized for. Never a silent truncation and never a silent drop (R114)."
)
_HEAD_NODES_MESSAGE = (
    "graph 0 needs 7 edges and 4 nodes on its own, which exceeds max_nodes=3 "
    "(train.microbatch_caps.max_nodes). Micro-batching partitions at GRAPH boundaries, so a "
    "single graph is the atom and no split reduces it — this is out of the domain the caps "
    "were sized for. Never a silent truncation and never a silent drop (R114)."
)


def _offsets(counts) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _caps(max_edges: int, max_nodes: int) -> FusedGraphCapsSpec:
    return FusedGraphCapsSpec(max_fused_edges=max_edges, max_fused_nodes=max_nodes)


def _reference_plan(ec, nc, max_edges: int, max_nodes: int) -> list[tuple[int, int]]:
    """An INDEPENDENT transcription of the order-preserving greedy rule, written from the
    design text (F816_10_DESIGN §4.2 / graph_wire_split's docstring) rather than from the
    implementation. Two transcriptions of one stated rule disagree exactly where the rule was
    misread — which is the only place a shared-planner reuse can go wrong silently."""
    parts: list[tuple[int, int]] = []
    start, acc_e, acc_n = 0, 0, 0
    for i in range(len(ec)):
        if (acc_e + int(ec[i]) > max_edges or acc_n + int(nc[i]) > max_nodes) and i > start:
            parts.append((start, i))
            start, acc_e, acc_n = i, 0, 0
        acc_e += int(ec[i])
        acc_n += int(nc[i])
    parts.append((start, len(ec)))
    return parts


def _assert_partition_properties(ec, nc, max_edges: int, max_nodes: int, parts) -> None:
    """The five properties one ordered contiguous bounded partition must satisfy."""
    b = len(ec)
    assert parts, "a non-empty pop must produce at least one bounded forward"
    assert parts[0][0] == 0 and parts[-1][1] == b, (
        f"the plan {parts} does not cover [0, {b})")
    for (a0, a1), (n0, n1) in zip(parts, parts[1:], strict=False):
        assert a0 < a1, f"empty or inverted part {(a0, a1)}"
        assert a1 == n0, f"parts {(a0, a1)} and {(n0, n1)} are not contiguous"
    covered = [g for g0, g1 in parts for g in range(g0, g1)]
    assert covered == list(range(b)), (
        f"the plan is not an ordered cover of [0, {b}) — a graph was dropped, duplicated or "
        "reordered, and a reordered plan mis-assigns every policy downstream of it")
    for g0, g1 in parts:
        assert int(np.sum(ec[g0:g1])) <= max_edges, (
            f"part {(g0, g1)} breaches max_fused_edges={max_edges}")
        assert int(np.sum(nc[g0:g1])) <= max_nodes, (
            f"part {(g0, g1)} breaches max_fused_nodes={max_nodes}")
    for g0, g1 in parts[:-1]:
        assert (int(np.sum(ec[g0:g1 + 1])) > max_edges
                or int(np.sum(nc[g0:g1 + 1])) > max_nodes), (
            f"part {(g0, g1)} is not maximal — graph {g1} fits and was split off anyway, "
            "which pays a launch floor for nothing")
    assert list(parts) == _reference_plan(ec, nc, max_edges, max_nodes), (
        "the plan disagrees with an independent transcription of the stated greedy rule")


# ═══ FG1 — the planner ═══════════════════════════════════════════════════════════════════
def test_fg1_01_a_pop_that_fits_under_both_members_is_one_forward() -> None:
    """FG1-01 — no split when none is needed, at the EXACT boundary on both members.

    Totals equal to the caps are legal; this is the `>=`-for-`>` off-by-one's home, so it is a
    named row rather than a value a randomiser might happen to draw."""
    ec = np.array([5, 9, 3, 9], dtype=np.int64)
    nc = np.array([4, 2, 7, 1], dtype=np.int64)
    caps = _caps(int(ec.sum()), int(nc.sum()))
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    assert parts == ((0, 4),), (
        f"a pop whose totals sit exactly at both caps must run as ONE forward; got {parts}")
    _assert_partition_properties(ec, nc, caps.max_fused_edges, caps.max_fused_nodes, parts)


@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg1_02_a_pop_over_a_member_splits_at_graph_boundaries(member: str) -> None:
    """FG1-02 — the cut lands on a GRAPH boundary, driven by EITHER member.

    The node-driven arm is the MB-19 mutation transplanted: an edges-only implementation
    passes every edge-shaped row here and produces an unbounded N term, which §1.4 of the
    design shows is the LARGER of the two in the worst case an edge-only cap admits."""
    ec = np.array([10, 10, 10, 10], dtype=np.int64)
    nc = np.array([4, 4, 4, 4], dtype=np.int64)
    caps = _caps(25, 10 ** 9) if member == "edges" else _caps(10 ** 9, 9)
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    assert len(parts) > 1, (
        f"the {member} member did not bind — an implementation blind to it bounds nothing")
    _assert_partition_properties(ec, nc, caps.max_fused_edges, caps.max_fused_nodes, parts)


def test_fg1_03_partition_properties_over_randomised_inputs() -> None:
    """FG1-03 — the five properties on 100% of >=200 randomised inputs. A counter-example is
    a HALT, not a rate: a partition property is binary."""
    rng = np.random.default_rng(SEED)
    checked = 0
    for _ in range(240):
        b = int(rng.integers(1, 24))
        ec = rng.integers(1, 500, size=b).astype(np.int64)
        nc = rng.integers(1, 60, size=b).astype(np.int64)
        caps = _caps(int(rng.integers(int(ec.max()), int(ec.sum()) + 1)),
                     int(rng.integers(int(nc.max()), int(nc.sum()) + 1)))
        parts = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
        _assert_partition_properties(ec, nc, caps.max_fused_edges, caps.max_fused_nodes,
                                     parts)
        checked += 1
    assert checked >= 200, f"the row requires >=200 randomised inputs; ran {checked}"


def test_fg1_04_an_empty_pop_plans_zero_forwards() -> None:
    """FG1-04 — `B == 0` returns `()`, never one empty part.

    An empty part would hand `collate_graph_batch` a zero-graph wire inside the production
    inference loop, where (unlike the trainer) nothing downstream raises on it — the failure
    would surface as an FFI length mismatch three frames later, if at all."""
    empty = np.zeros(1, dtype=np.int64)
    assert plan_fused_forwards(empty, empty, _caps(10, 10)) == ()


def test_fg1_05_the_plan_is_identical_over_repeated_calls() -> None:
    """FG1-05 — a pure function of `(edge counts, node counts, caps)`. 100 repeats of the same
    inputs return the identical tuple; a partition that depended on host state would still be
    legal on every call and no bound-checking row could see it."""
    ec = np.array([7, 3, 11, 2, 9, 4], dtype=np.int64)
    nc = np.array([3, 2, 5, 1, 4, 2], dtype=np.int64)
    caps = _caps(15, 8)
    first = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    for i in range(100):
        assert plan_fused_forwards(_offsets(ec), _offsets(nc), caps) == first, (
            f"call {i} produced a different plan — the partition is not a pure function")


def test_fg1_05_the_plan_matches_the_shared_train_side_planner() -> None:
    """FG1-05 second limb — ONE greedy loop, two name authorities (D-2).

    `plan_fused_forwards` must be an adapter over `plan_microbatches`, not a second
    transcription: two implementations of one algorithm agree right up until they diverge, and
    the divergence would be a memory bound that is correct on one arm only."""
    ec = np.array([6, 6, 6, 6, 6], dtype=np.int64)
    nc = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    assert plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(13, 10 ** 9)) == \
        plan_microbatches(_offsets(ec), _offsets(nc), 13, 10 ** 9)


@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg1_06_a_single_over_cap_graph_refuses_by_name(member: str) -> None:
    """FG1-06 — the refusal names the graph, its `(N, E)`, WHICH member, that member's value
    and the INFERENCE key path. Never a truncation, never a drop, never a runtime cap-raise.

    The last is refused on `graph_wire_split.py`'s own recorded grounds: clamping the cap up
    at runtime is tune-to-green (R61) and makes the peak-allocation bound unprovable."""
    ec = np.array([4, 31, 5], dtype=np.int64)
    nc = np.array([3, 17, 4], dtype=np.int64)
    cap_e = 30 if member == "max_fused_edges" else 10 ** 9
    cap_n = 16 if member == "max_fused_nodes" else 10 ** 9
    with pytest.raises(FusedGraphOverCap) as exc:
        plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(cap_e, cap_n))
    msg = str(exc.value)
    assert "graph 1" in msg, f"the offending graph's index is not named: {msg!r}"
    assert "31" in msg and "17" in msg, f"the graph's (E, N) is not named: {msg!r}"
    assert member in msg, f"the breached member is not named: {msg!r}"
    assert str(cap_e if member == "max_fused_edges" else cap_n) in msg, (
        f"the member's VALUE is not named, so an operator cannot tell how far over it is: "
        f"{msg!r}")
    assert f"inference.fused_graph_caps.{member}" in msg, (
        f"the refusal does not name the config key path an operator must edit: {msg!r}")


def test_fg1_06_the_inference_refusal_never_names_the_train_side_key() -> None:
    """FG1-06 second limb — the R73 name-truth claim, stated negatively.

    The shared planner bakes `train.microbatch_caps` into its message. A refusal that leaked
    it out of the inference adapter would send an operator to re-mint a key that had nothing
    to do with the failure — a false provenance record, which is worse than no message."""
    ec = np.array([99], dtype=np.int64)
    nc = np.array([9], dtype=np.int64)
    with pytest.raises(FusedGraphOverCap) as exc:
        plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(10, 10 ** 9))
    assert "train.microbatch_caps" not in str(exc.value), (
        "the inference adapter leaked the TRAIN key into its refusal (D-2/R73)")
    assert "max_edges" not in str(exc.value).replace("max_fused_edges", ""), (
        "the inference adapter leaked the TRAIN member name `max_edges` into its refusal")


def test_fg1_07_a_trainer_side_handler_does_not_catch_the_inference_refusal() -> None:
    """FG1-07 — the two refusals are diagnosable APART (D-2).

    Asserted twice, structurally and behaviourally, because the structural half alone would
    survive an implementation that raised `GraphMicroBatchOverCap` from the adapter and only
    ANNOTATED it. A trainer-side `except GraphMicroBatchOverCap` that swallowed an
    inference-side refusal would turn a run-fatal memory refusal into a silently skipped
    forward on the wrong seam."""
    assert not issubclass(FusedGraphOverCap, GraphMicroBatchOverCap), (
        "`FusedGraphOverCap` is a subclass of the train-side refusal — every "
        "`except GraphMicroBatchOverCap` in the trainer now swallows an inference refusal")
    assert not issubclass(GraphMicroBatchOverCap, FusedGraphOverCap), (
        "the inverse subclassing is equally fatal, in the other direction")
    ec = np.array([99], dtype=np.int64)
    nc = np.array([9], dtype=np.int64)
    caught_by_trainer_handler = False
    try:
        try:
            plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(10, 10 ** 9))
        except GraphMicroBatchOverCap:
            caught_by_trainer_handler = True
    except FusedGraphOverCap:
        pass
    assert not caught_by_trainer_handler, (
        "a trainer-side `except GraphMicroBatchOverCap` caught an inference-side refusal")


@pytest.mark.parametrize(
    ("member", "expected"),
    [("edges", _HEAD_EDGES_MESSAGE), ("nodes", _HEAD_NODES_MESSAGE)],
)
def test_fg1_08_the_default_key_preserves_every_existing_callers_message(
    member: str, expected: str
) -> None:
    """FG1-08 — D-2's load-bearing half: the new `key=` parameter's DEFAULT hides no
    authority, because every caller that does not pass it keeps its exact current message.

    Frozen VERBATIM. A default that silently reworded the train-side refusal would make this
    a behaviour change dressed as a refactor, and `tests/train/test_graph_microbatch.py`'s
    OF2-7 row asserts only the exception TYPE, so nothing else in the tree would notice."""
    max_edges, max_nodes = (6, 100) if member == "edges" else (100, 3)
    with pytest.raises(GraphMicroBatchOverCap) as exc:
        plan_microbatches(np.array([0, 7], dtype=np.int64),
                          np.array([0, 4], dtype=np.int64), max_edges, max_nodes)
    assert str(exc.value) == expected, (
        "the DEFAULT `key=` changed an existing caller's refusal text; D-2 admits the "
        "parameter only because the default is behaviour-preserving")


def test_fg1_08_the_key_parameter_is_keyword_only_and_defaulted() -> None:
    """FG1-08 second limb — the parameter is KEYWORD-ONLY with the train key as its default.

    A positional parameter would break
    `tests/train/test_graph_microbatch_authority.py:580-582`, which calls `plan_microbatches`
    with four POSITIONAL arguments — and that file MUST NOT MOVE (design §8), because its
    frozen AST census is the whole reason the inference members are not named
    `max_edges`/`max_nodes`. A REQUIRED parameter (the design's original shape) would break it
    outright; D-2 overruled that."""
    params = inspect.signature(plan_microbatches).parameters
    assert "key" in params, "`plan_microbatches` gained no `key=` parameter (D-2)"
    key = params["key"]
    assert key.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"`key` must be KEYWORD-ONLY; it is {key.kind}, which changes positional arity")
    assert key.default == "train.microbatch_caps", (
        f"`key`'s default must be the train key verbatim; it is {key.default!r}")


# ═══ FG2 — the bound actually bounds ═════════════════════════════════════════════════════
#: An adversarial bank of `(edge counts, node counts, caps)`, each row naming the shape it
#: is adversarial ABOUT. Uniform banks hide exactly the cases a greedy gets wrong.
_BANK: list[tuple[str, list[int], list[int], int, int]] = [
    ("one dominant graph among tiny ones", [1, 1, 97, 1, 1], [1, 1, 41, 1, 1], 97, 41),
    ("every graph exactly at the cap", [10] * 8, [4] * 8, 10, 4),
    ("every graph one below the cap", [9] * 8, [3] * 8, 10, 4),
    ("ragged, edge-bound", [3, 17, 5, 11, 2, 19, 7], [2, 6, 3, 4, 1, 7, 3], 20, 10 ** 9),
    ("ragged, node-bound", [3, 17, 5, 11, 2, 19, 7], [2, 6, 3, 4, 1, 7, 3], 10 ** 9, 8),
    ("ragged, both bind", [3, 17, 5, 11, 2, 19, 7], [2, 6, 3, 4, 1, 7, 3], 20, 8),
    ("edge-light node-heavy (V-D's grave)", [2, 2, 2, 2, 2, 2], [50, 50, 50, 50, 50, 50],
     10 ** 9, 100),
    ("single graph", [7], [3], 7, 3),
]


@pytest.mark.parametrize(("label", "ec", "nc", "cap_e", "cap_n"), _BANK,
                         ids=[row[0] for row in _BANK])
def test_fg2_01_every_part_satisfies_both_members_over_the_bank(
    label: str, ec: list[int], nc: list[int], cap_e: int, cap_n: int
) -> None:
    """FG2-01 — the bound BOUNDS, on every part, on both members, over the adversarial bank.

    The "every graph exactly at the cap" row is the one that forces many parts: a greedy that
    accumulates before checking emits `M == 1` and breaches by 8x."""
    e = np.asarray(ec, dtype=np.int64)
    n = np.asarray(nc, dtype=np.int64)
    parts = plan_fused_forwards(_offsets(e), _offsets(n), _caps(cap_e, cap_n))
    _assert_partition_properties(e, n, cap_e, cap_n, parts)


def test_fg2_01_near_cap_graphs_force_one_forward_each() -> None:
    """FG2-01 second limb — the `M == B` worst case the mechanism admits, asserted rather
    than argued. Design §6.2 prices it; a planner that quietly merged two near-cap graphs
    would make that price wrong AND breach the bound."""
    ec = np.asarray([10] * 8, dtype=np.int64)
    nc = np.asarray([4] * 8, dtype=np.int64)
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(10, 4))
    assert len(parts) == 8, (
        f"eight graphs each exactly at the cap must run as eight forwards; got {len(parts)}")


@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg2_02_at_the_cap_is_legal_and_one_over_is_not(member: str) -> None:
    """FG2-02 — the off-by-one, on both members, in all four quadrants.

    `total == cap` is ONE part; `total == cap + 1` is TWO. `single == cap` is legal;
    `single == cap + 1` REFUSES rather than emitting a part that breaches its own bound. An
    implementation with `>=` for `>` passes the first and third and fails the second and
    fourth, which is why all four are one row."""
    ec = np.asarray([5, 5], dtype=np.int64)
    nc = np.asarray([3, 3], dtype=np.int64)
    at_cap = _caps(10, 10 ** 9) if member == "max_fused_edges" else _caps(10 ** 9, 6)
    assert plan_fused_forwards(_offsets(ec), _offsets(nc), at_cap) == ((0, 2),), (
        "a pop whose total sits EXACTLY at the cap is legal and must not be split")

    one_over = _caps(9, 10 ** 9) if member == "max_fused_edges" else _caps(10 ** 9, 5)
    assert plan_fused_forwards(_offsets(ec), _offsets(nc), one_over) == ((0, 1), (1, 2)), (
        "a pop one unit over the cap must split; an admitted over-bound part is not a bound")

    single_e = np.asarray([5], dtype=np.int64)
    single_n = np.asarray([3], dtype=np.int64)
    exact = _caps(5, 10 ** 9) if member == "max_fused_edges" else _caps(10 ** 9, 3)
    assert plan_fused_forwards(_offsets(single_e), _offsets(single_n), exact) == ((0, 1),), (
        "a single graph EXACTLY at the cap fits and must not refuse")
    under = _caps(4, 10 ** 9) if member == "max_fused_edges" else _caps(10 ** 9, 2)
    with pytest.raises(FusedGraphOverCap):
        plan_fused_forwards(_offsets(single_e), _offsets(single_n), under)
