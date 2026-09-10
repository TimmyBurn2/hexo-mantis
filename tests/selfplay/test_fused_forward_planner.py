# >300 justify (R8): one claim — `plan_fused_forwards` is the same greedy partition under a
# different name authority — over one shared arithmetic rig, and the partition properties only
# prove that claim alongside the rows that check which config key the refusal names.
"""Cover the fused-forward planner's partition, its bound and its typed refusal."""
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

#: The over-cap refusal `plan_microbatches` produces at HEAD, frozen verbatim: the default
#: `key=` must reproduce it byte for byte, and a paraphrase cannot hold that claim.
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
    """Transcribe the greedy rule from the design text, independently of the implementation."""
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


def test_fg1_01_a_pop_that_fits_under_both_members_is_one_forward() -> None:
    """Prove totals sitting exactly at both caps are legal and run as one forward."""
    ec = np.array([5, 9, 3, 9], dtype=np.int64)
    nc = np.array([4, 2, 7, 1], dtype=np.int64)
    caps = _caps(int(ec.sum()), int(nc.sum()))
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    assert parts == ((0, 4),), (
        f"a pop whose totals sit exactly at both caps must run as ONE forward; got {parts}")
    _assert_partition_properties(ec, nc, caps.max_fused_edges, caps.max_fused_nodes, parts)


@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg1_02_a_pop_over_a_member_splits_at_graph_boundaries(member: str) -> None:
    """Prove the cut lands on a graph boundary driven by either member.

    The node arm is the one an edges-only implementation fails while passing every other row.
    """
    ec = np.array([10, 10, 10, 10], dtype=np.int64)
    nc = np.array([4, 4, 4, 4], dtype=np.int64)
    caps = _caps(25, 10 ** 9) if member == "edges" else _caps(10 ** 9, 9)
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    assert len(parts) > 1, (
        f"the {member} member did not bind — an implementation blind to it bounds nothing")
    _assert_partition_properties(ec, nc, caps.max_fused_edges, caps.max_fused_nodes, parts)


def test_fg1_03_partition_properties_over_randomised_inputs() -> None:
    """Prove the five partition properties over >=200 randomised inputs."""
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
    """Prove an empty pop plans zero forwards, never one empty part.

    Nothing downstream of the inference loop raises on a zero-graph wire, unlike the trainer.
    """
    empty = np.zeros(1, dtype=np.int64)
    assert plan_fused_forwards(empty, empty, _caps(10, 10)) == ()


def test_fg1_05_the_plan_is_identical_over_repeated_calls() -> None:
    """Prove the plan is a pure function of counts and caps over repeated calls."""
    ec = np.array([7, 3, 11, 2, 9, 4], dtype=np.int64)
    nc = np.array([3, 2, 5, 1, 4, 2], dtype=np.int64)
    caps = _caps(15, 8)
    first = plan_fused_forwards(_offsets(ec), _offsets(nc), caps)
    for i in range(100):
        assert plan_fused_forwards(_offsets(ec), _offsets(nc), caps) == first, (
            f"call {i} produced a different plan — the partition is not a pure function")


def test_fg1_05_the_plan_matches_the_shared_train_side_planner() -> None:
    """Prove the fused planner is an adapter over the shared one, not a second transcription."""
    ec = np.array([6, 6, 6, 6, 6], dtype=np.int64)
    nc = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    assert plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(13, 10 ** 9)) == \
        plan_microbatches(_offsets(ec), _offsets(nc), 13, 10 ** 9)


@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg1_06_a_single_over_cap_graph_refuses_by_name(member: str) -> None:
    """Prove the refusal names the graph, its counts, the breached member, its value and the
    inference key path — never a truncation, a drop or a runtime cap-raise."""
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
    """Prove the inference refusal never leaks the train-side key it shares a planner with."""
    ec = np.array([99], dtype=np.int64)
    nc = np.array([9], dtype=np.int64)
    with pytest.raises(FusedGraphOverCap) as exc:
        plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(10, 10 ** 9))
    assert "train.microbatch_caps" not in str(exc.value), (
        "the inference adapter leaked the TRAIN key into its refusal (D-2/R73)")
    assert "max_edges" not in str(exc.value).replace("max_fused_edges", ""), (
        "the inference adapter leaked the TRAIN member name `max_edges` into its refusal")


def test_fg1_07_a_trainer_side_handler_does_not_catch_the_inference_refusal() -> None:
    """Prove the two refusals stay diagnosable apart, structurally and behaviourally.

    The structural half alone would pass an adapter that raised the train type and annotated it.
    """
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
    """Prove the default `key=` leaves every existing caller's refusal text byte-identical.

    No other row asserts that text — they assert the exception type only.
    """
    max_edges, max_nodes = (6, 100) if member == "edges" else (100, 3)
    with pytest.raises(GraphMicroBatchOverCap) as exc:
        plan_microbatches(np.array([0, 7], dtype=np.int64),
                          np.array([0, 4], dtype=np.int64), max_edges, max_nodes)
    assert str(exc.value) == expected, (
        "the DEFAULT `key=` changed an existing caller's refusal text; D-2 admits the "
        "parameter only because the default is behaviour-preserving")


def test_fg1_08_the_key_parameter_is_keyword_only_and_defaulted() -> None:
    """Prove `key` is keyword-only with the train key as its default.

    A positional or required parameter would change the arity existing four-positional callers
    rely on.
    """
    params = inspect.signature(plan_microbatches).parameters
    assert "key" in params, "`plan_microbatches` gained no `key=` parameter (D-2)"
    key = params["key"]
    assert key.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"`key` must be KEYWORD-ONLY; it is {key.kind}, which changes positional arity")
    assert key.default == "train.microbatch_caps", (
        f"`key`'s default must be the train key verbatim; it is {key.default!r}")


#: Adversarial `(edge counts, node counts, caps)` rows, each naming the shape it is
#: adversarial about; a uniform bank hides exactly the cases a greedy gets wrong.
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
    """Prove every part satisfies both members over the adversarial bank."""
    e = np.asarray(ec, dtype=np.int64)
    n = np.asarray(nc, dtype=np.int64)
    parts = plan_fused_forwards(_offsets(e), _offsets(n), _caps(cap_e, cap_n))
    _assert_partition_properties(e, n, cap_e, cap_n, parts)


def test_fg2_01_near_cap_graphs_force_one_forward_each() -> None:
    """Prove near-cap graphs force one forward each — the worst case the mechanism admits."""
    ec = np.asarray([10] * 8, dtype=np.int64)
    nc = np.asarray([4] * 8, dtype=np.int64)
    parts = plan_fused_forwards(_offsets(ec), _offsets(nc), _caps(10, 4))
    assert len(parts) == 8, (
        f"eight graphs each exactly at the cap must run as eight forwards; got {len(parts)}")


@pytest.mark.parametrize("member", ["max_fused_edges", "max_fused_nodes"])
def test_fg2_02_at_the_cap_is_legal_and_one_over_is_not(member: str) -> None:
    """Prove at-the-cap is legal and one-over is not, on both members, in all four quadrants.

    A `>=`-for-`>` implementation passes two of the four, so they travel as one row.
    """
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
