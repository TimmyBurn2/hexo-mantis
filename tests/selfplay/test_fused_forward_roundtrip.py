# >300 justify (R8): the rows here are ONE claim — the split forward IS the un-split forward,
# positionally — and every one of them, including the three mutation self-tests, binds the SAME
# comparison helper. Separating the self-tests would let the oracle be weakened without the file
# that proves it detects a transposition ever being opened.
"""The concat-then-submit round trip for a split fused forward.

THE HIGHEST-RISK ORACLE IN THE PACKET. There is ZERO in-repo precedent — the train-side split
accumulates GRADIENTS and never concatenates per-part OUTPUT ARRAYS back into one
positionally-ordered result. The failure mode is sharper than the trainer's: a transposition
assigns a POLICY TO THE WRONG GAME STATE. And the FFI covers half of it — its per-id
`policy_dst_slot.len() != leaf_probs.len()` check is defeated by two same-length graphs and
there is NO check on `values[i]` ordering, so on the value axis this suite is the only
instrument that exists.

The rig therefore uses UNEQUAL per-graph legal counts AND distinct value sentinels, and asserts
the two axes SEPARATELY. POSITIONALLY EXACT, NOT BIT-EXACT, because GPU reductions are not
order-invariant across batch shapes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import _fused_graph_harness as H

_Result = tuple[list[int], np.ndarray, np.ndarray, np.ndarray]

#: Ragged on purpose (D-3): no two adjacent graphs share a legal-node count, so a swap of any
#: neighbouring pair changes the flat probs layout as well as its contents.
_RAGGED_LEGAL = [2, 5, 3, 7, 4, 6, 1, 8]


def assert_positional_round_trip(expected: _Result, actual: _Result) -> None:
    """The oracle: `actual` (split) must reproduce `expected` (un-split) POSITION FOR POSITION.
    Four separable claims — ids, segment structure, probs, values — asserted separately so a
    failure names which one moved. `rtol=1e-6` is positional exactness, not bit-exactness."""
    exp_ids, exp_probs, exp_offsets, exp_values = expected
    act_ids, act_probs, act_offsets, act_values = actual

    assert act_ids == exp_ids, (
        f"the request ids moved: {act_ids} != {exp_ids}. The ids travel in the pop order the "
        "Rust producer built and the split must not touch them — a re-ordered id list "
        "mis-assigns every result in the pop.")
    assert np.array_equal(act_offsets, exp_offsets), (
        f"the submitted `legal_offsets` moved: {act_offsets} != {exp_offsets}. The submit "
        "must carry the UNSLICED offsets; a part's own re-based offsets segment the "
        "concatenated array wrongly from the first part onward.")
    assert act_probs.shape == exp_probs.shape, (
        f"the probs length moved: {act_probs.shape} != {exp_probs.shape} — a part's output "
        "was dropped or duplicated.")
    assert act_values.shape == exp_values.shape, (
        f"the values length moved: {act_values.shape} != {exp_values.shape}")
    np.testing.assert_allclose(
        act_probs, exp_probs, rtol=1e-6, atol=1e-7,
        err_msg="the concatenated probs are not the un-split probs, position for position — "
                "a transposition here assigns a policy to the WRONG GAME STATE")
    np.testing.assert_allclose(
        act_values, exp_values, rtol=1e-6, atol=1e-7,
        err_msg="the concatenated values are not the un-split values, position for position. "
                "NOTHING downstream checks `values[i]` ordering (review Finding 9) — this "
                "assertion is the only instrument on that axis")


def _drive(monkeypatch: Any, payload: Any, cap_e: int, cap_n: int) -> tuple[_Result, Any]:
    server, batcher, _net = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=cap_e, max_fused_nodes=cap_n)
    assert batcher.failures == [], (
        f"the drive failed instead of serving: {batcher.failures}")
    assert len(batcher.results) == 1, (
        f"exactly ONE submit per pop is the FFI contract; got {len(batcher.results)}")
    return batcher.results[0], server


@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg3_01_the_split_forward_is_the_unsplit_forward_positionally(
    monkeypatch, member: str
) -> None:
    """Split == un-split, position for position, driven from BOTH members. The two drives differ
    ONLY in the caps, so any difference is attributable to the split and nothing else."""
    payload = H.build_payload(_RAGGED_LEGAL)
    ec, nc = H.per_graph_counts(payload)
    whole, whole_server = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)

    cap_e = int(ec.max()) + 1 if member == "edges" else 10 ** 9
    cap_n = int(nc.max()) + 1 if member == "nodes" else 10 ** 9
    split, split_server = _drive(monkeypatch, payload, cap_e, cap_n)

    fusion = split_server.batch_timing_snapshot()["fusion"]
    assert fusion["fusion_parts"] > 1, (
        f"the {member} cap did not bind — this row proves nothing about a split it never ran")
    assert whole_server.batch_timing_snapshot()["fusion"]["fusion_parts"] == 1, (
        "the reference drive split too; it is no longer the un-split reference")
    assert_positional_round_trip(whole, split)


def test_fg3_01_a_maximally_split_pop_still_round_trips(monkeypatch) -> None:
    """Second limb — `M == B`, one graph per forward, where every part is a single-graph batch
    whose own `legal_offsets` are `[0, L_g]`: the shape a re-based-offset bug looks correct in."""
    payload = H.build_payload(_RAGGED_LEGAL)
    ec, nc = H.per_graph_counts(payload)
    whole, _ = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)
    split, server = _drive(monkeypatch, payload, int(ec.max()), int(nc.max()))
    assert server.batch_timing_snapshot()["fusion"]["fusion_parts"] == len(_RAGGED_LEGAL), (
        "the caps admit exactly one graph per forward, so M must equal B")
    assert_positional_round_trip(whole, split)


def test_fg3_02_each_graphs_value_survives_the_split_in_its_own_slot(monkeypatch) -> None:
    """The value axis, asserted on its own and against the ids, because the two axes fail
    independently: a bug that concatenates `values_parts` in plan order but `probs_parts` in
    reverse reds only the probs assertion. DISCLOSED: a differential row cannot see a defect that
    corrupts BOTH sides identically, which is a HEAD defect rather than a split defect."""
    payload = H.build_payload(_RAGGED_LEGAL)
    ec, _nc = H.per_graph_counts(payload)
    (_ids_w, _p_w, _o_w, values_whole), _ = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)
    (ids_s, _p_s, _o_s, values_split), split_server = _drive(
        monkeypatch, payload, int(ec.max()) + 1, 10 ** 9)
    assert split_server.batch_timing_snapshot()["fusion"]["fusion_parts"] > 1, (
        "the cap did not bind — a differential row over two identical drives proves nothing")

    assert len(values_whole) == len(ids_s) == len(_RAGGED_LEGAL)
    for i, (a, b) in enumerate(zip(values_whole, values_split, strict=True)):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-7), (
            f"graph {i}'s value sentinel changed across the split ({a} -> {b}) — this leaf's "
            "search would back up another position's evaluation")


#: Two graphs with the SAME legal-node count, and a third of a different count so the split is
#: forced between them. `_RAGGED_LEGAL` is all-distinct, which is this row's exact blind spot.
_EQUAL_LEGAL = [3, 5, 5]


def test_fg3_08_two_graphs_of_EQUAL_legal_count_cannot_be_transposed(monkeypatch) -> None:
    """The transposition every length check in the stack is blind to: the FFI's only per-id
    defence is `meta.policy_dst_slot.len() != leaf_probs.len()`, defeated when two graphs carry
    the SAME legal-node count, and there is no per-id check on `values[i]`. RED-TEAM injected such
    a swap and the whole fused-forward family stayed GREEN, so this row is the one that reds."""
    payload = H.build_payload(_EQUAL_LEGAL)
    ec, nc = H.per_graph_counts(payload)
    assert nc[1] == nc[2] and ec[1] == ec[2], (
        "the premise of this row is two graphs the length checks cannot tell apart; got "
        f"nodes {nc.tolist()} edges {ec.tolist()}")

    whole, whole_server = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)
    split, split_server = _drive(monkeypatch, payload, int(ec.max()), int(nc.max()))

    assert whole_server.batch_timing_snapshot()["fusion"]["fusion_parts"] == 1, (
        "the reference drive split too; it is no longer the un-split reference")
    parts = split_server.batch_timing_snapshot()["fusion"]["fusion_parts"]
    assert parts == len(_EQUAL_LEGAL), (
        f"one graph per forward is what puts the two equal-length graphs in ADJACENT parts, "
        f"which is where a same-length swap lives; got {parts} parts")

    assert_positional_round_trip(whole, split)

    # The value axis again, per-slot: a reader must see that the EQUAL-LENGTH pair kept its own
    # sentinels, because that is the pair no length check protects.
    values_whole, values_split = whole[3], split[3]
    for i in (1, 2):
        assert values_whole[i] == pytest.approx(values_split[i], rel=1e-6, abs=1e-7), (
            f"graph {i} is one of the two EQUAL-legal-count graphs and its value moved "
            f"({values_whole[i]} -> {values_split[i]}) — a transposition no length check in "
            "the FFI or the collate can see")
    assert values_whole[1] != pytest.approx(values_whole[2], rel=1e-9, abs=1e-12), (
        "the two equal-length graphs must carry DISTINGUISHABLE value sentinels or this row "
        "is green under a swap by construction")


def test_fg3_03_one_submit_per_pop_against_the_unsliced_legal_offsets(monkeypatch) -> None:
    """The FFI's four self-consistency checks are satisfied by the ONE submit: `values.len() == n`,
    `legal_offsets.len() == n + 1`, `lo[0] == 0`, `lo[n] == probs.len()`."""
    payload = H.build_payload(_RAGGED_LEGAL)
    ec, _nc = H.per_graph_counts(payload)
    (ids, probs, offsets, values), server = _drive(
        monkeypatch, payload, int(ec.max()) + 1, 10 ** 9)

    n = len(ids)
    assert ids == list(range(1, n + 1)), "the ids must arrive in the producer's pop order"
    assert len(values) == n, f"values.len()={len(values)} != n={n}"
    assert len(offsets) == n + 1, f"legal_offsets.len()={len(offsets)} != n+1={n + 1}"
    assert int(offsets[0]) == 0, "lo[0] must be 0"
    assert int(offsets[n]) == len(probs), (
        f"lo[n]={int(offsets[n])} != probs.len()={len(probs)}")
    assert np.array_equal(offsets, np.asarray(payload.legal_offsets, dtype=np.int64)), (
        "the submitted offsets are not the wire's own UNSLICED offsets")
    assert server.batch_timing_snapshot()["fusion"]["fusion_parts"] > 1, (
        "this row must run over an actual split")


def test_fg3_03_a_pop_that_fits_submits_exactly_as_head_does(monkeypatch) -> None:
    """Second limb — the M == 1 path production takes when the caps do not bind is unchanged, and
    it is the path every smoke config runs, so a regression there is invisible to splitting rows."""
    payload = H.build_payload([3, 4])
    (ids, probs, offsets, values), server = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)
    assert ids == [1, 2]
    assert server.forward_count == 1
    assert server.batch_timing_snapshot()["fusion"]["fusion_parts"] == 1
    for g in range(2):
        seg = probs[int(offsets[g]):int(offsets[g + 1])]
        assert seg.sum() == pytest.approx(1.0), (
            f"graph {g}'s segment does not normalise: {seg}")
    assert np.isfinite(values).all()


def _synthetic_pair() -> tuple[_Result, list[np.ndarray], list[np.ndarray]]:
    """Return a hand-built un-split result plus the per-part pieces a two-part plan would produce.
    Deliberately NOT produced by the server: these rows are about the comparison helper."""
    part_a_probs = np.array([0.4, 0.6, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    part_b_probs = np.array([0.7, 0.3, 0.25, 0.25, 0.25, 0.25, 0.5], dtype=np.float32)
    part_a_values = np.array([-0.10, -0.20], dtype=np.float32)
    part_b_values = np.array([-0.30, -0.40, -0.50], dtype=np.float32)
    offsets = np.array([0, 2, 6, 8, 12, 13], dtype=np.int64)
    whole = (
        [1, 2, 3, 4, 5],
        np.concatenate([part_a_probs, part_b_probs]),
        offsets,
        np.concatenate([part_a_values, part_b_values]),
    )
    return whole, [part_a_probs, part_b_probs], [part_a_values, part_b_values]


def test_fg3_04_reversing_the_parts_before_concatenation_is_detected() -> None:
    """`probs_parts[::-1]` before `np.concatenate` is detected: every part still ran, the array is
    the right length and `lo[n] == probs.len()` still passes, so only a positional comparison
    sees it — and only on a ragged rig."""
    whole, probs_parts, values_parts = _synthetic_pair()
    corrupted = (whole[0], np.concatenate(probs_parts[::-1]), whole[2], whole[3])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_05_swapping_two_graphs_values_is_detected() -> None:
    """Swapping two graphs' values is detected — the mutation nothing else in the stack can see,
    since the FFI checks a per-id probs SEGMENT LENGTH and nothing about `values[i]`."""
    whole, _probs_parts, values_parts = _synthetic_pair()
    values = np.concatenate(values_parts).copy()
    values[0], values[1] = values[1], values[0]
    corrupted = (whole[0], whole[1], whole[2], values)
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_06_submitting_a_parts_own_rebased_offsets_is_detected() -> None:
    """The LAST part's own re-based `legal_offsets` submitted in place of the unsliced ones is
    detected: `slice_graph_wire` re-bases every offset array, so this is a one-character mistake
    that still satisfies `lo[0] == 0`."""
    whole, _probs_parts, _values_parts = _synthetic_pair()
    rebased = whole[2] - whole[2][2]
    corrupted = (whole[0], whole[1], rebased, whole[3])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_06_a_dropped_part_is_detected() -> None:
    """Second limb — a plan whose last part never ran. The FFI's length check would catch it at
    the seam; the oracle must catch it HERE, where the message names the split."""
    whole, probs_parts, values_parts = _synthetic_pair()
    corrupted = (whole[0], probs_parts[0], whole[2], values_parts[0])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_07_the_per_graph_sentinels_are_pairwise_separable(monkeypatch) -> None:
    """The precondition FG3-01/02 rest on: if two graphs produced identical prob segments and
    values, a swap would be undetectable and those rows would pass VACUOUSLY. The separation must
    also be orders above the helper's `rtol`."""
    payload = H.build_payload(_RAGGED_LEGAL)
    (ids, probs, offsets, values), _ = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)

    segments = [tuple(np.round(probs[int(offsets[g]):int(offsets[g + 1])], 9).tolist())
                for g in range(len(ids))]
    assert len(set(segments)) == len(segments), (
        "two graphs produced identical probability segments — a transposition between them "
        "would be invisible and FG3-01 would pass vacuously")
    assert len(set(values.tolist())) == len(values), (
        "two graphs produced identical value sentinels — FG3-02 would pass vacuously")
    gaps = [abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]]
    assert min(gaps) > 1e-4, (
        f"the closest pair of value sentinels differ by {min(gaps)}, which is not "
        "comfortably above the helper's 1e-6 tolerance")
