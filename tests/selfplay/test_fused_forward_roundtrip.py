# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence.
# The rows here are ONE claim — "the split forward IS the un-split forward, positionally" —
# and every one of them, including the three mutation self-tests, binds the SAME comparison
# helper. The helper is the oracle; the rows are the drives and the proofs that the helper has
# teeth. Separating the self-tests into another file would let the helper be weakened without
# the file that proves it detects a transposition ever being opened.
"""⊕ F-816-10 F3 — the concat-then-submit round trip (dispatcher ruling D-3).

Written by ORACLE-WRITE **before** the feature exists. THE HIGHEST-RISK ORACLE IN THE PACKET,
and the reasons are on the record rather than asserted:

1. **Zero in-repo precedent.** The train-side split (`train/coordinator/dispatch.py`) splits
   and ACCUMULATES GRADIENTS against a whole-step denominator; it never concatenates per-part
   OUTPUT ARRAYS back into one positionally-ordered result (review Finding 6). The inference
   side must concatenate the parts in plan order and submit ONCE against the UNSLICED
   `legal_offsets`. That is new code with no shape to copy.
2. **The failure mode is sharper than the trainer's.** A gradient-accumulation bug degrades
   training quietly; a transposition here assigns a POLICY TO THE WRONG GAME STATE, and the
   search then expands the wrong move with full confidence.
3. **The FFI covers half of it and none of the other half.** `submit_graph_inference_results`
   (`crates/mantis-bridge/src/inference.rs`) checks `meta.policy_dst_slot.len() !=
   leaf_probs.len()` per id — a PARTIAL defence on probs, defeated by two same-length graphs —
   and has NO check whatsoever on `values[i]` ordering (review Finding 9). On the value axis
   this suite is the only instrument that exists.

So the rig uses UNEQUAL per-graph legal-node counts across parts AND distinct per-graph value
sentinels (D-3), and asserts the two axes SEPARATELY, so a probs transposition and a values
transposition cannot be confused for one another or hidden behind one another.

POSITIONALLY EXACT, NOT BIT-EXACT (D-6). GPU reductions are not reduction-order-invariant
across batch shapes, so numeric bit-exactness is neither claimed nor needed; the comparison is
a tight tolerance, and `test_fg3_07_...` asserts the sentinel separation is many orders above
it, which is what makes the tolerance safe rather than convenient.

The defect each row is the ONLY witness to:

- **FG3-01** — a plan applied in the wrong order, a part's output dropped, or a part's probs
  concatenated at the wrong offset. Driven from BOTH members, because an edges-only split and
  a nodes-driven split produce different part boundaries over the same wire.
- **FG3-02** — a values transposition. Nothing downstream checks it; see (3) above.
- **FG3-03** — a per-part submit (which would break the FFI's `lo[n] == probs.len()`
  self-consistency check), a re-based `legal_offsets` submitted in place of the unsliced one,
  or `request_ids` re-ordered to match the plan.
- **FG3-04/05/06** — the ORACLE going blind. Each is a LAW-07 mutation self-test on the
  comparison helper itself: it feeds the helper a deliberately corrupted result and requires
  it to raise. These are GREEN at authorship BY DESIGN — they test the oracle's detection
  power, which exists before the feature does.
- **FG3-07** — a rig whose sentinels are not separable, which would make FG3-01/02 pass
  vacuously under any transposition.
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
    """The oracle. `actual` (split) must reproduce `expected` (un-split) POSITION FOR POSITION.

    Four separable claims, asserted separately so a failure names which one moved:
    the request ids, the segment structure, the probs, and the values. `rtol=1e-6` is the
    D-6 wording applied numerically — positional exactness, not bit-exactness — and
    `test_fg3_07_...` pins that the rig's sentinel separation is orders above it.
    """
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


# ═══ FG3-01/02 — the round trip ══════════════════════════════════════════════════════════
@pytest.mark.parametrize("member", ["edges", "nodes"])
def test_fg3_01_the_split_forward_is_the_unsplit_forward_positionally(
    monkeypatch, member: str
) -> None:
    """FG3-01 — split == un-split, position for position, driven from BOTH members.

    The two drives differ ONLY in the caps: same wire, same net, same loop. Any difference in
    the submitted arrays is therefore attributable to the split and to nothing else."""
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
    """FG3-01 second limb — `M == B`, one graph per forward. The concatenation does the most
    work it can ever do, and every part is a single-graph batch whose own `legal_offsets` are
    `[0, L_g]` — the shape a re-based-offset bug looks most correct in."""
    payload = H.build_payload(_RAGGED_LEGAL)
    ec, nc = H.per_graph_counts(payload)
    whole, _ = _drive(monkeypatch, payload, 10 ** 9, 10 ** 9)
    split, server = _drive(monkeypatch, payload, int(ec.max()), int(nc.max()))
    assert server.batch_timing_snapshot()["fusion"]["fusion_parts"] == len(_RAGGED_LEGAL), (
        "the caps admit exactly one graph per forward, so M must equal B")
    assert_positional_round_trip(whole, split)


def test_fg3_02_each_graphs_value_survives_the_split_in_its_own_slot(monkeypatch) -> None:
    """FG3-02 — the value axis, asserted on its own and against the ids.

    Read separately from FG3-01 because the two axes fail independently: a bug that
    concatenates `values_parts` in plan order but `probs_parts` in reverse reds only the probs
    assertion, and the inverse bug reds only this one. Folding them into one comparison would
    let either failure be reported as the other.

    DISCLOSED RESIDUAL: a differential row cannot see a defect that corrupts BOTH sides
    identically (a global `values.reshape` transposition, say). That class is a HEAD defect
    rather than a split defect and is not what this packet introduces; FG3-05's mutation
    self-test is what covers the helper's own detection power independently of any drive."""
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


# ═══ FG3-08 — the EQUAL-LENGTH shape, where every length check is blind ══════════════════
#: Two graphs with the SAME legal-node count, and a third of a different count so the split
#: is forced between them. `_RAGGED_LEGAL` is deliberately all-distinct, which is the right
#: default for most rows and the exact blind spot for this one.
_EQUAL_LEGAL = [3, 5, 5]


def test_fg3_08_two_graphs_of_EQUAL_legal_count_cannot_be_transposed(monkeypatch) -> None:
    """FG3-08 — the transposition that every length check in the stack is blind to.

    WHY THIS ROW EXISTS, AND WHY THE REST OF THE FILE DOES NOT COVER IT. The FFI's only
    per-id defence against a mis-assigned policy is
    `meta.policy_dst_slot.len() != leaf_probs.len()` (`crates/mantis-bridge/src/inference.rs`),
    which is defeated outright when two graphs carry the SAME legal-node count — and there is
    no per-id check on `values[i]` ordering at all (REVIEW-design Finding 9, D-3). Every other
    payload in this file uses mutually distinct legal counts, so a swap of two parts is caught
    by a length mismatch rather than by the ordering assertion, and the ordering assertion is
    the thing this file exists to make. RED-TEAM proved the gap was real rather than
    theoretical: it injected a swap of two same-length parts into `_run_graph_loop` and the
    whole 69-row fused-forward family stayed GREEN while a genuine per-graph transposition
    occurred (F816_10_REDTEAM.md, H4).

    So: equal legal counts on graphs 1 and 2, a split forced between them, and BOTH axes
    compared positionally against the un-split reference. Under the injected swap this row is
    the one that reds.
    """
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

    # The value axis again, explicitly and per-slot: `assert_positional_round_trip` compares
    # the arrays, but a reader of this row must see that the EQUAL-LENGTH pair specifically
    # kept its own sentinels, because that is the pair no length check protects.
    values_whole, values_split = whole[3], split[3]
    for i in (1, 2):
        assert values_whole[i] == pytest.approx(values_split[i], rel=1e-6, abs=1e-7), (
            f"graph {i} is one of the two EQUAL-legal-count graphs and its value moved "
            f"({values_whole[i]} -> {values_split[i]}) — a transposition no length check in "
            "the FFI or the collate can see")
    assert values_whole[1] != pytest.approx(values_whole[2], rel=1e-9, abs=1e-12), (
        "the two equal-length graphs must carry DISTINGUISHABLE value sentinels or this row "
        "is green under a swap by construction")


# ═══ FG3-03 — one submit, unsliced offsets ═══════════════════════════════════════════════
def test_fg3_03_one_submit_per_pop_against_the_unsliced_legal_offsets(monkeypatch) -> None:
    """FG3-03 — the FFI's four self-consistency checks are satisfied by the ONE submit.

    `values.len() == n`, `legal_offsets.len() == n + 1`, `lo[0] == 0`,
    `lo[n] == probs.len()` (`crates/mantis-bridge/src/inference.rs`). Asserted here on the
    PYTHON side because the fake batcher stands in for Rust: if these four hold, the real
    batcher accepts the submit; if any fails, the run dies at the FFI with a message about
    lengths rather than about the split."""
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
    """FG3-03 second limb — the M == 1 path production takes when the caps do not bind is
    unchanged: one collate, one forward, one submit, probs summing to 1 per segment. The
    non-binding path is the one every smoke config runs (FG5-07), so a regression there is
    invisible to every splitting row."""
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


# ═══ FG3-04/05/06 — the oracle's own teeth (LAW-07 mutation self-tests) ══════════════════
def _synthetic_pair() -> tuple[_Result, list[np.ndarray], list[np.ndarray]]:
    """A hand-built un-split result plus the per-part pieces a two-part plan would produce.

    Deliberately NOT produced by the server: these three rows must hold before the feature
    exists, because they are about the comparison helper and not about the implementation."""
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
    """FG3-04 — the mutation D-3 names first: `probs_parts[::-1]` before `np.concatenate`.

    Every part still ran, every graph still got a policy, the array is still the right length
    and the FFI's `lo[n] == probs.len()` check still passes. Only a positional comparison sees
    it, and only if the parts have UNEQUAL lengths or unequal contents — which is why the rig
    is ragged."""
    whole, probs_parts, values_parts = _synthetic_pair()
    corrupted = (whole[0], np.concatenate(probs_parts[::-1]), whole[2], whole[3])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_05_swapping_two_graphs_values_is_detected() -> None:
    """FG3-05 — the mutation NOTHING else in the stack can see (review Finding 9).

    The FFI checks a per-id probs SEGMENT LENGTH and nothing about `values[i]`. Two graphs'
    values swapped is a legal-looking submit that backs up the wrong evaluation at two leaves,
    forever, silently."""
    whole, _probs_parts, values_parts = _synthetic_pair()
    values = np.concatenate(values_parts).copy()
    values[0], values[1] = values[1], values[0]
    corrupted = (whole[0], whole[1], whole[2], values)
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_06_submitting_a_parts_own_rebased_offsets_is_detected() -> None:
    """FG3-06 — the third mutation D-3 names: the LAST part's own re-based `legal_offsets`
    submitted in place of the wire's unsliced ones.

    `slice_graph_wire` re-bases every offset array so each part is a valid wire ON ITS OWN;
    submitting that re-based array is a one-character mistake (`sub.legal_offsets` for
    `payload.legal_offsets`) that still satisfies `lo[0] == 0` for a single-part plan."""
    whole, _probs_parts, _values_parts = _synthetic_pair()
    rebased = whole[2] - whole[2][2]
    corrupted = (whole[0], whole[1], rebased, whole[3])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


def test_fg3_06_a_dropped_part_is_detected() -> None:
    """FG3-06 second limb — a plan whose last part never ran (an early `break`, a `continue`
    in the part loop). The concatenation is shorter and the FFI's length check would catch it
    at the seam; the oracle must catch it HERE, where the message names the split."""
    whole, probs_parts, values_parts = _synthetic_pair()
    corrupted = (whole[0], probs_parts[0], whole[2], values_parts[0])
    with pytest.raises(AssertionError):
        assert_positional_round_trip(whole, corrupted)


# ═══ FG3-07 — the rig's own separability ═════════════════════════════════════════════════
def test_fg3_07_the_per_graph_sentinels_are_pairwise_separable(monkeypatch) -> None:
    """FG3-07 — the precondition FG3-01/02 rest on, asserted rather than assumed.

    If two graphs produced identical prob segments and identical values, a swap between them
    would be undetectable and FG3-01/02 would pass VACUOUSLY. The separation is also required
    to be many orders above the `rtol=1e-6` the helper uses, which is what makes the D-6
    tolerance safe rather than convenient."""
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
