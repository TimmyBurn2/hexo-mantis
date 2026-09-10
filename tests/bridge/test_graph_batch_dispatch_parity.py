"""Batch dispatch — cross-FFI parity + per-id demux integrity.

The block-diagonal fuse is pinned in `queue_fuse_pin.rs`; what was NOT pinned anywhere is that
routing N leaves through ONE pop yields per-leaf results identical to routing them one at a
time. A bad demux breaks that SILENTLY — every leaf still gets *a* policy, just the wrong one.
The eval/arena decode driver shares this submit, so leaving it serial would make the two arms
disagree about batching. Per-segment probabilities depend on the SEGMENT WIDTH alone, so serial
and fused forwards see bit-identical inputs: any output difference is the dispatch, never the
arithmetic.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from mantis import _engine

# Five positions with DELIBERATELY different legal-set widths (measured: 126 / 138 / 156 / 230 /
# 196 legal cells); equal widths would hide a segment-offset bug, and `test_ragged_widths_...`
# asserts the fixture disagrees with itself.
_POSITIONS: list[tuple[list[tuple[int, int, int]], int, int]] = [
    ([(0, 0, 1)], -1, 2),
    ([(0, 0, 1), (1, 0, -1)], 1, 1),
    ([(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 0, -1)], 1, 2),
    ([(0, 0, 1), (5, 5, -1)], 1, 2),
    ([(0, 0, 1), (1, 0, -1), (3, 3, 1)], -1, 2),
]


@pytest.fixture
def graph_batcher():
    """A fresh graph `InferenceBatcher` per test — in-flight state must not travel."""
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    batcher = _engine.InferenceBatcher(encoding_spec=spec)
    yield batcher
    batcher.close()


def _segment_probs(offsets: np.ndarray) -> np.ndarray:
    """A normalized ramp per segment, determined by the width ALONE so serial and fused agree
    exactly, and summing to 1 per segment so the assemble-side invariant holds."""
    flat = np.zeros((int(offsets[-1]),), dtype=np.float32)
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        w = e - s
        if w > 0:
            ramp = np.arange(1, w + 1, dtype=np.float64)
            flat[s:e] = (ramp / ramp.sum()).astype(np.float32)
    return flat


def _segment_widths(offsets: np.ndarray) -> list[int]:
    return [int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)]


def _policy_width(dense: Any, overflow: Any) -> int:
    """The assembled policy's own legal-set width: in-window slots carry a nonzero probability
    and off-window cells ride `overflow`."""
    return int(np.count_nonzero(np.asarray(dense, dtype=np.float64))) + len(overflow)


def _round_trip(
    batcher: Any, positions: list, *, values_from: str = "width"
) -> tuple[list, int]:
    """One blocking `submit_graphs_and_wait` served by a producer driven HERE, returning
    `(results, pops)`. `pops == 1` for an N-position call is the dispatch claim: the serial
    submit it replaces could only put one graph in the queue at a time."""
    box: dict[str, Any] = {}

    def submitter() -> None:
        try:
            box["res"] = batcher.submit_graphs_and_wait(positions)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread
            box["exc"] = exc

    thread = threading.Thread(target=submitter, daemon=True)
    thread.start()

    pops = 0
    served = 0
    # BUDGETED BY WALL TIME, NOT BY ITERATION COUNT: a fixed count worked only while the submit
    # held the GIL through the enqueue. With the leaf BUILD inside `py.detach`, empty pops
    # return immediately and every iteration burns while the submitter is still building.
    deadline = time.monotonic() + 30.0
    while served < len(positions) and time.monotonic() < deadline:
        ids, wire = batcher.next_graph_batch(len(positions), 200)
        ids = list(ids)
        if not ids:
            continue
        pops += 1
        offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
        widths = _segment_widths(offsets)
        values = (
            np.asarray(widths, dtype=np.float32)
            if values_from == "width"
            else np.arange(len(ids), dtype=np.float32)
        )
        batcher.submit_graph_inference_results(
            ids, _segment_probs(offsets), offsets, values
        )
        served += len(ids)

    thread.join(timeout=30)
    assert not thread.is_alive(), "submit_graphs_and_wait never returned"
    if "exc" in box:
        raise box["exc"]
    return box["res"], pops


def test_fused_batch_returns_per_graph_results_identical_to_serial_forwards(
    graph_batcher,
) -> None:
    """One N-graph pop vs N one-graph pops must agree, per graph. The serial arm is the
    pre-change dispatch, reproduced by calling with one position at a time."""
    serial = [_round_trip(graph_batcher, [p])[0][0] for p in _POSITIONS]
    fused, pops = _round_trip(graph_batcher, _POSITIONS)

    assert pops == 1, f"the whole batch must be served by ONE pop, took {pops}"
    assert len(fused) == len(serial) == len(_POSITIONS)
    for i, (f, s) in enumerate(zip(fused, serial, strict=True)):
        f_dense, f_overflow, f_value = f
        s_dense, s_overflow, s_value = s
        np.testing.assert_array_equal(
            np.asarray(f_dense), np.asarray(s_dense), err_msg=f"dense policy drift at {i}"
        )
        assert dict(f_overflow) == dict(s_overflow), f"overflow map drift at {i}"
        assert f_value == s_value, f"value drift at {i}"


def test_ragged_widths_do_not_bleed_across_graphs_in_the_fused_batch(
    graph_batcher,
) -> None:
    """Positions with DIFFERENT legal-set widths in one batch: each graph's assembled policy
    must carry its OWN width, and the fixture is asserted to disagree with itself first."""
    fused, pops = _round_trip(graph_batcher, _POSITIONS)

    assert pops == 1
    widths = [_policy_width(dense, overflow) for dense, overflow, _ in fused]
    assert len(set(widths)) > 1, "fixture must exercise ragged widths, else the pin is vacuous"
    # `values_from="width"` stamps each segment's width into its value, so a crossed segment is
    # arithmetically visible rather than merely plausible.
    for i, (dense, overflow, value) in enumerate(fused):
        assert _policy_width(dense, overflow) == pytest.approx(value), (
            f"leaf {i} assembled a width-{_policy_width(dense, overflow)} policy from a "
            f"width-{int(value)} segment — the demux crossed two graphs"
        )


def test_each_request_id_receives_its_own_segment_of_the_flat_probs(
    graph_batcher,
) -> None:
    """Each id's policy is assembled from ITS retained `policy_dst_slot`, checked by feeding a
    fused batch whose values are position-encoded. The pop assertion is the dispatch claim
    itself: under the serial submit, `next_graph_batch` could only hand back one id."""
    fused, pops = _round_trip(graph_batcher, _POSITIONS, values_from="index")

    assert pops == 1, "the batch must arrive fused, not one at a time"
    assert len(fused) == len(_POSITIONS)
    for i, (dense, overflow, value) in enumerate(fused):
        assert value == pytest.approx(float(i)), f"waiter {i} received another graph's value"
        # Each policy still sums to 1 over its own legal set; a one-entry shift breaks that.
        mass = float(np.asarray(dense, dtype=np.float64).sum()) + sum(
            p for _, p in overflow
        )
        assert mass == pytest.approx(1.0, abs=1e-3)


def test_a_segment_length_mismatch_fails_that_id_and_the_tail_rather_than_crossing_them(
    graph_batcher,
) -> None:
    """The die-loud arm: a segment whose length disagrees with the retained `policy_dst_slot`
    must raise and fail the remaining ids, never assemble a shifted policy. Under the serial
    submit only ONE id is in flight and the tail does not exist, so the pop-width assertion is
    what binds this to the batched dispatch."""
    box: dict[str, Any] = {}

    def submitter() -> None:
        try:
            batcher_result = graph_batcher.submit_graphs_and_wait(_POSITIONS)
            box["res"] = batcher_result
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread
            box["exc"] = exc

    thread = threading.Thread(target=submitter, daemon=True)
    thread.start()

    request_ids: list[int] = []
    offsets = np.zeros((1,), dtype=np.int64)
    for _ in range(200):
        ids, wire = graph_batcher.next_graph_batch(len(_POSITIONS), 200)
        request_ids = list(ids)
        if request_ids:
            offsets = np.asarray(wire.legal_offsets, dtype=np.int64)
            break
    assert len(request_ids) == len(_POSITIONS), (
        "the batch must arrive fused for the tail to exist at all, got "
        f"{len(request_ids)} of {len(_POSITIONS)}"
    )

    probs = _segment_probs(offsets)
    bad_offsets = offsets.copy()
    bad_offsets[1] += 1  # shift ONE interior boundary: every later segment is misaligned
    with pytest.raises(ValueError, match="segment len .* != n_legal"):
        graph_batcher.submit_graph_inference_results(
            request_ids,
            probs,
            bad_offsets,
            np.zeros((len(request_ids),), dtype=np.float32),
        )

    thread.join(timeout=30)
    assert not thread.is_alive(), "a failed batch must not orphan its waiters"
    assert "exc" in box, "the caller must see the failure, never a shifted policy"
    assert "segment len" in str(box["exc"])
