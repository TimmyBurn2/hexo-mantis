"""Collate BYTE-PARITY against the captured old outputs.

No tolerance exists on the parity arms: the collate path is reshape/copy-only, so ANY byte
difference is real contract drift, not numerical noise. The capture's own `verify_fixtures.py`
reproduced these outputs bit-exactly from the reloaded npz, so a mismatch here is a port
defect, never a fixture artifact.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from _retired_batch_fields import RETIRED_BATCH_FIELDS
from _wire_geometry import geometry_kwargs
import pytest
import torch

from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    collate_graph_batch,
    segment_softmax,
    stone_mask_from_batch,
)
from mantis.train.losses import segment_softmax as train_segment_softmax

# The capture ran on CPU with `torch.set_num_threads(1)`; B-04/B-05 reproduce that regime.
CAPTURE_TORCH_THREADS = 1
# Seed for the B-06 property battery (test-local; B-06 compares two NEW-side copies).
DUPLICATION_BATTERY_SEED = 20260723


@pytest.fixture
def single_threaded_torch():
    """Pin torch to 1 CPU thread for the bit-parity arms, then restore the process default."""
    previous = torch.get_num_threads()
    torch.set_num_threads(CAPTURE_TORCH_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


#: The capture's geometry, READ OFF THE REGISTRY ROW it was built at, never typed.
GEOMETRY: dict[str, int] = geometry_kwargs()


def _collate(fields: dict[str, Any], **kw: Any):
    """The geometry is stated on every call — `collate_graph_batch` requires it."""
    return collate_graph_batch(GraphWirePayload(**fields), **{**GEOMETRY, **kw})


#: The ONE authority lives in `_retired_batch_fields`. The capture `.npz` is NOT regenerated to
#: drop retired keys: a byte-parity capture whose bytes get rewritten when the code changes has
#: stopped being a capture. The retirement is asserted POSITIVELY below rather than skipped,
#: because a silent `continue` over an unmatched golden key is a check that stopped checking.


def _assert_tensor_parity(batch, golden: dict[str, np.ndarray], label: str) -> None:
    for field, expected in golden.items():
        if field in RETIRED_BATCH_FIELDS:
            assert not hasattr(batch, field), (
                f"{label}.{field}: the batch produces a field this capture records as RETIRED. "
                "Either the retirement was reverted without updating this list, or a field was "
                "re-added under a retired name — both need saying out loud, not passing quietly"
            )
            continue
        actual = getattr(batch, field).detach().cpu().numpy()
        assert actual.shape == expected.shape, (
            f"{label}.{field}: shape {actual.shape} != {expected.shape}"
        )
        assert actual.dtype == expected.dtype, (
            f"{label}.{field}: dtype {actual.dtype} != {expected.dtype}"
        )
        assert np.array_equal(actual, expected), (
            f"{label}.{field}: byte parity lost vs the captured old collate output — the "
            "contract reader changed the bytes it hands the NN"
        )


def test_collate_output_byte_parity_b6(payload_fields, collated_golden):
    """All 12 output tensors of the B=6 mixed-spread wire are element-wise identical (f32
    bit-exact, int exact) to the captured old outputs. FAIL = the port is not equivalent on the
    real multi-graph fused batch."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    _assert_tensor_parity(batch, collated_golden["b6"], "b6")


def test_collate_output_byte_parity_b1(payload_fields, collated_golden):
    """The degenerate single-graph batch is byte-identical to capture. FAIL = the B=1 path
    (node_offsets[0]==0 => local==global) drifted from the general one."""
    batch = _collate(payload_fields("b1"), expected_version=1, device="cpu", semantic="full")
    _assert_tensor_parity(batch, collated_golden["b1"], "b1")


@pytest.mark.parametrize("semantic", ["full", "off"])
def test_collate_empty_batch_parity(payload_fields, collated_golden, semantic):
    """B=0 collates SUCCESSFULLY and byte-identically under both semantic modes; the capture's
    two outputs were per-tensor sha-equal, so one golden serves both. FAIL = the empty-batch arm
    drifted, in shape, dtype, or existence."""
    batch = _collate(payload_fields("b0"), expected_version=1, device="cpu", semantic=semantic)
    _assert_tensor_parity(batch, collated_golden["b0"], f"b0[{semantic}]")


def test_segment_softmax_bit_parity(payload_fields, hotpath_golden, single_threaded_torch):
    """`segment_softmax(captured_logits, batch.legal_offsets)` is BIT-exact against the captured
    old output (same torch minor, CPU, 1 thread). The logits vector is loaded from the capture,
    not regenerated, so the seed cannot drift. FAIL = the ragged softmax the Rust assembler
    normalizes against changed."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    logits = torch.from_numpy(hotpath_golden["logits"].copy())
    expected = hotpath_golden["segment_softmax"]

    probs = segment_softmax(logits, batch.legal_offsets).detach().cpu().numpy()
    assert probs.dtype == expected.dtype, f"dtype {probs.dtype} != {expected.dtype}"
    assert np.array_equal(probs, expected), (
        "segment_softmax output is not bit-identical to capture; PREREG permits NO tolerance "
        "here without a root-caused mechanism in IMPL_NOTES + dispatcher sign-off"
    )


def test_stone_mask_bit_parity(payload_fields, hotpath_golden, single_threaded_torch):
    """`stone_mask_from_batch(batch)` equals the captured mask exactly. FAIL = the value-head
    pooling subset changed, which silently re-weights every value target."""
    batch = _collate(payload_fields("b6"), expected_version=1, device="cpu", semantic="full")
    expected = hotpath_golden["stone_mask"]

    mask = stone_mask_from_batch(batch).detach().cpu().numpy()
    assert mask.dtype == expected.dtype == np.bool_
    assert np.array_equal(mask, expected), "stone mask drifted vs capture"
    assert int(mask.sum()) == int(expected.sum()) == 60


def _battery() -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Property battery: single graph, many graphs, large magnitudes, count-1 segments."""
    gen = torch.Generator().manual_seed(DUPLICATION_BATTERY_SEED)
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    cases.append(("single_graph",
                  torch.randn(17, generator=gen, dtype=torch.float32),
                  torch.tensor([0, 17], dtype=torch.long)))
    counts = [3, 1, 8, 1, 12, 5]
    offsets = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.long)
    cases.append(("many_graphs_with_count_1_segments",
                  torch.randn(int(offsets[-1]), generator=gen, dtype=torch.float32), offsets))
    cases.append(("large_magnitude",
                  torch.randn(20, generator=gen, dtype=torch.float32) * 500.0,
                  torch.tensor([0, 5, 20], dtype=torch.long)))
    cases.append(("all_count_1",
                  torch.randn(4, generator=gen, dtype=torch.float32),
                  torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)))
    return cases


def test_train_reads_the_selfplay_segment_softmax_and_never_a_copy():
    """§c.6's declared duplication is RESOLVED (R367(a)): `train.losses.segment_softmax` IS the selfplay authority, one object, so the two cannot drift."""
    assert train_segment_softmax is segment_softmax
    for label, logits, offsets in _battery():
        probs = segment_softmax(logits, offsets)
        seg_sums = torch.stack([
            probs[int(offsets[i]):int(offsets[i + 1])].sum() for i in range(len(offsets) - 1)
        ])
        assert torch.allclose(seg_sums, torch.ones_like(seg_sums), atol=1e-6), (
            f"{label}: per-segment probabilities must sum to 1"
        )


