"""A4-2: the sync-free helpers and the pinned `non_blocking` H2D path change no byte."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from _wire_geometry import geometry_kwargs

from mantis.model.dist65 import VALUE_SUPPORT, decode_binned_value
from mantis.model.gnn import _node_offsets_to_batch_vec
from mantis.selfplay.graph_collate import (
    GraphWirePayload,
    collate_graph_batch,
    segment_softmax,
    stone_mask_from_batch,
)

_GEOMETRY = geometry_kwargs()
_TENSORS = ("x", "edge_index", "edge_attr", "legal_offsets", "legal_node_gather",
            "node_offsets", "n_stones")


def _sized_and_unsized(node_offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_total = int(node_offsets[-1])
    counts = node_offsets[1:] - node_offsets[:-1]
    unsized = torch.repeat_interleave(
        torch.arange(node_offsets.shape[0] - 1, dtype=torch.long), counts)
    return _node_offsets_to_batch_vec(node_offsets, n_total), unsized


def _unsized_segment_softmax(logits: torch.Tensor, legal_offsets: torch.Tensor) -> torch.Tensor:
    """The pre-A4-2 body, op for op, without `output_size`."""
    counts = legal_offsets[1:] - legal_offsets[:-1]
    b = int(legal_offsets.shape[0]) - 1
    seg = torch.repeat_interleave(torch.arange(b, dtype=torch.long), counts)
    seg_max = torch.full((b,), float("-inf"), dtype=logits.dtype)
    seg_max.scatter_reduce_(0, seg, logits, reduce="amax", include_self=False)
    ex = torch.exp(logits - seg_max[seg])
    denom = torch.zeros(b, dtype=logits.dtype)
    denom.scatter_add_(0, seg, ex)
    return ex / denom[seg]


def test_output_size_changes_no_element_of_the_batch_vec() -> None:
    offsets = torch.tensor([0, 3, 4, 9, 9, 15], dtype=torch.long)
    sized, unsized = _sized_and_unsized(offsets)
    assert torch.equal(sized, unsized)


def test_segment_softmax_and_stone_mask_are_unchanged_on_cpu(payload_fields) -> None:
    batch = collate_graph_batch(GraphWirePayload(**payload_fields("b6")), device="cpu",
                                semantic="off", **_GEOMETRY)
    logits = torch.randn(int(batch.legal_offsets[-1]), generator=torch.Generator().manual_seed(7))
    assert torch.equal(segment_softmax(logits, batch.legal_offsets),
                       _unsized_segment_softmax(logits, batch.legal_offsets))
    mask = stone_mask_from_batch(batch)
    expect = torch.zeros(int(batch.x.shape[0]), dtype=torch.bool)
    for g in range(batch.n_graphs):
        start = int(batch.node_offsets[g])
        expect[start:start + int(batch.n_stones[g])] = True
    assert torch.equal(mask, expect)


def test_decode_uses_one_resident_support_equal_to_the_module_constant() -> None:
    logits = torch.randn(5, VALUE_SUPPORT.numel(), generator=torch.Generator().manual_seed(3))
    probs = torch.softmax(logits, dim=-1)
    expect = (probs * VALUE_SUPPORT).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    assert torch.equal(decode_binned_value(logits), expect)
    assert decode_binned_value(logits).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — the pinned H2D path is CUDA-only")
def test_pinned_non_blocking_collate_is_byte_identical_to_a_pageable_copy(payload_fields) -> None:
    payload = GraphWirePayload(**payload_fields("b6"))
    on_cpu = collate_graph_batch(payload, device="cpu", semantic="off", **_GEOMETRY)
    on_cuda = collate_graph_batch(payload, device="cuda", semantic="off", **_GEOMETRY)
    torch.cuda.synchronize()
    for name in _TENSORS:
        host, dev = getattr(on_cpu, name), getattr(on_cuda, name)
        assert dev.device.type == "cuda", name
        assert dev.dtype == host.dtype and tuple(dev.shape) == tuple(host.shape), name
        assert np.array_equal(dev.cpu().numpy(), host.numpy()), f"{name} differs across the H2D path"
    sized, unsized = _sized_and_unsized(on_cuda.node_offsets.cpu())
    assert torch.equal(sized, unsized)


def test_the_resident_support_built_under_inference_mode_still_serves_autograd() -> None:
    """The cache is shared by the serving forward (inference mode) and the trainer (autograd)."""
    from mantis.model import dist65

    dist65._SUPPORT_ON.pop(torch.device("cpu"), None)
    with torch.inference_mode():
        decode_binned_value(torch.zeros(2, VALUE_SUPPORT.numel()))
    logits = torch.zeros(2, VALUE_SUPPORT.numel(), requires_grad=True)
    decode_binned_value(logits).sum().backward()
    assert logits.grad is not None
