"""The readout's per-graph sums (value pools, served softmax, policy loss) repeat exactly and stay fp32-accurate."""
from __future__ import annotations

import pytest
import torch

from mantis.model.gnn import segment_lengths, segment_sums
from mantis.selfplay.graph_collate import segment_ids, segment_softmax

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — the atomic sums jitter only on CUDA")


def _segments(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(20260925)
    lens = torch.randint(300, 900, (64,), generator=gen)
    lens[5] = 0
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)])
    values = torch.randn(int(lens.sum()), 512, generator=gen)
    return values.to(device), segment_ids(offsets.to(device), total=int(lens.sum())), offsets.to(device)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_CUDA)])
def test_segment_sums_match_fp64_and_zero_an_empty_segment(device: str) -> None:
    """Every segment within fp32 rounding of the fp64 sum; an empty graph sums to exactly 0."""
    values, seg, offsets = _segments(device)
    got = segment_sums(values, segment_lengths(seg, 64))
    ref = torch.stack([values[offsets[i]:offsets[i + 1]].double().sum(0) for i in range(64)])
    assert got.dtype is values.dtype and got.shape == (64, 512)
    assert torch.equal(got[5], torch.zeros_like(got[5]))
    assert float((got.double() - ref).abs().max()) < 1e-3


@_CUDA
def test_segment_sums_repeat_exactly_where_the_atomic_sum_does_not() -> None:
    """CONTROL first: the atomic `index_add_` over the same segments must differ across ten repeats."""
    values, seg, _offsets = _segments("cuda")
    atomic = [torch.zeros(64, 512, device="cuda").index_add_(0, seg, values) for _ in range(10)]
    assert any(not torch.equal(atomic[0], a) for a in atomic[1:]), "the atomic control never jittered: no reading"
    lengths = segment_lengths(seg, 64)
    runs = [segment_sums(values, lengths) for _ in range(10)]
    assert all(torch.equal(runs[0], r) for r in runs[1:])


@_CUDA
def test_the_served_softmax_repeats_exactly() -> None:
    """The per-graph softmax the server returns: bit-identical on a repeat."""
    values, _seg, offsets = _segments("cuda")
    logits = values[:, 0].contiguous()
    first, second = segment_softmax(logits, offsets), segment_softmax(logits, offsets)
    assert torch.equal(first, second)
