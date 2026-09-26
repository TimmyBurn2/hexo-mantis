"""The segment sums' backward keeps no values alive and equals autograd's; the CPU message sum keeps an fp64 input fp64."""
from __future__ import annotations

import pytest
import torch

from mantis.model.gine import gine_message_sum
from mantis.model.gnn import segment_sums


def _plain(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return torch.segment_reduce(values.float(), "sum", offsets=offsets, axis=0).to(values.dtype)


def _saved_bytes(fn, *args: torch.Tensor) -> int:
    saved: list[int] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t.numel() * t.element_size())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return sum(saved)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(37,), (37, 5)])
def test_the_forward_and_gradient_equal_the_plain_path_bitwise(dtype: torch.dtype, shape: tuple) -> None:
    gen = torch.Generator().manual_seed(3)
    offsets = torch.tensor([0, 4, 4, 20, 37])
    values = torch.randn(shape, generator=gen).to(dtype)
    grad = torch.randn((4, *shape[1:]), generator=gen).to(dtype)
    a, b = values.clone().requires_grad_(), values.clone().requires_grad_()
    out_a, out_b = segment_sums(a, offsets), _plain(b, offsets)
    assert torch.equal(out_a, out_b)
    (ga,), (gb,) = torch.autograd.grad(out_a, (a,), grad), torch.autograd.grad(out_b, (b,), grad)
    assert torch.equal(ga, gb)


def test_the_backward_keeps_only_the_offsets() -> None:
    offsets = torch.tensor([0, 500, 1000])
    values = torch.randn(1000, 64, requires_grad=True)
    kept = _saved_bytes(segment_sums, values, offsets)
    assert kept <= offsets.numel() * offsets.element_size(), kept
    assert _saved_bytes(_plain, values, offsets) > values.numel() * 4, "the control no longer saves values"


@pytest.mark.parametrize("with_div", [False, True], ids=["no-div", "div"])
def test_an_fp64_input_is_summed_and_differentiated_in_fp64_on_cpu(with_div: bool) -> None:
    gen = torch.Generator().manual_seed(5)
    xs = torch.randn(30, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    e = torch.randn(90, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    src, dst = torch.randint(0, 30, (90,), generator=gen), torch.randint(0, 30, (90,), generator=gen)
    div = torch.randint(1, 9, (30, 1), generator=gen).double() if with_div else None
    grad = torch.randn(30, 8, generator=gen, dtype=torch.float64)
    got = gine_message_sum(xs, e, src, dst, None, div)
    want = torch.zeros(30, 8, dtype=torch.float64).index_add(0, dst, (xs.index_select(0, src) + e).relu())
    want = want if div is None else want / div
    assert got.dtype == torch.float64 and torch.equal(got, want)
    for a, b in zip(torch.autograd.grad(got, (xs, e), grad), torch.autograd.grad(want, (xs, e), grad), strict=True):
        assert torch.equal(a, b)
