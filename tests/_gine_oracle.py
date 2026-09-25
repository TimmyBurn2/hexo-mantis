"""The pre-L2 GINE aggregation (bf16 `index_add_`) and the exact controls, kept ONLY for L2's witnesses."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import torch
from torch import Tensor

from mantis.model.gine import _GINEConv


SumFn = Callable[[Tensor, Tensor, int, "Tensor | None"], Tensor]


def forward_with(sum_fn: SumFn) -> Callable[..., Tensor]:
    """`_GINEConv.forward` with its message, gather and update as at fc37f3f2 and the aggregation swapped for `sum_fn`."""
    def forward(self: _GINEConv, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                agg_divisor: Tensor | None = None) -> Tensor:
        n = x.shape[0]
        if edge_index.shape[1] == 0:
            agg = x.new_zeros((n, x.shape[1]))
            agg = agg if agg_divisor is None else agg / agg_divisor.to(agg.dtype)
            return self.nn(agg + (1.0 + self.eps) * x)
        e = self.lin(edge_attr)
        msg = (x.to(e.dtype).index_select(0, edge_index[0]) + e).relu()
        return self.nn(sum_fn(msg, edge_index[1], n, agg_divisor) + (1.0 + self.eps) * x)
    return forward


def index_add_sum(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
    """The pre-L2 aggregation alone: a `msg`-dtype `index_add_`, divided in that dtype."""
    agg = msg.new_zeros((n, msg.shape[1])).index_add_(0, dst, msg)
    return agg if divisor is None else agg / divisor.to(agg.dtype)


def exact_sum(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
    """The control that must pass: an fp32 buffer, deterministic `index_put_`, one rounding."""
    agg = msg.new_zeros((n, msg.shape[1]), dtype=torch.float32).index_put_((dst,), msg.float(), accumulate=True)
    return (agg if divisor is None else agg / divisor.float()).to(msg.dtype)


def fp64_sum(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
    """The reference every aggregation is measured against."""
    agg = msg.new_zeros((n, msg.shape[1]), dtype=torch.float64).index_add_(0, dst, msg.double())
    return agg if divisor is None else agg / divisor.double()


@contextmanager
def aggregating_with(sum_fn: SumFn) -> Iterator[None]:
    """Every `_GINEConv` in the process aggregates through `sum_fn` inside the block."""
    current = _GINEConv.forward
    _GINEConv.forward = forward_with(sum_fn)  # type: ignore[method-assign]
    try:
        yield
    finally:
        _GINEConv.forward = current  # type: ignore[method-assign]


def index_add_aggregation() -> AbstractContextManager[None]:
    """Every `_GINEConv` aggregates the pre-L2 way inside the block."""
    return aggregating_with(index_add_sum)


def capturing(into: list[tuple[Tensor, Tensor, int, Tensor | None]]) -> AbstractContextManager[None]:
    """The pre-L2 aggregation, appending each layer's real `(msg, dst, n, divisor)` to `into` on the host."""
    def record(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
        into.append((msg.detach().cpu(), dst.cpu(), n, None if divisor is None else divisor.detach().cpu()))
        return index_add_sum(msg, dst, n, divisor)
    return aggregating_with(record)


def synthetic_batch(device: str, *, n_graphs: int = 64, nodes: int = 780, degree: int = 16,
                    seed: int = 20260925) -> dict[str, Tensor]:
    """A B-64-sized fused batch with the real hot spot: every graph's last node is a dummy wired to every real node."""
    gen = torch.Generator().manual_seed(seed)
    real = nodes - 1
    src, dst = [], []
    for g in range(n_graphs):
        base = g * nodes
        s = torch.randint(0, real, (real * degree,), generator=gen) + base
        d = torch.arange(real).repeat_interleave(degree) + base
        dummy = torch.full((real,), base + real)
        ids = torch.arange(real) + base
        src += [s, ids, dummy]
        dst += [d, dummy, ids]
    n_total = n_graphs * nodes
    stone = torch.rand(n_total, generator=gen) < 0.15
    legal = torch.rand(n_total, generator=gen) < 0.5
    dummy_rows = torch.arange(n_graphs) * nodes + real
    stone[dummy_rows] = False
    legal[dummy_rows] = False
    legal &= ~stone
    batch = {
        "x": torch.randn(n_total, 11, generator=gen),
        "edge_index": torch.stack((torch.cat(src), torch.cat(dst))).long(),
        "edge_attr": torch.randn(sum(t.numel() for t in src), 5, generator=gen),
        "legal_index": legal.nonzero().flatten(),
        "stone_mask": stone,
        "node_offsets": torch.arange(n_graphs + 1) * nodes,
    }
    return {k: v.to(device) for k, v in batch.items()}
