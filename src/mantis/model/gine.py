"""Self-contained pure-PyTorch GINE axis-graph net definition (no torch_geometric).

`_GINEConv`, `RepresentationNetwork` and `PolicyHead` live here; `gnn.py`/`gnn_v2.py` import
sideways. `HeXONet` and `ValueHead` are BURIED, goldens frozen at
`tests/fixtures/model_graves/hexonet_grave_v1.json` so a resurrection can be proved
bit-identical, and the conformance suite's T11 section holds the grave dead. State-dict keys
mirror the reference GINEConv exactly, so a strix checkpoint loads `strict=True` and a BC
checkpoint loads under the `representation.`/`policy_head.` prefixes. Architecture pinned by the
checkpoint's model_config: hidden_dim=128, num_layers=4, conv_type=gine, pre_norm=True,
use_jk=True, jk_mode=cat (heads see L*hidden=512), policy_hidden=128, value_hidden=32,
graph_type=axis; node dim 11, edge_attr dim 5.

Attribution: the representation and policy modules follow the public SootyOwl/hexo-strix HeXONet
forward pass (MIT), reimplemented pure-torch. Licence-required, and it STAYS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

#: Edges per fp32 chunk of the sum: bounds the fp32 copy of the messages the aggregation reads.
_AGGREGATE_CHUNK_EDGES = 1 << 18


@torch.library.custom_op("mantis::gine_aggregate", mutates_args=())
def gine_aggregate(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
    """Per-node fp32 message sum, divided in fp32, rounded once; an op so Inductor cannot lower it to atomics, `index_add_` on CPU where `index_put_` races."""
    agg = torch.zeros((n, msg.shape[1]), dtype=torch.float32, device=msg.device)
    for start in range(0, msg.shape[0], _AGGREGATE_CHUNK_EDGES):
        stop = start + _AGGREGATE_CHUNK_EDGES
        if msg.is_cuda:
            agg.index_put_((dst[start:stop],), msg[start:stop].float(), accumulate=True)
        else:
            agg.index_add_(0, dst[start:stop], msg[start:stop].float())
    if divisor is not None:
        agg = agg / divisor.float()
    return agg.to(msg.dtype)


@gine_aggregate.register_fake
def _gine_aggregate_fake(msg: Tensor, dst: Tensor, n: int, divisor: Tensor | None) -> Tensor:
    return msg.new_empty((n, msg.shape[1]))


def _gine_aggregate_setup(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: Tensor) -> None:
    _msg, dst, _n, divisor = inputs
    ctx.save_for_backward(dst, divisor)


def _gine_aggregate_backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple:
    dst, divisor = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    g = grad if divisor is None else (grad.float() / divisor.float()).to(grad.dtype)
    return g.index_select(0, dst), None, None, None


gine_aggregate.register_autograd(_gine_aggregate_backward, setup_context=_gine_aggregate_setup)


@torch.library.custom_op("mantis::gine_gather", mutates_args=())
def gine_gather(x: Tensor, src: Tensor) -> Tensor:
    """`x[src]`; its gradient is a graph aggregation over `src`, so it sums through `gine_aggregate`, never atomics."""
    return x.index_select(0, src)


@gine_gather.register_fake
def _gine_gather_fake(x: Tensor, src: Tensor) -> Tensor:
    return x.new_empty((src.shape[0], x.shape[1]))


def _gine_gather_setup(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: Tensor) -> None:
    x, src = inputs
    ctx.save_for_backward(src)
    ctx.n = x.shape[0]  # pyright: ignore[reportAttributeAccessIssue]


def _gine_gather_backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple:
    (src,) = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    return gine_aggregate(grad, src, ctx.n, None), None  # pyright: ignore[reportAttributeAccessIssue]


gine_gather.register_autograd(_gine_gather_backward, setup_context=_gine_gather_setup)


class _GINEConv(nn.Module):
    """Plain-torch GINEConv (sum aggregation, edge-feature injection), state-dict keys mirroring
    the reference GINEConv. `edge_in` is the width of the edge tensor handed to `forward` — the
    representation applies `edge_proj` (5→128) ONCE, so each conv's own `lin` is Linear(128→128),
    matching the checkpoint shapes."""

    eps: Tensor

    def __init__(self, hidden: int, edge_in: int) -> None:
        super().__init__()
        self.register_buffer("eps", torch.zeros(1))
        self.nn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.lin = nn.Linear(edge_in, hidden)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        agg_divisor: Tensor | None = None,
    ) -> Tensor:
        """`agg_divisor` is `(N, 1)` and divides the aggregated messages per node when given.
        The one aggregation authority stays here because the block is dtype-critical under
        autocast and a second copy is a second place for that care to drift."""
        n = x.shape[0]
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            # The edge projection is evaluated FIRST and the node tensor aligned to ITS dtype
            # BEFORE the gather: `index_select` is dtype-PRESERVING, so gathering the fp32
            # pre-norm tensor under bf16 autocast materialises an [E, H] copy at 2x the width,
            # on the one tensor that scales with E (a single 8.94 GiB request killed a run).
            e = self.lin(edge_attr)
            msg = (gine_gather(x.to(e.dtype), src) + e).relu()
            agg = gine_aggregate(msg, dst, n, agg_divisor)
        else:
            agg = x.new_zeros((n, x.shape[1]))
            if agg_divisor is not None:
                agg = agg / agg_divisor.to(agg.dtype)
        out = agg + (1.0 + self.eps) * x
        return self.nn(out)


class RepresentationNetwork(nn.Module):
    """GINE axis-graph representation: pre-norm residual blocks + JK-cat, output `L * hidden`."""

    def __init__(self, in_dim: int = 11, hidden: int = 128, num_layers: int = 4,
                 edge_dim: int = 5) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        # Each conv's edge input is the ALREADY-projected edge tensor: Linear(hidden->hidden).
        self.convs = nn.ModuleList([_GINEConv(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.output_dim = num_layers * hidden
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.input_proj(x)  # (N, H)
        # Projected ONCE and reused; each layer's own `lin` re-projects THAT tensor (H->H).
        projected_edge_attr = self.edge_proj(edge_attr)
        hs: list[Tensor] = []
        for conv, norm in zip(self.convs, self.norms, strict=False):
            residual = x
            xn = norm(x)                                  # pre-norm
            xc = conv(xn, edge_index, projected_edge_attr)
            x = xc + residual
            x = self.activation(x)
            hs.append(x)
        # jk_mode="cat": final_norm(H) applied to EACH h_i, then concat.
        hs = [self.final_norm(h) for h in hs]
        return torch.cat(hs, dim=-1)                      # (N, L*H)


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, policy_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )
