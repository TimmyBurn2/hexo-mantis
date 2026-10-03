"""Self-contained GINE axis-graph net definition (no torch_geometric; one Triton op on CUDA).

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

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

#: Table GEMMs run on at least this many rows, so they skip the small-M kernels that split a reduction (bit-equal to
#: the per-edge forward on sm_86, pinned by the coded parity tests); codes in [V, rows) would read a bias-only row.
_TABLE_ROWS = 1024


def csr_edges(edge_index: Tensor, edge_attr: Tensor, n: int) -> tuple[Tensor, Tensor, Tensor | None]:
    """On CUDA the edges stably sorted by destination plus the `[n+1]` row pointer the fused sum walks; CPU keeps its order."""
    if not edge_index.is_cuda or edge_index.shape[1] == 0:
        return edge_index, edge_attr, None
    dst = edge_index[1]
    perm = torch.argsort(dst, stable=True)
    # Integer counts (exact under any order), not `searchsorted`, whose Inductor lowering refuses a view.
    counts = torch.zeros(n + 1, dtype=torch.long, device=dst.device).index_add_(0, dst + 1, torch.ones_like(dst))
    return edge_index[:, perm], edge_attr[perm], counts.cumsum(0)


@torch.library.custom_op("mantis::gine_message_sum", mutates_args=())
def gine_message_sum(xs: Tensor, e: Tensor, src: Tensor, dst: Tensor, rowptr: Tensor | None,
                     divisor: Tensor | None, code: Tensor | None = None) -> Tensor:
    """Per-node Σ relu(xs[src] + e) / divisor in at least fp32, rounded once, atomic-free: a fused kernel over `csr_edges` on CUDA, `index_add_` on CPU; with `code`, `e` is a table and edge j reads its row `code[j]`.

    Raises:
        ValueError: CUDA inputs without `csr_edges`' row pointer.
        TypeError: CUDA inputs wider than fp32, which the kernels would narrow.
    """
    if xs.is_cuda:
        from mantis.model import _gine_triton

        if rowptr is None:
            raise ValueError("gine_message_sum on CUDA needs csr_edges' row pointer and destination-sorted edges")
        if xs.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"gine_message_sum's CUDA kernels sum bf16/fp16/fp32 in fp32; {xs.dtype} would narrow")
        return _gine_triton.message_sum(xs, e, src, rowptr, divisor, code)
    acc = torch.promote_types(torch.float32, xs.dtype)
    msg = (xs.index_select(0, src) + (e if code is None else e.index_select(0, code.long()))).relu()
    agg = torch.zeros((xs.shape[0], xs.shape[1]), dtype=acc).index_add_(0, dst, msg.to(acc))
    return (agg if divisor is None else agg / divisor.to(acc)).to(xs.dtype)


@gine_message_sum.register_fake
def _gine_message_sum_fake(xs: Tensor, e: Tensor, src: Tensor, dst: Tensor, rowptr: Tensor | None,
                           divisor: Tensor | None, code: Tensor | None = None) -> Tensor:
    return xs.new_empty(xs.shape)


def _gine_message_sum_setup(ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: Tensor) -> None:
    ctx.save_for_backward(*inputs)


def _gine_message_sum_backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple:
    xs, e, src, dst, rowptr, divisor, code = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    if code is not None:
        raise RuntimeError("gine_message_sum over a coded table has no backward; train on per-edge rows")
    if xs.is_cuda:
        from mantis.model import _gine_triton

        grad_xs, grad_e = _gine_triton.message_grads(grad, xs, e, src, rowptr, divisor)
        return grad_xs, grad_e, None, None, None, None, None
    acc = torch.promote_types(torch.float32, xs.dtype)
    g = grad if divisor is None else (grad.to(acc) / divisor.to(acc)).to(grad.dtype)
    grad_e = torch.where((xs.index_select(0, src) + e) > 0, g.index_select(0, dst), 0).to(e.dtype)
    grad_xs = torch.zeros(xs.shape, dtype=acc).index_add_(0, src, grad_e.to(acc)).to(xs.dtype)
    return grad_xs, grad_e, None, None, None, None, None


gine_message_sum.register_autograd(_gine_message_sum_backward, setup_context=_gine_message_sum_setup)


class _GINEConv(nn.Module):
    """GINEConv over `gine_message_sum`, keys mirroring the reference; `edge_in` is the already-projected edge width (`lin` is 128→128)."""

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
        rowptr: Tensor | None = None,
        edge_code: Tensor | None = None,
    ) -> Tensor:
        """`agg_divisor` `(N, 1)` divides each node's sum; a `rowptr` means the edges come in `csr_edges` order; with `edge_code`, `edge_attr` holds the vocabulary's rows and edge j reads row `edge_code[j]`."""
        n = x.shape[0]
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            if rowptr is None:
                if edge_code is None:
                    edge_index, edge_attr, rowptr = csr_edges(edge_index, edge_attr, n)
                else:
                    edge_index, edge_code, rowptr = csr_edges(edge_index, edge_code, n)
                src, dst = edge_index[0], edge_index[1]
            e = self.lin(edge_attr)
            # `xs` takes `e`'s dtype: the op's per-edge rounding reads it, and a wider gather doubles the [E, H] width.
            agg = gine_message_sum(x.to(e.dtype), e, src, dst, rowptr, agg_divisor, edge_code)
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

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_vocab: Tensor | None = None) -> Tensor:
        """`(N, L*H)`; given `edge_vocab`, `edge_attr` is each edge's code in it. Raises: ValueError — `edge_vocab` under grad (no backward)."""
        edge_index, projected_edge_attr, rowptr, code = self._edge_inputs(x, edge_index, edge_attr, edge_vocab)
        x = self.input_proj(x)  # (N, H)
        hs: list[Tensor] = []
        for i in range(self.num_layers):
            x = self.activation(self._layer(i, x, edge_index, projected_edge_attr, None, rowptr, code) + x)
            hs.append(x)
        # jk_mode="cat": final_norm(H) applied to EACH h_i, then concat.
        hs = [self.final_norm(h) for h in hs]
        return torch.cat(hs, dim=-1)                      # (N, L*H)


    def _edge_inputs(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                     edge_vocab: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """`(edge_index, projected, rowptr, code)` in `csr_edges` order, projected ONCE per edge or, given `edge_vocab`, per vocabulary row."""
        if edge_vocab is None:
            edge_index, edge_attr, rowptr = csr_edges(edge_index, edge_attr, x.shape[0])
            return edge_index, self.edge_proj(edge_attr), rowptr, None
        if torch.is_grad_enabled():
            raise ValueError("coded edges serve only: the vocabulary path has no backward; train on per-edge rows")
        edge_index, code, rowptr = csr_edges(edge_index, edge_attr, x.shape[0])
        pad = edge_vocab.new_zeros(max(0, _TABLE_ROWS - edge_vocab.shape[0]), edge_vocab.shape[1])
        return edge_index, self.edge_proj(torch.cat((edge_vocab, pad))), rowptr, code

    def _layer(self, i: int, x: Tensor, edge_index: Tensor, projected_edge_attr: Tensor,
               divisor: Tensor | None, rowptr: Tensor | None, code: Tensor | None = None) -> Tensor:
        """Layer `i`'s pre-norm conv; in training recomputed in backward, so no layer keeps its [E, H] edge tensor."""
        args = (i, x, edge_index, projected_edge_attr, divisor, rowptr, code)
        if torch.is_grad_enabled() and x.requires_grad:
            # RNG-free region, so no state is stashed for the replay.
            return cast(Tensor, checkpoint(self._conv, *args, use_reentrant=False, preserve_rng_state=False))
        return self._conv(*args)

    def _conv(self, i: int, x: Tensor, edge_index: Tensor, projected_edge_attr: Tensor,
              divisor: Tensor | None, rowptr: Tensor | None, code: Tensor | None = None) -> Tensor:
        return self.convs[i](self.norms[i](x), edge_index, projected_edge_attr, divisor, rowptr, code)


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, policy_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )
