"""Self-contained pure-PyTorch GINE axis-graph net definition (no torch_geometric).

This is the graph net DEFINITION — `_GINEConv`, `RepresentationNetwork`,
`PolicyHead`. It lives in `model/` (the definition's home); `gnn.py` and
`gnn_v2.py` import `RepresentationNetwork`/`PolicyHead` sideways from here.

GRAVE (R322(d), SEAM-B2 Leg 2): `HeXONet` and `ValueHead` were BURIED here on
2026-08-30. The structural reachability census found `HeXONet` in no `build_net`
dispatch branch and referenced nowhere in `src/`, `tools/` or `tests/`;
`ValueHead` was reachable only from it. This docstring used to say "a downstream
bot wraps `HeXONet` (bots → model, DAG-correct)" — that bot does not exist in
`src/mantis/bots/` and the sentence is corrected rather than kept. Their goldens
are frozen at `tests/fixtures/model_graves/hexonet_grave_v1.json` (state-dict
shapes, parameter count, and a seeded forward digest), so a resurrection can be
proved bit-identical to what was buried; the grave is held dead by the
conformance suite's T11 section.

The message passing is a small sum-scatter, reimplemented in plain torch:
    m_{i<-j} = ReLU(x_j + lin(e_{j->i}))
    out_i    = MLP( sum_{j->i} m_{i<-j} + (1 + eps) * x_i )
State-dict keys mirror the reference GINEConv exactly so a strix checkpoint loads
`strict=True` and a BC checkpoint loads under the `representation.`/`policy_head.`
transfer prefixes.

Architecture pinned by the checkpoint's model_config: hidden_dim=128, num_layers=4,
conv_type=gine, pre_norm=True, use_jk=True, jk_mode=cat (heads see L*hidden=512),
policy_hidden=128, value_hidden=32, graph_type=axis; node feature dim = 11,
edge_attr dim = 5.

Attribution: the representation and policy modules follow the public
SootyOwl/hexo-strix HeXONet forward pass (MIT), reimplemented pure-torch. The
attribution is licence-required and STAYS: the modules it covers are still here,
and only the value module and the top-level net were buried.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _GINEConv(nn.Module):
    """Plain-torch GINEConv (sum aggregation, edge-feature injection).

    State-dict keys mirror the reference GINEConv under this module:
      `eps` (buffer, shape (1,)), `nn.0.*` / `nn.2.*` (the MLP), `lin.*` (edge Linear).

    `edge_in` is the width of the edge tensor handed to `forward`: the
    representation applies its `edge_proj` (5→128) ONCE and hands the projected
    (128-dim) tensor to every layer, so each conv's own `lin` is Linear(128→128),
    matching the checkpoint shapes.
    """

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

        The one aggregation authority stays here rather than being copied into an arch that
        wants a degree-normalized node: this block is dtype-critical under autocast (LAW-06)
        and a second copy of it is a second place for that care to drift. `None` — every v1
        call — skips the division entirely, so the v1 arithmetic is unchanged.
        """
        n = x.shape[0]
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            # message from src -> dst using that edge's attr. The edge projection is
            # evaluated FIRST and the node tensor aligned to ITS dtype BEFORE the gather:
            # under autocast the projection is the regime witness (bf16, LAW-06), while
            # `index_select` is dtype-PRESERVING and would otherwise materialise an
            # [E, H] copy of the fp32 pre-norm tensor -- 2x the width the bf16 regime
            # implies, on the one tensor that scales with E (WP12-R D-FIX F1, R179;
            # CARD-RUN5-GPU-OOM died on a single 8.94 GiB request here). `Tensor.to`
            # returns `self` when the dtype already matches, so with autocast OFF this
            # is an exact no-op and the fp32 deploy arm is bit-unchanged.
            e = self.lin(edge_attr)
            xs = x.to(e.dtype)
            msg = (xs.index_select(0, src) + e).relu()
            # `agg` is built from `xs`: `index_add_` requires matching dtypes.
            agg = xs.new_zeros((n, xs.shape[1]))
            agg.index_add_(0, dst, msg)
        else:
            agg = x.new_zeros((n, x.shape[1]))
        if agg_divisor is not None:
            agg = agg / agg_divisor.to(agg.dtype)
        out = agg + (1.0 + self.eps) * x
        return self.nn(out)


class RepresentationNetwork(nn.Module):
    """GINE axis-graph representation with pre-norm residual blocks + JK-cat.

    Config: conv_type=gine, pre_norm=True, use_jk=True, jk_mode=cat. Output dim is
    `num_layers * hidden` (the JK-cat concatenation the heads consume).
    """

    def __init__(self, in_dim: int = 11, hidden: int = 128, num_layers: int = 4,
                 edge_dim: int = 5) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        # Each conv's edge input is the ALREADY-projected (hidden-dim) edge tensor,
        # so conv.lin is Linear(hidden->hidden).
        self.convs = nn.ModuleList([_GINEConv(hidden, hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.output_dim = num_layers * hidden
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.input_proj(x)  # (N, H)
        # Project edge_attr ONCE and reuse the projected tensor across layers; each
        # GINE layer's own `lin` then re-projects THAT (H-dim) tensor (H->H).
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
