"""Self-contained pure-PyTorch GINE axis-graph net definition (no torch_geometric).

This is the graph net DEFINITION — `_GINEConv`, `RepresentationNetwork`,
`PolicyHead`, `ValueHead`, `HeXONet`. It lives in `model/` (the definition's
home); `gnn.py` imports `RepresentationNetwork`/`PolicyHead` sideways from here,
and a downstream bot wraps `HeXONet` (bots → model, DAG-correct). This inverts
the old latent model → bots import edge.

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

Attribution: representation/policy/value modules follow the public
SootyOwl/hexo-strix HeXONet forward pass (MIT), reimplemented pure-torch.
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

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
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


class ValueHead(nn.Module):
    def __init__(self, in_dim: int, value_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )


class HeXONet(nn.Module):
    """HeXONet — representation + policy + scalar-tanh value.

    forward() consumes a single graph and returns
    `(policy_logits_over_legal_nodes, value_scalar)`:
      - policy: PolicyHead MLP over the LEGAL-move node embeddings (legal-node order).
      - value:  ValueHead MLP over the MEAN of STONE-node embeddings, tanh scalar.
    """

    EDGE_DIM = 5

    def __init__(self, in_dim: int = 11, hidden: int = 128, num_layers: int = 4,
                 policy_hidden: int = 128, value_hidden: int = 32) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(in_dim, hidden, num_layers, self.EDGE_DIM)
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, policy_hidden)
        self.value_head = ValueHead(head_in, value_hidden)

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        emb = self.representation(x, edge_index, edge_attr)   # (N, L*H)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)  # (num_legal,)
        if stone_mask.any():
            pooled = emb[stone_mask].mean(dim=0)
        else:
            pooled = emb.mean(dim=0)
        value = self.value_head.mlp(pooled).squeeze(-1)       # scalar
        return policy_logits, value
