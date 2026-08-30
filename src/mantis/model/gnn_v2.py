"""`GnnNetV2` — the `gnn_axis_v1` wire, two model-side mechanisms swapped in.

WHAT V2 IS, and it is exactly two components (WP-AXIS2 candidates A and C(i), model-side half):

  * **Readout (A).** `concat(stone-masked mean, max over REAL nodes)`, so the value head sees a
    max statistic. The win condition is a max and the v1 readout is a mean; a mean over a
    growing node set dilutes a single strong signal, which is what GNN-1 names.
  * **Dummy aggregation (C(i)).** The virtual node's incoming aggregation is degree-normalized,
    so `‖agg[dummy]‖` stops scaling with the real-node count. Real-node GINE sums — and the
    count signal they carry — are UNTOUCHED, which is what keeps this from becoming the global
    mean aggregation that would destroy GINE's injectivity premise.

WHAT V2 IS NOT. No wire change: V2 consumes `gnn_axis_v1` and adds NO registry row, so no
`node_feat_dim`, no `contract_version`, no builder question is opened. No config key. No arch
field naming a property V2 claims. No training and no strength claim — the behavioral witnesses
this module is built against are forward-only comparisons of FUNCTION FORM at random init, and
F-01 is the standing fence: static probes once passed while self-play collapsed to 0–1 %.

TWO DERIVATIONS THAT MUST STAY DERIVATIONS, both of them wire facts rather than constants:

  * The dummy is the node that is NEITHER legal NOR a stone. There is no `real_mask` on the
    wire — the masks it carries are `legal` and `stone` — and `dummy_idx == n_real` makes the
    dummy the last row of each graph, so a literal `N-1` would work today and would be a
    code-side constant standing in for a wire fact the moment the builder reorders.
  * The dummy's in-degree is counted off the edge list, never taken as `n_real`.

BC/STRIX WARMSTART SURVIVES, deliberately. `BC_TRANSFER_PREFIXES` is
`("representation.", "policy_head.")` and `load_representation_policy_from_bc` raises on ANY
key mismatch, so a trunk change that renamed or re-shaped a `representation.*` tensor would
destroy the warmstart path. `RepresentationNetworkV2` therefore holds the same modules under the
same names at the same shapes — C(i) lives entirely in the forward, not in the parameters. This
is the property that separates A/C(i) from candidate B, which cannot preserve it.
"""
from __future__ import annotations

import torch
from torch import Tensor

from mantis.model.arch import GnnArchV2
from mantis.model.gine import RepresentationNetwork
from mantis.model.gnn import GnnNet, _node_offsets_to_batch_vec, segment_mean_with_fallback

__all__ = ["GnnNetV2", "RepresentationNetworkV2", "segment_max_with_fallback"]


def segment_max_with_fallback(
    emb: Tensor, mask: Tensor, batch_vec: Tensor, num_graphs: int
) -> Tensor:
    """Per-graph max over `mask`-selected nodes; falls back to ALL nodes where none are masked.

    Deliberately the same shape as `segment_mean_with_fallback`, including the fallback, so the
    two halves of the V2 readout degenerate the same way on the same graphs rather than one of
    them producing a sentinel the other never would.

    MASK BY SENTINEL, NOT BY `nonzero`. `emb[mask]` decomposes into `aten::nonzero` +
    `aten::index` and `nonzero` forces a host-device sync to report its data-dependent length —
    the same cost `forward_batch`'s legal gather was rewritten to avoid. Filling the unmasked
    rows with the dtype's minimum keeps the reduction over a fixed shape.

    Args:
        emb: `(N, D)` node embeddings (block-diagonal batch).
        mask: `(N,)` bool — the preferred subset.
        batch_vec: `(N,)` long — graph id per node, in `[0, num_graphs)`.
        num_graphs: B.

    Returns:
        `(num_graphs, D)` per-graph maxima.
    """
    d = emb.shape[1]
    device = emb.device
    dtype = emb.dtype
    floor = torch.finfo(dtype).min

    # `scatter_reduce_`, not `index_reduce_`: the latter is a beta API that warns on every
    # process, and this runs on the trainer's own forward. The index it needs is `batch_vec`
    # widened to `emb`'s shape, and `expand` is a stride-0 VIEW — it allocates nothing, which
    # matters because the naive `repeat` here would be an `(N, D)` int64 tensor on the one
    # axis that scales with the batch.
    scatter_index = batch_vec.unsqueeze(-1).expand(-1, d)

    masked = emb.masked_fill(~mask.unsqueeze(-1), floor)
    masked_max = torch.full((num_graphs, d), floor, device=device, dtype=dtype)
    masked_max.scatter_reduce_(0, scatter_index, masked, "amax", include_self=False)

    all_max = torch.full((num_graphs, d), floor, device=device, dtype=dtype)
    all_max.scatter_reduce_(0, scatter_index, emb, "amax", include_self=False)

    counts = torch.zeros(num_graphs, device=device, dtype=dtype)
    counts.index_add_(0, batch_vec, mask.to(dtype))
    return torch.where((counts == 0).unsqueeze(-1), all_max, masked_max)


class RepresentationNetworkV2(RepresentationNetwork):
    """V1's trunk, parameter-identical, with the dummy node's aggregation degree-normalized.

    Same modules under the same names at the same shapes — see the module docstring on why that
    is load-bearing rather than incidental. The whole of C(i) is the divisor computed here and
    handed to each conv.
    """

    def forward(  # type: ignore[override] — V2's trunk needs the mask V1's has no use for
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        normalize_mask: Tensor | None = None,
    ) -> Tensor:
        """`normalize_mask` is `(N,)` bool, True on nodes whose aggregation is degree-normalized.

        `None` reproduces V1's forward exactly, which is what makes W-C2 constructible: with the
        dummy's edges removed there is nothing to normalize and the two nets must agree exactly.
        """
        divisor = None
        if normalize_mask is not None and edge_index.shape[1] > 0:
            dst = edge_index[1]
            in_degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            in_degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            divisor = torch.where(
                normalize_mask, in_degree.clamp(min=1.0), torch.ones_like(in_degree)
            ).unsqueeze(-1)

        x = self.input_proj(x)
        projected_edge_attr = self.edge_proj(edge_attr)
        hs: list[Tensor] = []
        for conv, norm in zip(self.convs, self.norms, strict=False):
            residual = x
            xn = norm(x)
            xc = conv(xn, edge_index, projected_edge_attr, divisor)
            x = xc + residual
            x = self.activation(x)
            hs.append(x)
        hs = [self.final_norm(h) for h in hs]
        return torch.cat(hs, dim=-1)


class GnnNetV2(GnnNet):
    """The V2 graph net: mean+max readout over a degree-normalized-dummy trunk.

    Constructed from a declared `GnnArchV2`. The two overrides below ARE the arch: the trunk it
    builds and the width its readout pools. Nothing else differs from `GnnNet`, which is the
    seam's own claim — adding a model kind is a component swap behind the contract.
    """

    def __init__(self, arch: GnnArchV2) -> None:
        super().__init__(arch)  # type: ignore[arg-type] — the field sets are identical by design

    @staticmethod
    def build_representation(arch: GnnArchV2) -> RepresentationNetwork:  # type: ignore[override]
        return RepresentationNetworkV2(arch.in_dim, arch.hidden, arch.num_layers, arch.edge_dim)

    @staticmethod
    def pooled_width(head_in: int) -> int:
        """V2 pools TWO statistics — the stone-masked mean and the max over real nodes."""
        return 2 * head_in

    @staticmethod
    def real_mask_from_batch(stone_mask: Tensor, legal_index: Tensor) -> Tensor:
        """`real = stone | legal`, and the dummy is what is left. Derived, never `N-1`."""
        real = stone_mask.clone()
        real[legal_index] = True
        return real

    def node_embeddings(  # type: ignore[override] — V2's trunk takes the normalization mask
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        normalize_mask: Tensor | None = None,
    ) -> Tensor:
        return self.representation(x, edge_index, edge_attr, normalize_mask)

    def forward_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_index: Tensor,
        stone_mask: Tensor,
        node_offsets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """V1's contract shape and V1's returns; the readout and the trunk are V2's.

        Args:
            x: `(N_total, in_dim)` node features.
            edge_index: `(2, E_total)` int64, per-graph offsets already applied.
            edge_attr: `(E_total, edge_dim)` edge features.
            legal_index: `(Lg,)` int64 rows of the legal nodes, strictly ascending.
            stone_mask: `(N_total,)` bool, True on stone nodes.
            node_offsets: `(B+1,)` int64 ptr array; `None` means one graph.

        Returns:
            `(policy_logits, value, bin_logits)`, as `GnnNet.forward_batch`.
        """
        assert legal_index.dtype == torch.long, (
            f"legal_index must be int64 rows (the contract's legal_node_gather), got "
            f"{legal_index.dtype} — a bool mask here is the pre-R284 call shape"
        )
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        n_total = x.shape[0]
        if node_offsets is None:
            node_offsets = torch.tensor([0, n_total], dtype=torch.long, device=x.device)
        num_graphs = node_offsets.shape[0] - 1

        real_mask = self.real_mask_from_batch(stone_mask, legal_index)
        emb = self.representation(x, edge_index, edge_attr, ~real_mask)
        legal_emb = emb.index_select(0, legal_index)
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)

        batch_vec = _node_offsets_to_batch_vec(node_offsets)
        pooled = torch.cat(
            (
                segment_mean_with_fallback(emb, stone_mask, batch_vec, num_graphs),
                segment_max_with_fallback(emb, real_mask, batch_vec, num_graphs),
            ),
            dim=-1,
        )
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value, bin_logits

    @torch.no_grad()
    def forward_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The deploy twin. Non-delegating for V1's reason — the MEAN half still carries the
        ~5e-7 accumulation-order drift that made the pair separate; the MAX half adds no drift
        term, because a maximum is order-independent in exact arithmetic and under IEEE
        `maximum` for non-NaN inputs.

        Args:
            x: `(N, in_dim)` node features for ONE graph.
            edge_index: `(2, E)` int64.
            edge_attr: `(E, edge_dim)` edge features.
            legal_mask: `(N,)` bool, True on legal nodes.
            stone_mask: `(N,)` bool, True on stone nodes.

        Returns:
            `(policy_logits (num_legal,), value (scalar), bin_logits (n_value_bins,))`.
        """
        assert legal_mask.dtype == torch.bool, f"legal_mask must be bool, got {legal_mask.dtype}"
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        real_mask = legal_mask | stone_mask
        emb = self.representation(x, edge_index, edge_attr, ~real_mask)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)
        mean_part = emb[stone_mask].mean(dim=0) if stone_mask.any() else emb.mean(dim=0)
        max_part = (
            emb[real_mask].amax(dim=0) if real_mask.any() else emb.amax(dim=0)
        )
        value, bin_logits = self.value_head(torch.cat((mean_part, max_part), dim=-1))
        return policy_logits, value.squeeze(0), bin_logits
