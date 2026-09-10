"""`GnnNetV2` — the `gnn_axis_v1` wire, two model-side mechanisms swapped in: a
`concat(stone-masked mean, max over REAL nodes)` readout, because the win condition is a max and
a mean over a growing node set dilutes it; and a degree-normalized dummy aggregation, so
`‖agg[dummy]‖` stops scaling with the real-node count while real-node GINE sums stay UNTOUCHED.

V2 consumes `gnn_axis_v1` and adds no registry row, config key or arch field. Two wire facts
stay DERIVED: the dummy is the node that is NEITHER legal NOR a stone, and its in-degree is
counted off the edge list. `RepresentationNetworkV2` keeps V1's module names and shapes because
`load_representation_policy_from_bc` raises on ANY key mismatch.
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

    Masking is by SENTINEL, not `emb[mask]`, which decomposes into `aten::nonzero` and forces a
    host-device sync to report its data-dependent length.
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

    # `scatter_reduce_`, not the beta `index_reduce_`, which warns on every process. `expand` is
    # a stride-0 VIEW, so the widened index allocates nothing where `repeat` would allocate (N, D).
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
    """V1's trunk, parameter-identical — same module names and shapes, so the BC warmstart still
    loads — with the dummy node's aggregation degree-normalized."""

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        normalize_mask: Tensor | None = None,
    ) -> Tensor:
        """`normalize_mask` is `(N,)` bool, True on nodes whose aggregation is degree-normalized;
        `None` reproduces V1's forward exactly.
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
    """The V2 graph net: mean+max readout over a degree-normalized-dummy trunk."""

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

    def node_embeddings(  # type: ignore[override]
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
        """The deploy twin, non-delegating: the MEAN half carries a ~5e-7 accumulation-order
        drift and the MAX half adds none.

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
