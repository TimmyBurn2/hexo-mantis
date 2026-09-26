"""Production GNN net — GINE representation, policy head, dist65 pooled value head.
`forward_single` deliberately does NOT delegate to `forward_batch`: batched segment-pooling
changes accumulation order by ~5e-7.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict, cast

import torch
import torch.nn as nn
from torch import Tensor

from mantis.model.arch import GnnArch
from mantis.model.dist65 import N_VALUE_BINS, decode_binned_value
from mantis.model.gine import PolicyHead, RepresentationNetwork


class GnnDist65ValueHead(nn.Module):
    """Pooled MLP -> 65 bin logits -> decoded scalar; pool-agnostic, pooling happens above."""

    def __init__(self, in_dim: int, hidden: int = 32, n_bins: int = N_VALUE_BINS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2_bins = nn.Linear(hidden, n_bins)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """Map pooled (..., in_dim) to value (..., 1) in [-1, 1] and (..., n_bins) logits."""
        h = self.relu(self.fc1(pooled))
        bin_logits = self.fc2_bins(h)
        value = decode_binned_value(bin_logits)
        return value, bin_logits


def _node_offsets_to_batch_vec(node_offsets: Tensor, n_total: int) -> Tensor:
    """Turn a (B+1,) i64 ptr array into an (N,) i64 graph-id per node, sized from the host."""
    # `output_size` is the host-known N: without it `repeat_interleave` syncs to size its output.
    counts = node_offsets[1:] - node_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(node_offsets.shape[0] - 1, device=node_offsets.device, dtype=torch.long),
        counts,
        output_size=n_total,
    )


class _SegmentSums(torch.autograd.Function):
    """`segment_reduce`'s sum with a backward that keeps only the offsets, where autograd's keeps the input and output."""

    @staticmethod
    def forward(ctx: Any, values: Tensor, offsets: Tensor) -> Tensor:
        ctx.save_for_backward(offsets)
        ctx.rows = values.shape[0]
        return torch.segment_reduce(values.float(), "sum", offsets=offsets, axis=0).to(values.dtype)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, None]:
        (offsets,) = ctx.saved_tensors
        segments = torch.arange(offsets.shape[0] - 1, device=offsets.device)
        rows = torch.repeat_interleave(segments, offsets.diff(), output_size=ctx.rows)
        return grads[0].float().index_select(0, rows).to(grads[0].dtype), None


def segment_sums(values: Tensor, offsets: Tensor) -> Tensor:
    """Fixed-order fp32 sums along dim 0 of the `[B+1]` CSR segments; offsets because `lengths=` syncs, fp32 because it sums in the input dtype."""
    return cast(Tensor, _SegmentSums.apply(values, offsets))


def segment_mean_with_fallback(emb: Tensor, mask: Tensor, node_offsets: Tensor) -> Tensor:
    """Per-graph mean over `mask`-selected nodes, falling back to ALL nodes where none.

    Args:
        emb:        (N, D) node embeddings (block-diagonal batch).
        mask:       (N,) bool — the preferred subset (stone nodes).
        node_offsets: (B+1,) long — the CSR pointer of each graph's contiguous node rows.
    Returns:
        (num_graphs, D) pooled vectors.
    """
    dtype = emb.dtype
    mask_f = mask.to(dtype)

    masked_sums = segment_sums(emb * mask_f.unsqueeze(-1), node_offsets)
    masked_counts = segment_sums(mask_f, node_offsets)
    all_sums = segment_sums(emb, node_offsets)
    all_counts = (node_offsets[1:] - node_offsets[:-1]).to(dtype)

    use_fallback = masked_counts == 0
    denom = torch.where(use_fallback, all_counts.clamp(min=1.0), masked_counts.clamp(min=1.0))
    numer = torch.where(use_fallback.unsqueeze(-1), all_sums, masked_sums)
    return numer / denom.unsqueeze(-1)


class GnnNet(nn.Module):
    """Production GNN: GINE representation, policy head and dist65 pooled value head."""

    def __init__(self, arch: GnnArch) -> None:
        super().__init__()
        self.representation = self.build_representation(arch)
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, arch.policy_hidden)
        # `pooled_width` is the readout seam: a subclass widens the value head here rather
        # than rebuilding it, and the override reads no instance state.
        self.value_head = GnnDist65ValueHead(
            self.pooled_width(head_in), arch.value_hidden, arch.n_value_bins
        )

    @staticmethod
    def build_representation(arch: GnnArch) -> RepresentationNetwork:
        """Build the trunk this arch declares — one of the two seams a graph arch swaps."""
        return RepresentationNetwork(arch.in_dim, arch.hidden, arch.num_layers, arch.edge_dim)

    @staticmethod
    def pooled_width(head_in: int) -> int:
        """Return the width the value head consumes; V1 pools ONE statistic, the mean."""
        return head_in

    def node_embeddings(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Return (N, L*H) node embeddings for a possibly batched, disjoint graph."""
        return self.representation(x, edge_index, edge_attr)

    def forward_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_index: Tensor,
        stone_mask: Tensor,
        node_offsets: Tensor | None = None,
        *,
        trunk: Callable[..., Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a grad-capable forward over a disjoint-union batch of collated graphs.

        Args:
            x:            (N_total, in_dim) node features (all graphs concatenated).
            edge_index:   (2, E_total) int64, per-graph node offsets already applied.
            edge_attr:    (E_total, edge_dim) edge features.
            legal_index:  (Lg,) int64 rows of the legal-move nodes, strictly ascending. A bool
                          mask here FAILS CLOSED, never slowly and never silently.
            stone_mask:   (N_total,) bool — True on stone nodes (for value pooling).
            node_offsets: (B+1,) int64 non-decreasing ptr array; `None` == one graph.
            trunk:        a stand-in for `self.representation` over the same parameters, or `None`.
        Returns:
            policy_logits: (num_legal_total,) per-legal-node logits, in gather order.
            value:        (B, 1) decoded value per graph, in [-1, 1].
            bin_logits:   (B, n_value_bins) raw dist65 bin logits per graph.
        """
        assert legal_index.dtype == torch.long, (
            f"legal_index must be int64 rows (the contract's legal_node_gather), got "
            f"{legal_index.dtype} — a bool mask here is the pre-R284 call shape"
        )
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        n_total = x.shape[0]
        device = x.device
        if node_offsets is None:
            node_offsets = torch.tensor([0, n_total], dtype=torch.long, device=device)

        emb = (self.representation if trunk is None else trunk)(x, edge_index, edge_attr)
        # Sync-free gather: `emb[bool_mask]` runs `aten::nonzero`, which host-syncs on CUDA,
        # while `index_select` knows its length from `legal_index.numel()`. Byte-identical
        # because both are row copies and the wire's gather is strictly ascending.
        legal_emb = emb.index_select(0, legal_index)
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)

        pooled = segment_mean_with_fallback(emb, stone_mask, node_offsets)
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value, bin_logits

    def forward_batch_heads(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_index: Tensor,
        stone_mask: Tensor,
        node_offsets: Tensor | None = None,
        *,
        trunk: Callable[..., Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """`forward_batch`'s three outputs plus the auxiliary policy logits — `None` on a net with one policy head; the trainer's ONE entry, so a head set is a subclass and never a branch."""
        policy_logits, value, bin_logits = self.forward_batch(
            x, edge_index, edge_attr, legal_index, stone_mask, node_offsets, trunk=trunk)
        return policy_logits, value, bin_logits, None

    def policy_heads(self) -> tuple[str, ...]:
        """The attribute names of this net's policy heads, main first."""
        return ("policy_head",)

    @torch.no_grad()
    def forward_single(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        legal_mask: Tensor,
        stone_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the deploy path: ONE graph in, no batch dimension on the outputs.

        Does NOT delegate to `forward_batch` (~5e-7 accumulation drift); keep the two in sync.

        Returns:
            (policy_logits (num_legal,), value (scalar), bin_logits (n_value_bins,)).
        """
        assert legal_mask.dtype == torch.bool, f"legal_mask must be bool, got {legal_mask.dtype}"
        assert stone_mask.dtype == torch.bool, f"stone_mask must be bool, got {stone_mask.dtype}"
        emb = self.representation(x, edge_index, edge_attr)
        legal_emb = emb[legal_mask]
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)
        if stone_mask.any():
            pooled = emb[stone_mask].mean(dim=0)
        else:
            pooled = emb.mean(dim=0)
        # pooled has no batch dim, so only value needs the squeeze to a true scalar.
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value.squeeze(0), bin_logits


class BcTransferReport(TypedDict):
    """The return shape of `load_from_bc`."""

    loaded_keys: list[str]
    reinit_keys: list[str]
    verified_tensors: int


def _reinit_keys(own_keys: Sequence[str], reinit: Sequence[str]) -> list[str]:
    """The tensors `reinit` names by module prefix; a name matching nothing is refused.

    Raises:
        ValueError: a `reinit` entry matches no tensor of the net, or is listed twice.
    """
    if len(set(reinit)) != len(reinit):
        raise ValueError(f"load_from_bc: reinit lists a head twice: {list(reinit)}")
    keys: list[str] = []
    for head in reinit:
        matched = [k for k in own_keys if k == head or k.startswith(head + ".")]
        if not matched:
            raise ValueError(
                f"load_from_bc: reinit names {head!r}, which matches no tensor of the net "
                f"(top-level modules: {sorted({k.split('.')[0] for k in own_keys})})"
            )
        keys.extend(matched)
    return sorted(keys)


def load_from_bc(
    net: GnnNet,
    bc_state_dict: Mapping[str, Tensor],
    *,
    reinit: Sequence[str],
) -> BcTransferReport:
    """Load EVERY tensor of a BC checkpoint onto `net` (strict both ways), then put the heads
    `reinit` names back to the fresh init; every landed tensor is verified `torch.equal`.

    Raises:
        ValueError:   a `reinit` entry matches no tensor, or repeats.
        RuntimeError: a key mismatch between source and net, or a failed landed-verify.
    """
    own_sd = net.state_dict()
    reinit_keys = _reinit_keys(list(own_sd), reinit)
    # A head `reinit` names may be ABSENT from the source: a fresh head the parent never had.
    missing = own_sd.keys() - bc_state_dict.keys() - set(reinit_keys)
    unexpected = bc_state_dict.keys() - own_sd.keys()
    if missing or unexpected:
        raise RuntimeError(
            "load_from_bc: state-dict key mismatch — "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    # Fresh tensors are captured BEFORE the load and the reinit keys are never loaded: a
    # re-initialised head may differ in SHAPE, which `load_state_dict` refuses under any `strict`.
    fresh = {k: own_sd[k].detach().clone() for k in reinit_keys}
    net.load_state_dict({k: v for k, v in bc_state_dict.items() if k not in fresh}, strict=False)

    landed = net.state_dict()
    verified = 0
    for key, source in bc_state_dict.items():
        expected = fresh[key] if key in fresh else source
        if not torch.equal(landed[key], expected.to(device=landed[key].device, dtype=landed[key].dtype)):
            raise RuntimeError(
                f"load_from_bc: landed-verify FAILED for {key!r} (the load did not land this "
                "tensor byte-equal)."
            )
        verified += 1
    return {
        "loaded_keys": sorted(k for k in bc_state_dict if k not in fresh),
        "reinit_keys": reinit_keys,
        "verified_tensors": verified,
    }
