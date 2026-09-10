"""Production GNN net — GINE representation, policy head, dist65 pooled value head.
`forward_single` deliberately does NOT delegate to `forward_batch`: batched segment-pooling
changes accumulation order by ~5e-7.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from mantis.model.arch import GnnArch
from mantis.model.dist65 import N_VALUE_BINS, decode_binned_value
from mantis.model.gine import PolicyHead, RepresentationNetwork

# The prefixes that load byte-compatibly from a BC-prefit checkpoint; `value_head.*` is
# excluded because the dist65 head has a different architecture.
BC_TRANSFER_PREFIXES: tuple[str, ...] = ("representation.", "policy_head.")


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


def _node_offsets_to_batch_vec(node_offsets: Tensor) -> Tensor:
    """Turn a (B+1,) i64 ptr array into an (N,) i64 graph-id per node."""
    counts = node_offsets[1:] - node_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(node_offsets.shape[0] - 1, device=node_offsets.device, dtype=torch.long),
        counts,
    )


def segment_mean_with_fallback(
    emb: Tensor, mask: Tensor, batch_vec: Tensor, num_graphs: int
) -> Tensor:
    """Per-graph mean over `mask`-selected nodes, falling back to ALL nodes where none.

    Args:
        emb:        (N, D) node embeddings (block-diagonal batch).
        mask:       (N,) bool — the preferred subset (stone nodes).
        batch_vec:  (N,) long — graph id per node, in [0, num_graphs).
        num_graphs: B.
    Returns:
        (num_graphs, D) pooled vectors.
    """
    d = emb.shape[1]
    device = emb.device
    dtype = emb.dtype
    mask_f = mask.to(dtype)

    masked_sums = torch.zeros(num_graphs, d, device=device, dtype=dtype)
    masked_sums.index_add_(0, batch_vec, emb * mask_f.unsqueeze(-1))
    masked_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
    masked_counts.index_add_(0, batch_vec, mask_f)

    all_sums = torch.zeros(num_graphs, d, device=device, dtype=dtype)
    all_sums.index_add_(0, batch_vec, emb)
    all_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
    all_counts.index_add_(0, batch_vec, torch.ones_like(mask_f))

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
        num_graphs = node_offsets.shape[0] - 1

        emb = self.representation(x, edge_index, edge_attr)
        # Sync-free gather: `emb[bool_mask]` runs `aten::nonzero`, which host-syncs on CUDA,
        # while `index_select` knows its length from `legal_index.numel()`. Byte-identical
        # because both are row copies and the wire's gather is strictly ascending.
        legal_emb = emb.index_select(0, legal_index)
        policy_logits = self.policy_head.mlp(legal_emb).squeeze(-1)

        batch_vec = _node_offsets_to_batch_vec(node_offsets)
        pooled = segment_mean_with_fallback(emb, stone_mask, batch_vec, num_graphs)
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
    """The return shape of `load_representation_policy_from_bc`."""

    loaded_keys: list[str]
    verified_tensors: int


def load_representation_policy_from_bc(
    net: GnnNet,
    bc_state_dict: Mapping[str, Tensor],
    *,
    prefixes: Sequence[str] = BC_TRANSFER_PREFIXES,
    verify_n: int | None = None,
    seed: int = 0,
) -> BcTransferReport:
    """Load ONLY the transfer-prefix tensors from a BC checkpoint onto `net`.

    STRICT on those prefixes, and a landed-verify pass `allclose`-checks what was transferred,
    guarding a silent key-mismatch drop under `strict=False`.

    Raises:
        RuntimeError: on a key mismatch, or a failed landed-verify.
    """
    own_sd = net.state_dict()
    own_keys_for_prefixes = {k for k in own_sd if k.startswith(tuple(prefixes))}
    src = {k: v for k, v in bc_state_dict.items() if k.startswith(tuple(prefixes))}

    missing = own_keys_for_prefixes - src.keys()
    unexpected = src.keys() - own_keys_for_prefixes
    if missing or unexpected:
        raise RuntimeError(
            "load_representation_policy_from_bc: state-dict key mismatch for "
            f"prefixes={prefixes} — missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    net.load_state_dict(src, strict=False)

    reloaded_sd = net.state_dict()
    rng = random.Random(seed)
    verified = 0
    for prefix in prefixes:
        keys = sorted(k for k in own_keys_for_prefixes if k.startswith(prefix))
        sample = keys if verify_n is None else rng.sample(keys, min(verify_n, len(keys)))
        for k in sample:
            loaded = reloaded_sd[k]
            source = src[k].to(device=loaded.device, dtype=loaded.dtype)
            if not torch.allclose(loaded, source):
                raise RuntimeError(
                    f"load_representation_policy_from_bc: landed-verify FAILED for {k!r} "
                    "(strict=False load did not land this tensor)."
                )
            verified += 1

    return {"loaded_keys": sorted(src.keys()), "verified_tensors": verified}
