"""Production GNN net — GINE representation + policy head + dist65 pooled value head.

Ships the probe-284k class (hidden=128, num_layers=4, JK-cat, edge_dim=5,
in_dim=11). The trunk (`representation` / `policy_head`) is imported sideways from
`.gine` (was an up-import from bots); the value head is the dist65 pooled MLP, and
bins/decode use `.dist65` (was an up-import from training). `forward_single` is the
bit-exact single-graph deploy path and deliberately does NOT delegate to
`forward_batch` (batched segment-pooling changes accumulation order ~5e-7).
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

# The two state-dict prefixes that load byte-compatibly from a BC-prefit
# checkpoint. `value_head.*` is excluded — the probe's value head is unsupervised
# and this module's dist65 head has a different architecture; it is always
# fresh-initialized.
BC_TRANSFER_PREFIXES: tuple[str, ...] = ("representation.", "policy_head.")


class GnnDist65ValueHead(nn.Module):
    """Stone-masked-pooled MLP -> 65 bin logits -> decoded scalar.

    Consumes the GINE JK-cat pooled vector (`in_dim` = `num_layers * hidden`).
    Pooling (stone-mask mean, all-nodes fallback) happens in `GnnNet` / the free
    functions below; this head is pool-agnostic, taking a `(B, in_dim)` (or
    `(in_dim,)` for a single graph) pooled vector.
    """

    def __init__(self, in_dim: int, hidden: int = 32, n_bins: int = N_VALUE_BINS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2_bins = nn.Linear(hidden, n_bins)

    def forward(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """pooled: (..., in_dim) -> (value (..., 1) in [-1,1], bin_logits (..., n_bins))."""
        h = self.relu(self.fc1(pooled))
        bin_logits = self.fc2_bins(h)
        value = decode_binned_value(bin_logits)
        return value, bin_logits


def _node_offsets_to_batch_vec(node_offsets: Tensor) -> Tensor:
    """(B+1,) i64 ptr array -> (N,) i64 graph-id per node (torch.repeat_interleave)."""
    counts = node_offsets[1:] - node_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(node_offsets.shape[0] - 1, device=node_offsets.device, dtype=torch.long),
        counts,
    )


def segment_mean_with_fallback(
    emb: Tensor, mask: Tensor, batch_vec: Tensor, num_graphs: int
) -> Tensor:
    """Per-graph mean over `mask`-selected nodes; falls back to ALL nodes for any
    graph with zero masked nodes. Batched generalization of `emb[stone_mask].mean(dim=0)`
    else `emb.mean(dim=0)`.

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
    """Production HeXONet-equivalent GNN: GINE representation + policy head + dist65
    pooled value head (probe-284k class). Constructed from a declared `GnnArch`."""

    def __init__(self, arch: GnnArch) -> None:
        super().__init__()
        self.representation = self.build_representation(arch)
        head_in = self.representation.output_dim
        self.policy_head = PolicyHead(head_in, arch.policy_hidden)
        # `pooled_width` is the readout seam: a subclass whose readout concatenates more than
        # one statistic widens the value head here rather than rebuilding it afterwards, which
        # would leave a wrongly-shaped module constructed for one moment and in the state dict
        # if construction then raised. Called from `__init__` deliberately, and it reads no
        # instance state, so the override is a pure function of `head_in`.
        self.value_head = GnnDist65ValueHead(
            self.pooled_width(head_in), arch.value_hidden, arch.n_value_bins
        )

    @staticmethod
    def build_representation(arch: GnnArch) -> RepresentationNetwork:
        """The trunk this arch declares. The second of the two seams a graph arch swaps."""
        return RepresentationNetwork(arch.in_dim, arch.hidden, arch.num_layers, arch.edge_dim)

    @staticmethod
    def pooled_width(head_in: int) -> int:
        """Width of the vector the value head consumes. V1 pools ONE statistic (the mean)."""
        return head_in

    def node_embeddings(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """(N, L*H) node embeddings for a (possibly batched/disjoint) graph."""
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
        """Grad-capable forward over a disjoint-union batch of graphs (block-diagonal
        contract shape, already collated into tensors).

        Args:
            x:            (N_total, in_dim) node features (all graphs concatenated).
            edge_index:   (2, E_total) int64, per-graph node offsets already applied.
            edge_attr:    (E_total, edge_dim) edge features.
            legal_index:  (Lg,) int64 — the contract's `legal_node_gather`: the ROWS of the
                          legal-move (empty) nodes, strictly ascending (wire contract check
                          13). This is the gather the ragged output is paired by. Passing the
                          `(N_total,) bool` mask here FAILS CLOSED — `AssertionError` from the
                          dtype guard below, or (under `python -O`, where asserts vanish) a
                          `RuntimeError` from `index_select`. Never a slow path, and never a
                          silent one.
            stone_mask:   (N_total,) bool — True on stone nodes (for value pooling).
            node_offsets: (B+1,) int64 non-decreasing ptr array, `[0]=0`, `[B]=N_total`.
                          `None` == single graph (B=1): treated as `[0, N_total]`.
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
        # SYNC-FREE GATHER (R284(b), P-MASK). `emb[bool_mask]` decomposes into
        # `aten::nonzero` + `aten::index`, and `nonzero` must report a data-dependent output
        # length to the host — torch's own doc: "When input is on CUDA, torch.nonzero() causes
        # host-device synchronization." In a fixed-worker serve loop that sync is pure latency:
        # the policy/value kernels cannot enqueue until the trunk drains. `index_select` knows
        # its output length from `legal_index.numel()` and needs no round trip.
        # BYTE-IDENTICAL, not approximately so: both forms are pure row copies and the wire's
        # gather is strictly ascending (check 13), which is exactly when mask order and gather
        # order coincide. Pinned by tests/model/test_pmask_gather_parity.py, mutation included.
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
        """Deploy path: ONE graph in, no batch dimension on the outputs.

        Returns (policy_logits_over_legal_nodes (num_legal,), value (scalar Tensor),
        bin_logits (n_value_bins,)). Deliberately does NOT delegate to `forward_batch`:
        routing through the batched segment-pooling changes accumulation order
        (~5e-7 drift). Keep in sync with forward_batch semantics.
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
        # pooled is (in_dim,) -- no batch dim -- so value_head returns value (1,)
        # and bin_logits (n_value_bins,) already unbatched; only value needs the
        # squeeze to a true scalar.
        value, bin_logits = self.value_head(pooled)
        return policy_logits, value.squeeze(0), bin_logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BcTransferReport(TypedDict):
    """The docstring-stated return shape of `load_representation_policy_from_bc`."""

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
    """Load ONLY `representation.*` / `policy_head.*` tensors from a BC checkpoint
    state dict onto `net`; `value_head.*` is left fresh-initialized.

    STRICT on the two transfer prefixes: every `net` key under `prefixes` must be
    present in `bc_state_dict` and vice versa, else raises. After load, a
    landed-verify pass checks the transferred tensors (`verify_n=None` → all) with
    `torch.allclose` against the source — guarding a silent key-mismatch drop under
    `strict=False`.

    Returns `{"loaded_keys": [...], "verified_tensors": int}`. Raises `RuntimeError`
    on key mismatch or a failed landed-verify.
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
