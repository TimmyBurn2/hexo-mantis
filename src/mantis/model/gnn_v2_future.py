"""`GnnNetV2SoftPolicyFuture` — the soft-policy net plus a training-only future-occupancy head over the legal embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mantis.model.arch import GnnArchV2SoftPolicyFuture
from mantis.model.gnn_v2 import GnnNetV2SoftPolicy

#: The head's three classes per legal cell, in the ring's label order: empty, the mover's, the opponent's.
FUTURE_CLASSES = 3
#: The head's starting logits: each side fills about 1.8 % of a radius-8 ring row's legal cells, so it begins at the base rate.
FUTURE_PRIOR_LOGITS = (0.0, -4.0, -4.0)
#: The attribute a net carries the head under: what a warm start's `reinit` names when it starts fresh.
FUTURE_HEAD = "future_head"
#: The lineage tag a stamp carries once a warm start has begun the head fresh.
AUX_FUTURE_LINEAGE = "+aux_future"

__all__ = ["AUX_FUTURE_LINEAGE", "FUTURE_CLASSES", "FUTURE_HEAD", "FUTURE_PRIOR_LOGITS", "FutureOccupancyHead",
           "GnnNetV2SoftPolicyFuture"]


class FutureOccupancyHead(nn.Module):
    """Per legal node, three logits over who holds the cell twenty placements on; quiet at the start (zero last layer)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, FUTURE_CLASSES))
        last = self.mlp[2]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(torch.tensor(FUTURE_PRIOR_LOGITS))

    def forward(self, legal_emb: Tensor) -> Tensor:
        """`(num_legal_total, FUTURE_CLASSES)` logits."""
        return self.mlp(legal_emb)


class GnnNetV2SoftPolicyFuture(GnnNetV2SoftPolicy):
    """The soft-policy net plus `future_head`; `forward_batch`/`forward_single` serve V2's outputs exactly, only `forward_batch_heads` evaluates it."""

    def __init__(self, arch: GnnArchV2SoftPolicyFuture) -> None:
        super().__init__(arch)  # type: ignore[arg-type] — the field sets are identical by design
        self.future_head = FutureOccupancyHead(self.representation.output_dim, arch.policy_hidden)

    def future_heads(self) -> tuple[str, ...]:
        return (FUTURE_HEAD,)

    def _heads(self, legal_emb: Tensor, pooled: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        policy_logits, value, bin_logits, aux_logits, _ = super()._heads(legal_emb, pooled)
        return policy_logits, value, bin_logits, aux_logits, self.future_head(legal_emb)
