"""`train.policy_loss_trough_abort` -> a frozen spec, or `None` on the explicit OFF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyLossTroughAbortSpec:
    """The validated terms; with `null` neither the spec nor any member exists."""

    delta_nats: float
    consec: int
    max_step: int


def resolve_policy_loss_trough_abort(train_section: Any) -> PolicyLossTroughAbortSpec | None:
    """Return the validated trough-halt terms, or None when explicitly OFF."""
    block = train_section.policy_loss_trough_abort
    if block is None:
        return None
    return PolicyLossTroughAbortSpec(
        delta_nats=float(block.delta_nats), consec=int(block.consec), max_step=int(block.max_step),
    )


__all__ = ["PolicyLossTroughAbortSpec", "resolve_policy_loss_trough_abort"]
