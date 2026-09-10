"""The ONE read path for the draw-rate abort family.

`train.draw_rate_abort` is read HERE and nowhere else; the composition root passes the resolved
spec into `StepCoordinatorConfig.draw_rate_abort`, whose four fields reach
`check_draw_rate_collapse(threshold=…, min_step=…, consec=…)` and `pooled_draw_rate(…,
N_pool_min=…)`.

ARMING IS A PROPERTY OF THIS VALUE: there is no boolean enable beside it, because a boolean
could contradict it. `None` is ARMED=NO and a spec is ARMED=YES — disjoint TYPES, not two
regions of one range.

THE FOUR TRAVEL TOGETHER. A threshold without `min_step` / `N_pool_min` is a hair-trigger: an
`N_pool_min` of 4 lets one drawn game in four meet a threshold of 0.25. Guards settable
independently of the threshold could be set to nothing, so the block arrives whole or not at
all, and all four are pre-registered at mint rather than tuned.

No code-side default and no numeric disable sentinel: the schema block is the sole authority, a
missing key is rejected at load naming the key, and `gt=0, le=1` means no NUMBER can disarm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrawRateAbortSpec:
    """The resolved terms of the draw-rate collapse hard abort.

    Defined beside the resolver rather than being the pydantic block, so nothing in
    `mantis.train` imports the schema class to consume it.
    """

    threshold: float
    min_step: int
    N_pool_min: int
    #: Rides here and not on the coordinator dataclass because a term of a DISARMED abort is
    #: not a fact: with `train.draw_rate_abort: null` neither the spec nor `consec` exists.
    consec: int


def resolve_draw_rate_abort(train_section: Any) -> DrawRateAbortSpec | None:
    """Return the validated draw-rate abort terms, or None when explicitly OFF."""
    block = train_section.draw_rate_abort
    if block is None:
        return None
    return DrawRateAbortSpec(
        threshold=float(block.threshold),
        min_step=int(block.min_step),
        N_pool_min=int(block.N_pool_min),
        consec=int(block.consec),
    )


__all__ = ["DrawRateAbortSpec", "resolve_draw_rate_abort"]
