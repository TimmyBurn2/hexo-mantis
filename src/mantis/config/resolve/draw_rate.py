"""`resolve_draw_rate_abort` — the ONE read path for the draw-rate abort family
(WPAX Phase D, CARD-DRAWRATE-KEY / R65, re-scoped by R80, shaped by R79/R83).

`train.draw_rate_abort` is read HERE and nowhere else. The composition root
(`mantis.run.compose_run`) passes the resolved spec into
`StepCoordinatorConfig.draw_rate_abort`, which `coordinator/step.py`'s hard-abort gate
gates the whole draw-rate check on and whose three fields reach, respectively,
`check_draw_rate_collapse(threshold=…, min_step=…)` and
`pool.per_worker_draw_rates(min_samples=…)`.

ARMING IS A PROPERTY OF THIS VALUE (R79) — there is no boolean enable beside it, because a
boolean could contradict it (`enabled=true, block=null` → the flag lies; `enabled=false,
threshold=0.25` → the config lies). `None` is ARMED=NO, explicitly; a spec is ARMED=YES on
those terms. The two are disjoint TYPES, not two regions of one range.

THE THREE TRAVEL TOGETHER, and that is the point (R80). A threshold without `min_step` /
`min_samples` is ADJ-14's hair-trigger: the shipped inclusion rule `len(dq) > 0` counted a
worker after ONE game, so one drawn game per worker saturated the pool mean at 1.0 — at or
above every legal threshold. Guards that could be set independently of the threshold could
be set to nothing, so the block arrives whole or not at all.

RUN-SCOPED CONSTANTS (R82/R85): `threshold`, `min_step` and `min_samples` are
pre-registered at mint prereg and that is "the only place they may change" — not tunables,
and never hand-varied in a committed config (R1).

No code-side default and no numeric disable sentinel (R1/R49/R83): the schema block is the
sole authority, a missing key never reaches here (pydantic rejects it at load, naming the
key), and `gt=0, le=1` means no NUMBER can disarm or over-shoot the metric's own range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrawRateAbortSpec:
    """The resolved terms of the draw-rate collapse hard abort.

    A frozen dataclass defined beside the resolver rather than the pydantic block itself:
    `train/coordinator/config.py` is the DAG-clean seam layer and types every injected
    collaborator against a local `Protocol` (`DrawRateAbortLike`), so nothing in
    `mantis.train` needs to import the schema class to consume this.
    """

    threshold: float
    min_step: int
    min_samples: int


def resolve_draw_rate_abort(train_section: Any) -> DrawRateAbortSpec | None:
    """Return the validated draw-rate abort terms, or None when explicitly OFF."""
    block = train_section.draw_rate_abort
    if block is None:
        return None
    return DrawRateAbortSpec(
        threshold=float(block.threshold),
        min_step=int(block.min_step),
        min_samples=int(block.min_samples),
    )


__all__ = ["DrawRateAbortSpec", "resolve_draw_rate_abort"]
