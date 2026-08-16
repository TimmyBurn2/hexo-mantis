"""`resolve_ply_cap_adjudication` / `resolve_strength_floor` — the ONE read path for the two
early-strength eval-posture blocks (F-R-P2B-5).

`eval.ply_cap_adjudication` and `eval.strength_floor` are read HERE and nowhere else. The eval
pipeline (`mantis.eval.pipeline.EvalPipeline._build_round_spec`) calls both, puts the resolved
specs on the `RoundSpec`, and the worker consumes them across the process seam — the same
shape `resolve_draw_rate_abort` uses to reach `StepCoordinatorConfig`.

ARMING IS A PROPERTY OF THE VALUE (R79). There is no boolean enable beside either block,
because a boolean could contradict it (`enabled=true, block=null` -> the flag lies;
`enabled=false, criterion=...` -> the config lies). `None` is ARMED=NO, explicitly; a spec is
ARMED=YES on those terms. The two are disjoint TYPES, not two regions of one range.

BOTH SHIP DISARMED, AND THAT IS THE POINT OF THE SEAM. Every committed config mints `null`
for both, so every consumer below takes its `None` arm and the run's observable behaviour —
the arena's capped-game label, the round's phase order, the events on the stream, the sidecar
result JSON's key set — is byte-identical to the tree before these keys existed. The
mechanism ships; the values are mint-prereg rows the operator owns
(`plan/EVAL_POSTURE_OPTIONS.md` is the decision material, not an authority).

Two resolvers rather than one, because they are two facts: a run may want the ply-cap
criterion without the ladder floor or the reverse, and a single combined spec would make the
two arm together whether or not that was meant. They share a module because they share one
prereg decision and one measured finding, so a reader who opens either finds the other.

No code-side default and no numeric disable sentinel (R1): the schema blocks are the sole
authority, a missing key never reaches here (pydantic rejects it at load, naming the key), and
the disarmed posture is a TYPE (`None`) that no in-range number can imitate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlyCapAdjudicationSpec:
    """The resolved terms of ply-cap adjudication.

    A frozen dataclass beside the resolver rather than the pydantic block itself, for the
    reason `DrawRateAbortSpec` gives: this value crosses the eval process seam as JSON
    (`RoundSpec.to_dict`), so nothing in `mantis.eval` or `mantis.arena` needs to import the
    schema class — or pydantic — to consume it.
    """

    criterion: str
    min_margin: int


@dataclass(frozen=True)
class StrengthFloorSpec:
    """The resolved terms of the ladder strength floor. Same seam reason as above."""

    probe_games: int
    min_decisive_rate: float
    min_winrate: float


def resolve_ply_cap_adjudication(eval_section: Any) -> PlyCapAdjudicationSpec | None:
    """Return the validated ply-cap adjudication terms, or None when explicitly OFF."""
    block = eval_section.ply_cap_adjudication
    if block is None:
        return None
    return PlyCapAdjudicationSpec(
        criterion=str(block.criterion),
        min_margin=int(block.min_margin),
    )


def resolve_strength_floor(eval_section: Any) -> StrengthFloorSpec | None:
    """Return the validated strength-floor terms, or None when explicitly OFF."""
    block = eval_section.strength_floor
    if block is None:
        return None
    return StrengthFloorSpec(
        probe_games=int(block.probe_games),
        min_decisive_rate=float(block.min_decisive_rate),
        min_winrate=float(block.min_winrate),
    )


__all__ = [
    "PlyCapAdjudicationSpec",
    "StrengthFloorSpec",
    "resolve_ply_cap_adjudication",
    "resolve_strength_floor",
]
