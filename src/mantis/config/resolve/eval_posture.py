"""`resolve_ply_cap_adjudication` / `resolve_strength_floor` — the ONE read path for the two
early-strength eval-posture blocks.

`eval.ply_cap_adjudication` and `eval.strength_floor` are read HERE and nowhere else. The eval
pipeline calls both, puts the resolved specs on the `RoundSpec`, and the worker consumes them
across the process seam.

Arming is a property of the VALUE: there is no boolean enable beside either block, because a
boolean could contradict it. `None` is ARMED=NO, explicitly; a spec is ARMED=YES. The two are
disjoint TYPES, not two regions of one range, so no in-range number can imitate the disarmed
posture and no code-side default exists — a missing key is rejected at load, naming the key.

Two resolvers rather than one, because a run may want the ply-cap criterion without the ladder
floor or the reverse, and a single combined spec would make the two arm together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlyCapAdjudicationSpec:
    """The resolved terms of ply-cap adjudication.

    A frozen dataclass beside the resolver rather than the pydantic block itself: this value
    crosses the eval process seam as JSON, so nothing in `mantis.eval` or `mantis.arena` needs
    to import the schema class — or pydantic — to consume it.
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
