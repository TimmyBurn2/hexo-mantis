"""`train.ply_cap_abort` -> a frozen spec, or `None` on the explicit OFF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlyCapAbortSpec:
    """The validated terms; with `null` neither the spec nor any member exists."""

    rate: float
    window_games: int
    min_step: int


def resolve_ply_cap_abort(train_section: Any) -> PlyCapAbortSpec | None:
    """Return the validated ply-cap halt terms, or None when explicitly OFF."""
    block = train_section.ply_cap_abort
    if block is None:
        return None
    return PlyCapAbortSpec(
        rate=float(block.rate), window_games=int(block.window_games), min_step=int(block.min_step),
    )


__all__ = ["PlyCapAbortSpec", "resolve_ply_cap_abort"]
