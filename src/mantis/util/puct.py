"""The PUCT tree's constants as one value: produced by `mantis.config.resolve.puct`, handed to every tree a run builds."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PuctConstants:
    """The PUCT selection and quiescence constants a tree is built with; none has a code-side default."""

    c_puct: float
    fpu_reduction: float
    quiescence_enabled: bool
    quiescence_blend_2: float

    def tree_kwargs(self) -> dict[str, Any]:
        """The keyword arguments `MCTSTree` takes."""
        return dataclasses.asdict(self)


__all__ = ["PuctConstants"]
