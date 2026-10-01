"""`selfplay.mcts`'s PUCT constants -> a frozen spec, the one read path for every tree a run builds."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mantis.util.puct import PuctConstants


def resolve_puct_constants(config: Any) -> PuctConstants:
    """The minted `selfplay.mcts` constants, from a `RunConfig` or its plain mapping (a checkpoint stamp's).

    Raises:
        KeyError, TypeError: a mapping without `selfplay.mcts` or one of its four keys, or a null value.
    """
    mcts = config["selfplay"]["mcts"] if isinstance(config, Mapping) else config.selfplay.mcts
    read = mcts.__getitem__ if isinstance(mcts, Mapping) else (lambda key: getattr(mcts, key))
    return PuctConstants(
        c_puct=float(read("c_puct")), fpu_reduction=float(read("fpu_reduction")),
        quiescence_enabled=bool(read("quiescence_enabled")), quiescence_blend_2=float(read("quiescence_blend_2")),
    )


__all__ = ["PuctConstants", "resolve_puct_constants"]
