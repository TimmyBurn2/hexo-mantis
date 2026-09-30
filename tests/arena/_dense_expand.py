"""A per-leaf `(policy, value)` stub adapted to `DeployHeadPlayer.expand_fn` over the dense expand."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

InferStub = Callable[[Any], tuple[list[float], float]]


def dense_expand(infer: InferStub) -> Callable[[Any, list[Any]], None]:
    """An expand collaborator calling `infer` once per leaf, in order, then `expand_and_backup`."""
    def _expand(tree: Any, leaves: list[Any]) -> None:
        results = [infer(leaf) for leaf in leaves]
        tree.expand_and_backup([p for p, _ in results], [v for _, v in results])
    return _expand
