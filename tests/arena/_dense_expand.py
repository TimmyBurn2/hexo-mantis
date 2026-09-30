"""A per-leaf `(policy, value)` stub adapted to `DeployHeadPlayer.expand_fn` over the dense expand."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

InferStub = Callable[[Any], tuple[list[float], float]]

#: The dense policy stride the stubs fill.
_STRIDE = 362


def dense_expand(infer: InferStub) -> Callable[[Any, list[Any]], None]:
    """An expand collaborator calling `infer` once per leaf, in order, then `expand_and_backup`."""
    def _expand(tree: Any, leaves: list[Any]) -> None:
        results = [infer(leaf) for leaf in leaves]
        tree.expand_and_backup([p for p, _ in results], [v for _, v in results])
    return _expand


def peaked_infer(calls: list[int]) -> InferStub:
    """The red team's peaked net: a distance-decay prior and a position-dependent value."""
    def _infer(board: Any) -> tuple[list[float], float]:
        calls.append(1)
        legal = board.legal_moves()
        recent = [(q, r) for (q, r, _p) in board.get_stones()][-4:]
        weights = []
        for q, r in legal:
            d = min(
                (abs(q - sq) + abs(r - sr) + abs((q + r) - (sq + sr))) / 2 for sq, sr in recent
            ) if recent else 0
            weights.append(math.exp(-1.5 * d))
        total = sum(weights)
        policy = [0.0] * _STRIDE
        for (q, r), w in zip(legal, weights, strict=True):
            flat = board.to_flat(q, r)
            if flat < _STRIDE:
                policy[flat] = w / total
        value = ((board.zobrist_hash() % 2001) / 1000.0 - 1.0) * 0.5
        return policy, value
    return _infer
