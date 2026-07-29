"""DeployHeadPlayer — the deploy-matched candidate head (design §a.2 deploy_head.py).

Gumbel-SH completed-Q ARGMAX with gumbel scale g=0 (no root noise, no softmax knob): with
g=0 every Gumbel(0,1) root-noise term is exactly 0, so the SH-winner score collapses to
`log(max(prior_i, 1e-8)) + sigma_i` with `sigma_i = (c_visit + max_n_all) * c_scale *
clamp(q_i, -1, 1)` and `max_n_all` = the max visit count over ALL root children (ported
mechanics: hexo_rl/eval/deploy_strength_eval.py:109-207, gumbel_search_py.py:178-227).

`select_argmax_child` is the ORACLE-CHOSEN pure surface (tests/arena/test_deploy_head.py):
the smallest testable unit of the deploy head's internal decision, hand-checkable with a
calculator. `PuctEvalPlayer` is deliberately NOT built (legacy temp-0.5 opponents are dead
— run3 dropped them).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mantis._engine import MCTSTree

#: `get_root_children_info()` row shape (bridge stub _engine.pyi):
#: (coord, pool_idx, prior, visits, q)
ChildInfo = tuple[tuple[int, int], int, float, int, float]

InferFn = Callable[[Any], tuple[list[float], float]]


def select_argmax_child(
    children_info: list[ChildInfo], *, c_visit: float, c_scale: float
) -> tuple[int, int]:
    """Pure g=0 completed-Q argmax over root children (ORACLE-CHOSEN SEAM).

    `sigma_i = (c_visit + max_n_all) * c_scale * clamp(q_i, -1, 1)`; the winner maximizes
    `log(max(prior_i, 1e-8)) + sigma_i`. NOT a PUCT visit-count argmax — the highest-visit
    child need not win (test-pinned).
    """
    if not children_info:
        raise ValueError("select_argmax_child: no root children to select from")
    max_n_all = max(visits for _coord, _idx, _prior, visits, _q in children_info)

    best_coord: tuple[int, int] | None = None
    best_score = float("-inf")
    for coord, _pool_idx, prior, _visits, q in children_info:
        clamped_q = max(-1.0, min(1.0, q))
        sigma = (c_visit + max_n_all) * c_scale * clamped_q
        score = math.log(max(prior, 1e-8)) + sigma
        if score > best_score:
            best_score = score
            best_coord = coord
    assert best_coord is not None
    return best_coord


class DeployHeadPlayer:
    """The deploy-matched candidate head: g=0 completed-Q argmax over an injected
    `infer_fn`-driven `MCTSTree` search. NO dirichlet/epsilon/gumbel_scale (or any
    softmax-knob) constructor parameters exist — g=0 is structural, not a knob."""

    def __init__(
        self,
        *,
        infer_fn: InferFn,
        n_sims: int,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
    ) -> None:
        self._infer_fn = infer_fn
        self._n_sims = int(n_sims)
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        self._tree: MCTSTree | None = None

    def name(self) -> str:
        return "deploy_head"

    def new_game(self) -> None:
        self._tree = MCTSTree()

    def select_move(self, board: Any) -> tuple[int, int]:
        tree = self._tree if self._tree is not None else MCTSTree()
        self._tree = tree
        tree.new_game(board)
        for _ in range(self._n_sims):
            leaves = tree.select_leaves(1)
            if not leaves:
                break
            policies: list[list[float]] = []
            values: list[float] = []
            for leaf in leaves:
                policy, value = self._infer_fn(leaf)
                policies.append(policy)
                values.append(value)
            tree.expand_and_backup(policies, values)
        children_info = tree.get_root_children_info()
        return select_argmax_child(children_info, c_visit=self._c_visit, c_scale=self._c_scale)


__all__ = ["ChildInfo", "DeployHeadPlayer", "InferFn", "select_argmax_child"]
