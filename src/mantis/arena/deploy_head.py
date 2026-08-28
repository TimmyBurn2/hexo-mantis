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
from mantis.util.device import release_cuda_cache

#: `get_root_children_info()` row shape (bridge stub _engine.pyi):
#: (coord, pool_idx, prior, visits, q)
ChildInfo = tuple[tuple[int, int], int, float, int, float]

InferFn = Callable[[Any], tuple[list[float], float]]

#: The graph collaborator (WP12-R Phase EVALDECODE): given the live tree and the leaves
#: `select_leaves` just returned, run the whole decode+expand itself. It exists because
#: the graph seam's no-drop expand needs FOUR producer outputs (dense, overflow, value,
#: builder centre) plus two spec constants — a shape `InferFn`'s `(policy, value)` return
#: cannot carry without dropping exactly the half this card exists to stop dropping.
ExpandFn = Callable[[MCTSTree, list[Any]], None]


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
    `infer_fn`- or `expand_fn`-driven `MCTSTree` search. NO dirichlet/epsilon/
    gumbel_scale (or any softmax-knob) constructor parameters exist — g=0 is structural,
    not a knob.

    EXACTLY ONE of `infer_fn=` (the grid arm: leaf -> `(policy, value)`, then the dense
    `expand_and_backup`) or `expand_fn=` (the graph arm: the collaborator owns the decode
    and the no-drop `expand_and_backup_ls_graph`) is supplied. Neither and both are named
    `ValueError`s: a defaulted arm or a polymorphic `infer_fn` would reintroduce the
    silent pick this card exists to remove, and a second player class would fork the g=0
    argmax authority below.
    """

    def __init__(
        self,
        *,
        infer_fn: InferFn | None = None,
        expand_fn: ExpandFn | None = None,
        n_sims: int,
        leaf_batch_size: int,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
    ) -> None:
        if (infer_fn is None) == (expand_fn is None):
            supplied = "both" if infer_fn is not None else "neither"
            raise ValueError(
                f"DeployHeadPlayer takes EXACTLY ONE of infer_fn= (grid) or expand_fn= "
                f"(graph); {supplied} was supplied. There is no default arm — picking one "
                f"here would decide the decode contract silently."
            )
        # R318(b): REQUIRED and never defaulted. A default would be a search-regime constant
        # nobody minted, and the value it would take (1) is the defect itself.
        if int(leaf_batch_size) < 1:
            raise ValueError(
                f"DeployHeadPlayer: leaf_batch_size={leaf_batch_size!r} must be >= 1. It is "
                f"the config's own selfplay.leaf_batch_size (schema `ge=1`), threaded here so "
                f"deploy searches under the regime the net's targets were generated in."
            )
        self._infer_fn = infer_fn
        self._expand_fn = expand_fn
        self._n_sims = int(n_sims)
        self._leaf_batch_size = int(leaf_batch_size)
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
        # R318(b): leaves are selected in batches of `leaf_batch_size`, the SAME knob the
        # self-play worker reads. What changes is the number of BLOCKING round-trips — n_sims
        # of them at k=1, roughly n_sims/k above it — never the amount of search.
        #
        # THE BUDGET ADVANCES BY LEAVES RETURNED, NOT BY LEAVES REQUESTED, and the difference
        # is load-bearing (R318(b)(iii), "fixed nodes"). `select_leaves(k)` yields FEWER than k
        # on a cold tree — measured 1, 1, then k — so crediting the request would spend ~11%
        # fewer nodes at k=8 than at k=1, making a self-play THROUGHPUT knob silently change
        # deploy STRENGTH. `selfplay/worker.py` credits the request instead; that divergence is
        # deliberate and is the one place this head does not mirror it, because self-play's
        # budget is a data-generation rate while this one is the strength dial a promotion gate
        # is read against. The two are matched on WIDTH, which is what deploy-matching means
        # here; they were never matched on node count (n_simulations 50 vs deploy_sims 128).
        sims_done = 0
        while sims_done < self._n_sims:
            current_batch = min(self._leaf_batch_size, self._n_sims - sims_done)
            leaves = tree.select_leaves(current_batch)
            if not leaves:
                break
            if self._expand_fn is not None:
                self._expand_fn(tree, leaves)
            else:
                assert self._infer_fn is not None  # ctor guarantees exactly one arm
                policies: list[list[float]] = []
                values: list[float] = []
                for leaf in leaves:
                    policy, value = self._infer_fn(leaf)
                    policies.append(policy)
                    values.append(value)
                tree.expand_and_backup(policies, values)
            sims_done += len(leaves)
        children_info = tree.get_root_children_info()
        try:
            move = select_argmax_child(children_info, c_visit=self._c_visit, c_scale=self._c_scale)
        finally:
            release_cuda_cache()
        return move


__all__ = ["ChildInfo", "DeployHeadPlayer", "ExpandFn", "InferFn", "select_argmax_child"]
