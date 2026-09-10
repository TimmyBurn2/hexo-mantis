"""DeployHeadPlayer — the deploy-matched candidate head.

It runs the RUN'S OWN search: `search_kind` comes from the resolver
`SelfPlayHParams.from_config` reads and reaches the `MCTSTree.configure_search` setter the
self-play worker calls, so "deploy-matched" is a construction, not a coincidence. `puct` plays
the MOST-VISITED root child; `gumbel` plays Sequential Halving's own answer; the hybrid this
replaced (PUCT descent with a g=0 Gumbel root pick) is a third algorithm no bar can be matched
to. The Gumbel draw is SEEDED from the round's `seed_base` mixed with this player's game and
move counters, so a replayed round draws the same noise.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mantis._engine import MCTSTree
from mantis.util.device import release_cuda_cache

#: `get_root_children_info()` row shape: (coord, pool_idx, prior, visits, q).
ChildInfo = tuple[tuple[int, int], int, float, int, float]

InferFn = Callable[[Any], tuple[list[float], float]]

#: The graph collaborator runs the whole decode+expand itself: `InferFn`'s `(policy, value)`
#: return cannot carry the four producer outputs the no-drop graph expand needs.
ExpandFn = Callable[[MCTSTree, list[Any]], None]

#: Odd 64-bit multipliers, so distinct (game, move) pairs cannot collide inside one round.
_GAME_STRIDE = 0x9E3779B97F4A7C15
_MOVE_STRIDE = 0xBF58476D1CE4E5B9
_SEED_MASK = (1 << 64) - 1


class DeployHeadPlayer:
    """The deploy-matched candidate head: the run's own search over an `MCTSTree`.

    EXACTLY ONE of `infer_fn=` (grid: leaf -> `(policy, value)`, then dense `expand_and_backup`)
    or `expand_fn=` (graph: the collaborator owns the decode and the no-drop expand) is supplied.

    Raises:
        ValueError: neither or both of `infer_fn`/`expand_fn`; `leaf_batch_size < 1`;
            `gumbel_m < 1`; `search_kind` is not a kind the engine implements (raised by
            `MCTSTree.configure_search` on the first `new_game`).
    """

    def __init__(
        self,
        *,
        infer_fn: InferFn | None = None,
        expand_fn: ExpandFn | None = None,
        n_sims: int,
        leaf_batch_size: int,
        c_visit: float,
        c_scale: float,
        search_kind: str,
        gumbel_m: int,
        gumbel_seed: int,
    ) -> None:
        if (infer_fn is None) == (expand_fn is None):
            supplied = "both" if infer_fn is not None else "neither"
            raise ValueError(
                f"DeployHeadPlayer takes EXACTLY ONE of infer_fn= (grid) or expand_fn= "
                f"(graph); {supplied} was supplied. There is no default arm — picking one "
                f"here would decide the decode contract silently."
            )
        # `c_visit`, `c_scale`, `leaf_batch_size` and `gumbel_m` are REQUIRED schema keys, never
        # defaulted: a default equal to today's minted value is still a second authority.
        if int(leaf_batch_size) < 1:
            raise ValueError(
                f"DeployHeadPlayer: leaf_batch_size={leaf_batch_size!r} must be >= 1. It is "
                f"the config's own selfplay.leaf_batch_size (schema `ge=1`), threaded here so "
                f"deploy searches under the regime the net's targets were generated in."
            )
        if int(gumbel_m) < 1:
            raise ValueError(
                f"DeployHeadPlayer: gumbel_m={gumbel_m!r} must be >= 1. It is the config's "
                f"own selfplay.gumbel_m (schema `ge=1`), threaded for leaf_batch_size's "
                f"reason — a bar that considered a different number of root actions than the "
                f"run did is not deploy-matched."
            )
        self._infer_fn = infer_fn
        self._expand_fn = expand_fn
        self._n_sims = int(n_sims)
        self._leaf_batch_size = int(leaf_batch_size)
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        self._search_kind = str(search_kind)
        self._gumbel_m = int(gumbel_m)
        self._gumbel_seed = int(gumbel_seed) & _SEED_MASK
        self._game_index = 0
        self._move_index = 0
        self._tree: MCTSTree | None = None
        #: The LAST search's root for the game record, `(root_value, children)` from rows this
        #: head already computes. A plain attribute: the consumer reads it once per ply.
        self.last_root: tuple[float, list[ChildInfo]] | None = None

    def name(self) -> str:
        return "deploy_head"

    @property
    def search_kind(self) -> str:
        """The kind this head searches with — the run's own `search.kind`."""
        return self._search_kind

    def new_game(self) -> None:
        self._tree = self._fresh_tree()
        self._game_index += 1
        self._move_index = 0
        self.last_root = None

    def _fresh_tree(self) -> MCTSTree:
        """A tree configured with the RUN's search kind. `configure_search` runs ONCE per tree:
        under `gumbel` it allocates a per-node raw-value vector that does not change per ply."""
        tree = MCTSTree()
        tree.configure_search(self._search_kind, self._c_visit, self._c_scale)
        return tree

    def _evaluate(self, tree: MCTSTree, leaves: list[Any]) -> None:
        if self._expand_fn is not None:
            self._expand_fn(tree, leaves)
            return
        assert self._infer_fn is not None  # ctor guarantees exactly one arm
        policies: list[list[float]] = []
        values: list[float] = []
        for leaf in leaves:
            policy, value = self._infer_fn(leaf)
            policies.append(policy)
            values.append(value)
        tree.expand_and_backup(policies, values)

    def _move_seed(self) -> int:
        return (
            self._gumbel_seed
            ^ ((self._game_index * _GAME_STRIDE) & _SEED_MASK)
            ^ ((self._move_index * _MOVE_STRIDE) & _SEED_MASK)
        ) & _SEED_MASK

    def select_move(self, board: Any) -> tuple[int, int]:
        """Search `n_sims` LEAVES and return the move this run's search kind picks.

        The root's own evaluation is one of the N on both arms, and the budget advances by leaves
        RETURNED rather than requested — `select_leaves(k)` yields fewer than k on a cold tree, so
        crediting the request would let a throughput knob change deploy strength.

        Raises:
            ValueError: the search produced no root children, so there is no move to pick.
        """
        tree = self._tree if self._tree is not None else self._fresh_tree()
        self._tree = tree
        tree.new_game(board)
        try:
            move = self._search(tree)
        finally:
            release_cuda_cache()
        self._move_index += 1
        return move

    def _search(self, tree: MCTSTree) -> tuple[int, int]:
        root_leaves = tree.select_leaves(1)
        if root_leaves:
            self._evaluate(tree, root_leaves)
        sims_done = len(root_leaves)

        if self._search_kind == "gumbel":
            move = self._drive_gumbel(tree, sims_done)
        else:
            move = self._drive_puct(tree, sims_done)

        children_info = tree.get_root_children_info()
        # Captured from the tree the decision read, so a recorded root always matches its move.
        self.last_root = (float(tree.root_value()), children_info)
        if move is None:
            raise ValueError(
                "DeployHeadPlayer: the search produced no root children, so there is no "
                "move to pick. A head that guessed here would put a move nobody searched "
                "on a promotion bar."
            )
        return move

    def _drive_puct(self, tree: MCTSTree, sims_done: int) -> tuple[int, int] | None:
        # Batched by `leaf_batch_size`, the SAME knob the self-play worker reads: only the number
        # of blocking round-trips changes. Clamped to the remaining budget so N is exact.
        while sims_done < self._n_sims:
            current_batch = min(self._leaf_batch_size, self._n_sims - sims_done)
            leaves = tree.select_leaves(current_batch)
            if not leaves:
                break
            self._evaluate(tree, leaves)
            sims_done += len(leaves)
        top = tree.get_top_visits(1)
        return top[0][0] if top else None

    def _drive_gumbel(self, tree: MCTSTree, sims_done: int) -> tuple[int, int] | None:
        if tree.root_n_children() == 0:
            return None
        budget = max(0, self._n_sims - sims_done)
        tree.gumbel_root_begin(self._gumbel_m, budget, self._move_seed())
        spent = 0
        while spent < budget:
            child = tree.gumbel_root_select(self._c_visit, self._c_scale)
            if child is None:
                break
            tree.forced_root_child = child
            try:
                # ONE leaf: consecutive sims at a considered level land on DIFFERENT children.
                leaves = tree.select_leaves(1)
                if not leaves:
                    break
                self._evaluate(tree, leaves)
            finally:
                tree.forced_root_child = None
            spent += len(leaves)
        return tree.gumbel_root_best_move(self._c_visit, self._c_scale)


__all__ = ["ChildInfo", "DeployHeadPlayer", "ExpandFn", "InferFn"]
