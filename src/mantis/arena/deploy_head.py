"""DeployHeadPlayer — the deploy-matched candidate head (design §a.2 deploy_head.py).

IT RUNS THE RUN'S OWN SEARCH, AND THAT IS THE WHOLE POINT. The head takes `search_kind`
from `mantis.config.resolve.resolve_search_kind` — the SAME resolver
`SelfPlayHParams.from_config` reads — and hands it to the SAME `MCTSTree.configure_search`
setter the self-play worker calls. LAW-15's "deploy-matched" is then a construction rather
than a coincidence between two call sites.

WHAT WAS DELETED HERE. The head used to run a HYBRID: a plain PUCT tree whose ROOT pick was
the Gumbel scoring function with its noise term set to zero (`select_argmax_child`, the g=0
"completed-Q argmax"). That is neither of the two searches the repo implements — the tree
descended by PUCT, no Sequential Halving ran anywhere, and the move was chosen by a rule no
self-play worker has ever used. A bar that plays a third algorithm cannot be matched to
anything. The two arms are now:

* `puct` — PUCT descent, and the move is the MOST-VISITED root child. That is what
  self-play's own visit-count policy picks at its deploy setting, which is what
  "deploy-matched" has to mean on this arm.
* `gumbel` — Gumbel-Top-k root sampling with Sequential Halving, and the move is
  Sequential Halving's own answer (the highest-scoring of the most-visited children).

THE GUMBEL DRAW IS SEEDED, EXPLICITLY. It is the head's one stochastic term, and a
promotion bar has to be a reproducible instrument (LAW-15). The seed is the round's
`seed_base`, mixed with this player's own game and move counters, so a replayed round draws
the same noise.
"""
from __future__ import annotations

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

#: Mixing constants for the per-move Gumbel seed. Odd 64-bit multipliers, so distinct
#: (game, move) pairs cannot collide inside one round.
_GAME_STRIDE = 0x9E3779B97F4A7C15
_MOVE_STRIDE = 0xBF58476D1CE4E5B9
_SEED_MASK = (1 << 64) - 1


class DeployHeadPlayer:
    """The deploy-matched candidate head: the run's own search, driven by an injected
    `infer_fn` or `expand_fn` over an `MCTSTree`.

    EXACTLY ONE of `infer_fn=` (the grid arm: leaf -> `(policy, value)`, then the dense
    `expand_and_backup`) or `expand_fn=` (the graph arm: the collaborator owns the decode
    and the no-drop `expand_and_backup_ls_graph`) is supplied. Neither and both are named
    `ValueError`s: a defaulted arm or a polymorphic `infer_fn` would reintroduce the
    silent pick this card exists to remove.

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
        # AUDIT-1 F-39. `c_visit`/`c_scale` lost their `= 50.0` / `= 1.0` defaults for
        # `leaf_batch_size`'s reason, one row over: they are REQUIRED schema keys
        # (`selfplay.c_visit`, `selfplay.c_scale`) that the eval head was never given, so the
        # deploy-matched bar searched at this signature's numbers instead of the run's. A
        # default that happens to equal today's minted value is still a second authority — and
        # the moment the key is re-minted, LAW-15's "deploy-matched" claim stops being true
        # with no config diff to show for it.
        #
        # R318(b): REQUIRED and never defaulted. A default would be a search-regime constant
        # nobody minted, and the value it would take (1) is the defect itself.
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
        #: The run's `search.kind`, NEVER defaulted — see the module docstring. Validated by
        #: the engine on the first `configure_search`, which refuses an unknown kind.
        self._search_kind = str(search_kind)
        self._gumbel_m = int(gumbel_m)
        self._gumbel_seed = int(gumbel_seed) & _SEED_MASK
        self._game_index = 0
        self._move_index = 0
        self._tree: MCTSTree | None = None
        #: R344(b) — the LAST search's root, for the game record. Read by
        #: `arena.match._play_one_game` off WHICHEVER player just moved, so on a
        #: deploy-head-vs-deploy-head game both sides publish one. `(root_value, children)`
        #: with `children` the `get_root_children_info()` rows this head already computes and,
        #: before this, discarded one line before returning its move. `None` until the first
        #: `select_move`. It is a plain attribute rather than a callback because the consumer
        #: (`arena.match._play_one_game`) reads it once per ply and owns what to keep.
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
        """A tree configured with the RUN's search kind.

        `configure_search` is called ONCE per tree, never per move: under `gumbel` it
        allocates the per-node raw-value vector, and re-allocating that every ply would be
        real work for a value that does not change.
        """
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

        THE ROOT'S OWN EVALUATION IS ONE OF THE N, on both arms — the same charge the
        self-play drive makes, so `n_sims` means the same amount of network work here as
        it does there.

        THE BUDGET ADVANCES BY LEAVES RETURNED, NOT BY LEAVES REQUESTED, and the difference
        is load-bearing (R318(b)(iii), "fixed nodes"). `select_leaves(k)` yields FEWER than k
        on a cold tree — measured 1, 1, then k — so crediting the request would spend ~11%
        fewer nodes at k=8 than at k=1, making a self-play THROUGHPUT knob silently change
        deploy STRENGTH.

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
        # The root: ONE leaf, charged against the budget on both arms.
        root_leaves = tree.select_leaves(1)
        if root_leaves:
            self._evaluate(tree, root_leaves)
        sims_done = len(root_leaves)

        if self._search_kind == "gumbel":
            move = self._drive_gumbel(tree, sims_done)
        else:
            move = self._drive_puct(tree, sims_done)

        children_info = tree.get_root_children_info()
        # Captured from the same tree the decision read, so a recorded root can never
        # describe a different search than the move beside it.
        self.last_root = (float(tree.root_value()), children_info)
        if move is None:
            raise ValueError(
                "DeployHeadPlayer: the search produced no root children, so there is no "
                "move to pick. A head that guessed here would put a move nobody searched "
                "on a promotion bar."
            )
        return move

    def _drive_puct(self, tree: MCTSTree, sims_done: int) -> tuple[int, int] | None:
        # R318(b): leaves are selected in batches of `leaf_batch_size`, the SAME knob the
        # self-play worker reads. What changes is the number of BLOCKING round-trips — n_sims
        # of them at k=1, roughly n_sims/k above it — never the amount of search. The batch is
        # clamped to the REMAINING budget so the search stops at exactly N (R335(c)).
        while sims_done < self._n_sims:
            current_batch = min(self._leaf_batch_size, self._n_sims - sims_done)
            leaves = tree.select_leaves(current_batch)
            if not leaves:
                break
            self._evaluate(tree, leaves)
            sims_done += len(leaves)
        # The move self-play's own visit-count policy picks at its deploy setting: the
        # most-visited child.
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
                # ONE leaf: consecutive simulations at a considered level land on DIFFERENT
                # children by construction, so forcing a whole batch into one candidate
                # would be a different algorithm.
                leaves = tree.select_leaves(1)
                if not leaves:
                    break
                self._evaluate(tree, leaves)
            finally:
                tree.forced_root_child = None
            spent += len(leaves)
        return tree.gumbel_root_best_move(self._c_visit, self._c_scale)


__all__ = ["ChildInfo", "DeployHeadPlayer", "ExpandFn", "InferFn"]
