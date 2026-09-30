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

#: The collaborator runs the whole decode + no-drop expand for a batch of leaves itself.
ExpandFn = Callable[[MCTSTree, list[Any]], None]

#: Odd 64-bit multipliers, so distinct (game, move) pairs cannot collide inside one round.
_GAME_STRIDE = 0x9E3779B97F4A7C15
_MOVE_STRIDE = 0xBF58476D1CE4E5B9
_SEED_MASK = (1 << 64) - 1


class DeployHeadPlayer:
    """The deploy-matched candidate head: the run's own search over an `MCTSTree`.

    Raises:
        ValueError: `leaf_batch_size < 1`; `gumbel_m < 1`; `search_kind` is not a kind the
            engine implements, or `tactics` a block it cannot arm (raised by `MCTSTree` on the first `new_game`).
    """

    def __init__(
        self,
        *,
        expand_fn: ExpandFn,
        n_sims: int,
        leaf_batch_size: int,
        c_visit: float,
        c_scale: float,
        q_rescale: bool,
        search_kind: str,
        gumbel_m: int,
        gumbel_seed: int,
        tactics: dict[str, Any] | None,
    ) -> None:
        # `c_visit`, `c_scale`, `q_rescale`, `leaf_batch_size` and `gumbel_m` are REQUIRED schema
        # keys, never defaulted: a default equal to today's minted value is still a second authority.
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
        self._expand_fn = expand_fn
        self._n_sims = int(n_sims)
        self._leaf_batch_size = int(leaf_batch_size)
        self._c_visit = float(c_visit)
        self._c_scale = float(c_scale)
        self._q_rescale = bool(q_rescale)
        self._search_kind = str(search_kind)
        self._gumbel_m = int(gumbel_m)
        self._gumbel_seed = int(gumbel_seed) & _SEED_MASK
        #: The resolved `deploy.search.tactics` block the tree arms, or `None` (the module off).
        self._tactics = tactics
        self._game_index = 0
        self._move_index = 0
        self._tree: MCTSTree | None = None
        #: The LAST search's root for the game record, `(root_value, children)` from rows this
        #: head already computes. A plain attribute: the consumer reads it once per ply.
        self.last_root: tuple[float, list[ChildInfo]] | None = None
        #: The LAST move's descents, both searches' where it searched again (LADDER-1's budget witness reads it).
        self.last_sims: int | None = None
        #: The LAST search's tactics rows (`MCTSTree.tactics_counters`), `None` with the module off.
        self.last_tactics: dict[str, int] | None = None

    def name(self) -> str:
        return "deploy_head"

    @property
    def search_kind(self) -> str:
        """The kind this head searches with — the run's own `deploy.search.kind`."""
        return self._search_kind

    def new_game(self) -> None:
        self._tree = self._fresh_tree()
        self._game_index += 1
        self._move_index = 0
        self.last_root = None
        self.last_sims = None
        self.last_tactics = None

    def _fresh_tree(self) -> MCTSTree:
        """A tree configured with the RUN's search kind and σ. `configure_search` runs ONCE per
        tree: under `gumbel` it allocates a per-node raw-value vector that does not change per
        ply, and the root calls below read the σ it set (one σ per tree)."""
        tree = MCTSTree()
        tree.configure_search(self._search_kind, self._c_visit, self._c_scale, self._q_rescale)
        tree.configure_tactics(self._tactics)
        return tree

    def _move_seed(self) -> int:
        return (
            self._gumbel_seed
            ^ ((self._game_index * _GAME_STRIDE) & _SEED_MASK)
            ^ ((self._move_index * _MOVE_STRIDE) & _SEED_MASK)
        ) & _SEED_MASK

    def select_move(self, board: Any) -> tuple[int, int]:
        """Search `n_sims` DESCENTS and return the move this run's search kind picks.

        Every descent counts (leaves returned, inline descents, table hits); armed tactics play a decided root stone
        unsearched, the audit may swap the move for a hold, and where it held on nothing a re-search's winner plays.

        Raises:
            ValueError: no root children to pick from, or armed tactics at a radius below 5.
            RuntimeError: the root's tactics refused (`MCTSTree.root_offence`).
        """
        tree = self._tree if self._tree is not None else self._fresh_tree()
        self._tree = tree
        self.last_tactics = None
        tree.new_game(board)
        decided = tree.root_offence() if self._tactics is not None else None
        if decided is not None:
            self.last_root, self.last_sims = None, 0
            self.last_tactics = tree.tactics_counters()
            self._move_index += 1
            return decided
        try:
            move = self._search(tree)
        finally:
            release_cuda_cache()
        if self._tactics is not None:
            move = self._audit(tree, move)
            self.last_tactics = tree.tactics_counters()
        self._move_index += 1
        return move

    def _audit(self, tree: MCTSTree, move: tuple[int, int]) -> tuple[int, int]:
        """The audit's move, or, where it held on nothing and the target sat on its vetoes, a re-search's winner."""
        move = tree.root_audit(move)
        if not (tree.searched_all_vetoed() and tree.begin_research()):
            return move
        spent = self.last_sims or 0
        try:
            move = self._search(tree)
        finally:
            release_cuda_cache()
        self.last_sims = (self.last_sims or 0) + spent
        return move

    def _search(self, tree: MCTSTree) -> tuple[int, int]:
        root_leaves = tree.select_leaves(1)
        if root_leaves:
            self._expand_fn(tree, root_leaves)
        sims_done = len(root_leaves) + tree.last_inline_descents() + tree.last_tt_hits()

        if self._search_kind == "gumbel":
            move, spent = self._drive_gumbel(tree, sims_done)
        else:
            move, spent = self._drive_puct(tree, sims_done)
        self.last_sims = spent

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

    def _drive_puct(self, tree: MCTSTree, sims_done: int) -> tuple[tuple[int, int] | None, int]:
        # Batched by `leaf_batch_size`, the SAME knob the self-play worker reads: only the number
        # of blocking round-trips changes. Clamped to the remaining budget so N is exact.
        while sims_done < self._n_sims:
            current_batch = min(self._leaf_batch_size, self._n_sims - sims_done)
            leaves = tree.select_leaves(current_batch)
            unserved = tree.last_inline_descents() + tree.last_tt_hits()
            if not leaves and not unserved:
                break
            if leaves:
                self._expand_fn(tree, leaves)
            sims_done += len(leaves) + unserved
        top = tree.get_top_visits(1)
        return (top[0][0] if top else None), sims_done

    def _drive_gumbel(self, tree: MCTSTree, sims_done: int) -> tuple[tuple[int, int] | None, int]:
        if tree.root_n_children() == 0:
            return None, sims_done
        budget = max(0, self._n_sims - sims_done)
        tree.gumbel_root_begin(self._gumbel_m, budget, self._move_seed())
        spent = 0
        while spent < budget:
            child = tree.gumbel_root_select()
            if child is None:
                break
            # The FORCED descent: the halving's chosen child is the one descended, not PUCT's pick.
            leaves = tree.select_leaves_forced([child])
            unserved = tree.last_inline_descents() + tree.last_tt_hits()
            if not leaves and not unserved:
                break
            if leaves:
                self._expand_fn(tree, leaves)
            spent += len(leaves) + unserved
        return tree.gumbel_root_best_move(), sims_done + spent


__all__ = ["ChildInfo", "DeployHeadPlayer", "ExpandFn"]
