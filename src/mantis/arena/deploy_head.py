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
from mantis.arena.eval_cache import GameEvalCache
from mantis.util.device import release_cuda_cache
from mantis.util.puct import PuctConstants

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
        puct: PuctConstants,
        eval_cache: GameEvalCache | None = None,
        early_stop: bool = False,
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
        #: The run's `selfplay.mcts` constants every tree is built with: the bridge holds no default.
        self._puct = puct
        self._game_index = 0
        self._move_index = 0
        self._tree: MCTSTree | None = None
        #: The LAST search's root for the game record, `(root_value, children)` from rows this
        #: head already computes. A plain attribute: the consumer reads it once per ply.
        self.last_root: tuple[float, list[ChildInfo]] | None = None
        #: The LAST search's descents, the head's own count (LADDER-1's budget witness reads it).
        self.last_sims: int | None = None
        #: The LAST search's tactics rows (`MCTSTree.tactics_counters`), `None` with the module off.
        self.last_tactics: dict[str, int] | None = None
        #: The per-game cache `expand_fn` serves through, emptied by `new_game`; `None` when it has none.
        self._eval_cache = eval_cache
        #: PUCT ends a search once its visit leader cannot be overtaken by the descents left; off, it spends them all.
        self._early_stop = bool(early_stop)
        #: Whether the LAST search ended at that stop rather than at its budget.
        self.last_stopped = False
        #: The head's own lever rows, cumulative: stops and the descents they left, and the PUCT select calls.
        self._rows = dict.fromkeys(("stop_fired", "stop_saved", "select_calls", "select_overlaps",
                                    "select_network_leaves"), 0)

    def name(self) -> str:
        return "deploy_head"

    @property
    def search_kind(self) -> str:
        """The kind this head searches with — the run's own `deploy.search.kind`."""
        return self._search_kind

    def new_game(self) -> None:
        if self._eval_cache is not None:
            self._eval_cache.new_game()
        self._tree = self._fresh_tree()
        self._game_index += 1
        self._move_index = 0
        self.last_root = None
        self.last_sims = None
        self.last_tactics = None

    def search_rows(self) -> dict[str, int]:
        """The head's levers' cumulative rows, each keyed by its lever: `stop_*`, `select_*`, and `cache_*` with a cache."""
        rows = dict(self._rows)
        if self._eval_cache is not None:
            rows.update({f"cache_{k}": v for k, v in self._eval_cache.counters().items()})
        return rows

    def _fresh_tree(self) -> MCTSTree:
        """A tree configured with the RUN's search kind and σ. `configure_search` runs ONCE per
        tree: under `gumbel` it allocates a per-node raw-value vector that does not change per
        ply, and the root calls below read the σ it set (one σ per tree)."""
        tree = MCTSTree(**self._puct.tree_kwargs())
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

        The budget counts every descent, the root's own among them: leaves RETURNED, inline descents and table hits; an
        empty board plays the origin, armed tactics a decided stone, both unsearched; the audit may swap in a hold.

        Raises:
            ValueError: no root children to pick from, or armed tactics at a radius below 5.
            RuntimeError: the root's tactics refused (`MCTSTree.root_offence`).
        """
        self.last_tactics = None
        self.last_stopped = False
        if not board.get_stones():
            # The first stone is the origin, as the official rule plays it.
            self.last_root, self.last_sims = None, 0
            self._move_index += 1
            return (0, 0)
        tree = self._tree if self._tree is not None else self._fresh_tree()
        self._tree = tree
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
            move = tree.root_audit(move)
            self.last_tactics = tree.tactics_counters()
        self._move_index += 1
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
        calls, overlaps, leaves = tree.select_counters()
        self._rows["select_calls"] += calls
        self._rows["select_overlaps"] += overlaps
        self._rows["select_network_leaves"] += leaves

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
            left = self._n_sims - sims_done
            if self._early_stop and _leader_fixed(tree, left):
                self.last_stopped = True
                self._rows["stop_fired"] += 1
                self._rows["stop_saved"] += left
                break
            # Up to a batch of network leaves, refilled past table and solver descents, within the budget left.
            leaves = tree.select_leaves_filled(self._leaf_batch_size, left)
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


def _leader_fixed(tree: MCTSTree, left: int) -> bool:
    """No runner-up can reach the visit leader with `left` descents, so the most visited child is decided."""
    top = tree.get_top_visits(2)
    return len(top) == 1 or (len(top) == 2 and top[0][1] - top[1][1] > left)


__all__ = ["ChildInfo", "DeployHeadPlayer", "ExpandFn"]
