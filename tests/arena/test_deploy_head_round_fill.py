"""A PUCT round refills past table and solver descents to a batch of net leaves, within the budget left."""
from __future__ import annotations

from typing import Any

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

from _dense_expand import dense_expand
from _minted_puct import MINTED_PUCT
from test_deploy_head_budget_spent import _LEAF_TACTICS, _open_four_board, _peaked_infer


class _Spy:
    """Forwards the real tree, recording each fill call's ask and what it returned."""

    def __init__(self, inner: Any, calls: list[tuple[int, int, int, int]]) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_calls", calls)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != "select_leaves_filled":
            return attr

        def fill(network: int, descents: int) -> list:
            leaves = attr(network, descents)
            unserved = self._inner.last_inline_descents() + self._inner.last_tt_hits()
            self._calls.append((network, descents, len(leaves), unserved))
            return leaves
        return fill


def test_a_round_refills_past_table_and_solver_descents_within_the_budget_left() -> None:
    calls: list[tuple[int, int, int, int]] = []
    head = DeployHeadPlayer(expand_fn=dense_expand(_peaked_infer([])), n_sims=512, leaf_batch_size=8, c_visit=50.0,
                            c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=7,
                            tactics=_LEAF_TACTICS, puct=MINTED_PUCT)
    real = head._fresh_tree
    head._fresh_tree = lambda: _Spy(real(), calls)
    head.new_game()
    head.select_move(_open_four_board())
    assert head.last_sims == 512, "the refill never overspends"
    refilled = 0
    for network, descents, served, unserved in calls:
        assert served <= network and served + unserved <= descents
        refilled += served == network and unserved > 0
    assert refilled > 0, "no round carried a full batch past a table or solver descent: the refill never fired"
    rows = head.search_rows()
    # The tree counts the root's own one-leaf call too, which no fill call makes.
    assert rows["select_network_leaves"] == sum(c[2] for c in calls) + 1 and rows["select_calls"] == len(calls) + 1


def test_the_round_fill_keeps_a_plain_board_on_its_budget() -> None:
    head = DeployHeadPlayer(expand_fn=dense_expand(_peaked_infer([])), n_sims=256, leaf_batch_size=8, c_visit=50.0,
                            c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=7, tactics=None,
                            puct=MINTED_PUCT)
    head.new_game()
    board = Board()
    for q, r in [(0, 0), (2, -1), (1, 1), (-1, 2), (3, 0)]:
        board.apply_move(q, r)
    head.select_move(board)
    assert head.last_sims == 256
