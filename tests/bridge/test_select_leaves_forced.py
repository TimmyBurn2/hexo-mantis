"""`MCTSTree.select_leaves_forced` returns one leaf per forced child even on a transposition (A-1, R355(b))."""
from __future__ import annotations

import pytest

from mantis._engine import Board, MCTSTree

_STRIDE = 362


def _uniform(_board: Board) -> list[float]:
    return [1.0 / _STRIDE] * _STRIDE


def _expanded_root() -> MCTSTree:
    board = Board.with_encoding_name("gnn_axis_v1")
    tree = MCTSTree()
    tree.configure_search("gumbel", 50.0, 1.0, True)
    tree.new_game(board)
    (root,) = tree.select_leaves(1)
    tree.expand_and_backup([_uniform(root)], [0.0])
    return tree


def test_one_leaf_comes_back_per_forced_child_every_time() -> None:
    tree = _expanded_root()
    children = [row[1] for row in tree.get_root_children_info()][:3]
    # Descend into the same child 40 times: after the first, every descent below it lands on
    # nodes a sibling order already created — the transposition case the batch path short-cuts.
    for _ in range(40):
        leaves = tree.select_leaves_forced([children[0]])
        assert len(leaves) == 1, "a forced descent must return its leaf, TT hit or not"
        tree.expand_and_backup([_uniform(leaves[0])], [0.0])
    assert tree.get_top_visits(1)[0][1] == 40


def test_a_batch_of_forced_children_returns_one_leaf_each() -> None:
    tree = _expanded_root()
    children = [row[1] for row in tree.get_root_children_info()][:4]
    leaves = tree.select_leaves_forced(children)
    assert len(leaves) == 4
    tree.expand_and_backup([_uniform(b) for b in leaves], [0.0] * 4)


def test_a_child_the_root_does_not_own_is_refused() -> None:
    tree = _expanded_root()
    with pytest.raises(ValueError):
        tree.select_leaves_forced([10**6])
