"""The deploy head's root tactics: a decided root is played with no search, and the audit vets the searched move."""
from __future__ import annotations

from typing import Any

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

from _dense_expand import InferStub, dense_expand

_STRIDE = 362

#: T2's `fix219@17`: P2 to move with two stones holds a one-turn strict win at (-4, -11), (-4, -10).
_FIX219 = [(-1, -8), (-4, -8), (-4, -9), (-5, -8), (0, -10), (1, -13), (-6, -9), (-5, -7), (3, -13), (-6, -12),
           (-5, -10), (-6, -15), (4, -16), (-1, -14), (2, -15), (-13, 0), (-10, 0)]
#: With `_FIX219[:16]` on the board (P1 to move with one stone) only these stop P2's win; `(-10, 0)` does not.
_HOLDS = {(-4, -12), (-4, -11), (-4, -10), (-3, -12), (-2, -13)}
_AUDIT = {"turns": 2, "nodes": 256, "k": 4, "m": 2, "total_nodes": 65536, "mode": "hold"}


def _block(*, root_nodes: int, audit: dict[str, Any] | None) -> dict[str, Any]:
    """A tactics block with the leaf solver off, so only the root's tactics can see a blunder at depth one."""
    return {"kind": "strict_turn", "leaf_turns": 2, "leaf_nodes": 0, "root_turns": 2, "root_nodes": root_nodes,
            "audit": audit}


def _played(stones: list[tuple[int, int]]) -> Board:
    board = Board()
    for q, r in stones:
        board.apply_move(q, r)
    return board


def _peaked_on(cells: list[tuple[int, int]], calls: list[int]) -> InferStub:
    """A net that puts most of its prior on `cells` in order (halving each rank) and values every leaf at 0."""
    def _infer(board: Board) -> tuple[list[float], float]:
        calls.append(1)
        legal = board.legal_moves()
        policy = [0.0] * _STRIDE
        for q, r in legal:
            flat = board.to_flat(q, r)
            if flat < _STRIDE:
                policy[flat] = 1e-7
        for rank, (q, r) in enumerate(cells):
            flat = board.to_flat(q, r)
            if (q, r) in legal and flat < _STRIDE:
                policy[flat] = 2.0 ** -rank
        total = sum(policy)
        return [p / total for p in policy], 0.0
    return _infer


def _head(tactics: dict[str, Any], infer: InferStub, kind: str = "puct", n_sims: int = 64) -> DeployHeadPlayer:
    head = DeployHeadPlayer(
        expand_fn=dense_expand(infer), n_sims=n_sims, leaf_batch_size=8, c_visit=50.0, c_scale=1.0,
        q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7, tactics=tactics,
    )
    head.new_game()
    return head


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_a_proven_root_is_played_with_no_search_and_its_second_stone_at_the_next_call(kind: str) -> None:
    calls: list[int] = []
    head = _head(_block(root_nodes=256, audit=None), _peaked_on([], calls), kind)
    board = _played(_FIX219)
    assert head.select_move(board) == (-4, -11)
    assert (head.last_sims, head.last_root, calls) == (0, None, [])
    assert head.last_tactics is not None and head.last_tactics["root_proofs_found"] == 1
    board.apply_move(-4, -11)
    assert head.select_move(board) == (-4, -10)
    assert head.last_tactics is not None and head.last_tactics["proof_stones_played"] == 1
    assert calls == []


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_audit_swaps_a_searched_move_that_allows_a_proven_win_for_one_that_holds(kind: str) -> None:
    calls: list[int] = []
    infer = _peaked_on([(-10, 0), (-4, -11), (-2, -13)], calls)
    board = _played(_FIX219[:16])
    # Two descents: the kind picks from its prior (Gumbel's seeded draw can pass the peak); any non-hold loses.
    plain = _head(_block(root_nodes=0, audit=None), infer, kind, n_sims=2)
    assert plain.select_move(board) not in _HOLDS, "the searched move lets P2 win: the audit has work"
    head = _head(_block(root_nodes=0, audit=_AUDIT), infer, kind, n_sims=2)
    move = head.select_move(board)
    assert move in _HOLDS
    assert head.last_tactics is not None and head.last_tactics["root_vetoes"] == 1
    assert head.last_sims == 2, "the audit spends solver nodes, not descents"


def test_the_inverted_known_bad_plays_into_a_proven_win() -> None:
    calls: list[int] = []
    infer = _peaked_on([(-4, -11), (-10, 0)], calls)
    board = _played(_FIX219[:16])
    plain = _head(_block(root_nodes=0, audit=None), infer, n_sims=2)
    assert plain.select_move(board) == (-4, -11)
    head = _head(_block(root_nodes=0, audit={**_AUDIT, "mode": "inverted"}), infer, n_sims=2)
    assert head.select_move(board) not in _HOLDS
    assert head.last_tactics is not None and head.last_tactics["root_vetoes"] == 1
