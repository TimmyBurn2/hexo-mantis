"""The deploy head's root tactics: a decided root is played with no search, and the audit vets the searched move."""
from __future__ import annotations

import math
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
    rows = head.last_tactics
    assert rows is not None and rows["root_vetoes"] == 1
    # The audit spends solver nodes, not descents; a search whose every visit went to the veto is searched again.
    assert head.last_sims == 2 * (1 + rows["research_count"])


def test_the_inverted_known_bad_plays_into_a_proven_win() -> None:
    calls: list[int] = []
    infer = _peaked_on([(-4, -11), (-10, 0)], calls)
    board = _played(_FIX219[:16])
    plain = _head(_block(root_nodes=0, audit=None), infer, n_sims=2)
    assert plain.select_move(board) == (-4, -11)
    head = _head(_block(root_nodes=0, audit={**_AUDIT, "mode": "inverted"}), infer, n_sims=2)
    assert head.select_move(board) not in _HOLDS
    assert head.last_tactics is not None and head.last_tactics["audit_swaps"] == 1


#: P1 filler first; P2 builds a five (0..4, 0) capped at (-1, 0), whose window (1..6, 0) also needs (6, 0), and a
#: five (0..4, 6) capped at both ends; P1 then moves with two against the forced cells (5, 0), (5, 6), (6, 0).
_FORCED = [(0, -6), (0, 0), (1, 0), (-1, 0), (-1, 6), (2, 0), (3, 0), (6, 6), (-6, -6), (4, 0), (0, 6), (6, -6),
           (-6, 12), (1, 6), (2, 6), (0, 12), (6, 12), (3, 6), (4, 6)]


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_a_first_stone_that_leaves_the_head_lost_on_cover_is_swapped_for_the_block_that_holds(kind: str) -> None:
    calls: list[int] = []
    infer = _peaked_on([(6, 0), (5, 0), (5, 6)], calls)
    plain = _head(_block(root_nodes=0, audit=None), infer, kind, n_sims=2)
    assert plain.select_move(_played(_FORCED)) == (6, 0), "after (6, 0) no second stone covers both fives"
    head = _head(_block(root_nodes=0, audit=_AUDIT), infer, kind, n_sims=2)
    assert head.select_move(_played(_FORCED)) in {(5, 0), (5, 6)}
    assert head.last_tactics is not None and head.last_tactics["audit_swaps"] == 1


#: Five non-holds the prior ranks first, then a hold: at two descents every visited move is one of the five.
_LOSING_FIRST = [(-10, 0), (-11, 0), (-12, 0), (-3, -10), (-6, -10), (-4, -11)]


def test_a_root_whose_every_searched_move_is_vetoed_is_searched_again_without_them_and_plays_that_winner() -> None:
    """PLANTED BREAK: drop `select_move`'s re-search and the head plays the audit's best hold, a proven loss."""
    calls: list[int] = []
    board = _played(_FIX219[:16])
    head = _head(_block(root_nodes=0, audit=_AUDIT), _peaked_on(_LOSING_FIRST, calls), "puct", n_sims=2)
    assert head.select_move(board) == (-4, -11)
    rows = head.last_tactics
    assert rows is not None
    assert (rows["root_vetoes"], rows["best_holds"], rows["research_count"], rows["research_over_hold"]) == (1, 1, 1, 0)
    assert head.last_sims == rows["descents"] == 4, "both searches spend their budget, and both count"
    assert head.last_root is not None and (-10, 0) not in {c for c, *_ in head.last_root[1]}, "the re-search's root"


#: A distance-decay net's game, read by a fresh Gumbel head (64 descents, seed 9): its audit holds on a move the
#: target left with no mass, so the head searches again and that search's winner replaces the hold.
_GUMBEL_HOLD_OVERRIDE = [(-1, 0), (2, 1), (-2, -2), (-1, -3), (5, 1), (0, -6), (1, -6), (5, 2), (-2, -3), (-1, -2),
                         (-3, 0), (0, -3), (5, 0), (-2, -1), (-4, 1), (-6, 3)]


def _distance_decay(board: Board) -> tuple[list[float], float]:
    """The red team's peaked net (`test_deploy_head_budget_spent`): a distance-decay prior, a hashed value."""
    legal = board.legal_moves()
    recent = [(q, r) for (q, r, _p) in board.get_stones()][-4:]
    weights = [math.exp(-1.5 * min((abs(q - sq) + abs(r - sr) + abs((q + r) - (sq + sr))) / 2 for sq, sr in recent))
               for q, r in legal]
    policy = [0.0] * _STRIDE
    for (q, r), w in zip(legal, weights, strict=True):
        flat = board.to_flat(q, r)
        if flat < _STRIDE:
            policy[flat] = w / sum(weights)
    return policy, ((board.zobrist_hash() % 2001) / 1000.0 - 1.0) * 0.5


def test_a_gumbel_root_searched_again_over_an_audited_hold_is_counted() -> None:
    audit = {"turns": 4, "nodes": 256, "k": 4, "m": 4, "total_nodes": 4000, "mode": "hold"}
    block = {"kind": "strict_turn", "leaf_turns": 2, "leaf_nodes": 64, "root_turns": 4, "root_nodes": 2000,
             "audit": audit}
    head = DeployHeadPlayer(
        expand_fn=dense_expand(_distance_decay), n_sims=64, leaf_batch_size=8, c_visit=50.0, c_scale=1.0,
        q_rescale=True, search_kind="gumbel", gumbel_m=16, gumbel_seed=9, tactics=block,
    )
    head.new_game()
    head.select_move(_played(_GUMBEL_HOLD_OVERRIDE))
    rows = head.last_tactics
    assert rows is not None
    assert (rows["research_count"], rows["research_over_hold"], rows["best_holds"]) == (1, 1, 0), rows
    assert head.last_sims == rows["descents"] == 128
