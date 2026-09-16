"""EVERY search kind spends EXACTLY `n_sims` leaves — the check R355(b) makes a suite section.

THE DEFECT, measured (INVESTIGATION-1 A-1): `_drive_gumbel` called `select_leaves(1)` under a
forced root child; a transposition hit expanded the leaf inline and returned nothing, the loop
read that as exhaustion, and the Gumbel deploy head served 0.19–0.42 of a 512-sim budget on a
40-stone board under a peaked prior. Every Gumbel-head reading since 2026-09-09 (F-48, F-50,
F-51, CARD-DEPLOY-HEAD-BUDGET, GAME_QUALITY's Gumbel line) was through it. PUCT spent exactly.
The witness counts LEAVES EVALUATED, the only place the budget is observable from outside.
"""
from __future__ import annotations

import math
import random

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

_STRIDE = 362


def _mid_game_board(n_stones: int, seed: int) -> Board:
    """A random legal position of `n_stones` stones under the real cadence — the red team's
    `random_board`, on the rules-default board."""
    rng = random.Random(seed)
    board = Board()
    for _ in range(n_stones):
        if board.check_win():
            break
        legal = board.legal_moves()
        q, r = legal[rng.randrange(len(legal))]
        board.apply_move(q, r)
    return board


def _peaked_infer(calls: list[int]):
    """The red team's peaked net: a prior decaying with distance from the last four stones and a
    value that DEPENDS on the position (off its hash), so completed Qs differ between siblings
    and the descent re-walks paths — the shape that read 0.19 of 512 through the old driver."""
    def _infer(board):
        calls.append(1)
        legal = board.legal_moves()
        recent = [(q, r) for (q, r, _p) in board.get_stones()][-4:]
        weights = []
        for q, r in legal:
            d = min(
                (abs(q - sq) + abs(r - sr) + abs((q + r) - (sq + sr))) / 2 for sq, sr in recent
            ) if recent else 0
            weights.append(math.exp(-1.5 * d))
        total = sum(weights)
        policy = [0.0] * _STRIDE
        for (q, r), w in zip(legal, weights, strict=True):
            flat = board.to_flat(q, r)
            if flat < _STRIDE:
                policy[flat] = w / total
        value = ((board.zobrist_hash() % 2001) / 1000.0 - 1.0) * 0.5
        return policy, value
    return _infer


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
@pytest.mark.parametrize("n_sims", [64, 512])
def test_every_kind_spends_exactly_its_budget(kind: str, n_sims: int) -> None:
    calls: list[int] = []
    player = DeployHeadPlayer(
        infer_fn=_peaked_infer(calls), n_sims=n_sims, leaf_batch_size=8, c_visit=50.0,
        c_scale=1.0, q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7,
    )
    player.new_game()
    player.select_move(_mid_game_board(40, seed=3))
    assert len(calls) == n_sims, (
        f"{kind} at {n_sims}: served {len(calls)} leaves against a budget of {n_sims}"
    )


def test_the_gumbel_head_spends_its_budget_on_a_second_board_too() -> None:
    """A second position, because the defect's rate depended on the board's transpositions."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        infer_fn=_peaked_infer(calls), n_sims=256, leaf_batch_size=1, c_visit=50.0,
        c_scale=1.0, q_rescale=False, search_kind="gumbel", gumbel_m=16, gumbel_seed=11,
    )
    player.new_game()
    player.select_move(_mid_game_board(12, seed=5))
    assert len(calls) == 256
