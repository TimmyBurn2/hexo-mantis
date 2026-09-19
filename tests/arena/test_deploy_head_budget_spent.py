"""EVERY search kind spends EXACTLY `n_sims` leaves (R355(b); the old Gumbel driver served 0.19–0.42 of 512, A-1)."""
from __future__ import annotations

import math
import random

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer, InferFn

_STRIDE = 362


def _mid_game_board(n_stones: int, seed: int) -> Board:
    """A random legal position of `n_stones` stones under the real cadence (the red team's)."""
    rng = random.Random(seed)
    board = Board()
    for _ in range(n_stones):
        if board.check_win():
            break
        legal = board.legal_moves()
        q, r = legal[rng.randrange(len(legal))]
        board.apply_move(q, r)
    return board


def _peaked_infer(calls: list[int]) -> InferFn:
    """The red team's peaked net: a distance-decay prior and a position-dependent value."""
    def _infer(board: Board) -> tuple[list[float], float]:
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


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_head_reports_the_leaves_it_spent_as_its_own_counter(kind: str) -> None:
    """LADDER-1's budget witness reads `last_sims`, the head's count, never the caller's tally of infer calls."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        infer_fn=_peaked_infer(calls), n_sims=96, leaf_batch_size=8, c_visit=50.0,
        c_scale=1.0, q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7,
    )
    player.new_game()
    assert player.last_sims is None
    player.select_move(_mid_game_board(30, seed=3))
    assert player.last_sims == 96 == len(calls)
