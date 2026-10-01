"""EVERY search kind spends EXACTLY `n_sims` descents, tactics off or on, whatever backed each (net, table, terminal)."""
from __future__ import annotations

import math
import random

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

from _dense_expand import InferStub, dense_expand
from _minted_puct import MINTED_PUCT

_STRIDE = 362

#: The leaf wiring alone, the design's leaf budgets: no root offence, no audit.
_LEAF_TACTICS = {"kind": "strict_turn", "leaf_turns": 2, "leaf_nodes": 64, "root_turns": 8, "root_nodes": 0,
                 "audit": None}

#: The solver terminals by kind, as `MCTSTree.tactics_counters` names them.
_TERMINALS = ("terminal_win1", "terminal_lost_on_cover", "terminal_strict_win", "terminal_six")


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


def _open_four_board() -> Board:
    """P1 to move with two stones against P2's open four: forced children, then decided leaves below them."""
    board = Board()
    for q, r in [(0, 0), (10, 5), (11, 5), (1, 0), (2, 0), (12, 5), (13, 5)]:
        board.apply_move(q, r)
    return board


def _five_board() -> Board:
    """P2 holds five on r = 3 and places two: its finish is a terminal child the peaked prior piles onto."""
    board = Board()
    for q, r in [(0, 0), (0, 3), (1, 3), (0, -3), (5, -5), (2, 3), (3, 3), (-5, 0), (-4, -2), (4, 3), (-3, 6), (-6, 2),
                 (-2, -6)]:
        board.apply_move(q, r)
    return board


def _peaked_infer(calls: list[int]) -> InferStub:
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


@pytest.mark.parametrize("tactics", [None, _LEAF_TACTICS], ids=["tactics-off", "tactics-on"])
@pytest.mark.parametrize("kind", ["puct", "gumbel"])
@pytest.mark.parametrize("n_sims", [64, 512])
def test_every_kind_spends_exactly_its_budget(kind: str, n_sims: int, tactics: dict | None) -> None:
    """PLANTED BREAK: drop `last_tt_hits()` from `_drive_puct`'s count and the transposing PUCT cases overspend."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        expand_fn=dense_expand(_peaked_infer(calls)), n_sims=n_sims, leaf_batch_size=8, c_visit=50.0,
        c_scale=1.0, q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7, tactics=tactics, puct=MINTED_PUCT,
    )
    player.new_game()
    if tactics is None:
        player.select_move(_mid_game_board(40, seed=3))
        _assert_every_descent_counted(player, calls, n_sims, f"{kind} at {n_sims}")
        return
    player.select_move(_open_four_board())
    rows = player.last_tactics
    assert rows is not None
    inline = sum(rows[k] for k in _TERMINALS) + rows["terminal_revisits"]
    assert player.last_sims == rows["descents"] == n_sims, f"{kind} at {n_sims}: {rows}"
    assert rows["served_leaves"] == len(calls), "every served leaf is one net call"
    assert rows["served_leaves"] + inline + rows["table_hits"] == rows["descents"], (
        f"{kind} at {n_sims}: the rows do not add up: {rows}"
    )
    assert inline > 0, f"{kind} at {n_sims}: no decided leaf, so the tactics-on case proves nothing: {rows}"
    # PUCT's two-stone turns transpose; Gumbel's forced descents have no table path.
    assert (rows["table_hits"] > 0) == (kind == "puct"), f"{kind} at {n_sims}: the table case: {rows}"


def _assert_every_descent_counted(player: DeployHeadPlayer, calls: list[int], n_sims: int, label: str) -> None:
    """The head spent `n_sims` descents, each one backup through the root; the net served no more than that."""
    assert player._tree is not None
    assert player.last_sims == n_sims == player._tree.root_visits(), (
        f"{label}: the head counted {player.last_sims} and the root saw {player._tree.root_visits()} backups"
    )
    assert len(calls) <= n_sims, f"{label}: served {len(calls)} leaves against a budget of {n_sims}"


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_plain_head_spends_its_budget_where_its_search_revisits_a_win(kind: str) -> None:
    """PLANTED BREAK: count a table-path terminal in no counter (the plain head's early end, 43 of 256) and PUCT reds."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        expand_fn=dense_expand(_peaked_infer(calls)), n_sims=256, leaf_batch_size=8, c_visit=50.0,
        c_scale=1.0, q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7, tactics=None, puct=MINTED_PUCT,
    )
    player.new_game()
    player.select_move(_five_board())
    _assert_every_descent_counted(player, calls, 256, f"{kind} at a revisited win")


def test_the_gumbel_head_spends_its_budget_on_a_second_board_too() -> None:
    """A second position, because the defect's rate depended on the board's transpositions."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        expand_fn=dense_expand(_peaked_infer(calls)), n_sims=256, leaf_batch_size=1, c_visit=50.0,
        c_scale=1.0, q_rescale=False, search_kind="gumbel", gumbel_m=16, gumbel_seed=11, tactics=None, puct=MINTED_PUCT,
    )
    player.new_game()
    player.select_move(_mid_game_board(12, seed=5))
    assert len(calls) == 256


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_head_reports_the_leaves_it_spent_as_its_own_counter(kind: str) -> None:
    """LADDER-1's budget witness reads `last_sims`, the head's count, never the caller's tally of infer calls."""
    calls: list[int] = []
    player = DeployHeadPlayer(
        expand_fn=dense_expand(_peaked_infer(calls)), n_sims=96, leaf_batch_size=8, c_visit=50.0,
        c_scale=1.0, q_rescale=True, search_kind=kind, gumbel_m=16, gumbel_seed=7, tactics=None, puct=MINTED_PUCT,
    )
    player.new_game()
    assert player.last_sims is None
    player.select_move(_mid_game_board(30, seed=3))
    _assert_every_descent_counted(player, calls, 96, kind)
