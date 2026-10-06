"""The early stop: a stopped PUCT search plays the whole budget's move and counts its stops; off, it spends all."""
from __future__ import annotations

import random

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

from _dense_expand import dense_expand
from _minted_puct import MINTED_PUCT

_STRIDE = 362


def _board(n_stones: int, seed: int) -> Board:
    rng = random.Random(seed)
    board = Board()
    for _ in range(n_stones):
        if board.check_win():
            break
        legal = board.legal_moves()
        board.apply_move(*legal[rng.randrange(len(legal))])
    return board


def _peaked(board: Board) -> tuple[list[float], float]:
    """One legal cell takes most of the prior, so its visit lead outruns the budget left well before the end."""
    legal = board.legal_moves()
    policy = [0.0] * _STRIDE
    favourite = legal[board.zobrist_hash() % len(legal)]
    for q, r in legal:
        flat = board.to_flat(q, r)
        if flat < _STRIDE:
            policy[flat] = 0.9 if (q, r) == favourite else 0.1 / max(1, len(legal) - 1)
    return policy, ((board.zobrist_hash() % 1001) / 1000.0 - 0.5) * 0.2


def _head(early_stop: bool, n_sims: int = 256) -> DeployHeadPlayer:
    head = DeployHeadPlayer(expand_fn=dense_expand(_peaked), n_sims=n_sims, leaf_batch_size=8, c_visit=50.0,
                            c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=7, tactics=None,
                            puct=MINTED_PUCT, early_stop=early_stop)
    head.new_game()
    return head


@pytest.mark.parametrize("seed", range(12))
def test_a_stopped_search_plays_the_move_the_whole_budget_plays(seed: int) -> None:
    board = _board(20 + seed, seed)
    full, stopped = _head(False), _head(True)
    assert stopped.select_move(board) == full.select_move(board)
    assert full.last_sims == 256 and not full.last_stopped


def test_a_stop_fires_only_with_the_leader_fixed_and_is_counted() -> None:
    head = _head(True)
    fired = saved = 0
    for seed in range(12):
        head.select_move(_board(20 + seed, seed))
        assert head._tree is not None
        if head.last_stopped:
            top = head._tree.get_top_visits(2)
            left = 256 - int(head.last_sims or 0)
            assert left > 0 and (len(top) == 1 or top[0][1] - top[1][1] > left), (seed, top, left)
            fired, saved = fired + 1, saved + left
        else:
            assert head.last_sims == 256
    assert fired > 0, "the peaked net never let the leader run away, so this proves nothing"
    rows = head.search_rows()
    assert rows is not None and (rows["stop_fired"], rows["stop_saved"]) == (fired, saved)


def test_the_switch_off_spends_every_descent_and_counts_no_stop() -> None:
    head = _head(False)
    for seed in range(6):
        head.select_move(_board(20 + seed, seed))
        assert head.last_sims == 256 and not head.last_stopped
    rows = head.search_rows()
    assert rows is not None and (rows["stop_fired"], rows["stop_saved"]) == (0, 0)


@pytest.mark.parametrize(("rel", "switch"), [("src/mantis/eval/worker.py", False), ("tools/dash/engine/engines.py", False),
                                             ("src/mantis/diagnostics/acceptance_witness.py", False),
                                             ("tools/ladder/backends.py", True)])
def test_every_consumer_states_its_switch(rel: str, switch: bool) -> None:
    """The ladder plays the stop; cells, the in-run gate, the analyzer and the witness read whole budgets."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "build_candidate_player"]
    assert calls, rel
    for call in calls:
        value = {k.arg: k.value for k in call.keywords}.get("early_stop")
        assert isinstance(value, ast.Constant) and value.value is switch, (rel, call.lineno)
