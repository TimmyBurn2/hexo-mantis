"""Within-turn tree reuse: stone 2 starts on stone 1's chosen subtree, and each stone spends its own budget in new descents."""
from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

from _dense_expand import dense_expand
from test_deploy_head_budget_spent import _LEAF_TACTICS, _peaked_infer
from _minted_puct import MINTED_PUCT

_REPO = Path(__file__).resolve().parents[2]


def _two_stone_board(n_stones: int, seed: int) -> Board:
    """A random position under the real cadence whose side to move holds both its stones."""
    rng = random.Random(seed)
    board = Board()
    while len(board.get_stones()) < n_stones or board.moves_remaining != 2:
        legal = board.legal_moves()
        board.apply_move(*legal[rng.randrange(len(legal))])
        assert not board.check_win(), "the fixture's seed must not end the game"
    return board


def _head(n_sims: int, tactics: dict | None, *, early_stop: bool = False, calls: list[int] | None = None) -> DeployHeadPlayer:
    head = DeployHeadPlayer(
        expand_fn=dense_expand(_peaked_infer([] if calls is None else calls)), n_sims=n_sims, leaf_batch_size=8,
        c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=7, tactics=tactics,
        puct=MINTED_PUCT, early_stop=early_stop, tree_reuse=True,
    )
    head.new_game()
    return head


@pytest.mark.parametrize("early_stop", [False, True], ids=["whole", "early-stop"])
@pytest.mark.parametrize("tactics", [None, _LEAF_TACTICS], ids=["tactics-off", "tactics-on"])
@pytest.mark.parametrize("n_sims", [64, 256])
def test_each_reused_stone_spends_its_budget_and_keeps_the_first_stones_visits(
    n_sims: int, tactics: dict | None, early_stop: bool,
) -> None:
    """PLANTED BREAK: count the kept visits against stone 2's budget (`sims_done = root_visits()`) and stone 2 underspends."""
    searched = 0
    for seed in range(12):
        calls: list[int] = []
        head = _head(n_sims, tactics, early_stop=early_stop, calls=calls)
        board = _two_stone_board(21 + 2 * seed, seed)
        first = head.select_move(board)
        if head.last_root is None:
            continue  # the root solver decided stone 1, so nothing was searched to keep
        kept = next(c[3] for c in head.last_root[1] if c[0] == first)
        board.apply_move(*first)
        head.select_move(board)
        if head.last_root is None:
            continue
        searched += 1
        tree, rows = head._tree, head.search_rows()
        assert tree is not None and head.last_sims is not None
        assert rows["reuse_reroots"] == 1 and rows["reuse_inherited"] == kept > 0, (seed, rows, kept)
        if head.last_stopped:
            top = tree.get_top_visits(2)
            left = n_sims - head.last_sims
            assert left > 0 and (len(top) == 1 or top[0][1] - top[1][1] > left), (seed, top, left)
        else:
            assert head.last_sims == n_sims, f"seed {seed}: stone 2 spends its budget in new descents"
        assert tree.root_visits() == kept + head.last_sims, f"seed {seed}: the root holds the kept and the new"
        if tactics is not None:
            assert head.last_tactics is not None and head.last_tactics["descents"] == head.last_sims, head.last_tactics
        assert len(calls) <= 2 * n_sims, f"seed {seed}: served {len(calls)} leaves over two stones"
    assert searched >= 3, f"only {searched} turns searched both stones, so the witness went unread"


def test_reuse_stays_inside_the_turn() -> None:
    """The opponent's two stones end it: our next stone 1 starts fresh, and only stone 2 re-roots."""
    head = _head(96, None)
    board = _two_stone_board(25, 3)
    rng = random.Random(3)
    second_stones = 0
    while second_stones < 3 and not board.check_win():
        for stone in (1, 2):
            board.apply_move(*head.select_move(board))
            assert head._tree is not None
            assert (head._tree.root_visits() == 96) == (stone == 1), (second_stones, stone)
            second_stones += stone == 2
            if board.check_win():
                break
        for _ in range(2):
            if not board.check_win():
                legal = board.legal_moves()
                board.apply_move(*legal[rng.randrange(len(legal))])
    assert second_stones == 3, "the game ended before three turns, so the fixture's seed must move"
    assert head.search_rows()["reuse_reroots"] == 3


def test_reuse_needs_puct() -> None:
    with pytest.raises(ValueError, match="tree_reuse needs search_kind 'puct'"):
        DeployHeadPlayer(expand_fn=dense_expand(_peaked_infer([])), n_sims=64, leaf_batch_size=8, c_visit=50.0,
                         c_scale=1.0, q_rescale=True, search_kind="gumbel", gumbel_m=16, gumbel_seed=7, tactics=None,
                         puct=MINTED_PUCT, tree_reuse=True)


def test_a_head_without_reuse_records_no_reuse_rows() -> None:
    head = DeployHeadPlayer(expand_fn=dense_expand(_peaked_infer([])), n_sims=64, leaf_batch_size=8, c_visit=50.0,
                            c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16, gumbel_seed=7, tactics=None,
                            puct=MINTED_PUCT)
    head.new_game()
    board = _two_stone_board(21, 0)
    board.apply_move(*head.select_move(board))
    head.select_move(board)
    assert head._tree is not None and head._tree.root_visits() == 64
    assert not any(k.startswith("reuse_") for k in head.search_rows())


@pytest.mark.parametrize("rel", ["src/mantis/eval/worker.py", "tools/dash/engine/engines.py",
                                 "src/mantis/diagnostics/acceptance_witness.py", "tools/ladder/backends.py"])
def test_only_the_ladder_can_reuse(rel: str) -> None:
    """Cells, the in-run gate, the analyzer and the witness search every stone fresh; the ladder states its switch."""
    tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "build_candidate_player"]
    assert calls, rel
    for call in calls:
        stated = {k.arg: k.value for k in call.keywords}.get("tree_reuse")
        names = {n.id for n in ast.walk(stated) if isinstance(n, ast.Name)} if stated is not None else set()
        if rel == "tools/ladder/backends.py":
            assert "LADDER_TREE_REUSE" in names, (rel, call.lineno)
        else:
            assert stated is None or (isinstance(stated, ast.Constant) and stated.value is False), (rel, call.lineno)
