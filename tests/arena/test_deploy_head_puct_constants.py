"""The deploy head builds every tree from the run's `selfplay.mcts` constants; the bridge holds none of its own."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from mantis._engine import Board, MCTSTree
from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.config.loader import load_config
from mantis.config.resolve.puct import PuctConstants, resolve_puct_constants

from _dense_expand import dense_expand
from _minted_puct import MINTED_PUCT

_STRIDE = 362
_DEV_EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"


def _infer(board: Board) -> tuple[list[float], float]:
    legal = board.legal_moves()
    policy = [0.0] * _STRIDE
    for q, r in legal:
        flat = board.to_flat(q, r)
        if flat < _STRIDE:
            policy[flat] = 1.0 / len(legal)
    return policy, ((board.zobrist_hash() % 2001) / 1000.0 - 1.0) * 0.5


def _visits(puct: PuctConstants) -> list[int]:
    head = DeployHeadPlayer(expand_fn=dense_expand(_infer), n_sims=96, leaf_batch_size=8, c_visit=50.0, c_scale=1.0,
                            q_rescale=False, search_kind="puct", gumbel_m=16, gumbel_seed=7, tactics=None, puct=puct)
    head.new_game()
    board = Board()
    for q, r in [(0, 0), (1, 0), (0, 1), (-1, 0), (2, -1)]:
        board.apply_move(q, r)
    head.select_move(board)
    assert head.last_root is not None
    return [child[3] for child in head.last_root[1]]


def test_the_bridge_tree_takes_no_constant_it_was_not_given() -> None:
    """PLANTED BREAK: restore a signature default and a tree builds on a number no config states."""
    with pytest.raises(TypeError):
        MCTSTree()  # type: ignore[call-arg]


def test_the_heads_c_puct_reaches_its_search() -> None:
    """PLANTED BREAK: build the head's tree from fixed constants and both searches read the same visits."""
    low = _visits(dataclasses.replace(MINTED_PUCT, c_puct=0.25))
    high = _visits(dataclasses.replace(MINTED_PUCT, c_puct=8.0))
    assert sum(low) == sum(high), "both searches spend the same budget"
    assert low != high, "the head searched the same tree at two c_puct values: the constant never reached it"


def test_the_resolver_reads_a_config_and_its_stamp_mapping_alike() -> None:
    """A deploy head built from a live config and one built from a checkpoint stamp search under one set of constants."""
    config = load_config(_DEV_EXAMPLE)
    assert resolve_puct_constants(config) == resolve_puct_constants(json.loads(json.dumps(config.model_dump())))
