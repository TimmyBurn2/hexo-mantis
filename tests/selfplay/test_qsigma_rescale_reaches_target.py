"""The pin: `selfplay.q_rescale` reaches the completed-Q TARGET BUILDER, set each way on a shipped config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mantis._engine import Board, MCTSTree
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.selfplay.hparams import SelfPlayHParams

_REPO = Path(__file__).resolve().parents[2]
#: Any census member: the rows set `selfplay.q_rescale` themselves, so two targets differ in that key alone.
_BASE = production_configs(_REPO)[0]
_STRIDE = 19 * 19 + 1
# The ring stores no Q (FORCED_MOVE_CENSUS §"What the record cannot answer"), so the two candidates'
# Q's are PLANTED through the real backup on one fixed root and read back in the root's view.
_A, _B = (2, 1), (1, 2)
_ROOT = ((0, 0), (1, 0), (0, 1))
_ROWS = {"dq_0.01": (0.105, 0.095), "dq_0.2": (0.2, 0.0)}


def _hparams(q_rescale: bool) -> SelfPlayHParams:
    raw = load_config(_BASE).model_dump()
    raw["selfplay"]["q_rescale"] = q_rescale
    return SelfPlayHParams.from_config(RunConfig.model_validate(raw).model_dump())


def _root_board() -> Board:
    board = Board.with_encoding_name("gnn_axis_r8")
    for q, r in _ROOT:
        board.apply_move(q, r)
    assert board.moves_remaining == 2, "the root's second stone keeps the child in the root's own view"
    return board


def _two_candidate_prior(board: Board) -> list[float]:
    prior = [0.0] * _STRIDE
    prior[board.to_flat(*_A)] = 0.5
    prior[board.to_flat(*_B)] = 0.5
    return prior


def _child_row(tree: MCTSTree, cell: tuple[int, int]) -> tuple[int, float, int, float]:
    row = next(r for r in tree.get_root_children_info() if tuple(r[0]) == cell)
    return row[1], row[2], row[3], row[4]


def _target(hp: SelfPlayHParams, q_a: float, q_b: float) -> tuple[np.ndarray, float, int, tuple[int, int]]:
    """The real target under `hp`'s σ: (masses, Δq read back, max_n, the two flat indices)."""
    board = _root_board()
    tree = MCTSTree()
    # THE setter the self-play workers call, fed from the hparams the config resolved to.
    tree.configure_search(hp.search_kind, hp.c_visit, hp.c_scale, hp.q_rescale)
    tree.new_game(board)
    (root,) = tree.select_leaves(1)
    tree.expand_and_backup([_two_candidate_prior(root)], [(q_a + q_b) / 2.0])
    for cell, value in ((_A, q_a), (_B, q_b)):
        idx = _child_row(tree, cell)[0]
        (leaf,) = tree.select_leaves_forced([idx])
        tree.expand_and_backup([[1.0 / _STRIDE] * _STRIDE], [value])
    _, prior_a, n_a, got_a = _child_row(tree, _A)
    _, prior_b, n_b, got_b = _child_row(tree, _B)
    assert prior_a == prior_b == pytest.approx(0.5)
    masses = np.asarray(tree.get_improved_policy(), dtype=np.float64)
    return masses, got_a - got_b, max(n_a, n_b), (board.to_flat(*_A), board.to_flat(*_B))


def _entropy(masses: np.ndarray) -> float:
    m = masses[masses > 0.0]
    return float(-(m * np.log(m)).sum())


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_rescale_on_and_off_build_different_targets_on_the_same_root(row: str) -> None:
    q_a, q_b = _ROWS[row]
    raw, _, _, _ = _target(_hparams(False), q_a, q_b)
    rescaled, _, _, _ = _target(_hparams(True), q_a, q_b)
    assert raw.shape == rescaled.shape == (_STRIDE,)
    assert not np.allclose(raw, rescaled, atol=1e-6), "the key did not reach the target builder"


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_under_raw_q_the_logit_gap_is_visit_scale_times_dq(row: str) -> None:
    hp = _hparams(False)
    assert hp.q_rescale is False
    q_a, q_b = _ROWS[row]
    masses, dq, max_n, (fa, fb) = _target(hp, q_a, q_b)
    assert dq == pytest.approx(q_a - q_b, abs=1e-3)
    gap = float(np.log(masses[fa]) - np.log(masses[fb]))
    assert gap == pytest.approx((hp.c_visit + max_n) * hp.c_scale * dq, abs=1e-6)


def test_the_near_equal_row_is_one_hot_under_rescale_and_not_under_raw_q() -> None:
    q_a, q_b = _ROWS["dq_0.01"]
    rescaled, _, _, _ = _target(_hparams(True), q_a, q_b)
    raw, _, _, _ = _target(_hparams(False), q_a, q_b)
    assert _entropy(rescaled) < 1e-3
    assert _entropy(raw) >= 1e-3
