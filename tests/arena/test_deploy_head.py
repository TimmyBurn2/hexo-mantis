"""⊕ WP11-A arena — deploy-matched head, g=0 completed-Q argmax (design §a.2 deploy_head.py,
§b arena/test_deploy_head.py).

RED-at-import until IMPL writes `mantis.arena.deploy_head`. Frozen mechanics ported from
hexo_rl/hexo_rl/eval/deploy_strength_eval.py:109-207 (`DeployHeadBot`) driven through
hexo_rl/hexo_rl/eval/gumbel_search_py.py:178-227 (`run_gumbel_on_board`, gumbel_scale=0.0):
with g=0 every Gumbel(0,1) root-noise term is exactly 0, so the SH-winner score collapses
to `log(max(prior_i, 1e-8)) + sigma_i` with
`sigma_i = (c_visit + max_n_all) * c_scale * clamp(q_i, -1.0, 1.0)` and `max_n_all` = the max
visit count over ALL root children (gumbel_search_py.py:190-192,220-223). No candidate
elimination is exercised here (single-phase final-score argmax is what the oracle pins);
`get_root_children_info()` shape (bridge stub _engine.pyi): list of
`(coord: tuple[int,int], pool_idx: int, prior: float, visits: int, q: float)`.

ORACLE-CHOSEN SEAM: `mantis.arena.deploy_head.select_argmax_child(children_info, *,
c_visit: float, c_scale: float) -> tuple[int,int]` is the minimal pure surface implementing
that frozen formula (cited above) that this suite can hand-check with a calculator — it is
not a redesign, just the smallest testable unit of `DeployHeadPlayer`'s internal decision.
"""
from __future__ import annotations

import inspect
import math

from mantis.arena.deploy_head import DeployHeadPlayer, select_argmax_child

# c_visit/c_scale defaults mirror MCTSTree.get_improved_policy's own defaults
# (bridge stub _engine.pyi: c_visit: float = 50.0, c_scale: float = 1.0).
_C_VISIT = 50.0
_C_SCALE = 1.0


def test_gumbel_greedy_argmax_selection_on_synthetic_stats():
    # Deliberately constructed so the highest-VISIT child does NOT win — pinning that the
    # deploy head is a completed-Q sigma argmax, never a PUCT visit-count argmax.
    children_info = [
        ((0, 0), 0, 0.1, 100, 0.01),   # most-visited, low prior, low q
        ((1, 1), 1, 0.6, 5, 0.9),      # least-visited, high prior + q — must win
        ((2, 2), 2, 0.3, 50, 0.3),
    ]
    max_n_all = 100  # max visits over ALL children

    def _score(prior: float, q: float) -> float:
        sigma = (_C_VISIT + max_n_all) * _C_SCALE * max(-1.0, min(1.0, q))
        return math.log(max(prior, 1e-8)) + sigma

    score_a = _score(0.1, 0.01)   # log(0.1) + 150*0.01   =  -2.302585 +   1.5  =  -0.802585
    score_b = _score(0.6, 0.9)    # log(0.6) + 150*0.9    =  -0.510826 + 135.0  = 134.489174
    score_c = _score(0.3, 0.3)    # log(0.3) + 150*0.3    =  -1.203973 +  45.0  =  43.796027
    assert score_b > score_a and score_b > score_c, "sanity: hand-computed scores"

    winner = select_argmax_child(children_info, c_visit=_C_VISIT, c_scale=_C_SCALE)
    assert winner == (1, 1)


def test_argmax_ties_broken_by_score_not_input_order():
    # A second synthetic case with a different winner, guarding against an implementation
    # that always returns the first/last tuple regardless of score.
    children_info = [
        ((3, 3), 0, 0.9, 40, -0.5),
        ((4, 4), 1, 0.2, 40, 0.8),
    ]
    winner = select_argmax_child(children_info, c_visit=_C_VISIT, c_scale=_C_SCALE)
    assert winner == (4, 4)


def test_no_dirichlet_no_temperature_parameters_exist():
    sig = inspect.signature(DeployHeadPlayer.__init__)
    forbidden = {"temperature", "dirichlet", "epsilon", "gumbel_scale", "alpha"}
    present = forbidden & set(sig.parameters)
    assert not present, f"DeployHeadPlayer must not expose {present} — g=0 is structural, not a knob"


def test_deploy_head_is_deterministic_given_fixed_inference():
    from mantis._engine import Board

    def infer_fn(_board):
        # Fixed dummy policy+value: uniform logits over the 19x19+pass=362 action space
        # (registry.toml policy_logit_count for v6_live2_ls), value 0.0 — a stub good
        # enough to exercise search determinism without a real net.
        return [0.0] * 362, 0.0

    player_a = DeployHeadPlayer(infer_fn=infer_fn, n_sims=8, c_visit=_C_VISIT, c_scale=_C_SCALE)
    player_b = DeployHeadPlayer(infer_fn=infer_fn, n_sims=8, c_visit=_C_VISIT, c_scale=_C_SCALE)

    board_a = Board.with_encoding_name("v6_live2_ls")
    board_b = Board.with_encoding_name("v6_live2_ls")
    player_a.new_game()
    player_b.new_game()

    move_a = player_a.select_move(board_a)
    move_b = player_b.select_move(board_b)
    assert move_a == move_b, "identical fixed inference must yield an identical deploy move"
