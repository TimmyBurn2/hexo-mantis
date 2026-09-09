"""⊕ the deploy-matched head runs THE RUN'S OWN SEARCH.

WHAT THIS FILE USED TO PIN, AND WHY IT DOES NOT ANY MORE. `select_argmax_child` was the
"g=0 completed-Q argmax": a PUCT tree whose ROOT pick was the Gumbel scoring function with
its noise term set to zero. That is a THIRD algorithm — the tree descended by PUCT, no
Sequential Halving ran anywhere, and the move was chosen by a rule no self-play worker has
ever used — so the bar it produced was matched to nothing. The hybrid is deleted with its
oracle. What replaces it is the property the hybrid was standing in for: the head searches
with the run's `search.kind`, and each kind's move rule is that kind's own.

ORACLE: `DeployHeadPlayer(search_kind=...)` -> `MCTSTree.configure_search`, the SAME setter
`runner::game::run_worker_thread` calls. The kind is READ BACK off the head and off the
tree, so "the bar searched the way the run searched" is checkable rather than asserted.
"""
from __future__ import annotations

import inspect

import pytest

from mantis._engine import Board
from mantis.arena.deploy_head import DeployHeadPlayer

#: `selfplay.c_visit` / `selfplay.c_scale` as the committed configs mint them. STATED here
#: rather than imported so this file does not silently re-anchor when the keys are re-minted
#: — these are the head's inputs, not its subject.
_C_VISIT = 50.0
_C_SCALE = 1.0
#: v6_live2_ls's `policy_logit_count` (19x19 + pass).
_STRIDE = 362


def _uniform_infer(_board):
    """Fixed dummy policy+value: uniform logits over the action space, value 0.0 — a stub
    good enough to exercise search determinism without a real net."""
    return [0.0] * _STRIDE, 0.0


def _head(kind: str, **over):
    kwargs = dict(
        infer_fn=_uniform_infer,
        n_sims=8,
        leaf_batch_size=1,
        c_visit=_C_VISIT,
        c_scale=_C_SCALE,
        search_kind=kind,
        gumbel_m=4,
        gumbel_seed=20260909,
    )
    kwargs.update(over)
    return DeployHeadPlayer(**kwargs)


def _board():
    return Board.with_encoding_name("v6_live2_ls")


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_head_reports_and_configures_the_kind_it_was_given(kind: str):
    player = _head(kind)
    assert player.search_kind == kind
    player.new_game()
    # The TREE is the thing that has to be configured — a head that merely remembered the
    # string while its tree ran PUCT is exactly the coincidence this replaces.
    assert player._tree is not None
    assert player._tree.search_kind == kind


def test_an_unknown_kind_is_refused_by_the_engine_rather_than_defaulted():
    player = _head("mctx")
    with pytest.raises(ValueError, match="search.kind"):
        player.new_game()


def test_no_dirichlet_no_temperature_parameters_exist():
    sig = inspect.signature(DeployHeadPlayer.__init__)
    forbidden = {"temperature", "dirichlet", "epsilon", "gumbel_scale", "alpha"}
    present = forbidden & set(sig.parameters)
    assert not present, f"DeployHeadPlayer must not expose {present} — these are not knobs"


def test_the_search_regime_keys_are_required_and_have_no_defaults():
    """Every knob that decides the SEARCH is required — a default is a regime nobody minted.

    `c_visit`/`c_scale` carry AUDIT-1 F-39's history: they were defaulted on this
    signature, so the deploy-matched bar searched at numbers no config authored. The same
    reasoning covers `search_kind` and `gumbel_m`.
    """
    sig = inspect.signature(DeployHeadPlayer.__init__)
    for name in ("n_sims", "leaf_batch_size", "c_visit", "c_scale", "search_kind",
                 "gumbel_m", "gumbel_seed"):
        assert sig.parameters[name].default is inspect.Parameter.empty, (
            f"{name} must be REQUIRED — a default here is a search-regime constant nobody "
            f"minted, and LAW-15's deploy-matched claim stops being true with no config "
            f"diff to show for it"
        )


@pytest.mark.parametrize("kind", ["puct", "gumbel"])
def test_the_head_is_deterministic_given_fixed_inference(kind: str):
    """Both kinds replay. The Gumbel draw is SEEDED (LAW-15: a bar is a reproducible
    instrument), so two heads at one seed pick one move."""
    player_a, player_b = _head(kind), _head(kind)
    player_a.new_game()
    player_b.new_game()
    assert player_a.select_move(_board()) == player_b.select_move(_board())


def test_the_gumbel_seed_is_load_bearing():
    """A different seed is allowed to move the answer — otherwise the draw is not reaching
    the search and the `gumbel` arm is PUCT wearing another name."""
    moves = set()
    for seed in range(24):
        player = _head("gumbel", gumbel_seed=seed, n_sims=8, gumbel_m=8)
        player.new_game()
        moves.add(player.select_move(_board()))
    assert len(moves) > 1, (
        "24 seeds produced ONE move — the Gumbel noise is not reaching the root sampler"
    )


def test_the_puct_arm_plays_the_most_visited_child():
    """PUCT's deploy move is the most-visited root child — self-play's own deploy pick.

    Read off the tree the head just searched, so this pins the RULE rather than re-deriving
    it from the same numbers the head used.
    """
    player = _head("puct", n_sims=16)
    player.new_game()
    move = player.select_move(_board())
    top = player._tree.get_top_visits(1)
    assert top, "a searched root must have visited children"
    assert move == top[0][0]


def test_the_budget_is_leaves_and_the_root_is_one_of_them():
    """`n_sims` means N LEAVES of network work on both arms, root included.

    The stub counts its own calls, which is the only place the leaf count is observable
    from outside the engine.
    """
    for kind in ("puct", "gumbel"):
        calls = []

        def counting(board, _calls=calls):
            _calls.append(1)
            return _uniform_infer(board)

        player = _head(kind, infer_fn=counting, n_sims=12, leaf_batch_size=1)
        player.new_game()
        player.select_move(_board())
        assert len(calls) == 12, (
            f"{kind}: served {len(calls)} leaves against a budget of 12. The root's own "
            f"evaluation is charged on both arms — N means N leaves."
        )
