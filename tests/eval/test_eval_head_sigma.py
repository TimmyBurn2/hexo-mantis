"""The eval head's σ is the run's: `q_rescale` rides `RoundSpec` beside `c_visit`/`c_scale` and
reaches the deploy head's tree, so the bar scores its root the way self-play did.
"""
from __future__ import annotations

import dataclasses

import pytest

from mantis.arena.deploy_head import DeployHeadPlayer
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.rounds import RoundSpec


def test_round_spec_requires_q_rescale_and_round_trips_it() -> None:
    names = {f.name for f in dataclasses.fields(RoundSpec)}
    assert "q_rescale" in names
    field = next(f for f in dataclasses.fields(RoundSpec) if f.name == "q_rescale")
    assert field.default is dataclasses.MISSING, "q_rescale is a σ term: never defaulted"


@pytest.mark.parametrize("rescale", [True, False])
def test_the_head_built_for_a_round_holds_the_round_sigma(rescale: bool) -> None:
    """`build_candidate_player` threads `q_rescale` into the SAME `configure_search` the
    self-play workers call; the tree reads it back."""

    # The graph arm binds the engine lazily (first expansion); `new_game` only configures the
    # tree, so no inference engine is needed to read the σ back.
    spec = lookup("gnn_axis_v1")
    player = worker.build_candidate_player(
        object(), 1, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=0.1,
        q_rescale=rescale, search_kind="gumbel", gumbel_m=4, gumbel_seed=0,
    )
    assert isinstance(player, DeployHeadPlayer)
    player.new_game()
    assert player._tree is not None
    c_visit, c_scale, got_rescale = player._tree.search_sigma
    assert (c_visit, c_scale, got_rescale) == (50.0, pytest.approx(0.1), rescale)
