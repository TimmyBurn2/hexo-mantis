"""The strix rung's sims row (RUNG-2): `None` on production rounds, a named refusal for a strix job there."""
from __future__ import annotations

import dataclasses

import pytest

from mantis.eval.rounds import GateSpec, RoundSpec
from mantis.eval.worker import _model_sims_for_kind


def _spec(**over):
    base = dict(
        round_id="r", round_index=0, step=0, candidate_snapshot="", best_snapshot=None, best_step=None,
        encoding="gnn_axis_r8", worker_device="cpu", rung_jobs=[], random_floor_games=0,
        gate=GateSpec(stride=1, screen_games=0, confirm_games=0, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=512, opening_book="book_v1_s20260625_p4",
                      bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=1, run_gate=False),
        random_model_sims=96, sealbot_model_sims=512, seed_base=1, round_timeout_sec=1.0,
        result_path="", progress_path="", ladder_bootstrap_resamples=1, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1, game_record=None, ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=None, leaf_batch_size=8, max_plies=256, c_visit=50.0, c_scale=1.0,
        q_rescale=True, search_kind="puct", gumbel_m=16, inference_batching=None, leaf_build_threads=1,
        concurrency=1, rung_concurrency=1,
    )
    base.update(over)
    return RoundSpec(**base)


def test_a_production_round_carries_no_strix_sims_and_a_strix_rung_on_it_is_a_named_refusal():
    spec = _spec()
    assert spec.strix_model_sims is None
    assert _model_sims_for_kind(spec, "sealbot") == 512
    with pytest.raises(ValueError, match="strix_model_sims"):
        _model_sims_for_kind(spec, "strix")


def test_the_strix_rung_tool_threads_its_sims_through_the_same_lookup():
    spec = _spec(strix_model_sims=256)
    assert _model_sims_for_kind(spec, "strix") == 256
    assert RoundSpec.from_dict(spec.to_dict()).strix_model_sims == 256, "the child reads it back"
    assert dataclasses.replace(spec, strix_model_sims=128).strix_model_sims == 128
