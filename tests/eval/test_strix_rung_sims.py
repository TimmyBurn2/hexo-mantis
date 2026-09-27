"""The external rungs' sims row: `None` on production rounds, a named refusal for a strix or six job there."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from mantis.bots.protocol import RungUnresolvable
from mantis.eval import worker
from mantis.eval.rounds import GateSpec, RoundSpec, RungJob
from mantis.eval.worker import _model_sims_for_kind


def _spec(**over):
    base = dict(
        round_id="r", round_index=0, step=0, candidate_snapshot="", best_snapshot=None, best_step=None,
        encoding="gnn_axis_r8", worker_device="cpu", rung_jobs=[], random_floor_games=0,
        gate=GateSpec(stride=1, screen_games=0, confirm_games=0, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=512, opening_book="book_v1_s20260625_p4",
                      bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=1, run_gate=False, sequential=None),
        random_model_sims=96, seed_base=1, round_timeout_sec=1.0,
        result_path="", progress_path="",
        game_record=None, ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=None, leaf_batch_size=8, max_plies=256, c_visit=50.0, c_scale=1.0,
        q_rescale=True, search_kind="puct", gumbel_m=16, inference_batching=None, leaf_build_threads=1,
        concurrency=1, rung_concurrency=1,
    )
    base.update(over)
    return RoundSpec(**base)


def test_a_production_round_carries_no_strix_sims_and_a_strix_rung_on_it_is_a_named_refusal():
    spec = _spec()
    assert spec.rung_model_sims is None
    assert _model_sims_for_kind(spec, "random") == 96
    with pytest.raises(ValueError, match="rung_model_sims"):
        _model_sims_for_kind(spec, "strix")
    with pytest.raises(ValueError, match="sealbot rung was deleted"):
        _model_sims_for_kind(spec, "sealbot")


def test_the_strix_rung_tool_threads_its_sims_through_the_same_lookup():
    spec = _spec(rung_model_sims=256)
    assert _model_sims_for_kind(spec, "strix") == 256
    assert RoundSpec.from_dict(spec.to_dict()).rung_model_sims == 256, "the child reads it back"
    assert dataclasses.replace(spec, rung_model_sims=128).rung_model_sims == 128


def test_a_six_job_reads_the_same_candidate_sims_and_is_refused_on_a_production_round():
    with pytest.raises(ValueError, match="rung_model_sims"):
        _model_sims_for_kind(_spec(), "six")
    assert _model_sims_for_kind(_spec(rung_model_sims=256), "six") == 256


def test_a_six_job_resolves_at_its_own_nodes_on_the_rounds_worker_device(monkeypatch):
    seen: dict = {}

    def _capture(kind, **kw):
        seen.update(kind=kind, **kw)
        raise RungUnresolvable(rung=kind, reason="captured")

    monkeypatch.setattr(worker, "resolve_bot", _capture)
    job = RungJob(name="six", bot="six", variant="gen0030", opponent_sims=16, opening_book="b",
                  deploy_matched=True, games=2, bootstrap_resamples=1, bootstrap_ci_level=0.95, bootstrap_seed=1)
    with pytest.raises(RungUnresolvable):
        worker._play_rung_block(_spec(rung_model_sims=256, worker_device="cuda"), job, None, None,
                                encoding_spec=None, adjudicator=None, progress=None, games=None)
    assert seen == {"kind": "six", "opponent_sims": 16, "variant": "gen0030", "device": "cuda"}


@pytest.mark.parametrize("concurrency, made", [(1, 1), (8, 9)])
def test_the_rung_block_probes_its_opponent_first_and_closes_every_one_it_made(monkeypatch, concurrency, made):
    """The eager pair resolves the opponent before any game; under concurrency it never plays, so it closes at once."""
    bots: list = []

    class _Bot:
        def __init__(self) -> None:
            self.closed = False
            bots.append(self)

        def close(self) -> None:
            self.closed = True

    def _match(candidate, opponent, openings, *, player_factory, concurrency, **_kw):
        if concurrency > 1:
            assert bots[0].closed, "the probe's opponent is released before the games start"
            for _ in range(concurrency):
                player_factory()
        assert not any(b.closed for b in bots[1:]) and (concurrency > 1 or not bots[0].closed)
        return []

    sink = SimpleNamespace(sink=lambda *a, **k: (lambda record: None))
    monkeypatch.setattr(worker, "resolve_bot", lambda kind, **kw: _Bot)
    monkeypatch.setattr(worker, "build_candidate_player", lambda *a, **k: object())
    monkeypatch.setattr(worker, "round_openings", lambda *a, **k: [])
    monkeypatch.setattr(worker, "play_paired_match", _match)
    job = RungJob(name="six", bot="six", variant="gen0030", opponent_sims=16, opening_book="b",
                  deploy_matched=True, games=2, bootstrap_resamples=1, bootstrap_ci_level=0.95, bootstrap_seed=1)
    worker._play_rung_block(_spec(rung_model_sims=256, rung_concurrency=concurrency), job, None, None,
                            encoding_spec=None, adjudicator=None, progress=sink, games=sink)
    assert len(bots) == made and all(b.closed for b in bots)
