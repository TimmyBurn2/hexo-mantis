"""The sequential promotion gate: the LLR, its bounds, the loop's stops, and once through a real round."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from _fused_caps import CAPS
from mantis.eval.sequential import (
    SD_FLOOR,
    SequentialGateSpec,
    gsprt_decision,
    gsprt_llr,
    llr_bounds,
    run_sequential_gate,
)


def _spec(**over: object) -> SequentialGateSpec:
    base = dict(mu0=0.52, mu1=0.62, alpha=0.05, beta=0.10, check_every_pairs=8, min_pairs=16,
                max_pairs=104, at_max_pairs="sign")
    base.update(over)
    return SequentialGateSpec(**base)  # type: ignore[arg-type]


def _records(pair_scores: list[float], start: int) -> list[dict]:
    """Two legs per opening whose draw-aware mean is the pair score, keyed like the arena's records."""
    out = []
    for i, score in enumerate(pair_scores):
        opening = f"o{start + i:04d}"
        legs = {1.0: ("p1", "p1"), 0.75: ("p1", "draw"), 0.5: ("p1", "p2"), 0.25: ("draw", "p2"),
                0.0: ("p2", "p2")}[score]
        for leg, winner in enumerate(legs):
            out.append({"opening_id": opening, "winner": winner, "candidate_color": 1,
                        "p1": "cand", "p2": "anchor", "moves": ((leg, i),)})
    return out


def test_the_llr_is_van_den_berghs_normalized_t_over_pairs() -> None:
    scores = [1.0] * 16
    llr = gsprt_llr(scores, mu0=0.52, mu1=0.62)
    sd = SD_FLOOR  # sixteen identical scores: the sd floor, not a division by zero
    t, t0, t1 = (1.0 - 0.5) / sd, (0.52 - 0.5) / sd, (0.62 - 0.5) / sd
    want = 0.5 * 16 * math.log((1 + (t - t0) ** 2) / (1 + (t - t1) ** 2))
    assert llr == pytest.approx(want)
    assert llr > 0
    # exactly midway between the hypotheses the evidence is zero, whatever the spread
    assert gsprt_llr([0.5, 0.75] * 8, mu0=0.5, mu1=0.75) == pytest.approx(0.0)
    assert gsprt_llr([0.0] * 16, mu0=0.52, mu1=0.62) < 0


def test_the_bounds_are_walds() -> None:
    lower, upper = llr_bounds(alpha=0.05, beta=0.10)
    assert upper == pytest.approx(math.log(0.90 / 0.05))
    assert lower == pytest.approx(math.log(0.10 / 0.95))


def test_the_decision_maps_the_llr_to_accept_reject_continue() -> None:
    lower, upper = llr_bounds(alpha=0.05, beta=0.10)
    assert gsprt_decision(upper + 1e-9, lower, upper, at_max=False, at_max_pairs="sign") == "accept"
    assert gsprt_decision(lower - 1e-9, lower, upper, at_max=False, at_max_pairs="sign") == "reject"
    assert gsprt_decision(0.0, lower, upper, at_max=False, at_max_pairs="sign") == "continue"
    assert gsprt_decision(0.5, lower, upper, at_max=True, at_max_pairs="sign") == "accept"
    assert gsprt_decision(0.0, lower, upper, at_max=True, at_max_pairs="sign") == "reject"
    assert gsprt_decision(-0.5, lower, upper, at_max=True, at_max_pairs="sign") == "reject"


def test_at_max_pairs_promote_accepts_an_undecided_candidate_at_the_cap_and_nowhere_else() -> None:
    # At the cap the bounds still decide before it fires, and a reject is still a reject.
    lower, upper = llr_bounds(alpha=0.05, beta=0.10)
    assert gsprt_decision(-0.5, lower, upper, at_max=True, at_max_pairs="promote") == "accept"
    assert gsprt_decision(0.0, lower, upper, at_max=True, at_max_pairs="promote") == "accept"
    assert gsprt_decision(lower - 1e-9, lower, upper, at_max=True, at_max_pairs="promote") == "reject"
    assert gsprt_decision(-0.5, lower, upper, at_max=False, at_max_pairs="promote") == "continue"
    assert gsprt_decision(upper + 1e-9, lower, upper, at_max=False, at_max_pairs="promote") == "accept"


class _Scripted:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[int, int]] = []

    def __call__(self, start: int, end: int) -> list[dict]:
        self.calls.append((start, end))
        return _records(self.scores[start:end], start)


def test_a_strong_candidate_is_accepted_at_the_first_check() -> None:
    player = _Scripted([1.0] * 104)
    records, verdict = run_sequential_gate(player, _spec())
    assert verdict.decision == "promote" and verdict.stopped == "accept"
    assert verdict.pairs_played == 16 and player.calls == [(0, 16)]
    assert len(records) == 32


def test_a_weak_candidate_is_rejected_at_the_first_check() -> None:
    player = _Scripted([0.0] * 104)
    _records_, verdict = run_sequential_gate(player, _spec())
    assert verdict.decision == "reject" and verdict.stopped == "reject"
    assert verdict.pairs_played == 16


def test_an_undecided_candidate_runs_to_max_pairs_in_batches_and_the_sign_decides() -> None:
    # exactly between mu0 and mu1 at every check: the LLR is 0, never crosses, the max decides
    player = _Scripted([0.5, 0.75] * 52)
    _records_, verdict = run_sequential_gate(player, _spec(mu0=0.5, mu1=0.75))
    assert verdict.stopped == "max" and verdict.pairs_played == 104
    assert verdict.decision == "reject", "an LLR of exactly 0 at the max is not evidence for H1"
    assert player.calls == [(0, 16)] + [(n, n + 8) for n in range(16, 104, 8)]
    assert verdict.checks == 12


def test_under_promote_the_same_undecided_candidate_is_promoted_at_the_cap() -> None:
    player = _Scripted([0.5, 0.75] * 52)
    _records_, verdict = run_sequential_gate(player, _spec(mu0=0.5, mu1=0.75, at_max_pairs="promote"))
    assert verdict.stopped == "max" and verdict.pairs_played == 104 and verdict.decision == "promote"
    # a clearly worse candidate is still rejected, before the cap
    _records_, verdict = run_sequential_gate(_Scripted([0.0] * 104), _spec(at_max_pairs="promote"))
    assert verdict.decision == "reject" and verdict.stopped == "reject" and verdict.pairs_played == 16


def test_the_last_batch_is_clipped_to_max_pairs() -> None:
    player = _Scripted([0.5, 0.75] * 52)
    _records_, verdict = run_sequential_gate(player, _spec(mu0=0.5, mu1=0.75, min_pairs=10,
                                                            check_every_pairs=7, max_pairs=20))
    assert player.calls == [(0, 10), (10, 17), (17, 20)] and verdict.pairs_played == 20


def test_the_spec_refuses_a_band_that_cannot_stop() -> None:
    with pytest.raises(ValueError, match="mu1"):
        _spec(mu0=0.6, mu1=0.55)
    with pytest.raises(ValueError, match="max_pairs"):
        _spec(min_pairs=32, max_pairs=16)
    with pytest.raises(ValueError, match="check_every_pairs"):
        _spec(check_every_pairs=0)
    with pytest.raises(ValueError, match="at_max_pairs"):
        _spec(at_max_pairs="llr")


_ENC = "gnn_axis_v1"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(seed: int):
    import torch

    from mantis.encoding import lookup
    from mantis.model import GnnArch, build_net

    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _round_spec(tmp_path, sequential: dict | None):
    from mantis.config.resolve.inference_batching import InferenceBatchingSpec
    from mantis.eval.rounds import GateSpec, RoundSpec
    from mantis.eval.snapshot import write_model_snapshot

    candidate, best = tmp_path / "candidate.pt", tmp_path / "best.pt"
    write_model_snapshot(_net(seed=1), candidate)
    write_model_snapshot(_net(seed=2), best)
    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55, screen_confirm_lo=0.44,
        deploy_sims=2, opening_book=_BOOK, bootstrap_resamples=10, min_distinct_per_pair=1,
        seed_base=_SEED, run_gate=True, sequential=sequential,
    )
    return RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16,
        max_plies=32, leaf_build_threads=1, concurrency=1, rung_concurrency=1,
        round_index=0, round_id="sequential_wiring", step=1, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=_ENC, worker_device="cpu",
        gate=gate, rung_jobs=[], random_floor_games=0,
        random_model_sims=2, seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"), progress_path=str(tmp_path / "progress.txt"),
        game_record=None, ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=CAPS,
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )


def test_the_worker_plays_the_sequential_gate_in_batches_and_reports_the_rule(tmp_path, monkeypatch) -> None:
    """The armed block reaches `_play_gate_block`: batches under `gate_sequential`, the gate result naming the rule."""
    from mantis.eval import worker

    calls: list[int] = []
    real = worker.play_paired_match

    def _spy(candidate, opponent, openings, **kwargs):
        calls.append(len(list(openings)))
        return real(candidate, opponent, openings, **kwargs)

    monkeypatch.setattr(worker, "play_paired_match", _spy)
    seq = {"mu0": 0.52, "mu1": 0.62, "alpha": 0.05, "beta": 0.10,
           "check_every_pairs": 1, "min_pairs": 2, "max_pairs": 3, "at_max_pairs": "sign"}
    result = worker.run_round(_round_spec(tmp_path, seq))

    gate = result["gate"]
    assert gate["rule"] == "gsprt"
    assert gate["stopped"] in ("accept", "reject", "max") and isinstance(gate["llr"], float)
    assert 2 <= gate["pairs_played"] <= 3 and gate["n_pooled"] == 2 * gate["pairs_played"]
    assert gate["n_confirm"] == 0 and gate["wr_confirm"] == gate["wr_screen"]
    assert calls[0] == 2 and all(c == 1 for c in calls[1:]) and len(calls) <= 2
    phases = {json.loads(line)["phase"] for line in Path(tmp_path / "progress.txt").read_text(encoding="utf-8").splitlines()}
    assert phases == {"gate_sequential"}, phases
    expected = {"accept": True, "reject": False, "max": gate["llr"] > 0.0}[gate["stopped"]]
    assert gate["promoted"] is (expected and not gate["low_power"])


def test_a_null_block_keeps_the_screen_confirm_rule(tmp_path) -> None:
    from mantis.eval import worker

    gate = worker.run_round(_round_spec(tmp_path, None))["gate"]
    assert gate["rule"] == "screen_confirm" and gate["llr"] is None and gate["pairs_played"] is None
