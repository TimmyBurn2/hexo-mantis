# >300 justify (R8): the file is ONE claim — with both postures `null`, behaviour is identical
# to the tree before the keys existed — and each surface it is checked on sits beside its own
# ARMED mutation arm. Split them and an inertness suite rots into one that passes on dead code.
"""The inertness proof for the two early-strength eval postures.

The claim was narrow and total: with `eval.ply_cap_adjudication: null` and
`eval.strength_floor: null` the run's observable behaviour is identical to the tree before
these keys existed. One half no longer holds — a ruling ARMED `eval.strength_floor` on the
production set — so the claim splits: `ply_cap_adjudication` is INERT everywhere, and
`strength_floor` is ARMED on exactly `_ARMED_STRENGTH_FLOOR` and inert on the rest.

THE ARMED SET IS A CLOSED, NAMED CONSTANT AND NOT A PREDICATE OVER THE FILES: a row that read
whatever the configs happen to say would go green on an arming that arrived without a ruling,
which is the event this suite exists to refuse.

"Observable" is enumerated one test per surface: the shipped value, the resolvers, the round
spec, the sidecar result JSON, the event stream, and the arena's capped-game label (pinned next
door in `tests/arena/test_ply_cap_adjudication.py`). Each is paired with an ARMED mutation arm
in the same test or the one below it — without those, the suite proves only that the code is
unreachable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema.core import StrengthFloorConfig
from mantis.config.resolve.eval_posture import (
    PlyCapAdjudicationSpec,
    StrengthFloorSpec,
    resolve_ply_cap_adjudication,
    resolve_strength_floor,
)
from mantis.eval.rounds import RoundSpec, _REQUIRED_RESULT_KEYS
from mantis.eval.worker import _build_adjudicator, _round_result

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO / "configs"


def _config_paths() -> list[Path]:
    paths = sorted(discover_configs(_CONFIG_DIR))
    assert paths, "no configs discovered — this suite must never be vacuous"
    return paths


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
def test_every_committed_config_states_both_postures_and_states_them_disarmed(path) -> None:
    """The keys are PRESENT — a missing key is an error, not a disarmed posture — and their
    value is the explicit `null`, read off the FILE because the claim is about what was
    minted."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "ply_cap_adjudication" in raw["eval"], (
        f"{path.name}: the posture must be STATED; `extra='forbid'` plus a required field "
        f"means an absent key is a load error, and silence is never a posture (R1)"
    )
    assert "strength_floor" in raw["eval"]
    assert raw["eval"]["ply_cap_adjudication"] is None, (
        f"{path.name} arms ply-cap adjudication. Arming is a MINT-PREREG event with "
        f"operator-owned values; this suite is the instrument that refuses one arriving "
        f"without a ruling."
    )
    floor = raw["eval"]["strength_floor"]
    if path.name in _ARMED_STRENGTH_FLOOR:
        assert isinstance(floor, dict), (
            f"{path.name} is in the RULED armed set (R326 / Δ10.5) and must carry a real "
            f"block; got {floor!r}. A ruled arming that silently reverted to `null` would "
            "disarm the gate-integrity guard with nothing announcing it"
        )
        assert set(floor) == set(StrengthFloorConfig.model_fields), (
            f"{path.name}: an armed floor states ALL of its terms — the set is read off the "
            f"schema, so a fourth term cannot arrive half-minted; got {sorted(floor)}"
        )
    else:
        assert floor is None, (
            f"{path.name} arms the strength floor, and it is NOT in the ruled armed set "
            f"{sorted(_ARMED_STRENGTH_FLOOR)}. Arming is a MINT-PREREG event with "
            "operator-owned values; this row is what refuses one arriving without a ruling"
        )


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
def test_the_resolvers_return_none_except_where_a_ruling_armed_them(path) -> None:
    cfg = load_config(path)
    assert resolve_ply_cap_adjudication(cfg.eval) is None
    floor = resolve_strength_floor(cfg.eval)
    if path.name in _ARMED_STRENGTH_FLOOR:
        assert floor is not None, (
            f"{path.name} is in the ruled armed set but its resolver still answers None — the "
            "value would be minted and inert, which is the silently-disabled-knob class R1 and "
            "LAW-08 exist to kill"
        )
    else:
        assert floor is None


#: The configs a RULING has armed `eval.strength_floor` on. CLOSED and NAMED: it is widened only
#: by a mint act with a ruling behind it, and it is NOT derived from the files, because a
#: predicate over `configs/` would go vacuous on exactly the event this suite exists to catch.
_ARMED_STRENGTH_FLOOR = frozenset({"run6.yaml"})


def _armed_config():
    """`dev_example.yaml`'s raw payload with both postures armed, RE-VALIDATED through
    `RunConfig` rather than `model_copy`, so this also proves the armed shapes are
    config-legal."""
    from mantis.config.schema import RunConfig

    raw = yaml.safe_load((_CONFIG_DIR / "dev_example.yaml").read_text(encoding="utf-8"))
    raw["eval"]["ply_cap_adjudication"] = {
        "criterion": "longest_run_margin", "min_margin": 2,
    }
    raw["eval"]["strength_floor"] = {
        "probe_games": 4, "min_decisive_rate": 0.5, "min_winrate": 0.5,
    }
    return RunConfig.model_validate(raw)


def test_the_resolvers_BITE_on_an_armed_block() -> None:
    """The mutation arm for the resolvers: without it the `is None` assertions above would pass
    against a resolver that returned `None` unconditionally."""
    armed = _armed_config()
    ply = resolve_ply_cap_adjudication(armed.eval)
    floor = resolve_strength_floor(armed.eval)
    assert isinstance(ply, PlyCapAdjudicationSpec)
    assert (ply.criterion, ply.min_margin) == ("longest_run_margin", 2)
    assert isinstance(floor, StrengthFloorSpec)
    assert (floor.probe_games, floor.min_decisive_rate, floor.min_winrate) == (4, 0.5, 0.5)


def _spec_from(config_name: str, tmp_path: Path) -> RoundSpec:
    """Drive the PRODUCTION `_build_round_spec`: the claim is about what the pipeline threads,
    so a hand-built spec would prove nothing about the wiring."""
    from mantis.eval.pipeline import DrainCaps, EvalPipeline

    cfg = load_config(_CONFIG_DIR / config_name)
    pipeline = EvalPipeline(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1,
        eval_cfg=cfg.eval,
        caps=DrainCaps(final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
                       eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0),
        encoding=cfg.identity.encoding, run_id=cfg.run_id, spool_dir=tmp_path / "spool", game_record_dir=str(tmp_path / "spool") + "_games",
        ladder_state_path=tmp_path / "ladder.json", promotion=None, sink=None,
        # These fixtures assert the POSTURE fields, so the memory bound is `None` here.
        fused_graph_caps=None,
        inference_batching=None,
    )
    try:
        spec, _alloc, _gate, _path = pipeline._build_round_spec(
            _StubModel(), 1, None, round_id="r000001_1", round_idx=1, terminal=False,
        )
        return spec
    finally:
        pipeline.stop()


class _StubModel:
    """The snapshot writer needs `.arch` and a state dict; nothing here runs a forward."""

    arch = {"kind": "stub"}

    def state_dict(self) -> dict[str, Any]:
        return {}


def test_the_production_round_spec_carries_what_the_config_states(tmp_path, monkeypatch) -> None:
    """An armed config's round spec must CARRY the floor across the process seam; a spec that
    dropped it would leave the value minted, audited and inert."""
    monkeypatch.setattr(
        "mantis.eval.pipeline.write_model_snapshot", lambda model, path: str(path)
    )
    spec = _spec_from("run6.yaml", tmp_path)
    assert spec.ply_cap_adjudication is None
    assert "run6.yaml" in _ARMED_STRENGTH_FLOOR, "this row's premise is the ruled armed set"
    assert spec.strength_floor is not None, (
        "run5's armed floor did not reach the round spec — minted and inert"
    )
    assert spec.strength_floor.probe_games >= 1


def test_the_round_spec_survives_a_json_round_trip_on_both_arms() -> None:
    """The specs cross a process seam as JSON, so `to_dict`/`from_dict` must rebuild them on
    both arms — an armed one left as a raw mapping gives the worker attribute errors."""
    import json

    from mantis.eval.rounds import GameRecordTarget, GateSpec

    base = dict(
        round_index=0, round_id="r1", step=1, candidate_snapshot="c.pt", best_snapshot=None, best_step=None,
        encoding="gnn_axis_v1", worker_device="cpu",
        gate=GateSpec(stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=1, opening_book="b",
                      bootstrap_resamples=1, min_distinct_per_pair=1, seed_base=1,
                      run_gate=False),
        rung_jobs=[], random_floor_games=0, random_model_sims=1, sealbot_model_sims=1,
        seed_base=1, round_timeout_sec=1.0,
        result_path="r.json", progress_path="p.txt", ladder_bootstrap_resamples=1,
        ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1,
        game_record=None,
    )
    # `RoundSpec` carries the fused-forward memory bound in the SAME shape as the two postures;
    # its own round-trip is pinned elsewhere, so here it rides as `None`.
    disarmed = RoundSpec(**base, ply_cap_adjudication=None, strength_floor=None,
                         leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1, concurrency=1,
                         fused_graph_caps=None,
                         inference_batching=None)
    back = RoundSpec.from_dict(json.loads(json.dumps(disarmed.to_dict())))
    assert back.ply_cap_adjudication is None and back.strength_floor is None
    assert back == disarmed

    armed = RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128, leaf_build_threads=1, concurrency=1,
        **base,
        ply_cap_adjudication=PlyCapAdjudicationSpec(criterion="longest_run_margin",
                                                    min_margin=2),
        strength_floor=StrengthFloorSpec(probe_games=4, min_decisive_rate=0.5,
                                         min_winrate=0.5),
           fused_graph_caps=None,
           inference_batching=None,
    )
    back_armed = RoundSpec.from_dict(json.loads(json.dumps(armed.to_dict())))
    assert back_armed == armed
    assert isinstance(back_armed.strength_floor, StrengthFloorSpec)
    assert isinstance(back_armed.ply_cap_adjudication, PlyCapAdjudicationSpec)

    # `game_record` is the field the CHILD dereferences by ATTRIBUTE the moment a round starts:
    # left as a raw mapping it raises in a subprocess whose stderr nobody is reading.
    targeted = RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, max_plies=128,
        leaf_build_threads=1, concurrency=1,
        **{**base, "game_record": GameRecordTarget(record_dir="/tmp/games", run_id="r6")},
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=None, inference_batching=None,
    )
    back_target = RoundSpec.from_dict(json.loads(json.dumps(targeted.to_dict())))
    assert back_target == targeted
    assert isinstance(back_target.game_record, GameRecordTarget), (
        "the target came back as a raw mapping; the child would die on its first attribute read"
    )
    assert back_target.game_record.record_dir == "/tmp/games"


def _disarmed_spec() -> Any:
    class _S:
        step = 7
        ply_cap_adjudication = None
        strength_floor = None
    return _S()


def test_the_disarmed_result_payload_key_set_is_exactly_the_required_six() -> None:
    """Not "equivalent" — IDENTICAL, key set included, so a consumer that iterates the payload
    sees the same set it saw before these postures existed."""
    result = _round_result(
        _disarmed_spec(), gate_result=None, rungs_result={}, skipped_rungs=[],
        random_result={"games": 0, "wr": None}, floor_payload=None,
        adjudicator=_build_adjudicator(_disarmed_spec()),
    )
    assert set(result) == set(_REQUIRED_RESULT_KEYS)
    assert "strength_floor" not in result
    assert "ply_cap_adjudication" not in result


def test_the_armed_result_payload_GAINS_exactly_the_two_posture_keys() -> None:
    """The mutation arm for the payload: the extras are real, and they appear only when armed."""
    from mantis.arena.adjudicate import CRITERION_LONGEST_RUN, PlyCapAdjudicator

    class _S:
        step = 7
        ply_cap_adjudication = PlyCapAdjudicationSpec(criterion=CRITERION_LONGEST_RUN,
                                                      min_margin=2)
        strength_floor = StrengthFloorSpec(probe_games=4, min_decisive_rate=0.5,
                                           min_winrate=0.5)

    adj = _build_adjudicator(_S())
    assert isinstance(adj, PlyCapAdjudicator)
    result = _round_result(
        _S(), gate_result=None, rungs_result={}, skipped_rungs=[],
        random_result={"games": 0, "wr": None},
        floor_payload={"passed": False}, adjudicator=adj,
    )
    assert set(result) - set(_REQUIRED_RESULT_KEYS) == {
        "strength_floor", "ply_cap_adjudication"
    }
    assert result["ply_cap_adjudication"]["criterion"] == CRITERION_LONGEST_RUN
    assert result["ply_cap_adjudication"]["adjudicated"] == 0


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _FakePipeline:
    """The emitter under test, lifted off `EvalPipeline` without building a poller thread —
    `_emit_posture_events` runs as the PRODUCTION method, only the collaborators are stubs."""

    def __init__(self, sink) -> None:
        self._sink = sink
        self._floor_checked_total = 0
        self._floor_skipped_total = 0


def _emit(raw: dict[str, Any]) -> tuple[_RecordingSink, _FakePipeline]:
    from mantis.eval.pipeline import EvalPipeline

    sink = _RecordingSink()
    fake = _FakePipeline(sink)
    EvalPipeline._emit_posture_events(fake, {"round_id": "r1", "step": 3}, raw)
    return sink, fake


def test_a_disarmed_round_emits_no_posture_event_and_moves_no_counter() -> None:
    sink, fake = _emit({"rungs": {}, "gate": None})
    assert sink.events == []
    assert (fake._floor_checked_total, fake._floor_skipped_total) == (0, 0)


def test_an_armed_failing_round_emits_the_floor_event_with_BOTH_totals() -> None:
    """A fire rate needs its denominator: `skipped_total` alone cannot tell "the floor never
    fires" from "the floor never ran"."""
    sink, fake = _emit({"strength_floor": {"passed": False, "decisive_rate": 0.0}})
    assert [e["event"] for e in sink.events] == ["eval_strength_floor"]
    payload = sink.events[0]
    assert payload["round_id"] == "r1" and payload["step"] == 3
    assert payload["passed"] is False
    assert payload["checked_total"] == 1 and payload["skipped_total"] == 1
    assert (fake._floor_checked_total, fake._floor_skipped_total) == (1, 1)


def test_an_armed_PASSING_round_advances_only_the_checked_total() -> None:
    from mantis.eval.pipeline import EvalPipeline

    sink = _RecordingSink()
    fake = _FakePipeline(sink)
    EvalPipeline._emit_posture_events(
        fake, {"round_id": "r1", "step": 3}, {"strength_floor": {"passed": True}}
    )
    EvalPipeline._emit_posture_events(
        fake, {"round_id": "r2", "step": 4}, {"strength_floor": {"passed": False}}
    )
    assert [e["checked_total"] for e in sink.events] == [1, 2]
    assert [e["skipped_total"] for e in sink.events] == [0, 1]


def test_an_armed_adjudication_round_emits_its_own_tally_event() -> None:
    sink, _fake = _emit({"ply_cap_adjudication": {
        "criterion": "longest_run_margin", "min_margin": 2,
        "adjudicated": 5, "candidate": 3, "opponent": 1, "draw": 1,
    }})
    assert [e["event"] for e in sink.events] == ["eval_ply_cap_adjudication"]
    assert sink.events[0]["adjudicated"] == 5
    assert sink.events[0]["criterion"] == "longest_run_margin"
