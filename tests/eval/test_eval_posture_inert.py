# >300 justify (R8), no line count stated (G-DFIX-4 / R192(e)). The file is ONE claim —
# "with both postures `null`, behaviour is identical to the tree before the keys existed" —
# and the claim is only as strong as the number of surfaces it is checked on, so the six
# surfaces enumerated below are conjuncts of a single assertion rather than six topics. Each
# also carries its own ARMED mutation arm immediately beside it; splitting the file would put
# an inertness assertion and the mutation that proves it non-vacuous in different files, which
# is exactly how an inertness suite rots into a suite that passes against dead code.
"""THE INERTNESS PROOF for the two early-strength eval postures (F-R-P2B-5).

The claim under test WAS narrow and total: **with `eval.ply_cap_adjudication: null` and
`eval.strength_floor: null` — the value every committed config mints — the run's observable
behaviour is identical to the tree before these keys existed.**

**ONE HALF OF THAT IS NO LONGER TRUE, AND THE ROWS BELOW SAY SO BY NAME RATHER THAN BY GOING
QUIET.** RECAL-SITTING-5's mint (R326, values R324(d), scope Δ10.5) ARMED `eval.strength_floor`
on the PRODUCTION SET — `run5.yaml`, `run6.yaml` and `shakedown_20260807.yaml` — and on
nothing else. run6 joins at its own mint: `RUN6_MINT_PREREG.md`'s `strength floor` row is
CONFIRM at run5's three armed values, carried unchanged, which is the ruling this row asks
for (R338 / R337(a)).
`ply_cap_adjudication` is untouched and still inert everywhere. So the claim splits:

  * `ply_cap_adjudication` — INERT on all seven, the original claim, unchanged;
  * `strength_floor` — ARMED on exactly `_ARMED_STRENGTH_FLOOR`, INERT on the rest.

**THE ARMED SET IS A CLOSED, NAMED CONSTANT AND NOT A PREDICATE OVER THE FILES.** That is the
whole value of this suite: a row that simply read whatever the configs happen to say would go
green on an arming that arrived without a ruling, which is the exact event it exists to refuse.
Widening `_ARMED_STRENGTH_FLOOR` is a mint act with a ruling behind it, not a maintenance edit.

"Observable" is enumerated rather than gestured at, one test per surface:

  1. the SHIPPED VALUE — every config under `configs/` STATES both postures, and states them
     `null` except where the ruling arms one, so the claim is about the tree as committed and
     not about a hypothetical config;
  2. the RESOLVERS — they return `None` for every committed config except the armed pair, where
     they must return a real spec, so nothing downstream is ever handed a spec by accident;
  3. the ROUND SPEC — the pipeline's own `_build_round_spec` carries what the config states;
  4. the RESULT JSON — the worker's sidecar payload carries EXACTLY the six required keys,
     no `strength_floor` and no `ply_cap_adjudication`, so a consumer iterating the key set
     sees the same set it saw before;
  5. the EVENT STREAM — `_emit_posture_events` emits nothing and moves no counter;
  6. the ARENA — the capped-game label is `"draw"` (that surface is pinned next door, in
     `tests/arena/test_ply_cap_adjudication.py`, against the same fixture as its mutation).

Every one of those is paired with an ARMED mutation in the same test or the one below it. An
inertness suite with no mutation arm proves only that the code is unreachable, which is the
one thing nobody doubted; the mutation arms are what make each `assert` above a statement
about a live seam.
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


# ── 1. the shipped value ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.name)
def test_every_committed_config_states_both_postures_and_states_them_disarmed(path) -> None:
    """R1: the keys are PRESENT (a missing key is an error, not a disarmed posture) and their
    value is the explicit `null`. Read off the FILE, not off the loaded model, because the
    claim is about what was minted."""
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


# ── 2. the resolvers ───────────────────────────────────────────────────────────────────
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


#: The configs a RULING has armed `eval.strength_floor` on. CLOSED, NAMED, and widened only by
#: a mint act with a ruling behind it — R326 / the RECAL-SITTING-5 forwarding §0.3, values
#: R324(d) (`probe_games 4`, `min_decisive_rate 0.25`, `min_winrate 0.0`), scope Δ10.5 (the
#: production pair only, because no armed-abort row exists for `strength_floor`).
#:
#: NOT derived from the files. A predicate over `configs/` would make every row below vacuous
#: on exactly the event this suite exists to catch: an arming that arrived without a ruling.
_ARMED_STRENGTH_FLOOR = frozenset({"run5.yaml", "run6.yaml", "shakedown_20260807.yaml"})


def _armed_config():
    """`dev_example.yaml`'s own raw payload with both postures armed, RE-VALIDATED through
    `RunConfig`. Going back through the schema (rather than `model_copy`, which skips
    validation) means this helper also proves the armed shapes are config-legal."""
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
    """The mutation arm for (2). Without it the two `is None` assertions above would pass
    against a resolver that returned `None` unconditionally."""
    armed = _armed_config()
    ply = resolve_ply_cap_adjudication(armed.eval)
    floor = resolve_strength_floor(armed.eval)
    assert isinstance(ply, PlyCapAdjudicationSpec)
    assert (ply.criterion, ply.min_margin) == ("longest_run_margin", 2)
    assert isinstance(floor, StrengthFloorSpec)
    assert (floor.probe_games, floor.min_decisive_rate, floor.min_winrate) == (4, 0.5, 0.5)


# ── 3. the round spec ──────────────────────────────────────────────────────────────────
def _spec_from(config_name: str, tmp_path: Path) -> RoundSpec:
    """Drive the PRODUCTION `_build_round_spec`, not a hand-built spec — the claim is about
    what the pipeline threads, so a hand-built spec would prove nothing about the wiring."""
    from mantis.eval.pipeline import DrainCaps, EvalPipeline

    cfg = load_config(_CONFIG_DIR / config_name)
    pipeline = EvalPipeline(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
        eval_cfg=cfg.eval,
        caps=DrainCaps(final_eval_drain_timeout_sec=1.0, eval_final_drain_safety_factor=1.0,
                       eval_final_drain_hard_cap_sec=1.0, terminal_eval_hard_cap_sec=1.0),
        encoding=cfg.identity.encoding, run_id=cfg.run_id, spool_dir=tmp_path / "spool",
        ladder_state_path=tmp_path / "ladder.json", promotion=None, sink=None,
        # F-816-10 D-1: resolved once in the parent, carried on every RoundSpec.
        # These fixtures assert the POSTURE fields, so the bound is `None` here.
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
    """`run5.yaml` is in the ruled armed set, so its round spec must CARRY the floor across the
    process seam. A spec that dropped it would leave the value minted, audited and inert — the
    knob reporting armed while nothing reads it."""
    monkeypatch.setattr(
        "mantis.eval.pipeline.write_model_snapshot", lambda model, path: str(path)
    )
    spec = _spec_from("run5.yaml", tmp_path)
    assert spec.ply_cap_adjudication is None
    assert "run5.yaml" in _ARMED_STRENGTH_FLOOR, "this row's premise is the ruled armed set"
    assert spec.strength_floor is not None, (
        "run5's armed floor did not reach the round spec — minted and inert"
    )
    assert spec.strength_floor.probe_games >= 1


def test_the_round_spec_survives_a_json_round_trip_on_both_arms() -> None:
    """The specs cross a process seam as JSON, so `to_dict`/`from_dict` must rebuild them —
    on the `None` arm (unchanged) and on the armed arm (rehydrated to the dataclass, not left
    as a raw mapping, which would silently give the worker attribute errors)."""
    import json

    from mantis.eval.rounds import GateSpec

    base = dict(
        round_id="r1", step=1, candidate_snapshot="c.pt", best_snapshot=None, best_step=None,
        encoding="gnn_axis_v1", worker_device="cpu",
        gate=GateSpec(stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
                      screen_confirm_lo=0.44, deploy_sims=1, opening_book="b",
                      bootstrap_resamples=1, min_distinct_per_pair=1, seed_base=1,
                      run_gate=False),
        rung_jobs=[], random_floor_games=0, random_model_sims=1, sealbot_model_sims=1,
        kraken_model_sims=1, strix_model_sims=1, seed_base=1, round_timeout_sec=1.0,
        result_path="r.json", progress_path="p.txt", ladder_bootstrap_resamples=1,
        ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1,
    )
    # F-816-10 D-1: `RoundSpec` carries the fused-forward memory bound in the SAME shape
    # as the two postures — a resolver-produced frozen dataclass that `asdict`/`from_dict`
    # round-trips. Its own round-trip (both arms) is pinned by
    # tests/selfplay/test_fused_graph_caps_construction.py; here it rides as `None`.
    disarmed = RoundSpec(**base, ply_cap_adjudication=None, strength_floor=None,
                         leaf_batch_size=1, c_visit=50.0, c_scale=1.0, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
                         fused_graph_caps=None,
                         inference_batching=None)
    back = RoundSpec.from_dict(json.loads(json.dumps(disarmed.to_dict())))
    assert back.ply_cap_adjudication is None and back.strength_floor is None
    assert back == disarmed

    armed = RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, amp_dtype="bf16", max_plies=128, leaf_build_threads=1,
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


# ── 4. the sidecar result JSON ─────────────────────────────────────────────────────────
def _disarmed_spec() -> Any:
    class _S:
        step = 7
        ply_cap_adjudication = None
        strength_floor = None
    return _S()


def test_the_disarmed_result_payload_key_set_is_exactly_the_required_six() -> None:
    """Not "equivalent" — IDENTICAL, key set included. A consumer that iterates the payload
    (a future display, the manifest checker, a diff over two runs' sidecars) must see the
    same set it saw before these postures existed."""
    result = _round_result(
        _disarmed_spec(), gate_result=None, rungs_result={}, skipped_rungs=[],
        random_result={"games": 0, "wr": None}, floor_payload=None,
        adjudicator=_build_adjudicator(_disarmed_spec()),
    )
    assert set(result) == set(_REQUIRED_RESULT_KEYS)
    assert "strength_floor" not in result
    assert "ply_cap_adjudication" not in result


def test_the_armed_result_payload_GAINS_exactly_the_two_posture_keys() -> None:
    """The mutation arm for (4): the extras are real, and they appear only when armed —
    the same discipline `_broken_result`'s `detail`/`exception_class` extras follow."""
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


# ── 5. the event stream ────────────────────────────────────────────────────────────────
class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _FakePipeline:
    """The emitter under test, lifted off `EvalPipeline` without building a poller thread.

    `_emit_posture_events` is an unbound method invoked against this stand-in, so the code
    exercised is the PRODUCTION method — only the collaborators are stubs.
    """

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
    """LAW-18 wants a FIRE RATE. `skipped_total` alone cannot tell "the floor never fires"
    from "the floor never ran", so the denominator rides the same payload."""
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
