"""Bounds census for the two `Block | None` eval postures (F-R-P2B-5).

`tests/config/test_eval_schema_bounds.py` cannot cover these: its payload mints both postures
disarmed, and a field inside a `null` block has no value to put out of domain. So the census
for the five inner leaves lives here, on an ARMED fixture, with the same shape as its sibling
— one out-of-domain rejection and one in-domain boundary acceptance per field.

Two claims beyond the bounds:

  * the DISJOINT-TYPES posture (R79) — `null` and a block are the two representable states,
    and there is no third spelling. In particular there is no `enabled:` boolean beside
    either block and no numeric disable sentinel inside one, because either would be a
    second authority over "is this armed" that can contradict the first.
  * NO CODE-SIDE DEFAULT (R1) — an absent key is an error that NAMES the key, not a silently
    disarmed posture. That distinction is the whole reason the shipped configs state `null`
    explicitly rather than omitting the key.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.schema import PlyCapAdjudicationConfig, RunConfig, StrengthFloorConfig

_REPO = Path(__file__).resolve().parents[2]
_DEV = _REPO / "configs" / "dev_example.yaml"

_ARMED_PLY = {"criterion": "longest_run_margin", "min_margin": 2}
_ARMED_FLOOR = {"probe_games": 4, "min_decisive_rate": 0.5, "min_winrate": 0.5}


def _payload(*, ply: Any = None, floor: Any = None) -> dict:
    """A committed config's own raw payload, with the two postures substituted.

    Derived from `configs/dev_example.yaml` rather than hand-written so this file cannot
    drift out of schema-completeness the way a transcribed payload does — a new required leaf
    anywhere in `RunConfig` arrives here for free.
    """
    raw = yaml.safe_load(_DEV.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["eval"]["ply_cap_adjudication"] = copy.deepcopy(ply)
    raw["eval"]["strength_floor"] = copy.deepcopy(floor)
    return raw


def test_the_baseline_payload_is_still_valid_disarmed_and_armed() -> None:
    """Guard the premise: neither posture may reject a healthy minted config, and the armed
    shapes this file mutates must themselves be legal or every rejection below is vacuous."""
    RunConfig.model_validate(_payload())
    RunConfig.model_validate(_payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR))


# ── the disjoint-types posture (R79) ───────────────────────────────────────────────────
def test_null_and_a_block_are_the_two_states_and_null_is_what_ships() -> None:
    cfg = RunConfig.model_validate(_payload())
    assert cfg.eval.ply_cap_adjudication is None
    assert cfg.eval.strength_floor is None

    armed = RunConfig.model_validate(_payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR))
    assert isinstance(armed.eval.ply_cap_adjudication, PlyCapAdjudicationConfig)
    assert isinstance(armed.eval.strength_floor, StrengthFloorConfig)


@pytest.mark.parametrize("key", ["ply_cap_adjudication", "strength_floor"])
def test_an_absent_posture_key_is_an_error_that_NAMES_it(key: str) -> None:
    """R1: missing key = error. `None` is a STATED posture; silence is not a posture at all,
    and a code-side default would be the duplicated-authority class R1 exists to kill."""
    raw = _payload()
    del raw["eval"][key]
    with pytest.raises(ValidationError) as ei:
        RunConfig.model_validate(raw)
    assert key in str(ei.value)
    assert "missing" in str(ei.value).lower()


@pytest.mark.parametrize("block,extra", [
    ("ply_cap_adjudication", {**_ARMED_PLY, "enabled": True}),
    ("strength_floor", {**_ARMED_FLOOR, "enabled": True}),
])
def test_no_enable_flag_may_be_smuggled_in_beside_the_terms(block: str, extra: dict) -> None:
    """A boolean beside the terms is a second authority over "is this armed" and can
    contradict the first. `extra="forbid"` makes it unrepresentable; this pins that it is the
    POSTURE blocks it is unrepresentable on, not merely somewhere in the tree."""
    raw = _payload()
    raw["eval"][block] = extra
    with pytest.raises(ValidationError) as ei:
        RunConfig.model_validate(raw)
    assert "enabled" in str(ei.value)


@pytest.mark.parametrize("block,partial", [
    ("ply_cap_adjudication", {"criterion": "longest_run_margin"}),
    ("strength_floor", {"probe_games": 4}),
])
def test_the_terms_travel_together_or_not_at_all(block: str, partial: dict) -> None:
    """R80's rule, on this fact. A criterion with no margin cannot be evaluated and a probe
    size with no bar measures nothing, so a HALF-armed block must not load."""
    raw = _payload()
    raw["eval"][block] = partial
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


# ── the bounds census ──────────────────────────────────────────────────────────────────
_OUT_OF_DOMAIN = [
    ("ply_cap_adjudication", "criterion", "centre_control"),
    ("ply_cap_adjudication", "criterion", "draw"),
    ("ply_cap_adjudication", "min_margin", 0),
    ("ply_cap_adjudication", "min_margin", -1),
    ("strength_floor", "probe_games", 0),
    ("strength_floor", "probe_games", -1),
    ("strength_floor", "min_decisive_rate", -0.1),
    ("strength_floor", "min_decisive_rate", 1.1),
    ("strength_floor", "min_winrate", -0.1),
    ("strength_floor", "min_winrate", 1.1),
]


@pytest.mark.parametrize("block,field,value", _OUT_OF_DOMAIN,
                         ids=[f"{b}.{f}={v}" for b, f, v in _OUT_OF_DOMAIN])
def test_an_out_of_domain_posture_value_is_rejected_by_name(block, field, value) -> None:
    """A value outside the mechanism's own range must be a NAMED load-time error, never a
    downstream surprise. `min_margin=0` is the load-bearing row: the margin is a signed
    difference between two equally-measured sides, so 0 means "measured equal" and a rule
    that awarded on it would award every capped game to whichever side the comparison
    happened to test first."""
    armed = {"ply_cap_adjudication": _ARMED_PLY, "strength_floor": _ARMED_FLOOR}
    raw = _payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR)
    block_value = dict(armed[block])
    block_value[field] = value
    raw["eval"][block] = block_value
    with pytest.raises(ValidationError) as ei:
        RunConfig.model_validate(raw)
    assert field in str(ei.value)


_IN_DOMAIN_BOUNDARY = [
    ("ply_cap_adjudication", "criterion", "immediate_win_margin"),
    ("ply_cap_adjudication", "min_margin", 1),
    ("strength_floor", "probe_games", 1),
    ("strength_floor", "min_decisive_rate", 0.0),
    ("strength_floor", "min_decisive_rate", 1.0),
    ("strength_floor", "min_winrate", 0.0),
    ("strength_floor", "min_winrate", 1.0),
]


@pytest.mark.parametrize("block,field,value", _IN_DOMAIN_BOUNDARY,
                         ids=[f"{b}.{f}={v}" for b, f, v in _IN_DOMAIN_BOUNDARY])
def test_the_boundary_values_of_each_field_LOAD(block, field, value) -> None:
    """The other half of the census: a bound that also rejects legal values is a bug of its
    own. `min_winrate: 0.0` in particular must load — it is the explicit "no win-rate bar"
    posture, said out loud in the config rather than by an absent key."""
    armed = {"ply_cap_adjudication": _ARMED_PLY, "strength_floor": _ARMED_FLOOR}
    raw = _payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR)
    block_value = dict(armed[block])
    block_value[field] = value
    raw["eval"][block] = block_value
    cfg = RunConfig.model_validate(raw)
    assert getattr(getattr(cfg.eval, block), field) == value
