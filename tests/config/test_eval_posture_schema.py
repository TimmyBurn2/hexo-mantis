"""Bounds census for the two `Block | None` eval postures.

The sibling bounds suite cannot cover these: its payload mints both postures disarmed, and a
field inside a `null` block has no value to put out of domain. So the census for the inner
leaves lives here on an armed fixture, one out-of-domain rejection and one in-domain boundary
acceptance per field.

Two claims beyond the bounds: `null` and a block are the only two representable states, with
no `enabled:` boolean and no numeric disable sentinel that could be a second authority over
"is this armed"; and an absent key is an error that NAMES the key, never a silently disarmed
posture, which is why the shipped configs state `null` explicitly.
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
    """Return a committed config's own raw payload with the two postures substituted.

    Derived rather than hand-written, so a new required leaf anywhere in `RunConfig` arrives
    here for free instead of leaving this file out of schema-completeness.
    """
    raw = yaml.safe_load(_DEV.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["eval"]["ply_cap_adjudication"] = copy.deepcopy(ply)
    raw["eval"]["strength_floor"] = copy.deepcopy(floor)
    return raw


def test_the_baseline_payload_is_still_valid_disarmed_and_armed() -> None:
    """Guard the premise: the disarmed and armed baseline payloads must both validate."""
    RunConfig.model_validate(_payload())
    RunConfig.model_validate(_payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR))


def test_null_and_a_block_are_the_two_states_and_null_is_what_ships() -> None:
    cfg = RunConfig.model_validate(_payload())
    assert cfg.eval.ply_cap_adjudication is None
    assert cfg.eval.strength_floor is None

    armed = RunConfig.model_validate(_payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR))
    assert isinstance(armed.eval.ply_cap_adjudication, PlyCapAdjudicationConfig)
    assert isinstance(armed.eval.strength_floor, StrengthFloorConfig)


@pytest.mark.parametrize("key", ["ply_cap_adjudication", "strength_floor"])
def test_an_absent_posture_key_is_an_error_that_NAMES_it(key: str) -> None:
    """Prove an absent posture key is an error naming it: `None` is stated, silence is not a posture."""
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
    """Prove no `enabled` flag may sit beside the terms: it would be a second authority over arming."""
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
    """Prove a half-armed block does not load: a criterion with no margin cannot be evaluated."""
    raw = _payload()
    raw["eval"][block] = partial
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


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
    """Prove an out-of-domain posture value is rejected by name at load time.

    `min_margin=0` is the load-bearing row: the margin is a signed difference, so 0 means
    "measured equal" and awarding on it would award by comparison order.
    """
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
    """Prove each field's boundary values load; `min_winrate: 0.0` is the stated no-bar posture."""
    armed = {"ply_cap_adjudication": _ARMED_PLY, "strength_floor": _ARMED_FLOOR}
    raw = _payload(ply=_ARMED_PLY, floor=_ARMED_FLOOR)
    block_value = dict(armed[block])
    block_value[field] = value
    raw["eval"][block] = block_value
    cfg = RunConfig.model_validate(raw)
    assert getattr(getattr(cfg.eval, block), field) == value
