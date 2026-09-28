"""`search.tactics` (v39): REQUIRED in both homes with `null` the explicit off, refused under self-play, one resolver."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mantis._engine import MCTSTree
from mantis.config.loader import load_config
from mantis.config.resolve.tactics import (
    AUDIT_INVERTED,
    MissingTacticsError,
    resolve_deploy_tactics,
    tactics_block,
)
from mantis.config.schema import RunConfig

_CONFIGS = sorted(Path(__file__).resolve().parents[2].joinpath("configs").glob("*.yaml"))
_BLOCK: dict[str, Any] = {
    "kind": "strict_turn", "leaf_turns": 3, "leaf_nodes": 256, "root_turns": 8, "root_nodes": 20000,
    "audit": {"turns": 8, "nodes": 2000, "k": 4, "m": 4, "total_nodes": 40000},
}


def _dump() -> dict[str, Any]:
    return load_config(str(_CONFIGS[0])).model_dump()


def _with(home: str, block: Any) -> dict[str, Any]:
    dump = copy.deepcopy(_dump())
    dump[home]["search"]["tactics"] = block
    return dump


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.name)
def test_every_minted_config_states_the_explicit_off_in_both_homes(path: Path) -> None:
    config = load_config(str(path))
    assert config.deploy.search.tactics is None
    assert config.selfplay.search.tactics is None
    assert resolve_deploy_tactics(config.model_dump()) is None


def test_the_key_is_required_not_defaulted() -> None:
    dump = _dump()
    del dump["deploy"]["search"]["tactics"]
    with pytest.raises(ValidationError, match="tactics"):
        RunConfig.model_validate(dump)
    with pytest.raises(MissingTacticsError, match="REQUIRED"):
        resolve_deploy_tactics(dump)


def test_a_deploy_block_validates_and_the_bridge_arms_what_the_resolver_hands_it() -> None:
    config = RunConfig.model_validate(_with("deploy", _BLOCK))
    armed = resolve_deploy_tactics(config.model_dump())
    assert armed == {**_BLOCK, "audit": {**_BLOCK["audit"], "mode": "hold"}}
    tree = MCTSTree()
    tree.configure_tactics(armed)
    assert tree.tactics_armed
    off = RunConfig.model_validate(_with("deploy", {**_BLOCK, "audit": None}))
    assert resolve_deploy_tactics(off.model_dump())["audit"] is None


def test_the_inverted_audit_is_a_resolver_argument_never_a_schema_leaf() -> None:
    assert tactics_block(_BLOCK, audit_mode=AUDIT_INVERTED)["audit"]["mode"] == "inverted"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_with("deploy", {**_BLOCK, "audit": {**_BLOCK["audit"], "mode": "inverted"}}))


def test_a_selfplay_block_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match="selfplay.search.tactics is set"):
        RunConfig.model_validate(_with("selfplay", _BLOCK))


@pytest.mark.parametrize("bad", [
    {"kind": "per_stone"}, {"leaf_turns": 0}, {"root_turns": 41}, {"leaf_nodes": -1}, {"root_nodes": True},
    {"audit": {**_BLOCK["audit"], "k": 0}}, {"audit": {**_BLOCK["audit"], "total_nodes": 0}},
])
def test_a_block_outside_the_bridge_bounds_does_not_mint(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_with("deploy", {**_BLOCK, **bad}))
