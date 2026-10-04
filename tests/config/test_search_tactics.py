"""`search.tactics` (v39): REQUIRED in both homes with `null` the explicit off, each home armed through one resolver."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mantis._engine import MCTSTree
from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.tactics import (
    ARMS,
    AUDIT_INVERTED,
    MissingTacticsError,
    arm_block,
    resolve_deploy_tactics,
    resolve_selfplay_tactics,
    tactics_block,
)
from mantis.config.schema import RunConfig
from mantis.selfplay.hparams import SelfPlayHParams, build_runner_config, resolve_pool_encoding
from _minted_puct import MINTED_PUCT

_CONFIGS = discover_configs(Path(__file__).resolve().parents[2] / "configs")
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
def test_every_minted_config_states_the_off_or_the_block_of_record_alike_in_both_homes(path: Path) -> None:
    """A config states its posture in both homes: the explicit off, or the deploy block of record armed alike."""
    config = load_config(str(path)).model_dump()
    deploy, selfplay = config["deploy"]["search"]["tactics"], config["selfplay"]["search"]["tactics"]
    assert deploy == selfplay and deploy in (None, _BLOCK), f"{path.name}: deploy {deploy}, self-play {selfplay}"
    assert (resolve_deploy_tactics(config) is None) == (deploy is None)


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
    tree = MCTSTree(**MINTED_PUCT.tree_kwargs())
    tree.configure_tactics(armed)
    assert tree.tactics_armed
    off = RunConfig.model_validate(_with("deploy", {**_BLOCK, "audit": None}))
    assert resolve_deploy_tactics(off.model_dump())["audit"] is None


def test_the_inverted_audit_is_a_resolver_argument_never_a_schema_leaf() -> None:
    assert tactics_block(_BLOCK, audit_mode=AUDIT_INVERTED)["audit"]["mode"] == "inverted"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_with("deploy", {**_BLOCK, "audit": {**_BLOCK["audit"], "mode": "inverted"}}))


def test_a_selfplay_block_validates_and_every_worker_tree_arms_what_the_resolver_hands_it() -> None:
    config = RunConfig.model_validate(_with("selfplay", _BLOCK))
    armed = resolve_selfplay_tactics(config.model_dump())
    assert armed == {**_BLOCK, "audit": {**_BLOCK["audit"], "mode": "hold"}}
    assert resolve_deploy_tactics(config.model_dump()) is None, "the homes are read apart"
    hp = SelfPlayHParams.from_config(config.model_dump())
    assert hp.tactics == armed
    dump = config.model_dump()
    runner_config = build_runner_config(hp, spec_dims=resolve_pool_encoding(dump),
                                        encoding_name=dump["identity"]["encoding"])
    assert runner_config.tactics_armed
    with pytest.raises(MissingTacticsError, match="selfplay.search.tactics is absent"):
        dump = config.model_dump()
        del dump["selfplay"]["search"]["tactics"]
        resolve_selfplay_tactics(dump)


@pytest.mark.parametrize("bad", [
    {"kind": "per_stone"}, {"leaf_turns": 0}, {"root_turns": 41}, {"leaf_nodes": -1}, {"root_nodes": True},
    {"root_nodes": 2**63}, {"audit": {**_BLOCK["audit"], "total_nodes": 2**63}},
    {"audit": {**_BLOCK["audit"], "k": 0}}, {"audit": {**_BLOCK["audit"], "total_nodes": 0}},
])
def test_a_block_outside_the_bridge_bounds_does_not_mint(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(_with("deploy", {**_BLOCK, **bad}))


def test_each_arm_resolves_the_one_block_and_the_bridge_arms_every_one() -> None:
    assert arm_block("plain", _BLOCK) is None and arm_block("plain", None) is None
    assert arm_block("full", _BLOCK) == tactics_block(_BLOCK)
    assert arm_block("audit-off", _BLOCK) == tactics_block({**_BLOCK, "audit": None})
    assert arm_block("known-bad", _BLOCK) == tactics_block(_BLOCK, audit_mode=AUDIT_INVERTED)
    for arm in ("full", "audit-off", "known-bad"):
        MCTSTree(**MINTED_PUCT.tree_kwargs()).configure_tactics(arm_block(arm, _BLOCK))
    assert set(ARMS) == {"plain", "full", "audit-off", "known-bad"}


@pytest.mark.parametrize(("arm", "block", "match"), [
    ("inverted", _BLOCK, "not one of"), ("full", None, "none was given"),
    ("known-bad", {**_BLOCK, "audit": None}, "null"), ("full", {**_BLOCK, "leaf_turns": 0}, "leaf_turns"),
    ("audit-off", {**_BLOCK, "mode": "hold"}, "mode"),
])
def test_an_arm_that_cannot_resolve_is_refused_by_name(arm: str, block: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        arm_block(arm, block)
