"""`resolve_deploy_tactics` — THE read path for `deploy.search.tactics` (v39): the block as the bridge arms it, or `None`."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mantis.config.schema import TacticsConfig

_KEY = "deploy.search.tactics"
_AUDIT_MEMBERS: tuple[str, ...] = ("turns", "nodes", "k", "m", "total_nodes")
#: The mode a minted block's audit arms; the inverted known-bad is a cell tool's arm, never a config's.
AUDIT_HOLD = "hold"
AUDIT_INVERTED = "inverted"


#: The A/B arms over one block, the candidate's only: none, the block, its audit off, its audit inverted (the known-bad).
ARMS: tuple[str, ...] = ("plain", "full", "audit-off", "known-bad")


class MissingTacticsError(ValueError):
    """`deploy.search.tactics` is absent (not `null`) at some named level."""


def tactics_block(block: Mapping[str, Any] | None, *, audit_mode: str = AUDIT_HOLD) -> dict[str, Any] | None:
    """A schema-valid block as `MCTSTree.configure_tactics` takes it (the audit's `mode` added), or `None`."""
    if block is None:
        return None
    audit = block["audit"]
    return {
        "kind": block["kind"],
        "leaf_turns": block["leaf_turns"],
        "leaf_nodes": block["leaf_nodes"],
        "root_turns": block["root_turns"],
        "root_nodes": block["root_nodes"],
        "audit": None if audit is None else {**{m: audit[m] for m in _AUDIT_MEMBERS}, "mode": audit_mode},
    }


def arm_block(arm: str, block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The bridge block `arm` arms from a raw `search.tactics` block; Raises: ValueError — unknown arm, bad block."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} is not one of {ARMS}")
    if arm == "plain":
        return None
    if block is None:
        raise ValueError(f"arm {arm!r} overlays a tactics block and none was given")
    valid = TacticsConfig.model_validate(block).model_dump()
    if arm == "audit-off":
        return tactics_block({**valid, "audit": None})
    if valid["audit"] is None:
        raise ValueError(f"arm {arm!r} arms the block's audit, which is null")
    return tactics_block(valid, audit_mode=AUDIT_INVERTED if arm == "known-bad" else AUDIT_HOLD)


def resolve_deploy_tactics(full_config: Any) -> dict[str, Any] | None:
    """The deploy head's block from a config dump, or `None` for `null`; Raises: MissingTacticsError — the key absent."""
    deploy = full_config.get("deploy") if isinstance(full_config, Mapping) else None
    search = deploy.get("search") if isinstance(deploy, Mapping) else None
    if not isinstance(search, Mapping) or "tactics" not in search:
        raise MissingTacticsError(
            f"{_KEY} is absent. The key is REQUIRED (`null` is the explicit OFF), so a config reaching "
            "here without it was not built through the one loader; a code-side OFF would report as configured"
        )
    return tactics_block(search["tactics"])
