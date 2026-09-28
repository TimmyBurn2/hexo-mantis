"""`resolve_deploy_tactics` — THE read path for `deploy.search.tactics` (v39): the block as the bridge arms it, or `None`."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_KEY = "deploy.search.tactics"
_AUDIT_MEMBERS: tuple[str, ...] = ("turns", "nodes", "k", "m", "total_nodes")
#: The mode a minted block's audit arms; the inverted known-bad is a cell tool's arm, never a config's.
AUDIT_HOLD = "hold"
AUDIT_INVERTED = "inverted"


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
