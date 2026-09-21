"""`resolve_gnn_widths` — the refusing front door for `model.gnn` (v35), run at every fresh boot before the build reads the block; the parse is restated here because this package imports no `mantis.model` (torch)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mantis.config.resolve.arch_scope import refuse_outside_its_arch

_SECTION, _FIELD = "model", "gnn"
_KEY = f"{_SECTION}.{_FIELD}"
_MEMBERS: tuple[str, ...] = ("hidden", "num_layers")


class MissingGnnWidthsError(ValueError):
    """The graph trunk's widths are not declared at some named level — a configuration ERROR."""


@dataclass(frozen=True)
class GnnWidthsSpec:
    """The resolved trunk shape: `hidden` and `num_layers`, together (the readout is their product)."""

    hidden: int
    num_layers: int


def resolve_gnn_widths(full_config: Any) -> GnnWidthsSpec:
    """Return the declared graph trunk widths; Raises: ArchScopedKeyOutsideItsArchError — a non-graph representation, refused first; MissingGnnWidthsError — no `model` section, no `gnn` block, or a member short."""
    refuse_outside_its_arch(full_config, _SECTION, _FIELD)
    if not isinstance(full_config, Mapping) or _SECTION not in full_config:
        raise MissingGnnWidthsError(
            f"{_KEY}: the config carries no `{_SECTION}` section, so the trunk's shape cannot be "
            "read — the dataclass default would report as configured (R1/LAW-11)"
        )
    block = full_config[_SECTION].get(_FIELD) if isinstance(full_config[_SECTION], Mapping) else None
    if not isinstance(block, Mapping):
        raise MissingGnnWidthsError(
            f"{_KEY} is absent. The block is REQUIRED on a graph config, so a config reaching "
            "here without it was not built through the one loader"
        )
    missing = [m for m in _MEMBERS if m not in block]
    if missing:
        raise MissingGnnWidthsError(f"{_KEY} is missing {missing}; both members are REQUIRED")
    return GnnWidthsSpec(hidden=int(block["hidden"]), num_layers=int(block["num_layers"]))


__all__ = ["GnnWidthsSpec", "MissingGnnWidthsError", "resolve_gnn_widths"]
