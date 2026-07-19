"""`build_net` — the ONE model construction authority (repo_design §3).

Dispatch on the declared arch dataclass (a closed union — no wildcard on the kind,
LAW-11): `CnnArch` → `HexTacToeNet`, `GnnArch` → `GnnNet`. The arch travels on the
declared dataclass; nobody infers it from module structure (the old
`model_representation` live-`nn.Module` sniff is DELETED and grep-gate-banned).

`RepresentationMismatch` is defined in `arch` (the lowest layer that raises it) and
re-exported here so `mantis.model.build.RepresentationMismatch` resolves.
"""
from __future__ import annotations

import torch.nn as nn

from mantis.model.arch import CnnArch, GnnArch, ModelArch, RepresentationMismatch
from mantis.model.cnn import HexTacToeNet
from mantis.model.gnn import GnnNet

__all__ = ["build_net", "RepresentationMismatch"]


def build_net(arch: ModelArch) -> nn.Module:
    """Construct the model for `arch` — the ONE authority.

    Grid (`CnnArch`) → `HexTacToeNet(arch)`; graph (`GnnArch`) → `GnnNet(arch)`.
    The construction builds the SAME nn layers (same names/order) as the old kwargs
    ctors → state-dict byte-identical. An arch that is neither raises
    `RepresentationMismatch` (unreachable for the closed union — explicit, no silent
    default)."""
    if isinstance(arch, CnnArch):
        return HexTacToeNet(arch)
    # Defensive runtime check: the declared type is a closed union, but a caller can
    # still pass a non-arch at runtime (build_net(object()) → RepresentationMismatch).
    if isinstance(arch, GnnArch):  # pyright: ignore[reportUnnecessaryIsInstance]
        return GnnNet(arch)
    raise RepresentationMismatch(
        f"build_net: arch is neither CnnArch nor GnnArch (got {type(arch).__name__})."
    )
