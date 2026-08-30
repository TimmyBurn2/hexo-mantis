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

from mantis.model.arch import CnnArch, GnnArch, GnnArchV2, ModelArch, RepresentationMismatch
from mantis.model.cnn import HexTacToeNet
from mantis.model.gnn import GnnNet
from mantis.model.gnn_v2 import GnnNetV2

__all__ = ["build_net", "RepresentationMismatch"]


def build_net(arch: ModelArch) -> nn.Module:
    """Construct the model for `arch` — the ONE authority.

    Grid (`CnnArch`) → `HexTacToeNet(arch)`; graph (`GnnArch`) → `GnnNet(arch)`.
    The construction builds the SAME nn layers (same names/order) as the old kwargs
    ctors → state-dict byte-identical. An arch that is neither raises
    `RepresentationMismatch` (unreachable for the closed union — explicit, no silent
    default).

    TD-2 (WPAX Phase P): this also supplies the BUILD side of the arch-travels-with-the-
    model convention. `eval/snapshot.py:45-51` documents that convention and RAISES without
    it (`getattr(model, "arch", None)`, `:48`), and `:82` implements the LOAD side — nothing
    implemented the build side, so the terminal eval (`eval/pipeline.py:312`) and the anchor
    snapshot died on the first snapshot of a freshly-built net. This is NOT the repo_design
    §3 ban: what §3 bans is DERIVING arch metadata from a live module's structure; what
    happens below is carrying the DECLARED dataclass instance — the very object §3 says
    arch travels on — as a handle."""
    net: nn.Module
    if isinstance(arch, CnnArch):
        net = HexTacToeNet(arch)
    # GnnArchV2 IS TESTED BEFORE GnnArch, and the order is load-bearing rather than stylistic:
    # were V2 ever made a subclass of V1, `isinstance(arch, GnnArch)` would match it and this
    # function would silently build V1's net for a V2 arch. V2 is a SIBLING dataclass so the
    # order is not what saves us today — it is the second line of defence, and the pair is
    # pinned by tests/model/test_arch_v2_dispatch.py.
    elif isinstance(arch, GnnArchV2):
        net = GnnNetV2(arch)
    # Defensive runtime check: the declared type is a closed union, but a caller can
    # still pass a non-arch at runtime (build_net(object()) → RepresentationMismatch).
    elif isinstance(arch, GnnArch):  # pyright: ignore[reportUnnecessaryIsInstance]
        net = GnnNet(arch)
    else:
        raise RepresentationMismatch(
            f"build_net: arch is none of CnnArch, GnnArch, GnnArchV2 "
            f"(got {type(arch).__name__})."
        )
    # THE declared instance, never a copy or a re-derivation: a copy would be a second
    # authority for the run's identity. Plain attribute assignment, so the handle lands in
    # `__dict__` and in NONE of `_parameters` / `_buffers` / `_modules` — `state_dict()` is
    # byte-identical and LAW-12's checkpoint key set is unchanged.
    net.arch = arch  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType] — plain __dict__ handle; nn.Module.__setattr__ is annotated Tensor | Module only
    return net
