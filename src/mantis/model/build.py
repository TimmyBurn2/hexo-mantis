"""`build_net` — the ONE model construction authority.

Dispatch is on the declared arch dataclass, a closed union with no wildcard on the kind:
`GnnArch` → `GnnNet`, `GnnArchV2` → `GnnNetV2`. The arch travels on that dataclass and is
never inferred from module structure.

`RepresentationMismatch` is defined in `arch`, the lowest layer that raises it, and
re-exported here.
"""
from __future__ import annotations

import torch.nn as nn

from mantis.model.arch import GnnArch, GnnArchV2, ModelArch, RepresentationMismatch
from mantis.model.gnn import GnnNet
from mantis.model.gnn_v2 import GnnNetV2

__all__ = ["build_net", "RepresentationMismatch"]


def build_net(arch: ModelArch) -> nn.Module:
    """Construct the model for `arch`, and attach the declared arch to it as a handle.

    The attached handle is the build side of the arch-travels-with-the-model convention the
    snapshot and load paths require; it carries the declared dataclass instance and derives
    nothing from the live module's structure.

    Raises:
        RepresentationMismatch: `arch` is neither `GnnArch` nor `GnnArchV2`.
    """
    net: nn.Module
    # GnnArchV2 is tested first as a second line of defence: were V2 ever made a subclass of
    # V1, `isinstance(arch, GnnArch)` would match it and silently build V1's net.
    if isinstance(arch, GnnArchV2):
        net = GnnNetV2(arch)
    # The declared type is a closed union, but a caller can still pass a non-arch at runtime.
    elif isinstance(arch, GnnArch):  # pyright: ignore[reportUnnecessaryIsInstance]
        net = GnnNet(arch)
    else:
        raise RepresentationMismatch(
            f"build_net: arch is neither GnnArch nor GnnArchV2 "
            f"(got {type(arch).__name__})."
        )
    # THE declared instance, never a copy: a copy would be a second authority for the run's
    # identity. Plain assignment keeps the handle in `__dict__` and out of `_parameters` /
    # `_buffers` / `_modules`, so `state_dict()` is byte-identical.
    net.arch = arch  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType] — plain __dict__ handle; nn.Module.__setattr__ is annotated Tensor | Module only
    return net
