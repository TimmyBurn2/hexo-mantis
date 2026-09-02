"""mantis.model — nets (GNN + CNN), dist65 value codec, build_net authority.

Public API: `build_net` + the declared arch dataclasses (`ModelArch`/`CnnArch`/
`GnnArch`/`GnnArchV2`) + `RepresentationMismatch`; the arch-kind vocabulary
(`ARCH_KINDS`/`ARCH_KINDS_BY_REPRESENTATION`/`INCUMBENT_ARCH_KIND`) and the `select_arch`
selector; the nets `HexTacToeNet` / `GnnNet` / `GnnNetV2`; the dist65 primitives;
`amp_dtype_for`; `net_param_hash`.
"""
from __future__ import annotations

from mantis.model.amp import amp_dtype_for
from mantis.model.arch import (
    ARCH_KIND_ROW,
    ARCH_KINDS,
    ARCH_KINDS_BY_REPRESENTATION,
    INCUMBENT_ARCH_KIND,
    CnnArch,
    GnnArch,
    GnnArchV2,
    ModelArch,
    RepresentationMismatch,
    UnknownArchKind,
    arch_from_spec_and_config,
    declared_arch_kind,
    select_arch,
)
from mantis.model.build import build_net
from mantis.model.cnn import HexTacToeNet, compile_model
from mantis.model.dist65 import (
    N_VALUE_BINS,
    VALUE_SUPPORT,
    binned_value_loss,
    decode_binned_value,
    scalar_to_two_hot,
)
from mantis.model.gnn import (
    GnnDist65ValueHead,
    GnnNet,
    load_representation_policy_from_bc,
)
from mantis.model.gnn_v2 import GnnNetV2
from mantis.model.identity import net_param_hash

__all__ = [
    "ARCH_KIND_ROW",
    "ARCH_KINDS",
    "ARCH_KINDS_BY_REPRESENTATION",
    "INCUMBENT_ARCH_KIND",
    "N_VALUE_BINS",
    "VALUE_SUPPORT",
    "CnnArch",
    "GnnArch",
    "GnnArchV2",
    "GnnDist65ValueHead",
    "GnnNet",
    "GnnNetV2",
    "HexTacToeNet",
    "ModelArch",
    "RepresentationMismatch",
    "UnknownArchKind",
    "amp_dtype_for",
    "arch_from_spec_and_config",
    "declared_arch_kind",
    "binned_value_loss",
    "build_net",
    "compile_model",
    "decode_binned_value",
    "load_representation_policy_from_bc",
    "net_param_hash",
    "scalar_to_two_hot",
    "select_arch",
]
