"""Declared model-arch dataclasses + the spec/config → arch adapter.

Arch metadata travels on these frozen dataclasses (repo_design §3): a caller
retains the declared `CnnArch`/`GnnArch` and hands it to `build_net`; nobody
infers arch by reading attributes off a live `nn.Module` (that sniff — the old
`model_representation` — is DELETED and grep-gate-banned; see `tests/model/
test_arch_ban.py`).

`arch_from_spec_and_config` consumes a resolved encoding spec (from
`mantis.encoding`) + a plain `Mapping` config — it imports NO `mantis.config`,
so the model layer builds and tests without the config package. There is NO
representation default (LAW-11): an absent `spec.representation` is an error,
never a silent "grid".

`RepresentationMismatch` is defined here (the lowest layer that raises it — both
this adapter and `build.build_net` do) and re-exported by `build` and the package
`__init__`, so `mantis.model.build.RepresentationMismatch` and
`mantis.model.RepresentationMismatch` both resolve.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Representation = Literal["grid", "graph"]


class RepresentationMismatch(ValueError):
    """`spec.representation` is unknown/absent, or incompatible with the requested
    model config (a non-`dist65` `value_head_type` under `representation="graph"`,
    or a graph encoding missing its node/edge geometry fields).

    Message is prefixed `"RepresentationMismatch: "` to mirror the engine seam's
    raised-`ValueError` convention.
    """

    def __init__(self, msg: str) -> None:
        super().__init__(f"RepresentationMismatch: {msg}")


@dataclass(frozen=True)
class CnnArch:
    """Declared grid (CNN) architecture. `build_net` constructs `HexTacToeNet`
    from this; the fields reproduce the old kwargs ctor byte-for-byte."""

    board_size: int
    in_channels: int
    filters: int = 128
    res_blocks: int = 12
    se_reduction_ratio: int = 4
    value_head_type: Literal["scalar", "dist65"] = "scalar"
    n_value_bins: int = 65
    input_channels: tuple[int, ...] | None = None
    representation: Literal["grid"] = "grid"   # closed tag


@dataclass(frozen=True)
class GnnArch:
    """Declared graph (GNN) architecture. `build_net` constructs `GnnNet` from
    this; the graph net ships only a dist65 value head."""

    in_dim: int
    edge_dim: int
    hidden: int = 128
    num_layers: int = 4
    policy_hidden: int = 128
    value_hidden: int = 32
    n_value_bins: int = 65
    representation: Literal["graph"] = "graph"


ModelArch = CnnArch | GnnArch

# Grid hparam config keys → CnnArch field names (config override wins; absent →
# the dataclass field default, which is the sole default authority — R1).
_GRID_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("filters", "filters"),
    ("res_blocks", "res_blocks"),
    ("se_reduction_ratio", "se_reduction_ratio"),
    ("value_head_type", "value_head_type"),
    ("n_value_bins", "n_value_bins"),
)

# Graph hparam config keys → GnnArch field names.
_GRAPH_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("gnn_hidden", "hidden"),
    ("gnn_num_layers", "num_layers"),
    ("gnn_policy_hidden", "policy_hidden"),
    ("gnn_value_hidden", "value_hidden"),
    ("n_value_bins", "n_value_bins"),
)


def arch_from_spec_and_config(spec: Any, config: Mapping[str, Any]) -> ModelArch:
    """Resolved encoding spec + plain config mapping → declared arch dataclass.

    Mapping-typed (NOT config-schema-typed) so the model layer is self-contained
    and testable without the config package. NO representation default (LAW-11):
    absent `spec.representation` → `RepresentationMismatch`.
    """
    rep = getattr(spec, "representation", None)
    if rep is None:
        raise RepresentationMismatch(
            f"spec {getattr(spec, 'name', spec)!r} has no representation attribute "
            "— cannot infer a model arch (no dense-by-default, LAW-11)."
        )
    if rep == "grid":
        kw: dict[str, Any] = {}
        for cfg_key, field in _GRID_CONFIG_KEYS:
            if cfg_key in config and config[cfg_key] is not None:
                kw[field] = config[cfg_key]
        ic = config.get("input_channels")
        if ic is not None:
            kw["input_channels"] = tuple(int(c) for c in ic)
        return CnnArch(
            board_size=int(spec.board_size),
            in_channels=int(spec.n_planes),
            **kw,
        )
    if rep == "graph":
        node_feat_dim = getattr(spec, "node_feat_dim", None)
        edge_feat_dim = getattr(spec, "edge_feat_dim", None)
        if node_feat_dim is None or edge_feat_dim is None:
            raise RepresentationMismatch(
                f"encoding {getattr(spec, 'name', '?')!r} declares "
                "representation='graph' but is missing node_feat_dim/edge_feat_dim."
            )
        declared_vht = config.get("value_head_type")
        if declared_vht is not None and declared_vht != "dist65":
            raise RepresentationMismatch(
                f"representation='graph' (encoding {getattr(spec, 'name', '?')!r}) "
                f"only ships a dist65 value head; got value_head_type={declared_vht!r}."
            )
        kw = {}
        for cfg_key, field in _GRAPH_CONFIG_KEYS:
            if cfg_key in config and config[cfg_key] is not None:
                kw[field] = config[cfg_key]
        return GnnArch(in_dim=int(node_feat_dim), edge_dim=int(edge_feat_dim), **kw)
    raise RepresentationMismatch(
        f"spec.representation={rep!r} for encoding "
        f"{getattr(spec, 'name', '?')!r} — expected 'grid' or 'graph'."
    )
