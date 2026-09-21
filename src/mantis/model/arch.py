"""Declared model-arch dataclasses plus the spec/config to arch adapter.

Arch metadata travels on these frozen dataclasses: a caller retains the declared arch and hands
it to `build_net`, and nobody infers arch by reading attributes off a live `nn.Module` — that
sniff is deleted and grep-gate-banned. `arch_from_spec_and_config` consumes a resolved encoding
spec and a plain `Mapping`; the `model.gnn` parser and the soft-policy kinds table are the
config package's (the `model -> config` edge, R367(a): one implementation per thing); there is
NO representation default. `RepresentationMismatch` is defined here, the lowest layer that
raises it, and re-exported by `build` and the package.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mantis.config.resolve.gnn_widths import MissingGnnWidthsError, resolve_gnn_widths

Representation = Literal["graph"]


class RepresentationMismatch(ValueError):
    """`spec.representation` is unknown or absent, or incompatible with the requested model
    config. The message is prefixed `"RepresentationMismatch: "` to mirror the engine seam's
    raised-`ValueError` convention."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"RepresentationMismatch: {msg}")


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


@dataclass(frozen=True)
class GnnArchV2:
    """Declared graph architecture, V2 — the `gnn_axis_v1` WIRE, two model-side mechanisms.

    A SIBLING of `GnnArch`, never a subclass: `build_net` and the checkpoint loader both dispatch
    by type, and a subclass would satisfy `isinstance(arch, GnnArch)` and silently build V1. The
    mechanisms are a `concat(stone-masked mean, max over real nodes)` value readout and a
    degree-normalized dummy aggregation. NO FIELD NAMES A PROPERTY V2 CLAIMS — a per-arch field
    summarising a per-position fact is a lie with a type — and the field set matches `GnnArch`
    exactly, because the swap is the ARCH, not a knob on one.
    """

    in_dim: int
    edge_dim: int
    hidden: int = 128
    num_layers: int = 4
    policy_hidden: int = 128
    value_hidden: int = 32
    n_value_bins: int = 65
    representation: Literal["graph"] = "graph"


@dataclass(frozen=True)
class GnnArchV2SoftPolicy:
    """V2's trunk and heads plus an AUXILIARY SOFT-POLICY head (R366(b): a second policy head on the searched target at `model.aux_soft_policy.target_temperature`, KataGo's soft head in the sparse-row regime); served outputs are V2's exactly; a SIBLING with V2's field set, never a subclass, for `build_net`'s reason."""

    in_dim: int
    edge_dim: int
    hidden: int = 128
    num_layers: int = 4
    policy_hidden: int = 128
    value_hidden: int = 32
    n_value_bins: int = 65
    representation: Literal["graph"] = "graph"


ModelArch = GnnArch | GnnArchV2 | GnnArchV2SoftPolicy


class UnknownArchKind(ValueError):
    """A requested arch kind is not in `ARCH_KINDS`, or is not available on that representation.
    A `ValueError` for `RepresentationMismatch`'s reason: naming an arch this build does not have
    is a configuration ERROR, and the nearest member of the union is never substituted."""


#: THE ARCH-KIND VOCABULARY — the ONE naming authority for "which model kind is this", keyed by
#: the declared dataclass's own name, so the token and the type cannot drift apart and a new arch
#: adds exactly one row. It lives HERE and not in the checkpoint loader, which first needed it: a
#: kind vocabulary is a MODEL fact and the loader is a consumer, which is also the import
#: direction the DAG requires.
ARCH_KINDS: dict[str, type] = {

    "GnnArch": GnnArch,
    "GnnArchV2": GnnArchV2,
    "GnnArchV2SoftPolicy": GnnArchV2SoftPolicy,
}

#: Which kinds a representation admits — the pairing rule, stated for the arch side. `graph`
#: admits several kinds since GnnNetV2 landed, which is exactly why a selector has to exist.
ARCH_KINDS_BY_REPRESENTATION: dict[str, tuple[str, ...]] = {
    "graph": ("GnnArch", "GnnArchV2", "GnnArchV2SoftPolicy"),
}

#: THE INCUMBENT KIND PER REPRESENTATION — a statement about HISTORY, not a default, which is
#: why it is named and pinned rather than inlined: a default answers "what should we build when
#: nobody said?", this answers "what has this tree always built?", a fact the conformance suite
#: executes against the real minted files. Until a config carries `identity.arch_kind`,
#: resolution goes through THIS table; once it does, the row is the authority. Absence is not a
#: fallback with a guess in it — it is the statement "this config predates the row".
INCUMBENT_ARCH_KIND: dict[str, str] = {"graph": "GnnArch"}

#: THE ONE CONFIG ROW that names an arch kind — dotted, as `RunConfig` spells it. Read by
#: `declared_arch_kind` and nowhere else, so the row has exactly one reader to change.
ARCH_KIND_ROW = "identity.arch_kind"


def declared_arch_kind(config: Mapping[str, Any]) -> str | None:
    """The `identity.arch_kind` row of a plain config mapping, or `None` when the config does not
    carry it. Mapping-typed for the same reason the rest of this module is: no `mantis.config`
    import. A row present but not a string is returned as-is and refused downstream by
    `select_arch`, which names the kind it was given — this reader never coerces."""
    identity = config.get("identity")
    if not isinstance(identity, Mapping):
        return None
    return identity.get("arch_kind")

#: THE ONE CONFIG BLOCK that sizes the graph trunk, `model.gnn.{hidden, num_layers}` (v35); a
#: mapping with no `model` block at all (a strip's synthetic config) falls to the dataclass defaults.
GNN_WIDTHS_BLOCK = ("model", "gnn")
_GNN_WIDTH_FIELDS: tuple[str, ...] = ("hidden", "num_layers")

# Flat head-width keys → GnnArch field names; no schema leaf carries them.
_GRAPH_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("gnn_policy_hidden", "policy_hidden"),
    ("gnn_value_hidden", "value_hidden"),
    ("n_value_bins", "n_value_bins"),
)


def gnn_widths_block(arch: Any) -> dict[str, int]:
    """`arch`'s trunk widths as the `model.gnn` block a config carries — the inverse of `declared_gnn_widths`."""
    return {field: int(getattr(arch, field)) for field in _GNN_WIDTH_FIELDS}


def declared_gnn_widths(config: Mapping[str, Any]) -> dict[str, int]:
    """The `model.gnn` widths of a plain config mapping as arch-field kwargs, `{}` with no `model` block (a pre-v35 stamp); Raises: RepresentationMismatch — a `model` block whose `gnn` member is absent or short of a width; ArchScopedKeyOutsideItsArchError — the block on a non-graph config."""
    if config.get(GNN_WIDTHS_BLOCK[0]) is None:
        return {}
    try:
        spec = resolve_gnn_widths(config)
    except MissingGnnWidthsError as exc:
        raise RepresentationMismatch(
            f"{'.'.join(GNN_WIDTHS_BLOCK)} must carry {list(_GNN_WIDTH_FIELDS)} on a graph config: {exc}"
        ) from exc
    return {"hidden": spec.hidden, "num_layers": spec.num_layers}


def arch_from_spec_and_config(spec: Any, config: Mapping[str, Any]) -> ModelArch:
    """Resolved encoding spec plus plain config mapping to the arch the CONFIG selects.

    THE PRODUCTION ENTRY POINT for a caller that holds a run's config. The `identity.arch_kind`
    row, when present, is the authority and is handed to `select_arch` verbatim; without it,
    resolution goes to the representation's INCUMBENT kind — a history fact pinned against every
    minted file. A caller that holds an ARTIFACT and no config does not come here: the stamp is
    its authority, and reading this table for it would be a second answer.

    Raises:
        RepresentationMismatch: `spec.representation` is absent or unknown, or the encoding is
            incompatible with the requested config.
        UnknownArchKind: the config's row names a kind this build does not have, or one its
            representation does not admit.
    """
    rep = getattr(spec, "representation", None)
    if rep is None:
        raise RepresentationMismatch(
            f"spec {getattr(spec, 'name', spec)!r} has no representation attribute "
            "— cannot infer a model arch (no dense-by-default, LAW-11)."
        )
    # The read-path half of the arch partition for `model.gnn` (v35): a declared representation
    # the spec does not have is refused before the widths are read.
    identity = config.get("identity")
    declared_rep = identity.get("representation") if isinstance(identity, Mapping) else None
    if isinstance(declared_rep, str) and declared_rep != str(rep):
        raise RepresentationMismatch(
            f"the config declares identity.representation={declared_rep!r} but encoding "
            f"{getattr(spec, 'name', '?')!r} is representation={rep!r}; the trunk widths under "
            f"{'.'.join(GNN_WIDTHS_BLOCK)} are scoped to {rep!r} and are not read for another"
        )
    declared = declared_arch_kind(config)
    if declared is not None:
        return select_arch(spec, config, arch_kind=declared)
    incumbent = INCUMBENT_ARCH_KIND.get(str(rep))
    if incumbent is None:
        raise RepresentationMismatch(
            f"spec.representation={rep!r} for encoding "
            f"{getattr(spec, 'name', '?')!r} — expected 'grid' or 'graph'."
        )
    return select_arch(spec, config, arch_kind=incumbent)


def select_arch(spec: Any, config: Mapping[str, Any], *, arch_kind: str) -> ModelArch:
    """Resolved encoding spec, plain config mapping and an EXPLICIT arch kind to a declared arch.

    `arch_kind` is keyword-only with NO default: a caller that does not know which kind it wants
    is not entitled to one, and that absence is what keeps this from becoming a second answer to
    "what does production build".

    Args:
        spec: a resolved encoding spec carrying `representation` and the geometry fields.
        config: a plain config mapping; the trunk widths are read off its `model.gnn` block
            (`declared_gnn_widths`); no `model` block at all falls to the dataclass default.
        arch_kind: a member of `ARCH_KINDS`, admitted by `spec.representation`.

    Raises:
        UnknownArchKind: `arch_kind` is not known, or the representation does not admit it.
            Named separately from `RepresentationMismatch` because "you asked for an arch that
            does not exist" and "your encoding and your model disagree" send a reader to two
            different places.
        RepresentationMismatch: `spec.representation` is absent or unknown, the graph geometry
            fields are missing, or a non-`dist65` value head was requested on a graph arch.
    """
    rep = getattr(spec, "representation", None)
    if rep is None:
        raise RepresentationMismatch(
            f"spec {getattr(spec, 'name', spec)!r} has no representation attribute "
            "— cannot infer a model arch (no dense-by-default, LAW-11)."
        )
    if arch_kind not in ARCH_KINDS:
        raise UnknownArchKind(
            f"arch_kind={arch_kind!r} is not a known model kind; this build has "
            f"{sorted(ARCH_KINDS)}. An unknown kind is REFUSED, never resolved to the nearest "
            "member of the union."
        )
    admitted = ARCH_KINDS_BY_REPRESENTATION.get(str(rep), ())
    if arch_kind not in admitted:
        raise UnknownArchKind(
            f"arch_kind={arch_kind!r} is not admitted by representation={rep!r}, which admits "
            f"{sorted(admitted)}. The pairing rule is `ARCH_KINDS_BY_REPRESENTATION` and it is "
            "closed: an arch and an encoding that disagree do not silently build."
        )
    if rep == "graph":  # noqa: RET503 — the closed set is exhausted by the guards above
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
        kw: dict[str, Any] = declared_gnn_widths(config)
        for cfg_key, field in _GRAPH_CONFIG_KEYS:
            if cfg_key in config and config[cfg_key] is not None:
                kw[field] = config[cfg_key]
        # The kind chooses the dataclass; the FIELDS are identical, which is the point of V2
        # being a sibling rather than a knob.
        cls = ARCH_KINDS[arch_kind]
        return cls(in_dim=int(node_feat_dim), edge_dim=int(edge_feat_dim), **kw)
    raise RepresentationMismatch(
        f"spec.representation={rep!r} for encoding "
        f"{getattr(spec, 'name', '?')!r} — expected 'graph'."
    )
