# >300 justify (R8). NO LINE COUNT is stated (G-DFIX-4 / R192(e), derive-or-delete). This module
# is ONE authority — the declared arch dataclasses, the kind vocabulary that names them, the
# pairing rule that says which representation admits which, the incumbent-by-history table, the
# config row that can override it, and the two resolvers that read all of the above. Splitting
# any of those out would put "which arch is this" in two files, which is the duplicate-authority
# class this module's own docstrings warn against three times over.
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


@dataclass(frozen=True)
class GnnArchV2:
    """Declared graph architecture, V2 — the `gnn_axis_v1` WIRE, two model-side mechanisms.

    A SIBLING of `GnnArch`, never a subclass. `build.build_net` and
    `train.checkpoints._arch_from_dict` both dispatch by type/kind, and a subclass would
    satisfy `isinstance(arch, GnnArch)` and silently build V1 — the hazard filed against the
    `_arch_from_dict` twin and never closed until this landed.

    The two mechanisms, which is what V2 IS (WP-AXIS2 candidates A and C(i), model-side half):
    a `concat(stone-masked mean, max over real nodes)` value readout, and a degree-normalized
    dummy aggregation. NO FIELD NAMES A PROPERTY V2 CLAIMS — fields name mechanisms or widths,
    because a per-arch field summarising a per-position fact is a lie with a type (R307(b)), and
    the conformance suite reads this union structurally.

    The field set matches `GnnArch` exactly, which is the point: the swap is the ARCH, not a
    knob on one. `representation` stays `"graph"` because the wire is unchanged — V2 consumes
    `gnn_axis_v1` and adds no registry row.
    """

    in_dim: int
    edge_dim: int
    hidden: int = 128
    num_layers: int = 4
    policy_hidden: int = 128
    value_hidden: int = 32
    n_value_bins: int = 65
    representation: Literal["graph"] = "graph"


ModelArch = CnnArch | GnnArch | GnnArchV2


class UnknownArchKind(ValueError):
    """A requested arch kind is not in `ARCH_KINDS`, or is not available on that representation.

    A `ValueError` for `RepresentationMismatch`'s reason: naming an arch this build does not
    have is a configuration ERROR, and the nearest member of the union is never substituted.
    """


#: THE ARCH-KIND VOCABULARY — the ONE naming authority for "which model kind is this"
#: (R322(d), candidate D). Keyed by the declared dataclass's own name, so the token and the
#: type cannot drift apart and a new arch adds exactly one row.
#:
#: It lives HERE and not in `mantis.train.checkpoints`, which is where B1 first needed it: a
#: kind vocabulary is a MODEL fact, and the loader is one of its consumers rather than its
#: owner. `checkpoints` imports this — the import direction the repo DAG requires anyway,
#: since `mantis.model` may not import `mantis.train`.
ARCH_KINDS: dict[str, type] = {
    "CnnArch": CnnArch,
    "GnnArch": GnnArch,
    "GnnArchV2": GnnArchV2,
}

#: Which kinds a representation admits — the pairing rule `SEAM_V1_DESIGN` §2.1 names as the
#: missing half of encoder/arch identity, stated for the arch side. `graph` admits TWO kinds
#: since GnnNetV2 landed, which is exactly why a selector has to exist.
ARCH_KINDS_BY_REPRESENTATION: dict[str, tuple[str, ...]] = {
    "grid": ("CnnArch",),
    "graph": ("GnnArch", "GnnArchV2"),
}

#: THE INCUMBENT KIND PER REPRESENTATION — a statement about HISTORY, not a default, and the
#: distinction is the whole reason this table is named and pinned rather than inlined.
#:
#: A default answers "what should we build when nobody said?"; this answers "what has this
#: tree always built?", which is a fact with a witness: every shipped config selects it, and
#: `tests/model/test_arch_selector.py` executes that against the real minted files rather than
#: asserting it here. `arch_from_spec_and_config` resolves through this so its behaviour is
#: byte-for-byte what it was before the selector existed, and `select_arch` — which takes the
#: kind EXPLICITLY and has no default at all — is the surface a caller uses to build anything
#: else.
#:
#: **THE CONFIG ROW EXISTS AND IS EMPTY IN EVERY SHIPPED CONFIG (R330(e) / R323(b)).** The
#: selector row is `identity.arch_kind` (`ARCH_KIND_ROW`), an OPTIONAL schema leaf, and the run6
#: mint is what first writes a value into it — a mint act, not an engine one, which is why the
#: plumbing lands here first and the row later. Until a config carries the row,
#: `arch_from_spec_and_config` resolves through THIS table; once it does, the row is the
#: authority and this table is not consulted. Absence is therefore not a fallback with a guess in
#: it: it is the statement "this config predates the row", and what such a config has always
#: built is the incumbent, pinned against the real minted files by
#: `tests/model/conformance/test_arch_selector_makes_v2_selectable.py`.
INCUMBENT_ARCH_KIND: dict[str, str] = {"grid": "CnnArch", "graph": "GnnArch"}

#: THE ONE CONFIG ROW that names an arch kind — dotted, as `RunConfig` spells it. Read by
#: `declared_arch_kind` and nowhere else, so the row has exactly one reader to change.
ARCH_KIND_ROW = "identity.arch_kind"


def declared_arch_kind(config: Mapping[str, Any]) -> str | None:
    """The `identity.arch_kind` row of a plain config mapping, or `None` when the config does not
    carry it (every config minted before the run6 mint, R323(b)).

    Mapping-typed for the same reason the rest of this module is: no `mantis.config` import.
    A row that is present but not a string is returned as-is and refused downstream by
    `select_arch`, which names the kind it was given — this reader never coerces.
    """
    identity = config.get("identity")
    if not isinstance(identity, Mapping):
        return None
    return identity.get("arch_kind")

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
    """Resolved encoding spec + plain config mapping → the arch the CONFIG selects.

    THE PRODUCTION ENTRY POINT for a caller that holds a run's config (R330(e)). The config's
    `identity.arch_kind` row, when it carries one, is the authority and is handed to
    `select_arch` verbatim — an unknown or non-admitted kind is refused there by name. A config
    that does not carry the row predates the run6 mint (R323(b)) and resolves to the
    representation's INCUMBENT kind, which is a history fact pinned against every minted file,
    not a guess: `select_arch(..., arch_kind=INCUMBENT_ARCH_KIND[representation])`, byte-for-byte
    what this function built before the row existed.

    A caller that holds an ARTIFACT and no config does not come here: the artifact's stamp is
    its authority (`mantis.train.checkpoints.stamped_arch_kind`), and reading this table for it
    would be a second answer to "which arch is this".

    Mapping-typed (NOT config-schema-typed) so the model layer is self-contained
    and testable without the config package.

    Raises:
        RepresentationMismatch: `spec.representation` is absent or unknown (LAW-11: no
            dense-by-default), or the encoding is incompatible with the requested config.
        UnknownArchKind: the config's `identity.arch_kind` row names a kind this build does not
            have, or one its representation does not admit.
    """
    rep = getattr(spec, "representation", None)
    if rep is None:
        raise RepresentationMismatch(
            f"spec {getattr(spec, 'name', spec)!r} has no representation attribute "
            "— cannot infer a model arch (no dense-by-default, LAW-11)."
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
    """Resolved encoding spec + plain config mapping + an EXPLICIT arch kind → declared arch.

    THE SELECTOR (R322(d), candidate D). `arch_kind` is keyword-only and has NO default: a
    caller that does not know which kind it wants is not entitled to one, and the absence of a
    default is what keeps this from becoming a second answer to "what does production build".

    Args:
        spec: a resolved encoding spec carrying `representation` and the geometry fields.
        config: a plain config mapping; per-arch width/depth keys are read from it where
            present, and an absent key falls to the dataclass field's own default, which is
            the sole default authority (R1).
        arch_kind: a member of `ARCH_KINDS`, admitted by `spec.representation` per
            `ARCH_KINDS_BY_REPRESENTATION`.

    Raises:
        UnknownArchKind: `arch_kind` is not a known kind, or the representation does not
            admit it. Named separately from `RepresentationMismatch` because "you asked for an
            arch that does not exist" and "your encoding and your model disagree" send a reader
            to two different places.
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
        kw = {}
        for cfg_key, field in _GRAPH_CONFIG_KEYS:
            if cfg_key in config and config[cfg_key] is not None:
                kw[field] = config[cfg_key]
        # The kind chooses the dataclass; the FIELDS are identical, which is the point of V2
        # being a sibling rather than a knob (see `GnnArchV2`'s own docstring).
        cls = ARCH_KINDS[arch_kind]
        return cls(in_dim=int(node_feat_dim), edge_dim=int(edge_feat_dim), **kw)
    raise RepresentationMismatch(
        f"spec.representation={rep!r} for encoding "
        f"{getattr(spec, 'name', '?')!r} — expected 'grid' or 'graph'."
    )
