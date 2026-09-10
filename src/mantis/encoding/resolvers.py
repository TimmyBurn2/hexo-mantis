# >300 lines: ports the OLD resolver surface whole (config/checkpoint/state-dict resolvers +
# corpus/anchor/held-out registries + the unified detector) as one cohesive delegating shim;
# splitting would scatter the single resolver authority.
"""Encoding resolvers — config-form, checkpoint-form, state-dict detection.

The `resolve_*` functions are the blessed paths to construct an `EncodingSpec` outside the
registry itself. The state-dict detector is UNIFIED: marker/stamp FIRST for grid AND graph, then
a single deterministic shape fallback over the registered set. A filename is NEVER a dispatch
signal.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis._engine import RegistrySpec as EncodingSpec
from mantis.encoding._probes import GNN_GRAPH_MARKER_KEY as _GNN_GRAPH_MARKER_KEY
from mantis.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)
from mantis.encoding.registry import (
    _load as _load_registry,
)


class ShapeMismatchError(Exception):
    """Raised when state-dict shapes contradict an EncodingSpec."""


class EncodingDeclarationConflictError(EncodingRegistryError):
    """Raised when a config declares an encoding in TWO shapes that DISAGREE. NOT a subclass of
    `MissingEncodingError`: the anchor maps that to `None`, so a corrupt declaration must not
    degrade into an absent one."""


class AmbiguousGraphMarkerError(EncodingRegistryError):
    """An unstamped graph state dict cannot be resolved: >1 graph encoding is registered. The
    marker key says the checkpoint is a graph, never WHICH graph, and the two rows differ only in
    a geometry no stamp records — so a guess here is unfalsifiable downstream."""


class MissingEncodingError(EncodingRegistryError):
    """Raised when an encoding value is absent (LAW-11). A subclass of `EncodingRegistryError`, so
    a caller wanting to distinguish "never specified" from "unknown to the registry" can catch
    this one; the retired default arm is dead and an absent encoding is always an error."""


# Sentinel used by expand_auto_paths to detect unresolved artifact paths.
_AUTO = "<auto>"


def normalize_encoding_name(enc: Any) -> str:
    """Coerce a config encoding value to its registry name string, accepting a str, a dict carrying
    ``version``/``name``, or an object with ``.name``.

    Raises:
        MissingEncodingError: if ``enc`` is ``None`` — an explicit encoding is required (LAW-11).
    """
    if enc is None:
        raise MissingEncodingError(
            "encoding value is None; an explicit encoding name/dict/EncodingSpec "
            "is required (LAW-11, R28) — the v6 default arm is retired"
        )
    if isinstance(enc, str):
        return enc
    if isinstance(enc, Mapping):
        name = enc.get("name", enc.get("version"))
        if name is None:
            raise MissingEncodingError(
                "encoding mapping carries neither 'name' nor 'version'; an explicit "
                "encoding is required (LAW-11, R28) — the v6 default arm is retired"
            )
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"encoding mapping name/version must be a string; "
                f"got {type(name).__name__}: {name!r}"
            )
        return name
    name = getattr(enc, "name", None)
    if isinstance(name, str):
        return name
    raise EncodingRegistryError(
        f"cannot extract encoding name from {type(enc).__name__}: {enc!r}"
    )


_SCATTERED_KEYS_TO_FIELD: dict[str, str] = {
    "board_size": "board_size",
    "cluster_window_size": "cluster_window_size",
    "cluster_threshold": "cluster_threshold",
    "legal_move_radius": "legal_move_radius",
    "n_planes": "n_planes",
    "in_channels": "n_planes",
}


def _check_scattered_keys(cfg: Mapping[str, Any], spec: EncodingSpec) -> None:
    """Raise EncodingRegistryError if any scattered key disagrees with spec: where a key is present
    in the config AND the registry spec has a non-None value, the integers must match."""
    if not cfg:
        return
    disagreements: list[str] = []
    for cfg_key, spec_field in _SCATTERED_KEYS_TO_FIELD.items():
        cfg_val = cfg.get(cfg_key)
        if cfg_val is None:
            continue
        spec_val = getattr(spec, spec_field, None)
        if spec_val is None:
            continue
        try:
            cfg_int = int(cfg_val)
        except (TypeError, ValueError):
            disagreements.append(
                f"  - {cfg_key}={cfg_val!r} (config) is not an int; "
                f"{spec_field}={spec_val} (encoding {spec.name!r})"
            )
            continue
        if cfg_int != int(spec_val):
            disagreements.append(
                f"  - {cfg_key}={cfg_val} (config) vs {spec_field}={spec_val} "
                f"(encoding {spec.name!r} from registry.toml)"
            )
    if disagreements:
        raise EncodingRegistryError(
            f"variant config has scattered key(s) that disagree with the "
            f"declared encoding {spec.name!r}:\n"
            + "\n".join(disagreements)
            + f"\n\nRemove the scattered key(s) and let the registry decide. "
            f"Registered encodings: {sorted(_load_registry())}. "
            f"Schema: docs/contracts/registry.md."
        )


# Canonical artifact paths per encoding name, repo-relative and keyed by registered encoding name;
# only registered specs reach these lookups.

_CORPUS_PATHS: dict[str, str] = {
    "gnn_axis_v1": "data/gnn_corpus_v1.hexg",
    "gnn_axis_r8": "data/gnn_corpus_r8.hexg",
}

_CORPUS_SHA_PINS: dict[str, str] = {
    # Launch-pinned sha256: a corpus with a pin must be byte-identical across hosts, and absence
    # of an encoding here means "no launch pin enforced". The dict stays because the mechanism
    # does, and an EMPTY pin set is a truthful "none enforced".
}


def resolve_corpus_sha_pin(spec: Any) -> str | None:
    """Launch-pinned sha256 for encoding *spec*'s canonical corpus, or `None` when no pin is
    registered — callers must treat `None` as "not enforced"."""
    return _CORPUS_SHA_PINS.get(spec.name)


# Held-out corpus registry — sha256 -> (label, on-disk byte size). A held-out corpus loaded
# through a TRAINING corpus path is a hard, labelled error.
_HELDOUT_CORPUS_SHAS: dict[str, tuple[str, int]] = {
    "s5_post20260704": (
        "88f99c2b5fea7495484e4e9cc1af831d1e053221dc7e0f9c8f5d3ab6f27aa69e",
        12872280,
    ),
}


def held_out_shas() -> frozenset[str]:
    """All registered held-out corpus sha256 values; any of these loaded through a TRAINING corpus
    path is a hard, labelled error — see `assert_not_heldout_sha`."""
    return frozenset(sha for sha, _size in _HELDOUT_CORPUS_SHAS.values())


def assert_not_heldout_sha(actual_sha: str, *, path: Any) -> None:
    """Raise if *actual_sha* is a registered held-out corpus sha.

    Call this from any training-path corpus loader BEFORE using the file's contents.

    Args:
        actual_sha: sha256 of the file actually on disk (freshly streamed).
        path: the path being loaded (for the error message only).

    Raises:
        ValueError: if `actual_sha` matches a registered held-out sha.
    """
    for label, (sha, _size) in _HELDOUT_CORPUS_SHAS.items():
        if actual_sha == sha:
            raise ValueError(
                f"corpus at {path} is the HELD-OUT set {label!r} "
                f"(sha {actual_sha[:12]}…) — held-out corpora are reserved for "
                f"future BC/architecture reads and must NEVER enter a training "
                f"corpus load. This looks like a misconfigured "
                f"pretrained_buffer_path pointing at a held-out artifact."
            )



def _assert_no_registry_overlap() -> None:
    """`_CORPUS_SHA_PINS` and `_HELDOUT_CORPUS_SHAS` must never share a sha256: a held-out set
    also registered as a launch corpus pin would silently defeat both gates. Called once at import
    so a bad registry entry fails loudly, and directly callable from tests."""
    overlap = set(_CORPUS_SHA_PINS.values()) & held_out_shas()
    if overlap:
        raise EncodingRegistryError(
            f"corpus sha registries overlap: {sorted(overlap)!r} present in "
            f"BOTH _CORPUS_SHA_PINS and _HELDOUT_CORPUS_SHAS — a held-out set "
            f"and a launch-pinned training corpus cannot share a sha256."
        )


_assert_no_registry_overlap()


# The graph lineage warm-starts from `identity.warm_start`, a minted config row, not a path table.
_ANCHOR_PATHS: dict[str, str] = {}


# Architecture resolver — ONE registry-derived map from an encoding NAME to the arch facts
# consumers used to hardcode. Every field is computed from `lookup(name)`.
@dataclass(frozen=True)
class ArchSpec:
    """Registry-derived architecture facts for a single encoding: a thin, typed, immutable view
    over the registry `EncodingSpec`, every field computed from `lookup(name)` and never
    hardcoded."""

    name: str
    k_max: int                     # = spec.k_max
    policy_logit_count: int        # = spec.policy_logit_count


def resolve_arch(name: Any) -> ArchSpec:
    """Resolve an encoding NAME (str / dict / EncodingSpec) to its `ArchSpec` — the one
    registry-derived resolver, so never shape-sniff a checkpoint or hardcode a plane count."""
    spec = lookup(normalize_encoding_name(name))
    return ArchSpec(
        name=spec.name,
        k_max=spec.k_max,
        policy_logit_count=spec.policy_logit_count,
    )


def resolve_corpus_path(spec: Any) -> Path:
    """Canonical corpus npz for an encoding.

    Raises:
        EncodingRegistryError: if no canonical path is registered for spec.name.
    """
    p = _CORPUS_PATHS.get(spec.name)
    if p is None:
        raise EncodingRegistryError(
            f"No canonical corpus path registered for encoding {spec.name!r}. "
            "Add an entry to _CORPUS_PATHS in mantis/encoding/resolvers.py."
        )
    return Path(p)


def resolve_anchor_path(spec: Any) -> Path:
    """Canonical bootstrap anchor checkpoint for an encoding.

    Raises:
        EncodingRegistryError: if no canonical path is registered for spec.name.
    """
    p = _ANCHOR_PATHS.get(spec.name)
    if p is None:
        raise EncodingRegistryError(
            f"No canonical anchor path registered for encoding {spec.name!r}. "
            "Add an entry to _ANCHOR_PATHS in mantis/encoding/resolvers.py."
        )
    return Path(p)


def expand_auto_paths(config: dict[str, Any], spec: Any) -> None:
    """Expand ``<auto>`` literals in *config* using the canonical artifact paths, in place. Handles
    both flat top-level keys and the nested keys in variant YAML files, and only where the current
    value is the literal string ``"<auto>"``."""
    if config.get("corpus_npz") == _AUTO:
        config["corpus_npz"] = str(resolve_corpus_path(spec))
    if config.get("bootstrap_anchor") == _AUTO:
        config["bootstrap_anchor"] = str(resolve_anchor_path(spec))

    mixing = config.get("mixing")
    if isinstance(mixing, dict) and mixing.get("pretrained_buffer_path") == _AUTO:
        mixing["pretrained_buffer_path"] = str(resolve_corpus_path(spec))
        # Stamp provenance so a corpus loader can require a sha pin for THIS path — only
        # <auto>-resolved paths carry the flag.
        mixing["_pretrained_buffer_path_auto_resolved"] = True

    eval_cfg = config.get("eval_pipeline")
    if isinstance(eval_cfg, dict):
        opponents = eval_cfg.get("opponents")
        if isinstance(opponents, dict):
            anchor_cfg = opponents.get("bootstrap_anchor")
            if isinstance(anchor_cfg, dict) and anchor_cfg.get("path") == _AUTO:
                anchor_cfg["path"] = str(resolve_anchor_path(spec))


def resolve_from_config(cfg: Mapping[str, Any] | None) -> EncodingSpec:
    """Return an `EncodingSpec` from a config mapping — THE one authority for where in a config an
    encoding is declared.

    Three DECLARED shapes are accepted: `cfg['encoding']` as a string or as a mapping carrying
    `version`, and `cfg['identity']['encoding']`, which is what `RunConfig` dumps. There is no
    precedence between them: a config carrying BOTH must carry the SAME name, and a disagreement
    raises `EncodingDeclarationConflictError` rather than picking a side. The nested shape is not a
    fallback — an absent declaration still raises, and a caller-side injection of one shape into
    the other would be the code-side default authority LAW-11 forbids.

    Raises:
        MissingEncodingError: if `cfg` is `None`, declares an encoding in NONE of the three
            shapes, or has a mapping-form `encoding` with no `version` key.
    """
    if cfg is None:
        raise MissingEncodingError(
            "resolve_from_config(None): a config mapping is required (LAW-11, "
            "R28) — the v6 default arm is retired"
        )
    section = cfg.get("encoding")
    nested = None
    identity = cfg.get("identity")
    if isinstance(identity, Mapping):
        nested = identity.get("encoding")
    if section is None:
        section = nested
    elif nested is not None:
        # BOTH shapes declared. Disagreement is CORRUPT INPUT, not a precedence question, so the
        # one authority refuses to pick a winner; flat malformation raises its own error below
        # before the comparison can pass.
        flat_name = section if isinstance(section, str) else (
            section.get("version") if isinstance(section, Mapping) else None
        )
        if isinstance(flat_name, str) and isinstance(nested, str) and flat_name != nested:
            raise EncodingDeclarationConflictError(
                f"config declares TWO encodings that disagree: flat "
                f"`encoding` says {flat_name!r}, nested `identity.encoding` says "
                f"{nested!r}. A dual-shape config whose declarations disagree is "
                "corrupt input — fix the config; no precedence arm exists (R104)"
            )
    if section is None:
        raise MissingEncodingError(
            "config declares no encoding: neither a flat `encoding: <name>` key "
            "nor a nested `identity.encoding` one. An explicit declaration is "
            "required (LAW-11, R28) — the v6 default arm is retired"
        )
    spec: EncodingSpec
    if isinstance(section, str):
        spec = lookup(section)
    elif isinstance(section, Mapping):
        version = section.get("version")
        if version is None:
            raise MissingEncodingError(
                "config's 'encoding' mapping has no 'version' key; an "
                "explicit version is required (LAW-11, R28) — the v6 "
                "default arm is retired"
            )
        if not isinstance(version, str):
            raise EncodingRegistryError(
                f"encoding.version must be a string; got {type(version).__name__}"
            )
        spec = lookup(version)
    else:
        raise EncodingRegistryError(
            f"encoding section must be str or mapping; got {type(section).__name__}"
        )

    _check_scattered_keys(cfg, spec)
    return spec


def _graph_specs() -> list[EncodingSpec]:
    """Registered GRAPH-representation specs. Derived, so pruning back to one re-arms
    the marker branch without an edit."""
    return [s for s in all_specs() if s.representation == "graph"]


#: REPORT-ONLY. The function below dispatches on an ARCH-STRUCTURAL key, so a V3 graph arch that
#: renames its trunk entry resolves nothing, and `checkpoints.load_legacy_weights` — the loader
#: for exactly these artifacts — REFUSES to shape-sniff, which is the posture that governs.
#: NOTHING ON A DISPATCH PATH MAY CALL THIS: the encoding a run uses comes from a stamp or an
#: explicit declaration. `mantis.encoding.audit_sections` calls it to REPORT a declared-vs-inferred
#: reconciliation, which selects no behaviour.
def detect_encoding_from_state_dict(
    state: Mapping[str, Any],
    ckpt_label: str,
    strict: bool = False,
) -> EncodingSpec | None:
    """Detect a registry encoding from a model state-dict.

    UNIFIED precedence for grid and graph alike: an embedded ``metadata['encoding_name']`` STAMP
    wins outright, then the graph-representation MARKER key, then ONE deterministic shape fallback
    matching uniquely over the registered grid set. The filename is used ONLY in error text.

    Args:
        state: Model state-dict (key → tensor), optionally with a `metadata` envelope.
        ckpt_label: Free-form label/path used only in error messages.
        strict: If True, raise ValueError on no canonical match; else None.
    """
    # 1. STAMP — an embedded encoding_name beats shape for every kind.
    meta = state.get("metadata") if hasattr(state, "get") else None
    if isinstance(meta, Mapping):
        stamped = meta.get("encoding_name")
        if isinstance(stamped, str):
            return lookup(stamped)

    # 2. MARKER — resolve by marker BEFORE any shape probe, so it beats shape and filename.
    if _GNN_GRAPH_MARKER_KEY in state:
        graph = _graph_specs()
        if len(graph) != 1:
            raise AmbiguousGraphMarkerError(
                f"checkpoint {ckpt_label} carries the graph marker key but no encoding stamp, "
                f"and {len(graph)} graph encodings are registered "
                f"({', '.join(sorted(s.name for s in graph))}). The marker says GRAPH; it has "
                "never said WHICH graph. Refusing rather than picking one (LAW-11): an "
                "unstamped r6 checkpoint resolved as r8 differs only in a geometry no stamp "
                "records. Stamp the checkpoint or pass the encoding explicitly."
            )
        return graph[0]

    if strict:
        raise ValueError(
            f"checkpoint {ckpt_label} carries neither an encoding stamp nor the graph marker "
            "key; the dense shape fallback went with the grid path (R346(f)). Stamp the "
            "checkpoint or pass the encoding explicitly (LAW-11)."
        )
    return None
def resolve_from_checkpoint(path: str | Path) -> EncodingSpec:
    """Return an `EncodingSpec` for a saved checkpoint — from its STAMP, never its shape.

    This read `torch.load(weights_only=False)`, executing arbitrary pickle against a contract that
    asserts no pickle-exec fallback exists; absent a stamp it also fell through to a shape sniffer
    dispatching on an ARCH-STRUCTURAL key, so a graph arch that renames `input_proj` was silently
    classified as GRID. An artifact with no `encoding_name` now RAISES by name, deliberately
    harder than the old warning, since proceeding on a guess is what left artifacts unstamped.

    Raises:
        EncodingRegistryError: the payload is not a mapping, carries no
            `metadata['encoding_name']`, or that field is not a string.
    """
    import torch

    d = torch.load(path, map_location="cpu", weights_only=True)
    meta = d.get("metadata") if isinstance(d, dict) else None
    if isinstance(meta, dict) and "encoding_name" in meta:
        name = meta["encoding_name"]
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"checkpoint {path}: metadata['encoding_name'] is "
                f"{type(name).__name__}, expected str"
            )
        return lookup(name)

    raise EncodingRegistryError(
        f"checkpoint {path}: no metadata['encoding_name'] to resolve an encoding from. This "
        "used to fall through to a state-dict SHAPE inference, which dispatches on an "
        "arch-structural key and on conv widths — so a renamed graph trunk read as grid "
        "(AUDIT-1 F-20). An artifact's encoding is its STAMP's. Pass the encoding explicitly, "
        "or re-stamp the artifact."
    )
