# >300 lines: ports the OLD resolver surface whole (config/checkpoint/state-dict
# resolvers + corpus/anchor/held-out registries + the unified detector) as one
# cohesive delegating shim; splitting would scatter the single resolver authority.
"""Encoding resolvers — config-form, checkpoint-form, state-dict detection.

The `resolve_*` functions are the blessed paths to construct an `EncodingSpec`
outside the registry itself. The state-dict detector is UNIFIED (LOCKED #7): the
two historic divergent detectors (filename-first vs shape-first) converge into
ONE precedence — marker/stamp FIRST for grid AND graph, then a single
deterministic shape fallback over the registered set. A filename is NEVER a
dispatch signal.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mantis._engine import RegistrySpec as EncodingSpec
from mantis.encoding._probes import FIRST_CONV_KEYS as _FIRST_CONV_KEYS
from mantis.encoding._probes import GNN_GRAPH_MARKER_KEY as _GNN_GRAPH_MARKER_KEY
from mantis.encoding._probes import POLICY_FC_KEYS as _POLICY_FC_KEYS
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


class MissingEncodingError(EncodingRegistryError):
    """Raised when an encoding value is absent (R28, LAW-11).

    A subclass of `EncodingRegistryError` — a caller catching the parent for
    "encoding trouble in general" keeps working, while a caller wanting to
    distinguish "never specified" from "specified but unknown to the
    registry" can catch this subclass specifically. The retired behaviour
    silently resolved an absent encoding to the "v6" default; that default
    arm is killed — an absent encoding is now always an error.
    """


# Sentinel used by expand_auto_paths to detect unresolved artifact paths.
_AUTO = "<auto>"


def normalize_encoding_name(enc: Any) -> str:
    """Coerce a config encoding value to its registry name string.

    Accepts the three forms that show up at consumer sites:
      - str ``"v6"``                           → returned as-is
      - dict ``{"version": "v6", ...}`` or
             ``{"name": "v6", ...}``           → version/name extracted
      - object with ``.name`` (EncodingSpec)   → ``.name`` returned

    Raises:
        MissingEncodingError: if ``enc`` is ``None`` — an explicit encoding
            name/dict/EncodingSpec is required (LAW-11, R28); the v6 default
            arm is retired.
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
    """Raise EncodingRegistryError if any scattered key disagrees with spec.

    Consistency rule: if a key is present in the config AND the registry spec
    has a non-None value for the corresponding field, the integers must match.
    Keys absent from the config or with `None` registry values are skipped.
    """
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


# ---------------------------------------------------------------------------
# Canonical artifact paths per encoding name. Repo-relative (no leading slash).
# Keyed by registered encoding name; only registered specs reach these lookups.
# ---------------------------------------------------------------------------

_CORPUS_PATHS: dict[str, str] = {
    "v6":          "data/bootstrap_corpus.npz",
    "v6_live2_ls": "data/bootstrap_corpus_v6_live2_ls.npz",
    "v6w25":       "data/bootstrap_corpus_v6w25.npz",
    "gnn_axis_v1": "data/gnn_corpus_v1.hexg",
}

_CORPUS_SHA_PINS: dict[str, str] = {
    # Launch-pinned sha256 — a corpus with a pin must be byte-identical across
    # hosts. Absence of an encoding here means "no launch pin enforced".
    "v6_live2_ls": "3813edc2fb10a7c5ab976a0293e38cbba0fd6b84e5295630f339ca421b345c97",
}


def resolve_corpus_sha_pin(spec: Any) -> str | None:
    """Launch-pinned sha256 for encoding *spec*'s canonical corpus, if any.

    Returns the lowercase-hex sha256 string, or `None` when no pin is
    registered for `spec.name` — callers must treat `None` as "not enforced".
    """
    return _CORPUS_SHA_PINS.get(spec.name)


# ---------------------------------------------------------------------------
# Held-out corpus registry — sha256 -> (label, on-disk byte size). A held-out
# corpus loaded through a TRAINING corpus path is a hard, labelled error. The
# size is a cheap stat-only pre-filter before a full sha256 stream.
# ---------------------------------------------------------------------------
_HELDOUT_CORPUS_SHAS: dict[str, tuple[str, int]] = {
    "s5_post20260704": (
        "88f99c2b5fea7495484e4e9cc1af831d1e053221dc7e0f9c8f5d3ab6f27aa69e",
        12872280,
    ),
}


def held_out_shas() -> frozenset[str]:
    """All registered held-out corpus sha256 values.

    Any of these loaded through a TRAINING corpus path is a hard, labelled
    error — see `assert_not_heldout_sha`.
    """
    return frozenset(sha for sha, _size in _HELDOUT_CORPUS_SHAS.values())


def assert_not_heldout_sha(actual_sha: str, *, path: Any) -> None:
    """Raise if *actual_sha* is a registered held-out corpus sha.

    Held-out corpora exist ONLY for future BC/architecture reads; they must
    NEVER enter a training corpus load. Call this from any training-path corpus
    loader BEFORE using the file's contents.

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


def heldout_size_bytes() -> frozenset[int]:
    """On-disk byte sizes of every registered held-out artifact.

    Cheap (stat-only) pre-filter for callers that want to skip the
    `assert_not_heldout_sha` sha256 stream unless a candidate file's size
    could possibly match.
    """
    return frozenset(size for _sha, size in _HELDOUT_CORPUS_SHAS.values())


def _assert_no_registry_overlap() -> None:
    """Resolver-level static invariant: `_CORPUS_SHA_PINS` and
    `_HELDOUT_CORPUS_SHAS` must never share a sha256 — a held-out set
    accidentally also registered as a launch corpus pin (or vice versa) would
    silently defeat both gates. Called once at import time (below) so a bad
    registry entry fails loudly at first import; also directly callable from
    tests against a monkeypatched registry.
    """
    overlap = set(_CORPUS_SHA_PINS.values()) & held_out_shas()
    if overlap:
        raise EncodingRegistryError(
            f"corpus sha registries overlap: {sorted(overlap)!r} present in "
            f"BOTH _CORPUS_SHA_PINS and _HELDOUT_CORPUS_SHAS — a held-out set "
            f"and a launch-pinned training corpus cannot share a sha256."
        )


_assert_no_registry_overlap()


_ANCHOR_PATHS: dict[str, str] = {
    "v6":          "checkpoints/bootstrap_model_v6.pt",
    "v6_live2_ls": "checkpoints/bootstrap_model_v6_live2.pt",
    "v6w25":       "checkpoints/bootstrap_model_v6w25.pt",
}


# ---------------------------------------------------------------------------
# Architecture resolver — ONE registry-derived map from an encoding NAME to the
# arch facts consumers used to hardcode (plane count, kept-index list, stone
# slots, policy width). Every field is computed from `lookup(name)`.
#
# Source-plane semantics fixed by the v6 wire format:
#   0       → current-player stone, t0          (always kept-slot 0)
#   8       → opponent stone, t0
#   1,2,3   → current-player history t-1..t-3
#   9,10,11 → opponent history t-1..t-3
#   16,17   → turn-phase scalars (moves_remaining / ply_parity)
# ---------------------------------------------------------------------------

_CUR_STONE_SRC_PLANE = 0
_OPP_STONE_SRC_PLANE = 8
_MOVES_REMAINING_SRC_PLANE = 16
_PLY_PARITY_SRC_PLANE = 17
_HISTORY_SRC_PLANES = frozenset({1, 2, 3, 9, 10, 11})
_TURN_PHASE_SRC_PLANES = frozenset({_MOVES_REMAINING_SRC_PLANE, _PLY_PARITY_SRC_PLANE})


def _kept_slot_of(kept: list[int], src_plane: int) -> int:
    """Position of ``src_plane`` within the encoding's kept-plane order."""
    return kept.index(src_plane)


@dataclass(frozen=True)
class ArchSpec:
    """Registry-derived architecture facts for a single encoding.

    A thin, typed, immutable view over the registry `EncodingSpec` — every
    field is computed from `lookup(name)`, never hardcoded.
    """

    name: str
    in_channels: int               # = spec.n_planes (model trunk in_channels)
    kept_indices: tuple[int, ...]  # = spec.kept_plane_indices (source→wire slice)
    cur_stone_slot: int            # kept-slot of source plane 0 (always 0)
    opp_stone_slot: int            # kept-slot of source plane 8
    k_max: int                     # = spec.k_max (multi-window cluster cap)
    policy_logit_count: int        # = spec.policy_logit_count
    history_planes: tuple[int, ...]     # kept-slots of source {1,2,3,9,10,11}
    turn_phase_planes: tuple[int, ...]  # kept-slots of source {16,17}


def resolve_arch(name: Any) -> ArchSpec:
    """Resolve an encoding NAME (str / dict / EncodingSpec) to its `ArchSpec`.

    The one registry-derived resolver: never shape-sniff a checkpoint, never
    hardcode a plane count or kept-index list — call this by name.
    """
    spec = lookup(normalize_encoding_name(name))
    kept = list(spec.kept_plane_indices)
    history = tuple(i for i, src in enumerate(kept) if src in _HISTORY_SRC_PLANES)
    turn_phase = tuple(i for i, src in enumerate(kept) if src in _TURN_PHASE_SRC_PLANES)
    return ArchSpec(
        name=spec.name,
        in_channels=spec.n_planes,
        kept_indices=tuple(kept),
        cur_stone_slot=_kept_slot_of(kept, _CUR_STONE_SRC_PLANE),
        opp_stone_slot=_kept_slot_of(kept, _OPP_STONE_SRC_PLANE),
        k_max=spec.k_max,
        policy_logit_count=spec.policy_logit_count,
        history_planes=history,
        turn_phase_planes=turn_phase,
    )


def cur_stone_slot(spec: Any) -> int:
    """Slice index of the current-player t0 stone plane (source plane 0)."""
    return _kept_slot_of(list(spec.kept_plane_indices), _CUR_STONE_SRC_PLANE)


def opp_stone_slot(spec: Any) -> int:
    """Slice index of the opponent t0 stone plane (source plane 8) within the
    encoding's kept-plane order. Derived from the registry, never hardcoded."""
    return _kept_slot_of(list(spec.kept_plane_indices), _OPP_STONE_SRC_PLANE)


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
    """Expand ``<auto>`` literals in *config* using the canonical artifact paths.

    Mutates *config* in-place. Handles both flat top-level keys and the nested
    keys present in variant YAML files. Only expands when the current value is
    the literal string ``"<auto>"``.
    """
    if config.get("corpus_npz") == _AUTO:
        config["corpus_npz"] = str(resolve_corpus_path(spec))
    if config.get("bootstrap_anchor") == _AUTO:
        config["bootstrap_anchor"] = str(resolve_anchor_path(spec))

    mixing = config.get("mixing")
    if isinstance(mixing, dict) and mixing.get("pretrained_buffer_path") == _AUTO:
        mixing["pretrained_buffer_path"] = str(resolve_corpus_path(spec))
        # Stamp provenance so a corpus loader can require a sha pin for THIS
        # path — only <auto>-resolved paths carry the flag.
        mixing["_pretrained_buffer_path_auto_resolved"] = True

    eval_cfg = config.get("eval_pipeline")
    if isinstance(eval_cfg, dict):
        opponents = eval_cfg.get("opponents")
        if isinstance(opponents, dict):
            anchor_cfg = opponents.get("bootstrap_anchor")
            if isinstance(anchor_cfg, dict) and anchor_cfg.get("path") == _AUTO:
                anchor_cfg["path"] = str(resolve_anchor_path(spec))


def resolve_from_config(cfg: Mapping[str, Any] | None) -> EncodingSpec:
    """Return an `EncodingSpec` from a config mapping. THE one authority for
    *where in a config an encoding is declared* (CARD-POOL-ENCODING-BRIDGE / TD-4).

    Accepts three DECLARED shapes, in precedence order:
      - `cfg['encoding'] = "v6w25"`             (legacy flat, string form)
      - `cfg['encoding'] = {'version': 'v6'}`   (legacy flat, mapping form)
      - `cfg['identity']['encoding'] = "v6w25"` (WP8 nested — what `RunConfig`
        actually dumps; `IdentityConfig.encoding` is a required, defaultless,
        registry-cross-checked field, `config/schema/core.py:50`)

    The nested shape is NOT a fallback and NOT a default: it reads a key the
    schema requires the operator to declare, and an absent declaration still
    raises. Before TD-4 this knowledge was duplicated in two private bridges
    (`train/trainer/core.py::_resolve_spec`, `train/orchestrator.py`) and absent
    from five other call sites — the pool among them, which is why mode
    PREFLIGHT could not boot. Adding a shape here rather than a sixth private
    bridge is the R1 "one authority" requirement: a caller-side
    `d["encoding"] = d["identity"]["encoding"]` injection is exactly the
    code-side default authority R1 and LAW-11 forbid.

    Raises:
        MissingEncodingError: if `cfg` is `None`, declares an encoding in NONE
            of the three shapes, or has a mapping-form `encoding` with no
            `version` key — an explicit encoding is required (LAW-11, R28); the
            v6 default arm is retired.
    """
    if cfg is None:
        raise MissingEncodingError(
            "resolve_from_config(None): a config mapping is required (LAW-11, "
            "R28) — the v6 default arm is retired"
        )
    section = cfg.get("encoding")
    if section is None:
        identity = cfg.get("identity")
        if isinstance(identity, Mapping):
            section = identity.get("encoding")
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


def _grid_specs() -> list[EncodingSpec]:
    """Registered GRID-representation specs (graph encodings excluded)."""
    return [s for s in all_specs() if getattr(s, "representation", "grid") != "graph"]


def detect_encoding_from_state_dict(
    state: Mapping[str, Any],
    ckpt_label: str,
    strict: bool = False,
) -> EncodingSpec | None:
    """Detect a registry encoding from a model state-dict.

    UNIFIED precedence (LOCKED #7 — the filename-beats-shape KILL). Applies to
    ALL kinds (grid AND graph):

      1. STAMP  — an embedded ``metadata['encoding_name']`` wins outright.
      2. MARKER — the graph-representation marker key
                  (`_probes.GNN_GRAPH_MARKER_KEY`) → the graph encoding.
      3. ONE deterministic shape fallback — probe in_channels (first conv) and
         n_actions (policy fc), then match uniquely over the registered grid
         set. strict=True raises `ValueError` on ambiguity/miss; strict=False
         returns None. The filename (`ckpt_label`) is used ONLY in error text —
         NEVER as a dispatch signal.

    Args:
        state: Model state-dict (key → tensor), optionally with a `metadata`
               envelope carrying an `encoding_name` stamp.
        ckpt_label: Free-form label/path used only in error messages.
        strict: If True, raise ValueError on no canonical match; else None.
    """
    # 1. STAMP — an embedded encoding_name beats shape for every kind.
    meta = state.get("metadata") if hasattr(state, "get") else None
    if isinstance(meta, Mapping):
        stamped = meta.get("encoding_name")
        if isinstance(stamped, str):
            return lookup(stamped)

    # 2. MARKER — a graph state dict has no grid conv marker; resolve by marker
    #    BEFORE any shape probe so it beats shape (and filename).
    if _GNN_GRAPH_MARKER_KEY in state:
        return lookup("gnn_axis_v1")

    # 3. Deterministic shape fallback (grid).
    inp_w = state.get("trunk.input_conv.conv.weight")
    if inp_w is None:
        inp_w = state.get("trunk.input_conv.weight")
    if inp_w is None or getattr(inp_w, "dim", lambda: 0)() != 4:
        if strict:
            raise ValueError(
                f"checkpoint {ckpt_label} has no trunk.input_conv(.conv)?.weight; "
                "cannot detect encoding"
            )
        return None
    in_ch = int(inp_w.shape[1])
    n_actions: int | None = None
    for k in ("policy_fc.weight", "cluster_pool.policy_mlp.2.weight"):
        w = state.get(k)
        if w is not None and getattr(w, "dim", lambda: 0)() == 2:
            n_actions = int(w.shape[0])
            break

    candidates = [
        s
        for s in _grid_specs()
        if s.n_planes == in_ch
        and (n_actions is None or s.policy_logit_count == n_actions)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if strict:
            raise ValueError(
                f"checkpoint {ckpt_label}: unsupported in_channels={in_ch} "
                f"(n_actions={n_actions}); no registered grid encoding matches"
            )
        return None
    # Ambiguous: shape alone does not disambiguate (e.g. in_ch=8 with no
    # n_actions probe matches multiple encodings). No filename tiebreak (KILL).
    if strict:
        names = sorted(s.name for s in candidates)
        raise ValueError(
            f"checkpoint {ckpt_label}: shape (in_channels={in_ch}, "
            f"n_actions={n_actions}) is ambiguous across {names}; stamp "
            f"metadata['encoding_name'] explicitly"
        )
    return None


def resolve_from_checkpoint(path: str | Path) -> EncodingSpec:
    """Return an `EncodingSpec` for a saved checkpoint.

    Reads `ckpt['metadata']['encoding_name']` if present. Otherwise falls back
    to the unified state-dict detector and emits a `DeprecationWarning`.
    """
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("metadata") if isinstance(d, dict) else None
    if isinstance(meta, dict) and "encoding_name" in meta:
        name = meta["encoding_name"]
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"checkpoint {path}: metadata['encoding_name'] is "
                f"{type(name).__name__}, expected str"
            )
        return lookup(name)

    if isinstance(d, dict) and "model_state" in d:
        sd = d["model_state"]
    else:
        sd = d
    if not isinstance(sd, Mapping):
        raise EncodingRegistryError(
            f"checkpoint {path}: cannot extract state-dict for shape inference"
        )
    # Lazy import — compat delegates back into this module's unified detector.
    from mantis.encoding import compat

    name = compat.infer_encoding_from_state_dict(sd, str(path))
    warnings.warn(
        f"checkpoint {path} has no metadata['encoding_name']; "
        f"inferred {name!r} from state-dict shape. Stamp metadata explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lookup(name)


def resolve_encoding_for_eval(
    checkpoint_path: str | Path,
    encoding_override: str | None = None,
) -> EncodingSpec:
    """Resolve an EncodingSpec for an eval/smoke/probe consumer.

    Priority:
      1. ``encoding_override`` (CLI flag) — direct lookup.
      2. ``resolve_from_checkpoint(checkpoint_path)`` — metadata, then
         shape-inference fallback (DeprecationWarning).
      3. Re-raise with a clearer "pass --encoding explicitly" message.
    """
    if isinstance(encoding_override, str) and encoding_override:
        return lookup(encoding_override)

    try:
        return resolve_from_checkpoint(checkpoint_path)
    except EncodingRegistryError as e:
        registered = sorted(_load_registry())
        raise EncodingRegistryError(
            f"could not auto-detect encoding for checkpoint {checkpoint_path!s}; "
            f"pass --encoding explicitly. Registered encodings: {registered}. "
            f"Underlying error: {e}"
        ) from e


def validate_against_state_dict(
    spec: EncodingSpec, state_dict: Mapping[str, Any]
) -> None:
    """Cross-check spec.policy_logit_count + spec.n_planes against a state-dict.

    Probes common key names for the policy fc and first conv. Silently no-ops
    for keys that don't appear. Raises `ShapeMismatchError` on disagreement.
    """
    pfc = None
    for k in _POLICY_FC_KEYS:
        if k in state_dict:
            pfc = state_dict[k]
            break
    if pfc is not None:
        out_features = int(pfc.shape[0])
        if out_features != spec.policy_logit_count:
            raise ShapeMismatchError(
                f"policy_fc out_features {out_features} != "
                f"spec.policy_logit_count {spec.policy_logit_count} "
                f"for encoding {spec.name!r}"
            )

    conv = None
    for k in _FIRST_CONV_KEYS:
        if k in state_dict:
            conv = state_dict[k]
            break
    if conv is not None:
        in_channels = int(conv.shape[1])
        if in_channels != spec.n_planes:
            raise ShapeMismatchError(
                f"first conv in_channels {in_channels} != "
                f"spec.n_planes {spec.n_planes} for encoding {spec.name!r}"
            )
