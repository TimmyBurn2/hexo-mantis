"""Canonical encoding registry — delegating shim over `mantis._engine`.

Public API:
  - `EncodingSpec`               type alias for `mantis._engine.RegistrySpec`
  - `lookup(name)`               by-name registry access (stable-instance)
  - `all_specs()`                iterate every registered spec
  - `resolve_from_config(cfg)`   resolve from a config mapping
  - `resolve_from_checkpoint(p)` resolve from a saved checkpoint
  - `EncodingRegistryError`      raised on parse / lookup failure

Schema authoring lives in `crates/mantis-encoding/src/registry.toml`; the Rust parser
compiled into `mantis._engine` is the single source of truth.

At import, `_registry_sha_handshake` hashes the on-disk `registry.toml` and compares it to
the compiled `_engine.registry_sha()`, so a stale `.so` cannot silently serve a stale
registry.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib

from mantis import _engine
from mantis._engine import RegistrySpec as EncodingSpec
from mantis.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)

_LOG = logging.getLogger(__name__)

#: Anchors the handshake skipped, in order; read through `handshake_skipped`/`handshake_ran`.
_skipped: list[str] = []

# On-disk registry relative to the repo root; resolved by walking up from this file.
_REGISTRY_TOML_REL = pathlib.PurePosixPath("crates/mantis-encoding/src/registry.toml")


def _resolve_registry_toml() -> pathlib.Path | None:
    """Return the on-disk `registry.toml`, or None in an installed non-repo layout."""
    here = pathlib.Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / _REGISTRY_TOML_REL
        if candidate.is_file():
            return candidate
    return None


def _registry_sha_handshake(
    toml_path: pathlib.Path | None = None,
    *,
    engine=_engine,
) -> None:
    """Hard-error if the on-disk registry.toml drifted from the compiled sha.

    An installed non-repo layout has no crate-source TOML: that case skips with a logged
    reason rather than passing silently. Passing `toml_path` explicitly is the mutation
    self-test surface — a mutated copy must hard-error.

    Raises:
        EncodingRegistryError: the on-disk sha differs from `engine.registry_sha()`.
    """
    path = toml_path if toml_path is not None else _resolve_registry_toml()
    if path is None or not path.is_file():
        # A skip means the stale-`.so` guard did not run, so it warns rather than informs.
        # Only the default resolution records: an explicit `toml_path` is a test probe and
        # must not poison what the composition root publishes about the real run.
        if toml_path is None:
            _skipped.append(str(_REGISTRY_TOML_REL))
        _LOG.warning(
            "registry-sha handshake SKIPPED: on-disk registry.toml not found "
            "(installed-wheel / non-repo layout); trusting compiled "
            "_engine.registry_sha() WITHOUT COMPARING IT. Searched anchor: %s",
            _REGISTRY_TOML_REL,
        )
        return
    disk = hashlib.sha256(path.read_bytes()).digest()
    if disk != engine.registry_sha():
        raise EncodingRegistryError(
            f"registry.toml on disk ({path}) drifted from the compiled "
            f"_engine registry_sha; the extension was built against a different "
            f"registry. Rebuild the extension (uv sync)."
        )


#: Every anchor a handshake SKIPPED on, for publication into the event stream. Read it as a
#: module attribute, live — a from-imported value binds the list object, not the fact.
handshake_skipped: list[str] = _skipped


def handshake_ran() -> bool:
    """Return True iff the registry-sha handshake actually compared a sha this process."""
    return not _skipped


# Fire the handshake at import; it skips cleanly in an installed layout.
_registry_sha_handshake()


from mantis.encoding.resolvers import (  # noqa: E402 — after the import handshake
    ArchSpec,
    assert_not_heldout_sha,
    detect_encoding_from_state_dict,
    expand_auto_paths,
    held_out_shas,
    normalize_encoding_name,
    resolve_anchor_path,
    resolve_arch,
    resolve_corpus_path,
    resolve_corpus_sha_pin,
    resolve_from_checkpoint,
    resolve_from_config,
)

__all__ = [
    "ArchSpec",
    "EncodingSpec",
    "EncodingRegistryError",
    "all_specs",
    "assert_not_heldout_sha",
    "detect_encoding_from_state_dict",
    "expand_auto_paths",
    "held_out_shas",
    "lookup",
    "normalize_encoding_name",
    "resolve_anchor_path",
    "resolve_arch",
    "resolve_corpus_path",
    "resolve_corpus_sha_pin",
    "resolve_from_checkpoint",
    "resolve_from_config",
]
