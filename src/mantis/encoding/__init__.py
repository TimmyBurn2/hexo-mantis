"""Canonical encoding registry — delegating shim over `mantis._engine`.

Public API:
  - `EncodingSpec`               type alias for `mantis._engine.RegistrySpec`
  - `lookup(name)`               by-name registry access (stable-instance)
  - `all_specs()`                iterate every registered spec
  - `resolve_from_config(cfg)`   resolve from a config mapping
  - `resolve_from_checkpoint(p)` resolve from a saved checkpoint
  - `validate_against_state_dict(spec, sd)` cross-check shapes
  - `EncodingRegistryError`      raised on parse / lookup failure
  - `ShapeMismatchError`         raised by validate_against_state_dict

Schema authoring lives in `crates/mantis-encoding/src/registry.toml`; the Rust
parser (compiled into `mantis._engine`) is the single source of truth. This
package is a thin delegating shim over `mantis._engine.RegistrySpec`.

Import-time registry-sha handshake (NEW-BUILD): the on-disk `registry.toml` is
hashed and compared to the compiled `_engine.registry_sha()`; a drift hard-errors
(a stale `.so` cannot silently serve a stale registry). See
`_registry_sha_handshake`.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib

from mantis import _engine
from mantis._engine import RegistrySpec as EncodingSpec  # inv22 alias
from mantis.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)

_LOG = logging.getLogger(__name__)

# On-disk registry, relative to the repo root (dev/test layout). The handshake
# resolves it by walking up from this file for the crate-source TOML.
_REGISTRY_TOML_REL = pathlib.PurePosixPath("crates/mantis-encoding/src/registry.toml")


def _resolve_registry_toml() -> pathlib.Path | None:
    """Locate the on-disk `registry.toml` via a repo-root anchor.

    Walks up from this file looking for `crates/mantis-encoding/src/registry.toml`.
    Returns the first match, or `None` in a truly-installed non-repo layout where
    the crate source is not present (the handshake then SKIPs with a logged
    reason — never a silent pass).
    """
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

    Resolution (repo_design §3; DESIGN §g.3):
      - In a dev/test/repo layout, hash the on-disk `registry.toml` and compare
        to `engine.registry_sha()`; a mismatch raises `EncodingRegistryError`
        (rebuild the extension). This is the stale-`.so`/stale-registry guard.
      - In a truly-installed non-repo layout (the crate-source TOML is absent),
        SKIP with a logged reason — NEVER a silent pass (a missing file must not
        masquerade as a match).

    Exposed for the LAW-07 mutation self-test: pass a mutated tmp-copy path and
    the call hard-errors; the unmutated on-disk file passes.
    """
    path = toml_path if toml_path is not None else _resolve_registry_toml()
    if path is None or not path.is_file():
        _LOG.info(
            "registry-sha handshake SKIPPED: on-disk registry.toml not found "
            "(installed-wheel / non-repo layout); trusting compiled "
            "_engine.registry_sha(). Searched anchor: %s",
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


# Fire the handshake at import (dev/test-scoped; skips cleanly when installed).
_registry_sha_handshake()


from mantis.encoding.resolvers import (  # noqa: E402 — after the import handshake
    ArchSpec,
    ShapeMismatchError,
    assert_not_heldout_sha,
    cur_stone_slot,
    detect_encoding_from_state_dict,
    expand_auto_paths,
    held_out_shas,
    normalize_encoding_name,
    opp_stone_slot,
    resolve_anchor_path,
    resolve_arch,
    resolve_corpus_path,
    resolve_corpus_sha_pin,
    resolve_from_checkpoint,
    resolve_from_config,
    validate_against_state_dict,
)

__all__ = [
    "ArchSpec",
    "EncodingSpec",
    "EncodingRegistryError",
    "ShapeMismatchError",
    "all_specs",
    "assert_not_heldout_sha",
    "cur_stone_slot",
    "detect_encoding_from_state_dict",
    "expand_auto_paths",
    "held_out_shas",
    "lookup",
    "normalize_encoding_name",
    "opp_stone_slot",
    "resolve_anchor_path",
    "resolve_arch",
    "resolve_corpus_path",
    "resolve_corpus_sha_pin",
    "resolve_from_checkpoint",
    "resolve_from_config",
    "validate_against_state_dict",
]
