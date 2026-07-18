"""Encoding registry — thin delegating shim over `mantis._engine.RegistrySpec`.

Counterpart to `crates/mantis-encoding/src/registry/mod.rs`. The Rust loader
(compiled into `mantis._engine`) is the single source of truth; this module is a
delegating shim over it.

The registered-name set is sourced from ONE `_engine.all_specs()` call — there is
NO hand-synced name tuple in Python (the hardcoded name list is KILLED; the
registry.toml, parsed by the Rust loader, is the sole name authority). `_load`
builds its cache from that single compiled call (no per-name `from_registry`
loop). All entry points delegate to the compiled registry.
"""
from __future__ import annotations

from collections.abc import Iterable

from mantis import _engine
from mantis._engine import RegistrySpec as _EngineRegistrySpec


class EncodingRegistryError(Exception):
    """Raised on registry parse failure or unknown encoding lookup.

    Preserved for backwards compatibility with consumers that catch this
    exception class. The underlying Rust loader raises Python `ValueError`
    on unknown lookup; `lookup` translates that into this exception type.
    """


_REGISTRY_CACHE: dict[str, _EngineRegistrySpec] | None = None


def _load() -> dict[str, _EngineRegistrySpec]:
    """Return the cached registry dict.

    Built from ONE `_engine.all_specs()` call — the compiled registry is the
    sole name source (no hardcoded name tuple, no per-name lookup loop).
    Preserved as a private helper because `compat.py`, `resolvers.py`, and
    `audit_sections.py` all import it.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    _REGISTRY_CACHE = {spec.name: spec for spec in _engine.all_specs()}
    return _REGISTRY_CACHE


def lookup(name: str) -> _EngineRegistrySpec:
    """Return spec for `name` or raise `EncodingRegistryError`.

    Cached: repeated calls with the same name return the identical
    `RegistrySpec` instance, preserving the ``lookup(name) is lookup(name)``
    round-trip identity contract.
    """
    cache = _load()
    spec = cache.get(name)
    if spec is None:
        raise EncodingRegistryError(
            f"unknown encoding {name!r}; registered: {sorted(cache)}"
        )
    return spec


def all_specs() -> Iterable[_EngineRegistrySpec]:
    """Iterate every registered spec (registry.toml insertion order)."""
    return _load().values()
