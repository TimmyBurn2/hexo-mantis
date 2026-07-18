"""Backward-compat encoding inference for legacy checkpoints.

Used by `resolve_from_checkpoint` when a checkpoint lacks
`metadata['encoding_name']`.

The historic filename-first heuristic is KILLED. This module now delegates to
the ONE unified detector (`resolvers.detect_encoding_from_state_dict`), whose
precedence is marker/stamp FIRST for ALL kinds (grid AND graph), then a single
deterministic shape fallback over the registered set. A filename is NEVER a
dispatch signal. `infer_encoding_from_state_dict` preserves its NAME-returning,
`EncodingRegistryError`-raising surface for the consumers that still catch it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mantis.encoding.registry import EncodingRegistryError


def infer_encoding_from_state_dict(
    state_dict: Mapping[str, Any], path_hint: str = ""
) -> str:
    """Return the registered encoding name for a legacy checkpoint.

    Delegates to the unified `detect_encoding_from_state_dict` (strict mode):
    stamp → graph marker → deterministic shape fallback; the filename is only
    used in error text, never for dispatch. Raises `EncodingRegistryError` on
    ambiguity or no-match (preserving the exception type consumers catch).
    """
    # Lazy import — resolvers imports this module at top; break the cycle here.
    from mantis.encoding.resolvers import detect_encoding_from_state_dict

    try:
        spec = detect_encoding_from_state_dict(state_dict, path_hint, strict=True)
    except ValueError as exc:
        raise EncodingRegistryError(str(exc)) from exc
    # strict=True never returns None (it raises on miss); guard defensively.
    if spec is None:  # pragma: no cover — strict path always raises on miss
        raise EncodingRegistryError(
            f"could not infer encoding from state-dict (path_hint={path_hint!r})"
        )
    return spec.name
