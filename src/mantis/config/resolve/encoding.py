"""Encoding resolution authority (REBUILD of frozen resolve/encoding.py).

One rule for the encoding name: raise-on-conflict declared-vs-stamp, presence-before-normalize
(UNSPECIFIED sentinel, NOT None→"v6"), absent+stamp → stamp wins (metadata-wins). The frozen
absent+no-stamp "v6" terminal default is BANNED (LAW-11): it RAISES AbsentEncodingError.

Import constraint (DAG): stdlib + mantis.encoding ONLY — no torch, no eval/selfplay imports.
"""
from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mantis.encoding import normalize_encoding_name as _normalize

# Distinct sentinel for an ABSENT declaration (presence-before-normalize). NOT None: a None
# would normalize to a terminal default and make an absent key look PRESENT, I2-raising against
# any non-matching stamp.
UNSPECIFIED = types.new_class("_UnspecifiedEncodingSentinel", (), {})()


class EncodingConflictError(ValueError):
    """A PRESENT declared encoding disagrees with a PRESENT checkpoint stamp.

    Subclasses ValueError. The message NAMES both sources — a checkpoint may INFORM but never
    silently override a declared encoding.
    """

    def __init__(self, declared: str, stamp: str, *, stamp_source: str = "checkpoint"):
        self.declared = declared
        self.stamp = stamp
        self.stamp_source = stamp_source
        super().__init__(
            f"encoding conflict: declared={declared!r} vs {stamp_source}={stamp!r} "
            "(a checkpoint may INFORM but never silently override a declared encoding — "
            "re-stamp the checkpoint or fix the declared encoding)."
        )


class AbsentEncodingError(ValueError):
    """No encoding declared AND no checkpoint stamp (LAW-11).

    Δ-REBUILD vs frozen, which returned the terminal ``"v6"`` default. There is no dense/v6
    default: an absent identity key is an error.
    """


@dataclass(frozen=True)
class EncodingResolution:
    """Resolved encoding name + which source won (for provenance emission).

    ``source`` ∈ {"variant", "checkpoint"}. ``declared`` is the normalized declared name or the
    UNSPECIFIED sentinel; ``stamp`` is the normalized stamp name or None.
    """

    name: str
    source: str
    declared: Any
    stamp: str | None


def reconcile_encoding(
    declared: Any,
    stamp: str | None,
    *,
    stamp_source: str = "checkpoint",
) -> EncodingResolution:
    """Single raise-on-conflict rule (shared by both invocation surfaces when they land).

    ``declared`` is either the UNSPECIFIED sentinel (no declaration) or a name string; ``stamp``
    is the checkpoint's stamp name or None. Precedence:
      - declared PRESENT and stamp PRESENT and DIFFER → EncodingConflictError.
      - declared PRESENT (agrees, or no stamp) → declared wins, source="variant".
      - declared ABSENT and stamp PRESENT → stamp wins, source="checkpoint".
      - declared ABSENT and no stamp → AbsentEncodingError (LAW-11; no "v6" default).
    """
    declared_present = declared is not UNSPECIFIED
    if declared_present and stamp is not None:
        if declared != stamp:
            raise EncodingConflictError(declared, stamp, stamp_source=stamp_source)
        return EncodingResolution(declared, "variant", declared, stamp)
    if declared_present:
        return EncodingResolution(declared, "variant", declared, stamp)
    if stamp is not None:
        return EncodingResolution(stamp, "checkpoint", declared, stamp)
    raise AbsentEncodingError(
        "no encoding declared and no checkpoint stamp present: the identity key is required "
        "(LAW-11 — there is no terminal 'v6'/dense default). Declare identity.encoding "
        "explicitly, or resume from a stamped checkpoint."
    )


def normalize_declared(present: bool, raw: Any) -> object:
    """Presence-before-normalize: a present declaration → normalized name; absent → UNSPECIFIED."""
    return _normalize(raw) if present else UNSPECIFIED


def normalize_stamp(stamps: Mapping[str, Any]) -> str | None:
    """Normalize the checkpoint's encoding stamp (or None when no stamp is present)."""
    if "encoding" not in stamps:
        return None
    return _normalize(stamps["encoding"])
