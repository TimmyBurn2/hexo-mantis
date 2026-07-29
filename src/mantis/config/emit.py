"""Resolved-config emit surface (REBUILD — thin per-knob source-tagging).

The frozen deep-merge/layer-reconstruct machinery is DELETED (explicit-complete configs have no
layers to reconstruct). What survives is per-knob ``(value, source)`` tagging + ``to_event_payload``
for the resolved_config event (docs/contracts/event_manifest.md). No inputs_seen, no
precedence_family, no layer chain; "checkpoint" source is not producible in WP8 (no loader).

The payload carries the 7 schema leaf keys (source="file") plus the derived ``amp_dtype``
(source="derived") = 8 knobs. The 7-key schema portion is identical to O15's CONSUMER_REGISTRY's
original (pre-WP11-A/pre-WPSC) 8-key set, minus ``selfplay.legal_move_radius_schedule``
(WPSC Phase 2 SC-A2: the field no longer exists — the encoding registry alone is the radius
authority, DESIGN_P2.md §5/§9 — no replacement leaf is added, per the same precedent WP11-A
set of not threading every new schema leaf into this payload).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mantis.config.resolve.amp import resolve_amp_dtype
from mantis.config.resolve.encoding import reconcile_encoding
from mantis.config.schema import RunConfig

# Resolver-vocab source → thin WP8 emit vocab. A declared config value is a "file" source; the
# encoding resolver reports "variant"/"checkpoint" — remap so provenance speaks the emit vocab.
_SOURCE_REMAP = {"variant": "file", "default": "file", "cli": "cli", "checkpoint": "checkpoint"}


@dataclass(frozen=True)
class ResolvedKnob:
    """One resolved knob: its value + which source it came from ("file" | "cli" | "derived")."""

    value: Any
    source: str


@dataclass(frozen=True)
class ResolvedConfig:
    """Frozen bundle of resolved knobs (thin source-tagging; no merge provenance)."""

    _knobs: Mapping[str, ResolvedKnob]

    def provenance(self, knob: str) -> ResolvedKnob:
        try:
            return self._knobs[knob]
        except KeyError:
            raise KeyError(
                f"no resolved knob {knob!r}; resolved: {sorted(self._knobs)}"
            ) from None

    def to_event_payload(self) -> dict:
        """Render the resolved_config event payload: {event, knobs:{k:{value, source}}}."""
        return {
            "event": "resolved_config",
            "knobs": {
                knob: {"value": kb.value, "source": kb.source}
                for knob, kb in self._knobs.items()
            },
        }


def resolve_config(cfg: RunConfig) -> ResolvedConfig:
    """Build a ResolvedConfig from a validated RunConfig.

    The 8 schema leaves tag as "file" (declared config values); ``amp_dtype`` is the one derived
    knob (source="derived", from resolve_amp_dtype(representation)). The encoding routes through
    reconcile_encoding (declared, no stamp → source "variant") and remaps variant→"file" (NIT-2).
    """
    enc = reconcile_encoding(cfg.identity.encoding, None)
    knobs: dict[str, ResolvedKnob] = {
        "schema_version": ResolvedKnob(cfg.schema_version, "file"),
        "run_id": ResolvedKnob(cfg.run_id, "file"),
        "seed": ResolvedKnob(cfg.seed, "file"),
        "identity.encoding": ResolvedKnob(enc.name, _SOURCE_REMAP[enc.source]),
        "identity.representation": ResolvedKnob(cfg.identity.representation, "file"),
        "eval.random_model_sims": ResolvedKnob(cfg.eval.random_model_sims, "file"),
        "eval.sealbot_model_sims": ResolvedKnob(cfg.eval.sealbot_model_sims, "file"),
        "amp_dtype": ResolvedKnob(
            resolve_amp_dtype(cfg.identity.representation, cfg.train.amp_dtype), "derived"
        ),
    }
    return ResolvedConfig(knobs)
