"""WPBRIDGE Phase T — TD-4 / CARD-POOL-ENCODING-BRIDGE: `resolve_from_config` is THE one
authority for WHERE a config declares its encoding (R1 one-authority, LAW-11, LAW-08).

RED AT HEAD (`ca237d2`): `resolve_from_config` read only the flat top-level `encoding` key.
`RunConfig.model_dump()` carries `identity.encoding` and no flat key, so every caller that
did not privately re-implement the lift died on `MissingEncodingError`. Two callers DID
re-implement it (`train/trainer/core.py::_resolve_spec`, `train/orchestrator.py`) and five
did not — `selfplay/hparams.py::resolve_pool_encoding` (the pool), `selfplay/worker.py`,
`selfplay/inference_server.py`, `train/subsystems.py`, `encoding/audit_sections.py`. That
split is the defect: duplicated authority for a schema fact, which is what R1 forbids.

The census-reproducing repro, at HEAD:

    resolve_pool_encoding(load_config("configs/run5.yaml").model_dump(), arch=None)
    -> MissingEncodingError: config has no 'encoding' key ... (LAW-11, R28)

and it is why mode PREFLIGHT could not run a burst at all (parent rc 33, child rc 1).

The fix adds the nested shape to the ONE resolver and deletes both private bridges. Reading
`identity.encoding` is NOT a fallback and NOT a default: `IdentityConfig.encoding` is a
required, defaultless, registry-cross-checked field (`config/schema/core.py:50`), and a
config declaring an encoding in NO shape still raises. The LAW-11 arms are re-asserted here
rather than assumed, because widening an accept-set is exactly how a no-fallback posture
gets lost by accident.
"""
from __future__ import annotations

import pytest

from mantis.encoding.resolvers import MissingEncodingError, resolve_from_config

# ── the shape TD-4 was about ────────────────────────────────────────────────────────────


def test_nested_identity_encoding_resolves() -> None:
    """The WP8 nested shape — what `RunConfig` actually dumps — resolves."""
    spec = resolve_from_config({"identity": {"encoding": "v6", "representation": "grid"}})
    assert spec.name == "v6"


def test_nested_identity_resolves_the_graph_encoding_run5_declares() -> None:
    spec = resolve_from_config(
        {"identity": {"encoding": "gnn_axis_v1", "representation": "graph"}}
    )
    assert spec.name == "gnn_axis_v1"
    assert spec.representation == "graph"


def test_flat_shape_takes_precedence_over_nested() -> None:
    """Precedence is FLAT-first, preserving the deleted bridges' own order
    (`"encoding" not in cfg` guarded their nested arm). Under `extra="forbid"` a real
    `RunConfig` can never carry both, so this pins intent, not a live path."""
    spec = resolve_from_config(
        {"encoding": "v6", "identity": {"encoding": "gnn_axis_v1"}}
    )
    assert spec.name == "v6"


# ── LAW-11: the accept-set widened, the no-default posture did not ──────────────────────


def test_neither_shape_present_still_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({})


def test_identity_without_encoding_key_raises() -> None:
    """An `identity` block that omits `encoding` is an absent declaration, not a v6 config."""
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": {"representation": "grid"}})


def test_identity_not_a_mapping_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": "gnn_axis_v1"})


def test_identity_encoding_none_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": {"encoding": None}})


def test_error_message_names_both_shapes() -> None:
    """Name-truth (R73): the diagnostic must tell the operator every place it looked."""
    with pytest.raises(MissingEncodingError) as exc:
        resolve_from_config({})
    msg = str(exc.value)
    assert "identity.encoding" in msg
    assert "encoding: <name>" in msg
