"""`resolve_from_config` is THE one authority for WHERE a config declares its encoding.

It reads the nested `identity.encoding` that `RunConfig.model_dump()` carries as well as the
flat key; no caller re-implements the lift. Reading the nested shape is NOT a fallback and NOT
a default — a config declaring an encoding in NO shape still raises, and those arms are
re-asserted here because widening an accept-set is how a no-fallback posture gets lost.
"""
from __future__ import annotations

import pytest

from mantis.encoding.resolvers import MissingEncodingError, resolve_from_config

def test_nested_identity_encoding_resolves() -> None:
    """The WP8 nested shape — what `RunConfig` actually dumps — resolves."""
    spec = resolve_from_config({"identity": {"encoding": "gnn_axis_v1", "representation": "graph"}})
    assert spec.name == "gnn_axis_v1"


def test_nested_identity_resolves_the_graph_encoding_run5_declares() -> None:
    spec = resolve_from_config(
        {"identity": {"encoding": "gnn_axis_v1", "representation": "graph"}}
    )
    assert spec.name == "gnn_axis_v1"
    assert spec.representation == "graph"


def test_disagreeing_dual_shape_raises_not_a_precedence_pick() -> None:
    """A dual-shape config whose declarations DISAGREE is corrupt input: the one authority
    RAISES rather than picking a winner. `extra="forbid"` means no real `RunConfig` carries
    both, so this pins the intent, not a live path."""
    from mantis.encoding.resolvers import EncodingDeclarationConflictError

    with pytest.raises(EncodingDeclarationConflictError):
        resolve_from_config({"encoding": "gnn_axis_v1", "identity": {"encoding": "gnn_axis_r8"}})


# the accept-set widened; the no-default posture did not


def test_neither_shape_present_still_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({})


def test_identity_without_encoding_key_raises() -> None:
    """An `identity` block that omits `encoding` is an absent declaration, not a v6 config."""
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": {"representation": "graph"}})


def test_identity_not_a_mapping_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": "gnn_axis_v1"})


def test_identity_encoding_none_raises() -> None:
    with pytest.raises(MissingEncodingError, match="declares no encoding"):
        resolve_from_config({"identity": {"encoding": None}})


def test_error_message_names_both_shapes() -> None:
    """The diagnostic must tell the operator every place it looked."""
    with pytest.raises(MissingEncodingError) as exc:
        resolve_from_config({})
    msg = str(exc.value)
    assert "identity.encoding" in msg
    assert "encoding: <name>" in msg
