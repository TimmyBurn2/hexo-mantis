"""WPTS Phase P oracles — agreement-or-raise on the ONE encoding resolver (ADJ-25 / R104).

A config declaring an encoding in two shapes that DISAGREE is corrupt input, and
`resolve_from_config` RAISES the named `EncodingDeclarationConflictError` rather than
silently picking a winner (the operator rejected "precedence" as the question). The two
remaining private name-lifters (`train/anchor.py::_resolve_declared_encoding`,
`train/pretrain/validate.py::_config_encoding`) converge onto the resolver, finishing the
one-authority collapse WPBRIDGE started — so the conflict raise holds through every former
call site, and LAW-11's raise-arms are re-pinned at each (the TD-4 lesson: converging an
accept-set re-proves no-fallback).
"""
from __future__ import annotations

import pytest

from mantis.encoding.resolvers import (
    EncodingDeclarationConflictError,
    EncodingRegistryError,
    MissingEncodingError,
    resolve_from_config,
)
from mantis.train.anchor import _resolve_declared_encoding
from mantis.train.pretrain.validate import _config_encoding


# ── the conflict raise, at the resolver ──────────────────────────────────────────────────
def test_disagreeing_flat_string_and_nested_identity_raise_the_named_error() -> None:
    with pytest.raises(EncodingDeclarationConflictError) as exc:
        resolve_from_config({"encoding": "v6", "identity": {"encoding": "gnn_axis_v1"}})
    msg = str(exc.value)
    assert "v6" in msg and "gnn_axis_v1" in msg, (
        "the error must NAME both declarations — corrupt input is diagnosed, not hidden"
    )


def test_disagreeing_flat_mapping_and_nested_identity_raise_the_named_error() -> None:
    with pytest.raises(EncodingDeclarationConflictError):
        resolve_from_config({"encoding": {"version": "v6w25"},
                             "identity": {"encoding": "v6_live2_ls"}})


def test_the_conflict_error_is_not_a_missing_encoding() -> None:
    """A corrupt declaration must never be classified as an ABSENT one: a caller that maps
    `MissingEncodingError → None` (the anchor's legal absence affordance) must not be able
    to swallow a conflict."""
    assert not issubclass(EncodingDeclarationConflictError, MissingEncodingError)
    assert issubclass(EncodingDeclarationConflictError, EncodingRegistryError)


# ── agreement and single shapes still resolve (parity) ───────────────────────────────────
@pytest.mark.parametrize("cfg", [
    {"encoding": "v6w25", "identity": {"encoding": "v6w25"}},
    {"encoding": {"version": "v6w25"}, "identity": {"encoding": "v6w25"}},
])
def test_agreeing_dual_shape_resolves_like_single_shape(cfg) -> None:
    assert resolve_from_config(cfg).name == resolve_from_config({"encoding": "v6w25"}).name


def test_single_shapes_resolve_unchanged() -> None:
    assert resolve_from_config({"encoding": "v6"}).name == "v6"
    assert resolve_from_config({"identity": {"encoding": "gnn_axis_v1"}}).name == "gnn_axis_v1"
    assert resolve_from_config({"encoding": {"version": "v6_live2_ls"}}).name == "v6_live2_ls"


# ── LAW-11 raise-arms re-pinned at the resolver ──────────────────────────────────────────
@pytest.mark.parametrize("cfg", [None, {}, {"identity": {}}, {"identity": "v6"},
                                 {"identity": {"encoding": None}}])
def test_absent_declarations_still_raise_missing(cfg) -> None:
    with pytest.raises(MissingEncodingError):
        resolve_from_config(cfg)


def test_mapping_without_version_still_raises_missing() -> None:
    with pytest.raises(MissingEncodingError):
        resolve_from_config({"encoding": {}})


# ── the former call sites: ONE family member, same raise ─────────────────────────────────
def test_anchor_lifter_is_a_veneer_absence_is_none_conflict_raises() -> None:
    """The anchor's absence affordance survives (a WP10-only launch declares nothing), but a
    CONFLICT is corrupt input and must not degrade into 'no declaration'."""
    assert _resolve_declared_encoding({"identity": {"encoding": "v6w25"}}) == "v6w25"
    assert _resolve_declared_encoding({"encoding": "v6"}) == "v6"
    assert _resolve_declared_encoding({"encoding": {"version": "v6_live2_ls"}}) == "v6_live2_ls"
    assert _resolve_declared_encoding({}) is None
    assert _resolve_declared_encoding({"batch_size": 8}) is None
    assert _resolve_declared_encoding("not-a-dict") is None
    with pytest.raises(EncodingDeclarationConflictError):
        _resolve_declared_encoding({"encoding": "v6", "identity": {"encoding": "v6w25"}})


def test_anchor_private_name_form_is_dead() -> None:
    """`{'encoding': {'name': ...}}` was an anchor-private invention no other reader ever
    accepted; converging onto the one resolver kills it. Absence-shaped → None (deliberate
    narrowing, pinned)."""
    assert _resolve_declared_encoding({"encoding": {"name": "v6w25"}}) is None


def test_anchor_unregistered_name_now_raises_from_the_registry() -> None:
    """The old private read returned ANY string; the one resolver cross-checks the registry
    (LAW-11 posture). A narrowing, pinned deliberately."""
    with pytest.raises(EncodingRegistryError):
        _resolve_declared_encoding({"identity": {"encoding": "not_a_registered_encoding"}})


def test_validate_lifter_conflict_raises_and_absence_still_raises() -> None:
    with pytest.raises(EncodingDeclarationConflictError):
        _config_encoding({"encoding": "v6", "identity": {"encoding": "v6w25"}})
    with pytest.raises(MissingEncodingError):
        _config_encoding({"board_size": 19})


def test_validate_lifter_parity_on_legit_shapes() -> None:
    assert _config_encoding({"identity": {"encoding": "v6w25"}}) == "v6w25"
    assert _config_encoding({"encoding": "gnn_axis_v1"}) == "gnn_axis_v1"
    assert _config_encoding({"encoding": {"version": "v6_live2_ls"}}) == "v6_live2_ls"
