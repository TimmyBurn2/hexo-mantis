"""WPSC Phase 3 SC-B2 — dense fallbacks -> hard raises (R28, LAW-11; DESIGN_P3.md §3).

No dedicated resolvers-only test file exists at HEAD (confirmed by the design's own
grep census, re-checked here) — this is a genuinely new file, not a deviation from an
edit-shaped PREREG entry.

RED at HEAD (`507c23b`): `MissingEncodingError` does not exist yet; `normalize_encoding_
name(None)`/`resolve_from_config(None)`/`resolve_from_config({})`/`resolve_from_config(
{"encoding": {}})` all currently silently resolve to the v6 default instead of raising.
The three positive controls (explicit encoding, unaffected) already pass today.
"""
from __future__ import annotations

import pytest

from mantis.encoding.resolvers import (
    MissingEncodingError,
    normalize_encoding_name,
    resolve_from_config,
)


def test_normalize_encoding_name_none_raises() -> None:
    with pytest.raises(MissingEncodingError):
        normalize_encoding_name(None)


def test_resolve_from_config_none_raises() -> None:
    with pytest.raises(MissingEncodingError):
        resolve_from_config(None)


def test_resolve_from_config_no_encoding_key_raises() -> None:
    with pytest.raises(MissingEncodingError):
        resolve_from_config({})


def test_resolve_from_config_mapping_without_version_raises() -> None:
    with pytest.raises(MissingEncodingError):
        resolve_from_config({"encoding": {}})


def test_normalize_encoding_name_explicit_v6_unaffected() -> None:
    assert normalize_encoding_name("v6") == "v6"


def test_resolve_from_config_explicit_string_form_unaffected() -> None:
    spec = resolve_from_config({"encoding": "v6"})
    assert spec.name == "v6"


def test_resolve_from_config_explicit_mapping_form_unaffected() -> None:
    spec = resolve_from_config({"encoding": {"version": "v6"}})
    assert spec.name == "v6"
