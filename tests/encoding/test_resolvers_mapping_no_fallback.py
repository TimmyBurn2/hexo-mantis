"""WPSC Phase 3 REVIEW-impl MUST-FIX pin (R28/LAW-11): the FIFTH silent-v6 arm.

`normalize_encoding_name`'s Mapping branch carried `enc.get("name",
enc.get("version", "v6"))` — a never-enumerated fallback the SC-B2 caller census
and the frozen `test_resolvers_no_fallback.py` oracle both missed (found by
Phase 3 REVIEW-impl; reachable via `train/batch_assembly.py`). A mapping with
neither key now raises `MissingEncodingError`; keyed mappings still resolve.
"""
from __future__ import annotations

import pytest

from mantis.encoding import normalize_encoding_name
from mantis.encoding.resolvers import MissingEncodingError


def test_mapping_without_name_or_version_raises_missing_encoding_error():
    with pytest.raises(MissingEncodingError, match="neither 'name' nor 'version'"):
        normalize_encoding_name({"planes": 18})


def test_empty_mapping_raises_missing_encoding_error():
    with pytest.raises(MissingEncodingError):
        normalize_encoding_name({})


def test_mapping_with_name_still_resolves():
    assert normalize_encoding_name({"name": "v6"}) == "v6"


def test_mapping_with_version_still_resolves():
    assert normalize_encoding_name({"version": "v6w25"}) == "v6w25"
