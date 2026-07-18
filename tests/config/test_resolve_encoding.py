"""O1 — encoding reconcile decision-equivalence (resolve/encoding.reconcile_encoding).

Vectors captured from frozen hexo_rl/config/resolve/encoding.py::reconcile_declared_vs_stamp.
Two branches invert vs frozen (marked Δ-REBUILD): absent+no-stamp RAISES (LAW-11, no "v6").
"""
import pytest

from mantis.config.resolve.encoding import (
    UNSPECIFIED,
    AbsentEncodingError,
    EncodingConflictError,
    EncodingResolution,
    normalize_declared,
    normalize_stamp,
    reconcile_encoding,
)


def test_declared_agrees_with_stamp_variant_wins():
    res = reconcile_encoding("v6_live2_ls", "v6_live2_ls")
    assert isinstance(res, EncodingResolution)
    assert res.name == "v6_live2_ls"
    assert res.source == "variant"


def test_declared_conflicts_with_stamp_raises_naming_both():
    with pytest.raises(EncodingConflictError) as exc:
        reconcile_encoding("v6_live2_ls", "v6_live2")
    msg = str(exc.value)
    assert "v6_live2_ls" in msg and "v6_live2" in msg
    assert exc.value.declared == "v6_live2_ls"
    assert exc.value.stamp == "v6_live2"


def test_declared_present_no_stamp_variant_wins():
    res = reconcile_encoding("v6w25", None)
    assert res.name == "v6w25"
    assert res.source == "variant"


def test_absent_declared_stamp_present_checkpoint_wins():
    res = reconcile_encoding(UNSPECIFIED, "v6_live2_ls")
    assert res.name == "v6_live2_ls"
    assert res.source == "checkpoint"


def test_absent_declared_no_stamp_raises_delta_rebuild():
    # Δ-REBUILD: frozen returned ("v6", "default"); the terminal "v6" default is BANNED (LAW-11).
    with pytest.raises(AbsentEncodingError):
        reconcile_encoding(UNSPECIFIED, None)


def test_conflict_error_is_valueerror():
    assert issubclass(EncodingConflictError, ValueError)
    assert issubclass(AbsentEncodingError, ValueError)


def test_normalize_declared_presence_before_normalize():
    # absent key → UNSPECIFIED (NOT normalize(None)=="v6")
    assert normalize_declared(False, None) is UNSPECIFIED
    assert normalize_declared(True, "gnn_axis_v1") == "gnn_axis_v1"


def test_normalize_stamp():
    assert normalize_stamp({}) is None
    assert normalize_stamp({"encoding": "v6"}) == "v6"
