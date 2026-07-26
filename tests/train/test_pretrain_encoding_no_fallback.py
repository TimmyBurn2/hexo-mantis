"""Pretrain encoding resolution never defaults (R45, LAW-11/LAW-05).

Two arms lived here. `validate._config_encoding` silently resolved a checkpoint config
with no encoding to "v6"; `cli._resolve_encoding_name` silently pretrained a v6 model when
neither `--encoding` nor `--resume` was passed. The second was found by running gate 11
before landing it, and is the sixth arm of this class (ADJ-03) — it is on a TRAINING path,
so it is the more consequential of the two.

RED at `973822d`: both functions returned "v6" instead of raising.
"""
from __future__ import annotations

import argparse

import pytest

from mantis.encoding.resolvers import MissingEncodingError
from mantis.train.pretrain.cli import _resolve_encoding_name
from mantis.train.pretrain.validate import _config_encoding

# ── validate._config_encoding ────────────────────────────────────────────────────────


def test_checkpoint_config_with_no_encoding_at_all_raises():
    with pytest.raises(MissingEncodingError, match="carries no encoding"):
        _config_encoding({"board_size": 19})


def test_checkpoint_config_with_empty_encoding_mapping_raises():
    with pytest.raises(MissingEncodingError, match="no string 'version' key"):
        _config_encoding({"encoding": {}})


def test_checkpoint_config_with_non_string_version_raises():
    with pytest.raises(MissingEncodingError, match="no string 'version' key"):
        _config_encoding({"encoding": {"version": 6}})


def test_checkpoint_config_with_identity_but_non_string_encoding_raises():
    """`identity.encoding` present but not a string must not fall through to a default."""
    with pytest.raises(MissingEncodingError):
        _config_encoding({"identity": {"encoding": 6}})


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"identity": {"encoding": "v6w25"}}, "v6w25"),
        ({"encoding": "gnn_axis_v1"}, "gnn_axis_v1"),
        ({"encoding": {"version": "v6_live2_ls"}}, "v6_live2_ls"),
        # identity wins over a conflicting flat key — the WP8 nested form is authoritative
        ({"identity": {"encoding": "v6w25"}, "encoding": "v6"}, "v6w25"),
    ],
)
def test_explicit_encodings_still_resolve(cfg, expected):
    """Positive controls: closing the default arms must not break real resolution."""
    assert _config_encoding(cfg) == expected


# ── cli._resolve_encoding_name (ADJ-03, the sixth arm) ───────────────────────────────


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(encoding=kw.get("encoding"), resume=kw.get("resume"))


def test_pretrain_cli_without_encoding_or_resume_raises_the_class_error():
    """R45 names the convention by ERROR CLASS, so the CLI raises that class.

    REVIEW-impl rejected an earlier `SystemExit` here: the repo already has named errors
    for exactly this class, and an implementation that has to widen the rule's own
    statement of itself in order to comply has not matched the convention.
    """
    with pytest.raises(MissingEncodingError, match="no encoding specified"):
        _resolve_encoding_name(_args())


def test_pretrain_cli_error_names_both_ways_out():
    """The message must tell the operator how to proceed, not just that it failed."""
    with pytest.raises(MissingEncodingError) as exc:
        _resolve_encoding_name(_args())
    msg = str(exc.value)
    assert "--encoding" in msg
    assert "--resume" in msg


def test_pretrain_cli_boundary_converts_the_class_error_to_a_clean_message():
    """`pretrain()` turns it into one line, so a forgotten flag is not a traceback.

    This is what makes raising the class error at the resolver compatible with CLI
    ergonomics — the conversion happens once, at the boundary, not at the seam.
    """
    from mantis.train.pretrain.cli import pretrain

    with pytest.raises(SystemExit) as exc:
        pretrain(["--corpus", "/nonexistent"])
    assert "no encoding specified" in str(exc.value)


def test_pretrain_cli_explicit_encoding_still_wins():
    assert _resolve_encoding_name(_args(encoding="v6w25")) == "v6w25"
