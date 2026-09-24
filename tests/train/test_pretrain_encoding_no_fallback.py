"""Pretrain encoding resolution never defaults.

Two arms: a checkpoint config with no encoding, and a CLI invocation passing no
`--encoding`. Both must raise rather than silently pretrain a dense model.

The checkpoint arm is pinned on `resolve_from_config` itself — the one resolver any veneer
has to call.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mantis.encoding.resolvers import (
    EncodingRegistryError,
    MissingEncodingError,
    resolve_from_config,
)
from mantis.train.pretrain.cli import _resolve_encoding_name


def _config_encoding(config: dict) -> str:
    """What a config-encoding veneer does: the ONE resolver, by name."""
    return resolve_from_config(config).name


# the ONE resolver, on the shapes a checkpoint config arrives in


def test_checkpoint_config_with_empty_encoding_mapping_raises():
    with pytest.raises(MissingEncodingError, match="no 'version' key"):
        _config_encoding({"encoding": {}})


def test_checkpoint_config_with_non_string_version_raises():
    # present-but-malformed is `EncodingRegistryError`, which `MissingEncodingError` subclasses
    with pytest.raises(EncodingRegistryError, match="must be a string"):
        _config_encoding({"encoding": {"version": 6}})


def test_checkpoint_config_with_identity_but_non_string_encoding_raises():
    """`identity.encoding` present but not a string must not fall through to a default."""
    with pytest.raises(EncodingRegistryError):
        _config_encoding({"identity": {"encoding": 6}})


# cli._resolve_encoding_name


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(encoding=kw.get("encoding"))


def test_pretrain_cli_without_encoding_raises_the_class_error():
    """The convention is named by ERROR CLASS, so the CLI raises that class, not `SystemExit`."""
    with pytest.raises(MissingEncodingError, match="no encoding specified"):
        _resolve_encoding_name(_args())


def test_pretrain_cli_error_names_the_way_out():
    """The message must tell the operator how to proceed, not just that it failed."""
    with pytest.raises(MissingEncodingError) as exc:
        _resolve_encoding_name(_args())
    msg = str(exc.value)
    assert "--encoding" in msg


def test_pretrain_cli_boundary_converts_the_class_error_to_a_clean_message():
    """`pretrain()` turns the class error into one line at the BOUNDARY, so a forgotten flag
    is not a traceback.

    A real config path is passed rather than a fake one, so this row does not depend on the
    encoding check happening before the config load.
    """
    from mantis.train.pretrain.cli import pretrain

    config = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"
    with pytest.raises(SystemExit) as exc:
        pretrain(["--config", str(config)])
    assert "no encoding specified" in str(exc.value)


def test_pretrain_cli_explicit_encoding_still_wins():
    assert _resolve_encoding_name(_args(encoding="v6w25")) == "v6w25"
