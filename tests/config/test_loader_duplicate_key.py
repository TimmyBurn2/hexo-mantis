"""O13 — duplicate-key rejection (util.yaml_io.UniqueKeyLoader).

AUDIT-1 F-45 moved the loader to `mantis.util.yaml_io` (a DAG leaf) so the encoding audit
can read through THE parser instead of a strictly more permissive `yaml.safe_load`.

Frozen utils/config.py merely LOGGED a warning on overlap (last-wins). The new loader
HARD-ERRORS on any duplicate key, at any nesting depth. Safe construction preserved
(SafeLoader subclass).
"""
import textwrap
from pathlib import Path

import pytest
import yaml

from mantis.config.loader import DuplicateKeyError, load_config
from mantis.util.yaml_io import UniqueKeyLoader as _UniqueKeyLoader

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_top_level_duplicate_key_raises_naming_key():
    with pytest.raises((DuplicateKeyError, ValueError), match="seed"):
        yaml.load("seed: 1\nseed: 2\n", Loader=_UniqueKeyLoader)


def test_nested_duplicate_key_raises_naming_key():
    doc = textwrap.dedent(
        """
        identity:
          encoding: a
          encoding: b
        """
    )
    with pytest.raises((DuplicateKeyError, ValueError), match="encoding"):
        yaml.load(doc, Loader=_UniqueKeyLoader)


def test_clean_minted_config_loads_and_validates():
    cfg = load_config(REPO_ROOT / "configs" / "dev_example.yaml")
    assert cfg.run_id == "dev_example"


def test_mutation_self_test_base_safeloader_is_permissive():
    # Proves _UniqueKeyLoader's check is what bites: the base SafeLoader silently last-wins.
    assert yaml.safe_load("seed: 1\nseed: 2\n") == {"seed": 2}


def test_duplicate_key_error_is_valueerror():
    assert issubclass(DuplicateKeyError, ValueError)
