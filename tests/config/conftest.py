"""Config test fixtures (repo_design §8): production_config() loads the live run5.

The regime-parity oracles (O9–O12) assert *suite default == production default* by
deriving their expectations from this one fixture — no hardcoded regime knob divergent
from the shipped config (CONTEXT bug-class #5, fixture blindness).
"""
from pathlib import Path

import pytest

from mantis.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def production_config():
    """The live production run config (configs/run5.yaml), schema-validated."""
    return load_config(REPO_ROOT / "configs" / "run5.yaml")
