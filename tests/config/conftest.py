"""Config test fixtures (repo_design §8): production_config() loads the live run5.

The regime-parity oracles (O9–O12) assert *suite default == production default* by
deriving their expectations from this one fixture — no hardcoded regime knob divergent
from the shipped config (CONTEXT bug-class #5, fixture blindness).
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def production_config(smoke_run_config):
    """The live production run config (configs/run5.yaml), schema-validated.

    WPAX Phase S §2.2: re-expressed as a delegation to the root conftest's factory, so ONE
    loader call, ONE merge rule and ONE `model_validate` sit behind both fixture names
    (LAW-03). The NAME is unchanged — its users answer "the production config", while the
    factory answers "a config derived from any of the minted ones". The delegation goes
    through pytest's own fixture mechanism rather than `from conftest import ...`, because
    the bare module name `conftest` resolves to THIS file, not the root one (R5 bars the
    `sys.path` write that would fix that).
    """
    return smoke_run_config("run5.yaml")
