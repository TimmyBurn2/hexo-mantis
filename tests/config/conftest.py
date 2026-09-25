"""Config test fixtures (repo_design §8): production_config() loads the live production config.

The regime-parity oracles (O9–O12) assert *suite default == production default* by
deriving their expectations from this one fixture — no hardcoded regime knob divergent
from the shipped config (CONTEXT bug-class #5, fixture blindness).
"""
from pathlib import Path

import pytest

from mantis.config.census import production_configs

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=production_configs(REPO_ROOT), ids=lambda p: p.name)
def production_config(smoke_run_config, request):
    """Each production config of the census in turn, schema-validated.

    Re-expressed as a delegation to the root conftest's factory, so ONE
    loader call, ONE merge rule and ONE `model_validate` sit behind both fixture names.
    The NAME is unchanged — its users answer "the production config", while the
    factory answers "a config derived from any of the minted ones". The delegation goes
    through pytest's own fixture mechanism rather than `from conftest import ...`, because
    the bare module name `conftest` resolves to THIS file, not the root one, and no
    `sys.path` write may bridge that.
    """
    return smoke_run_config(request.param)
