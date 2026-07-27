"""Root conftest: single-seed determinism (repo_design §8).

One seed for the whole session, printed in the header; every test starts from an
identical RNG state via an autouse reseed of random (+ numpy/torch when installed —
neither is a scaffold dependency; the reseed arms itself when they arrive).
"""
import importlib.util
import os
import random
from pathlib import Path

import pytest

PYTEST_SEED = int(os.environ.get("PYTEST_SEED", "20260716"))
_HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
_HAVE_TORCH = importlib.util.find_spec("torch") is not None
_SEEDED_LIBS = ["random"] + (["numpy"] if _HAVE_NUMPY else []) + (["torch"] if _HAVE_TORCH else [])


def pytest_report_header(config):
    return f"PYTEST_SEED={PYTEST_SEED} (autouse reseed per test: {', '.join(_SEEDED_LIBS)})"


@pytest.fixture(autouse=True)
def _reseed():
    random.seed(PYTEST_SEED)
    if _HAVE_NUMPY:
        import numpy  # pyright: ignore[reportMissingImports] — guarded: arms when installed

        numpy.random.seed(PYTEST_SEED % (2**32))
    if _HAVE_TORCH:
        import torch  # pyright: ignore[reportMissingImports] — guarded: arms when installed

        torch.manual_seed(PYTEST_SEED)
    yield


@pytest.fixture
def seeded_libs() -> list[str]:
    return list(_SEEDED_LIBS)


# ── WPAX Phase S §2.2 option C — REAL RunConfigs for composition tests ───────────────────
# A composition test needs a schema-validated `RunConfig`, and the suite's answer so far was
# `SimpleNamespace()` (the F-A defect: a production axis pinned to one test-only value) or a
# fifteenth hand-written payload census (LAW-03). This factory takes neither: it derives a
# real `RunConfig` from an already-MINTED config through the ONE loader, so the base values
# are the ones CI gate 7 validates, and per-test deltas go back through
# `RunConfig.model_validate` — a test cannot construct a config the loader would reject.
# Imports are lazy so the root conftest stays scaffold-independent (same reason as _reseed).
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
MINTED_CONFIGS = ("dev_example.yaml", "run5.yaml", "smoke_gnn.yaml",
                  "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")


def _deep_merge(base: dict, over: dict) -> dict:
    """Section-wise merge: `train={"max_train_steps": 4}` overrides ONE key, not the block."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def make_run_config_from_minted(name: str = "smoke_gnn.yaml", **section_overrides):
    """A REAL RunConfig, derived from a MINTED config through the ONE loader.

    `name` is the axis: any of `MINTED_CONFIGS`. Overrides are per-section dicts and are
    re-validated, so every cross-field validator runs on the result.
    """
    from mantis.config.loader import load_config
    from mantis.config.schema import RunConfig

    base = load_config(CONFIGS_DIR / name).model_dump()
    return RunConfig.model_validate(_deep_merge(base, section_overrides))


@pytest.fixture
def smoke_run_config():
    """The factory itself, so a test can vary the config NAME as well as its deltas."""
    return make_run_config_from_minted
