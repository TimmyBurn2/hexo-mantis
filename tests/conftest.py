"""Root conftest: single-seed determinism (repo_design §8).

One seed for the whole session, printed in the header; every test starts from an
identical RNG state via an autouse reseed of random (+ numpy/torch when installed —
neither is a scaffold dependency; the reseed arms itself when they arrive).
"""
import importlib.util
import os
import random

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
