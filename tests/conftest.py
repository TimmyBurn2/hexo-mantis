"""Root conftest: one seed for the whole session, printed in the header.

Every test starts from an identical RNG state via an autouse reseed of random, plus
numpy/torch when installed — neither is a scaffold dependency, so the reseed self-arms.
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
    # The tier is DERIVED from the live marker expression, so the tier a run executed is read
    # off its own output rather than assumed from the command typed.
    markexpr = config.getoption("markexpr") or ""
    tier = f"-m {markexpr!r}" if markexpr else "NONE (whole tree, every marker)"
    return [
        f"PYTEST_SEED={PYTEST_SEED} (autouse reseed per test: {', '.join(_SEEDED_LIBS)})",
        f"TIER: {tier}",
    ]


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


@pytest.fixture(autouse=True)
def _restore_signal_dispositions():
    """Save and restore SIGINT + SIGTERM around every test.

    `compose_run` installs its signal handlers unconditionally, so every in-process drive
    mutates process-global handler state. Semantics are restore-AROUND, so inner save/restores
    nest cleanly inside this one.
    """
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture
def seeded_libs() -> list[str]:
    return list(_SEEDED_LIBS)


# Real RunConfigs for composition tests, derived from an already-minted config through the ONE
# loader, with per-test deltas re-validated — so a test cannot construct a config the loader
# would reject. Imports are lazy to keep this conftest scaffold-independent.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
MINTED_CONFIGS = ("dev_example.yaml", "run6.yaml", "smoke_preflight_armed.yaml")


def _deep_merge(base: dict, over: dict) -> dict:
    """Merge section-wise: `train={"max_train_steps": 4}` overrides one key, not the block."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def make_run_config_from_minted(name: str = "dev_example.yaml", **section_overrides):
    """Build a real RunConfig from a minted config through the one loader.

    `name` is any of `MINTED_CONFIGS`; overrides are per-section dicts and are re-validated, so
    every cross-field validator runs on the result.
    """
    from mantis.config.loader import load_config
    from mantis.config.schema import RunConfig

    base = load_config(CONFIGS_DIR / name).model_dump()
    return RunConfig.model_validate(_deep_merge(base, section_overrides))


@pytest.fixture
def smoke_run_config():
    """Return the factory itself, so a test can vary the config name as well as its deltas."""
    return make_run_config_from_minted


@pytest.fixture
def mk_graph_buffer():
    """Return a factory for a real `HexgBuffer` preloaded through the graph push path.

    The typed route refuses a shapeless fake at dispatch, so composition drives need this
    wherever a minted graph config's straight arm executes.
    """
    from mantis._engine import HexgBuffer

    def make(n_records: int = 8, capacity: int = 64, encoding: str = "gnn_axis_v1"):
        hb = HexgBuffer(capacity, encoding, 128)
        for i in range(n_records):
            stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
            hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i,
                                   True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i)
        return hb

    return make
