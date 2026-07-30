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


@pytest.fixture(autouse=True)
def _restore_signal_dispositions():
    """Save/restore SIGINT + SIGTERM around EVERY test (WPMAIN, DESIGN §3 additive).

    `mantis.run.compose_run` installs LAW-16's handlers unconditionally (leg 1: the install
    used to fire only on `run_training_loop`'s self-construct branch, which the composition
    root never takes — a live mechanism wired to a branch nothing runs). So every in-process
    `compose_run` drive now mutates PROCESS-GLOBAL handler state, and without this fixture
    one drive's handlers would decide a later test's fate.

    Three things stated rather than left implicit:
      * COST: it runs for the whole collection — two `signal.getsignal` + two `signal.signal`
        per test. Negligible, and disclosed rather than hidden.
      * ORDERING vs `_reseed` above: immaterial in substance. This fixture touches no RNG and
        `_reseed` touches no signal state, so the two are independent; it is declared after
        `_reseed` and that is the whole of the relationship.
      * NESTING: semantics are restore-AROUND. The existing inner save/restores
        (`tests/train/test_lifecycle_contract.py`'s `restore_signals`,
        `tests/train/test_launch_path_smoke.py`'s try/finally) nest cleanly inside it — each
        inner restore returns to what IT saved, then this one restores the pre-test
        disposition. Idempotent.
    """
    import signal

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


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


@pytest.fixture
def mk_graph_buffer():
    """Factory: a REAL `HexgBuffer` preloaded through the real graph push path — the
    smallest buffer the DECLARED graph training route serves (WPTS/TD-1, R102). Composition
    drives use it wherever a minted graph config's straight arm executes: the typed route
    refuses a shapeless fake at dispatch (`RepresentationRouteError`), by design."""
    from mantis._engine import HexgBuffer

    def make(n_records: int = 8, capacity: int = 64, encoding: str = "gnn_axis_v1"):
        hb = HexgBuffer(capacity, encoding)
        for i in range(n_records):
            stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
            hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i,
                                   True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i)
        return hb

    return make
