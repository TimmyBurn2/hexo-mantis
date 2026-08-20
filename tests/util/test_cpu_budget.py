"""Smoke net for the thread-budget helper (net addition — the module had no old test).

The subject relocated from `mantis.util.cpu_budget` to `tests/util/_cpu_budget.py` under
R289(q); it is imported as a tests/ helper (the `_value_health` / `_microbatch_harness`
convention), not from `mantis`.

Pins the stdlib-only thread-budget contract: detection returns a sane int, the
per-lib heuristic matches its documented examples, and `apply_auto_thread_budget`
sets the per-library env vars, is idempotent via `_MANTIS_THREAD_BUDGET_APPLIED`,
honours the `MANTIS_THREAD_BUDGET` override, and respects pre-existing vars via
setdefault. os.environ is isolated per test (a private copy) so the global
process env is never polluted.
"""
from __future__ import annotations

import os

import pytest

import _cpu_budget as cpu_budget
from _cpu_budget import (
    _THREAD_ENV_VARS,
    apply_auto_thread_budget,
    derive_per_lib,
    detect_cpu_budget,
)

_GUARD = "_MANTIS_THREAD_BUDGET_APPLIED"
_FORCE = "MANTIS_THREAD_BUDGET"


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate os.environ into a private copy with the budget keys stripped."""
    stripped = set(_THREAD_ENV_VARS) | {_GUARD, _FORCE}
    env = {k: v for k, v in os.environ.items() if k not in stripped}
    monkeypatch.setattr(cpu_budget.os, "environ", env)
    return env


def test_detect_cpu_budget_positive_int():
    b = detect_cpu_budget()
    assert isinstance(b, int)
    assert b >= 1


@pytest.mark.parametrize(
    "budget,n_workers,expected",
    [
        (16, 14, 3),     # laptop
        (41, 24, 5),     # rented
        (128, 24, 8),    # bare metal, clamped
        (128, None, 8),  # trainer-only, clamped
        (128, 0, 8),     # zero workers → divisor 4, clamped
        (2, None, 1),    # tiny host clamps to 1
    ],
)
def test_derive_per_lib_examples(budget, n_workers, expected):
    assert derive_per_lib(budget, n_workers) == expected


def test_apply_auto_thread_budget_sets_all_vars(clean_env):
    result = apply_auto_thread_budget(n_workers=8, silent=True)
    assert isinstance(result, dict)
    assert result["applied"] is True
    per_lib = result["per_lib"]
    assert per_lib >= 1
    for var in _THREAD_ENV_VARS:
        assert clean_env[var] == str(per_lib), f"{var} not set to per_lib"
        assert result[var] == str(per_lib)
    assert clean_env[_GUARD] == "1"


def test_apply_auto_thread_budget_idempotent(clean_env):
    first = apply_auto_thread_budget(n_workers=4, silent=True)
    assert first["applied"] is True
    second = apply_auto_thread_budget(n_workers=4, silent=True)
    assert second["applied"] is False
    # Values unchanged by the no-op second call.
    for var in _THREAD_ENV_VARS:
        assert second[var] == first[var]


def test_apply_auto_thread_budget_force_override(clean_env):
    clean_env[_FORCE] = "6"
    result = apply_auto_thread_budget(silent=True)
    assert result["applied"] is True
    assert result["per_lib"] == 6
    for var in _THREAD_ENV_VARS:
        assert clean_env[var] == "6"


def test_apply_auto_thread_budget_respects_preexisting(clean_env):
    clean_env["OMP_NUM_THREADS"] = "3"
    result = apply_auto_thread_budget(n_workers=8, silent=True)
    # setdefault must NOT overwrite an already-set var.
    assert clean_env["OMP_NUM_THREADS"] == "3"
    assert result["OMP_NUM_THREADS"] == "3"
