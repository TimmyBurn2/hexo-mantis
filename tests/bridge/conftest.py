"""Shared fixtures for the WP7 bridge-surface oracles.

`pyo3_runtime.PanicException` is not importable as a module (pyo3 registers it
lazily and never exposes an import path), so we capture the class object by
triggering one die-loud panic across the FFI and reading `type(exc)`. This is
the catchable-panic class every O11 / O13 oracle asserts against.
"""
import pytest

from mantis import _engine


@pytest.fixture(scope="session")
def panic_exception():
    """The `pyo3_runtime.PanicException` class (captured from a live panic)."""
    try:
        # Unknown-encoding lookup_or_panic — the guaranteed die-loud site.
        _engine.ReplayBuffer(4, "__bogus_encoding_for_panic_capture__")
    except BaseException as exc:  # noqa: BLE001 — capturing the panic class on purpose
        cls = type(exc)
        assert cls.__module__ == "pyo3_runtime", cls.__module__
        assert cls.__name__ == "PanicException", cls.__name__
        return cls
    raise AssertionError("ReplayBuffer with an unknown encoding did not panic")
