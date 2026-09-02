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
        # A multi-window encoding's dense kernel is `unimplemented!` — a genuine die-loud
        # site, and the one that remains after AUDIT-1 F-38.
        #
        # THE CAPTURE SITE MOVED, and the reason is the point of the suite it feeds. This was
        # `ReplayBuffer(4, "<unknown encoding>")`, which resolved through `lookup_or_panic`.
        # F-38 gave that constructor a named `ValueError` carrying the sorted registered set,
        # because every sibling — `HexgBuffer.__new__`, `SelfPlayRunner.__new__`,
        # `RegistrySpec.from_registry` — already did, and a `PanicException` for a mistyped
        # config value is not a design. So the site stopped panicking, which is progress, and
        # this fixture needs one that still does.
        #
        # What the suite is FOR is unchanged: `panic = "unwind"` (R2/LAW-13) means a panic
        # crosses the FFI as a catchable exception rather than aborting the process, and an
        # abort loses the run. That property needs a live witness for as long as any panic can
        # be reached — the goal is a tree with no reachable panics, and until then this proves
        # the profile setting still holds.
        _engine.Board.with_encoding_name("v6w25").to_tensor()
    except BaseException as exc:  # noqa: BLE001 — capturing the panic class on purpose
        cls = type(exc)
        assert cls.__module__ == "pyo3_runtime", cls.__module__
        assert cls.__name__ == "PanicException", cls.__name__
        return cls
    raise AssertionError("the multi-window dense kernel did not panic")
