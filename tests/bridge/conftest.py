"""Shared fixtures for the WP7 bridge-surface oracles.

`pyo3_runtime.PanicException` is not importable as a module (pyo3 registers it
lazily and never exposes an import path), so we capture the class object by
triggering one die-loud panic across the FFI and reading `type(exc)`. This is
the catchable-panic class every O11 / O13 oracle asserts against.
"""
import threading

import pytest

from mantis import _engine


@pytest.fixture(scope="session")
def panic_exception():
    """The `pyo3_runtime.PanicException` class (captured from a live panic)."""
    # The only live panic left in this tree is pyo3's OWN unsendable-pyclass assertion,
    # since no production path panics any more. `panic = "unwind"` must still cross the
    # FFI as a catchable exception, not an abort — this fixture is that live witness.
    captured: list[type] = []

    def _touch_off_thread() -> None:
        try:
            board.apply_move(1, 0)
        except BaseException as exc:  # noqa: BLE001 — capturing the panic class on purpose
            captured.append(type(exc))

    board = _engine.Board.with_encoding_name("gnn_axis_v1")
    board.apply_move(0, 0)
    thread = threading.Thread(target=_touch_off_thread)
    thread.start()
    thread.join()

    assert captured, "the unsendable pyclass did not panic when touched off-thread"
    cls = captured[0]
    assert cls.__module__ == "pyo3_runtime", cls.__module__
    assert cls.__name__ == "PanicException", cls.__name__
    return cls
