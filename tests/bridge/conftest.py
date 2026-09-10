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
    # THE CAPTURE SITE HAS MOVED TWICE, and each move is the point of the suite it feeds.
    # It was `ReplayBuffer(4, "<unknown encoding>")` until AUDIT-1 F-38 gave that constructor a
    # named `ValueError`; then the multi-window dense kernel's `unimplemented!`, until R346(f)
    # deleted the dense kernels with the grid path. What is left is pyo3's OWN unsendable
    # assertion — a `PyBoard` touched from a second thread — which is not a mantis panic at all
    # and is exactly why it is the right site now: no production path in this tree reaches for
    # a panic any more (CLAUDE.md's Rust rule), so the only live witness to the PROFILE is one
    # the framework raises.
    #
    # What the suite is FOR is unchanged: `panic = "unwind"` (R2/LAW-13) means a panic crosses
    # the FFI as a catchable exception rather than aborting the process, and an abort loses the
    # run. That needs a live witness for as long as ANY panic can be reached.
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
