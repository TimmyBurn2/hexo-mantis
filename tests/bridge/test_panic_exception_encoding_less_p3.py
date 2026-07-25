"""WPSC Phase 3 SC-B2 — `Board().to_tensor()` (no encoding bound) panics catchably
(R28; DESIGN_P3.md §3.2/§3.4). Mirrors `test_panic_exception.py::test_multi_window_to_
tensor_panics_catchably`'s exact idiom; reuses the same session-scoped `panic_exception`
fixture from `tests/bridge/conftest.py`.

DEVIATION FROM PREREG PATH (logged in ORACLE_NOTES_P3.md): PREREG names this suite's home
as the existing `tests/bridge/test_panic_exception.py` (an ADD, not a new file).
ORACLE-WRITE's writable surface here is NEW files only — ported to a sibling file with a
`_p3` suffix, matching the ORACLE_NOTES_P2.md house convention for the same class of
deviation. IMPL folds this test's body into the original file at port time.

RED at HEAD (`507c23b`): `Board()` (no encoding) `.to_tensor()` currently silently
resolves the v6 default and returns a tensor instead of panicking — this test currently
FAILS (no exception raised at all), not a collection error.
"""
from __future__ import annotations

from mantis import _engine


def test_encoding_less_board_to_tensor_panics_catchably(panic_exception) -> None:
    board = _engine.Board()
    try:
        board.to_tensor()
    except panic_exception as exc:
        assert "encoding-less" in str(exc)
    else:
        raise AssertionError("expected a PanicException, none raised")

    # Follow-on liveness check, same test (design's preferred shape, §3.4): the
    # interpreter must still be usable after the catch above (unwind, not abort).
    for _ in range(2):
        try:
            _engine.Board().to_tensor()
        except panic_exception:
            pass
    b = _engine.Board.with_encoding_name("v6")
    b.apply_move(0, 0)
    assert b.ply == 1
