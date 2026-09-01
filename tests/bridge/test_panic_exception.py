"""PanicException catchability (O11, LOCKED #4): a die-loud Rust panic crosses
the FFI as a CATCHABLE `pyo3_runtime.PanicException` — `panic = "unwind"`'s whole
point — NOT a process abort. RED-TEAM: the process must survive.
"""
from mantis import _engine


def test_unknown_encoding_lookup_panics_catchably(panic_exception):
    """`ReplayBuffer("bogus")` -> lookup_or_panic -> catchable PanicException."""
    try:
        _engine.ReplayBuffer(8, "__no_such_encoding__")
    except panic_exception as exc:
        assert "unknown encoding" in str(exc)
    else:
        raise AssertionError("expected a PanicException, none raised")


def test_multi_window_to_tensor_panics_catchably(panic_exception):
    """Board.to_tensor on a multi-window encoding (v6w25) hits the `unimplemented!`
    kernel — crosses as a catchable PanicException (route via get_cluster_views)."""
    board = _engine.Board.with_encoding_name("v6w25")
    try:
        board.to_tensor()
    except panic_exception as exc:
        assert "multi-window" in str(exc)
    else:
        raise AssertionError("expected a PanicException, none raised")


def test_encoding_less_board_to_tensor_panics_catchably(panic_exception):
    """Board().to_tensor() (no encoding bound) -> catchable PanicException
    (R28; DESIGN_P3.md §3.2/§3.4). Mirrors test_multi_window_to_tensor_
    panics_catchably's exact idiom."""
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


def test_process_survives_a_caught_panic(panic_exception):
    """After catching a panic the interpreter is still live and usable —
    proves unwind, not abort."""
    # DERIVED, not typed: the claim is that the registry SURVIVES the panics, so the
    # comparison is against the same surface read before them. A typed count made this row a
    # second authority over the size of the registered set, and it reds on any registry change
    # while saying nothing about unwinding (R192(e), derive-or-delete).
    before = len(_engine.all_specs())
    assert before > 0, "the registry was already empty; this row cannot show survival"
    for _ in range(3):
        try:
            _engine.ReplayBuffer(8, "__still_bogus__")
        except panic_exception:
            pass
    # The engine is still fully functional after repeated caught panics.
    assert len(_engine.all_specs()) == before
    b = _engine.Board.with_encoding_name("v6")
    b.apply_move(0, 0)
    assert b.ply == 1
