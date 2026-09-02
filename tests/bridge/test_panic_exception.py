"""PanicException catchability (O11, LOCKED #4): a die-loud Rust panic crosses
the FFI as a CATCHABLE `pyo3_runtime.PanicException` — `panic = "unwind"`'s whole
point — NOT a process abort. RED-TEAM: the process must survive.
"""
import pytest

from mantis import _engine


def test_unknown_encoding_lookup_raises_a_NAMED_error(panic_exception):
    """`ReplayBuffer("bogus")` — the refusal is unchanged, its FACE is not.

    AUDIT-1 F-38. This resolved through `lookup_or_panic` and reached Python as a
    `PanicException`, while every sibling constructor — `HexgBuffer.__new__`,
    `SelfPlayRunner.__new__`, `RegistrySpec.from_registry` — already returned a named
    `ValueError` carrying the sorted registered set. A mistyped encoding name in a config is
    an ordinary operator error and a panic is not how this repo reports one.
    """
    with pytest.raises(ValueError) as excinfo:
        _engine.ReplayBuffer(8, "__no_such_encoding__")
    assert "unknown encoding" in str(excinfo.value)
    assert "v6" in str(excinfo.value), "the registered set must be in the message"
    assert not isinstance(excinfo.value, panic_exception)


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


def test_encoding_less_board_to_tensor_raises_a_NAMED_error(panic_exception):
    """`Board().to_tensor()` (no encoding bound) — R28/LAW-11's refusal, which is unchanged.

    AUDIT-1 F-38 CHANGED ITS FACE, not its verdict. It was `panic!`, reaching Python as a
    `PanicException` and convertible only because the profile sets `panic = "unwind"`
    (R2/LAW-13) — a guarantee about the WORST case, not a design. CLAUDE.md's Rust rule is
    that fail-loud means a NAMED error that propagates, never a panic reached for in the first
    place. The row keeps `panic_exception` as a parameter so it also asserts the NEGATIVE: this
    site no longer produces one.
    """
    board = _engine.Board()
    with pytest.raises(ValueError) as excinfo:
        board.to_tensor()
    assert "encoding-less" in str(excinfo.value)
    assert not isinstance(excinfo.value, panic_exception), (
        "the refusal is a named ValueError now, not a panic crossing the FFI"
    )

    # Follow-on liveness check, same test (design's preferred shape, §3.4): the interpreter
    # must still be usable after the refusal above — and a NAMED error makes that trivially
    # true, which is the improvement. It is kept because the row's subject is the SITE, and a
    # site that refused once must go on refusing.
    for _ in range(2):
        with pytest.raises(ValueError):
            _engine.Board().to_tensor()
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
    # AUDIT-1 F-38 moved this row's DRIVER. `ReplayBuffer` with an unknown encoding is a named
    # `ValueError` now, so the repeated-panic loop is driven by the site that STILL panics:
    # the multi-window dense kernel's `unimplemented!`. The claim is unchanged and is the one
    # `panic = "unwind"` exists for — an ABORT would take the process with it, and no
    # assertion after it would run.
    multi_window = _engine.Board.with_encoding_name("v6w25")
    for _ in range(3):
        try:
            multi_window.to_tensor()
        except panic_exception:
            pass
    # The engine is still fully functional after repeated caught panics.
    assert len(_engine.all_specs()) == before
    b = _engine.Board.with_encoding_name("v6")
    b.apply_move(0, 0)
    assert b.ply == 1
