"""PanicException catchability (O11, LOCKED #4): a die-loud Rust panic crosses
the FFI as a CATCHABLE `pyo3_runtime.PanicException` — `panic = "unwind"`'s whole
point — NOT a process abort. RED-TEAM: the process must survive.
"""
import threading

import pytest

from mantis import _engine


#: Every constructor that takes an encoding NAME and answers with the registered set. Each
#: must refuse an unknown one with a named `ValueError` — never a panic.
#: `Board.with_encoding_name` refuses by name too but points at `registry.toml` instead of
#: listing the set, so it is not in this census.
_NAME_TAKING_CONSTRUCTORS = (
    ("HexgBuffer", lambda: _engine.HexgBuffer(8, "__no_such_encoding__", 128)),
    ("RegistrySpec.from_registry", lambda: _engine.RegistrySpec.from_registry("__no_such_encoding__")),
    ("SelfPlayRunner", lambda: _engine.SelfPlayRunner(
        _engine.SelfPlayRunnerConfig(encoding_name="__no_such_encoding__"))),
)


@pytest.mark.parametrize("label,construct", _NAME_TAKING_CONSTRUCTORS,
                         ids=[label for label, _ in _NAME_TAKING_CONSTRUCTORS])
def test_unknown_encoding_lookup_raises_a_NAMED_error(panic_exception, label, construct):
    """The refusal is unchanged, its FACE is not.

    AUDIT-1 F-38. `ReplayBuffer("bogus")` resolved through `lookup_or_panic` and reached
    Python as a `PanicException`, while every sibling constructor already returned a named
    `ValueError` carrying the sorted registered set. A mistyped encoding name in a config is
    an ordinary operator error and a panic is not how this repo reports one. `ReplayBuffer`
    itself went with the dense path (R346(f)); the siblings it was brought into line with are
    the census now, and the negative half — that none of them produces a panic — is the part
    that keeps the repair from silently reverting.
    """
    with pytest.raises(ValueError) as excinfo:
        construct()
    message = str(excinfo.value)
    assert "unknown encoding" in message or "not in registry" in message, message
    assert "gnn_axis_v1" in message, "the registered set must be in the message"
    assert not isinstance(excinfo.value, panic_exception)


def test_process_survives_a_caught_panic(panic_exception):
    """After catching a panic the interpreter is still live and usable —
    proves unwind, not abort."""
    # DERIVED, not typed: the claim is that the registry SURVIVES the panics, so the
    # comparison is against the same surface read before them. A typed count made this row a
    # second authority over the size of the registered set, and it reds on any registry change
    # while saying nothing about unwinding (R192(e), derive-or-delete).
    before = len(_engine.all_specs())
    assert before > 0, "the registry was already empty; this row cannot show survival"
    # THIS ROW'S DRIVER HAS MOVED TWICE. AUDIT-1 F-38 made `ReplayBuffer` with an unknown
    # encoding a named `ValueError`, so the loop moved to the multi-window dense kernel's
    # `unimplemented!`; R346(f) deleted that with the grid path. The driver is now pyo3's own
    # unsendable assertion — a `Board` touched from a second thread — which is the only panic
    # this tree can still reach, and reaching it repeatedly is what the claim needs. The claim
    # is unchanged and is the one `panic = "unwind"` exists for: an ABORT would take the
    # process with it, and no assertion after it would run.
    board = _engine.Board.with_encoding_name("gnn_axis_v1")
    board.apply_move(0, 0)

    def _touch_off_thread() -> None:
        try:
            board.apply_move(1, 0)
        except panic_exception:
            pass

    for _ in range(3):
        thread = threading.Thread(target=_touch_off_thread)
        thread.start()
        thread.join()
    # The engine is still fully functional after repeated caught panics.
    assert len(_engine.all_specs()) == before
    b = _engine.Board.with_encoding_name("gnn_axis_v1")
    b.apply_move(0, 0)
    assert b.ply == 1
