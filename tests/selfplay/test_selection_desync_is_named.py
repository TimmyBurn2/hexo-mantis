"""AUDIT-1 F-02 (Python face) — the MCTS desync is a named exception, not a matched panic.

`MCTSTree::select_one_leaf` did `board.apply_move_tracked(q, r).expect("selected move should
always be legal")`. `Board::apply_move` errs ONLY on occupancy — never on radius — so the
expect fired exactly when a child's stored `action_idx` decoded to a cell already on the
board. It DID fire in production: this repo's own `selfplay/worker.py` carried a
`BaseException` handler matching the panic's message text, `"cell already occupied"`, and
restarted the tree at root in batch-1 mode.

Three things were wrong with that recovery, and all three are closed engine-side:

* it triggered only when `current_batch > 1`, so the same desync at batch 1 propagated as an
  unnamed `PanicException`;
* on the RUST self-play arm nothing reached it at all — the unwind is caught by
  `runner::spawn::guard_worker`, which increments `worker_panics` and sets `running = false`,
  halting the run with NO reason in the fatal-defect latch, so R275(b)'s instrument never saw
  it;
* matching a panic's message text is a contract nothing enforces, and `panic = "unwind"`
  (R2/LAW-13) is what made the panic catchable at all — a guarantee about the worst case.
"""
from __future__ import annotations

import inspect

import pytest

from mantis import _engine


def test_the_named_exception_is_exported_from_the_engine() -> None:
    """A caller must be able to name it in an `except` clause — the whole point of retiring
    the message match."""
    assert hasattr(_engine, "SelectionDesync")
    assert issubclass(_engine.SelectionDesync, Exception)


def test_a_forced_root_child_outside_the_range_raises_ValueError_not_a_panic() -> None:
    """F-02's second trigger, at the FFI. `u32::MAX` index-panicked on the next descent, and
    any other foreign index descended into a node the root does not own — an uninitialised
    slot carries `action_idx = u32::MAX`, which decodes to (32767, 32767), a cell an
    UNBOUNDED board accepts. That arm produced neither a panic nor an error."""
    tree = _engine.MCTSTree(1.5, 1.0, 0.25, True, 0.3)
    board = _engine.Board()
    tree.new_game(board)
    # `forced_root_child` is a pyo3 PROPERTY (a `#[setter]`), so the refusal arrives on
    # assignment rather than from a method call.
    with pytest.raises(ValueError, match="not a child of the root"):
        tree.forced_root_child = 2**32 - 1
    tree.forced_root_child = None  # clearing is always allowed
    assert tree.forced_root_child is None


def test_a_short_expand_batch_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUDIT-1 F-22. The inner call took `n = min(pending, policies, values)` and DROPPED the
    rest, so a short batch from the inference side expanded the leading leaves and silently
    skipped the others — on `selfplay/worker.py` and on `arena/deploy_head.py`, the
    deploy-strength path. The graph sibling already carried C-1..C-4 guards for exactly this."""
    tree = _engine.MCTSTree(1.5, 1.0, 0.25, True, 0.3)
    board = _engine.Board.with_encoding_name("v6")
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    assert len(leaves) == 1
    with pytest.raises(ValueError, match="select_leaves returned"):
        tree.expand_and_backup([], [])


def test_the_worker_no_longer_matches_a_panics_message_text() -> None:
    """The recovery arm is DELETED, not merely bypassed. A string match on an exception's
    message is a contract nothing enforces — and this one covered one arm of two."""
    import mantis.selfplay.worker as worker

    source = inspect.getsource(worker)
    assert "cell already occupied" not in source, (
        "the panic-text match is back; the desync is a named error now and the recovery it "
        "performed (a silent tree reset) exported a search that did not happen"
    )
    assert "except BaseException" not in source, (
        "a BaseException handler around the search loop swallows KeyboardInterrupt too"
    )
