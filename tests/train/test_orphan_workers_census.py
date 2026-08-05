"""CARD-ORPHAN-WORKERS (R230) — SIGINT during active self-play leaves ZERO descendant
processes (process-tree census, bounded wait). The second-signal path too.

Oracle: spawn a child process simulating the eval pipeline's spawn child, register it,
send SIGINT to the test process — the child MUST be dead after a bounded wait.

Mutation: disable ``force_teardown_all`` (no-op) → the child survives the second signal
→ the census test reds.

Flip-set (R71 — class boundary, not the demo): the eval pipeline's spawn child is the
ONE orphanable child process in the system (the Rust runner uses threads, not
processes). The registry covers the class; the test exercises the class.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import time

import pytest

from mantis.train.lifecycle.signals import (
    ShutdownState,
    force_teardown_all,
    install_signal_handlers,
    register_child,
    unregister_child,
)


def _sleep_forever():
    """A child target that sleeps long enough to be torn down."""
    time.sleep(300)


@pytest.fixture
def restore_signals():
    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGINT, orig_int)
    signal.signal(signal.SIGTERM, orig_term)


def _spawn_child(ctx_name: str = "spawn"):
    ctx = multiprocessing.get_context(ctx_name)
    proc = ctx.Process(target=_sleep_forever, daemon=False)
    proc.start()
    return proc


def test_force_teardown_all_kills_registered_child():
    """ORACLE — a registered child is killed by force_teardown_all (terminate → join → kill)."""
    proc = _spawn_child()
    try:
        register_child(proc)
        assert proc.is_alive()
        force_teardown_all(grace_sec=2.0)
        proc.join(5.0)
        assert not proc.is_alive(), "child must be dead after force_teardown_all"
    finally:
        unregister_child(proc)
        if proc.is_alive():
            proc.kill()
            proc.join(2.0)


def test_force_teardown_all_idempotent_no_children():
    """ORACLE — force_teardown_all with no registered children is a no-op (no error)."""
    force_teardown_all(grace_sec=1.0)


def test_unregister_removes_child_from_registry():
    """ORACLE — an unregistered child is NOT killed by force_teardown_all."""
    proc = _spawn_child()
    try:
        register_child(proc)
        unregister_child(proc)
        force_teardown_all(grace_sec=1.0)
        proc.join(1.0)
        assert proc.is_alive(), "unregistered child must survive force_teardown_all"
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(2.0)


def test_second_signal_kills_registered_child(restore_signals, monkeypatch):
    """ORACLE — second SIGINT calls force_teardown_all then os._exit(1); the registered
    child is dead before the process exits."""
    from mantis.train.lifecycle import signals as sig_mod

    state = ShutdownState()
    install_signal_handlers(state)
    handler = signal.getsignal(signal.SIGINT)

    proc = _spawn_child()
    register_child(proc)
    assert proc.is_alive()

    exits: list = []
    monkeypatch.setattr(os, "_exit", lambda code=0: exits.append(code))

    handler(signal.SIGINT, None)  # 1st → cooperative
    assert state.running is False
    assert proc.is_alive(), "first signal must not kill the child"

    handler(signal.SIGINT, None)  # 2nd → force teardown + os._exit
    assert exits == [1], "second signal must call os._exit(1)"
    proc.join(5.0)
    assert not proc.is_alive(), "second signal must kill the registered child"


def test_mutation_disabled_teardown_leaves_child_alive(restore_signals, monkeypatch):
    """MUTATION — if force_teardown_all is disabled (no-op), the child survives the second
    signal. This test REDS against the disabled teardown, proving the oracle bites."""
    from mantis.train.lifecycle import signals as sig_mod

    state = ShutdownState()
    install_signal_handlers(state)
    handler = signal.getsignal(signal.SIGINT)

    proc = _spawn_child()
    register_child(proc)
    assert proc.is_alive()

    monkeypatch.setattr(sig_mod, "force_teardown_all", lambda: None)
    monkeypatch.setattr(os, "_exit", lambda code=0: None)

    handler(signal.SIGINT, None)  # 1st
    handler(signal.SIGINT, None)  # 2nd — teardown disabled

    # The child survives because force_teardown_all was disabled.
    proc.join(1.0)
    assert proc.is_alive(), "with teardown disabled, the child survives (mutation proof)"

    # Clean up.
    proc.kill()
    proc.join(2.0)
    unregister_child(proc)
