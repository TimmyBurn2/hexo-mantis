"""Lifecycle CONTRACT: SIGINT/SIGTERM save-then-exit, always-armed self-play stall watchdog,
disk guard, persist-fatal.

Signals use a real ShutdownState + monkeypatched sys.exit; the watchdog and disk guard use a
fake clock, an injected EventSink spy, and monkeypatched exit_fn / os.kill / free-space fn.
"""
from __future__ import annotations

import collections
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest
import torch

from mantis.train.emit import EventSink, NullEventSink  # noqa: F401 — the injected emit seam
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers
from mantis.train.lifecycle.watchdog import (
    DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC,  # noqa: F401 — pinned default (context law: 1800.0)
    SELFPLAY_STALL_EXIT_CODE,
    StallWatchdog,
    watchdog_snapshot_path,
)

GB = 1_000_000_000  # decimal GB: reproduces the old `usage.free / 1e9` divisor exactly, where
# `1024 ** 3` would have shifted every threshold by ~7.4%.


@pytest.fixture
def restore_signals():
    """Capture + restore the process SIGINT/SIGTERM handlers around a signal test."""
    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGINT, orig_int)
    signal.signal(signal.SIGTERM, orig_term)


def _fake_disk_usage(free_gb: float):
    """A shutil.disk_usage replacement reporting `free_gb` free on a 100 GB volume."""
    usage = collections.namedtuple("usage", "total used free")

    def f(_path):
        return usage(total=100 * GB, used=int((100 - free_gb) * GB), free=int(free_gb * GB))

    return f


def test_sigint_sets_save_then_exit_state(restore_signals):
    """One SIGINT flips running=False and shutdown_save=True. Bites: a signal that does not
    request a save."""
    state = ShutdownState()
    install_signal_handlers(state)
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)
    handler(signal.SIGINT, None)
    assert state.running is False
    assert state.shutdown_save is True


def test_sigterm_sets_save_then_exit_state(restore_signals):
    """One SIGTERM does the same. Bites: SIGTERM ignored (the disk-guard fail path relies on
    it)."""
    state = ShutdownState()
    install_signal_handlers(state)
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler)
    handler(signal.SIGTERM, None)
    assert state.running is False
    assert state.shutdown_save is True


def test_double_signal_force_exits(restore_signals, monkeypatch):
    """A second signal (stop_count>=2) force-tears-down registered children then calls
    os._exit(1). Bites: no forced exit, or exiting with children left orphaned."""
    from mantis.train.lifecycle import signals as sig_mod
    state = ShutdownState()
    install_signal_handlers(state)
    handler = signal.getsignal(signal.SIGINT)
    teardown_called = []
    monkeypatch.setattr(sig_mod, "force_teardown_all",
                        lambda: teardown_called.append(1))
    monkeypatch.setattr(os, "_exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    handler(signal.SIGINT, None)  # 1st → save-then-exit request
    assert state.stop_count >= 1
    with pytest.raises(SystemExit) as ei:
        handler(signal.SIGINT, None)  # 2nd → force teardown + os._exit(1)
    assert ei.value.code == 1
    assert state.stop_count >= 2
    assert teardown_called, "second signal must force-teardown children before os._exit"


def test_loop_saves_final_checkpoint_on_shutdown():
    """The loop, observing shutdown_save, calls trainer.save_checkpoint before returning.
    Bites: a shutdown that exits without the final save."""
    from mantis.train.loop import run_training_loop  # Slice 2 (lazy)

    class SpyTrainer:
        def __init__(self) -> None:
            self.saved: list = []

        def save_checkpoint(self, *a, **k):
            self.saved.append((a, k))
            return None

    spy = SpyTrainer()
    state = ShutdownState(running=False, shutdown_save=True)
    run_training_loop(trainer=spy, shutdown_state=state)  # 0 steps → observe shutdown_save → final save
    assert spy.saved, "loop must call trainer.save_checkpoint on shutdown_save before returning"


def _watchdog(spy_sink, fake_clock, *, timeout=DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC):
    exits: list = []
    snaps: list = []
    wd = StallWatchdog(
        timeout_sec=timeout, clock=fake_clock, sink=spy_sink,
        exit_fn=lambda code: exits.append(code),
        save_snapshot=lambda: snaps.append(1),
    )
    return wd, exits, snaps


def test_watchdog_arms_always(spy_sink, fake_clock):
    """The watchdog arms and emits selfplay_stall_watchdog_armed regardless of config. Bites: a
    conditionally-armed watchdog."""
    wd, _exits, _snaps = _watchdog(spy_sink, fake_clock)
    wd.arm(0)
    assert spy_sink.has("selfplay_stall_watchdog_armed")


def test_watchdog_resets_on_new_games(spy_sink, fake_clock):
    """tick(games_completed) with an increased count resets the stall clock. Bites: a stall
    clock that never resets, so it fires spuriously."""
    fake_clock.t = 0.0
    wd, exits, _snaps = _watchdog(spy_sink, fake_clock, timeout=1800.0)
    wd.arm(0)                      # baseline at t=0, games=0
    wd.tick(5, now=1000.0)         # NEW games (5>0) → reset stall clock to t=1000
    wd.tick(5, now=2500.0)         # same games; 2500-1000=1500 < 1800 → NO fire
    assert not exits, "a reset to t=1000 must prevent a fire at t=2500 (else 2500-0 > 1800 fires)"


def test_watchdog_fires_after_timeout(spy_sink, fake_clock):
    """No new games for >= timeout gives a loud selfplay_stall_watchdog log, a best-effort
    snapshot and os._exit(SELFPLAY_STALL_EXIT_CODE). Bites: no fire, wrong exit code, or a
    clean-shutdown attempt."""
    fake_clock.t = 0.0
    wd, exits, snaps = _watchdog(spy_sink, fake_clock, timeout=1800.0)
    wd.arm(0)                      # t=0, games=0
    wd.tick(0, now=1800.0)         # no new games; 1800-0 >= 1800 → FIRE
    assert exits == [SELFPLAY_STALL_EXIT_CODE]
    assert snaps, "the fire must attempt a best-effort snapshot"
    assert spy_sink.has("selfplay_stall_watchdog")


def test_watchdog_snapshot_path_is_distinct():
    """The fire-time snapshot targets <buffer>.watchdog, never the canonical replay_buffer.bin.
    Bites: the watchdog truncating the known-good resume buffer."""
    canonical = Path("/data/run/replay_buffer.bin")
    snap = watchdog_snapshot_path(canonical)
    assert str(snap).endswith(".watchdog")
    assert snap != canonical
    assert snap.name == "replay_buffer.bin.watchdog"


def test_watchdog_no_fire_when_timeout_nonpositive(spy_sink, fake_clock):
    """selfplay_stall_timeout_sec <= 0 never fires while the arm-log still emits. Bites: firing
    when the timeout is disabled."""
    fake_clock.t = 0.0
    wd, exits, _snaps = _watchdog(spy_sink, fake_clock, timeout=0.0)
    wd.arm(0)
    assert spy_sink.has("selfplay_stall_watchdog_armed")  # arm-log still emits
    wd.tick(0, now=1e9)            # enormous stall, but timeout<=0 → NO fire
    assert not exits


def test_disk_guard_emits_free_event(tmp_path, spy_sink, monkeypatch):
    """check_once emits a disk_free event with disk_free_gb through the injected EventSink.
    Bites: no emission."""
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(20))
    dg = DiskGuard(watch_path=tmp_path, interval_sec=60, warn_gb=10, fail_gb=5,
                   keep_all=False, sink=spy_sink)
    free = dg.check_once()
    assert free == pytest.approx(20, abs=0.5)
    events = spy_sink.named("disk_free")
    assert events and events[-1]["disk_free_gb"] == pytest.approx(20, abs=0.5)


def test_disk_guard_warns_below_warn_gb(tmp_path, spy_sink, monkeypatch):
    """free < warn_gb gives a warn log + disk_alert level=warn. Bites: no warning near the
    threshold."""
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(8))  # < warn_gb=10
    dg = DiskGuard(watch_path=tmp_path, interval_sec=60, warn_gb=10, fail_gb=5,
                   keep_all=False, sink=spy_sink)
    dg.check_once()
    alerts = spy_sink.named("disk_alert")
    assert alerts and alerts[-1]["level"] == "warn"


def test_disk_guard_sigterm_below_fail_gb(tmp_path, spy_sink, monkeypatch):
    """free < fail_gb gives disk_alert level=critical + os.kill(getpid(), SIGTERM). Bites: no
    SIGTERM, so the run burns disk to zero."""
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(3))  # < fail_gb=5
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    dg = DiskGuard(watch_path=tmp_path, interval_sec=60, warn_gb=10, fail_gb=5,
                   keep_all=False, sink=spy_sink)
    dg.check_once()
    alerts = spy_sink.named("disk_alert")
    assert alerts and alerts[-1]["level"] == "critical"
    assert kills == [(os.getpid(), signal.SIGTERM)]


def test_disk_guard_thresholds_ignore_keep_all(tmp_path, spy_sink, monkeypatch):
    """keep_all=True does NOT disable the disk thresholds. Bites: keep_all silencing the safety
    guard."""
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(3))  # < fail_gb=5
    kills: list = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    dg = DiskGuard(watch_path=tmp_path, interval_sec=60, warn_gb=10, fail_gb=5,
                   keep_all=True, sink=spy_sink)  # keep_all=True
    dg.check_once()
    assert kills == [(os.getpid(), signal.SIGTERM)], "keep_all must NOT silence the SIGTERM guard"


def test_buffer_persist_error_increments_counter_and_aborts(tmp_path, tiny_net, valid_config,
                                                            metadata_kwargs, monkeypatch):
    """A buffer/checkpoint save failure increments persist_errors_total and aborts run-fatally
    rather than swallowing it. Bites: a swallowed persist failure.

    `persist_errors_total` is a process-wide module GLOBAL, so the monkeypatch pins it to 0 AND
    restores the pre-test value at teardown: the heartbeat watchdog's persist-fatal rule is the
    literal `> 0`, and a leaked count would abort a later, healthy watchdog."""
    from mantis.train import checkpoints  # Slice 1
    monkeypatch.setattr(checkpoints, "persist_errors_total", 0)
    before = checkpoints.persist_errors_total

    def _boom(*_a, **_k):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", _boom)
    opt = torch.optim.AdamW(tiny_net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    with pytest.raises((OSError, RuntimeError)):  # aborts — the failure is NOT swallowed
        checkpoints.save_checkpoint(
            model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=100,
            config=valid_config, metadata_kwargs=metadata_kwargs, checkpoint_dir=tmp_path,
            kind="full",
        )
    assert checkpoints.persist_errors_total == before + 1  # counted (never except: pass)
    assert list(tmp_path.glob("*.ckpt")) == []
