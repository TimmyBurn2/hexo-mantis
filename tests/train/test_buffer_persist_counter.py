"""R-BUFFER-PERSIST-COUNTER closure oracles (WPCLEAN Phase RES; LAW-07 producer tests).

The debt (WPSC DEBT_DOSSIER, from WP13-A): `try_save_buffer` swallowed save failures —
WARN, uncounted, outside every counters_fn — including on the SHUTDOWN save, the last save
of a run. The closure: a module counter (`buffer_save_errors_total`), deliberately separate
from `checkpoints.persist_errors_total` (the watchdog's `> 0` FATAL feed — this path is
best-effort by design), with a live LAW-18 consumer in the coordinator's `monitor_gates`
payload. Every arm below is driven, not asserted from source.
"""
from __future__ import annotations

from types import SimpleNamespace

import mantis.train.buffer_persist as buffer_persist
import mantis.train.checkpoints as checkpoints
from mantis.train.buffer_persist import try_save_buffer
from mantis.train.coordinator.step import StepCoordinator


class _SaveExplodes:
    size = 3

    def save_to_path(self, path: str) -> None:
        raise OSError("disk on fire (test)")


class _SaveOk:
    size = 3

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_to_path(self, path: str) -> int:
        self.saved.append(path)
        return self.size


def _cfg(tmp_path) -> dict:
    return {"buffer_persist": True, "buffer_persist_path": str(tmp_path / "buf.bin")}


def test_a_failed_buffer_save_is_counted_and_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(buffer_persist, "buffer_save_errors_total", 0)
    monkeypatch.setattr(checkpoints, "persist_errors_total", 0)
    try_save_buffer(_SaveExplodes(), _cfg(tmp_path), "shutdown_signal")
    assert buffer_persist.buffer_save_errors_total == 1
    assert checkpoints.persist_errors_total == 0, (
        "a best-effort buffer save fed the persist-FATAL counter — the watchdog would "
        "abort (rc 43) a run over an optional snapshot"
    )


def test_the_recent_buffer_arm_is_counted_too(tmp_path, monkeypatch):
    monkeypatch.setattr(buffer_persist, "buffer_save_errors_total", 0)
    try_save_buffer(_SaveOk(), _cfg(tmp_path), "checkpoint_interval",
                    recent_buffer=_SaveExplodes())
    assert buffer_persist.buffer_save_errors_total == 1


def test_a_healthy_save_counts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(buffer_persist, "buffer_save_errors_total", 0)
    buf = _SaveOk()
    try_save_buffer(buf, _cfg(tmp_path), "checkpoint_interval")
    assert buf.saved and buffer_persist.buffer_save_errors_total == 0


def test_disabled_persistence_stays_a_true_no_op(tmp_path, monkeypatch):
    """The discriminating negative: the counter counts FAILURES, not the disabled arm."""
    monkeypatch.setattr(buffer_persist, "buffer_save_errors_total", 0)
    try_save_buffer(_SaveExplodes(), {"buffer_persist": False}, "shutdown_signal")
    assert buffer_persist.buffer_save_errors_total == 0


def test_monitor_gates_payload_reads_the_counter_live(monkeypatch):
    """The LAW-08 consumer arm, and it must be a LIVE module-attribute read: the payload
    reflects a count that moved AFTER import (a from-import of the int would freeze 0)."""
    monkeypatch.setattr(buffer_persist, "buffer_save_errors_total", 7)
    emitted: list[dict] = []
    sink = SimpleNamespace(emit=emitted.append)
    fake_coord = SimpleNamespace(
        _train_step=1, _gate_stats={}, _wr_history=[],
        monitor_cfg=SimpleNamespace(wr_hard_abort_enabled=False),
        _watchdog_counters=lambda: {},
    )
    cfg = SimpleNamespace(draw_rate_abort=None)
    StepCoordinator._emit_monitor_gates(fake_coord, cfg, sink)
    assert emitted and emitted[0]["event"] == "monitor_gates"
    assert emitted[0]["buffer_save_errors_total"] == 7
