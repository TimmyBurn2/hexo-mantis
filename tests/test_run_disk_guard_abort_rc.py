# >300 justify (R8): the seven rows are ONE claim — a fired disk guard is
# supervisor-distinguishable from a clean run — over ONE harness that every end-to-end row needs.
# R5 bars cross-test imports, so a split forks that harness, and it would also fork the drive's
# one safety property: the guard delivers a REAL SIGTERM here.
"""The disk-guard abort's process rc, DRIVEN.

The finding, measured before the fix: `DiskGuard.check_once` SIGTERMs its own pid below
`fail_gb`; the signal handler sets `shutdown_save`/`running` and NEVER `abort_rule`, which had
exactly ONE writer in all of `src/`. So `mantis.run.main` read `rule is None` and returned 0 — a
run the disk guard killed reported SUCCESS. REAL here: `main`, `launch_run`, `compose_run`, the
real `DiskGuard` on its real thread, the real signal handlers, a real minted config read back
through the ONE loader, and a REAL SIGTERM from the guard's own `os.kill`.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

import mantis.run as mantis_run
from mantis.config.armed_aborts import (
    DISK_SPACE_ABORT_RULE,
    MANIFEST,
    Status,
    exit_code_for_abort,
)
from mantis.monitor.heartbeat import DISK_SPACE_EXHAUSTED_EXIT_CODE
from mantis.run import RunCollaborators
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.signals import ShutdownState

_REPO = Path(__file__).resolve().parents[1]

#: The bounded burst; 3 is the smallest legal run at cadence 1.
_DRIVE_STEPS = 3

#: A guard cadence short enough to fire inside a sub-second burst; which side of the thresholds
#: a drive lands on is decided by the rigged usage, never by these numbers.
_DRIVE_GUARD = {"interval_sec": 0.02, "warn_gb": 4.0, "fail_gb": 2.0}

#: Rigged free space, in decimal GB (`disk_guard.py`'s `/1e9` divisor).
_HEALTHY_GB = 500.0
_CRITICAL_GB = 1.0


def _fake_disk_usage(free_gb: float):
    def _usage(_path):
        total = int(free_gb * 1_000_000_000) * 4
        return shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            total=total, used=total - int(free_gb * 1_000_000_000),
            free=int(free_gb * 1_000_000_000),
        )
    return _usage


class _Pool:
    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # F-816-2: the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    class _RunnerStats:
        mcts_mean_depth = 5.0
        mcts_mean_root_concentration = 0.1
        cluster_value_std_mean = 0.0
        cluster_policy_disagreement_mean = 0.0
        cluster_variance_sample_count = 0

    def __init__(self) -> None:
        self.started = False
        self._games = 0
        self.recent_move_histories: list = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("cannot join thread before it is started")

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return self._RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        return None

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self, on_step=None) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self.saves: list = []
        self._on_step = on_step

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        if self._on_step is not None:
            self._on_step(self.step)
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return {}

    def save_checkpoint(self, loss_info) -> None:
        self.saves.append(loss_info)


class _Drive:
    """What one `main()` drive observed: its rc, and the live objects it composed."""

    def __init__(self) -> None:
        self.rc: int | None = None
        self.handles: Any = None
        self.guards: list[DiskGuard] = []


def _write_config(tmp_path: Path, smoke_run_config) -> Path:
    """A REAL minted config written to disk, so `main --config` reads it back through the loader."""
    config = smoke_run_config(
        "dev_example.yaml", eval_enabled=False,
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
                 "disk_guard": dict(_DRIVE_GUARD)},
    )
    path = tmp_path / "drive.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request, *,
                free_gb: float, wait_for_fire: bool) -> _Drive:
    """Run `mantis.run.main(--config … --out-dir …)` over a rigged filesystem. `wait_for_fire`
    blocks the fake train step until the guard's latch is set, so the drive measures the FIX and
    never a race; the REAL sink is closed by a finalizer registered at its own construction,
    because `close_out` never touches it on a completed `compose_run`."""
    drive = _Drive()
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(shutil, "disk_usage", _fake_disk_usage(free_gb))

    class _RecordedGuard(DiskGuard):
        """The REAL guard; every behaviour is `super()`'s. Recorded so the drive can find it."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            drive.guards.append(self)

    def _await_fire(_step: int) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if drive.guards and drive.guards[-1].critical_fired:
                return
            time.sleep(0.005)
        raise AssertionError(
            "the rigged filesystem never drove the guard's critical arm inside 10 s — the "
            "drive's premise is broken, so nothing below would be measuring the fix"
        )

    trainer = _Trainer(on_step=_await_fire if wait_for_fire else None)
    out_dir = tmp_path / "out"
    collaborators = RunCollaborators(
        trainer=trainer, pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=out_dir / "logs", checkpoint_dir=out_dir / "checkpoints",
    )
    monkeypatch.setattr(mantis_run, "build_run_collaborators",
                        lambda **_kwargs: collaborators)
    monkeypatch.setattr(mantis_run, "DiskGuard", _RecordedGuard)

    real_build_run_safety = mantis_run.build_run_safety

    def _recording_build_run_safety(**kwargs):
        run_safety = real_build_run_safety(**kwargs)      # the REAL builder, unmodified
        request.addfinalizer(run_safety.sink.close)
        return run_safety

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build_run_safety)

    real_compose = mantis_run.compose_run

    def _recording_compose(**kwargs):
        drive.handles = real_compose(**kwargs)      # the REAL composer, unmodified
        return drive.handles

    monkeypatch.setattr(mantis_run, "compose_run", _recording_compose)

    config_path = _write_config(tmp_path, smoke_run_config)
    drive.rc = mantis_run.main(["--config", str(config_path), "--out-dir", str(out_dir)])
    return drive


def _await_signal(state: ShutdownState) -> None:
    """CPython delivers a signal at a bytecode boundary, so the handler may still be pending when
    `compose_run` returns. Bounded wait, then assert — never a SIGTERM left pending."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and state.stop_count < 1:
        time.sleep(0.005)


def _events(run_safety) -> list[dict]:
    return [json.loads(line) for line in
            Path(run_safety.sink.path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_run_the_disk_guard_killed_exits_47_and_a_clean_run_exits_0(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """THE A/B: same launcher, config shape and guard, one rigged volume apart, and the two runs
    must not hand a supervisor the same number. Measured before the fix: `abort_rule: null …
    MAIN_RC: 0`. MUTATION THAT REDS IT: suppress the recording, or drop the root's transfer."""
    fired = _drive_main(tmp_path / "fired", monkeypatch, smoke_run_config, mk_graph_buffer,
                        request, free_gb=_CRITICAL_GB, wait_for_fire=True)
    state = fired.handles.shutdown
    _await_signal(state)

    assert fired.guards and fired.guards[-1].critical_fired, (
        "premise: the guard's critical arm fired on the rigged volume"
    )
    levels = [event.get("level") for event in _events(fired.handles.run_safety)
              if event.get("event") == "disk_alert"]
    assert "critical" in levels, (
        f"premise: the run's OWN stream carries the critical alert; saw {levels}"
    )
    assert (state.stop_count, state.shutdown_save, state.running) == (1, True, False), (
        "premise: the guard's SIGTERM landed on LAW-16's handlers and requested "
        f"save-then-exit — got stop_count={state.stop_count}, "
        f"shutdown_save={state.shutdown_save}, running={state.running}"
    )
    assert state.abort_rule == DISK_SPACE_ABORT_RULE, (
        "the composition root must RECORD which rule stopped the run; at HEAD nothing did, "
        f"and `abort_rule is None` is the only thing that means a clean run. Got "
        f"{state.abort_rule!r}"
    )
    assert fired.rc == DISK_SPACE_EXHAUSTED_EXIT_CODE == 47, (
        "…and the launcher must hand the supervisor the registered code, not 0: reporting an "
        f"aborted run as clean relaunches it into the same full volume. Got {fired.rc!r}"
    )

    clean = _drive_main(tmp_path / "clean", monkeypatch, smoke_run_config, mk_graph_buffer,
                        request, free_gb=_HEALTHY_GB, wait_for_fire=False)
    assert clean.guards and not clean.guards[-1].critical_fired, (
        "premise: a healthy volume never fires the critical arm"
    )
    assert clean.handles.shutdown.abort_rule is None and clean.rc == 0, (
        "the CONTROL: a bounded run that reached its step ceiling is a clean stop and exits "
        f"0. Got rule={clean.handles.shutdown.abort_rule!r}, rc={clean.rc!r} — an oracle "
        "whose control also answered 47 would prove nothing about the guard"
    )
    assert fired.rc != clean.rc, (
        "R84's whole requirement, stated as the difference it is: a fired guard must be "
        "SUPERVISOR-DISTINGUISHABLE from a clean run"
    )


def test_the_rc_is_resolved_off_the_manifest_row_and_is_never_a_literal(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """The number must come from the manifest row, not a literal at the launcher. MUTATION THAT
    REDS IT: `return 47` beside the resolver call — a literal cannot follow a rewired row."""
    drive = _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                        request, free_gb=_CRITICAL_GB, wait_for_fire=True)
    _await_signal(drive.handles.shutdown)
    row = next(r for r in MANIFEST if r.name == DISK_SPACE_ABORT_RULE)
    assert drive.rc == row.exit_code == exit_code_for_abort(DISK_SPACE_ABORT_RULE), (
        f"the rc IS the row's `exit_code`, resolved; got rc={drive.rc!r} against "
        f"row={row.exit_code!r}"
    )
    import dataclasses
    rewired = tuple(dataclasses.replace(r, exit_code=91) if r.name == DISK_SPACE_ABORT_RULE
                    else r for r in MANIFEST)
    assert exit_code_for_abort(DISK_SPACE_ABORT_RULE, manifest=rewired) == 91, (
        "…and the resolver READS the row rather than branching on the rule's name"
    )


def test_suppressing_the_recording_collapses_the_rc_back_to_zero(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """THE MUTATION R84's template requires: `record_abort` neutered, the guard still fires, still
    SIGTERMs, still stops the run — and the process reports 0."""
    monkeypatch.setattr(ShutdownState, "record_abort", lambda self, rule: False)
    drive = _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                        request, free_gb=_CRITICAL_GB, wait_for_fire=True)
    state = drive.handles.shutdown
    _await_signal(state)
    assert drive.guards[-1].critical_fired and state.stop_count == 1, (
        "the guard's behaviour is UNTOUCHED by the mutation — it fired and it signalled"
    )
    assert state.abort_rule is None and drive.rc == 0, (
        "…and with the recording suppressed the run the guard killed reports SUCCESS, which "
        f"is RT-2 verbatim. Got rule={state.abort_rule!r}, rc={drive.rc!r}"
    )


def test_the_critical_arm_signals_once_per_run_while_the_alert_keeps_firing() -> None:
    """The guard polls on a condition that does not clear itself, so an UNLATCHED arm supplies the
    second press of the two-press force-exit ITSELF: `sys.exit(1)` from a signal handler, MID-SAVE,
    against `close_out`'s drain caps. `os.kill` is CAPTURED, not delivered, so the RED state is an
    assertion failure rather than a dead pytest process. MUTATION THAT REDS IT: drop the
    `if first_fire:` guard; in the other direction, latching the emit too drops the alert count."""
    kills: list = []
    alerts: list = []

    class _Sink:
        def emit(self, payload) -> None:
            if payload.get("event") == "disk_alert":
                alerts.append(payload["level"])

    guard = DiskGuard(watch_path=Path("."), interval_sec=60.0, warn_gb=4.0, fail_gb=2.0,
                      keep_all=False, sink=_Sink())
    real_kill, real_usage = os.kill, shutil.disk_usage
    try:
        os.kill = lambda pid, sig: kills.append((pid, sig))          # type: ignore[assignment]
        shutil.disk_usage = _fake_disk_usage(_CRITICAL_GB)           # type: ignore[assignment]
        for _ in range(3):
            guard.check_once()
    finally:
        os.kill, shutil.disk_usage = real_kill, real_usage           # type: ignore[assignment]

    assert kills == [(os.getpid(), signal.SIGTERM)], (
        "exactly ONE SIGTERM across three crossings — the second press is the operator's, "
        f"and a guard that supplies it force-exits the save it just asked for. Got {kills}"
    )
    assert alerts == ["critical", "critical", "critical"], (
        "…and the ALERT is not latched: the condition persists and an operator watching the "
        f"stream must keep seeing it. Got {alerts}"
    )
    assert guard.critical_fired is True


def test_the_latch_is_a_produced_fact_and_starts_false() -> None:
    """`critical_fired` is not a config proxy: it means "my critical arm signalled this process",
    produced by the guard and read after `stop()` joins the thread. MUTATION THAT REDS IT:
    initialise it True, or make it settable by the root."""
    guard = DiskGuard(watch_path=Path("."), interval_sec=60.0, warn_gb=4.0, fail_gb=2.0,
                      keep_all=False, sink=type("_S", (), {"emit": lambda self, p: None})())
    assert guard.critical_fired is False, "a guard that has not fired has not fired"
    with pytest.raises(AttributeError):
        guard.critical_fired = True     # type: ignore[misc]


def test_record_abort_is_set_once_and_the_first_fire_wins() -> None:
    """`record_abort` is THE writer for BOTH fire paths, and set-once stops two authorities
    disagreeing. MUTATION THAT REDS IT: a plain assignment, last writer wins."""
    state = ShutdownState()
    assert state.abort_rule is None
    assert state.record_abort("draw_rate_collapse") is True
    assert state.record_abort(DISK_SPACE_ABORT_RULE) is False, (
        "a second fire is a NO-OP, and says so in its return value"
    )
    assert state.abort_rule == "draw_rate_collapse", (
        "…and the rule that stopped the run is the one that stopped it"
    )


def test_the_manifest_row_is_required_and_its_pin_still_binds_the_transfer() -> None:
    """The row is REQUIRED because nothing is owed: `fail_gb` is minted with `gt=0` in a required
    block, so a validated `RunConfig` arms it by construction. What it is FOR is the drift, since
    `_dotted` short-circuits a mid-walk `None` to DISARMED. MUTATION THAT REDS IT: rename the
    constant, delete the transfer line, or move it above `disk_guard.stop()` — the last is the
    subtle one, since the latch would be read before the guard thread is joined."""
    row = next(r for r in MANIFEST if r.name == DISK_SPACE_ABORT_RULE)
    assert row.status is Status.REQUIRED and row.owner is None, (
        "a REQUIRED row carries no owner (an owner reads as already-excused); "
        f"got {row.status} / {row.owner!r}"
    )
    assert row.config_path == "monitor.disk_guard.fail_gb"
    assert row.exit_code == DISK_SPACE_EXHAUSTED_EXIT_CODE
    assert row.source_pin is not None, "R56 tamper-evidence"
    rel, text = row.source_pin
    assert text in (_REPO / rel).read_text(encoding="utf-8"), (
        f"the pinned text {text!r} is gone from {rel} — the mechanism the rc depends on was "
        "deleted, renamed or reordered, and the row now claims a fire path that is not there"
    )
    source = (_REPO / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    assert source.index("disk_guard.stop()") < source.index(text), (
        "the transfer must read the latch AFTER the guard thread is joined: ordered the "
        "other way, a guard firing during teardown is lost and the rc silently returns to 0"
    )
