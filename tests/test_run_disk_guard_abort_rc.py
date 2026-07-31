# >300 justify (R8), stated at this file's MEASURED size of 485 lines (`wc -l`, re-measured
# after the last edit — this run produced three stale headers by transcribing instead). The
# seven rows are ONE claim — a fired disk guard is supervisor-distinguishable from a clean run
# — over ONE harness: the collaborator stand-ins, the rigged `disk_usage`, the config writer
# and the `main()` driver are ~140 lines and every end-to-end row needs all of them. R5 bars
# cross-test imports, so a split forks that harness into two copies which then drift while both
# stay green, and it would ALSO fork the drive's one safety property — the guard delivers a
# REAL SIGTERM here, so the wait-for-fire hook and the bounded signal wait must stay in one
# place. Executable content is a minority: the rest is the mutation each row reds against,
# which is the whole of what R84's template asks a mutation test to carry.
"""⊕ WPMAIN RED-TEAM RT-2 / R132 — the disk-guard abort's process rc, DRIVEN.

The finding, measured on this branch before the fix: `DiskGuard.check_once` SIGTERMs its own
pid below `fail_gb`; `install_signal_handlers._stop` sets `shutdown_save=True` /
`running=False` and **never** `abort_rule`; `abort_rule` had exactly ONE writer in all of
`src/` (`train/coordinator/step.py`'s `_fire_hard_abort`). So `mantis.run.main` read
`rule is None` and returned **0**. A run the disk guard killed reported SUCCESS, and the
supervisor above relaunches into the same full volume — R44's class (a green that lies) on the
leg this WP armed for the first time in any run.

R132's mandate is R84's template verbatim: a registered exit code, resolved through
`exit_code_for_abort`, a manifest row, a contract doc in the same commit, and a MUTATION
proving a fired guard is supervisor-distinguishable from a clean run. This file is that
mutation, plus the second arm.

**What is real here and what is not.** Real: `mantis.run.main` (the launcher an operator
types), `launch_run`, `compose_run`, the real `DiskGuard` on its real thread, the real
`install_signal_handlers` handlers, a real `ShutdownState`, a real `HexgBuffer`, a real minted
config written to disk and read back through the ONE loader, the real manifest and the real
resolver. A REAL `SIGTERM` is delivered to this process by the guard's own `os.kill` in the
end-to-end drives — that is the mechanism under test and faking it would test the fake. Fake:
`build_run_collaborators`' three collaborators (trainer/pool are the injected seam every
composition drive in this suite stands in), and `shutil.disk_usage`, which is rigged so the
threshold can be crossed on demand — the alternative is filling a real volume, and it is the
house precedent (`tests/train/test_lifecycle_contract.py::_fake_disk_usage`).

**The rc measured is `main`'s RETURN VALUE**, which `run.py`'s `sys.exit(main())` hands the
OS unchanged; that two-line `__main__` glue is censused statically by
`tests/test_run_main_authority.py` and is not re-driven here. Everything between the guard's
`os.kill` and that number is live.

R5 bars cross-test imports, so the collaborator stand-ins below are re-derived rather than
imported from `tests/test_run_root_lifecycle.py` (which is byte-frozen at `7c28536` besides).
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

#: The bounded burst. 3 is the smallest legal run at cadence 1 (the reachability validator
#: spans cadence < actor_lag_threshold < max_train_steps).
_DRIVE_STEPS = 3

#: A guard cadence short enough to fire inside a sub-second burst. The thresholds are the
#: minted SHAPE (fail < warn, both > 0); which side of them a drive lands on is decided by the
#: rigged `shutil.disk_usage`, never by the numbers here.
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


# ── the drivable collaborators (injection-first contract) ─────────────────────────────
class _Pool:
    gumbel_mcts = True
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
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
    """A REAL minted config, bounded, guard-cadenced, written to disk so `main --config`
    reads it back through the ONE loader (no fixture object is smuggled past the CLI)."""
    config = smoke_run_config(
        "smoke_gnn.yaml", eval_enabled=False,
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
                 "disk_guard": dict(_DRIVE_GUARD)},
    )
    path = tmp_path / "drive.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8")
    return path


def _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, *, free_gb: float,
                wait_for_fire: bool) -> _Drive:
    """Run `mantis.run.main(--config … --out-dir …)` over a rigged filesystem.

    `wait_for_fire` blocks the fake train step until the guard's latch is set, so the drive
    measures the FIX and never a race: without it a 3-step burst can outrun a 0.02 s poll and
    the run would exit 0 for a reason that has nothing to do with the defect.
    """
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

    real_compose = mantis_run.compose_run

    def _recording_compose(**kwargs):
        drive.handles = real_compose(**kwargs)      # the REAL composer, unmodified
        return drive.handles

    monkeypatch.setattr(mantis_run, "compose_run", _recording_compose)

    config_path = _write_config(tmp_path, smoke_run_config)
    drive.rc = mantis_run.main(["--config", str(config_path), "--out-dir", str(out_dir)])
    return drive


def _await_signal(state: ShutdownState) -> None:
    """CPython delivers a signal to the main thread at a bytecode boundary, so the handler
    may still be pending when `compose_run` returns. Bounded wait, then assert — a race must
    fail this file loudly, never leave a SIGTERM pending into the next test."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and state.stop_count < 1:
        time.sleep(0.005)


def _events(run_safety) -> list[dict]:
    return [json.loads(line) for line in
            Path(run_safety.sink.path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ══ RT-2 — the rc is distinguishable ══════════════════════════════════════════════════
def test_a_run_the_disk_guard_killed_exits_47_and_a_clean_run_exits_0(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """THE A/B R132 names: same launcher, same config shape, same guard — one rigged volume
    apart — and the two runs must not hand a supervisor the same number.

    Measured before the fix (RED-TEAM probe E1, and again with no rigged filesystem at all in
    probe J1): `disk_alert_levels: ["critical"] … abort_rule: null … MAIN_RC: 0`. The guard
    stopped the run and the process said "clean".

    MUTATION THAT REDS IT: see the two below — suppress the recording, or drop the
    root's transfer. Either restores rc 0 while every other assertion in this file holds.
    """
    fired = _drive_main(tmp_path / "fired", monkeypatch, smoke_run_config, mk_graph_buffer,
                        free_gb=_CRITICAL_GB, wait_for_fire=True)
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
                        free_gb=_HEALTHY_GB, wait_for_fire=False)
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
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """R84's one-authority half. The number must come from the row, not from a second
    literal at the launcher — otherwise moving the row's `exit_code` leaves the process
    exiting the old number and the manifest lying about it.

    MUTATION THAT REDS IT: `return 47` in `main` beside the resolver call. Every assertion in
    the test above stays green; this one reds, because the rewired manifest moves the
    resolver's answer and a literal cannot follow it."""
    drive = _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                        free_gb=_CRITICAL_GB, wait_for_fire=True)
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
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """THE MUTATION R84's template requires, driven rather than described (R81 under R86: it
    kills the PRODUCTION writer, not a test helper, and its casualty is in-subject).

    `ShutdownState.record_abort` is the one writer of `abort_rule`. Neutered, the guard still
    fires, still SIGTERMs, still logs its critical alert, still stops the run through
    save-then-exit — and the process reports **0**. That is the defect exactly, restored, and
    it is what the two tests above would look like if the fix were deleted. An oracle nobody
    has seen red is not evidence."""
    monkeypatch.setattr(ShutdownState, "record_abort", lambda self, rule: False)
    drive = _drive_main(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                        free_gb=_CRITICAL_GB, wait_for_fire=True)
    state = drive.handles.shutdown
    _await_signal(state)
    assert drive.guards[-1].critical_fired and state.stop_count == 1, (
        "the guard's behaviour is UNTOUCHED by the mutation — it fired and it signalled"
    )
    assert state.abort_rule is None and drive.rc == 0, (
        "…and with the recording suppressed the run the guard killed reports SUCCESS, which "
        f"is RT-2 verbatim. Got rule={state.abort_rule!r}, rc={drive.rc!r}"
    )


# ══ RT-2b — the guard never supplies LAW-16's second press ════════════════════════════
def test_the_critical_arm_signals_once_per_run_while_the_alert_keeps_firing() -> None:
    """RT-2b. The guard polls every `interval_sec` (minted 60 s) on a condition that does not
    clear itself, so an UNLATCHED arm supplies the second press of the two-press force-exit
    ITSELF: `_stop` hits `stop_count >= 2` and `sys.exit(1)`s from a signal handler, at an
    arbitrary point in the main thread, MID-SAVE — against `close_out`'s 14400 s drain caps.
    The two-press force-exit is the OPERATOR's affordance and it stays theirs.

    `os.kill` is CAPTURED, not delivered: the RED state of this oracle must be an assertion
    failure, never a dead pytest process, and an unlatched arm would deliver a real second
    SIGTERM into a real handler here.

    MUTATION THAT REDS IT: drop the `if first_fire:` guard around `os.kill` — three kills.
    THE OTHER DIRECTION, equally required: latch the whole arm (skip the emit too) and the
    alert count drops to 1, hiding a disk condition that is still getting worse from the one
    observer watching the stream."""
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
    """The carrier's own contract. `critical_fired` is not a config proxy (R79) — there is no
    config value beside it; it is "my critical arm signalled this process", produced by the
    guard and read by the one composition root after `stop()` joins the thread.

    MUTATION THAT REDS IT: initialise it True (every composed run would then report rc 47),
    or make it a settable attribute the root could write — the fact would stop being the
    guard's."""
    guard = DiskGuard(watch_path=Path("."), interval_sec=60.0, warn_gb=4.0, fail_gb=2.0,
                      keep_all=False, sink=type("_S", (), {"emit": lambda self, p: None})())
    assert guard.critical_fired is False, "a guard that has not fired has not fired"
    with pytest.raises(AttributeError):
        guard.critical_fired = True     # type: ignore[misc]


# ══ the carrier and the row ═══════════════════════════════════════════════════════════
def test_record_abort_is_set_once_and_the_first_fire_wins() -> None:
    """`ShutdownState.record_abort` is THE writer of `abort_rule` for BOTH fire paths since
    R132 added the second one. Set-once is what stops two authorities disagreeing: a disk-full
    event during a draw-rate collapse must not re-label the collapse.

    MUTATION THAT REDS IT: a plain assignment (last writer wins) — the shape the field had
    when it was written at exactly one site and the invariant was prose."""
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
    """The row R132 mandates, and the one thing about it that can rot silently.

    REQUIRED and not DEFERRED because nothing is owed: `monitor.disk_guard.fail_gb` is a
    minted operator value on every committed config, its schema carries `gt=0`, and the block
    is a required field — so a validated `RunConfig` arms this row by construction. What the
    row is FOR is the drift: `_dotted` short-circuits a mid-walk `None` to DISARMED, so making
    the block nullable would go RED on run5 instead of the guard quietly disappearing again.

    MUTATION THAT REDS IT: rename the constant, delete the transfer line in `compose_run`, or
    move it above `disk_guard.stop()` — the last one is the subtle failure (the latch would be
    read before the guard thread is joined) and the pin's exact text is what catches it."""
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
