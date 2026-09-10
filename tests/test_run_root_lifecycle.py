# >300 justify (R8): every oracle below shares ONE real composed drive — real
# `build_run_safety`, `JsonlEventSink`, `HeartbeatWatchdog`, `DiskGuard` and signal handlers —
# and cross-test imports are barred, so splitting them would fork a second drivable harness AND
# a second real-subsystem boot per file.
"""LAW-16 at the composition root.

What this file exists to stop, measured and not inferred: signals were UNARMED in every composed
run, because `install_signal_handlers` fired only on the loop's self-construct branch while the
root always injects its own `ShutdownState` — a probe over 19 real drives found SIGINT at
`default_int_handler` and SIGTERM at `SIG_DFL` on all 19. The disk guard was never constructed,
and `resolve_config` / `to_event_payload` had zero production call sites.

Oracles fake nothing on these paths: `build_run_safety` is called FOR REAL and only wrapped to
record, `DiskGuard` is the REAL class subclassed to record kwargs, and the buffer is a real
`HexgBuffer`. Only the trainer, the pool and (in one oracle) `StepCoordinator` are substituted.
"""
from __future__ import annotations

import json
import signal
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

import mantis.run as mantis_run
from mantis.config.emit import resolve_config
from mantis.monitor.manifest import verify_manifest
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.monitor.manifest import DEFAULT_MANIFEST_PATH

_REPO = Path(__file__).resolve().parents[1]
#: The SHIPPED manifest, from its own module. Its path constant had zero references while three
#: test files rebuilt the path by hand, so deletion would have left the copies and no authority.
_MANIFEST = DEFAULT_MANIFEST_PATH

#: The bounded burst every drive runs; 3 is the smallest legal run at cadence 1.
_DRIVE_STEPS = 3

#: Disk-guard values: an interval short enough that the guard's thread emits inside a
#: sub-second burst, and thresholds low enough that it can NEVER fire on a real filesystem — a
#: critical alert SIGTERMs the pytest process.
_DRIVE_DISK_GUARD = {"interval_sec": 0.02, "warn_gb": 0.001, "fail_gb": 0.0005}

#: The producer-manifest row and the node it must name; both land in the SAME commit.
_RESOLVED_CONFIG_PRODUCER_TEST = (
    "tests/test_run_root_lifecycle.py::"
    "test_the_composed_boot_publishes_its_resolved_config_once_after_the_identity_witness"
)


@pytest.fixture(autouse=True)
def restore_signal_dispositions():
    """Save and restore the process-global SIGINT/SIGTERM handlers around each test, so one
    drive's handlers cannot decide another test's fate."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _Pool:
    """Drivable stand-in for `WorkerPool` at the injected seam; `start`/`stop` are recorded
    because the teardown ladder's contract is "pool stopped IFF started"."""

    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # the third outcome share.
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    class _RunnerStats:
        mcts_mean_depth = 5.0
        mcts_mean_root_concentration = 0.1
        cluster_value_std_mean = 0.0
        cluster_policy_disagreement_mean = 0.0
        cluster_variance_sample_count = 0

    def __init__(self, on_start=None) -> None:
        self.started = False
        self.stopped = False
        self._games = 0
        self._on_start = on_start
        self.recent_move_histories: list = []
        self.sync_payloads: list = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True
        if self._on_start is not None:
            self._on_start()

    def stop(self) -> None:
        if not self.started:
            raise RuntimeError("cannot join thread before it is started")
        self.stopped = True

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return self._RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    """Drivable stand-in for the trainer at the injected seam, conforming to the DECLARED
    train-step surface."""

    def __init__(self, on_step=None) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self.saves: list = []
        self._on_step = on_step
        # A REAL directory and a `save_checkpoint` returning a REAL path: both signal-save legs
        # write a resume sidecar BESIDE the checkpoint, so a fake returning None would leave the
        # leg that makes the stop resumable unmeasured.
        self.checkpoint_dir = Path(tempfile.mkdtemp(prefix="mantis-root-lifecycle-"))

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

    def save_checkpoint(self, loss_info) -> Path:
        self.saves.append(loss_info)
        path = self.checkpoint_dir / f"fake_{self.step:08d}_deadbeef.ckpt"
        path.write_bytes(b"fake-checkpoint")
        return path


class _RecordedDiskGuard(DiskGuard):
    """The REAL guard, with two observation points. Every behaviour is `super()`'s."""

    instances: list = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ctor_kwargs = dict(kwargs)
        self.stop_calls = 0
        type(self).instances.append(self)

    def stop(self) -> None:
        self.stop_calls += 1
        super().stop()


class _Recorders:
    """What the real subsystems did, observed without standing any of them in."""

    def __init__(self) -> None:
        self.run_safety: Any = None
        self.watchdog_stops = 0
        self.sink_closes = 0
        self.disk_guards: list[_RecordedDiskGuard] = []


def _install_recorders(monkeypatch, request) -> _Recorders:
    """Close the REAL sink after every drive: on the COMPLETED path `close_out` never touches
    it, which is bounded in production because both real callers exit the process right after,
    but this pytest process does not."""
    rec = _Recorders()
    real_build = mantis_run.build_run_safety
    _RecordedDiskGuard.instances = rec.disk_guards

    def _recording_build(**kwargs):
        run_safety = real_build(**kwargs)          # the REAL builder, unmodified
        rec.run_safety = run_safety
        real_stop, real_close = run_safety.watchdog.stop, run_safety.sink.close

        def _stop() -> None:
            rec.watchdog_stops += 1
            real_stop()

        def _close() -> None:
            rec.sink_closes += 1
            real_close()

        run_safety.watchdog.stop = _stop
        run_safety.sink.close = _close
        request.addfinalizer(run_safety.sink.close)
        return run_safety

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build)
    monkeypatch.setattr(mantis_run, "DiskGuard", _RecordedDiskGuard)
    return rec


def _bounded(smoke_run_config, **over):
    """A REAL minted graph config, bounded so the drive terminates, with eval OFF by the
    CONFIG's own value — no parameter can force it."""
    monitor = {"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
               "disk_guard": dict(_DRIVE_DISK_GUARD)}
    monitor.update(over.pop("monitor", {}))
    return smoke_run_config(
        "dev_example.yaml", eval_enabled=False,
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               "batch_size": 8},
        monitor=monitor, **over,
    )


def _events(run_safety) -> list[dict]:
    """The run's OWN event stream, read off the real sink's segment file."""
    return [json.loads(line) for line in
            Path(run_safety.sink.path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _installed(sig: int):
    """Assert the live disposition is a real handler BEFORE anything delivers a signal: a drive
    that sent SIGTERM at `SIG_DFL` would kill the test runner, so the RED state of this oracle
    must be an assertion failure, never a dead pytest."""
    handler = signal.getsignal(sig)
    assert handler not in (signal.SIG_DFL, signal.SIG_IGN) and callable(handler), (
        f"{signal.Signals(sig).name} is at {handler!r} during a composed run: LAW-16's "
        "save-then-exit is UNARMED, which is exactly the F-1 defect measured on 19 drives"
    )
    return handler


# O-D1 — the lifecycle contract at the root.
def test_the_signal_install_has_exactly_two_call_sites_and_one_of_them_is_the_root() -> None:
    """`install_signal_handlers` has exactly TWO call sites: the loop's self-construct branch and
    the composition root. It used to have one, on a branch nothing takes. A third site would be
    one silently disarmed shutdown path, since installation is process-global last-writer-wins."""
    import ast

    sites: set[str] = set()
    for path in sorted((_REPO / "src" / "mantis").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner: dict[Any, str] = {}

        def walk(node, name, owner=owner):
            for child in ast.iter_child_nodes(node):
                child_name = child.name if isinstance(
                    child, ast.FunctionDef | ast.AsyncFunctionDef) else name
                owner[child] = child_name
                walk(child, child_name)

        walk(tree, "<module>")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) \
                    == "install_signal_handlers":
                sites.add(f"{path.relative_to(_REPO)}::{owner.get(node, '<module>')}")
    assert sites == {"src/mantis/train/loop.py::run_training_loop",
                     "src/mantis/run.py::compose_run"}, (
        "LAW-16's handlers are installed at exactly two sites — the loop's self-construct "
        f"branch and the composition root; got {sorted(sites)}"
    )


def test_the_installed_handlers_are_bound_to_the_state_the_loop_actually_polls(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """The handlers close over the SAME `ShutdownState` the root injects, compared by IDENTITY:
    a root that installed over a different state passes every presence assertion and still never
    stops on a signal."""
    rec = _install_recorders(monkeypatch, request)
    captured: dict[str, Any] = {}

    def _capture(step: int) -> None:
        captured.setdefault("sigint", _installed(signal.SIGINT))
        captured.setdefault("sigterm", _installed(signal.SIGTERM))

    handles = mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_Trainer(on_step=_capture),
        pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert captured, "the drive never reached a step — nothing was observed"
    for name, handler in captured.items():
        cells = [cell.cell_contents for cell in (handler.__closure__ or ())]
        assert any(cell is handles.shutdown for cell in cells), (
            f"the installed {name} handler flips a DIFFERENT ShutdownState than the one the "
            f"loop polls; closure holds {cells!r}"
        )
    assert rec.run_safety is not None, "premise: the REAL build_run_safety ran"


def test_a_signal_mid_run_saves_then_exits(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """One REAL SIGTERM delivered from inside a step gives `running=False` + `shutdown_save=True`
    and the loop's final `trainer.save_checkpoint`. Handlers that set only `running=False` stop
    the run without its final checkpoint."""
    import os

    _install_recorders(monkeypatch, request)
    trainer = _Trainer()

    def _signal_at_first_step(step: int) -> None:
        if step == 1:
            _installed(signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)

    trainer._on_step = _signal_at_first_step
    handles = mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=trainer, pool=_Pool(),
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert handles.shutdown.shutdown_save is True, "one signal means SAVE then exit"
    assert handles.shutdown.running is False, "…and the loop must stop"
    assert trainer.saves, (
        "the final checkpoint is the save half of save-then-exit; without it a signalled "
        "run loses everything since the last interval save (LAW-16)"
    )
    assert trainer.step < _DRIVE_STEPS, (
        "the run stopped on the SIGNAL, not on its step ceiling — otherwise this test would "
        "pass with no handler installed at all"
    )


def test_a_second_signal_force_exits(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """The second press force-tears down all children then `os._exit(1)`. Driven by invoking the
    handler the ROOT installed, since a real second delivery would take pytest with it."""
    from mantis.train.lifecycle import signals as sig_mod
    import os
    teardown_called: list = []
    monkeypatch.setattr(sig_mod, "force_teardown_all", lambda: teardown_called.append(1))
    monkeypatch.setattr(os, "_exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    _install_recorders(monkeypatch, request)
    mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_Trainer(), pool=_Pool(),
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    handler = _installed(signal.SIGTERM)
    handler(signal.SIGTERM, None)
    with pytest.raises(SystemExit) as exit_info:
        handler(signal.SIGTERM, None)
    assert exit_info.value.code == 1, f"the second signal force-exits 1; got {exit_info.value.code!r}"
    assert teardown_called, "second signal must force-teardown children before os._exit"


def test_the_watchdog_and_the_disk_guard_are_both_armed_at_boot(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """The watchdog and the disk guard are both armed AND accounted for at teardown, asserted
    from the run's OWN stream — a guard thread outliving its run cannot pass as green."""
    rec = _install_recorders(monkeypatch, request)
    handles = mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=_Trainer(on_step=lambda _s: _sleep_a_beat()),
        pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    names = [event.get("event") for event in _events(handles.run_safety)]
    assert "heartbeat_watchdog_armed" in names, (
        f"the stall watchdog must be ARMED at boot, unconditionally (LAW-16); saw {set(names)}"
    )
    assert rec.disk_guards, (
        "the composition root must CONSTRUCT the disk guard — at HEAD `build_subsystems` is "
        "its only construction site and has zero callers (F-2-DISKGUARD, R121(b))"
    )
    guard = rec.disk_guards[-1]
    assert guard.ctor_kwargs["watch_path"] == Path(tmp_path / "ckpt"), (
        "the guard watches the CHECKPOINT dir — the volume the run actually fills"
    )
    assert "disk_free" in names, (
        f"an armed guard emits into the run's own stream (LAW-18); saw {sorted(set(names))}"
    )
    assert guard.stop_calls >= 1, (
        "the guard's thread is stopped on the way out; a daemon thread that outlives its "
        "run is a leak the teardown ladder exists to prevent (DESIGN §8)"
    )


def test_a_signal_delivered_during_composition_completes_the_boot_then_saves(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """A signal during `pool.start()` completes composition and returns with ZERO steps.

    Nothing between the handler install and the loop polls the state, so the loop's entry-set arm
    saves and returns and `close_out` drains. A pre-start bail-out was argued against as a
    rarely-exercised branch in the one composer, so the window is bounded and PINNED instead.
    """
    _install_recorders(monkeypatch, request)
    trainer = _Trainer()

    def _signal_during_pool_start() -> None:
        handler = _installed(signal.SIGTERM)
        handler(signal.SIGTERM, None)

    pool = _Pool(on_start=_signal_during_pool_start)
    handles = mantis_run.compose_run(
        config=_bounded(smoke_run_config), trainer=trainer, pool=pool,
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    assert pool.started is True, "composition COMPLETES through the window (§7 leg 1)"
    assert trainer.step == 0, "the entry-set arm saves before any step runs"
    assert trainer.saves, "…and it SAVES (loop.py:91-94) rather than returning empty-handed"
    assert handles.shutdown.shutdown_save is True and handles.shutdown.running is False
    assert pool.stopped is True, (
        "no worker survives the signalled boot — `close_out`'s guarded stop is the contract "
        "DESIGN §8 states as 'nothing is half-alive'"
    )


# O-D2 — partial composition leaks nothing.
class _CoordinatorSeamFailure(RuntimeError):
    """A distinctive failure at the coordinator seam, so "the ORIGINAL exception propagates" is
    an identity claim rather than a family claim."""


def test_a_failure_at_the_coordinator_seam_tears_everything_down_and_re_raises(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """Builder N succeeds, builder N+1 raises: no worker process and no non-daemon thread
    survives, and the ORIGINAL failure propagates.

    A bare propagate would leave a watchdog whose `exit_fn` is `os._exit` and a guard that will
    SIGTERM a process no longer running a run. Chaining as `__context__` rather than replacing
    the original is asserted by TYPE.
    """
    rec = _install_recorders(monkeypatch, request)

    def _raising_coordinator(**_kwargs):
        raise _CoordinatorSeamFailure("the coordinator seam refused this composition")

    monkeypatch.setattr(mantis_run, "StepCoordinator", _raising_coordinator)
    pool = _Pool()
    with pytest.raises(_CoordinatorSeamFailure):
        mantis_run.compose_run(
            config=_bounded(smoke_run_config), trainer=_Trainer(), pool=pool,
            buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
        )
    assert pool.started and pool.stopped, (
        "the pool WAS started, so it must be stopped — `pool.stop()` iff `pool_started` is "
        "the item-11 closure's own rule, and both halves are live on this path"
    )
    assert rec.watchdog_stops >= 1, "the watchdog poll thread must not outlive a failed compose"
    assert rec.disk_guards and rec.disk_guards[-1].stop_calls >= 1, (
        "…nor the disk-guard thread, whose critical arm SIGTERMs the process"
    )
    assert rec.sink_closes >= 1, "…and the event sink is closed, not left holding the segment"


# O-E3 — the resolved-config producer.
def test_the_composed_boot_publishes_its_resolved_config_once_after_the_identity_witness(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """`resolve_config` + `to_event_payload` are emitted exactly once, after the identity
    witness. Ordering is asserted, not just presence: `run_boot_identity` must land FIRST,
    because it has to exist even if the boot later wedges."""
    _install_recorders(monkeypatch, request)
    config = _bounded(smoke_run_config)
    handles = mantis_run.compose_run(
        config=config, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    events = _events(handles.run_safety)
    names = [event.get("event") for event in events]
    assert names.count("resolved_config") == 1, (
        f"exactly one resolved_config per run segment; got {names.count('resolved_config')}"
    )
    assert names.index("resolved_config") == names.index("run_boot_identity") + 1, (
        f"the resolved posture follows the identity witness immediately; got {names[:6]}"
    )
    published = events[names.index("resolved_config")]
    expected = resolve_config(config).to_event_payload()
    assert {key: published[key] for key in expected} == expected, (
        "the published payload must BE `to_event_payload(resolve_config(config))` — a "
        "hand-assembled copy is a second authority for the run's resolved posture"
    )


def test_the_shipped_manifest_claims_the_resolved_config_producer_and_names_this_test() -> None:
    """The manifest row claiming the event exists and names a producer test that does — a gate
    input with no producer row is the phantom-gate class."""
    verify_manifest(_MANIFEST, _REPO)  # must not raise
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    rows = [row for row in manifest["gates"]
            if row.get("id") == "resolved_config" or row.get("input") == "resolved_config"]
    assert len(rows) == 1, (
        f"exactly one shipped manifest row claims resolved_config; got {rows}"
    )
    producer = rows[0]["producer"]
    assert producer.get("module") == "mantis.run" and producer.get("symbol") == "compose_run", (
        f"the producer is the composition root itself (§5.4); got {producer}"
    )
    assert rows[0]["producer_test"] == _RESOLVED_CONFIG_PRODUCER_TEST, (
        f"the row must name the drive above; got {rows[0]['producer_test']!r}"
    )


def _sleep_a_beat() -> None:
    """A per-step pause long enough that the disk guard's thread gets several ticks inside a
    3-step burst: ~20 expected emissions against an assertion of >= 1."""
    import time

    time.sleep(0.15)
