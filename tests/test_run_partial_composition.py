# >300 justify (R8). The three oracles are
# one claim — "a composition that fails partway leaks nothing and says WHERE" — driven
# through ONE real composed boot (real `build_run_safety`, real `JsonlEventSink`, real
# `HeartbeatWatchdog`, real `DiskGuard`, real signal handlers). R5 bars cross-test imports
# and `tests/test_run_root_lifecycle.py` is BYTE-FROZEN (`7c28536`), so the drivable
# pool/trainer harness cannot be shared from either side; splitting these three would fork it
# a second time instead of once. One harness, one boot shape, three RED-TEAM findings that
# are all "what DESIGN §8 owes at the seams O-D2 does not reach".
"""⊕ WPMAIN — the teardown ladder's own boundary conditions (RED-TEAM RT-3 / RT-4 / RT-7).

`tests/test_run_root_lifecycle.py`'s O-D2 pins the LAST seam — `StepCoordinator`, where the
pool, the watchdog and the guard are all already up. RED-TEAM drove the seams BEFORE it and
found the ladder does not reach them:

- **RT-3 — the `pool_started` window.** `WorkerPool.start()` (`selfplay/pool.py:308-329`) is
  three sub-starts: the inference server, the Rust runner, then the stats thread. The flag
  was set AFTER the call returned, so a raise in sub-start #2 or #3 left #1 alive with the
  flag still `False` — and `_stop_pool_if_started(pool, pool_started=False)()` is a NO-OP.
  Driven: `partial_resource_still_live: TRUE`, `pool.stopped: false`. Silent worker leak.
- **RT-4 — the gap above the ladder.** `build_run_safety` OPENS the run's JSONL segment; the
  five construction steps between it and the old `try:` were outside both the ladder and (bar
  two) any seam. Driven at `build_eval_pipeline` on an eval-enabled minted config:
  `leaked_open_files: ["…/logs/events_smoke_gnn_seg0001.jsonl"]`, `notes: []` — the largest
  construction step in the composer, the one every `eval_enabled: true` config walks into,
  reaching the process boundary with the sink open and NO seam name for the preflight's
  rc-32/33 classifier to read.
- **RT-7 — the guard threading nobody reads.** Transposing `warn_gb`/`fail_gb` at the
  construction site (`run.py`'s six-line hand-off) was **FULL TIER GREEN: 2278 passed, 2
  skipped**. `tests/config/test_disk_guard_keys.py:166-171` names that exact defect — *"a
  transposed `warn_gb`/`fail_gb` is a guard that kills the run at the warning threshold"* —
  and pins it AT THE RESOLVER, where three independent values go in and three named fields
  come out and the transposition cannot happen. The place it CAN happen is the hand-off, and
  `DiskGuardConfig`'s `fail_gb < warn_gb` model rule guards the CONFIG layer only. Under the
  mutation a run5 minted 10/5 SIGTERMs itself at 10 GB free and never warns.

Fakes, disclosed in full (R121(b): oracles fake nothing on the asserted path):

- `trainer` / `pool` are drivable COLLABORATORS, injected by `compose_run`'s own pinned
  contract (Q-INJECTION) — the same posture every wiring oracle in `tests/` uses. The buffer
  is the REAL `HexgBuffer` (the graph route refuses a shapeless fake at dispatch, by design).
- `build_run_safety` is called FOR REAL; the wrapper only records the real object it returned
  and installs recording delegates over `watchdog.stop` / `sink.close`.
- `DiskGuard` is the REAL class, subclassed to record its construction kwargs; `super()` on
  every path.
- The RT-3 pool's `start()` and the RT-4 `build_eval_pipeline` raiser ARE the subjects: there
  is no other way to make a partial start or an eval-pipeline wall happen.
"""
from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

import mantis.run as mantis_run
from mantis.config.resolve.disk_guard import resolve_disk_guard
from mantis.train.lifecycle.disk_guard import DiskGuard

#: The bounded burst every drive runs; 3 is the smallest legal run at cadence 1 (the
#: reachability validator spans cadence < actor_lag_threshold < max_train_steps).
_DRIVE_STEPS = 3

#: Disk-guard values for the drives. Deliberately three DISTINCT numbers: an assertion that
#: the guard received the resolver's values is vacuous if two of them are equal, and the
#: transposition RT-7 found is exactly a swap of two. Low enough that the critical arm can
#: NEVER fire on a real filesystem (it SIGTERMs the pytest process).
_DRIVE_DISK_GUARD = {"interval_sec": 0.02, "warn_gb": 0.001, "fail_gb": 0.0005}


@pytest.fixture(autouse=True)
def restore_signal_dispositions():
    """Every drive here installs process-global SIGINT/SIGTERM handlers (the root does, by
    design). Save and restore around each test so one drive's handlers cannot decide
    another test's fate."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _PartialStartFailure(RuntimeError):
    """Module-private: "the ORIGINAL exception propagates" must be an IDENTITY claim."""


class _EvalPipelineWall(RuntimeError):
    """Module-private, same reason, at the other seam."""


class _Pool:
    """Drivable stand-in for `WorkerPool` at the injected seam."""

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
        self.stopped = False
        self._games = 0
        self.recent_move_histories: list = []
        self.sync_payloads: list = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
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


class _PartiallyStartingPool(_Pool):
    """The RT-3 subject: `start()` brings a resource UP and then raises — `WorkerPool`'s
    real shape, where the inference server is live before the runner is asked to start."""

    def __init__(self) -> None:
        super().__init__()
        self.resource_live = False

    def start(self) -> None:
        self.resource_live = True          # sub-start #1 succeeded …
        raise _PartialStartFailure("the pool came up halfway and then refused")  # … #2 did not

    def stop(self) -> None:
        self.resource_live = False
        self.stopped = True


class _Trainer:
    """Drivable stand-in for the trainer at the injected seam, conforming to the DECLARED
    train-step surface (`train_step_from_graph_batch` / `_from_tensors`, R102)."""

    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self.saves: list = []

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return {}

    def save_checkpoint(self, loss_info) -> None:
        self.saves.append(loss_info)


class _RecordedDiskGuard(DiskGuard):
    """The REAL guard with one observation point; every behaviour is `super()`'s."""

    instances: list = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ctor_kwargs = dict(kwargs)
        type(self).instances.append(self)


class _Recorders:
    def __init__(self) -> None:
        self.run_safety: Any = None
        self.watchdog_stops = 0
        self.sink_closes = 0
        self.disk_guards: list[_RecordedDiskGuard] = []


def _install_recorders(monkeypatch) -> _Recorders:
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
        return run_safety

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build)
    monkeypatch.setattr(mantis_run, "DiskGuard", _RecordedDiskGuard)
    return rec


def _bounded(smoke_run_config, **over):
    """A REAL minted graph config, bounded so the drive terminates. `eval_enabled` is the
    CONFIG's own value (R120: no parameter can force it, so the config is the only route)."""
    monitor = {"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
               "disk_guard": dict(_DRIVE_DISK_GUARD)}
    monitor.update(over.pop("monitor", {}))
    over.setdefault("eval_enabled", False)
    return smoke_run_config(
        "smoke_gnn.yaml",
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               "batch_size": 8},
        monitor=monitor, **over,
    )


def _seam_names(exc: BaseException) -> list[str]:
    """The PEP 678 notes `_seam` attaches, as bare seam names."""
    prefix = "composition seam: "
    return [note[len(prefix):] for note in getattr(exc, "__notes__", [])
            if note.startswith(prefix)]


# ══ RT-3 — a partial `pool.start()` is still torn down ════════════════════════════════
def test_a_pool_that_comes_up_halfway_and_then_raises_is_still_stopped(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """RT-3. The ladder's own boundary condition, which O-D2 (the coordinator seam, where
    everything is already up) cannot reach.

    The contract DESIGN §8 states — *"no worker process and no non-daemon thread survives a
    failed compose, nothing is half-alive"* — is asserted against the pool's OWN resource
    flag, not against the fact that `stop()` was called: a `stop()` that runs on a pool the
    ladder believes never started is the no-op this finding is about.

    MUTATION THAT REDS IT (driven, RED-TEAM probe B1): restore `pool_started = True` to its
    position AFTER `pool.start()`. `partial_resource_still_live` goes TRUE and `pool.stopped`
    goes False, with every other oracle in the tree green — including O-D2, whose pool starts
    cleanly.

    The other three halves of the ladder are asserted here too, because a widening of the
    flag that dropped one of them would be a different leak at the same seam."""
    rec = _install_recorders(monkeypatch)
    pool = _PartiallyStartingPool()

    with pytest.raises(_PartialStartFailure) as wall:
        mantis_run.compose_run(
            config=_bounded(smoke_run_config), trainer=_Trainer(), pool=pool,
            buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
        )

    assert pool.resource_live is False, (
        "the pool brought a resource UP and then raised; the teardown ladder left it live. "
        "`pool_started` set AFTER `pool.start()` reports a half-started pool as never "
        "started, so the guarded stop is a no-op and the workers leak (RED-TEAM RT-3)"
    )
    assert pool.stopped is True, "…and the stop must actually have been the ladder's"
    assert rec.watchdog_stops >= 1, "the watchdog must not outlive a failed compose"
    assert rec.sink_closes >= 1, "…nor may the event sink be left holding the segment"
    assert _seam_names(wall.value) == ["pool.start"], (
        "the wall must name its seam and only its seam (DESIGN §8); got "
        f"{_seam_names(wall.value)}"
    )


# ══ RT-4 — the eval-pipeline wall is inside the ladder AND named ══════════════════════
def test_an_eval_pipeline_wall_names_its_seam_and_closes_the_sink(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """RT-4. `build_eval_pipeline` is the largest construction step in the composer and the
    one every `eval_enabled: true` config walks into — and it used to sit ABOVE the `try:`,
    outside the ladder, unseamed.

    Two independent assertions, because the finding is two defects at one site: the segment
    file the run's own sink opened was left OPEN (measured by RED-TEAM's probe B2 as a live
    file descriptor), and the failure reached the process boundary with `notes: []`, so the
    preflight's rc-32/33 classifier reads a stderr tail with no seam in it.

    MUTATION THAT REDS IT: move the `try:` back down to `pool.start()` (the sink-close
    recorder stays at 0), or drop the `with _seam("build_eval_pipeline"):` (the note list
    empties). Both are invisible to every other oracle: this is the only drive in the tree
    that fails at this seam.

    The pool assertion is the third leg — a wall ABOVE `pool.start()` must NOT call
    `pool.stop()`, which is the ORIGINAL hazard `_stop_pool_if_start_attempted` guards
    (`InferenceServer.join(timeout=5.0)` raises on a never-started thread). RT-3's widening
    kept it closed and this row is what says so."""
    rec = _install_recorders(monkeypatch)

    def _raising_eval_pipeline(**_kwargs):
        raise _EvalPipelineWall("the eval pipeline refused this composition")

    monkeypatch.setattr(mantis_run, "build_eval_pipeline", _raising_eval_pipeline)
    pool = _Pool()

    with pytest.raises(_EvalPipelineWall) as wall:
        mantis_run.compose_run(
            config=_bounded(smoke_run_config, eval_enabled=True), trainer=_Trainer(),
            pool=pool, buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
        )

    assert _seam_names(wall.value) == ["build_eval_pipeline"], (
        "an eval-pipeline wall must NAME its seam: DESIGN §8's contract is that a config "
        "which validates but cannot compose fails loud with the seam in the message, and "
        f"the preflight's rc-32/33 classifier reads that tail; got {_seam_names(wall.value)}"
    )
    assert rec.sink_closes >= 1, (
        "the run's JSONL segment was opened by `build_run_safety` and left OPEN by the "
        "wall: the teardown ladder must START at the sink, not at `pool.start()` (RT-4)"
    )
    assert rec.watchdog_stops >= 1, "the watchdog build is in the same ladder"
    assert pool.started is False and pool.stopped is False, (
        "a wall ABOVE the start must not call `pool.stop()` on a pool this run never "
        "touched — the never-started `join` hazard the closure exists to guard"
    )
    assert rec.disk_guards == [], "…and nothing downstream of the wall may have been armed"


# ══ RT-7 — the guard gets the resolver's OWN values, in the resolver's OWN slots ══════
def test_the_disk_guard_receives_exactly_what_its_resolver_resolved(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """RT-7 — the DR-11 question at the END of the path nobody reads.

    `tests/config/test_disk_guard_keys.py` closes the resolver end: the `data.pop
    ("disk_guard")` in `resolve_monitor_config` is legitimate BECAUSE another reader exists,
    and the generalised-pop form is banned by source census. What no test read is the
    six-line hand-off from that reader to `DiskGuard(...)` — three floats, by keyword, in
    the one place a transposition is typeable.

    MUTATION THAT REDS IT (driven, RED-TEAM M6): swap the two kwargs at the construction
    site — `warn_gb=guard_spec.fail_gb, fail_gb=guard_spec.warn_gb`. FULL TIER GREEN before
    this row: 2278 passed, 2 skipped, and 44/44 on the four focused suites, because
    `DiskGuardConfig`'s `fail_gb < warn_gb` model rule constrains the CONFIG's own leaves and
    nothing constrains what reaches the guard. Under it a run5 minted 10/5 SIGTERMs itself at
    10 GB free and never warns once.

    Asserted against the RESOLVER's output, never against the literals this drive minted: a
    row that restates the test's own input is the self-satisfying species, and it would go
    green again the moment the resolver started lying."""
    rec = _install_recorders(monkeypatch)
    config = _bounded(smoke_run_config)
    expected = resolve_disk_guard(config.monitor)
    assert len({expected.interval_sec, expected.warn_gb, expected.fail_gb}) == 3, (
        "vacancy guard: this drive's three thresholds must be DISTINCT or a transposition "
        f"assertion cannot fail; got {expected}"
    )

    mantis_run.compose_run(
        config=config, trainer=_Trainer(), pool=_Pool(),
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert rec.disk_guards, "premise: the root constructed the guard (LAW-16 leg 3)"
    kwargs = rec.disk_guards[-1].ctor_kwargs
    assert kwargs["warn_gb"] == expected.warn_gb, (
        "the guard's WARN threshold is not the resolved `monitor.disk_guard.warn_gb`. A "
        "transposed pair is a guard that kills the run at the warning threshold and never "
        f"warns — the defect test_disk_guard_keys.py names by name; got {kwargs}"
    )
    assert kwargs["fail_gb"] == expected.fail_gb, (
        f"…and the CRITICAL threshold is not the resolved `fail_gb`; got {kwargs}"
    )
    assert kwargs["interval_sec"] == expected.interval_sec, (
        f"…nor is the poll cadence the resolved `interval_sec`; got {kwargs}"
    )
    assert kwargs["watch_path"] == Path(tmp_path / "ckpt"), (
        "the guard watches the CHECKPOINT dir — the volume the run actually fills"
    )
