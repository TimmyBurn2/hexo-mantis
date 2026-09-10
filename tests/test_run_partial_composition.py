# >300 justify (R8): the three oracles are one claim — "a composition that fails partway leaks
# nothing and says WHERE" — driven through ONE real composed boot. Cross-test imports are barred
# and the sibling lifecycle file is BYTE-FROZEN, so splitting these would fork the drivable
# pool/trainer harness a second time instead of once.
"""The teardown ladder's own boundary conditions — the seams BEFORE the coordinator, where the
ladder did not reach.

The `pool_started` flag was set AFTER `start()` returned, so a raise in its second or third
sub-start left the first alive with the flag `False` and the conditional stop a NO-OP. The five
construction steps between `build_run_safety` — which OPENS the run's JSONL segment — and the
old `try:` were outside both the ladder and any seam. And transposing `warn_gb`/`fail_gb` at the
guard construction site was FULL TIER GREEN, because the resolver test pins the transposition
where it cannot happen and the config model rule guards the CONFIG layer only.

Fakes, disclosed: `trainer` and `pool` are drivable collaborators injected through the composer's
own pinned contract, the buffer is a REAL `HexgBuffer`, `build_run_safety` is called FOR REAL and
only wrapped to record, and `DiskGuard` is the REAL class subclassed to record kwargs.
"""
from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

import mantis.run as mantis_run
from mantis.config.resolve.disk_guard import resolve_disk_guard
from mantis.train.lifecycle.disk_guard import DiskGuard

#: The bounded burst every drive runs; 3 is the smallest legal run at cadence 1.
_DRIVE_STEPS = 3

#: Disk-guard values, deliberately three DISTINCT numbers: an assertion that the guard received
#: the resolver's values is vacuous if two are equal, and the transposition is exactly a swap of
#: two. Low enough that the critical arm can NEVER fire on a real filesystem.
_DRIVE_DISK_GUARD = {"interval_sec": 0.02, "warn_gb": 0.001, "fail_gb": 0.0005}


@pytest.fixture(autouse=True)
def restore_signal_dispositions():
    """Save and restore the process-global SIGINT/SIGTERM handlers around each test, so one
    drive's handlers cannot decide another test's fate."""
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
    """The partial-start subject: `start()` brings a resource UP and then raises — `WorkerPool`'s
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


def _install_recorders(monkeypatch, request) -> _Recorders:
    """Close the REAL sink after every drive: on the COMPLETED path `close_out` never touches it,
    which is bounded in production because both real callers exit the process right after."""
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
    """A REAL minted graph config, bounded so the drive terminates; `eval_enabled` is the
    CONFIG's own value, no parameter being able to force it."""
    monitor = {"actor_lag_threshold_steps": _DRIVE_STEPS - 1,
               "disk_guard": dict(_DRIVE_DISK_GUARD)}
    monitor.update(over.pop("monitor", {}))
    over.setdefault("eval_enabled", False)
    return smoke_run_config(
        "dev_example.yaml",
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               "batch_size": 8},
        monitor=monitor, **over,
    )


def _seam_names(exc: BaseException) -> list[str]:
    """The PEP 678 notes `_seam` attaches, as bare seam names."""
    prefix = "composition seam: "
    return [note[len(prefix):] for note in getattr(exc, "__notes__", [])
            if note.startswith(prefix)]


# A partial `pool.start()` is still torn down.
def test_a_pool_that_comes_up_halfway_and_then_raises_is_still_stopped(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """A pool that came up halfway is stopped, asserted against the pool's OWN resource flag
    rather than the fact that `stop()` was called — a `stop()` on a pool the ladder believes
    never started is the no-op this is about. Mutation: put `pool_started = True` back AFTER
    `pool.start()`. The other three halves of the ladder are asserted here too."""
    rec = _install_recorders(monkeypatch, request)
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


# The eval-pipeline wall is inside the ladder AND named.
def test_an_eval_pipeline_wall_names_its_seam_and_closes_the_sink(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """`build_eval_pipeline` fails inside the ladder, with its seam named.

    Two assertions, because the finding is two defects at one site: the segment the run's own
    sink opened was left OPEN, and the failure reached the process boundary with `notes: []`. The
    third leg is that a wall ABOVE `pool.start()` must NOT call `pool.stop()`.
    """
    rec = _install_recorders(monkeypatch, request)

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


# The guard gets the resolver's OWN values, in the resolver's OWN slots.
def test_the_disk_guard_receives_exactly_what_its_resolver_resolved(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """The hand-off from the resolver to `DiskGuard(...)` — three floats, by keyword, in the one
    place a transposition is typeable — passes the resolver's values into its own slots.

    Swapping the two kwargs was FULL TIER GREEN, because the config model rule constrains the
    CONFIG's leaves and nothing constrained what reached the guard. Asserted against the
    RESOLVER's output, never the literals this drive minted.
    """
    rec = _install_recorders(monkeypatch, request)
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
