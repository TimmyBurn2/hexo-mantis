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

from pathlib import Path

import pytest

import mantis.run as mantis_run
from mantis.config.resolve.disk_guard import resolve_disk_guard
from _drivable import DrivablePoolStub, DrivableTrainerStub
from _root_recorders import bounded, install_recorders


class _PartialStartFailure(RuntimeError):
    """Module-private: "the ORIGINAL exception propagates" must be an IDENTITY claim."""


class _EvalPipelineWall(RuntimeError):
    """Module-private, same reason, at the other seam."""


class _PartiallyStartingPool(DrivablePoolStub):
    """The partial-start subject: `start()` brings a resource UP and then raises — `WorkerPool`'s
    real shape, where the inference server is live before the runner is asked to start."""

    def __init__(self) -> None:
        super().__init__(game_per_read=True)
        self.resource_live = False

    def start(self) -> None:
        self.resource_live = True          # sub-start #1 succeeded …
        raise _PartialStartFailure("the pool came up halfway and then refused")  # … #2 did not

    def stop(self) -> None:
        self.resource_live = False
        self.stopped = True


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
    rec = install_recorders(monkeypatch, request)
    pool = _PartiallyStartingPool()

    with pytest.raises(_PartialStartFailure) as wall:
        mantis_run.compose_run(
            config=bounded(smoke_run_config), trainer=DrivableTrainerStub(), pool=pool,
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
    rec = install_recorders(monkeypatch, request)

    def _raising_eval_pipeline(**_kwargs):
        raise _EvalPipelineWall("the eval pipeline refused this composition")

    monkeypatch.setattr(mantis_run, "build_eval_pipeline", _raising_eval_pipeline)
    pool = DrivablePoolStub(game_per_read=True)

    with pytest.raises(_EvalPipelineWall) as wall:
        mantis_run.compose_run(
            config=bounded(smoke_run_config, eval_enabled=True), trainer=DrivableTrainerStub(),
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
    rec = install_recorders(monkeypatch, request)
    config = bounded(smoke_run_config)
    expected = resolve_disk_guard(config.monitor)
    assert len({expected.interval_sec, expected.warn_gb, expected.fail_gb}) == 3, (
        "vacancy guard: this drive's three thresholds must be DISTINCT or a transposition "
        f"assertion cannot fail; got {expected}"
    )

    mantis_run.compose_run(
        config=config, trainer=DrivableTrainerStub(), pool=DrivablePoolStub(game_per_read=True),
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
