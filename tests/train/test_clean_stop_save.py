# >300 justify (R8): the ten rows are ONE claim — the third save leg: a run that reaches its own
# declared terminus writes EXACTLY ONE stamped checkpoint, a rigged failure is
# supervisor-distinguishable, and an in-loop abort still writes nothing — over one seam running
# from `step.py`'s O2 arm through a latch on the coordinator to `loop.py`'s post-loop guard and on
# to the registered persist-fatal chain. R5 bars cross-test imports, so the local `StepCoordinator`
# harness must live here; a split forks it into copies that drift while both stay green.
"""The third save leg: a run that reaches `stop_step` writes exactly one final checkpoint.

Before this leg, a run reaching `stop_step` exited 0 with `checkpoints/` EMPTY — the O2 arm set
`shutdown.running = False` and returned without saving, `loop.py`'s `_final_save()` fires only on
`shutdown_save` (which only a SIGNAL sets), and the periodic arm is guarded by `interval > 0`
against a `checkpoint_interval` every minted config mints at 0.

Leg 3 is its OWN semantic: leg 1 means "a resumption point", leg 2 "we were interrupted", leg 3
"the run finished". Exactly-once comes from the DRIVER, not from any latch inside the leg, so the
real `run_training_loop` is the only honest instrument for it.

Real here: the `StepCoordinator`, its config from the PRODUCTION builder off a MINTED config, the
`ShutdownState`, `run_training_loop`, `_fire_hard_abort`, the draw-rate gate. Fake: the TRAINER (a
counter — the subject is control flow), the pool/buffer collaborators, and the sink.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

import dataclasses
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import HEARTBEAT_SOURCES, PERSIST_FATAL_EXIT_CODE, HeartbeatRegistry
from mantis.run import _step_coordinator_config, launch_run
from mantis.train import checkpoints
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog
from mantis.train.lifecycle.signals import ShutdownState
from mantis.train.loop import run_training_loop

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through (R5 bars cross-test imports, so each
    file that needs one builds it)."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



#: The declaration a `StepCoordinator` reads on the graph route: the identity it dispatches on
#: plus the two sections the route's own resolvers read. The caps are the NON-BINDING pair.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


_REPO = Path(__file__).resolve().parents[2]
_MINTED = load_config(_REPO / "configs" / "dev_example.yaml")

#: The coordinator's non-leg knobs come from a MINTED config through the production resolvers,
#: never from literals here (R1).
_DRAIN_CAPS = resolve_drain_caps(_MINTED.monitor)
_KNOBS = resolve_coordinator_knobs(_MINTED.train)
#: The ARMING cadence, from the same minted config. Harnesses that set `log_interval` MIRROR it
#: onto `gate_interval`, which is the shipped posture — every committed config mints them equal.
_GATE_INTERVAL = _MINTED.monitor.gate_interval


def _mirrored(settings: dict) -> dict:
    """Mirror the GATE cadence onto the NARRATION cadence unless a drive names it.

    That mirroring is the SHIPPED posture: every committed config mints `monitor.gate_interval`
    equal to its own `train.log_interval`."""
    settings.setdefault("gate_interval", settings["log_interval"])
    return settings

#: The declared terminus for the leg-3 drives.
_CEILING = 5
#: The trainer's step at O2 entry, DELIBERATELY DISTINCT from `_CEILING`: the `clean_stop_save`
#: event carries BOTH `step` and `stop_step`, and equal values would let an implementation that
#: emits one number twice — or swaps the two fields — pass forever.
_RESUMED_STEP = 7
#: The CANONICAL shape — the run arriving exactly at its ceiling — so the file covers both `>`
#: and `==` on the O2 predicate.
_LOOP_MAX_STEPS = 5
#: Far above any step the abort drive reaches, so its abort is unambiguously BELOW the terminus.
_ABORT_CEILING = 1000
_ABORT_RULE = "draw_rate_collapse"

#: What the fake trainer's `save_checkpoint` hands back. The leg must publish THE WRITER'S return
#: value on the event, not a path it re-derives from the checkpoint dir.
_SAVED_PATH = Path("/checkpoints/oracle_00000007_deadbeef.ckpt")

#: The armed smoke's OWN minted `train.max_train_steps`. Asserted as a PREMISE, never used as a
#: drive: measured at 474.6 s against the 300 s tier ceiling.
_SMOKE_CONFIG = "smoke_preflight_armed.yaml"
_MINTED_BOUND = 200
#: The drive bound, fixed by a PRE-REGISTERED measurement and its binding decision rule — the
#: largest member of {200, 100, 50, 32, 16} measuring <= 300 s on the dev box. Measured, single
#: run each, with the leg absent: 200 -> 474.6 s, 100 -> 319.5 s, 50 -> 240.2 s.
_OC7_BOUND = 50


class _Pool:
    """A pool whose draw counts are the only thing the draw-rate gate reads off it."""

    def __init__(self, *, draws: int = 0, completed: int = 0) -> None:
        self.games_completed = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
        self.draws = draws
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self._counts = (int(draws), int(completed))

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return self._counts

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return SimpleNamespace(mcts_mean_depth=5.0, mcts_mean_root_concentration=0.1,
                               cluster_value_std_mean=0.0,
                               cluster_policy_disagreement_mean=0.0,
                               cluster_variance_sample_count=0)

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    """The instrument. `attempts` and `saves` are DISTINCT on purpose: a rigged failure must show
    that the leg REACHED the writer (`attempts == 1`) and that no artefact resulted (`saves == 0`).
    One counter could not tell "the leg never called" from "the call failed"."""

    def __init__(self, *, step: int = 0, raises: BaseException | None = None,
                 on_save: Any = None) -> None:
        self.step = step
        self.model = object()
        self.device = "cpu"
        self.attempts = 0
        self.saves = 0
        self._raises = raises
        self._on_save = on_save

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def save_checkpoint(self, loss_info) -> Path:
        self.attempts += 1
        if self._on_save is not None:
            self._on_save()
        if self._raises is not None:
            raise self._raises
        self.saves += 1
        return _SAVED_PATH


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # DELEGATED to a real `HexgBuffer` rather than faked: the dispatcher collates the wire for
        # real before the trainer stub sees it, so a hand-built payload would be a second format.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder — this file's deltas only. `None` is the EXPLICIT
    disarmed draw-rate posture; neither the builder nor this factory gives it a default (R1)."""
    return dataclasses.replace(
        _step_coordinator_config(stop_step=_CEILING, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **_mirrored({"eval_interval": 0, "log_interval": 1, "min_buf_size": 10,
                     **overrides}),
    )


def _harness(*, trainer: _Trainer, config: StepCoordinatorConfig, pool: _Pool | None = None,
             shutdown: ShutdownState | None = None) -> SimpleNamespace:
    pool = pool if pool is not None else _Pool()
    shutdown = shutdown if shutdown is not None else ShutdownState()
    sink = _Sink()
    coord = StepCoordinator(
        trainer=trainer, buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None,
        config=config,
        full_config=_GRAPH_FULL_CONFIG,
        train_cfg={}, mixing_cfg={}, sink=sink, heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, shutdown=shutdown, sink=sink)


def test_the_ceiling_arm_saves_once_and_publishes_the_clean_stop_leg() -> None:
    """The leg fires, once, and publishes what it did, on ONE real `StepCoordinator.step()`.

    `abort_rule is None` is asserted because a clean stop and an abort both write
    `running = False`, and `abort_rule` is the ONLY thing that tells them apart. `checkpoint_saved`
    is asserted because NOTHING in `src/` reads it. The event's two step fields are read against
    DISTINCT numbers, so emitting one twice, or swapping them, reds here.
    """
    trainer = _Trainer(step=_RESUMED_STEP)
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING))
    assert h.shutdown.running is True and trainer.attempts == 0, (
        "premise: a fresh ShutdownState is born running and nothing has saved yet — without "
        "this the assertions below could be true of a coordinator that never ran"
    )

    outcome = h.coord.step()

    assert trainer.saves == 1, (
        "R137 leg (a): reaching `stop_step` is the run's own declared terminus and MUST write "
        f"the final checkpoint through the ONE writer; got {trainer.saves} saves"
    )
    assert h.shutdown.running is False, "O2 still stops the run — unchanged behaviour"
    assert h.shutdown.abort_rule is None, (
        "a run that COMPLETED is not a run that aborted; leg 3 neither reads nor writes "
        f"`abort_rule` (DESIGN_CS §2.5.2); got {h.shutdown.abort_rule!r}"
    )
    assert h.coord.clean_stop_saved is True, (
        "the latch the loop-side guard reads to keep leg 2 from writing a SECOND final "
        "artefact at the same step (W-1, DESIGN_CS §2.4)"
    )
    assert outcome.checkpoint_saved is True, (
        "parity with the O3 arm (`step.py:368`). Nothing in `src/` reads this field, which is "
        "why it is pinned here and nowhere else"
    )
    events = h.sink.named("clean_stop_save")
    assert len(events) == 1, (
        f"exactly ONE clean_stop_save event (LAW-18: 'did the run's final save happen, and to "
        f"what file' must be answerable from the ONE channel); got {events}"
    )
    assert events[0]["step"] == _RESUMED_STEP, (
        f"the event's `step` is the coordinator's own train step ({_RESUMED_STEP}), not the "
        f"ceiling; got {events[0]['step']!r}"
    )
    assert events[0]["stop_step"] == _CEILING, (
        f"…and `stop_step` is the DECLARED terminus ({_CEILING}), a different number; got "
        f"{events[0]['stop_step']!r}"
    )
    assert events[0]["path"] == str(_SAVED_PATH), (
        "the event publishes THE WRITER'S returned path — a re-derived directory string would "
        f"name a file that may not exist; got {events[0]['path']!r}"
    )


def test_the_real_loop_drives_the_ceiling_leg_exactly_once() -> None:
    """The leg does not re-fire because the driver keeps driving.

    The REAL `run_training_loop`, not two hand calls to `step()`: the leg carries NO internal
    latch, so a hand re-entry would save twice under the CORRECT code and measure nothing. Zero
    `shutdown_save` events is the second half — a run that FINISHED must not say it was interrupted.
    """
    trainer = _Trainer(step=_CEILING)
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING))

    state = run_training_loop(trainer=trainer, shutdown_state=h.shutdown,
                              coordinator=h.coord, sink=h.sink, max_steps=_LOOP_MAX_STEPS)

    assert trainer.saves == 1, (
        f"exactly ONE final checkpoint on a clean {_LOOP_MAX_STEPS}-bounded drive: not 0 (the "
        f"leg never fired) and not {_LOOP_MAX_STEPS} (the arm stopped saving but not driving); "
        f"got {trainer.saves}"
    )
    assert state.running is False and state.abort_rule is None
    assert h.sink.named("shutdown_save") == [], (
        "leg 2 is the INTERRUPTED semantic and must not appear on a run that finished; got "
        f"{h.sink.named('shutdown_save')}"
    )


def test_a_signal_landing_inside_the_final_write_still_leaves_one_checkpoint() -> None:
    """A signal landing inside the final write still leaves ONE checkpoint.

    The one reachable double-save window: a signal landing between the entry to
    `_clean_stop_save` and `loop.py`'s post-loop test, a window spanning a full-envelope
    `torch.save`. The "same content ⇒ same filename ⇒ idempotent" defence is FALSE: `sha8` covers
    a microsecond-resolution `created_utc`, so two saves at one step are two DISTINCT files.
    """
    shutdown = ShutdownState()
    trainer = _Trainer(step=_CEILING,
                       on_save=lambda: setattr(shutdown, "shutdown_save", True))
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING), shutdown=shutdown)

    run_training_loop(trainer=trainer, shutdown_state=shutdown, coordinator=h.coord,
                      sink=h.sink, max_steps=_LOOP_MAX_STEPS)

    assert trainer.attempts == 1, (
        "premise 1: leg 3 REACHED the writer, which is the only thing that opens the W-1 "
        f"window this row is about; got {trainer.attempts} attempts"
    )
    assert shutdown.shutdown_save is True, (
        "premise 2: the rigged signal really did land during that write — without it this row "
        "would be OC-2a with extra steps"
    )
    assert trainer.saves == 1, (
        "W-1 closed: the loop must NOT run `_final_save()` on top of a completed leg-3 write. "
        f"Two saves at one step are two DISTINCT artefacts, not an idempotent one; got "
        f"{trainer.saves}"
    )
    assert h.sink.named("shutdown_save") == [], (
        "and leg 2 never even announced itself — the guard is read BEFORE `_final_save`'s own "
        f"pre-write emit; got {h.sink.named('shutdown_save')}"
    )


class _CoordinatorPublishingTheLatch:
    """A stand-in leg-3 coordinator: it saves, latches, sets `shutdown_save` and stops the run,
    all from INSIDE `step()`.

    Setting `shutdown_save` from inside `step()` is REQUIRED: an already-set `shutdown_save` at
    loop ENTRY calls `_final_save()` and RETURNS, never reaching the post-loop guard.
    """

    def __init__(self, *, trainer: _Trainer, shutdown: ShutdownState) -> None:
        self.clean_stop_saved = False
        self._trainer = trainer
        self._shutdown = shutdown

    def step(self) -> None:
        self._trainer.save_checkpoint(None)
        self.clean_stop_saved = True
        self._shutdown.shutdown_save = True
        self._shutdown.running = False


class _CoordinatorPublishingNothing:
    """The wiring bug: a coordinator that reaches the guard WITHOUT the flag. Deliberately a
    distinct class rather than the one above with the attribute deleted — a `del` leaves a class
    attribute reachable and would make the drive lie about what it is measuring."""

    def __init__(self, *, trainer: _Trainer, shutdown: ShutdownState) -> None:
        self._trainer = trainer
        self._shutdown = shutdown

    def step(self) -> None:
        self._shutdown.shutdown_save = True
        self._shutdown.running = False


def test_the_loop_guard_latches_leg_two_out_after_a_clean_stop_save() -> None:
    """The loop guard latches leg 2 out after a clean-stop save. The subject is `loop.py`: a
    coordinator that already wrote the run's FINAL artefact must not buy a second `_final_save()`
    merely because `shutdown_save` is also set."""
    shutdown = ShutdownState()
    trainer = _Trainer(step=_CEILING)
    coord = _CoordinatorPublishingTheLatch(trainer=trainer, shutdown=shutdown)
    sink = _Sink()

    run_training_loop(trainer=trainer, shutdown_state=shutdown, coordinator=coord, sink=sink,
                      max_steps=_LOOP_MAX_STEPS)

    assert coord.clean_stop_saved is True, "premise: the stand-in really took its leg-3 arm"
    assert trainer.saves == 1, (
        "the guard must latch leg 2 out; a second `_final_save()` writes a DUPLICATE final "
        f"artefact at the same step (R137); got {trainer.saves}"
    )


def test_a_coordinator_publishing_no_latch_makes_the_loop_raise() -> None:
    """A coordinator publishing no latch makes the loop RAISE, never degrade to a silent `False`.

    A silent `False` re-opens the window the guard exists to close, and invisibly: the run would
    write two final checkpoints and nothing would say so.
    """
    shutdown = ShutdownState()
    trainer = _Trainer(step=_CEILING)
    coord = _CoordinatorPublishingNothing(trainer=trainer, shutdown=shutdown)

    with pytest.raises(TypeError) as exc_info:
        run_training_loop(trainer=trainer, shutdown_state=shutdown, coordinator=coord,
                          sink=_Sink(), max_steps=_LOOP_MAX_STEPS)

    assert "clean_stop_saved" in str(exc_info.value), (
        "the refusal must NAME the member that is missing, or the next reader cannot tell a "
        f"wiring bug from a bug in the loop; got {str(exc_info.value)!r}"
    )
    assert type(coord).__name__ in str(exc_info.value), (
        "…and it must name the offending coordinator TYPE, which is the only thing that "
        f"points at the miswired call site; got {str(exc_info.value)!r}"
    )


def test_a_rigged_final_save_failure_propagates_out_of_step() -> None:
    """A rigged final-save failure propagates out of `step()` — leg 3 catches NOTHING (LAW-14).
    A swallowed failure is a run reporting success having written nothing, and a SECOND authority
    for a storage fault's exit code beside the registered persist-fatal chain."""
    trainer = _Trainer(step=_RESUMED_STEP, raises=OSError("rigged: the volume went away"))
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING))

    with pytest.raises(OSError, match="rigged"):
        h.coord.step()

    assert trainer.attempts == 1, (
        "premise: the leg REACHED the writer exactly once — this is what separates 'the save "
        f"failed' from 'the leg never called'; got {trainer.attempts} attempts"
    )


def test_a_failed_final_save_claims_nothing_in_the_stream_or_the_latch() -> None:
    """A failed final save claims nothing in the stream or the latch.

    An event named for a save is a CLAIM the save happened, so leg 3 emits AFTER the write. The
    latch is set after the write too, so a failure leaves it `False` and the guard does NOT skip
    leg 2 on a run whose leg-3 write died.
    """
    trainer = _Trainer(step=_RESUMED_STEP, raises=OSError("rigged: the volume went away"))
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING))

    with pytest.raises(OSError, match="rigged"):
        h.coord.step()

    assert trainer.saves == 0 and trainer.attempts == 1, (
        "premise: one attempt, no artefact — the exact state the two assertions below are "
        f"about; got attempts={trainer.attempts} saves={trainer.saves}"
    )
    assert h.sink.named("clean_stop_save") == [], (
        "a `clean_stop_save` event on a run that wrote nothing is a false record of the run's "
        f"PRODUCT, and it is what a pre-write emit would produce every time; got "
        f"{h.sink.named('clean_stop_save')}"
    )
    assert h.coord.clean_stop_saved is False, (
        "and the latch must stay False, or the loop-side guard would suppress leg 2 on a run "
        "whose leg-3 write failed — turning one lost save into two"
    )


def test_a_real_write_failure_counts_once_and_the_watchdog_fires_forty_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_net, optim_scaler_sched,
    valid_config, metadata_kwargs, spy_sink, fake_clock,
) -> None:
    """A real write failure counts once and the independent watchdog fires 43.

    No new exit code is authored: 43 already exists, is registered, and is already fired by the
    INDEPENDENT watchdog. This row adds a SECOND live producer for that registered input, driven
    end to end. It is deliberately NOT killed by any leg mutation — the counter is incremented
    INSIDE `_write_v2_payload` before the re-raise, so a swallow at the LEG site does not move it.
    `persist_errors_total` is a module GLOBAL pinned to 0 and restored, since the watchdog's rule
    is a literal `> 0` and a leaked count aborts a later, healthy watchdog.
    """
    monkeypatch.setattr(checkpoints, "persist_errors_total", 0)
    before = checkpoints.persist_errors_total

    def _boom(*_a, **_k):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(torch, "save", _boom)
    opt, scaler, sched = optim_scaler_sched
    with pytest.raises((OSError, RuntimeError)):
        checkpoints.save_checkpoint(
            model=tiny_net, optimizer=opt, scaler=scaler, scheduler=sched, step=_CEILING,
            config=valid_config, metadata_kwargs=metadata_kwargs, checkpoint_dir=tmp_path,
            kind="full",
        )
    assert checkpoints.persist_errors_total - before == 1, (
        "LAW-14: counted EXACTLY once by the ONE writer, never `except: pass`; got "
        f"{checkpoints.persist_errors_total - before}"
    )
    assert sorted(tmp_path.glob("*.ckpt")) == [], (
        "and no artefact exists for the failed step — a partially written checkpoint would be "
        "worse than none, because it would be loadable-looking"
    )

    exits: list[int] = []
    watchdog = HeartbeatWatchdog(
        registry=HeartbeatRegistry(clock=fake_clock),
        deadlines={source: 1800.0 for source in HEARTBEAT_SOURCES},
        sink=spy_sink,
        counters_fn=lambda: checkpoints.persist_errors_total,
        heartbeat_file=tmp_path / "hb.json", file_interval_sec=0.0, poll_interval_sec=0.1,
        clock=fake_clock, save_snapshot=lambda: None, exit_fn=exits.append,
    )
    watchdog.poll_once()

    assert exits == [PERSIST_FATAL_EXIT_CODE] and PERSIST_FATAL_EXIT_CODE == 43, (
        "the registered chain resolves a storage fault to 43 — the code this card REUSES "
        f"rather than authoring a new one beside (§3.2/§3.5); got {exits}"
    )


def test_an_in_loop_hard_abort_below_the_ceiling_writes_no_product_checkpoint() -> None:
    """An in-loop hard abort below the ceiling writes no product checkpoint. SCOPED TO CLASS A.

    Class A is recorded by `_fire_hard_abort` INSIDE the burst, so the loop exits at the `while`,
    `step()` is never re-entered and O2 is never evaluated — no leg-3 checkpoint, STRUCTURALLY.
    Class B (rc 47, rc 48) is recorded in `compose_run`'s ENCLOSING teardown, strictly after leg 3
    has written, so an unscoped "aborted ⇒ no checkpoint" row would assert something FALSE.
    """
    spec = DrawRateAbortSpec(threshold=0.25, min_step=0, N_pool_min=50, consec=3)
    trainer = _Trainer(step=0)
    h = _harness(trainer=trainer, pool=_Pool(draws=900, completed=1000),
                 config=_config(stop_step=_ABORT_CEILING, draw_rate_abort=spec, log_interval=1))
    for _ in range(12):
        if not h.shutdown.running:
            break
        h.pool.games_completed += 5
        h.coord.step()

    assert h.shutdown.running is False and h.shutdown.abort_rule == _ABORT_RULE, (
        "premise: a 0.9 pooled draw rate fired the REAL in-loop gate through the REAL "
        f"`_fire_hard_abort`; got running={h.shutdown.running} rule={h.shutdown.abort_rule!r}"
    )
    assert 0 < trainer.step < _ABORT_CEILING, (
        "premise: the run TRAINED and then aborted strictly BELOW its declared terminus — a "
        f"drive that never trained would satisfy the assertions below vacuously; got step "
        f"{trainer.step} against a {_ABORT_CEILING} ceiling"
    )
    assert trainer.attempts == 0 and trainer.saves == 0, (
        "an in-loop abort must not ship a 'product' checkpoint: the run did NOT reach its "
        f"terminus, and leg 3's artefact means 'the run finished'; got {trainer.attempts} "
        f"attempts / {trainer.saves} saves"
    )
    assert h.coord.clean_stop_saved is False, (
        "…and the clean-completion latch is untouched, so a later `shutdown_save` would still "
        "get its rescue write — leg 3 must not suppress leg 2 on a run that never completed"
    )
    assert h.sink.named("clean_stop_save") == [], (
        f"nor may the stream claim a clean stop on an aborted run; got "
        f"{h.sink.named('clean_stop_save')}"
    )


@pytest.mark.integration
def test_a_clean_run_at_the_minted_bound_leaves_one_stamped_checkpoint(
    tmp_path: Path, smoke_run_config,
) -> None:
    """A clean run at the minted bound leaves exactly ONE stamped checkpoint.

    DEVIATION, pre-registered and measured, not chosen after a red: the armed smoke's own minted
    bound of 200 measured 474.6 s against the tier's 300 s ceiling, 100 at 319.5 s, 50 at 240.2 s.
    The binding rule takes the largest member of {200, 100, 50, 32, 16} measuring <= 300 s, which
    is 50, and the property under test is BOUND-INDEPENDENT. The measurements were taken BEFORE
    this row existed, so the bound could not be lowered to make a red go away.

    Nothing about the RUN is routed around, and the artefact is read back through THE loader. NOT
    asserted, deliberately: that this checkpoint proves the run was clean — it does not (Class B).

    DISARMED IN THIS DRIVE, disclosed: `train.draw_rate_abort` is `None` for THIS config only. A
    50-step drive is below the rule's jurisdiction, so on a slow host the early 100%-ply-cap-draw
    regime (an UNTRAINED net, not a collapsed one) crosses the evidence bar and aborts a healthy
    run. The schema permits no armed-but-unfireable posture.
    """
    minted = smoke_run_config(_SMOKE_CONFIG)
    assert int(minted.train.max_train_steps) == _MINTED_BOUND, (
        "premise: R137's literal 200 IS this config's minted bound, so the deviation below is "
        "a wall-clock one and nothing else. If the mint moves this number, M-0 must be "
        f"re-measured — not this assertion re-aimed; got {minted.train.max_train_steps!r}"
    )
    config = smoke_run_config(
        _SMOKE_CONFIG, train={"max_train_steps": _OC7_BOUND, "draw_rate_abort": None}
    )
    assert int(config.train.max_train_steps) == _OC7_BOUND, (
        "premise: the M-0 bound really reached the coordinator's `stop_step` authority — a "
        f"section override that silently failed would drive 200 and blow the tier; got "
        f"{config.train.max_train_steps!r}"
    )
    assert config.train.draw_rate_abort is None, (
        "premise: the disarm above really REACHED the coordinator — the same section-merge "
        "check the bound gets, on the other override. A silently-failed merge would leave the "
        "rule armed and this drive would red on a slow host with rule='draw_rate_collapse', "
        "which is the exact failure this test's disclosure block exists to prevent being read "
        f"as a real collapse; got {config.train.draw_rate_abort!r} (RQ-20 / R288(c) grant)"
    )

    handles = launch_run(config=config, out_dir=tmp_path)

    assert handles.shutdown.running is False and handles.shutdown.abort_rule is None, (
        "premise: a CLEAN completion — the run reached its terminus and fired no abort; got "
        f"running={handles.shutdown.running} rule={handles.shutdown.abort_rule!r}"
    )
    assert int(handles.coordinator.trainer.step) == _OC7_BOUND, (
        f"…and the learner reached the bound EXACTLY; got {handles.coordinator.trainer.step!r}"
    )

    residents = sorted((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(residents) == 1, (
        f"R137's literal: exactly ONE final checkpoint. Not zero (leg 3 absent — the pre-R137 "
        f"truth) and not two (W-1 re-opened, or a second write authority); got "
        f"{[p.name for p in residents]}"
    )
    assert sorted((tmp_path / "checkpoints").glob("*.quarantine")) == [], (
        "and leg 3 did NOT take the survive-run quarantine branch: `Trainer.save_checkpoint` "
        "passes `allow_quarantine` by omission, so an unstampable terminal artefact must "
        "RAISE, never land as a `.quarantine` the count above would not see (LAW-12/R3: an "
        f"artifact that cannot be stamped cannot be written); got "
        f"{[p.name for p in (tmp_path / 'checkpoints').glob('*.quarantine')]}"
    )

    ckpt = checkpoints.load_checkpoint(residents[0], expected_run_id=config.run_id)
    assert ckpt.metadata.step == _OC7_BOUND, (
        "the artefact is stamped at the TERMINUS, which is what makes it the run's product "
        f"rather than a mid-run resumption point; got {ckpt.metadata.step!r}"
    )
    assert ckpt.metadata.run_id == config.run_id, (
        f"…carries this run's lineage; got {ckpt.metadata.run_id!r}"
    )
    assert ckpt.metadata.encoding_name == config.identity.encoding == "gnn_axis_v1", (
        "…and the DECLARED encoding, resolved through the registry (LAW-11: no dense-by-"
        f"default, absent encoding is an error); got {ckpt.metadata.encoding_name!r}"
    )
    assert ckpt.metadata.created_utc.endswith("Z"), (
        f"the immutable stamp is ISO-8601 Z (LAW-12); got {ckpt.metadata.created_utc!r}"
    )
    datetime.fromisoformat(ckpt.metadata.created_utc.removesuffix("Z"))
    assert ckpt.kind == "full" and ckpt.optimizer_state is not None, (
        "…and the FULL envelope was written, not a weights-only strip — the terminal artefact "
        "has to be resumable and evaluatable, which is the whole reason legs 1 and 2 write one"
    )
