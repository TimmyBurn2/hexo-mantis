# >300 justify (R8).
# The ten rows are ONE claim — R137's THIRD save leg: a run that reaches its own declared
# terminus writes EXACTLY ONE stamped checkpoint, a rigged failure is supervisor-distinguishable,
# and an in-loop abort still writes nothing — over one seam that runs from `step.py`'s O2 arm
# through a latch on the coordinator to `loop.py`'s post-loop guard and on to the registered
# persist-fatal chain. R5 bars cross-test imports, so measuring it needs the local
# `StepCoordinator` harness (the `tests/train/test_abort_exit_signal.py` shape) in this file;
# a split forks that harness into copies that then drift while both stay green, and it would
# separate the leg (OC-1) from the driver that makes it exactly-once (OC-2a/OC-2b) and from
# the guard that closes W-1 (OC-3a/OC-3b) — three halves of one argument. Executable content
# is a minority; the rest is the per-row "what defect is this the only witness to" rationale
# LAW-07 asks each row to carry, and the reachability note R166 asks each drive to state.
"""⊕ WP12-R Phase CS / OC-1..OC-5, OC-7 (R137 / CARD-CLEANSTOP-SAVE) — the third save leg.

R129 measured the hole and R137 carded it: at HEAD a run that reaches `stop_step` exits **0**
with `checkpoints/` **EMPTY**. `coordinator/step.py`'s O2 arm sets `shutdown.running = False`
and returns without saving; `loop.py`'s `_final_save()` fires only on `shutdown_save`, which
only a SIGNAL sets; and the trainer's periodic arm is guarded by `interval > 0` against a
`checkpoint_interval` every minted config mints at 0 — and, at the commit this file was
written, did not exist at all on the graph representation run5 declares (DESIGN_CS §1.5,
F-CS-1). **WP12-R CARD-CS2 (R173) has since falsified that second clause**: both step tails
now call the ONE resolver `Trainer._maybe_periodic_checkpoint`, so the graph arm exists and
evaluates, and the minted `0` is the sole reason it stays silent. So the 1e6-step run's
PRODUCT — the terminal weights the terminal eval and the deploy tag are about — is never
written.

Leg 3 is its OWN semantic, not a differently-triggered `shutdown_save` (DESIGN_CS §2.3): leg 1
means "a resumption point", leg 2 means "we were interrupted", leg 3 means "the run finished".

The defect each row is the ONLY witness to:

- **OC-1** — the leg not firing at all, i.e. the whole card. Drives ONE real
  `StepCoordinator.step()` at the ceiling and reads all five observables the leg publishes:
  the call on the injected trainer, `shutdown.running`, `shutdown.abort_rule` (still `None` —
  a clean stop is not an abort, R84), the `clean_stop_saved` latch, the outcome's
  `checkpoint_saved` flag (asserted HERE precisely because nothing in `src/` reads it —
  DESIGN_CS §2.2 N-6), and the ONE `clean_stop_save` event with its three fields.
- **OC-2a** — a leg that re-fires because the driver keeps driving. It cannot be measured by
  calling `step()` twice by hand: the leg carries NO internal latch (deliberate — exactly-once
  comes from the DRIVER, DESIGN_CS §2.4), so a hand re-entry saves twice under the CORRECT
  code too. The real `run_training_loop` is the only honest instrument (PREREG_CS §3.0 T-2).
- **OC-2b** — W-1, the one reachable double-save window, driven verbatim on the REAL objects:
  a signal landing INSIDE leg 3's own multi-second `torch.save`. The only row driving the real
  leg and the real loop guard together.
- **OC-3a** — leg exclusivity at the guard: `shutdown_save` set during the leg must not buy a
  second `_final_save()`.
- **OC-3b** — a coordinator that publishes no `clean_stop_saved` must RAISE, never degrade to
  a silent `False`, which would re-open W-1 with nobody noticing.
- **OC-4a** — a swallowed final-save failure: a run reporting success having written nothing
  (LAW-14; leg 3 catches nothing).
- **OC-4b** — an event stream claiming a save that never happened. The event is emitted AFTER
  the write, deliberately UNLIKE `loop.py:87` which emits before its own (Q-CS-6).
- **OC-4c** — LAW-14's rc authority, at the seam the registered producer test already uses:
  a REAL write failure counts once on `checkpoints.persist_errors_total` and the INDEPENDENT
  watchdog's `poll_once()` fires **43**. No new exit code is authored by this card.
- **OC-5** — an in-loop abort shipping a "product" checkpoint. **SCOPED TO CLASS A**; the
  docstring names Class B, which CAN coexist with a leg-3 artefact.
- **OC-7** — a checkpoint written outside the ONE stamp path, or stamped with the wrong
  step/lineage. R137's literal, at the armed smoke's own minted bound.

**What is real here and what is not.** Real in OC-1..OC-5: the `StepCoordinator`, its config
built by the PRODUCTION builder off a MINTED config, the `ShutdownState`, `run_training_loop`,
`_fire_hard_abort`, the draw-rate gate. Fake: the TRAINER (a counter — the subject is control
flow and the trainer is its instrument), the pool/buffer collaborators, and the sink (a spy).
OC-3a/OC-3b additionally fake the COORDINATOR, because their subject is the LOOP guard — which
is exactly why OC-2b exists (PREREG_CS §1, N-4). OC-4c's only substitution is `exit_fn` plus a
killed `torch.save`, the pattern `tests/monitor/test_persist_fatal.py` and
`tests/train/test_lifecycle_contract.py` already use. OC-7 fakes nothing about the run.
"""
from __future__ import annotations

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

_REPO = Path(__file__).resolve().parents[2]
_MINTED = load_config(_REPO / "configs" / "dev_example.yaml")

#: The coordinator's non-leg knobs come from a MINTED config through the production
#: resolvers, never from literals here (WPMINT Phase K-A/K-B; R1).
_DRAIN_CAPS = resolve_drain_caps(_MINTED.monitor)
_KNOBS = resolve_coordinator_knobs(_MINTED.train)
#: R242 (ADJ-D12): the builder's FIFTH config-authored parameter — `monitor.gate_interval`,
#: the ARMING cadence, from the same minted config. Harnesses that set `log_interval` MIRROR
#: it onto `gate_interval`, which is the shipped posture (every committed config mints the
#: two equal), so these drives keep exactly the cadence they had before R242's split.
_GATE_INTERVAL = _MINTED.monitor.gate_interval


def _mirrored(settings: dict) -> dict:
    """R242 (ADJ-D12): the GATE cadence mirrors the NARRATION cadence unless a drive names it.

    That mirroring is the SHIPPED posture, not a convenience — every committed config mints
    `monitor.gate_interval` equal to its own `train.log_interval` — so a drive here that moves
    only `log_interval` keeps exactly the cadence it had before R242 split the two knobs.
    """
    settings.setdefault("gate_interval", settings["log_interval"])
    return settings

#: The declared terminus for the leg-3 drives.
_CEILING = 5
#: The trainer's step at O2 entry for OC-1/OC-4, DELIBERATELY DISTINCT from `_CEILING`: the
#: `clean_stop_save` event carries BOTH `step` and `stop_step`, and equal values would let an
#: implementation that emits one number twice — or swaps the two fields — pass forever. This
#: is also a REAL production state, not a contrivance: DESIGN_CS §2.4 names the resumed run
#: whose `trainer.step` already exceeds `stop_step` as firing O2 on its first `step()`.
_RESUMED_STEP = 7
#: OC-2a/OC-2b drive the CANONICAL shape instead — the run arriving exactly at its ceiling —
#: so the file covers both `>` and `==` on the O2 predicate.
_LOOP_MAX_STEPS = 5
#: OC-5's ceiling: far above any step its drive reaches, so its abort is unambiguously BELOW
#: the terminus and the leg's predicate is genuinely false when the abort fires.
_ABORT_CEILING = 1000
_ABORT_RULE = "draw_rate_collapse"

#: What the fake trainer's `save_checkpoint` hands back. The leg must publish THE WRITER'S
#: return value on the event, not a path it re-derives from the checkpoint dir.
_SAVED_PATH = Path("/checkpoints/oracle_00000007_deadbeef.ckpt")

#: The armed smoke's OWN minted `train.max_train_steps`, which is also R137's literal ("a
#: clean 200-step run"). Asserted as a PREMISE by OC-7, never used as its drive: M-0 measured
#: it at 474.6 s against the 300 s tier ceiling (PREREG_CS §5.3).
_SMOKE_CONFIG = "smoke_preflight_armed.yaml"
_MINTED_BOUND = 200
#: OC-7's drive bound, fixed by the PRE-REGISTERED M-0 measurement and its binding decision
#: rule — the largest member of {200, 100, 50, 32, 16} measuring <= 300 s on the dev box.
#: Measured, single run each, at `32ec7b9` with the leg absent: 200 -> 474.6 s, 100 -> 319.5 s,
#: 50 -> 240.2 s. The measurement PRECEDED this row; it was never lowered after seeing a red.
_OC7_BOUND = 50


# ── the minimum real-coordinator harness (local by necessity — R5 bars cross-test imports) ──
class _Pool:
    """A pool whose draw counts are the only thing OC-5's gate reads off it."""

    def __init__(self, *, draws: int = 0, completed: int = 0) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
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
    """The instrument. `attempts` and `saves` are DISTINCT on purpose: a rigged failure must
    show that the leg REACHED the writer (`attempts == 1`) and that no artefact resulted
    (`saves == 0`). One counter could not tell "the leg never called" from "the call failed",
    which is exactly the difference between mutations M-1 and M-5."""

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

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e["event"] == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder — this file's deltas only. `None` is the EXPLICIT
    disarmed draw-rate posture; the builder gives it no default and neither does this factory
    (R1: no code-side default for anything an assertion's meaning depends on)."""
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
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
        train_cfg={}, mixing_cfg={}, sink=sink, heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, shutdown=shutdown, sink=sink)


# ══ OC-1 — the leg fires, once, and publishes what it did ══════════════════════════════
def test_the_ceiling_arm_saves_once_and_publishes_the_clean_stop_leg() -> None:
    """OC-1 — the whole card, on ONE real `StepCoordinator.step()` at the terminus.

    MUTATIONS THAT RED IT: M-1 (delete the leg call — `saves`/event/latch all go to 0/False
    together), M-6 (delete only the `save_checkpoint(...)` inside the leg — `saves` goes to 0
    while the event and the latch stay, which is M-6's signature and distinguishes it from
    M-1), M-9 (delete the emit), M-11 (call the leg twice), M-12 (revert the outcome flag),
    M-2 (`running` never flips).

    `abort_rule is None` is asserted because a clean stop and an abort both write
    `running = False`, and `abort_rule` is the ONLY thing that tells them apart (R84). The
    outcome's `checkpoint_saved` is asserted because NOTHING in `src/` reads it (DESIGN_CS
    §2.2 N-6) — leg 3 sets it for parity with O3, and without this row that write would be
    unwitnessed in the whole tree.

    The event's two step fields are read against DISTINCT numbers (`_RESUMED_STEP` 7 vs
    `_CEILING` 5), so an implementation emitting one of them twice, or swapping them, reds
    here rather than agreeing with itself forever.
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


# ══ OC-2a/OC-2b — exactly once under the REAL driver ═══════════════════════════════════
def test_the_real_loop_drives_the_ceiling_leg_exactly_once() -> None:
    """OC-2a — a leg that re-fires because the driver keeps driving.

    Why the REAL `run_training_loop` and not two hand calls to `step()` (PREREG_CS §3.0 T-2):
    the leg carries NO internal latch. Exactly-once is a property of the DRIVER — O2 sets
    `running = False` and returns, and `while shutdown_state.running` is tested BEFORE each
    call. A hand re-entry would save twice under the CORRECT code, so it would measure
    nothing. `max_steps=5` bounds the drive so a mutant that leaves `running` True still
    terminates and is READABLE (that is M-2's only sound witness; under a production driver,
    which passes no `max_steps`, M-2 does not terminate at all).

    MUTATIONS THAT RED IT: M-1/M-6 (`saves` 1→0), M-2 (`saves` 1→5), M-11 (1→2).

    Zero `shutdown_save` events is the second half: leg 3 is its OWN semantic, and a run that
    FINISHED must not leave a stream saying it was interrupted (DESIGN_CS §2.3).
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
    """OC-2b — W-1 verbatim, on the REAL coordinator AND the REAL loop guard.

    W-1 is the ONE reachable double-save window (DESIGN_CS §2.4): a SIGINT/SIGTERM landing
    between the entry to `_clean_stop_save` and `loop.py`'s post-loop test. The window spans a
    full-envelope `torch.save` — `model_state` + `optimizer_state` + `scaler_state` +
    `scheduler_state` — which is multi-second on run5. It is driven here by making the
    trainer's `save_checkpoint` set `shutdown_save` as a SIDE EFFECT, which is what a signal
    arriving mid-write looks like from the loop's side.

    The naive "same content ⇒ same filename ⇒ idempotent" defence is FALSE and must not be
    leaned on: `sha8` is `content_sha8` over a payload carrying `metadata.created_utc`, which
    is microsecond-resolution, so two saves at one step are two DISTINCT files.

    This is the only row driving the real leg and the real guard TOGETHER — OC-3a/OC-3b reach
    the guard through a fake coordinator and OC-1 reaches the leg without the loop, so without
    this row the production combination is never executed (PREREG_CS §1, N-4).

    MUTATIONS THAT RED IT: M-3 (drop the guard's conjunct → `saves` 1→2), M-1/M-6 (1→0).
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


# ══ OC-3a/OC-3b — the loop-side guard ══════════════════════════════════════════════════
class _CoordinatorPublishingTheLatch:
    """A coordinator standing in for a leg-3 `StepCoordinator`: it saves, latches, sets
    `shutdown_save` (the mid-write signal) and stops the run — all from INSIDE `step()`.

    Setting `shutdown_save` from inside `step()` is REQUIRED, not incidental (PREREG_CS §3.0
    T-1): `loop.py:91-94` observes an already-set `shutdown_save` at ENTRY, calls
    `_final_save()` and RETURNS, so an entry-set drive never reaches the post-loop guard these
    two rows exist to measure.
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
    distinct class rather than the one above with the attribute deleted — a `del` leaves a
    class attribute reachable and would make the drive lie about what it is measuring."""

    def __init__(self, *, trainer: _Trainer, shutdown: ShutdownState) -> None:
        self._trainer = trainer
        self._shutdown = shutdown

    def step(self) -> None:
        self._shutdown.shutdown_save = True
        self._shutdown.running = False


def test_the_loop_guard_latches_leg_two_out_after_a_clean_stop_save() -> None:
    """OC-3a — leg exclusivity at the guard, through a coordinator whose own `step()` saves.

    The subject here is `loop.py`, not `step.py`: a coordinator that already wrote the run's
    FINAL artefact must not buy a second `_final_save()` merely because `shutdown_save` is
    also set. MUTATION THAT REDS IT: M-3, dropping the `not _clean_stop_already_saved(...)`
    conjunct → two saves at one step, i.e. two distinct FINAL artefacts.
    """
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
    """OC-3b — the wiring bug must be LOUD, never a silent `False`.

    A silent `False` re-opens exactly the W-1 window the guard exists to close, and it does so
    invisibly: the run would write two final checkpoints and nothing would say so. This is
    `close_out`'s posture on `disarm_staleness` verbatim — a duck-typed object missing the
    member is a wiring bug, and a wiring bug must not degrade into "no guard".

    MUTATIONS THAT RED IT: M-4 (return `False` when the attribute is absent — the defect
    itself), and M-3 (dropping the conjunct deletes the ONLY call site of the only source of
    this `TypeError`, so nothing raises at all).
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


# ══ OC-4a/OC-4b/OC-4c — LAW-14: a final-save failure is fatal, counted, and never faked ══
def test_a_rigged_final_save_failure_propagates_out_of_step() -> None:
    """OC-4a — LAW-14: leg 3 catches NOTHING.

    A swallowed final-save failure is a run that reports success having written nothing, and
    it would also be a SECOND authority for a storage fault's exit code beside the registered
    persist-fatal chain (OC-4c). MUTATIONS THAT RED IT: M-5 (`try/except Exception: pass`
    around the leg's save — the defect itself), M-1/M-6 (no save call left in `step()` at all,
    so nothing raises and this fails DID-NOT-RAISE).
    """
    trainer = _Trainer(step=_RESUMED_STEP, raises=OSError("rigged: the volume went away"))
    h = _harness(trainer=trainer, config=_config(stop_step=_CEILING))

    with pytest.raises(OSError, match="rigged"):
        h.coord.step()

    assert trainer.attempts == 1, (
        "premise: the leg REACHED the writer exactly once — this is what separates 'the save "
        f"failed' from 'the leg never called'; got {trainer.attempts} attempts"
    )


def test_a_failed_final_save_claims_nothing_in_the_stream_or_the_latch() -> None:
    """OC-4b — an event named for a save is a CLAIM the save happened.

    Leg 3 emits AFTER the write, deliberately unlike `loop.py:87`, which emits `shutdown_save`
    BEFORE its own — so every failed shutdown save today leaves a stream asserting a save that
    never happened (Q-CS-6, queued rather than silently mirrored). The latch is likewise set
    after the write, so a failure leaves it `False` and the loop-side guard does NOT skip
    leg 2 on a run whose leg-3 write died.

    MUTATIONS THAT RED IT: M-10 (move the emit ABOVE the save — G-CS-1: the "ONLY witness of
    the ORDER" claim that stood here was FALSE and unrun; EXECUTED, M-10 also reds OC-1 on its
    `path` assert, so OC-1 is a second, independent witness), M-5 (swallow → latch and emit
    both run after a failure), M-6 (delete the save, keep the emit → the event fires anyway).
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
    """OC-4c — LAW-14's rc AUTHORITY, at the seam the registered producer test already uses.

    This card authors NO new exit code: `PERSIST_FATAL_EXIT_CODE = 43` already exists, is
    already registered (`monitor/producer_manifest.yaml`, id `persist_fatal`), and is already
    fired by the INDEPENDENT watchdog. What this row adds is a SECOND live producer for that
    registered input, driven end to end: a real `torch.save` fault inside the ONE writer bumps
    `checkpoints.persist_errors_total` by exactly 1 and writes no `.ckpt`, and the watchdog's
    `poll_once()` observes the counter and fires 43.

    Deliberately NOT killed by any leg mutation (PREREG_CS §3.2): it witnesses the SHARED
    chain, and its killer is the pre-existing `tests/monitor/test_persist_fatal.py` battery.
    Reading its green as coverage of M-5 would be the phantom-coverage error — the counter is
    incremented INSIDE `_write_v2_payload` before the re-raise, so a swallow at the LEG site
    does not move it. That is why OC-4a and OC-4b exist separately.

    `persist_errors_total` is a process-wide module GLOBAL; the monkeypatch pins it to 0 AND
    restores the pre-test value at teardown, so the increment cannot leak into another suite
    (the watchdog's persist rule is a literal `> 0`, so a leaked count aborts a later, healthy
    watchdog on inherited state).
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


# ══ OC-5 — an in-loop abort ships no product checkpoint ════════════════════════════════
def test_an_in_loop_hard_abort_below_the_ceiling_writes_no_product_checkpoint() -> None:
    """OC-5 — **SCOPED TO CLASS A: the IN-LOOP aborts.**

    Class A (`grad_norm_hard_abort`, `draw_rate_collapse`, `sealbot_wr_abort`) is recorded by
    `_fire_hard_abort` INSIDE the burst: `running = False` is set there, the loop exits at
    `loop.py`'s `while`, `step()` is never re-entered and O2 is never evaluated. So a Class-A
    abort gets no leg-3 checkpoint STRUCTURALLY, and that is what this row measures.

    **Class B is the counter-class, and it CAN coexist with a leg-3 artefact** —
    `disk_space_exhausted` (rc 47) and `terminal_eval_broken` (rc 48) are recorded in
    `compose_run`'s ENCLOSING teardown, strictly AFTER `close_out`, and therefore strictly
    after leg 3 has written (DESIGN_CS §2.5.1). A row asserting the unscoped "aborted ⇒ no
    checkpoint" would be asserting something FALSE at HEAD. The clean-vs-aborted distinction
    is carried by `ShutdownState.abort_rule` and its rc — never by the filesystem (Q-CS-9).

    MUTATIONS THAT RED IT: M-2b (MOVE the save to the top of `step()` so every call saves →
    `attempts` 0→1 on a run that ABORTED), M-8 (weaken the O2 predicate to drop the step
    comparison → the first `step()` saves and stops, so the abort never fires and `abort_rule`
    goes to `None`).
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


# ══ OC-7 — R137's literal, on the real artefact through THE stamp path ═════════════════
@pytest.mark.integration
def test_a_clean_run_at_the_minted_bound_leaves_one_stamped_checkpoint(
    tmp_path: Path, smoke_run_config,
) -> None:
    """OC-7 — R137's literal: *"a clean 200-step run ends with exactly ONE final checkpoint"*.

    **DEVIATION, pre-registered and measured, not chosen after a red.** R137's 200 is also the
    armed smoke's own minted `train.max_train_steps`, so the literal and the config agree — and
    M-0 (PREREG_CS §5.3) measured that bound at **474.6 s** against the tier's 300 s ceiling,
    with 100 at 319.5 s and 50 at **240.2 s**. The binding decision rule takes the largest
    member of {200, 100, 50, 32, 16} measuring <= 300 s, which is **50**, and requires the
    deviation be stated: *the property under test — exactly ONE checkpoint, at
    `step == stop_step`, through the LAW-12 stamp path — is BOUND-INDEPENDENT; R137's 200 is
    the illustrative length of a clean run, and OC-6 independently carries the end-to-end truth
    at 16 on the production `main` rc path.* The three measurements were taken BEFORE this row
    existed, on a tree where the leg does not exist, precisely so the bound could not be
    lowered to make a red go away.

    R64 posture — nothing about the RUN is routed around: real `init_trainer` -> `build_net`,
    real `WorkerPool` self-play on CPU, real graph replay buffer, real coordinator, real
    `close_out`. The artefact is read back through THE loader (`checkpoints.load_checkpoint`,
    `torch.load(weights_only=True)`, provenance re-verified against the filename), never by a
    hand-rolled parse.

    What it is the only witness to: a checkpoint written OUTSIDE the one LAW-12 stamp path, or
    stamped with the wrong step or lineage. MUTATIONS THAT RED IT: M-1/M-6 (no file), M-2b
    (many files), M-7 (strip `encoding_name` from the trainer's `metadata_kwargs` →
    `_build_stamped_metadata` raises and an unstampable artefact is NOT written — "an artifact
    that cannot be stamped cannot be written"), M-8 (the artefact lands at the wrong step).

    NOT asserted, deliberately: that the presence of this checkpoint proves the run was clean.
    It does not (Class B, OC-5's docstring). Cleanliness is `abort_rule`, asserted separately.
    """
    minted = smoke_run_config(_SMOKE_CONFIG)
    assert int(minted.train.max_train_steps) == _MINTED_BOUND, (
        "premise: R137's literal 200 IS this config's minted bound, so the deviation below is "
        "a wall-clock one and nothing else. If the mint moves this number, M-0 must be "
        f"re-measured — not this assertion re-aimed; got {minted.train.max_train_steps!r}"
    )
    config = smoke_run_config(_SMOKE_CONFIG, train={"max_train_steps": _OC7_BOUND})
    assert int(config.train.max_train_steps) == _OC7_BOUND, (
        "premise: the M-0 bound really reached the coordinator's `stop_step` authority — a "
        f"section override that silently failed would drive 200 and blow the tier; got "
        f"{config.train.max_train_steps!r}"
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
