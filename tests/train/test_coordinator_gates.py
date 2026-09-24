"""ORACLE — the coordinator gate seam.

step() never consumes the eval KICK return and makes ZERO blocking eval calls (the sealbot-WR
consumer that once sat at the async result seam left with the rung, R362(c); a completed round
is routed to promotion by `drain._route_eval_result`). Also covered: draw-rate gate wiring on the
LIVE producer, the emission wiring, and the `train_step` heartbeat beats.

>300 justify: one coordinator seam, one harness shared by every gate row; splitting the file
would duplicate that harness and let the two halves drift apart.
"""
from __future__ import annotations

import dataclasses

import pytest
from types import SimpleNamespace
from typing import Any

from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from _drivable import DrivablePoolStub
from _graph_drive import DEV_DRAIN_CAPS, DEV_GATE_INTERVAL, DEV_KNOBS, GRAPH_FULL_CONFIG, GraphSampleBuffer
from _monitor_config import monitor_config
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState


class FakeTrainer:
    """Conforms to the DECLARED seam (WPTS/TD-1 re-point, R90a): typed entry points +
    `device`; the dead `train_step` fake is gone with the card."""

    def __init__(self, grad_norm: float = 0.1) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self._gn = grad_norm

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": self._gn,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3, "opp_reply_loss": 0.0,
                "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def save_checkpoint(self, loss_info) -> None:
        return None


class FakeEvalPipeline:
    """Async eval pipeline: `run_evaluation` (the KICK) returns an ack; `drain_pending` is a
    spy that MUST stay uncalled from step() (a blocking drain in step() = the run3 wedge)."""

    def __init__(self, kick_result: dict) -> None:
        self.kick_result = kick_result
        self.run_calls = 0
        self.drain_calls = 0
        self.poll_calls = 0
        self.apply_calls = 0

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.run_calls += 1
        return self.kick_result

    def drain_pending(self):
        self.drain_calls += 1
        return None

    def poll_completed(self):
        # WP11-A: step()'s non-blocking poll at the top of every iteration. This fixture
        # never has a completed round ready — the kick-return-is-never-consumed invariant this
        # test pins is entirely about the KICK ack, not this seam.
        self.poll_calls += 1
        return None

    def apply_gate_decision(self, result):
        self.apply_calls += 1
        return None


class BeatSpy:
    def __init__(self) -> None:
        self.beats: list[str] = []

    def __call__(self, source: str) -> None:
        self.beats.append(source)


class SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _make_config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder, never a hand-written kwarg census: restating every
    field made ten test files agree with `StepCoordinatorConfig` by maintenance rather than
    construction. `stop_step`/`draw_rate_abort` are passed EXPLICITLY because neither the
    builder nor this factory gives them a default, and `gate_interval` MIRRORS `log_interval`
    unless a drive names it — the SHIPPED posture stated once."""
    settings = {"eval_interval": 1, "log_interval": 1, "min_buf_size": 10, **overrides}
    settings.setdefault("gate_interval", settings["log_interval"])
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None,
                                 drain_caps=DEV_DRAIN_CAPS, gate_interval=DEV_GATE_INTERVAL,
                                 knobs=DEV_KNOBS),
        **settings,
    )


def _make_coordinator(*, pool=None, config=None, eval_pipeline=None, heartbeat=None,
                      monitor_cfg=None, trainer_step: int = 0):
    pool = pool or DrivablePoolStub()
    trainer = FakeTrainer()
    trainer.step = trainer_step  # a resumed trainer: the coordinator reads it at construction
    buffer = GraphSampleBuffer()
    shutdown = ShutdownState()
    sink = SpySink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(),
        # WPTS/TD-1: the straight arm resolves its route from the DECLARED identity — these
        # unit drives declare the grid identity FakeBuffer's sampler serves.
        config=config or _make_config(),
        full_config=GRAPH_FULL_CONFIG,
        sink=sink, heartbeat=heartbeat, monitor_cfg=monitor_cfg or monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, buffer=buffer,
                           shutdown=shutdown, sink=sink, eval_pipeline=eval_pipeline)


def _drive_until_stopped(h, *, games_per_step=5, cap=12):
    """Drive step() up to `cap` times, bumping games each iteration so the burst runs; stop
    when the coordinator flips running=False (a gate fired)."""
    last = None
    for _ in range(cap):
        if not h.shutdown.running:
            break
        h.pool.games_completed += games_per_step
        last = h.coord.step()
    return last


# O-06 — the kick return is an ACK, never a result
def test_step_does_not_consume_the_kick_return_and_never_blocks() -> None:
    """The eval KICK returns a result-shaped ack (`promoted: True`) that WOULD apply a promotion
    if wrongly consumed; step() must route nothing from it and make ZERO blocking calls. Bites a
    consumer wired to the kick return, and a blocking eval re-entering step()."""
    pipe = FakeEvalPipeline(kick_result={"step": 30000, "promoted": True})
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5
    h.coord.step()
    assert pipe.run_calls >= 1, "the eval kick must have fired at the boundary"
    assert pipe.drain_calls == 0, "step() must make ZERO blocking drain calls"
    assert pipe.apply_calls == 0, (
        "the kick return must NEVER be consumed as a round result (the masked-dead-gate class)"
    )
    assert h.shutdown.running is True, "no fire may come from the kick return"


def test_steps_per_hour_after_a_resume_counts_steps_since_boot() -> None:
    """B-2 (R355(e)): booted at 23 829 the rate read 23 829 + d over the hours since boot (4.79e9)."""
    h = _make_coordinator(trainer_step=23_829)
    started = h.coord._run_started
    h.coord._clock = SimpleNamespace(now=lambda: started + 3600.0, sleep=lambda _s: None)
    h.pool.games_completed = 5
    h.coord.step()
    sph = h.sink.named("iteration_complete")[-1]["steps_per_hour"]
    steps_since_boot = h.trainer.step - 23_829
    assert steps_since_boot > 0
    assert sph == round(steps_since_boot / 3600.0 * 3600.0, 1), (
        f"steps_per_hour must be steps SINCE BOOT over the run clock, got {sph}"
    )


class _KickSpy:
    """A pipeline that records the step of every kick and completes nothing."""

    def __init__(self) -> None:
        self.kicks: list[int] = []
        self.round_counter = 0

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.kicks.append(int(step))
        return {"kicked": True}

    def drain_pending(self):
        return None

    def poll_completed(self):
        return None


def _boot_at(step: int, *, eval_interval: int):
    return _make_coordinator(config=_make_config(eval_interval=eval_interval, log_interval=1),
                             eval_pipeline=_KickSpy(), trainer_step=step)


def test_a_fresh_run_crossing_the_boundary_kicks_once() -> None:
    h = _boot_at(2999, eval_interval=3000)
    h.pool.games_completed = 5
    h.coord.step()
    assert h.eval_pipeline.kicks == [3000]
    h.pool.games_completed += 5
    h.coord.step()
    assert h.eval_pipeline.kicks == [3000], "the round is kicked once, not on every later step"


def test_a_resume_exactly_at_the_boundary_with_no_record_kicks_it() -> None:
    """B-3 (R355(e)): resumed at 3000 the old rule tested `3001 % 3000` and never kicked round 1."""
    h = _boot_at(3000, eval_interval=3000)
    h.coord.restore_eval_round_state(-1)  # a sidecar that predates the field
    h.pool.games_completed = 5
    h.coord.step()
    assert h.eval_pipeline.kicks == [3001], "round 1 kicks on the first step after the resume"


def test_a_resume_whose_sidecar_says_the_round_was_kicked_does_not_repeat_it() -> None:
    h = _boot_at(3000, eval_interval=3000)
    h.coord.restore_eval_round_state(3000)  # the kick at 3000 is on the record
    h.pool.games_completed = 5
    h.coord.step()
    assert h.eval_pipeline.kicks == []


def test_a_re_minted_eval_interval_across_a_resume_re_derives_the_round_from_the_kicked_step() -> None:
    """The record is a STEP: kicked at 24000 under 3000, resumed under 2000 the next round is 26000."""
    h = _boot_at(24000, eval_interval=2000)
    h.coord.restore_eval_round_state(24000)
    h.pool.games_completed = 5
    h.coord.step()
    assert h.eval_pipeline.kicks == []
    h.coord._train_step = 25999
    h.trainer.step = 25999
    h.pool.games_completed += 5
    h.coord.step()
    assert h.eval_pipeline.kicks == [26000]


def test_a_kick_record_past_the_boot_step_is_refused() -> None:
    from mantis.train.resume_state import ResumeStateError

    h = _boot_at(3000, eval_interval=3000)
    with pytest.raises(ResumeStateError, match="past the boot step"):
        h.coord.restore_eval_round_state(3001)


def test_draw_rate_gate_fires_on_live_producer() -> None:
    """The `draw_rate_collapse` producer test, keyed on the LIVE pooled rate and never on a NaN
    draw-target phantom: a sustained 0.9 pooled rate over sufficient evidence, past min_step,
    fires. Grad-norm is quiet, so the fire is draw-rate."""
    pool = DrivablePoolStub(draw_counts=(90, 100))
    cfg = _make_config(draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                        N_pool_min=10, consec=3))
    h = _make_coordinator(pool=pool, config=cfg)
    _drive_until_stopped(h)
    assert h.shutdown.running is False, "a sustained pool draw-rate collapse must hard-abort"
    aborts = h.sink.named("hard_abort")
    assert aborts and any("draw" in str(e) for e in aborts), (
        f"the draw-rate gate must emit a hard_abort naming its rule: {aborts}"
    )


def test_a_resume_restores_the_draw_rate_window_so_the_third_observation_fires() -> None:
    """B-7 (R355(e)): the resume emptied the abort windows (run7's draw-rate abort moved 25k -> 26k)."""
    spec = DrawRateAbortSpec(threshold=0.4, min_step=0, N_pool_min=10, consec=3)
    before = _make_coordinator(pool=DrivablePoolStub(draw_counts=(90, 100)),
                               config=_make_config(draw_rate_abort=spec))
    for _ in range(2):
        before.pool.games_completed += 5
        before.coord.step()
    assert before.shutdown.running is True and len(before.coord._draw_rate_history) == 2
    carried = before.coord.guard_state()

    resumed = _make_coordinator(pool=DrivablePoolStub(draw_counts=(90, 100)),
                                config=_make_config(draw_rate_abort=spec))
    resumed.coord.restore_guard_state(carried)
    resumed.pool.games_completed += 5
    resumed.coord.step()
    assert resumed.shutdown.running is False, "the third observation, first after the resume, fires"

    fresh = _make_coordinator(pool=DrivablePoolStub(draw_counts=(90, 100)),
                              config=_make_config(draw_rate_abort=spec))
    fresh.pool.games_completed += 5
    fresh.coord.step()
    assert fresh.shutdown.running is True, "the control: without the restore one observation is one"


def test_guard_state_round_trips_through_json_and_tolerates_an_empty_one() -> None:
    import json

    h = _make_coordinator()
    h.coord._consec_high_gn = 2
    h.coord._initial_policy_loss = 2.5
    h.trainer.skipped_steps = 4
    state = json.loads(json.dumps(h.coord.guard_state()))
    other = _make_coordinator()
    other.coord.restore_guard_state(state)
    assert other.coord._consec_high_gn == 2 and other.coord._initial_policy_loss == 2.5
    assert other.trainer.skipped_steps == 4
    other.coord.restore_guard_state({})  # a pre-field sidecar: nothing to restore, nothing raised
    # A pre-R362 sidecar's sealbot ring is ignored, not refused.
    other.coord.restore_guard_state({"wr_history": [[3000, 0.4]], "wr_history_rung": "sealbot_d5"})
    assert not hasattr(other.coord, "_wr_history")


def test_draw_rate_gate_default_off_does_not_fire() -> None:
    """On the EXPLICITLY disarmed posture a high draw rate NEVER fires. Bites a gate that ships
    hot against the config the operator actually wrote."""
    pool = DrivablePoolStub(draw_counts=(99, 100))
    cfg = _make_config()  # draw_rate_abort is None — EXPLICITLY off
    h = _make_coordinator(pool=pool, config=cfg)
    _drive_until_stopped(h, cap=6)
    assert h.shutdown.running is True, (
        "a `null` draw_rate_abort means the gate cannot fire, however bad the draw rate"
    )


# O-22 — emission wiring + heartbeat beats + monitor_gates
def test_log_interval_emits_training_step() -> None:
    """One step() crossing a log_interval boundary emits exactly one `training_step`, one
    `iteration_complete` and one `monitor_gates` summary. Bites alert rules with no live
    payload producer."""
    beat = BeatSpy()
    h = _make_coordinator(heartbeat=beat)
    h.pool.games_completed = 5
    h.coord.step()
    assert len(h.sink.named("training_step")) == 1
    assert len(h.sink.named("iteration_complete")) == 1
    assert len(h.sink.named("monitor_gates")) == 1, (
        "the LAW-18 monitor_gates per-gate summary must emit once per log_interval"
    )


def test_iteration_complete_carries_both_rate_gap_metrics() -> None:
    """The two rate gap metrics share ONE hook and both come from the coordinator's OWN
    counters over the same run clock. Producer-tested so the cutover floors have a live
    emitter, and cross-pinned: sph/gph must equal steps/games, which bites a hardcoded value, a
    wrong counter and a dropped injection."""
    h = _make_coordinator()
    # Pin the run clock 60 s after start: real elapsed here is MICROSECONDS, so the two now()
    # reads inside one emission would dominate the rates.
    started = h.coord._run_started
    h.coord._clock = SimpleNamespace(now=lambda: started + 60.0,
                                     sleep=lambda _s: None)
    h.pool.games_completed = 5
    h.coord.step()
    event = h.sink.named("iteration_complete")[-1]
    sph = event["steps_per_hour"]
    gph = event["games_per_hour"]
    assert isinstance(sph, float) and sph > 0.0, (
        f"steps_per_hour must be a LIVE measurement on this drive, got {sph!r}"
    )
    steps, games = h.trainer.step, h.pool.games_completed
    assert steps > 0 and games > 0 and gph > 0
    assert sph == round(steps / 60.0 * 3600.0, 1), (
        f"steps_per_hour must be the coordinator's own step counter over the run clock: "
        f"sph={sph} steps={steps}"
    )
    assert gph == round(games / 60.0 * 3600.0, 1), (
        f"games_per_hour must share the same clock: gph={gph} games={games}"
    )


def test_log_interval_boundaries_are_evaluated_per_training_step() -> None:
    """With `log_interval=5` and a burst of 4, 20 training steps must produce EXACTLY 4
    `training_step` + `monitor_gates` emissions at 5/10/15/20. Bites the once-per-burst
    boundary test, which hits a boundary only when the post-burst step is an exact multiple —
    here just step 20, thinning the stream and both gates' sampling by roughly the burst.
    `iteration_complete` is decoupled and emits per burst, at `[4, 8, 12, 16, 20]`."""
    cfg = _make_config(log_interval=5, max_train_burst=4, training_steps_per_game=4.0,
                       draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None)
    h = _make_coordinator(config=cfg)
    for _ in range(5):
        h.pool.games_completed += 5
        h.coord.step()

    assert h.trainer.step == 20, "5 outer iterations × burst 4 must run 20 training steps"
    assert [e["step"] for e in h.sink.named("training_step")] == [5, 10, 15, 20], (
        "training_step stays log_interval-gated (R210: training_step alerting stays gated)"
    )
    assert [e["step"] for e in h.sink.named("iteration_complete")] == [4, 8, 12, 16, 20], (
        "iteration_complete emits per coordinator step (per burst) after R210's decoupling, "
        "NOT per log_interval boundary. The step value is the post-burst _train_step."
    )
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [5, 10, 15, 20], (
        "monitor_gates rides monitor.gate_interval (R242), which `_make_config` mirrors onto "
        "log_interval here exactly as every committed config does"
    )


def test_gate_interval_boundaries_are_evaluated_per_training_step() -> None:
    """The GATE-INTERVAL twin of the row above: the per-training-step evaluation property has
    to hold SEPARATELY on each knob. `gate_interval=5` with `log_interval=1000` and a burst of
    4 must give EXACTLY 4 summaries at 5/10/15/20 and ZERO `training_step` events; testing once
    per burst would hit only step 20 and stretch the `consec` window by the mean burst."""
    cfg = _make_config(log_interval=1000, gate_interval=5, max_train_burst=4,
                       training_steps_per_game=4.0, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=None)
    h = _make_coordinator(config=cfg)
    for _ in range(5):
        h.pool.games_completed += 5
        h.coord.step()

    assert h.trainer.step == 20, "5 outer iterations × burst 4 must run 20 training steps"
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [5, 10, 15, 20], (
        "the gate boundary is tested PER TRAINING STEP; once per burst would give [20] alone"
    )
    assert h.sink.named("training_step") == [], (
        "and narration is silent throughout — the two knobs are independent (R242)"
    )


def test_gate_sampling_cadence_follows_gate_interval_not_the_burst() -> None:
    """The live-producer gate must sample once per GATE-INTERVAL BOUNDARY, not once per outer
    iteration: with `gate_interval=5`, burst 4 and a sustained 0.9 draw rate it collects its
    3rd sample at step 15 and fires THERE, where a once-per-burst implementation could not have
    fired yet. The subject used to be `log_interval`, and that identity was the defect."""
    pool = DrivablePoolStub(draw_counts=(90, 100))
    cfg = _make_config(log_interval=5, gate_interval=5, max_train_burst=4,
                       training_steps_per_game=4.0,
                       draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                        N_pool_min=10, consec=3),
                       hard_gn_threshold=1e9)
    h = _make_coordinator(pool=pool, config=cfg)
    _drive_until_stopped(h, cap=8)

    assert h.shutdown.running is False
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and "draw" in str(aborts[0])
    assert aborts[0]["step"] == 15, (
        f"the 3rd gate sample lands at step 15 (boundaries 5/10/15), got {aborts[0]['step']}"
    )
    gates = h.sink.named("monitor_gates")
    at_fire = next(e for e in gates if e["step"] == 15)
    assert [e["step"] for e in gates] == [5, 10, 15], (
        "one gate summary per gate_interval boundary and NO MORE — the fire stops the run, so "
        "a 4th summary would mean the burst kept gating past the abort"
    )
    assert at_fire["gates"]["draw_rate_collapse"]["checks"] == 3, (
        "exactly 3 gate samples were taken — one per gate_interval boundary, not per burst"
    )


def test_grad_norm_gate_fires_with_the_uniform_contract() -> None:
    """The `grad_norm_hard_abort` producer test — the kept gate's DECISION: a sustained grad
    norm above the threshold for `hard_gn_min_steps` consecutive steps stops the run AND emits
    ONE `hard_abort` event naming the rule. It only wrote a log line before, so the one
    unconditionally-active hard abort was invisible in the ONE channel."""
    cfg = _make_config(hard_gn_threshold=0.5, hard_gn_min_steps=3)
    h = _make_coordinator(config=cfg)
    h.trainer._gn = 10.0                              # sustained instability
    _drive_until_stopped(h)

    assert h.shutdown.running is False, "3 consecutive high-gn steps must hard-abort"
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1, f"exactly one abort decision, got {aborts}"
    assert aborts[0]["rule"] == "grad_norm_hard_abort"
    assert "grad" in str(aborts[0]).lower() and aborts[0]["step"] == 3
    assert h.sink.named("monitor_gates")[-1]["gates"]["grad_norm_hard_abort"]["fires"] == 1


def test_grad_norm_gate_does_not_fire_below_the_consecutive_count() -> None:
    """A single high-gn step (the consecutive counter reset by a healthy step) must NOT fire;
    only a sustained run of `hard_gn_min_steps` does. Bites a gate that aborts on one spike."""
    cfg = _make_config(hard_gn_threshold=0.5, hard_gn_min_steps=3)
    h = _make_coordinator(config=cfg)
    for gn in (10.0, 0.1, 10.0, 0.1):
        h.trainer._gn = gn
        h.pool.games_completed += 5
        h.coord.step()
    assert h.shutdown.running is True
    assert h.sink.named("hard_abort") == []


def test_step_loop_beats() -> None:
    """The `heartbeat.train_step` producer test: the step loop beats at step() entry AND once
    per burst training step. Bites a step loop the watchdog cannot see."""
    beat = BeatSpy()
    h = _make_coordinator(heartbeat=beat)
    h.pool.games_completed = 5
    outcome = h.coord.step()
    train_beats = [b for b in beat.beats if b == "train_step"]
    assert train_beats == ["train_step"] * (1 + outcome.steps_run), (
        f"train_step beats must equal 1 entry + {outcome.steps_run} burst steps, got {beat.beats}"
    )


# ══ RED-TEAM F12 / F9 — the result seam's step stamping + the watchdog counter consumer ═
def test_monitor_gates_publishes_the_watchdog_best_effort_counters() -> None:
    """`BestEffortCounters` had ZERO consumers despite documenting `snapshot()` as what the
    in-run summary events publish — a dead surface. The `monitor_gates` summary carries it now,
    so a degraded fire-path effect is readable IN RUN and not only in the fire's own event."""
    watchdog = SimpleNamespace(counters=SimpleNamespace(
        snapshot=lambda: {"watchdog_file_mirror": 3}))
    h = _make_coordinator()
    h.coord.heartbeat_watchdog = watchdog
    h.pool.games_completed = 5
    h.coord.step()
    summary = h.sink.named("monitor_gates")[-1]
    assert summary["watchdog_best_effort"] == {"watchdog_file_mirror": 3}


def test_a_malformed_guard_value_is_a_named_resume_error() -> None:
    from mantis.train.resume_state import ResumeStateError

    h = _make_coordinator()
    with pytest.raises(ResumeStateError, match="malformed"):
        h.coord.restore_guard_state({"consec_high_gn": "three"})
