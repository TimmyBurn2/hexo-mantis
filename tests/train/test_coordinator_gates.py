"""⊕ O-06 (+ O-22, O-03 wiring, LAW-18 monitor_gates) — the coordinator gate seam.

RED-at-import until IMPL writes `mantis.monitor.rules` + `mantis.monitor.config`. ORACLE-FIRST
(⊕): the top-level `import mantis.monitor.rules` raises ModuleNotFoundError before any port
code exists; the file goes GREEN only when IMPL wires the coordinator seam.

The centrepiece (MUST-1, O-06 / P-06): the sealbot-WR consumer lives at the ASYNC eval-RESULT
seam `StepCoordinator.on_eval_round_complete(result)` — the new-side twin of old
`step_coordinator.py` L1168-1200 (the `_pending_eval_result` drain), NEVER the eval kick
return. The fake eval pipeline's `run_evaluation` (kick) returns an ack WITHOUT `wr_sealbot`;
the test asserts step() consumes NOTHING from the kick and makes ZERO blocking calls
(`drain_pending` spy == 0). Completed results are delivered ONLY via a direct
`on_eval_round_complete(...)` call (simulating WP11-A's non-blocking drain callback).

Also covered: O-03 draw-rate gate wiring at log_interval cadence (fires on the LIVE
producer), O-22 emission wiring (`training_step` + `iteration_complete` + `monitor_gates`
once per log_interval boundary), and the `train_step` heartbeat beats. (O-04 stride5-spam was
REMOVED at close-out per operator directive B.)

Sealbot posture (operator G-3): the default `MonitorConfig` ships `wr_hard_abort_enabled=False`
= WARN-ONLY — a sustained collapse emits a visible `sealbot_wr_warn` and does NOT stop the
run; the one-field flip `wr_hard_abort_enabled=True` restores the A/B/C hard-abort as a
CAPABILITY (test_sealbot_hard_abort_capability_when_enabled).

IMPL API constraints (see ORACLE_NOTES): StepCoordinator.__init__ gains `heartbeat`,
`monitor_cfg`, and (for close-out) `heartbeat_watchdog` kwargs; `on_eval_round_complete` and
`_wr_history` are added; a hard-abort gate emits a `hard_abort` event naming its rule.

>300 justify: one coordinator seam, one set of fakes (pool / trainer / buffer / async eval
pipeline / beat + sink spies) shared by every gate row — splitting the file would duplicate
the harness and let the two halves drift apart, which is the failure this suite exists to
catch. Added post-review (F-3/F-4): the non-degenerate cadence rows (`log_interval > 1` AND
`max_train_burst > 1`) and the grad-norm decision row.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import check_sealbot_wr_hard_abort  # noqa: F401 — RED-at-import anchor
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

#: WPMINT Phase K-A stage 0: the four drain caps are `monitor.drain.*` (R93/DR-11), so a
#: harness reads them from a MINTED config rather than restating them — the same rule the
#: rest of this file's coordinator config now follows (see `_make_config`).
_DRAIN_CAPS = resolve_drain_caps(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor)
#: WPMINT Phase K-B: the builder's fourth config-authored parameter, from the same minted
#: config — the 19 coordinator knobs are `train.*` keys now, not builder literals.
_KNOBS = resolve_coordinator_knobs(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").train)


# ── fakes ─────────────────────────────────────────────────────────────────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class FakePool:
    def __init__(self, *, stride5=1, draw_counts=(0, 0)) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True                 # → iteration_complete cluster stats are None
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []   # empty → emit_axis_distribution returns early
        self._stride5 = stride5
        self._draw_counts = (int(draw_counts[0]), int(draw_counts[1]))
        self.counts_calls = 0

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        # WPMINT Phase DS (R92): the `WorkerPoolLike` surface serves RAW COUNTS
        # `(draws, completed)` and takes no evidence bar — the bar is applied at the abort
        # decision (`pooled_draw_rate`). The statistic's own oracle is
        # tests/selfplay/test_drawrate_pooled_statistic.py.
        self.counts_calls += 1
        return self._draw_counts

    def current_stride5_p90(self) -> int:
        return self._stride5

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


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


class FakeBuffer:
    def __init__(self, size: int = 1000, capacity: int = 100_000) -> None:
        self.size = size
        self.capacity = capacity

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        # The grid route's sampler (WPTS dispatcher); rows are opaque to FakeTrainer.
        return (None,) * 9


class FakeEvalPipeline:
    """Async eval pipeline: `run_evaluation` (the KICK) returns an ack; `drain_pending` is a
    spy that MUST stay uncalled from step() (a blocking drain in step() = the run3 wedge)."""

    def __init__(self, kick_result: dict) -> None:
        self.kick_result = kick_result
        self.run_calls = 0
        self.drain_calls = 0
        self.poll_calls = 0

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.run_calls += 1
        return self.kick_result

    def drain_pending(self):
        self.drain_calls += 1
        return None

    def poll_completed(self):
        # WP11-A: step()'s non-blocking poll at the top of every iteration. This fixture
        # never has a completed round ready — the kick-return-is-never-consumed-for-WR
        # invariant this test pins is entirely about the KICK ack, not this seam.
        self.poll_calls += 1
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
    """DERIVED from the production builder, never a hand-written 24-kwarg census.

    WPMINT Phase K-A stage 0. This used to restate every field; ten test files restated the
    same ones, so they agreed with `StepCoordinatorConfig` by maintenance rather than by
    construction and every new coordinator knob cost ten edits. The shape here is the one
    `tests/train/test_drawrate_abort_threading.py` and
    `tests/config/test_drawrate_arming_authority.py` already used and the census named as
    the in-repo precedent: build with the shipped builder, state only this file's deltas.
    `stop_step`/`draw_rate_abort` are passed EXPLICITLY (`None` is the disarmed posture,
    and this harness is not about the draw-rate abort) because the builder gives them no
    default and neither does this factory.
    """
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        **{"eval_interval": 1, "log_interval": 1, "min_buf_size": 10, **overrides},
    )


def _make_coordinator(*, pool=None, config=None, eval_pipeline=None, heartbeat=None,
                      monitor_cfg=None):
    pool = pool or FakePool()
    trainer = FakeTrainer()
    buffer = FakeBuffer()
    shutdown = ShutdownState()
    sink = SpySink()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None,
        # WPTS/TD-1: the straight arm resolves its route from the DECLARED identity — these
        # unit drives declare the grid identity FakeBuffer's sampler serves.
        config=config or _make_config(),
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
        train_cfg={}, mixing_cfg={},
        sink=sink, heartbeat=heartbeat, monitor_cfg=monitor_cfg or MonitorConfig(),
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


# ══ O-06 — sealbot at the ASYNC RESULT seam ══════════════════════════════════════════
def test_step_does_not_consume_the_kick_return_and_never_blocks() -> None:
    """O-06 / P-06 — the eval KICK returns a collapse `wr_sealbot=0.01` that WOULD fire if
    (wrongly) consumed; step() must NOT append it to `_wr_history`, must NOT fire, and must
    make ZERO blocking calls (`drain_pending` spy == 0). Bites a gate wired to the kick return
    and a blocking eval re-entering the step path (the run3 wedge)."""
    pipe = FakeEvalPipeline(kick_result={"step": 30000, "wr_sealbot": 0.01})
    h = _make_coordinator(eval_pipeline=pipe)
    h.pool.games_completed = 5
    h.coord.step()
    assert pipe.run_calls >= 1, "the eval kick must have fired at the boundary"
    assert pipe.drain_calls == 0, "step() must make ZERO blocking drain calls"
    assert list(getattr(h.coord, "_wr_history", [])) == [], (
        "the kick return must NEVER be consumed for WR (the masked-dead-gate class)"
    )
    assert h.shutdown.running is True, "no fire may come from the kick return"


def test_sealbot_default_is_warn_only_and_does_not_shut_down() -> None:
    """O-06 / P-06 (operator G-3) — the manifest `sealbot_wr_warn` producer test, the SHIPPED
    DEFAULT posture. Delivering N consecutive low-WR results via `on_eval_round_complete`
    (WP11-A's drain callback) on a run whose `MonitorConfig` ships `wr_hard_abort_enabled=False`
    emits a VISIBLE `sealbot_wr_warn` carrying the de-diagnosed trajectory fact (Objective-A
    off-distribution OR Objective-B strength regression — never asserting one) and does NOT
    stop the run. Warn-only that emits NOTHING would be the silently-disabled class."""
    h = _make_coordinator()                          # default MonitorConfig() = warn-only
    assert h.coord.monitor_cfg.wr_hard_abort_enabled is False, "shipped default is warn-only"
    for _ in range(3):                              # 3 consecutive < 0.05 past min_step 15000
        h.coord.on_eval_round_complete({"step": 30000, "wr_sealbot": 0.01})
    assert h.shutdown.running is True, "warn-only must NOT stop the run (operator G-3)"
    assert h.sink.named("hard_abort") == [], "warn-only must emit no hard_abort"
    warns = h.sink.named("sealbot_wr_warn")
    assert warns, "a sustained collapse must emit a VISIBLE sealbot_wr_warn (never silent)"
    msg = " ".join(str(e) for e in warns)
    assert "Objective-A" in msg and "Objective-B" in msg, (
        "the warn message must be DE-DIAGNOSED (name BOTH mechanisms, assert neither): " + msg
    )
    # The warn is COUNTED (LAW-18), incrementing each round the collapse persists —
    # here round 2 (trigger A, 2 consecutive) and round 3 (trigger C) both warn.
    assert warns[-1]["warn_total"] == len(warns) == 2


def test_sealbot_hard_abort_capability_when_enabled() -> None:
    """O-06 / P-06 — the DECISION-PARITY CAPABILITY. With the one-field operator flip
    `wr_hard_abort_enabled=True`, the SAME N consecutive low-WR results fire the hard-abort
    exactly as before: running=False + a de-diagnosed `hard_abort` event. The A/B/C triggers
    are unchanged — only the DEFAULT disposition moved (operator G-3)."""
    h = _make_coordinator(monitor_cfg=MonitorConfig(wr_hard_abort_enabled=True))
    for _ in range(3):
        h.coord.on_eval_round_complete({"step": 30000, "wr_sealbot": 0.01})
    assert h.shutdown.running is False, "with the flag True a sustained collapse must hard-abort"
    aborts = h.sink.named("hard_abort")
    assert aborts and aborts[-1]["rule"] == "sealbot_wr_abort"
    msg = " ".join(str(e) for e in aborts)
    assert "HARD-ABORT" in msg and "Objective-A" in msg and "Objective-B" in msg
    assert h.sink.named("sealbot_wr_warn") == [], "the hard path must not also warn"


def test_sealbot_single_low_result_does_not_fire_or_warn() -> None:
    """O-06 — a single low result (a recovering dip) is not a sustained collapse: it must NOT
    fire AND must NOT warn (no trajectory event at all). Bites a warn on a single dip."""
    h = _make_coordinator()
    h.coord.on_eval_round_complete({"step": 30000, "wr_sealbot": 0.5})
    h.coord.on_eval_round_complete({"step": 31000, "wr_sealbot": 0.5})
    h.coord.on_eval_round_complete({"step": 32000, "wr_sealbot": 0.01})  # one dip
    assert h.shutdown.running is True
    assert h.sink.named("sealbot_wr_warn") == [], "a single dip must not warn"


def test_sealbot_absent_key_skips_and_counts() -> None:
    """O-06 / P-06 — a result with `wr_sealbot` absent/None ⇒ exactly one
    `sealbot_wr_gate_skipped` event per delivered round + a counter, and ZERO fires (LAW-18: the
    inert gate is loud, never silently dead until WP11-A lands the producer)."""
    h = _make_coordinator()
    h.coord.on_eval_round_complete({"step": 30000})                 # no wr_sealbot
    h.coord.on_eval_round_complete({"step": 31000, "wr_sealbot": None})
    skips = h.sink.named("sealbot_wr_gate_skipped")
    assert len(skips) == 2, "one skip event per delivered round with an absent/None key"
    assert h.shutdown.running is True, "a skipped round must never fire"


# ══ O-03 — draw-rate gate WIRING (LIVE producer, log_interval cadence) ════════════════
# (O-04 stride5-spam gate REMOVED at close-out per operator directive B.)
def test_draw_rate_gate_fires_on_live_producer() -> None:
    """O-03 — the manifest `draw_rate_collapse` producer test. Keyed on the LIVE
    `pooled_draw_rate(pooled_draw_counts(), N_pool_min=…)` (never the NaN
    `draw_target_fraction`): a sustained 0.9 pooled draw rate over sufficient evidence, past
    min_step, fires. Grad-norm is quiet, so the fire is draw-rate."""
    pool = FakePool(draw_counts=(90, 100))
    cfg = _make_config(draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                        N_pool_min=10, consec=3))
    h = _make_coordinator(pool=pool, config=cfg)
    _drive_until_stopped(h)
    assert h.shutdown.running is False, "a sustained pool draw-rate collapse must hard-abort"
    aborts = h.sink.named("hard_abort")
    assert aborts and any("draw" in str(e) for e in aborts), (
        f"the draw-rate gate must emit a hard_abort naming its rule: {aborts}"
    )


def test_draw_rate_gate_default_off_does_not_fire() -> None:
    """O-03 — on the EXPLICITLY disarmed posture (`train.draw_rate_abort: null`, WPAX Phase
    D — it used to be the code-side `threshold 0.0`), a high draw rate NEVER fires. Bites a
    gate that ships hot against the config the operator actually wrote."""
    pool = FakePool(draw_counts=(99, 100))
    cfg = _make_config()  # draw_rate_abort is None — EXPLICITLY off
    h = _make_coordinator(pool=pool, config=cfg)
    _drive_until_stopped(h, cap=6)
    assert h.shutdown.running is True, (
        "a `null` draw_rate_abort means the gate cannot fire, however bad the draw rate"
    )


# ══ O-22 — emission wiring + heartbeat beats + monitor_gates ══════════════════════════
def test_log_interval_emits_training_step() -> None:
    """O-22 / P-22 — the manifest `warn.training_step_alerts` producer test. One step() crossing
    a log_interval boundary emits exactly one `training_step` + one `iteration_complete` + one
    `monitor_gates` summary. Bites alert rules with no live payload producer (LAW-07)."""
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
    """R29 gap metrics (a) games/hr + (b) steps/hr share ONE hook — `iteration_complete` —
    and both come from the coordinator's OWN counters over the same run clock (WPBOX CB-3
    wiring). Producer-tested so the cutover floors have a live emitter, and the two rates
    are cross-pinned: sph/gph must equal steps/games (same elapsed, tolerance for the two
    now() reads) — which bites a hardcoded value, a wrong counter, and a dropped injection
    (None = NOT MEASURED would fail the float assert here, where the producer IS injected).
    """
    h = _make_coordinator()
    # Pin the run clock 60 s after start: real elapsed in this rig is MICROSECONDS, so the
    # two now() reads inside one emission would dominate the rates; a frozen clock makes
    # the cross-identity exact (modulo the emitter's 1-decimal rounding).
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
    """O-22 (F-3 regression) — with `log_interval=5` and a burst of 4 TRAINING steps per
    outer iteration, 20 training steps must produce EXACTLY 4 `training_step` + `monitor_gates`
    emissions, at steps 5/10/15/20.

    Bites the once-per-burst boundary test: evaluating `step % log_interval` only after the
    burst (when `_train_step` has already advanced by up to `max_train_burst`) hits a boundary
    only when the post-burst step happens to be an exact multiple — here just step 20, i.e. 1
    emission instead of 4, thinning the LAW-18 stream and both gates' sampling by ~the burst.

    WP12R Step 3 narration (R210, DESIGN §3.6): `iteration_complete` was DECOUPLED from
    `log_interval` — it now emits per coordinator step (per burst), NOT per `log_interval`
    boundary. So with 5 outer iterations × burst 4, `iteration_complete` emits 5 times at the
    post-burst step values `[4, 8, 12, 16, 20]` (the `_train_step` value at the O6 return),
    while `training_step`/`monitor_gates` stay `log_interval`-gated at `[5, 10, 15, 20]`.
    """
    cfg = _make_config(log_interval=5, max_train_burst=4, training_steps_per_game=4.0,
                       draw_rate_abort=None)
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
        "monitor_gates stays log_interval-gated (R210: training_step alerting stays gated)"
    )


def test_gate_sampling_cadence_follows_log_interval_not_the_burst() -> None:
    """O-03 (F-3 regression) — the live-producer gate must sample once per log_interval
    BOUNDARY, not once per outer iteration. With `log_interval=5`, burst 4 and a sustained
    draw rate of 0.9 (>= 0.4, consec 3), the draw-rate gate collects its 3rd sample at step 15
    and fires THERE. A once-per-burst implementation would sample at most at step 20 and could
    not have fired yet — the `consec` window silently stretched by the burst factor."""
    pool = FakePool(draw_counts=(90, 100))
    cfg = _make_config(log_interval=5, max_train_burst=4, training_steps_per_game=4.0,
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
    assert [e["step"] for e in gates] == [5, 10, 15], (
        "one gate summary per boundary, up to the boundary that fired"
    )
    assert gates[-1]["gates"]["draw_rate_collapse"]["checks"] == 3, (
        "exactly 3 gate samples were taken — one per log_interval boundary, not per burst"
    )


def test_grad_norm_gate_fires_with_the_uniform_contract() -> None:
    """The manifest `grad_norm_hard_abort` producer test (F-4) — the KEPT WP10 gate's
    DECISION: a sustained grad norm above `hard_gn_threshold` for `hard_gn_min_steps`
    consecutive training steps stops the run AND emits ONE `hard_abort` event naming the rule,
    exactly like every WP13-A gate (before F-4 it only wrote a log line, so the one
    unconditionally-active hard-abort was invisible in the ONE channel)."""
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
    """F-4 companion — a single high-gn step (the consecutive counter reset by a healthy
    step) must NOT fire; only a sustained run of `hard_gn_min_steps` does. Bites a gate that
    aborts on one spike."""
    cfg = _make_config(hard_gn_threshold=0.5, hard_gn_min_steps=3)
    h = _make_coordinator(config=cfg)
    for gn in (10.0, 0.1, 10.0, 0.1):
        h.trainer._gn = gn
        h.pool.games_completed += 5
        h.coord.step()
    assert h.shutdown.running is True
    assert h.sink.named("hard_abort") == []


def test_step_loop_beats() -> None:
    """O-22 — the manifest `heartbeat.train_step` producer test. The step loop beats `train_step`
    at step() entry AND once per burst training step (entry + burst iterations). Bites a step
    loop the watchdog cannot see."""
    beat = BeatSpy()
    h = _make_coordinator(heartbeat=beat)
    h.pool.games_completed = 5
    outcome = h.coord.step()
    train_beats = [b for b in beat.beats if b == "train_step"]
    assert train_beats == ["train_step"] * (1 + outcome.steps_run), (
        f"train_step beats must equal 1 entry + {outcome.steps_run} burst steps, got {beat.beats}"
    )


# ══ RED-TEAM F12 / F9 — the result seam's step stamping + the watchdog counter consumer ═
def test_sealbot_result_at_step_zero_is_not_rewritten() -> None:
    """RED-TEAM F12 — `payload.get("step") or self._train_step` rewrote a legitimate step 0
    (falsy!) to the current train step, mis-stamping the WR ring. Absence, not falsiness,
    selects the fallback."""
    h = _make_coordinator()
    h.coord._train_step = 777
    h.coord.on_eval_round_complete({"step": 0, "wr_sealbot": 0.25})
    assert list(h.coord._wr_history) == [(0, 0.25)], (
        f"a step-0 result must keep step 0, got {list(h.coord._wr_history)}"
    )


def test_monitor_gates_publishes_the_watchdog_best_effort_counters() -> None:
    """RED-TEAM F9 — `BestEffortCounters` had ZERO consumers despite documenting `snapshot()`
    as 'what the LAW-18 in-run summary events publish' (a LAW-08 dead surface). The
    `monitor_gates` summary now carries it, so a degraded fire-path effect is readable IN RUN
    and not only in the fire's own event."""
    watchdog = SimpleNamespace(counters=SimpleNamespace(
        snapshot=lambda: {"watchdog_file_mirror": 3}))
    h = _make_coordinator()
    h.coord.heartbeat_watchdog = watchdog
    h.pool.games_completed = 5
    h.coord.step()
    summary = h.sink.named("monitor_gates")[-1]
    assert summary["watchdog_best_effort"] == {"watchdog_file_mirror": 3}
