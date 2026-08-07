# >300 justify (R8), and NO tally is stated (G-DFIX-4 / R192(e), derive-or-delete). ONE claim
# — "arming rides monitor.gate_interval and narration rides train.log_interval, and neither
# decides the other" — asserted from many angles over ONE set of fakes and ONE builder-derived
# config factory. The pins are only evidence BECAUSE they share that factory: each states
# the two knobs APART, and a second file would need its own copy of the harness, which would
# then be a second authority on what a drive at "gate_interval=4, log_interval=5" even is.
# That is the drift this suite exists to catch, so the harness stays with the rows it drives.
# The RED-TEAM close (P7-P10) added rows and no harness: each of the four was a mutation the
# committed suite let through at zero failures, and every one of them is driven through the
# same `_config` / `_coordinator` / `_drive` triple the original rows use.
"""⊕ R242 / ADJ-D12 — the ARMING cadence is `monitor.gate_interval`, not `train.log_interval`.

THE DEFECT, as measured. `coordinator/step.py::_run_log_interval` early-returned unless
`self._train_step % cfg.log_interval == 0`, and BOTH the live hard-abort gates
(`_run_hard_abort_gates`, whose `draw_rate_collapse` row gate 12 audits ARMED on
`configs/run5.yaml`) and the LAW-18 `monitor_gates` summary sat downstream of that guard. At
run5's minted `train.log_interval: 1000` that means: no draw-rate observation could be taken,
and no `monitor_gates` event could exist, before training step 1000. Armed machinery with a
blind first kilometre — and the instrument that would have made the deadness readable was
switched off by the same knob.

R242 splits the two. `log_interval` is NARRATION (`training_step`, the 4 WARN rules, the axis
distribution); `monitor.gate_interval` is ARMING. Every committed config mints the two EQUAL,
so no shipped behaviour and no armed value moves as this lands — the re-scaled stride and the
`consec` re-derived in gate-interval units are mint-prereg rows, not code. What lands is the
CAPABILITY to state them apart, and the pins below are what make the split real rather than
described. Each names the mutation that reds it.

P1 gate visibility BELOW the narration boundary — the defect's direct inverse.
P2 the abort fires on exactly the Nth OBSERVATION under the new cadence (observations
   counted, not just the fire).
P3 the gate's check count is `steps // gate_interval` and NOT `steps // log_interval`.
P4 narration stays `log_interval`-gated — the half of R210 that survives R242.
P5 `iteration_complete` still emits per coordinator step, independent of BOTH knobs (R210).
P6 no code-side default anywhere: a config MISSING `monitor.gate_interval` fails to load.

The RED-TEAM close added four more, each named by the 0-failure mutation it now reds:
P7 the BUILDER's `gate_interval` parameter carries no default (the schema half is P6; this
   is the second-authority half, R1/MF-2 Attack B).
P8 a fire on a NON-FINAL burst iteration survives into `StepOutcome` (the OR-fold).
P9 the gate boundary runs and publishes with NO `loss_info` at all — the VALUE of the
   decoupling P1 pins only the SIGNATURE of.
P10 a SKIPPED boundary neither advances nor RESETS `consec` — the true cadence semantic,
   after this bundle shipped a false one in three places.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG_PATH = _REPO / "configs" / "dev_example.yaml"
_DEV_CONFIG = load_config(_DEV_CONFIG_PATH)
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval

#: run5's own minted narration cadence. Named, not invented: the defect is about what happens
#: BEFORE this many training steps, so every drive below that is "large log_interval" uses
#: exactly the number the production config ships.
_RUN5_LOG_INTERVAL = 1000


# ── fakes (the harness shape tests/train/test_coordinator_gates.py established) ──────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self, *, draw_counts: tuple[int, int] = (0, 0)) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self._draw_counts = (int(draw_counts[0]), int(draw_counts[1]))

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return self._draw_counts

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def save_checkpoint(self, loss_info) -> None:
        return None


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
        return [e for e in self.events if e.get("event") == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder (the WPMINT K-A/K-B precedent).

    Unlike the sibling harnesses this one does NOT mirror `gate_interval` onto `log_interval`:
    every drive here states BOTH knobs, because stating them apart is the whole subject.
    """
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **{"eval_interval": 10**9, "min_buf_size": 10, "max_train_burst": 4,
           "training_steps_per_game": 4.0, "hard_gn_threshold": 1e9, **overrides},
    )


def _coordinator(*, config: StepCoordinatorConfig, pool: _Pool | None = None):
    pool = pool or _Pool()
    trainer, buffer, sink = _Trainer(), _Buffer(), _Sink()
    shutdown = ShutdownState()
    coord = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        full_config={"identity": {"encoding": "v6_live2_ls", "representation": "grid"}},
        train_cfg={}, mixing_cfg={}, sink=sink, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, trainer=trainer, sink=sink,
                           shutdown=shutdown, config=config)


def _drive(h, *, outer: int, games_per_step: int = 5) -> None:
    """`outer` coordinator steps, stopping early if a gate stops the run."""
    for _ in range(outer):
        if not h.shutdown.running:
            break
        h.pool.games_completed += games_per_step
        h.coord.step()


def _checks(h, gate: str = "draw_rate_collapse") -> int:
    summaries = h.sink.named("monitor_gates")
    return 0 if not summaries else summaries[-1]["gates"][gate]["checks"]


# ══ P1 — gate visibility BELOW the narration boundary (the defect's direct inverse) ═════
def test_p1_gates_are_visible_and_advancing_far_below_the_log_interval_boundary() -> None:
    """P1 — at run5's `log_interval=1000` with `gate_interval=1`, the gates run and publish on
    every training step while ZERO `training_step` events have been emitted.

    At HEAD this was impossible by construction: both lived behind the `log_interval` guard,
    so twenty training steps produced zero `monitor_gates` events and zero gate checks.

    MUTATION THAT REDS IT: move `_run_hard_abort_gates` / `_emit_monitor_gates` back inside
    `_run_log_interval` (the HEAD shape) — `monitor_gates` goes empty and `checks` stays 0.
    """
    h = _coordinator(config=_config(log_interval=_RUN5_LOG_INTERVAL, gate_interval=1))
    _drive(h, outer=5)

    assert h.trainer.step == 20, "5 outer iterations x burst 4 must run 20 training steps"
    summaries = h.sink.named("monitor_gates")
    assert [e["step"] for e in summaries] == list(range(1, 21)), (
        "the LAW-18 gate summary must emit on every gate_interval boundary — here every "
        f"training step — and all 20 land far below log_interval 1000; got {summaries}"
    )
    assert _checks(h) == 20, (
        f"the draw-rate gate must have been CHECKED 20 times, not {_checks(h)}: an armed "
        "abort that cannot even take an observation for its first 1000 steps is the defect"
    )
    assert h.sink.named("training_step") == [], (
        "narration must still be silent here — this drive proves the gates are readable "
        "WITHOUT any log_interval boundary having been crossed"
    )

    # The gate half must not be conditioned on `loss_info`: its producer is the POOL, not the
    # trainer. Driven directly, with no loss dict anywhere in the call, it still publishes.
    before = len(h.sink.named("monitor_gates"))
    h.coord._run_gate_interval(h.config)
    assert len(h.sink.named("monitor_gates")) == before + 1, (
        "`_run_gate_interval` takes no `loss_info` and must not acquire one: a gate whose "
        "arming depends on the trainer having produced a loss dict is the same hidden "
        "coupling R242 removes"
    )


# ══ P2 — the abort fires on exactly the Nth OBSERVATION under the new cadence ═══════════
def test_p2_the_draw_rate_abort_fires_on_exactly_the_nth_gate_interval_observation() -> None:
    """P2 — armed `consec=3`, `gate_interval=2`, `log_interval=1000`: observations land at
    training steps 2/4/6 and the abort fires at step 6, on the THIRD one and not before.

    The observation COUNT is asserted, not merely the fire: a gate that fired for the right
    reason at the wrong cadence, or one that sampled twice per boundary, would both satisfy a
    fire-only assertion. The threshold/consec/min_step are chosen HERE and not read from a
    config — R242 moved no armed value, and this pin must not become a second authority over
    the ones the operator pre-registers.
    """
    pool = _Pool(draw_counts=(90, 100))
    cfg = _config(log_interval=_RUN5_LOG_INTERVAL, gate_interval=2,
                  draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                    N_pool_min=10, consec=3))
    h = _coordinator(config=cfg, pool=pool)
    _drive(h, outer=8)

    assert h.shutdown.running is False, (
        "a sustained pool draw-rate collapse must hard-abort on the gate cadence — at HEAD "
        "it could not fire at all before training step 1000"
    )
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and aborts[0]["rule"] == "draw_rate_collapse"
    assert aborts[0]["step"] == 6, (
        f"the 3rd observation lands at step 6 (boundaries 2/4/6), got {aborts[0]['step']}"
    )
    # The burst that fired runs to its end (`running=False` ends the OUTER loop, not the
    # in-flight burst — pre-existing, and why step 8 records a `hard_abort_after_stop`), so
    # the boundary list runs 2/4/6/8 and the fire is read at ITS OWN summary.
    boundaries = [e["step"] for e in h.sink.named("monitor_gates")]
    assert boundaries == [2, 4, 6, 8], f"one summary per gate_interval boundary; got {boundaries}"
    at_fire = next(e for e in h.sink.named("monitor_gates") if e["step"] == 6)
    assert at_fire["gates"]["draw_rate_collapse"]["checks"] == 3, (
        "exactly 3 observations had been taken when it fired — one per gate_interval "
        f"boundary — got {at_fire['gates']['draw_rate_collapse']['checks']}. `consec` counts "
        "OBSERVATIONS, so this number IS the abort's clock"
    )
    assert at_fire["gates"]["draw_rate_collapse"]["fires"] == 1


# ══ P3 — the check count follows gate_interval and NOT log_interval ═════════════════════
def test_p3_the_gate_check_count_follows_gate_interval_and_not_log_interval() -> None:
    """P3 — the anti-regression pin. With `gate_interval=4` and `log_interval=5` over 20
    training steps the gate is checked `20 // 4 = 5` times, which is NOT `20 // 5 = 4`.

    The two knobs are deliberately given values that DISAGREE on this drive, because equal
    values (the shipped posture) cannot distinguish the fixed code from the defect.

    MUTATION RUN, AND ITS RESULT: replacing `cfg.gate_interval` with `cfg.log_interval` in
    `_run_gate_interval`'s guard makes the count 4 and the boundaries [5, 10, 15, 20] — this
    test fails on both assertions.
    """
    h = _coordinator(config=_config(log_interval=5, gate_interval=4))
    _drive(h, outer=5)

    assert h.trainer.step == 20
    assert _checks(h) == 20 // 4 == 5, (
        f"the gate must be checked steps//gate_interval = 5 times, got {_checks(h)}. If this "
        "is 4 the guard is reading log_interval again"
    )
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [4, 8, 12, 16, 20], (
        "the gate summary rides gate_interval; [5, 10, 15, 20] would be the log_interval "
        "boundaries, i.e. the regression"
    )
    assert [e["step"] for e in h.sink.named("training_step")] == [5, 10, 15, 20], (
        "and narration rides log_interval on the SAME drive — the two cadences coexist"
    )


# ══ P4 — narration stays log_interval-gated (the half of R210 that survives) ════════════
def test_p4_narration_follows_log_interval_and_not_gate_interval() -> None:
    """P4 — with `log_interval=5` and `gate_interval=1` over 20 training steps there are 4
    `training_step` events, not 20. R242 supersedes R210 only in ARMING scope; "training_step
    alerting stays gated" is untouched, and this is the pin that keeps the split from
    over-reaching into narration.

    MUTATION THAT REDS IT: drop the `log_interval` guard from `_run_log_interval` (or point it
    at `gate_interval`) — the count becomes 20.
    """
    h = _coordinator(config=_config(log_interval=5, gate_interval=1))
    _drive(h, outer=5)

    steps = [e["step"] for e in h.sink.named("training_step")]
    assert steps == [5, 10, 15, 20], f"narration must ride log_interval alone; got {steps}"
    assert len(h.sink.named("monitor_gates")) == 20, (
        "while the gates ran on all 20 — the drive is only meaningful because the two "
        "cadences disagree here"
    )


# ══ P5 — R210 intact: iteration_complete is independent of BOTH knobs ═══════════════════
def test_p5_iteration_complete_still_emits_per_coordinator_step() -> None:
    """P5 — `iteration_complete` emits once per coordinator step (per burst) at
    `log_interval=1000` AND `gate_interval=1000`, i.e. with both boundaries un-crossed. R210's
    decoupling is not disturbed by R242's.

    MUTATION THAT REDS IT: re-couple `_emit_iteration_complete` to either boundary — the
    count drops from 5 to 0 on this drive.
    """
    h = _coordinator(config=_config(log_interval=_RUN5_LOG_INTERVAL,
                                    gate_interval=_RUN5_LOG_INTERVAL))
    _drive(h, outer=5)

    assert [e["step"] for e in h.sink.named("iteration_complete")] == [4, 8, 12, 16, 20], (
        "iteration_complete emits per coordinator step, INDEPENDENT of both cadences (R210)"
    )
    assert h.sink.named("training_step") == [] and h.sink.named("monitor_gates") == [], (
        "neither boundary was crossed on this drive, which is what makes the assertion above "
        "about independence rather than coincidence"
    )


# ══ P6 — no code-side default: a config MISSING the key fails to load ══════════════════
def test_p6_a_config_missing_monitor_gate_interval_fails_to_load(tmp_path) -> None:
    """P6 — R1/R242: the ARMING cadence has NO default anywhere, so a config that omits it is
    a LOAD FAILURE, not a run that quietly inherits a stride.

    The subject is a MISSING REQUIRED key, not `extra="forbid"`: the failure mode R242 guards
    against is a config saying nothing about arming and the code choosing for it. Driven
    against a real minted config with exactly that one key removed, through the real loader.

    MUTATION THAT REDS IT: give `MonitorSchemaConfig.gate_interval` any default at all (or
    let `compose_run` fall back to `log_interval`) — the load succeeds.
    """
    payload = yaml.safe_load(_DEV_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["monitor"].pop("gate_interval") == _GATE_INTERVAL, (
        "the minted config must carry the key for its removal to mean anything"
    )
    target = tmp_path / "no_gate_interval.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="gate_interval"):
        load_config(target)


def test_p6b_every_committed_config_mints_gate_interval_equal_to_its_log_interval() -> None:
    """P6 companion — the dispatcher rider, asserted rather than described: R242 authored a
    MECHANISM and moved NO armed value, and the way that is true is that every committed
    config mints `monitor.gate_interval == train.log_interval`.

    This is a PREREG BLANK wired to current effective behaviour. When the operator picks a
    real gate stride at mint prereg this test is expected to be RE-POINTED by that ruling —
    it is not a law that the two must agree forever, it is the record that they agree TODAY
    and therefore that this bundle changed no cadence. Deleting it instead of re-pointing it
    would erase the only in-repo evidence for that claim.

    F-P2B (R259 shakedown): the seventh committed config, `shakedown_20260807.yaml`, mints
    BOTH knobs to 100 — the equal-mint held only because MAIN ratified the ninth delta
    `train.log_interval 1000 -> 100` alongside `monitor.gate_interval 1000 -> 100`; the
    first mint carried gate_interval alone and this very assertion refused it. The count
    below ratchets 6 -> 7 so an eighth FLAT `configs/*.yaml` cannot slip past the equality
    sweep unseen — this file's glob is flat, so subdirectory/`.yml` shapes (legal per
    ADJ-13 F-1/R75) are the ONE discovery authority's subject (`discover_configs`, consumed
    by gates 7 and 12), not this sweep's.
    """
    configs = sorted((_REPO / "configs").glob("*.yaml"))
    assert len(configs) == 7, f"expected the seven committed configs, found {configs}"
    for path in configs:
        cfg = load_config(path)
        assert cfg.monitor.gate_interval == cfg.train.log_interval, (
            f"{path.name}: gate_interval {cfg.monitor.gate_interval} != log_interval "
            f"{cfg.train.log_interval}. R242 moved no armed value; a divergence here is an "
            "operator prereg decision and must arrive as one, not as an implementation edit"
        )


# ══ P7 — the BUILDER carries no default for the arming cadence (R1 / MF-2 Attack B) ═════
def test_p7_the_builder_takes_gate_interval_as_a_required_keyword_only_parameter() -> None:
    """P7 — `_step_coordinator_config`'s `gate_interval` parameter must have NO default.

    P6 pins the SCHEMA half (a config that omits the key fails to load). This is the BUILDER
    half, and they are different authorities: a `gate_interval: int = 1000` on this signature
    leaves every schema assertion, every `dataclasses.fields()` census and P6 itself GREEN,
    while any caller that forgets the argument silently inherits an ARMING cadence. That is
    exactly the migration MF-2 Attack B describes — the literal does not die, it moves from
    the builder BODY to the builder SIGNATURE — and `run.py`'s own docstring commits to it
    ("a required keyword-only parameter with no default").

    MUTATION RUN, AND ITS RESULT: the RED-TEAM gave the parameter a default and measured
    1267 collected, 0 failures over the full tier. Re-run with this pin in place it reds
    exactly this test. The schema-default case is caught four times over; the builder was not
    caught once.
    """
    param = inspect.signature(_step_coordinator_config).parameters.get("gate_interval")
    assert param is not None, (
        "`_step_coordinator_config` must take `gate_interval`: the ARMING cadence is a "
        "`monitor.*` config fact and arrives from `config.monitor.gate_interval`, never from "
        "a literal here and never from `knobs.log_interval`"
    )
    assert param.default is inspect.Parameter.empty, (
        f"gate_interval carries a parameter default ({param.default!r}) — a caller that omits "
        "it now inherits an arming cadence the config never authored (R1: the schema is the "
        "one authority), and the inherited posture would be invisible to every existing test"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


# ══ P8 — a fire on a NON-FINAL burst iteration survives into StepOutcome (the OR-fold) ══
def test_p8_a_fire_on_a_non_final_burst_iteration_survives_into_the_step_outcome() -> None:
    """P8 — `hard_abort_fired = self._run_gate_interval(cfg) or hard_abort_fired`: the fold,
    not a plain assignment.

    The drive fires at training step 6, the SECOND of a four-step burst (steps 5-8). The burst
    runs to its end — `running=False` ends the OUTER loop, not the in-flight burst — so step 8
    crosses another gate boundary and `_fire_hard_abort` returns False there (it records a
    `hard_abort_after_stop` instead). Under a plain assignment that False OVERWRITES the True
    from step 6 and the `StepOutcome` reports a clean burst on the iteration that killed the
    run.

    STATED HONESTLY, because overstating a pin is worse than not having it: `StepOutcome`'s
    `hard_abort_fired` has NO PRODUCTION CONSUMER TODAY. Measured, not assumed: repo-wide in
    `src/` the name appears exactly twice — its `StepOutcome` field declaration and the
    `step()` body that sets it — and `train/loop.py`'s driver calls `coordinator.step()` and
    discards the return. The run's real stop signal is `ShutdownState.running` /
    `abort_rule`, which the fire writes directly, so it is not lost. This pin
    defends the DECLARED contract of the outcome record and any future consumer of it, NOT a
    live failure. It is still worth pinning: the field is public, it is the only per-iteration
    record of the decision, and a silently-lossy fold is the kind of thing a future consumer
    inherits rather than discovers.

    MUTATION RUN, AND ITS RESULT: the RED-TEAM replaced the fold with `hard_abort_fired =
    self._run_gate_interval(cfg)` and measured 1267 collected, 0 failures over the full tier.
    Re-run with this pin in place it reds exactly this test, on the last assertion.
    """
    pool = _Pool(draw_counts=(90, 100))
    cfg = _config(log_interval=_RUN5_LOG_INTERVAL, gate_interval=2,
                  draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                    N_pool_min=10, consec=3))
    h = _coordinator(config=cfg, pool=pool)

    outcomes = []
    for _ in range(4):
        if not h.shutdown.running:
            break
        h.pool.games_completed += 5
        outcomes.append(h.coord.step())

    assert len(outcomes) == 2, (
        f"the run must stop after the burst that fired; got {len(outcomes)} iterations"
    )
    assert outcomes[0].hard_abort_fired is False, (
        "the first burst (steps 1-4) takes only two observations — nothing fires there, which "
        "is what makes the second outcome's True attributable"
    )
    assert outcomes[-1].steps_run == 4, (
        f"the burst ran to its end ({outcomes[-1].steps_run} steps), so the fire at step 6 was "
        "NOT its final iteration — that is the precondition this pin needs"
    )
    assert [e["step"] for e in h.sink.named("hard_abort")] == [6]
    assert [e["step"] for e in h.sink.named("hard_abort_after_stop")] == [8], (
        "step 8 crosses another gate boundary AFTER the stop and returns False — the value a "
        "plain assignment would let overwrite the fire"
    )
    assert outcomes[-1].hard_abort_fired is True, (
        "the burst that fired must report it. A plain assignment reports the LAST iteration's "
        "return, so a fire on any non-final iteration vanishes from the outcome record"
    )


# ══ P9 — the gate boundary runs with NO loss_info: the VALUE, not just the signature ════
def test_p9_the_gate_boundary_gates_and_emits_with_no_loss_info_at_all() -> None:
    """P9 — `_run_gate_interval` must gate and publish when `_last_loss_info` is `None` and
    when it is `{}`.

    P1 pins that `_run_gate_interval` takes no `loss_info` PARAMETER. That is the signature,
    and a signature is not the behaviour: adding `if not self._last_loss_info: return False`
    to the method body re-couples arming to the trainer having produced a loss dict without
    touching the signature at all. This drives the VALUE — a coordinator that has never run a
    training step, so `_last_loss_info` is genuinely `None`.

    STATED HONESTLY: inside the burst this mutation is NEAR-INERT, because D2 populates
    `_last_loss_info` before `_run_gate_interval` is reached on every iteration. What it
    breaks is the DECLARED decoupling — "the gates' producer is the POOL, not the trainer" —
    and any path that reaches the boundary before or without a training step. The pin defends
    the contract, not a live failure, and says so.

    MUTATION RUN, AND ITS RESULT: the RED-TEAM put `if not self._last_loss_info: return
    False` at the top of `_run_gate_interval` and measured 1267 collected, 0 failures over the
    full tier. Re-run with this pin in place it reds exactly this test.
    """
    h = _coordinator(config=_config(log_interval=_RUN5_LOG_INTERVAL, gate_interval=1))
    assert h.coord._last_loss_info is None, (
        "the drive's precondition: a coordinator that has never trained carries no loss dict"
    )

    h.coord._train_step = 1
    assert h.coord._run_gate_interval(h.config) is False
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [1], (
        "the gate boundary must publish with NO loss dict in existence — its producer is the "
        "POOL, and an arming decision that waits on the trainer is the coupling R242 removes"
    )
    assert _checks(h) == 1, "and the gate must have been CHECKED, not merely emitted about"

    # The falsy-but-present case takes the same arm as `None` under `if not ...`, so it is
    # driven too: the mutation is spelled `not`, and `{}` is what a trainer that returned an
    # empty loss dict would leave behind.
    h.coord._last_loss_info = {}
    h.coord._train_step = 2
    assert h.coord._run_gate_interval(h.config) is False
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [1, 2]
    assert _checks(h) == 2


# ══ P10 — a SKIPPED boundary neither advances nor RESETS consec (the true semantic) ═════
class _BlackoutPool(_Pool):
    """A pool whose `pooled_draw_counts` is SCRIPTED per call: one call per gate boundary.

    `(0, 0)` is below any `N_pool_min`, so `pooled_draw_rate` returns `None` = NO OBSERVATION
    (R92) and `_sample` skip-counts it. That is the production shape of an evidence blackout —
    the early-run regime where too few games have completed — not a synthetic hook.
    """

    def __init__(self, script: list[tuple[int, int]]) -> None:
        super().__init__()
        self._script = list(script)
        self.calls = 0

    def pooled_draw_counts(self) -> tuple[int, int]:
        index = self.calls
        self.calls += 1
        return self._script[index] if index < len(self._script) else (0, 0)


def test_p10_a_skipped_boundary_neither_advances_nor_resets_consec() -> None:
    """P10 — `consec` counts consecutive OBSERVATIONS, and observations are only ATTEMPTED
    once per boundary: a boundary that observes nothing neither advances NOR RESETS the
    counter, so the step span a fire covers is a LOWER BOUND and never a product.

    THE FALSE CLAIM THIS PINS AGAINST, verbatim from the bundle that shipped it
    (`step.py::_sample`): "'consecutive OBSERVATIONS' below means consecutive gate-interval
    boundaries, and a rule's `consec` is denominated in them." Measured FALSE by this drive.
    The same claim reached the MINT RECORD through `config/armed_aborts.py`'s
    `draw_rate_collapse` NOTE ("at run5's gate_interval 1000 the three samples still span 2000
    steps", "holds 25 samples by step 25000") and through `DrawRateAbortConfig.consec`'s own
    docstring — both of which hold ONLY if every boundary yields an observation, and the
    early-run regime where `N_pool_min` is unmet is precisely what R242 exists to instrument.
    `monitor/rules.py::check_draw_rate_collapse` had the semantic right all along
    ("`consec` counts consecutive OBSERVATIONS here, not consecutive gate runs"); the three
    texts are corrected to agree with it, and this is the drive that keeps them there.

    THE DRIVE: `gate_interval=1`, `consec=3`, so a boundary lands on every training step. An
    observation at step 1, an EIGHT-boundary blackout over steps 2-9, then observations at
    steps 10 and 11. The abort fires at step 11 on `consec=3` — three observations spanning
    ELEVEN steps, with eight unobserved boundaries between the first and the second. Under the
    false semantic the fire would have needed three consecutive boundaries and could not have
    happened at all.

    MUTATION RUN, AND ITS RESULT: making a skip RESET the counter (`history.clear()` in both
    of `_sample`'s skip arms) — i.e. implementing the false semantic the bundle described —
    leaves the abort UNFIRED on this drive (`assert (0 == 1)`, zero `hard_abort` events), and
    it reds nothing else in the five suites that touch the split.
    """
    script = [(9, 10)] + [(0, 0)] * 8 + [(9, 10), (9, 10)]
    pool = _BlackoutPool(script)
    cfg = _config(log_interval=_RUN5_LOG_INTERVAL, gate_interval=1,
                  draw_rate_abort=DrawRateAbortSpec(threshold=0.4, min_step=0,
                                                    N_pool_min=10, consec=3))
    h = _coordinator(config=cfg, pool=pool)
    _drive(h, outer=4)

    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and aborts[0]["rule"] == "draw_rate_collapse"
    assert aborts[0]["step"] == 11, (
        f"the 3rd OBSERVATION lands at step 11 (observations at 1, 10, 11; boundaries 2-9 "
        f"observed nothing), got {aborts[0]['step']}. If the counter reset on a skip this "
        "would be 12 or would never fire"
    )
    at_fire = next(e for e in h.sink.named("monitor_gates") if e["step"] == 11)
    stats = at_fire["gates"]["draw_rate_collapse"]
    assert (stats["checks"], stats["skips"], stats["fires"]) == (11, 8, 1), (
        f"eleven boundaries were ATTEMPTED, eight of them observed nothing, and the abort "
        f"fired on the third observation; got {stats}"
    )
    assert stats["checks"] - stats["skips"] == 3, (
        "three observations — `consec`. The gap between them is invisible to the rule, which "
        "is the whole finding"
    )
    assert aborts[0]["step"] > cfg.gate_interval * 3, (
        "`consec * gate_interval` (3) is a LOWER BOUND on the span the fire covers (11), not "
        "an equality — the claim the mint record now states, instead of the arithmetic it "
        "used to state"
    )
