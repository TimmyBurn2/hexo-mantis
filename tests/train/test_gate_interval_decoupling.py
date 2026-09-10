# >300 justify (R8). ONE claim — "arming rides monitor.gate_interval and narration rides
# train.log_interval, and neither decides the other" — asserted from many angles over ONE set
# of fakes and ONE builder-derived config factory. The pins are evidence only BECAUSE they
# share that factory: a second file would need its own harness copy, which would then be a
# second authority on what a drive at "gate_interval=4, log_interval=5" even is.
"""ORACLE — the ARMING cadence is `monitor.gate_interval`, not `train.log_interval`.

THE DEFECT, as measured: both the live hard-abort gates and the `monitor_gates` summary sat
behind `_run_log_interval`'s guard, so at a minted `train.log_interval: 1000` no draw-rate
observation could be taken and no `monitor_gates` event could exist before training step 1000
— armed machinery with a blind first kilometre, and the instrument that would have shown it
switched off by the same knob. Every committed config mints the two knobs EQUAL, so no armed
value moved; what landed is the CAPABILITY to state them apart, and each pin below names the
mutation that reds it.
"""
from __future__ import annotations

from mantis._engine import HexgBuffer

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from mantis.config.loader import discover_configs, load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real graph ring the coordinator stubs sample through (R5 bars cross-test imports,
    so each file that needs one builds it)."""
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb



#: The declaration a `StepCoordinator` reads on the graph route: the identity it dispatches on
#: plus the two sections the route's resolvers read. The caps are the template's NON-BINDING
#: pair — nothing here exercises a split.
_GRAPH_FULL_CONFIG: dict = {
    "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
    "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
              "fast_policy_weight": 0.0},
    "selfplay": {"n_workers": 1},
}


_REPO = Path(__file__).resolve().parents[2]
_DEV_CONFIG_PATH = _REPO / "configs" / "dev_example.yaml"
_DEV_CONFIG = load_config(_DEV_CONFIG_PATH)
_DRAIN_CAPS = resolve_drain_caps(_DEV_CONFIG.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV_CONFIG.train)
_GATE_INTERVAL = _DEV_CONFIG.monitor.gate_interval

#: run5's own minted narration cadence, named rather than invented: the defect is about what
#: happens BEFORE this many training steps.
_RUN5_LOG_INTERVAL = 1000


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self, *, draw_counts: tuple[int, int] = (0, 0)) -> None:
        self.games_completed = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # F-816-2: the third outcome share.
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
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        # DELEGATED to a real `HexgBuffer` rather than faked: the dispatcher collates the wire
        # for real, so a hand-built payload would be a second wire format to disagree with.
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _config(**overrides) -> StepCoordinatorConfig:
    """DERIVED from the production builder. Unlike the sibling harnesses this one does NOT
    mirror `gate_interval` onto `log_interval`: every drive states BOTH knobs, because stating
    them apart is the whole subject."""
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
        full_config=_GRAPH_FULL_CONFIG,
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


def test_p1_gates_are_visible_and_advancing_far_below_the_log_interval_boundary() -> None:
    """P1 — at `log_interval=1000` with `gate_interval=1` the gates run and publish on every
    training step while ZERO `training_step` events have been emitted; at HEAD twenty steps
    produced zero `monitor_gates` events. Killer: move the gates and the summary back inside
    `_run_log_interval`."""
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


def test_p2_the_draw_rate_abort_fires_on_exactly_the_nth_gate_interval_observation() -> None:
    """P2 — armed `consec=3`, `gate_interval=2`, `log_interval=1000`: observations land at steps
    2/4/6 and the abort fires at step 6, on the THIRD and not before. The observation COUNT is
    asserted, not merely the fire, because a gate firing at the wrong cadence satisfies a
    fire-only assertion. The terms are chosen here, so this pin is no second authority over the
    operator's pre-registered values."""
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
    # in-flight burst), so the boundary list runs 2/4/6/8 and step 8 records an after-stop.
    boundaries = [e["step"] for e in h.sink.named("monitor_gates")]
    assert boundaries == [2, 4, 6, 8], f"one summary per gate_interval boundary; got {boundaries}"
    at_fire = next(e for e in h.sink.named("monitor_gates") if e["step"] == 6)
    assert at_fire["gates"]["draw_rate_collapse"]["checks"] == 3, (
        "exactly 3 observations had been taken when it fired — one per gate_interval "
        f"boundary — got {at_fire['gates']['draw_rate_collapse']['checks']}. `consec` counts "
        "OBSERVATIONS, so this number IS the abort's clock"
    )
    assert at_fire["gates"]["draw_rate_collapse"]["fires"] == 1


def test_p3_the_gate_check_count_follows_gate_interval_and_not_log_interval() -> None:
    """P3 — with `gate_interval=4` and `log_interval=5` over 20 steps the gate is checked
    `20 // 4 = 5` times, not `20 // 5 = 4`; the knobs deliberately DISAGREE, because equal
    values cannot distinguish the fixed code from the defect. Killer: read `cfg.log_interval`
    in `_run_gate_interval`'s guard."""
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


def test_p4_narration_follows_log_interval_and_not_gate_interval() -> None:
    """P4 — with `log_interval=5` and `gate_interval=1` over 20 steps there are 4
    `training_step` events, not 20: the split must not over-reach into narration. Killer: drop
    the `log_interval` guard, or point it at `gate_interval`."""
    h = _coordinator(config=_config(log_interval=5, gate_interval=1))
    _drive(h, outer=5)

    steps = [e["step"] for e in h.sink.named("training_step")]
    assert steps == [5, 10, 15, 20], f"narration must ride log_interval alone; got {steps}"
    assert len(h.sink.named("monitor_gates")) == 20, (
        "while the gates ran on all 20 — the drive is only meaningful because the two "
        "cadences disagree here"
    )


def test_p5_iteration_complete_still_emits_per_coordinator_step() -> None:
    """P5 — `iteration_complete` emits once per coordinator step with BOTH boundaries
    un-crossed. Killer: re-couple `_emit_iteration_complete` to either boundary."""
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


def test_p6_a_config_missing_monitor_gate_interval_fails_to_load(tmp_path) -> None:
    """P6 — the ARMING cadence has NO default anywhere, so a config that omits it is a LOAD
    FAILURE rather than a run inheriting a stride. The subject is a MISSING REQUIRED key, driven
    against a real minted config through the real loader. Killer: give the schema field any
    default, or fall back to `log_interval`."""
    payload = yaml.safe_load(_DEV_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["monitor"].pop("gate_interval") == _GATE_INTERVAL, (
        "the minted config must carry the key for its removal to mean anything"
    )
    target = tmp_path / "no_gate_interval.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="gate_interval"):
        load_config(target)


def test_p6b_every_committed_config_mints_gate_interval_equal_to_its_log_interval() -> None:
    """P6 companion — every committed config mints `monitor.gate_interval == train.log_interval`,
    which is how "this moved no armed value" is true rather than said.

    A PREREG BLANK wired to current behaviour: when the operator picks a real gate stride this
    is expected to be RE-POINTED, not deleted. The count ratchets in BOTH directions so no
    config slips past the sweep, and enumeration goes through the ONE discovery authority.
    """
    configs = discover_configs(_REPO / "configs")
    assert len(configs) == 3, f"expected the three committed configs, found {configs}"
    for path in configs:
        cfg = load_config(path)
        assert cfg.monitor.gate_interval == cfg.train.log_interval, (
            f"{path.name}: gate_interval {cfg.monitor.gate_interval} != log_interval "
            f"{cfg.train.log_interval}. R242 moved no armed value; a divergence here is an "
            "operator prereg decision and must arrive as one, not as an implementation edit"
        )


def test_p7_the_builder_takes_gate_interval_as_a_required_keyword_only_parameter() -> None:
    """P7 — `_step_coordinator_config`'s `gate_interval` parameter must have NO default.

    P6 pins the SCHEMA half; this is the BUILDER half, a different authority. A default on this
    signature leaves every schema assertion and P6 itself GREEN while a caller that forgets the
    argument silently inherits an ARMING cadence — the literal moves from the body to the
    signature. Killer, measured: a default here reds only this test in the full tier.
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


def test_p8_a_fire_on_a_non_final_burst_iteration_survives_into_the_step_outcome() -> None:
    """P8 — `hard_abort_fired = self._run_gate_interval(cfg) or hard_abort_fired`: the fold,
    not a plain assignment.

    The drive fires at step 6, the SECOND of a four-step burst; the burst runs to its end, so
    step 8 crosses another boundary returning False, and a plain assignment would report a
    clean burst on the iteration that killed the run.

    STATED HONESTLY: this field has NO production consumer today — measured, the name appears
    twice in `src/` and the driver discards the return — so this defends the DECLARED contract,
    not a live failure. Killer, measured: a plain assignment reds only this test's last
    assertion.
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


def test_p9_the_gate_boundary_gates_and_emits_with_no_loss_info_at_all() -> None:
    """P9 — `_run_gate_interval` must gate and publish when `_last_loss_info` is `None` and when
    it is `{}`.

    P1 pins that the method takes no `loss_info` PARAMETER, but a signature is not the
    behaviour: an early return on a falsy `_last_loss_info` re-couples arming to the trainer
    without touching it. STATED HONESTLY, that mutation is NEAR-INERT inside the burst, where
    D2 populates the dict first; what it breaks is the DECLARED decoupling — the gates' producer
    is the POOL. Killer, measured: that early return reds only this test.
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

    # The falsy-but-present case takes the same arm as `None` under `if not ...`, so `{}` is
    # driven too — that is what a trainer returning an empty loss dict leaves behind.
    h.coord._last_loss_info = {}
    h.coord._train_step = 2
    assert h.coord._run_gate_interval(h.config) is False
    assert [e["step"] for e in h.sink.named("monitor_gates")] == [1, 2]
    assert _checks(h) == 2


class _BlackoutPool(_Pool):
    """A pool whose `pooled_draw_counts` is SCRIPTED per call, one call per gate boundary.
    `(0, 0)` is below any `N_pool_min`, so the rate is `None` = NO OBSERVATION and `_sample`
    skip-counts it — the production shape of an early-run evidence blackout."""

    def __init__(self, script: list[tuple[int, int]]) -> None:
        super().__init__()
        self._script = list(script)
        self.calls = 0

    def pooled_draw_counts(self) -> tuple[int, int]:
        index = self.calls
        self.calls += 1
        return self._script[index] if index < len(self._script) else (0, 0)


def test_p10_a_skipped_boundary_neither_advances_nor_resets_consec() -> None:
    """P10 — `consec` counts consecutive OBSERVATIONS, only ATTEMPTED once per boundary: a
    boundary that observes nothing neither advances NOR RESETS the counter, so the step span a
    fire covers is a LOWER BOUND and never a product.

    THE FALSE CLAIM THIS PINS AGAINST, shipped in three places at once: that "consecutive
    OBSERVATIONS" means consecutive boundaries. That holds only if every boundary yields an
    observation, and the early-run regime where `N_pool_min` is unmet is exactly what this
    cadence split exists to instrument.

    THE DRIVE: an observation at step 1, an EIGHT-boundary blackout over steps 2-9, then
    observations at 10 and 11; the abort fires at step 11 on three observations spanning eleven
    steps. Killer, measured: making a skip RESET the counter leaves the abort UNFIRED here and
    reds nothing else in the five suites that touch the split.
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
