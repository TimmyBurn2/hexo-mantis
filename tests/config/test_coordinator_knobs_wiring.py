""">300 justify (R8): one demonstration PER KEY — "set the knob, observe the consumer" — for
eighteen registry citations, so half the length is one row and one behavioural drive per knob.
The rest is the shared fakes harness BOTH halves use; splitting it would fork that harness into
copies free to drift apart, and cross-test imports are barred.

The 18 `train.*` step-coordinator knobs reach the consumers their registry entries NAME, proved
by MUTATION rather than by grep: four keys once reached nothing while both copies of the registry
cited a real function, because the resolver did `data.pop("drain")` and a grep cannot tell a
reader from a `pop`.

TWO HALVES, neither asserted from the other's side. TRANSPORT drives the REAL `compose_run` with
ONE key set to a distinguishable value and reads the field off the running coordinator, with
`_step_coordinator_config` deliberately NOT monkeypatched because it is the subject. BEHAVIOUR
drives a real `StepCoordinator` and observes the event, call, count or fire the registry entry
names — transport alone would pass on a coordinator that carried the value and never read it.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
_DEV = load_config(_CONFIGS / "dev_example.yaml")
_DRAIN_CAPS = resolve_drain_caps(_DEV.monitor)
_KNOBS = resolve_coordinator_knobs(_DEV.train)
#: The builder's FIFTH config-authored parameter — `monitor.gate_interval`, the ARMING cadence.
_GATE_INTERVAL = _DEV.monitor.gate_interval

#: Bounded so `compose_run` terminates; the three step-clock knobs move together because the
#: reachability validator spans them.
_DRIVE_STEPS = 4

#: ONE distinguishable value per schema key: every value differs from the minted one and from
#: every other knob's, so a field reporting a stale, defaulted or NEIGHBOURING number cannot
#: accidentally match. Also the key census — `_SCHEMA_TO_FIELD` must cover exactly these.
_DISTINGUISHABLE: dict[str, Any] = {
    "eval_interval": 37,
    "log_interval": 17,
    "min_buf_size": 29,
    "replay_capacity": 31_337,
    "replay_capacity_schedule": [{"step": 11, "capacity": 222_222}],
    "training_steps_per_game": 2.5,
    "max_train_burst": 7,
    "batch_size": 41,
    "augment": True,
    "recency_weight": 0.43,
    "hard_gn_threshold": 0.59,
    "hard_gn_min_steps": 61,
    "terminal_eval_enabled": False,
    "selfplay_stall_timeout_sec": 67.0,
}

#: schema leaf -> `StepCoordinatorConfig` field. Two names differ, and the rename is the point:
#: a bare `train.capacity` names nothing on its own.
_SCHEMA_TO_FIELD = {
    "replay_capacity": "capacity",
    "replay_capacity_schedule": "buffer_schedule",
}
_KNOB_KEYS = tuple(_DISTINGUISHABLE)


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self.games_completed = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # the third outcome share.
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
    def pooled_draw_counts(self) -> tuple[int, int]: return (0, 0)
    def current_stride5_p90(self) -> int: return 1
    def runner_stats(self) -> Any: return _RunnerStats()
    def sync_inference_weights(self, state_dict) -> None: ...
    def update_checkpoint_step(self, step: int) -> None: ...


class _ComposePool(_Pool):
    """`compose_run`'s drive needs `games_completed` to ADVANCE so the burst runs."""

    def __init__(self) -> None:
        super().__init__()
        self._games = 0

    @property                                        # type: ignore[override]
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    @games_completed.setter
    def games_completed(self, value: int) -> None:
        self._games = int(value)


class _Trainer:
    """The double conforms to the DECLARED seam (typed entry points + `device`); augment
    observation lives on the sampler the dispatcher threads it to."""

    def __init__(self, grad_norm: float = 0.1) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self._gn = grad_norm

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": self._gn,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def inference_state_dict(self) -> dict:
        return {"w": "SENTINEL"}

    def save_checkpoint(self, loss_info) -> None: ...


def _real_graph_ring(n_records: int = 8, capacity: int = 64):
    """A real `HexgBuffer` behind the recording fake — see `_Buffer.sample_graph_batch`."""
    from mantis._engine import HexgBuffer

    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i,
                               True, 1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb


class _Buffer:
    def __init__(self, size: int = 1000, capacity: int = 100_000) -> None:
        self.size = size
        self.capacity = capacity
        self.resizes: list[int] = []
        self.saves: list[str] = []
        self.augment_seen: list[bool] = []
        self.recent_frac_seen: list[float] = []
        self.sampled: list[int] = []
        self._real = _real_graph_ring()

    def resize(self, n: int) -> None:
        self.capacity = n
        self.resizes.append(int(n))

    def save_to_path(self, p) -> None:
        self.saves.append(str(p))

    def sample_graph_batch(self, n: int, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        """The graph route's sampler, RECORDED and then delegated to a real `HexgBuffer`: the
        dispatcher refuses a shapeless fake by design and everything downstream needs real bytes,
        so the fake records WHAT THE DISPATCHER ASKED FOR."""
        self.sampled.append(int(n))
        self.augment_seen.append(bool(augment))
        self.recent_frac_seen.append(float(recent_frac))
        return self._real.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


# HALF ONE — transport, through the REAL composition root.
def _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
                                 monitor_over=None, **train_over):
    """The `StepCoordinatorConfig` a REAL `compose_run` hands the running coordinator.

    The buffer is a REAL preloaded `HexgBuffer`, since a shapeless fake is refused at dispatch.
    `monitor_over` is the SECOND override axis and MERGES into the drive's monitor deltas rather
    than replacing them, because `smoke_run_config` is section-wise and replacing the block would
    drop `actor_lag_threshold_steps` and wedge the drive on the actor-lag abort.
    """
    import mantis.train.anchor as _anchor

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )
    config = smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               # The drive runs the REAL graph route per step, so the minted 256 batch is pure
               # drag; baseline(8) != mutated(41) keeps the transport assertion honest.
               "batch_size": 8,
               # `mixing_min_w` cannot be mutated alone against a minted `mixing_initial_w` of
               # 0.0 — a floor above the start is rejected by design — so both drives share a
               # raised start and the comparison stays one-key-at-a-time.
               **train_over},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1, **(monitor_over or {})},
        # `eval_enabled` and `run_id` are CONFIG facts: `compose_run` has no parameter for
        # either, so the drive's posture is declared where the config is.
        eval_enabled=False, run_id="knob_wiring",
    )
    handles = mantis.run.compose_run(
        config=config, trainer=_Trainer(), pool=_ComposePool(),
        # 32 records: above every distinguishable warmup floor (`min_buf_size: 29`), so no
        # mutated drive can wedge in warmup against a too-small real buffer.
        buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    return handles.coordinator.config


@pytest.mark.parametrize("key", _KNOB_KEYS)
def test_each_knob_reaches_the_coordinator_the_composition_root_builds(
    key, tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """Set ONE `train.*` knob to a distinguishable value; the coordinator the run holds must
    carry it, and NO sibling may move with it. Parametrized per key on purpose: folded into one
    drive the eighteen would share one failure signature, and every one of these was once a
    literal in the builder, so a wire feeding the whole spec from one field would satisfy any
    single-key assertion."""
    field = _SCHEMA_TO_FIELD.get(key, key)
    baseline = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config,
                                            mk_graph_buffer)
    mutated = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config,
                                           mk_graph_buffer, **{key: _DISTINGUISHABLE[key]})

    expected = _DISTINGUISHABLE[key]
    if key == "replay_capacity_schedule":
        expected = tuple({"step": s["step"], "capacity": s["capacity"]} for s in expected)

    assert getattr(baseline, field) != expected, (
        f"the test value for {key} is not distinguishable from the minted one — this oracle "
        "would pass on a knob that reached nothing"
    )
    assert getattr(mutated, field) == expected, (
        f"train.{key} did not reach StepCoordinatorConfig.{field}. That is DR-11's class on a "
        "new axis: the key is minted, schema-validated and registry-claimed while the run "
        "uses a code-side number"
    )
    moved = [other for other in _KNOB_KEYS
             if other != key
             and getattr(mutated, _SCHEMA_TO_FIELD.get(other, other))
             != getattr(baseline, _SCHEMA_TO_FIELD.get(other, other))]
    assert not moved, (
        f"setting {key} moved {moved} too — the eighteen must arrive independently, or one "
        "key's citation is really another key's"
    )


#: The ONE distinguishable ARMING cadence, held apart from `_DISTINGUISHABLE` because it is a
#: `monitor.*` key and that census is `train.*`-only. No config mints it and no other knob
#: carries it; the drive ASSERTS that against the loaded config rather than trusting this line.
_GATE_INTERVAL_MUTATED = 23


def test_the_composition_root_threads_monitor_gate_interval_and_never_log_interval(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
) -> None:
    """`monitor.gate_interval` reaches `StepCoordinatorConfig.gate_interval` through the REAL
    `compose_run`, and it is THAT key which arrives — never `train.log_interval`.

    A sibling test rather than a census row for a structural reason: the census drive passes
    every row through `smoke_run_config(train={...})`, so a `monitor.*` name there is a LOAD
    FAILURE rather than a mutation. The mutated value is SYNTHETIC because every committed config
    mints the two keys EQUAL, so no config-derived drive can tell them apart; the mutation that
    reds this is ONE line reading `log_interval` for the gate cadence, which produced 0 failures
    over the full tier before this test existed. Both directions are driven, because the mutation
    has two signatures and the second is invisible to the census.
    """
    baseline = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config,
                                            mk_graph_buffer)
    assert baseline.gate_interval == baseline.log_interval, (
        "the drive's own config mints the two knobs EQUAL (the shipped posture) — which is "
        "precisely why the mutated value below has to be synthetic: a config-derived drive "
        f"cannot distinguish them. Got {baseline.gate_interval} / {baseline.log_interval}"
    )
    assert baseline.gate_interval != _GATE_INTERVAL_MUTATED, (
        "the test value is not distinguishable from the minted one — this oracle would pass "
        "on a gate_interval that reached nothing"
    )

    # arm 1: the ARMING key moves, the coordinator follows it.
    armed = _composed_coordinator_config(
        tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer,
        monitor_over={"gate_interval": _GATE_INTERVAL_MUTATED},
    )
    assert armed.gate_interval == _GATE_INTERVAL_MUTATED, (
        f"monitor.gate_interval did not reach StepCoordinatorConfig.gate_interval (got "
        f"{armed.gate_interval}). If it is {baseline.log_interval} the composition root is "
        "threading train.log_interval — the exact defect R242 exists to close, re-entered "
        "through the one seam nothing else in the repo watches"
    )
    assert armed.log_interval == baseline.log_interval, (
        "and NOTHING else moved with it: the narration cadence is untouched by an arming-"
        "cadence change, or one key's citation is really the other's"
    )

    # arm 2: the NARRATION key moves, the arming cadence must NOT follow.
    narrated = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config,
                                            mk_graph_buffer,
                                            log_interval=_DISTINGUISHABLE["log_interval"])
    assert narrated.log_interval == _DISTINGUISHABLE["log_interval"]
    assert narrated.gate_interval == baseline.gate_interval, (
        f"train.log_interval dragged the ARMING cadence with it (got "
        f"{narrated.gate_interval}, expected the minted {baseline.gate_interval}). The two "
        "are one knob again, which is the pre-R242 shape"
    )


# HALF TWO — behaviour, at the consumer each registry entry NAMES.
def _coordinator(*, pretrained=None, bot=None, trainer=None, eval_pipeline=None,
                 mixing_cfg=None, **knob_over):
    """A real `StepCoordinator` whose config is DERIVED from the production builder."""
    # The GATE cadence mirrors the NARRATION cadence unless a drive names it — the shipped
    # posture — so a drive that moves only `log_interval` keeps the cadence it had.
    settings = {"eval_interval": 10**9, "log_interval": 1, "min_buf_size": 1, **knob_over}
    settings.setdefault("gate_interval", settings["log_interval"])
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, gate_interval=_GATE_INTERVAL,
                                 knobs=_KNOBS),
        **settings,
    )
    pool, buffer, sink = _Pool(), _Buffer(), _Sink()
    coord = StepCoordinator(
        trainer=trainer or _Trainer(), buffer=buffer, pretrained_buffer=pretrained,
        recent_buffer=None, pool=pool, eval_pipeline=eval_pipeline,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        # The straight arm resolves its route from the DECLARED identity: an identity-less
        # full_config raises MissingEncodingError.
        full_config={
            "identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
            # The graph route resolves microbatch caps and fast-policy weight from the run's
            # own `train` section, and a `train`-less full_config is a NAMED refusal.
            "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
                      "fast_policy_weight": 0.0},
            "search": {"kind": "puct"},
            "selfplay": {"n_workers": 1},
        },
        train_cfg={}, mixing_cfg=mixing_cfg or {}, sink=sink, bot_buffer=bot,
        monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, pool=pool, buffer=buffer, sink=sink,
                           trainer=coord.trainer, config=config)


def _drive(h, *, steps=4, games=5):
    """Drive `step()` up to `steps` times, stopping the moment a gate flips `running` off —
    a run that kept stepping past its own abort would report the wrong fire step."""
    outcomes = []
    for _ in range(steps):
        if not h.coord.shutdown.running:
            break
        h.pool.games_completed += games
        outcomes.append(h.coord.step())
    return outcomes

def test_eval_interval_decides_when_a_promotion_round_is_kicked() -> None:
    """`train.eval_interval` -> `step.py::_maybe_kick_eval`. Both sides of the boundary,
    because "never kicks" and "always kicks" each satisfy one alone."""
    kicks: list[int] = []
    pipeline = SimpleNamespace(
        run_evaluation=lambda model, step, best, **kw: kicks.append(step) or {"kicked": True},
        poll_completed=lambda: None, drain_pending=lambda: None, stop=lambda: None,
    )
    h = _coordinator(eval_pipeline=pipeline, eval_interval=2, log_interval=10**9)
    _drive(h, steps=6, games=1)
    assert kicks == [2, 4, 6], f"a kick must land on every multiple of 2; got {kicks}"

    kicks.clear()
    h = _coordinator(eval_pipeline=pipeline, eval_interval=5, log_interval=10**9)
    _drive(h, steps=6, games=1)
    assert kicks == [5], (
        f"at eval_interval 5 exactly one round fits in six steps; got {kicks}. If the knob "
        "reached nothing both drives would agree"
    )


def test_log_interval_decides_when_the_run_emits_and_when_the_gates_run() -> None:
    """`train.log_interval` -> `step.py::_run_log_interval`: the payload events and the WARN
    rules hang off this boundary, and `<= 0` kills the family — what the schema's `ge=1` makes
    unwritable and this arm shows the value DECIDES. It does NOT decide the gate summary any
    more, that coupling having given run5's hard aborts a blind first kilometre; the count is
    unchanged because `_coordinator` mirrors the two knobs, the shipped posture."""
    every = _coordinator(log_interval=1)
    _drive(every, steps=4, games=1)
    rare = _coordinator(log_interval=3)
    _drive(rare, steps=4, games=1)

    assert len(every.sink.named("training_step")) == 4
    assert len(rare.sink.named("training_step")) == 1, (
        "at log_interval 3 exactly one of four steps is a boundary; got "
        f"{len(rare.sink.named('training_step'))}"
    )
    assert rare.config.gate_interval == 3, (
        "this drive mirrors the two knobs (the shipped posture), which is what makes the "
        "gate-summary count below readable — it is gate_interval's doing, not log_interval's"
    )
    assert len(rare.sink.named("monitor_gates")) == 1, (
        "the LAW-18 gate summary rides monitor.gate_interval (R242), which this drive mints "
        "equal to log_interval exactly as every committed config does"
    )
    apart = _coordinator(log_interval=10**9, gate_interval=1)
    _drive(apart, steps=4, games=1)
    assert apart.sink.named("training_step") == [] and len(
        apart.sink.named("monitor_gates")) == 4, (
        "and stated APART they separate: no narration, four gate summaries. Before R242 this "
        "drive produced neither, which is the defect ADJ-D12 measured"
    )


def test_min_buf_size_decides_the_warmup_floor() -> None:
    """`train.min_buf_size` -> `step.py` O4: below it the learner sees nothing."""
    h = _coordinator(min_buf_size=10**6)
    assert _drive(h, steps=1)[0].in_warmup is True
    assert h.trainer.step == 0, "a warmup step must not train"

    h = _coordinator(min_buf_size=1)
    assert _drive(h, steps=1)[0].in_warmup is False


def test_replay_capacity_is_the_window_the_run_publishes_and_the_preflight_sizes() -> None:
    """`train.replay_capacity` -> `StepCoordinatorConfig.capacity` -> the `buffer_capacity` the
    warmup event publishes, and the size of the REAL engine buffer every boot constructs."""
    h = _coordinator(capacity=31_337, min_buf_size=10**6)
    _drive(h, steps=1)
    stats = h.sink.named("system_stats")
    assert stats and stats[-1]["buffer_capacity"] == 31_337, (
        f"the published window must be the configured one; got {stats}"
    )


def test_the_capacity_schedule_ramps_the_buffer_at_its_own_step() -> None:
    """`train.replay_capacity_schedule` -> `step.py` D1. One registry leaf, so both inner names
    are demonstrated here: the STEP decides when, the CAPACITY decides what."""
    h = _coordinator(buffer_schedule=({"step": 2, "capacity": 555_555},))
    _drive(h, steps=1, games=1)
    assert h.buffer.resizes == [], "the ramp must not fire before its own step"
    _drive(h, steps=3, games=1)
    assert h.buffer.resizes == [555_555], (
        f"the ramp must fire once, at step 2, to the configured capacity; got {h.buffer.resizes}"
    )


def test_training_steps_per_game_and_max_train_burst_set_the_step_budget() -> None:
    """Both -> `step.py` O6 `_steps_budget`. Driven together because the budget is
    `min(max(1, games*per_game), burst)` and each knob is the binding term in exactly one of the
    two drives — a wire feeding one from the other shows up as the wrong drive being clamped."""
    ratio_bound = _coordinator(training_steps_per_game=2.0, max_train_burst=100)
    assert _drive(ratio_bound, steps=1, games=3)[0].steps_run == 6

    burst_bound = _coordinator(training_steps_per_game=2.0, max_train_burst=4)
    assert _drive(burst_bound, steps=1, games=3)[0].steps_run == 4, (
        "the burst ceiling must bind when the ratio would exceed it"
    )


def test_batch_size_is_the_batch_the_sampler_is_asked_for() -> None:
    """`train.batch_size` -> `step.py::_run_training_step` -> the route's sampler. The lookup was
    `train_cfg.get("batch_size", full_config.get("batch_size", 256))`, both lookups missed on the
    production path, and the run's real batch size was the literal 256 while
    `StepCoordinatorConfig.batch_size` sat beside it unread. The drive uses 41 so a surviving
    literal is visible."""
    h = _coordinator(batch_size=41)
    _drive(h, steps=3, games=1)
    assert set(h.buffer.sampled) == {41}, (
        f"the sampler must be asked for the CONFIGURED batch; got {h.buffer.sampled}"
    )
    assert 256 not in set(h.buffer.sampled), (
        "the `256` literal must be gone, not merely shadowed"
    )


def test_augment_reaches_the_training_path() -> None:
    """`train.augment` -> the dispatcher's sampler (`sample_graph_batch(augment=)`): augment is a
    SAMPLING knob and travels to the buffer draw, not to the trainer. There is ONE reader
    left."""
    plain = _coordinator(augment=True)
    _drive(plain, steps=2, games=1)
    assert plain.buffer.augment_seen and all(plain.buffer.augment_seen)

    off = _coordinator(augment=False)
    _drive(off, steps=2, games=1)
    assert off.buffer.augment_seen and not any(off.buffer.augment_seen), (
        "the discriminating negative: a knob that reaches the sampler as True whatever the "
        "config says is not wired, it is hardcoded"
    )


def test_recency_weight_reaches_the_samplers_recency_window() -> None:
    """`train.recency_weight` -> `sample_graph_batch(recent_frac=)`, the one live reader."""
    h = _coordinator(recency_weight=0.43)
    _drive(h, steps=2, games=1)
    assert set(h.buffer.recent_frac_seen) == {0.43}, (
        f"the sampler must be handed the CONFIGURED recency fraction; got "
        f"{h.buffer.recent_frac_seen}"
    )


def test_the_grad_norm_knobs_decide_whether_the_hard_abort_fires() -> None:
    """`train.hard_gn_threshold` / `train.hard_gn_min_steps` -> `step.py` D3, in three drives:
    below the threshold nothing fires, above it the run stops after exactly `min_steps`
    consecutive breaches, and a `min_steps` beyond the drive keeps it silent — so each knob is
    the binding term in one drive and not the other. Shipped at `1e9` this gate can never fire,
    which is why it is a DEFERRED armed-abort row."""
    quiet = _coordinator(trainer=_Trainer(grad_norm=0.5), hard_gn_threshold=1.0,
                         hard_gn_min_steps=1)
    _drive(quiet, steps=4, games=1)
    assert quiet.coord.shutdown.running is True, "a grad norm below the threshold must not fire"

    loud = _coordinator(trainer=_Trainer(grad_norm=5.0), hard_gn_threshold=1.0,
                        hard_gn_min_steps=3)
    _drive(loud, steps=4, games=1)
    assert loud.coord.shutdown.running is False
    assert loud.trainer.step == 3, (
        f"the abort must fire on the third consecutive breach, not the first; got "
        f"{loud.trainer.step}"
    )

    patient = _coordinator(trainer=_Trainer(grad_norm=5.0), hard_gn_threshold=1.0,
                           hard_gn_min_steps=10**6)
    _drive(patient, steps=4, games=1)
    assert patient.coord.shutdown.running is True, (
        "the SAME breaching grad norm with a higher consecutive count must stay silent — "
        "that is hard_gn_min_steps deciding on its own"
    )


def test_terminal_eval_enabled_decides_whether_close_out_runs_a_terminal_round() -> None:
    """`train.terminal_eval_enabled` -> `coordinator/drain.py::run_terminal_eval`. This fact
    once had THREE authorities (no key, a dataclass `= True`, and a fallback); the key retires
    the last of them, which is why five composition tests stopped monkeypatching the builder."""
    from mantis.train.coordinator import drain

    calls: list[str] = []
    pipeline = SimpleNamespace(
        # A terminal round returns a ROUND RESULT whose `eval_broken_reason` the seam reads.
        run_evaluation=lambda *a, **k: calls.append("terminal") or {"eval_broken_reason": None},
        poll_completed=lambda: None, drain_pending=lambda: None, stop=lambda: None,
    )
    on = _coordinator(eval_pipeline=pipeline, terminal_eval_enabled=True)
    drain.run_terminal_eval(on.coord)
    assert calls == ["terminal"]

    calls.clear()
    off = _coordinator(eval_pipeline=pipeline, terminal_eval_enabled=False)
    drain.run_terminal_eval(off.coord)
    assert calls == [], (
        "`terminal_eval_enabled: false` must remove the run's last promotion opportunity — "
        "and it must do so from the CONFIG, not from a code-side posture"
    )


def test_selfplay_stall_timeout_is_the_budget_the_watchdog_arms_with() -> None:
    """`train.selfplay_stall_timeout_sec` -> `StallWatchdog(timeout_sec=)`, whose arm event
    publishes the value. The watchdog's own contract lets `<= 0` disable the fire while still
    emitting the arm log; the schema's `gt=0` makes that posture unwritable."""
    h = _coordinator(selfplay_stall_timeout_sec=67.0)
    _drive(h, steps=1, games=1)
    armed = h.sink.named("selfplay_stall_watchdog_armed")
    assert armed and armed[-1]["timeout_sec"] == 67.0, (
        f"the watchdog must arm on the configured budget; got {armed}"
    )
    assert armed[-1]["enabled"] is True


# No second authority survives anywhere on the path.
def test_the_builder_takes_knobs_as_a_required_keyword_only_parameter() -> None:
    """A parameter DEFAULT would move the authority from the builder BODY to its SIGNATURE,
    leaving every `dataclasses.fields()` assertion green while a caller that omitted the
    argument silently inherited eighteen postures."""
    param = inspect.signature(_step_coordinator_config).parameters.get("knobs")
    assert param is not None, (
        "`_step_coordinator_config` must take `knobs`: the eighteen are `train.*` keys and "
        "arrive from `resolve_coordinator_knobs`, never from a literal here"
    )
    assert param.default is inspect.Parameter.empty, (
        f"knobs carries a parameter default ({param.default!r}) — the literals did not die, "
        "they MIGRATED from the builder body to the builder signature"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_no_coordinator_field_carries_a_code_side_default_and_the_dead_six_are_gone() -> None:
    """EVERY `StepCoordinatorConfig` field must be MISSING-defaulted, and the six dead ones
    deleted rather than authored: a surviving default is a second authority a caller inherits,
    and a surviving DEAD field would have to become a config key with no live consumer."""
    fields = {f.name: f for f in dataclasses.fields(StepCoordinatorConfig)}
    for name, field in fields.items():
        assert (field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING), (
            f"StepCoordinatorConfig.{name} carries a code-side default ({field.default!r} / "
            f"{field.default_factory!r}) — the schema is then not its only authority (R1)"
        )
    dead = {"composition_interval", "value_probe_interval", "soft_ew_threshold",
            "soft_ew_min_pts", "instrumentation_enabled", "bot_corpus_path"}
    assert dead.isdisjoint(fields), (
        f"the six consumer-less fields must be DELETED, not authored; still present: "
        f"{sorted(dead & set(fields))}"
    )


def test_the_resolver_is_the_only_read_of_the_eighteen_keys(smoke_run_config) -> None:
    """`resolve_coordinator_knobs` returns exactly what the loaded config holds, key for key. A
    resolver that dropped, defaulted or CROSSED two fields would still satisfy the per-key
    mutations above, and the schema->field rename is where a crossing would be easiest."""
    config = smoke_run_config(train=dict(_DISTINGUISHABLE))
    spec = resolve_coordinator_knobs(config.train)
    assert isinstance(spec, CoordinatorKnobsSpec)
    for key, value in _DISTINGUISHABLE.items():
        field = _SCHEMA_TO_FIELD.get(key, key)
        expected = value
        if key == "replay_capacity_schedule":
            expected = tuple({"step": s["step"], "capacity": s["capacity"]} for s in value)
        assert getattr(spec, field) == expected, (
            f"train.{key} must resolve to CoordinatorKnobsSpec.{field}; got "
            f"{getattr(spec, field)!r}"
        )
    assert {f.name for f in dataclasses.fields(CoordinatorKnobsSpec)} == {
        _SCHEMA_TO_FIELD.get(key, key) for key in _DISTINGUISHABLE
    }, "the spec must carry exactly the eighteen authored knobs and nothing else"
    for field in dataclasses.fields(CoordinatorKnobsSpec):
        assert (field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING), (
            f"CoordinatorKnobsSpec.{field.name} carries a code-side default — the resolver "
            "would then build a spec the config did not fully author (R83's Attack A shape)"
        )
    assert not hasattr(CoordinatorKnobsSpec, "__post_init__"), (
        "`CoordinatorKnobsSpec` is frozen, and `object.__setattr__` inside a `__post_init__` "
        "is legal on a frozen dataclass — a default restored AFTER construction still reports "
        "MISSING to `dataclasses.fields()` (R83's Attack A)"
    )


def test_the_builder_holds_no_literal_for_any_authored_knob() -> None:
    """With `stop_step`, `draw_rate_abort`, `drain_caps` and `knobs` all arriving as parameters,
    EVERY field the builder sets comes from one of them. Driven by construction rather than by
    reading source: a surviving literal shows up as that field disagreeing with the spec."""
    distinguishable = CoordinatorKnobsSpec(**{
        _SCHEMA_TO_FIELD.get(key, key): (
            tuple({"step": s["step"], "capacity": s["capacity"]} for s in value)
            if key == "replay_capacity_schedule" else value)
        for key, value in _DISTINGUISHABLE.items()
    })
    built = _step_coordinator_config(stop_step=11, draw_rate_abort=None,
                                     drain_caps=_DRAIN_CAPS,
                                     gate_interval=_GATE_INTERVAL, knobs=distinguishable)
    for field in dataclasses.fields(CoordinatorKnobsSpec):
        assert getattr(built, field.name) == getattr(distinguishable, field.name), (
            f"`_step_coordinator_config` overrode {field.name} with something other than the "
            "resolved spec — a literal survives in the builder body"
        )
    assert built.stop_step == 11 and built.draw_rate_abort is None
