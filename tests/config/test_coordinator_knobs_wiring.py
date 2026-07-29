""">300 justify (R8), stated at this file's MEASURED size of 729 lines: it is R93's evidence
for NINETEEN registry citations, and R93's condition is one demonstration PER KEY — "set the
knob, observe the consumer", never a shared arm a sibling could ride. Roughly half the length
is one `_DISTINGUISHABLE` row and one behavioural drive per knob; the rest is the ~120-line
fakes harness that BOTH halves use (a real `compose_run` drive and a real `StepCoordinator`
drive). Splitting it would fork that harness into two copies free to drift apart, which is the
exact failure `tests/config/test_drain_caps_wiring.py`'s own justify names, and R5 bars the
cross-test import that would prevent it.

The 19 `train.*` step-coordinator knobs reach the consumers their registry entries NAME —
proved by MUTATION.

WPMINT Phase K-B, `CARD-COORD-KNOBS` (R78 as clarified by R80), method bound by R93.

WHY MUTATION AND NOT GREP. R93 exists because DR-11 found four keys — minted,
schema-validated, and claimed by BOTH copies of `CONSUMER_REGISTRY` with a citation naming a
REAL function — that reached nothing, because `resolve_monitor_config` did `data.pop("drain")`
and a grep cannot tell a reader from a `pop`. Every citation Phase K touches is therefore owed
a demonstration that the VALUE MOVES THE CONSUMER. A test that asserted
`resolve_coordinator_knobs` returns what the config says would be green under exactly that
defect: the resolver is not the consumer.

TWO HALVES, and neither is asserted from the other's side.

* **Transport** (`test_each_knob_reaches_the_coordinator_the_composition_root_builds`) drives
  the REAL `compose_run` on a REAL minted config with ONE key set to a distinguishable value
  and reads the field off `handles.coordinator.config` — the object the running coordinator
  holds. Loader, `resolve_coordinator_knobs`, `_step_coordinator_config` and
  `StepCoordinatorConfig` all run unpatched; `_step_coordinator_config` in particular is NOT
  monkeypatched here, unlike the five composition drives that suppress terminal eval through
  it, because it is the subject.
* **Behaviour** — one test per knob below, driving a real `StepCoordinator.step()` (or
  `close_out`) and observing the thing the registry entry names: an event, a call, a count, a
  fire. Transport alone would pass on a coordinator that carried the value and never read it,
  which is the dead-field half of the same finding (`batch_size` was a live-looking
  `StepCoordinatorConfig` field with no reader at all until this phase).

`train.replay_capacity_schedule` is ONE registry leaf under NIT-3 (a `list[SubModel]`, like
`eval.ladder.rungs`), so it gets one row and one behavioural drive covering both inner names.

R7 / gate 6: nothing here writes a `*.jsonl`; every drive writes under `tmp_path`.
R5: zero `sys.path` mutation; configs are loaded through the ONE loader by absolute path.
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

#: The drive is bounded so `compose_run` terminates; the three step-clock knobs move together
#: because the reachability validator spans them (DESIGN_S §6.6 MF-3).
_DRIVE_STEPS = 4

#: ONE distinguishable value per schema key. Every value differs from the minted one, and
#: none is a value another knob also carries, so a coordinator field that reported a stale,
#: defaulted or NEIGHBOURING number cannot accidentally match. The map is also the key
#: census: `_SCHEMA_TO_FIELD` below must cover exactly these.
_DISTINGUISHABLE: dict[str, Any] = {
    "eval_interval": 37,
    "log_interval": 17,
    "buffer_save_interval": 23,
    "min_buf_size": 29,
    "replay_capacity": 31_337,
    "replay_capacity_schedule": [{"step": 11, "capacity": 222_222}],
    "training_steps_per_game": 2.5,
    "max_train_burst": 7,
    "batch_size": 41,
    "augment": True,
    "recency_weight": 0.43,
    "mixing_initial_w": 0.47,
    "mixing_min_w": 0.13,
    "mixing_decay_steps": 53.0,
    "hard_gn_threshold": 0.59,
    "hard_gn_min_steps": 61,
    "terminal_eval_enabled": False,
    "bot_batch_share": 0.19,
    "selfplay_stall_timeout_sec": 67.0,
}

#: schema leaf -> `StepCoordinatorConfig` field. Three names differ, and the rename is the
#: point: `train.checkpoint_interval` (the TRAINER's periodic save) is a DIFFERENT authored
#: key, so the coordinator's buffer-save cadence could not keep that spelling, and a bare
#: `train.capacity` names nothing on its own.
_SCHEMA_TO_FIELD = {
    "buffer_save_interval": "checkpoint_interval",
    "replay_capacity": "capacity",
    "replay_capacity_schedule": "buffer_schedule",
}
_KNOB_KEYS = tuple(_DISTINGUISHABLE)


# ── fakes (the `tests/train/test_coordinator_gates.py` harness shape) ──────────────────
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self.games_completed = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
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
    def __init__(self, grad_norm: float = 0.1) -> None:
        self.step = 0
        self.model = object()
        self._gn = grad_norm
        self.augment_seen: list[bool] = []

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": self._gn,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        self.augment_seen.append(bool(augment))
        return self._loss()

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def inference_state_dict(self) -> dict:
        return {"w": "SENTINEL"}

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    def __init__(self, size: int = 1000, capacity: int = 100_000) -> None:
        self.size = size
        self.capacity = capacity
        self.resizes: list[int] = []
        self.saves: list[str] = []

    def resize(self, n: int) -> None:
        self.capacity = n
        self.resizes.append(int(n))

    def save_to_path(self, p) -> None:
        self.saves.append(str(p))


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


# ══ HALF ONE — transport, through the REAL composition root ════════════════════════════
def _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config, **train_over):
    """The `StepCoordinatorConfig` a REAL `compose_run` hands the running coordinator."""
    import mantis.train.anchor as _anchor

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )
    config = smoke_run_config(
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS,
               # `mixing_min_w` cannot be mutated alone against a minted `mixing_initial_w`
               # of 0.0 — `_mixing_floor_is_below_its_start` rejects a floor above the start,
               # by design. Both drives (baseline and mutated) therefore share a raised start,
               # so the comparison stays one-key-at-a-time.
               "mixing_initial_w": 1.0, **train_over},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1},
    )
    handles = mantis.run.compose_run(
        config=config, trainer=_Trainer(), pool=_ComposePool(), buffer=_Buffer(),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
        eval_enabled=False, run_id="knob_wiring",
    )
    return handles.coordinator.config


@pytest.mark.parametrize("key", _KNOB_KEYS)
def test_each_knob_reaches_the_coordinator_the_composition_root_builds(
    key, tmp_path, monkeypatch, smoke_run_config,
) -> None:
    """Set ONE `train.*` knob to a distinguishable value; the coordinator the run holds must
    carry it, and NO sibling may move with it.

    Parametrized per key on purpose: with the nineteen folded into one drive they would share
    one failure signature, and a knob that reached nothing would be masked by the eighteen
    that did. The independence arm is the other half — before this phase every one of these
    was a literal in `_step_coordinator_config`, so a wire that fed the whole spec from one
    field would satisfy any single-key assertion.
    """
    field = _SCHEMA_TO_FIELD.get(key, key)
    baseline = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config)
    mutated = _composed_coordinator_config(tmp_path, monkeypatch, smoke_run_config,
                                           **{key: _DISTINGUISHABLE[key]})

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
        f"setting {key} moved {moved} too — the nineteen must arrive independently, or one "
        "key's citation is really another key's"
    )


# ══ HALF TWO — behaviour, at the consumer each registry entry NAMES ════════════════════
def _coordinator(*, pretrained=None, bot=None, trainer=None, eval_pipeline=None,
                 mixing_cfg=None, **knob_over):
    """A real `StepCoordinator` whose config is DERIVED from the production builder."""
    config = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS, knobs=_KNOBS),
        **{"eval_interval": 10**9, "log_interval": 1, "min_buf_size": 1, **knob_over},
    )
    pool, buffer, sink = _Pool(), _Buffer(), _Sink()
    coord = StepCoordinator(
        trainer=trainer or _Trainer(), buffer=buffer, pretrained_buffer=pretrained,
        recent_buffer=None, pool=pool, eval_pipeline=eval_pipeline,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=config,
        full_config={}, train_cfg={}, mixing_cfg=mixing_cfg or {}, sink=sink, bot_buffer=bot,
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


def _mixed_batch_calls(monkeypatch, h, *, steps=2):
    """Drive the MIXED training path and capture every `assemble_mixed_batch` call.

    That path — not `trainer.train_step` — is where `batch_size`, `recency_weight`, the three
    `mixing_*` knobs and `bot_batch_share` are read, so it is where their citations have to be
    demonstrated. It needs a pretrained buffer, which `compose_run` does not build today
    (`pretrained_buffer=None`), so the drive is a direct `StepCoordinator` — the production
    consumer, reached the only way a caller can reach it.
    """
    import mantis.train.batch_assembly as batch_assembly

    seen: list[dict] = []

    def _spy(pretrained_buffer, buffer, recent_buffer, n_pre, n_self, batch_size,
             batch_size_cfg, recency_weight, bufs, train_step, *, augment=False,
             bot_buffer=None, n_bot=0):
        seen.append({"n_pre": n_pre, "n_self": n_self, "batch_size": batch_size,
                     "recency_weight": recency_weight, "augment": augment, "n_bot": n_bot,
                     "train_step": train_step})
        return SimpleNamespace(
            states=None, policies=None, outcomes=None, chain_planes=None, ownership=None,
            winning_line=None, is_full_search=None, n_recent_actual=0,
            position_indices=None, value_target_valid=None,
        )

    monkeypatch.setattr(batch_assembly, "assemble_mixed_batch", _spy)
    _drive(h, steps=steps)
    # `train_step` 0 is dropped: `exp(-0/decay) == 1` for EVERY horizon, so the first call
    # cannot distinguish a decay knob from any other. The schedule is only observable once
    # the run has taken a step, which is what the knob describes.
    seen = [call for call in seen if call["train_step"] > 0]
    assert seen, "the mixed-batch path was never reached — the drive witnesses nothing"
    return seen


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
    """`train.log_interval` -> `step.py::_run_log_interval`: the payload events, the WARN
    rules, BOTH live hard-abort gates and the LAW-18 `monitor_gates` summary all hang off this
    one boundary. WPMINT DR-7 measured that `<= 0` kills the whole family at once; the schema's
    `ge=1` makes that unwritable, and this is the arm that shows the value still DECIDES."""
    every = _coordinator(log_interval=1)
    _drive(every, steps=4, games=1)
    rare = _coordinator(log_interval=3)
    _drive(rare, steps=4, games=1)

    assert len(every.sink.named("training_step")) == 4
    assert len(rare.sink.named("training_step")) == 1, (
        "at log_interval 3 exactly one of four steps is a boundary; got "
        f"{len(rare.sink.named('training_step'))}"
    )
    assert len(rare.sink.named("monitor_gates")) == 1, (
        "the LAW-18 gate summary rides the same boundary — DR-7's finding is that these two "
        "cannot be separated, so they are asserted together"
    )


def test_buffer_save_interval_decides_the_replay_buffer_save_cadence(tmp_path, monkeypatch) -> None:
    """`train.buffer_save_interval` -> `step.py` D4 `_try_save_buffer`. NOT the trainer
    checkpoint cadence (`train.checkpoint_interval`, a different authored key) — the rename
    exists because two config keys with one spelling is R1's duplicated-authority class."""
    monkeypatch.chdir(tmp_path)
    #: `try_save_buffer` is a no-op unless the legacy mixing dict enables persistence, so the
    #: drive enables it — otherwise "no save happened" would be true for a reason that has
    #: nothing to do with the knob, and the `0` arm below would pass vacuously.
    persist = {"buffer_persist": True, "buffer_persist_path": str(tmp_path / "replay.bin")}

    off = _coordinator(checkpoint_interval=0, mixing_cfg=persist)
    outcomes = _drive(off, steps=4, games=1)
    assert off.buffer.saves == [] and not any(o.checkpoint_saved for o in outcomes), (
        "0 is the shipped posture: no cadence save at all (close-out and the shutdown signal "
        "still save, which is why a zero here disables nothing LAW-16 requires)"
    )

    on = _coordinator(checkpoint_interval=2, mixing_cfg=persist)
    outcomes = _drive(on, steps=4, games=1)
    assert len(on.buffer.saves) == 2, (
        f"at cadence 2 the buffer must be saved on steps 2 and 4; got {on.buffer.saves}"
    )
    assert [o.checkpoint_saved for o in outcomes] == [False, True, False, True], (
        f"…and the outcome record must say which steps saved; got {outcomes}"
    )


def test_min_buf_size_decides_the_warmup_floor() -> None:
    """`train.min_buf_size` -> `step.py` O4: below it the learner sees nothing."""
    h = _coordinator(min_buf_size=10**6)
    assert _drive(h, steps=1)[0].in_warmup is True
    assert h.trainer.step == 0, "a warmup step must not train"

    h = _coordinator(min_buf_size=1)
    assert _drive(h, steps=1)[0].in_warmup is False


def test_replay_capacity_is_the_window_the_run_publishes_and_the_preflight_sizes() -> None:
    """`train.replay_capacity` -> `StepCoordinatorConfig.capacity` -> the `buffer_capacity`
    the warmup event publishes (and, in the tool, the REAL `ReplayBuffer` the preflight
    builds)."""
    h = _coordinator(capacity=31_337, min_buf_size=10**6)
    _drive(h, steps=1)
    stats = h.sink.named("system_stats")
    assert stats and stats[-1]["buffer_capacity"] == 31_337, (
        f"the published window must be the configured one; got {stats}"
    )


def test_the_capacity_schedule_ramps_the_buffer_at_its_own_step() -> None:
    """`train.replay_capacity_schedule` -> `step.py` D1. One registry leaf (NIT-3), so both
    inner names are demonstrated here: the STEP decides when, the CAPACITY decides what."""
    h = _coordinator(buffer_schedule=({"step": 2, "capacity": 555_555},))
    _drive(h, steps=1, games=1)
    assert h.buffer.resizes == [], "the ramp must not fire before its own step"
    _drive(h, steps=3, games=1)
    assert h.buffer.resizes == [555_555], (
        f"the ramp must fire once, at step 2, to the configured capacity; got {h.buffer.resizes}"
    )


def test_training_steps_per_game_and_max_train_burst_set_the_step_budget() -> None:
    """Both -> `step.py` O6 `_steps_budget(new_games, per_game, burst)`. Driven together
    because the budget is `min(max(1, games*per_game), burst)` and each knob is the binding
    term in exactly one of the two drives — a wire that fed one from the other would show up
    as the wrong drive being clamped."""
    ratio_bound = _coordinator(training_steps_per_game=2.0, max_train_burst=100)
    assert _drive(ratio_bound, steps=1, games=3)[0].steps_run == 6

    burst_bound = _coordinator(training_steps_per_game=2.0, max_train_burst=4)
    assert _drive(burst_bound, steps=1, games=3)[0].steps_run == 4, (
        "the burst ceiling must bind when the ratio would exceed it"
    )


def test_batch_size_is_the_batch_the_assembler_is_asked_for(monkeypatch) -> None:
    """`train.batch_size` -> `step.py::_run_training_step`. THE finding this key closes: the
    line read `train_cfg.get("batch_size", full_config.get("batch_size", 256))`, both lookups
    missed on the production path, and the run's real batch size was the literal 256 while
    `StepCoordinatorConfig.batch_size` sat beside it unread (WPMINT K-A). The minted value is
    256 for that reason; the drive below uses 41 so a surviving literal is visible."""
    h = _coordinator(pretrained=_Buffer(size=500), batch_size=41)
    calls = _mixed_batch_calls(monkeypatch, h, steps=3)
    assert {call["batch_size"] for call in calls} == {41}, (
        f"the assembler must be asked for the CONFIGURED batch; got {calls}"
    )
    assert 256 not in {call["batch_size"] for call in calls}, (
        "the `256` literal must be gone, not merely shadowed"
    )


def test_augment_reaches_both_training_paths(monkeypatch) -> None:
    """`train.augment` -> `trainer.train_step(augment=)` on the plain path and
    `assemble_mixed_batch(augment=)` on the mixed one. Both, because the knob is read twice
    and a wire that fixed one would leave the other on a code-side False."""
    plain = _coordinator(augment=True)
    _drive(plain, steps=2, games=1)
    assert plain.trainer.augment_seen and all(plain.trainer.augment_seen)

    mixed = _coordinator(pretrained=_Buffer(size=500), augment=True)
    assert all(call["augment"] for call in _mixed_batch_calls(monkeypatch, mixed, steps=3))


def test_recency_weight_reaches_the_assemblers_recency_window(monkeypatch) -> None:
    """`train.recency_weight` -> `assemble_mixed_batch`'s recency weighting."""
    h = _coordinator(pretrained=_Buffer(size=500), recency_weight=0.43)
    assert {call["recency_weight"]
            for call in _mixed_batch_calls(monkeypatch, h, steps=3)} == {0.43}


def test_the_three_mixing_knobs_decide_the_pretrained_share_of_each_batch(monkeypatch) -> None:
    """The `mixing_*` trio -> `_compute_pretrained_weight` -> `n_pre`, the number of batch
    slots drawn from the pretrained corpus. Observed as `n_pre`, not as the weight, because
    `w_pre` is an intermediate and the batch composition is what the registry cites.

    Three drives, one per knob, over the same batch size: the floor alone, a decayed start
    above it, and a decay horizon long enough to keep the start intact. If any one of the
    three reached nothing, two of these would collapse onto the same `n_pre`.
    """
    floor_only = _coordinator(pretrained=_Buffer(size=500), batch_size=100,
                              mixing_initial_w=0.0, mixing_min_w=0.5, mixing_decay_steps=1.0)
    assert {call["n_pre"]
            for call in _mixed_batch_calls(monkeypatch, floor_only, steps=3)} == {50}, (
        "with initial_w 0 the floor is the whole schedule: 50 of 100 slots"
    )

    fast_decay = _coordinator(pretrained=_Buffer(size=500), batch_size=100,
                              mixing_initial_w=1.0, mixing_min_w=0.1, mixing_decay_steps=1e-9)
    assert {call["n_pre"]
            for call in _mixed_batch_calls(monkeypatch, fast_decay, steps=3)} == {10}, (
        "a decay horizon far below one step drives w_pre to the floor immediately"
    )

    slow_decay = _coordinator(pretrained=_Buffer(size=500), batch_size=100,
                              mixing_initial_w=1.0, mixing_min_w=0.1, mixing_decay_steps=1e9)
    assert {call["n_pre"]
            for call in _mixed_batch_calls(monkeypatch, slow_decay, steps=3)} == {100}, (
        "the SAME initial and floor with a long horizon must keep the start — that is "
        "mixing_decay_steps deciding, and it is the arm the other two cannot fake"
    )


def test_bot_batch_share_allocates_batch_slots_from_the_bot_corpus(monkeypatch) -> None:
    """`train.bot_batch_share` -> `n_bot = round(share * batch_size)`."""
    h = _coordinator(pretrained=_Buffer(size=500), bot=_Buffer(size=500), batch_size=100,
                     bot_batch_share=0.25)
    assert {call["n_bot"] for call in _mixed_batch_calls(monkeypatch, h, steps=3)} == {25}

    none = _coordinator(pretrained=_Buffer(size=500), bot=_Buffer(size=500), batch_size=100,
                        bot_batch_share=0.0)
    assert {call["n_bot"]
            for call in _mixed_batch_calls(monkeypatch, none, steps=3)} == {0}


def test_the_grad_norm_knobs_decide_whether_the_hard_abort_fires() -> None:
    """`train.hard_gn_threshold` / `train.hard_gn_min_steps` -> `step.py` D3. Three drives:
    below the threshold nothing fires, above it the run stops after exactly `min_steps`
    consecutive breaches, and a `min_steps` beyond the drive keeps it silent — so each knob
    is the binding term in one drive and not the other.

    This is also the gate WPMINT Phase K-B registered as a DEFERRED armed-abort row: shipped
    at `1e9` it can never fire, and `Mechanism.CONFIG_THRESHOLD_BELOW_CEILING` is what says so
    against `monitor.alert_grad_norm_max`.
    """
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
    """`train.terminal_eval_enabled` -> `coordinator/drain.py::run_terminal_eval`. Until
    WPMINT Phase K-A this fact had THREE authorities (no key, a dataclass `= True`, and a
    `getattr(cfg, ..., True)` fallback); K-A retired the fallback and this key retires the
    dataclass default, which is why five composition tests could stop monkeypatching the
    production builder to turn it off."""
    from mantis.train.coordinator import drain

    calls: list[str] = []
    pipeline = SimpleNamespace(
        run_evaluation=lambda *a, **k: calls.append("terminal") or {"kicked": True},
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
    publishes the value. LAW-16 calls this guard always-armed while `watchdog.py`'s own
    contract lets `<= 0` disable the fire AND still emit the arm log — the schema's `gt=0` is
    what makes that posture unwritable, so the arm below shows the value the config really
    sets."""
    h = _coordinator(selfplay_stall_timeout_sec=67.0)
    _drive(h, steps=1, games=1)
    armed = h.sink.named("selfplay_stall_watchdog_armed")
    assert armed and armed[-1]["timeout_sec"] == 67.0, (
        f"the watchdog must arm on the configured budget; got {armed}"
    )
    assert armed[-1]["enabled"] is True


# ══ no second authority survives anywhere on the path (R1/LAW-08/R83) ══════════════════
def test_the_builder_takes_knobs_as_a_required_keyword_only_parameter() -> None:
    """MF-2 Attack B on the fourth config-authored fact: a parameter DEFAULT would move the
    authority from the builder BODY to the builder SIGNATURE, leaving every
    `dataclasses.fields()` assertion green while a caller that omitted the argument silently
    inherited nineteen postures."""
    param = inspect.signature(_step_coordinator_config).parameters.get("knobs")
    assert param is not None, (
        "`_step_coordinator_config` must take `knobs`: the nineteen are `train.*` keys and "
        "arrive from `resolve_coordinator_knobs`, never from a literal here"
    )
    assert param.default is inspect.Parameter.empty, (
        f"knobs carries a parameter default ({param.default!r}) — the literals did not die, "
        "they MIGRATED from the builder body to the builder signature"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_no_coordinator_field_carries_a_code_side_default_and_the_dead_six_are_gone() -> None:
    """EVERY `StepCoordinatorConfig` field must be MISSING-defaulted, and the six dead ones
    must be deleted rather than authored.

    Both halves are the same rule seen from two sides. A surviving default is a second
    authority a caller inherits (R1/R83). A surviving DEAD field would have to become a config
    key to satisfy this class's "every field is config-authored" invariant, and a config key
    with no live consumer is the R1/LAW-08 violation the bijection exists to catch — which is
    why adjudication call K-a deleted them (re-verified at HEAD by grep AND by recording every
    attribute read on a live instance across the whole tier).
    """
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


def test_the_resolver_is_the_only_read_of_the_nineteen_keys(smoke_run_config) -> None:
    """`resolve_coordinator_knobs` returns exactly what the loaded config holds, key for key —
    the transport arm at the resolver. A resolver that dropped, defaulted or CROSSED two
    fields would still satisfy the per-key mutations above for whichever key it happened to
    carry, and the schema->field rename is exactly where a crossing would be easiest."""
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
    }, "the spec must carry exactly the nineteen authored knobs and nothing else"
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
    """The card's closing claim, asserted rather than described: with `stop_step`,
    `draw_rate_abort`, `drain_caps` and `knobs` all arriving as parameters, EVERY
    `StepCoordinatorConfig` field the builder sets comes from one of them.

    Driven by construction, not by reading source: build with a spec whose every value is
    distinguishable and assert the built object carries them. A surviving literal for any knob
    would show up as that field disagreeing with the spec.
    """
    distinguishable = CoordinatorKnobsSpec(**{
        _SCHEMA_TO_FIELD.get(key, key): (
            tuple({"step": s["step"], "capacity": s["capacity"]} for s in value)
            if key == "replay_capacity_schedule" else value)
        for key, value in _DISTINGUISHABLE.items()
    })
    built = _step_coordinator_config(stop_step=11, draw_rate_abort=None,
                                     drain_caps=_DRAIN_CAPS, knobs=distinguishable)
    for field in dataclasses.fields(CoordinatorKnobsSpec):
        assert getattr(built, field.name) == getattr(distinguishable, field.name), (
            f"`_step_coordinator_config` overrode {field.name} with something other than the "
            "resolved spec — a literal survives in the builder body"
        )
    assert built.stop_step == 11 and built.draw_rate_abort is None
