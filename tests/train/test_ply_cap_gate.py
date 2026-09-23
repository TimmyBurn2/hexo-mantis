"""The ply-cap attractor halt (R352(c)): rule, pool producer, coordinator gate, planted break."""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.ply_cap import PlyCapAbortSpec, resolve_ply_cap_abort
from _monitor_config import monitor_config
from mantis.monitor.rules import check_ply_cap_attractor
from mantis.run import _step_coordinator_config
from mantis.selfplay.instrumentation import PoolInstrumentation
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.util.constants import PLY_CAP_RING_GAMES
from mantis._engine import HexgBuffer
from _coordinator_pool import CoordinatorPoolStub

_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"


# The rule.

def test_the_rule_fires_strictly_above_the_rate_at_or_past_min_step() -> None:
    msg = check_ply_cap_attractor(0.51, 3000, rate=0.5, window_games=600, min_step=3000)
    assert msg is not None and "ply-cap" in msg and "0.51" in msg and "600" in msg


@pytest.mark.parametrize("observed", [0.5, 0.49, 0.0])
def test_the_rule_is_silent_at_or_below_the_rate(observed: float) -> None:
    assert check_ply_cap_attractor(observed, 3000, rate=0.5, window_games=600, min_step=3000) is None


def test_the_rule_is_silent_before_min_step_however_high_the_rate() -> None:
    assert check_ply_cap_attractor(0.99, 2999, rate=0.5, window_games=600, min_step=3000) is None


# The resolver.

def test_the_resolver_returns_none_on_the_explicit_off_and_the_spec_otherwise() -> None:
    train = load_config(_CONFIG).train
    assert train.ply_cap_abort is None
    assert resolve_ply_cap_abort(train) is None
    armed = SimpleNamespace(ply_cap_abort=SimpleNamespace(rate=0.5, window_games=600, min_step=3000))
    assert resolve_ply_cap_abort(armed) == PlyCapAbortSpec(0.5, 600, 3000)


# The producer: one ring, pool-wide, in completion order.

def _instrumentation() -> tuple[PoolInstrumentation, threading.Lock]:
    return PoolInstrumentation(log_investigation_metrics=False, cluster_threshold=8), threading.Lock()


def _complete(inst: PoolInstrumentation, lock: threading.Lock, terminal_reason: int) -> None:
    inst.on_game_complete(lock, 0, [], 0, terminal_reason, 1, 1, 1, 0)


def test_the_window_counts_cap_games_over_the_last_n_completions() -> None:
    inst, lock = _instrumentation()
    for reason in (2, 0, 2, 2, 1, 3, 2, 0):  # 2 = ply_cap
        _complete(inst, lock, reason)
    assert inst.ply_cap_window_counts(lock, 4) == (1, 4), "the LAST four: (1, 3, 2, 0) → one cap"
    assert inst.ply_cap_window_counts(lock, 8) == (4, 8)


def test_a_window_wider_than_the_games_played_reports_the_games_it_has() -> None:
    inst, lock = _instrumentation()
    for reason in (2, 2, 0):
        _complete(inst, lock, reason)
    assert inst.ply_cap_window_counts(lock, 600) == (2, 3), "games < window is the caller's skip"


def test_the_ring_holds_PLY_CAP_RING_GAMES_completions_and_no_more() -> None:
    inst, lock = _instrumentation()
    for _ in range(PLY_CAP_RING_GAMES):
        _complete(inst, lock, 2)
    _complete(inst, lock, 0)
    caps, games = inst.ply_cap_window_counts(lock, PLY_CAP_RING_GAMES + 100)
    assert games == PLY_CAP_RING_GAMES and caps == PLY_CAP_RING_GAMES - 1


# The gate through the coordinator.

def _filled_hexg(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    hb = HexgBuffer(capacity, "gnn_axis_v1", 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)][: 2 + (i % 2)]
        hb.push_graph_position(stones, [(2, 0, 0.6), (1, 1, 0.4)], 1, 30, 2 + i, True,
                               1.0 if i % 2 == 0 else -1.0, True, 10 + i)
    return hb


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 2.0, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self._hexg = _filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None: ...

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _harness(flags: list[int], spec: PlyCapAbortSpec | None, *, gate_interval: int = 4):
    cfg = load_config(_CONFIG)
    base = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=None, ply_cap_abort=spec,
        drain_caps=resolve_drain_caps(cfg.monitor), gate_interval=gate_interval,
        knobs=resolve_coordinator_knobs(cfg.train))
    config = dataclasses.replace(base, eval_interval=0, log_interval=gate_interval,
                                 min_buf_size=10, max_train_burst=1,
                                 training_steps_per_game=1.0, hard_gn_threshold=1e9)
    shutdown = ShutdownState()
    sink = _Sink()
    pool = CoordinatorPoolStub(flags)
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), bufs=None, config=config,
        full_config={"identity": {"encoding": "gnn_axis_v1", "representation": "graph"},
                     "train": {"microbatch_caps": {"max_edges": 100_000_000, "max_nodes": 4_000_000},
                               "fast_policy_weight": 0.0},
                     "selfplay": {"n_workers": 1}},
        train_cfg={}, mixing_cfg={}, sink=sink, heartbeat=None, monitor_cfg=monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _drive(h, steps: int) -> None:
    """One game completes per coordinator step, so game i lands before training step i + 1."""
    for _ in range(steps):
        if not h.shutdown.running:
            return
        h.pool.games_completed += 1
        h.coord.step()


#: A 6-game window past step 8: the attractor arrives as caps from game 6 on.
_SPEC = PlyCapAbortSpec(rate=0.5, window_games=6, min_step=8)
_ATTRACTOR = [0] * 6 + [1] * 30


def test_the_halt_fires_at_the_first_step_past_min_step_whose_window_is_over_the_rate() -> None:
    """PRODUCER TEST: caps from game 6 on, the 6-game window first exceeds 0.5 at game 10 → step 10."""
    h = _harness(_ATTRACTOR, _SPEC)
    _drive(h, 30)
    assert h.shutdown.running is False
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and aborts[0]["rule"] == "ply_cap_attractor", aborts
    assert aborts[0]["step"] == 10, aborts[0]
    assert h.shutdown.abort_rule == "ply_cap_attractor"
    assert h.pool.window_calls and set(h.pool.window_calls) == {6}, "the window is the MINTED one"


def test_the_gate_is_checked_every_training_step_not_only_at_gate_boundaries() -> None:
    """Step 10 is not a gate boundary (interval 4): a boundary-clocked gate would fire at 12."""
    h = _harness(_ATTRACTOR, _SPEC, gate_interval=4)
    _drive(h, 30)
    assert h.sink.named("hard_abort")[0]["step"] == 10


def test_min_step_gates_the_fire_and_not_the_observation() -> None:
    """Caps from game 0: every window reads 1.0, yet the fire waits for step 8."""
    h = _harness([1] * 40, _SPEC)
    _drive(h, 30)
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and aborts[0]["step"] == 8
    gates = h.sink.named("monitor_gates")[0]
    assert gates["step"] == 4 and gates["ply_cap_rate"] is None, (
        "at step 4 only 4 games exist: below the window is NO OBSERVATION, never a rate")


def test_a_rate_at_exactly_the_bar_does_not_fire() -> None:
    h = _harness([1, 0] * 40, _SPEC)  # every 6-window reads exactly 0.5
    _drive(h, 30)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["ply_cap_rate"] == pytest.approx(0.5)
    assert gates["gates"]["ply_cap_attractor"]["fires"] == 0
    assert gates["gates"]["ply_cap_attractor"]["checks"] >= 20


def test_the_explicit_off_posture_never_fires_and_is_skip_counted() -> None:
    h = _harness([1] * 40, None)
    _drive(h, 30)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["gates"]["ply_cap_attractor"]["skips"] > 0, "a disarmed gate is SKIP-counted, not silent"
    assert gates["ply_cap_abort_rate"] is None and gates["ply_cap_window_games"] is None
    assert h.pool.window_calls == [], "a disarmed gate reads no producer"


def test_below_the_window_the_gate_makes_no_observation_and_skip_counts() -> None:
    h = _harness([1] * 40, PlyCapAbortSpec(rate=0.5, window_games=600, min_step=1))
    _drive(h, 30)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["gates"]["ply_cap_attractor"]["skips"] >= 20
    assert gates["ply_cap_rate"] is None


def test_the_live_terms_ride_monitor_gates() -> None:
    h = _harness([0] * 40, _SPEC)
    _drive(h, 12)
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["ply_cap_abort_rate"] == 0.5 and gates["ply_cap_window_games"] == 6
    assert gates["ply_cap_rate"] == 0.0


def test_the_planted_break_a_dead_producer_is_caught_by_the_producer_test() -> None:
    """LAW-07's mutation self-test: a producer reporting NO games makes the fire above never come."""
    h = _harness(_ATTRACTOR, _SPEC)
    h.pool.ply_cap_window_counts = lambda window_games: (0, 0)  # type: ignore[method-assign]
    _drive(h, 30)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["gates"]["ply_cap_attractor"]["fires"] == 0
    assert gates["gates"]["ply_cap_attractor"]["skips"] >= 20, "a dead producer reads as SKIPS"


def test_the_manifest_row_is_deferred_with_a_step_floor_cadence_and_exit_50() -> None:
    from mantis.config.armed_aborts import MANIFEST, Cadence, Status, exit_code_for_abort
    from mantis.monitor.heartbeat import PLY_CAP_ATTRACTOR_EXIT_CODE

    row = next(r for r in MANIFEST if r.name == "ply_cap_attractor")
    assert row.status is Status.DEFERRED and row.cadence is Cadence.TRAIN_STEP_FLOOR
    assert row.cadence_paths == ("train.ply_cap_abort.min_step",)
    assert exit_code_for_abort("ply_cap_attractor") == PLY_CAP_ATTRACTOR_EXIT_CODE == 50
    # run7's minted floor: the fire can come at step 3000 exactly, in the train-step clock.
    assert Cadence.TRAIN_STEP_FLOOR.earliest_fire_step((3000,), period_steps=1.0) == 3000.0
