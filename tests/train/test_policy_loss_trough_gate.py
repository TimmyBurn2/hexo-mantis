"""The policy-loss trough halt (R350(b)(iv)): the rule, and its producer through the coordinator."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.config.resolve.policy_loss_trough import (
    PolicyLossTroughAbortSpec,
    resolve_policy_loss_trough_abort,
)
from _monitor_config import monitor_config
from mantis.monitor.rules import check_policy_loss_trough
from mantis.run import _step_coordinator_config
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from _graph_drive import GRAPH_FULL_CONFIG, filled_hexg
from _spy import SpyEventSink

_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml"


def test_the_rule_fires_on_consec_windows_above_the_reference_inside_the_bound() -> None:
    msg = check_policy_loss_trough([2.5, 2.6, 2.7], 4000, reference=2.28, delta_nats=0.2,
                                   consec=3, max_step=5000)
    assert msg is not None and "trough" in msg and "2.480" in msg


def test_the_rule_fires_at_exactly_reference_plus_delta() -> None:
    assert check_policy_loss_trough([2.48, 2.48, 2.48], 4000, reference=2.28, delta_nats=0.2,
                                    consec=3, max_step=5000) is not None


@pytest.mark.parametrize("history", [[2.5, 2.6], [2.5, 2.3, 2.7], [2.47, 2.47, 2.47]])
def test_the_rule_is_silent_short_of_consec_or_below_the_bar(history: list[float]) -> None:
    assert check_policy_loss_trough(history, 4000, reference=2.28, delta_nats=0.2, consec=3,
                                    max_step=5000) is None


def test_the_rule_is_silent_past_max_step_by_contract() -> None:
    """The halt names an EARLY signature; a rise at 20k is INVESTIGATION-1's, not this rule's."""
    assert check_policy_loss_trough([3.0, 3.0, 3.0], 5001, reference=2.28, delta_nats=0.2,
                                    consec=3, max_step=5000) is None
    assert check_policy_loss_trough([3.0, 3.0, 3.0], 5000, reference=2.28, delta_nats=0.2,
                                    consec=3, max_step=5000) is not None


def test_the_resolver_returns_none_on_the_explicit_off_and_the_spec_otherwise() -> None:
    train = load_config(_CONFIG).train
    assert train.policy_loss_trough_abort is None
    assert resolve_policy_loss_trough_abort(train) is None
    armed = SimpleNamespace(policy_loss_trough_abort=SimpleNamespace(delta_nats=0.2, consec=3,
                                                                     max_step=5000))
    assert resolve_policy_loss_trough_abort(armed) == PolicyLossTroughAbortSpec(0.2, 3, 5000)


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self.games_completed = 0
        self.n_workers = 1
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
    def buffer_composition(self) -> dict[str, Any]:
        return {}
    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)
    def current_stride5_p90(self) -> int:
        return 1
    def runner_stats(self) -> Any:
        return _RunnerStats()
    def update_checkpoint_step(self, step: int) -> None: ...


class _Trainer:
    """Its policy loss is a SCRIPT over the step counter."""

    def __init__(self, script) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"
        self._script = script

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": float(self._script(self.step)), "value_loss": 0.4,
                "grad_norm": 0.1, "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000
        self._hexg = filled_hexg()

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None: ...

    def sample_graph_batch(self, n: int, *, augment: bool = False, recent_frac: float = 0.0,
                           n_threads: int = 1):
        return self._hexg.sample_graph_batch(n, augment=augment, recent_frac=recent_frac,
                                             n_threads=n_threads)



def _harness(script, spec: PolicyLossTroughAbortSpec | None, *, gate_interval: int = 2):
    cfg = load_config(_CONFIG)
    base = _step_coordinator_config(
        stop_step=10**9, draw_rate_abort=None, policy_loss_trough_abort=spec, ply_cap_abort=None,
        drain_caps=resolve_drain_caps(cfg.monitor), gate_interval=gate_interval,
        knobs=resolve_coordinator_knobs(cfg.train))
    config = dataclasses.replace(base, eval_interval=0, log_interval=gate_interval,
                                 min_buf_size=10, max_train_burst=1,
                                 training_steps_per_game=1.0, hard_gn_threshold=1e9)
    shutdown = ShutdownState()
    sink = SpyEventSink()
    pool = _Pool()
    coord = StepCoordinator(
        trainer=_Trainer(script), buffer=_Buffer(),
        pool=pool, eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=shutdown, eval_model=object(), config=config,
        full_config=GRAPH_FULL_CONFIG,
        sink=sink, heartbeat=None, monitor_cfg=monitor_config(),
    )
    return SimpleNamespace(coord=coord, pool=pool, shutdown=shutdown, sink=sink)


def _drive(h, steps: int) -> None:
    for _ in range(steps):
        if not h.shutdown.running:
            return
        h.pool.games_completed += 1
        h.coord.step()


_SPEC = PolicyLossTroughAbortSpec(delta_nats=0.2, consec=3, max_step=5000)


def _rising(step: int) -> float:
    # Window 1 (steps 1-2) reads 2.28; every later window sits 0.3 above it.
    return 2.28 if step <= 2 else 2.58


def test_the_halt_fires_on_the_trough_signature_through_the_one_channel() -> None:
    """PRODUCER TEST: reference 2.28, three windows at +0.30, ONE `hard_abort` at boundary 4 (step 8)."""
    h = _harness(_rising, _SPEC)
    _drive(h, 20)
    assert h.shutdown.running is False
    aborts = h.sink.named("hard_abort")
    assert len(aborts) == 1 and aborts[0]["rule"] == "policy_loss_trough"
    assert aborts[0]["step"] == 8, aborts[0]
    assert h.shutdown.abort_rule == "policy_loss_trough"
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["gates"]["policy_loss_trough"]["fires"] == 1
    assert gates["policy_loss_reference"] == pytest.approx(2.28)


def test_a_flat_loss_never_fires_and_the_reference_is_the_first_window() -> None:
    h = _harness(lambda step: 2.28, _SPEC)
    _drive(h, 20)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["policy_loss_reference"] == pytest.approx(2.28)
    assert len(gates["policy_loss_window_means"]) == _SPEC.consec, "the ring is sized by consec"


def test_a_rise_that_is_not_consecutive_does_not_fire() -> None:
    def script(step: int) -> float:
        window = (step - 1) // 2
        return 2.28 if window == 0 or window % 2 == 0 else 2.58

    h = _harness(script, _SPEC)
    _drive(h, 20)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []


def test_the_explicit_off_posture_never_fires_however_bad_the_rise() -> None:
    h = _harness(lambda step: 2.28 if step <= 2 else 9.0, None)
    _drive(h, 20)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []
    gates = h.sink.named("monitor_gates")[-1]
    assert gates["gates"]["policy_loss_trough"]["skips"] > 0, "a disarmed gate is SKIP-counted, not silent"
    assert gates["policy_loss_trough_delta_nats"] is None


def test_past_max_step_the_halt_is_silent() -> None:
    h = _harness(_rising, PolicyLossTroughAbortSpec(delta_nats=0.2, consec=3, max_step=7))
    _drive(h, 20)
    assert h.shutdown.running is True and h.sink.named("hard_abort") == []


def test_the_manifest_row_is_deferred_with_the_bounded_cadence_and_exit_49() -> None:
    from mantis.config.armed_aborts import MANIFEST, Cadence, Status, exit_code_for_abort
    from mantis.monitor.heartbeat import POLICY_LOSS_TROUGH_EXIT_CODE

    row = next(r for r in MANIFEST if r.name == "policy_loss_trough")
    assert row.status is Status.DEFERRED and row.cadence is Cadence.GATE_INTERVAL_CONSEC_BOUNDED
    assert exit_code_for_abort("policy_loss_trough") == POLICY_LOSS_TROUGH_EXIT_CODE == 49
    # run7's proposed terms at gate_interval 1000: the 4th boundary, inside the bound.
    assert Cadence.GATE_INTERVAL_CONSEC_BOUNDED.earliest_fire_step((3, 5000), period_steps=1000.0) == 4000.0
    # A bound the windows cannot reach is "can never fire", never a fabricated step.
    assert Cadence.GATE_INTERVAL_CONSEC_BOUNDED.earliest_fire_step((3, 3000), period_steps=1000.0) == float("inf")


def test_a_window_left_unconsumed_at_a_boundary_bundle_is_folded_on_restore() -> None:
    """B-7: a boundary bundle is written before the boundary consumed its window; the restore folds it."""
    h = _harness(_rising, _SPEC, gate_interval=2)
    # Steps 1-4: window 1 becomes the reference at 2, window 2's mean lands at 4.
    _drive(h, 4)
    assert h.coord._policy_loss_reference is not None
    assert len(h.coord._policy_loss_window_means) == 1
    # A bundle written at step 6 INSIDE the step: the window holds step 5 (and the boundary 6
    # has not consumed it). Model that state directly.
    h.coord._train_step = 6
    h.coord._policy_loss_window = [2.58]
    carried = h.coord.guard_state()
    assert carried["policy_loss_window"] == [2.58]

    resumed = _harness(_rising, _SPEC, gate_interval=2)
    resumed.coord._train_step = 6
    resumed.coord.restore_guard_state(carried)
    assert resumed.coord._policy_loss_window == [], "folded, not carried into the next window"
    assert len(resumed.coord._policy_loss_window_means) == 2, "the boundary's mean was taken"
    # One more window (steps 7-8) is the third consecutive rise: the halt fires at 8.
    resumed.coord.trainer.step = 6
    _drive(resumed, 2)
    assert resumed.shutdown.running is False, "consec 3 met on the resumed process's first boundary"
