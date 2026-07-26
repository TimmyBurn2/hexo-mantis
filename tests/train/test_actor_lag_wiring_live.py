"""The composed lag-watchdog callables read the LIVE engine and the LIVE trainer.

REVIEW-impl MF-1. Every lag oracle injects its own callables, and every `compose_run` test
fakes `build_run_safety` — so the two lambdas the composition root actually hands to the
watchdog (`run.py`: `actor_ckpt_step_fn`, `learner_step_fn`) were pinned by **nothing**.
Replacing either with `lambda: 0`, or swapping them, passed all 1681 tests while blinding
or false-firing the exit-45 actor-lag invariant at run5.

That is the F-10 / LAW-07 phantom-gate class — a gate fed by nothing — on the very
invariant this WP ships. R4 is explicit: no gate input without a producer test. These are
that producer test.

Deliberately NOT frozen: written after ORACLE-WRITE, in response to a review finding.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mantis.run
from mantis.train.coordinator.config import StepCoordinatorConfig

_STOP_STEP = 3


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
    def __init__(self) -> None:
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
    def per_worker_draw_rates(self) -> dict[int, float]:
        return {}

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None: ...
    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.inference_sd = {"w": "SENTINEL"}

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


def _bounded_config() -> StepCoordinatorConfig:
    return StepCoordinatorConfig(
        eval_interval=0, log_interval=1, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=1, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        instrumentation_enabled=False, stop_step=_STOP_STEP,
        final_eval_drain_timeout_sec=900.0,
    )


def _compose_capturing_lag_fns(tmp_path, monkeypatch):
    """Run `compose_run` and return (captured_kwargs, pool, trainer).

    Captures what the composition root ACTUALLY hands `build_run_safety`, rather than
    what a test injects in its place.
    """
    captured: dict = {}
    pool, trainer = _Pool(), _Trainer()

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _capture)
    monkeypatch.setattr(mantis.run, "_default_step_coordinator_config", _bounded_config)
    mantis.run.compose_run(
        config=SimpleNamespace(), trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        eval_enabled=False,
    )
    return captured, pool, trainer


def test_composition_root_supplies_both_lag_callables(tmp_path, monkeypatch):
    captured, _pool, _trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch)
    for name in ("actor_ckpt_step_fn", "learner_step_fn"):
        assert name in captured, f"composition root did not supply {name}"
        assert callable(captured[name])


def test_learner_step_fn_reads_the_live_trainer(tmp_path, monkeypatch):
    """Not a captured snapshot: mutating the trainer must move the reading.

    A build-time snapshot would freeze `learner_step` and the lag would never grow, so the
    invariant could never fire however far the actor fell behind.
    """
    captured, _pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch)
    trainer.step = 4242
    assert captured["learner_step_fn"]() == 4242


def test_actor_ckpt_step_fn_reads_the_live_sync_engine(tmp_path, monkeypatch):
    """It must report the actor's real synced step, not a constant and not the trainer."""
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch)
    assert pool.step_calls, "harness precondition: the actor synced at least once"
    assert captured["actor_ckpt_step_fn"]() == pool.step_calls[-1]

    # Moving the LEARNER must not move the ACTOR reading — that would hide all lag.
    before = captured["actor_ckpt_step_fn"]()
    trainer.step = 99999
    assert captured["actor_ckpt_step_fn"]() == before


def test_the_two_lag_callables_are_not_swapped(tmp_path, monkeypatch):
    """Swapping them inverts the invariant into a permanent false-negative.

    `learner_step − actor_ckpt_step` would go negative rather than positive, so a starved
    actor would read as healthy no matter how far behind it fell.
    """
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch)
    trainer.step = pool.step_calls[-1] + 500

    learner = captured["learner_step_fn"]()
    actor = captured["actor_ckpt_step_fn"]()
    assert learner == trainer.step, "learner_step_fn is not reading the trainer"
    assert actor == pool.step_calls[-1], "actor_ckpt_step_fn is not reading the engine"
    assert learner - actor == 500, (
        f"lag must be learner-minus-actor and positive when the actor is behind; "
        f"got learner={learner} actor={actor}"
    )


def test_neither_callable_is_a_constant(tmp_path, monkeypatch):
    """Kills the `lambda: 0` stub directly — the cheapest way to blind the gate."""
    captured, pool, trainer = _compose_capturing_lag_fns(tmp_path, monkeypatch)
    trainer.step = 7
    assert captured["learner_step_fn"]() == 7
    trainer.step = 8
    assert captured["learner_step_fn"]() == 8, "learner_step_fn returns a constant"

    # The actor reading must be the engine's real synced step. The harness guarantees at
    # least one sync, so a zero here means the callable is a stub rather than a reading.
    actor = captured["actor_ckpt_step_fn"]()
    assert actor == pool.step_calls[-1] and actor > 0, (
        f"actor_ckpt_step_fn returned {actor}; expected the live synced step "
        f"{pool.step_calls[-1]}"
    )
