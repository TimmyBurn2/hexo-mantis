"""⊕ WPUF Phase U ORACLE — O-U3 behavioral half (+ O-U1 composition): with the gate,
promotion and eval machinery NEVER CONSTRUCTED (`eval_enabled=False`, `run.py`'s eval
branch not taken), continuous actor sync runs unimpaired through `compose_run`
(DESIGN_U §2.3/§8 O-U1 last bullet). Sync provably needs nothing the deploy side
provides, because the deploy side does not exist in the process.

RED-at-import until IMPL lands `mantis.train.actor_sync`.

DEVIATION FROM DESIGN PATH (logged in ORACLE_NOTES_U.md): DESIGN §8/§10 R-32 places this
test inside the existing `tests/test_run_composition.py`; ORACLE-WRITE's writable surface
is NEW files only (same precedent as tests/config/test_train_policy_value_target_
consistency.py's logged deviation), so it lives here. IMPL may merge it at port time.

Bounded by construction: the coordinator's `stop_step` terminates the loop (O2 sets
`shutdown.running=False`); no thread is started (build_run_safety is replaced by fakes);
no sleeps (the warmup/waiting branches are never entered: buffer above floor, fresh games
every step).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mantis.run
import mantis.train.actor_sync  # noqa: F401 — RED-at-import anchor (module does not exist yet)
from mantis.train.coordinator.config import StepCoordinatorConfig

_STOP_STEP = 5


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _SyncRecordingPool:
    """The routing-harness FakePool surface + start/stop + the ActorSyncTarget recorders.
    `games_completed` yields one fresh game per read so every step() runs one burst."""

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
        self.started = False
        self.stopped = False
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def check_producer_health(self) -> None:
        return None

    def per_worker_draw_rates(self) -> dict[int, float]:
        return {}

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

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

    def save_checkpoint(self, loss_info) -> None:
        return None


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None:
        return None

    def save_to_path(self, p) -> None:
        return None


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


def test_compose_run_syncs_actor_on_cadence_without_eval(tmp_path, monkeypatch) -> None:
    """The dependency-absence proof: `eval_enabled=False` means no gate, no promotion
    hooks, no eval pipeline exist ANYWHERE in the process, yet the pool records
    cadence-consistent weight pushes (fakes path = cadence 1, `_SMOKE_ACTOR_SYNC_
    CADENCE_STEPS`, the zero-staleness posture — DESIGN §5) and the actor's recorded
    step ends inside the cadence bound of the learner's."""
    pool = _SyncRecordingPool()
    trainer = _Trainer()

    def _fake_build_run_safety(**kwargs):
        return SimpleNamespace(
            sink=SimpleNamespace(emit=lambda e: None),
            registry=SimpleNamespace(beat=lambda s: None),
            watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
            heartbeat=lambda s: None,
        )

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_build_run_safety)
    # Bound the drive: the shipped smoke config's stop_step=0 never enters the burst loop;
    # a small positive stop_step exercises real training steps (harness move, not a pin).
    monkeypatch.setattr(mantis.run, "_default_step_coordinator_config", _bounded_config)

    handles = mantis.run.compose_run(
        config=SimpleNamespace(),  # no .train attribute → the smoke cadence (=1) path
        trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        eval_enabled=False,
    )

    assert handles.eval_pipeline is None, (
        "harness precondition: the deploy side must not exist in this process"
    )
    assert trainer.step >= 1, "harness precondition: at least one real training step ran"
    assert len(pool.sync_payloads) >= 1, (
        "with the gate/promotion machinery ABSENT, sync must run unimpaired — zero pushes "
        "means actor sync still depends on something the deploy side provides (R49 breach)"
    )
    assert all(sd is trainer.inference_sd for sd in pool.sync_payloads), (
        "every push must carry trainer.inference_state_dict()'s result (EMA-aware weights)"
    )
    assert pool.step_calls == sorted(set(pool.step_calls)), (
        f"recorded sync steps must be strictly increasing: {pool.step_calls}"
    )
    cadence = 1  # the fakes-path smoke cadence (DESIGN §5): the MOST-synced world
    assert trainer.step - pool.step_calls[-1] < cadence + 1, (
        f"actor_ckpt_step {pool.step_calls[-1]} must track learner_step {trainer.step} "
        f"within the cadence bound"
    )
    gaps = [b - a for a, b in zip(pool.step_calls, pool.step_calls[1:])]
    assert all(g <= cadence for g in gaps), (
        f"consecutive syncs must never be further apart than the cadence: {pool.step_calls}"
    )
