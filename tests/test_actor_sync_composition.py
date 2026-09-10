"""Continuous actor sync runs unimpaired with the gate, promotion and eval machinery ABSENT.

`eval_enabled=False` means none of the deploy side is constructed in this process, so sync
provably needs nothing it provides.

Bounded by construction: the coordinator's `stop_step` terminates the loop, no thread starts
(`build_run_safety` is faked), and no sleep is reached — the buffer is above floor and every
step sees fresh games.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mantis.run
import mantis.train.actor_sync  # noqa: F401 — import anchor

_STOP_STEP = 5


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _SyncRecordingPool:
    """Pool double with the sync recorders; `games_completed` yields one fresh game per read,
    so every step runs one burst."""

    def __init__(self) -> None:
        self._games = 0
        self.search_kind = "gumbel"
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draw_rate = 0.05  # the third outcome share
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

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

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
        self.device = "cpu"
        self.inference_sd = {"w": "SENTINEL"}

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

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


def test_compose_run_syncs_actor_on_cadence_without_eval(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
) -> None:
    """The dependency-absence proof: with no eval pipeline in the process at all, the pool
    still records cadence-consistent weight pushes and the actor's recorded step ends inside
    the cadence bound of the learner's."""
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
    # `stop_step` is config-authored, so this drives the PRODUCTION coordinator config with no
    # monkeypatch — which couples the step counts below to that builder's other knobs.

    handles = mantis.run.compose_run(
        # Reachability bound: cadence < threshold < max_train_steps, so _STOP_STEP >= 3.
        config=smoke_run_config(
            train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP,
                   # Real graph route per step; the minted 256 batch is drag.
                   "batch_size": 8},
            monitor={"actor_lag_threshold_steps": _STOP_STEP - 1},
            # The eval posture is the config's fact; no parameter can force it.
            eval_enabled=False),
        trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
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
    cadence = 1  # the composed config's cadence: the most-synced world
    assert trainer.step - pool.step_calls[-1] < cadence + 1, (
        f"actor_ckpt_step {pool.step_calls[-1]} must track learner_step {trainer.step} "
        f"within the cadence bound"
    )
    gaps = [b - a for a, b in zip(pool.step_calls, pool.step_calls[1:])]
    assert all(g <= cadence for g in gaps), (
        f"consecutive syncs must never be further apart than the cadence: {pool.step_calls}"
    )
