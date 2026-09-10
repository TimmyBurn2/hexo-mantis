"""`compose_run` with a REAL `RunConfig` syncs on the configured cadence.

Every other drive in the suite composes with `config=SimpleNamespace()`, which leaves
`_resolve_actor_sync_cadence_steps`'s real-config arm unexercised. Two axes stay pinned even
here because no test can vary them: `_step_coordinator_config` is monkeypatched and
`build_run_safety` is faked.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mantis.run
from mantis.config.schema.core import RunConfig
from mantis.train.coordinator.config import StepCoordinatorConfig

_CADENCE = 2
_STOP_STEP = 6


def _frozen_payload():
    """Load the frozen schema oracle's payload builder by path, since `tests` is not a package.

    Built from the payload rather than a shipped config so this pins the resolver's behaviour
    and not one config's current values.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "test_actor_sync_schema.py"
    spec = importlib.util.spec_from_file_location("_frozen_schema_for_fa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._payload


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _Pool:
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
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
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

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


#: The unpatched production builder, captured at import so the patch below can delegate to it
#: without re-entering itself.
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


def _bounded_config(**kwargs) -> StepCoordinatorConfig:
    """Apply the harness's own deltas over the real builder, passing config values through."""
    return dataclasses.replace(_PRODUCTION_BUILDER(**kwargs),
                               terminal_eval_enabled=False, eval_interval=1000,
                               log_interval=1, stop_step=_STOP_STEP)


def _real_run_config() -> RunConfig:
    payload = _frozen_payload()(
        # The reachability bound binds on the run length, so the threshold must fit inside
        # `cadence < threshold < max_train_steps`.
        train_over={"actor_sync_cadence_steps": _CADENCE, "max_train_steps": _STOP_STEP},
        monitor_over={"actor_lag_threshold_steps": _CADENCE + 2},
    )
    # The frozen payload mints `eval_enabled: True`, this file's production posture.
    assert payload["eval_enabled"] is True
    return RunConfig(**payload)


def _drive(monkeypatch, *, eval_enabled: bool = True):
    """Compose with a real RunConfig; return (captured build_run_safety kwargs, pool, trainer).

    `eval_enabled` is a config fact — `compose_run` has no such parameter — so the posture
    travels on the payload `_real_run_config` builds.
    """
    import mantis.train.anchor as _anchor

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
    # What this patch buys is `terminal_eval_enabled=False`: the terminal eval round would
    # reach eval/snapshot.py's `.arch` read on a fake model, and that knob has no config key.
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _bounded_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )
    return captured, pool, trainer, _real_run_config()


def test_a_real_run_config_actually_reaches_the_cadence_resolver(tmp_path, monkeypatch, mk_graph_buffer):
    """The premise. If the real arm is not taken, everything below is vacuous."""
    captured, pool, trainer, cfg = _drive(monkeypatch)
    assert mantis.run._resolve_actor_sync_cadence_steps(cfg) == _CADENCE, (
        "the real-config arm did not resolve the configured cadence — this test would "
        "otherwise pass while exercising the smoke path it exists to avoid"
    )


def test_sync_follows_the_configured_cadence_under_a_real_config(tmp_path, monkeypatch, mk_graph_buffer):
    """Kill a resolver unit-slip on its own: a `* 1000` slip makes the cadence unreachable
    inside the run, so the actor takes one unconditional first sync and then freezes."""
    captured, pool, trainer, cfg = _drive(monkeypatch)
    mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert trainer.step >= _STOP_STEP - 1, "harness precondition: the run actually stepped"
    expected_min = trainer.step // _CADENCE
    assert len(pool.sync_payloads) >= expected_min, (
        f"actor synced {len(pool.sync_payloads)}x over {trainer.step} steps at cadence "
        f"{_CADENCE}; expected at least {expected_min}. A single sync then silence is the "
        f"frozen actor this WP removed"
    )
    gaps = [b - a for a, b in zip(pool.step_calls, pool.step_calls[1:], strict=False)]
    assert all(g <= _CADENCE for g in gaps), (
        f"sync gaps {gaps} exceed the configured cadence {_CADENCE}"
    )


def test_lag_callables_read_live_sources_under_a_real_config(tmp_path, monkeypatch, mk_graph_buffer):
    """Kill a lag-lambda `getattr(config, X, None)` fallback on its own.

    Asserting only sync volume would let that survive; asserting only the lambdas would let a
    resolver unit-slip survive.
    """
    captured, pool, trainer, cfg = _drive(monkeypatch)
    mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    actor_fn, learner_fn = captured["actor_ckpt_step_fn"], captured["learner_step_fn"]
    actor_before = actor_fn()
    assert actor_before == pool.step_calls[-1], (
        "actor_ckpt_step_fn is not reading the live sync engine under a real config"
    )

    trainer.step += 777
    assert learner_fn() == trainer.step, "learner_step_fn is not reading the live trainer"
    assert actor_fn() == actor_before, (
        "actor_ckpt_step_fn moved when only the LEARNER advanced — it is reading the "
        "wrong source, which makes the lag invariant blind"
    )
    assert learner_fn() - actor_fn() == 777 + (trainer.step - 777 - actor_before), (
        "lag must be learner-minus-actor and grow when the actor falls behind"
    )
