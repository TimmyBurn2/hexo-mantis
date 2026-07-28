"""`compose_run` with a REAL `RunConfig` syncs on the configured cadence (RED-TEAM-2 F-A).

All 11 `compose_run(...)` call sites in the suite — including both tests written to close
RED-TEAM-1's F-1 — passed `config=SimpleNamespace()`. Nothing anywhere composed with a real
`.train`-bearing `RunConfig`, so `_resolve_actor_sync_cadence_steps`'s real-config arm was
exercised by nothing. RED-TEAM-2 put two plausible edits on that arm — a `* 1000` unit slip
in the resolver and a `getattr(config, X, None)` lag-lambda fallback in run.py's own house
idiom — and got **116/116 guard tests passing** while a real-config drive froze the actor at
step 1 with the watchdog reading lag 0.

**Scope, stated honestly: this closes an AXIS, not the class.** The class is "a
production-only axis pinned to a single test-only value across the whole suite" — this is
its third instance (`eval_enabled=False`, then `config=SimpleNamespace()`). This file varies
the config-shape axis wholesale: a real `RunConfig` routes the real `.train`, `.monitor`,
`.eval` and `.identity` arms in one drive. It does NOT close the two axes that remain pinned
in every behavioral drive, because no test can: `_default_step_coordinator_config` is
monkeypatched (production's `stop_step=0` never enters the burst loop) and `build_run_safety`
is faked. Those are where RED-TEAM-3 should look. The class-level defenses are structural
(collapsing the smoke seams, owed to R-TRAINCONFIG-SCHEMA) and operational (an ARMED lag
watchdog plus a run5 smoke-boot check that counts `actor_sync` events).

NOT frozen: written after ORACLE-WRITE in response to a RED-TEAM finding.
"""
from __future__ import annotations

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
    """Reuse the frozen schema oracle's payload builder, by PATH (R5: `tests` is not a
    package and no `sys.path` mutation is permitted; reading a frozen file is not editing
    it, R43). Built from the payload rather than from `configs/run5.yaml` so this test
    pins the RESOLVER's behaviour and not one shipped config's current values."""
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
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
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

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


def _bounded_config(*, stop_step, draw_rate_abort) -> StepCoordinatorConfig:
    return StepCoordinatorConfig(
        terminal_eval_enabled=False,
        eval_interval=1000, log_interval=1, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=1, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        # WPAX Phase D: `draw_rate_abort` is a required parameter with no default
        # and is passed THROUGH; `stop_step` stays the harness's own bound, which
        # is this patch's stated reason for existing.
        instrumentation_enabled=False, stop_step=_STOP_STEP,
        draw_rate_abort=draw_rate_abort,
        final_eval_drain_timeout_sec=900.0,
    )


def _real_run_config() -> RunConfig:
    payload = _frozen_payload()(
        # WPAX S-4: the reachability bound now binds on the RUN LENGTH (6), not on the
        # LR horizon (1 000 000), so the threshold must fit inside `cadence < threshold
        # < max_train_steps` — 2 < 4 < 6.
        train_over={"actor_sync_cadence_steps": _CADENCE, "max_train_steps": _STOP_STEP},
        monitor_over={"actor_lag_threshold_steps": _CADENCE + 2},
    )
    return RunConfig(**payload)


def _drive(monkeypatch, *, eval_enabled: bool = True):
    """Compose with a REAL RunConfig; return (captured build_run_safety kwargs, pool, trainer)."""
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
    # WPAX S-4 narrowed this patch's reason: `stop_step` is config-authored now (compose_run
    # overrides it from train.max_train_steps regardless), so what `_bounded_config` still
    # buys is `terminal_eval_enabled=False` — this drive is eval_enabled=True and the
    # production builder's terminal eval round reaches eval/snapshot.py's `.arch` read on a
    # fake model. That knob has no config key (R-TRAINCONFIG-SCHEMA / ADJ-08).
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _bounded_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(best_model=None, best_model_step=None,
                                      best_model_path=None, representation="grid"),
    )
    return captured, pool, trainer, _real_run_config()


def test_a_real_run_config_actually_reaches_the_cadence_resolver(tmp_path, monkeypatch):
    """The premise. If the real arm is not taken, everything below is vacuous."""
    captured, pool, trainer, cfg = _drive(monkeypatch)
    assert mantis.run._resolve_actor_sync_cadence_steps(cfg) == _CADENCE, (
        "the real-config arm did not resolve the configured cadence — this test would "
        "otherwise pass while exercising the smoke path it exists to avoid"
    )


def test_sync_follows_the_configured_cadence_under_a_real_config(tmp_path, monkeypatch):
    """Kills the resolver unit-slip half of RED-TEAM-2's mutation on its own.

    A `* 1000` slip makes the cadence unreachable inside the run, so the actor takes its
    single unconditional first sync and then freezes — exactly run3.
    """
    captured, pool, trainer, cfg = _drive(monkeypatch)
    mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
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


def test_lag_callables_read_live_sources_under_a_real_config(tmp_path, monkeypatch):
    """Kills the lag-lambda half on its own — so each edit of the pair dies alone.

    RED-TEAM-2 paired the resolver slip with a `getattr(config, X, None)` fallback on the
    lag lambda. Asserting only sync volume would let that half survive; asserting only the
    lambdas would let the resolver slip survive.
    """
    captured, pool, trainer, cfg = _drive(monkeypatch)
    mantis.run.compose_run(
        config=cfg, trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"), eval_enabled=True,
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
