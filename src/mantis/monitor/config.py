"""`MonitorConfig` — the monitor-side thresholds, with NO defaults: the schema field is the one default authority.

Inside `src/` only the monitor resolver constructs one, a 1:1 copy off a validated schema (an AST
census enforces it); tests build theirs through the same resolver (tests/_monitor_config.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class MonitorConfig:
    """Frozen monitor thresholds; every field has a named live consumer."""

    # training-step WARN rules (monitor/rules.py)
    alert_entropy_min: float
    collapse_threshold_nats: float
    alert_grad_norm_max: float
    alert_loss_increase_window: int

    # axis-distribution warn/alert (train/events.py) — one authority for the threshold.
    axis_warn: float
    axis_alert: float

    # heartbeat watchdog (train/lifecycle/heartbeat_watchdog.py): per-source staleness
    # deadlines. `deadline <= 0` disables that source's fire, and the arm-log still names it.
    heartbeat_deadline_train_step_sec: float
    heartbeat_deadline_inference_dispatch_sec: float
    heartbeat_deadline_selfplay_drain_sec: float
    # The eval pipeline's poller-thread source.
    heartbeat_deadline_eval_round_sec: float
    heartbeat_poll_interval_sec: float
    heartbeat_file_interval_sec: float
    # Teardown budget swapped in by `disarm_staleness()`: close-out waits are legally long,
    # but an unbounded teardown leaves both watchdog levels blind.
    heartbeat_close_out_deadline_sec: float
    # Hard budget for one optional effect in the fire path: `best_effort` catches exceptions,
    # not hangs, so a wedged filesystem would suspend the exit forever.
    heartbeat_fire_effect_timeout_sec: float

    # actor-lag invariant (train/lifecycle/heartbeat_watchdog.py): `learner_step −
    # actor_ckpt_step > N` exits 45 when armed, else one loud event per episode. The mechanism
    # ships wired and the config arms it.
    actor_lag_threshold_steps: int
    actor_lag_abort_enabled: bool

    # out-of-process supervisor flag defaults (monitor/supervise.py)
    supervisor_stale_after_sec: float
    supervisor_poll_interval_sec: float
    supervisor_kill_grace_sec: float
    supervisor_max_relaunches: int
