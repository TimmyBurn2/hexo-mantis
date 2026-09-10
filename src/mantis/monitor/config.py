"""`MonitorConfig` — the monitor-side threshold authority.

Frozen, explicit kwargs only, with no lenient `from_dict` ignoring unknown keys. The `wr_*`
values are old-lineage calibrations carried verbatim, flagged for re-anchor; the criterion
STRUCTURE is what is law.

CONSTRUCTION AUTHORITY: inside `src/`, exactly ONE place may construct this — the monitor
resolver, a 1:1 copy off a VALIDATED schema — and an AST census enforces it. A bare construction
is not obviously wrong at the call site: it yields a complete, valid-looking object that has
substituted these literals for whatever the operator minted, armed in the config and absent in
effect. Tests may construct one directly, since there the thresholds are the subject; what no
test may do is depend on a production path defaulting, and that path no longer exists.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorConfig:
    """Frozen monitor thresholds; every field has a named live consumer."""

    # training-step WARN rules (monitor/rules.py)
    alert_entropy_min: float = 1.0
    collapse_threshold_nats: float = 1.5
    alert_grad_norm_max: float = 10.0
    alert_loss_increase_window: int = 3

    # sealbot-WR trajectory instrument (monitor/rules.py), values verbatim from the old
    # lineage. `wr_hard_abort_enabled=False` warns on a sustained collapse without stopping the
    # run; True restores the hard abort, with the same triggers either way.
    wr_hard_abort_enabled: bool = False
    wr_rolling_consecutive_evals: int = 2
    wr_rolling_threshold: float = 0.10
    wr_rolling_min_step: int = 20000
    wr_collapse_from_peak_ratio: float = 0.5
    wr_collapse_min_step: int = 25000
    wr_collapse_consecutive_evals: int = 3
    wr_early_death_threshold: float = 0.05
    wr_early_death_min_step: int = 15000

    # axis-distribution warn/alert (train/events.py) — one authority for the threshold.
    axis_warn: float = 0.45
    axis_alert: float = 0.50

    # heartbeat watchdog (train/lifecycle/heartbeat_watchdog.py): per-source staleness
    # deadlines, the calibrated 1800 s stall timeout per stage. `deadline <= 0` disables that
    # source's fire, and the arm-log still names it.
    heartbeat_deadline_train_step_sec: float = 1800.0
    heartbeat_deadline_inference_dispatch_sec: float = 1800.0
    heartbeat_deadline_selfplay_drain_sec: float = 1800.0
    # The eval pipeline's poller-thread source.
    heartbeat_deadline_eval_round_sec: float = 1800.0
    heartbeat_poll_interval_sec: float = 5.0
    heartbeat_file_interval_sec: float = 15.0
    # Teardown budget swapped in by `disarm_staleness()`: close-out waits are legally long,
    # but an unbounded teardown leaves both watchdog levels blind.
    heartbeat_close_out_deadline_sec: float = 14400.0
    # Hard budget for one optional effect in the fire path: `best_effort` catches exceptions,
    # not hangs, so a wedged filesystem would suspend the exit forever.
    heartbeat_fire_effect_timeout_sec: float = 30.0

    # actor-lag invariant (train/lifecycle/heartbeat_watchdog.py): `learner_step −
    # actor_ckpt_step > N` exits 45 when armed, else one loud event per episode. The mechanism
    # ships wired and the config arms it.
    actor_lag_threshold_steps: int = 100
    actor_lag_abort_enabled: bool = False

    # out-of-process supervisor flag defaults (monitor/supervise.py)
    supervisor_stale_after_sec: float = 900.0
    supervisor_poll_interval_sec: float = 30.0
    supervisor_kill_grace_sec: float = 30.0
    supervisor_max_relaunches: int = 5
