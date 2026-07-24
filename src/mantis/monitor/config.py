"""`MonitorConfig` — the monitor-side threshold authority (WP13-A §c.6, decision D1).

Ports `hexo_rl/monitoring/config.py::MonitoringConfig` PRUNED to fields with a live
consumer, and drops its lenient `from_dict` (which silently ignored unknown keys — an R1
violation in spirit): construction is explicit kwargs only, the dataclass is frozen.

Dropped with their consumers: `alert_entropy_warn`, every `web_*`/`socketio_*`/`viewer_*`/
`ema_*`/`p0_win_rate_target_*` display knob, the history-retention knobs, ALL
`strength_*`/`robustness_*` (the KILLED phantom `strength_aggregate` chain), and
`sealbot_wr_revert_to_abort` (meaningless once sealbot-WR is the primary abort again).

D1 debt R-MONITORCONFIG-SCHEMA: these are code-side defaults, NOT minted `configs/` keys —
owed with R-TRAINCONFIG-SCHEMA before the run5 mint (one schema extension retires both).
The draw-rate threshold deliberately stays in `StepCoordinatorConfig` (WP10): one authority
per knob, no duplication. (The stride5-spam gate was REMOVED at close-out per operator
directive B; its WP10 `stride5_p90_*` knobs are now consumer-less again — debt
R-STRIDE5-ORPHAN-KNOBS.)

R1b honesty: the `wr_*` values are old-lineage (dense) calibrations carried verbatim and
are flagged for re-anchor at the run5 mint; the criterion STRUCTURE is the law.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorConfig:
    """Frozen monitor thresholds. Every field has a named live consumer (O-25)."""

    # ── the 4 training-step WARN rules (monitor/rules.py) ─────────────────────────────
    alert_entropy_min: float = 1.0
    collapse_threshold_nats: float = 1.5
    alert_grad_norm_max: float = 10.0
    alert_loss_increase_window: int = 3

    # ── sealbot-WR trajectory instrument (monitor/rules.py::sealbot_wr_trajectory_alert
    #    + check_sealbot_wr_hard_abort disposition wrapper) ─────────────────────────────
    # Values verbatim from the old `monitoring/config.py`. `wr_hard_abort_enabled` ships
    # FALSE = WARN-ONLY (operator G-3: warn chosen over hard-abort). With the flag False the
    # coordinator emits a visible `sealbot_wr_warn` on a sustained collapse and does NOT stop
    # the run; setting it True restores the A/B/C hard-abort exactly. The trajectory triggers
    # are unchanged — only the DEFAULT disposition moved.
    wr_hard_abort_enabled: bool = False
    wr_rolling_consecutive_evals: int = 2
    wr_rolling_threshold: float = 0.10
    wr_rolling_min_step: int = 20000
    wr_collapse_from_peak_ratio: float = 0.5
    wr_collapse_min_step: int = 25000
    wr_collapse_consecutive_evals: int = 3
    wr_early_death_threshold: float = 0.05
    wr_early_death_min_step: int = 15000

    # ── axis-distribution warn/alert (train/events.py::emit_axis_distribution) ────────
    # Folded in from the inline `config.get("monitors", {}).get(...)` code-side defaults,
    # which die: one authority for the threshold (SHOULD-D).
    axis_warn: float = 0.45
    axis_alert: float = 0.50

    # ── independent heartbeat watchdog (train/lifecycle/heartbeat_watchdog.py) ────────
    # Per-source staleness deadlines: the calibrated 1800 s stall timeout applied per
    # pipeline stage. `deadline <= 0` disables that source's fire (the arm-log still
    # names it — the WP10 visibility law).
    heartbeat_deadline_train_step_sec: float = 1800.0
    heartbeat_deadline_inference_dispatch_sec: float = 1800.0
    heartbeat_deadline_selfplay_drain_sec: float = 1800.0
    # WP11-A: the eval pipeline's poller-thread source (R-MONITORCONFIG-SCHEMA debt
    # pattern — a code-side default until the run5 mint, documented alongside its siblings).
    heartbeat_deadline_eval_round_sec: float = 1800.0
    heartbeat_poll_interval_sec: float = 5.0
    heartbeat_file_interval_sec: float = 15.0
    # Teardown budget swapped in by `disarm_staleness()` (RED-TEAM F2): close-out waits are
    # legally long, but an UNBOUNDED teardown left both watchdog levels blind. Matches the
    # drain/terminal-eval hard caps the close-out path is supposed to enforce.
    heartbeat_close_out_deadline_sec: float = 14400.0
    # Hard budget for ONE optional effect in the fire path (RED-TEAM F5): `best_effort`
    # catches exceptions, not hangs, so a wedged FS could suspend the exit forever.
    heartbeat_fire_effect_timeout_sec: float = 30.0

    # ── out-of-process supervisor flag defaults (monitor/supervise.py) ───────────────
    supervisor_stale_after_sec: float = 900.0
    supervisor_poll_interval_sec: float = 30.0
    supervisor_kill_grace_sec: float = 30.0
    supervisor_max_relaunches: int = 5
