"""A same-named, same-typed schema field for every `mantis.monitor.config.MonitorConfig` field,
minted at its current dataclass default, plus the four live-consumed drain / terminal-eval hard
caps — nested under `monitor.drain` because they gate the same close-out machinery as the
heartbeat fields rather than the trainer's step loop.
"""
from __future__ import annotations

from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel

#: The OPERATIONAL CONSTANTS here carry schema defaults and leave the YAML: a key defaults when
#: it names how the PROCESS is operated and no run has ever decided it differently. It stays
#: REQUIRED where a run really chooses it — `gate_interval`, the actor-lag pair,
#: `supervisor_kill_grace_sec`, the WR/axis/warn family — because those are ARMING.


class DiskGuardConfig(StrictModel):
    """The `DiskGuard` thresholds: ONE block, ONE resolver, THREE typed leaves, minted at
    60/10/5.

    Those numbers sat dead in `.get(...)` defaults over a key no config ever carried, so guard
    behaviour has never been measured on a box and revising them is a mint-prereg row. `keep_all`
    gets NO key — an inert carried knob the root passes explicitly.
    """

    interval_sec: float = Field(default=60.0, gt=0)
    warn_gb: float = Field(default=10.0, gt=0)
    fail_gb: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _fail_threshold_below_warn_threshold(self) -> DiskGuardConfig:
        # A `fail_gb` at or above `warn_gb` SIGTERMs the run before it ever warns, and both
        # `gt=0` bounds read legal on such a pair. Inert at the minted 60/10/5, deliberately.
        if self.fail_gb >= self.warn_gb:
            raise ValueError(
                f"monitor.disk_guard.fail_gb ({self.fail_gb}) must be < warn_gb "
                f"({self.warn_gb}): a critical threshold at or above the warning threshold "
                "makes the guard kill the run without ever having warned about it"
            )
        return self


class DrainCapsConfig(StrictModel):
    """The four live-consumed drain / terminal-eval-hard-cap fields; every field is `gt=0`
    because a `subprocess.join(0)` is not a real bound."""

    final_eval_drain_timeout_sec: float = Field(default=900.0, gt=0)
    eval_final_drain_safety_factor: float = Field(default=3.0, gt=0)
    eval_final_drain_hard_cap_sec: float = Field(default=14400.0, gt=0)
    terminal_eval_hard_cap_sec: float = Field(default=14400.0, gt=0)


class MonitorSchemaConfig(StrictModel):
    """Every `MonitorConfig` field, same name and type, minted at its dataclass default;
    `gate_interval`, `drain` and `disk_guard` are schema-only, which is what legitimates
    `resolve_monitor_config`'s three enumerated drops."""

    # The stride, in TRAINING STEPS, at which the live hard-abort gates run and `monitor_gates`
    # is published — a `monitor.*` key because it times the SAFETY machinery, not the trainer.
    # `ge=1`: at `<= 0` the hard-abort family stops being evaluated AND the event that would
    # make the deadness readable stops emitting, while gate 12 still audits the row ARMED.
    gate_interval: int = Field(ge=1)

    # the 4 training-step WARN rules (monitor/rules.py)
    alert_entropy_min: float = Field(ge=0)
    collapse_threshold_nats: float = Field(ge=0)
    alert_grad_norm_max: float = Field(ge=0)
    alert_loss_increase_window: int = Field(ge=0)

    # sealbot-WR trajectory instrument (monitor/rules.py)
    wr_hard_abort_enabled: bool
    wr_rolling_consecutive_evals: int = Field(ge=0)
    wr_rolling_threshold: float = Field(ge=0)
    wr_rolling_min_step: int = Field(ge=0)
    wr_collapse_from_peak_ratio: float = Field(ge=0)
    wr_collapse_min_step: int = Field(ge=0)
    wr_collapse_consecutive_evals: int = Field(ge=0)
    wr_early_death_threshold: float = Field(ge=0)
    wr_early_death_min_step: int = Field(ge=0)

    # axis-distribution warn/alert (train/events.py::emit_axis_distribution)
    axis_warn: float = Field(ge=0)
    axis_alert: float = Field(ge=0)

    # independent heartbeat watchdog. `deadline <= 0` disables that source's fire, so the bound
    # is `ge=0`, not `gt=0`, to keep the disabled sentinel mintable.
    heartbeat_deadline_train_step_sec: float = Field(default=1800.0, ge=0)
    heartbeat_deadline_inference_dispatch_sec: float = Field(default=1800.0, ge=0)
    heartbeat_deadline_selfplay_drain_sec: float = Field(default=1800.0, ge=0)
    heartbeat_deadline_eval_round_sec: float = Field(default=1800.0, ge=0)
    heartbeat_poll_interval_sec: float = Field(default=5.0, ge=0)
    heartbeat_file_interval_sec: float = Field(default=15.0, ge=0)
    heartbeat_close_out_deadline_sec: float = Field(default=14400.0, ge=0)
    heartbeat_fire_effect_timeout_sec: float = Field(default=30.0, ge=0)

    # actor-lag invariant. `ge=1`, no zero-disable sentinel — disablement is the arming flag's
    # job. Cross-checked > train.actor_sync_cadence_steps at the RunConfig level.
    actor_lag_threshold_steps: int = Field(ge=1)
    actor_lag_abort_enabled: bool

    # out-of-process supervisor flag defaults (monitor/supervise.py)
    supervisor_stale_after_sec: float = Field(default=900.0, ge=0)
    supervisor_poll_interval_sec: float = Field(default=30.0, ge=0)
    # NO default: the kill grace is the one supervisor knob a run decides — run6 mints 600 s
    # against the template's 30 s because its close-out really takes that long.
    supervisor_kill_grace_sec: float = Field(ge=0)
    supervisor_max_relaunches: int = Field(default=5, ge=0)

    # drain/terminal-eval hard caps; schema-only, see DrainCapsConfig.
    drain: DrainCapsConfig = DrainCapsConfig()

    # disk guard: read by `resolve_disk_guard` and threaded into `DiskGuard(...)` by
    # `compose_run`, not part of the 1:1 `MonitorConfig` copy — dropped by name like `drain`.
    disk_guard: DiskGuardConfig = DiskGuardConfig()
