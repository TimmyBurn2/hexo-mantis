"""`MonitorSchemaConfig` + `DrainCapsConfig` — R-MONITORCONFIG-SCHEMA closure (SC-A3,
DESIGN_P2.md §4). Every field of `mantis.monitor.config.MonitorConfig` (27 fields) gets a
same-named, same-typed schema field minted at its current dataclass default (zero behavior
change — nothing in `configs/` set any of these before this chunk). `DrainCapsConfig`
carries the 4 `StepCoordinatorConfig` drain/terminal-eval-hard-cap fields (DESIGN_P2.md
§4.3): a direct re-read of `eval/pipeline.py`/`run.py` confirms all four are live-consumed
subprocess-join bounds, not just the 2 `TARGET_RECON_REPORT.md` names — nested under
`monitor.drain` since it gates the same close-out/teardown machinery as the heartbeat
fields above it, not `train.step_coordinator` (out of scope, R-10).
"""
from __future__ import annotations

from pydantic import Field

from mantis.config.schema._base import StrictModel


class DrainCapsConfig(StrictModel):
    """The 4 live-consumed drain/terminal-eval-hard-cap fields (DESIGN_P2.md §4.3):
    `run.py`'s `DrainCaps` (all 4) + `eval/pipeline.py`'s `drain_budget_sec`/
    `_run_terminal_sync`. A zero-or-negative bound is domain-nonsense — a
    `subprocess.join(0)` is not a real bound — so every field is `Field(gt=0)`.
    """

    final_eval_drain_timeout_sec: float = Field(gt=0)
    eval_final_drain_safety_factor: float = Field(gt=0)
    eval_final_drain_hard_cap_sec: float = Field(gt=0)
    terminal_eval_hard_cap_sec: float = Field(gt=0)


class MonitorSchemaConfig(StrictModel):
    """Every `mantis.monitor.config.MonitorConfig` field (27), same name/type, minted at its
    current dataclass default (DESIGN_P2.md §4.2). `resolve_monitor_config`
    (`mantis.config.resolve.monitor`) is the pure 1:1 field-copy consumer onto the runtime
    `MonitorConfig` dataclass; `drain` is schema-only (feeds `DrainCaps`/
    `StepCoordinatorConfig`, not `MonitorConfig`)."""

    # ── the 4 training-step WARN rules (monitor/rules.py) ──────────────────────────────
    alert_entropy_min: float = Field(ge=0)
    collapse_threshold_nats: float = Field(ge=0)
    alert_grad_norm_max: float = Field(ge=0)
    alert_loss_increase_window: int = Field(ge=0)

    # ── sealbot-WR trajectory instrument (monitor/rules.py) ────────────────────────────
    wr_hard_abort_enabled: bool
    wr_rolling_consecutive_evals: int = Field(ge=0)
    wr_rolling_threshold: float = Field(ge=0)
    wr_rolling_min_step: int = Field(ge=0)
    wr_collapse_from_peak_ratio: float = Field(ge=0)
    wr_collapse_min_step: int = Field(ge=0)
    wr_collapse_consecutive_evals: int = Field(ge=0)
    wr_early_death_threshold: float = Field(ge=0)
    wr_early_death_min_step: int = Field(ge=0)

    # ── axis-distribution warn/alert (train/events.py::emit_axis_distribution) ─────────
    axis_warn: float = Field(ge=0)
    axis_alert: float = Field(ge=0)

    # ── independent heartbeat watchdog (train/lifecycle/heartbeat_watchdog.py) ─────────
    # `deadline <= 0` disables that source's fire (monitor/config.py's own contract) — a
    # bound of `ge=0`, not `gt=0`, preserves the ability to mint the disabled sentinel.
    heartbeat_deadline_train_step_sec: float = Field(ge=0)
    heartbeat_deadline_inference_dispatch_sec: float = Field(ge=0)
    heartbeat_deadline_selfplay_drain_sec: float = Field(ge=0)
    heartbeat_deadline_eval_round_sec: float = Field(ge=0)
    heartbeat_poll_interval_sec: float = Field(ge=0)
    heartbeat_file_interval_sec: float = Field(ge=0)
    heartbeat_close_out_deadline_sec: float = Field(ge=0)
    heartbeat_fire_effect_timeout_sec: float = Field(ge=0)

    # ── out-of-process supervisor flag defaults (monitor/supervise.py) ────────────────
    supervisor_stale_after_sec: float = Field(ge=0)
    supervisor_poll_interval_sec: float = Field(ge=0)
    supervisor_kill_grace_sec: float = Field(ge=0)
    supervisor_max_relaunches: int = Field(ge=0)

    # ── drain/terminal-eval hard caps (DESIGN_P2.md §4.3; schema-only, see DrainCapsConfig) ──
    drain: DrainCapsConfig
