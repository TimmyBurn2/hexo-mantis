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

from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel


class DiskGuardConfig(StrictModel):
    """The `mantis.train.lifecycle.disk_guard.DiskGuard` thresholds, minted (WPMAIN, R122).

    R121(b) MANDATES that the composition root construct the disk guard (LAW-16's third leg);
    R1 FORBIDS the construction values being literals or `dict.get` defaults. The only
    disposition satisfying both is a schema block, and R122 grants it as a FAMILY: ONE block,
    ONE resolver (`mantis.config.resolve.disk_guard.resolve_disk_guard`), THREE typed leaves,
    minted at 60/10/5 — the very numbers that sat dead in `build_subsystems`' `.get(...)`
    defaults over a key no schema and no config ever carried.

    `Field(gt=0)` on all three: a non-positive interval is not a cadence and a non-positive
    threshold is not a threshold. `keep_all` gets NO key — it is an inert carried knob whose
    only consumer is the "thresholds ignore it" pin, so the root passes `False` explicitly and
    `extra="forbid"` refuses a minted one rather than silently ignoring it.

    Values are revisable at MINT PREREG (the R85 pattern): the literals were dead, so nothing
    has ever measured guard behaviour on a real box. A revision is a prereg row, never an
    implementation edit.
    """

    interval_sec: float = Field(gt=0)
    warn_gb: float = Field(gt=0)
    fail_gb: float = Field(gt=0)

    @model_validator(mode="after")
    def _fail_threshold_below_warn_threshold(self) -> DiskGuardConfig:
        # A `fail_gb` at or above `warn_gb` means the run SIGTERMs itself before it ever
        # warns — a guard that skips its own warning stage. Both field-level `gt=0` bounds
        # read perfectly legal on such a pair, and the EQUAL case especially reads normal in
        # a config diff. Inert at the minted 60/10/5 (5 < 10 holds), deliberately: the house
        # precedent for an inert-at-mint validator is
        # `RunConfig._policy_target_completed_q_consistency`, cited so this is not mistaken
        # for R116 dead weight.
        if self.fail_gb >= self.warn_gb:
            raise ValueError(
                f"monitor.disk_guard.fail_gb ({self.fail_gb}) must be < warn_gb "
                f"({self.warn_gb}): a critical threshold at or above the warning threshold "
                "makes the guard kill the run without ever having warned about it"
            )
        return self


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
    `MonitorConfig` dataclass; `gate_interval`, `drain` and `disk_guard` are schema-only (they
    feed `StepCoordinatorConfig`, `DrainCaps`/`StepCoordinatorConfig` and `DiskGuard`
    respectively, never `MonitorConfig`) — which is what legitimates `resolve_monitor_config`'s
    three enumerated drops."""

    # ── the ARMING cadence (R242 / ADJ-D12; schema-only, see resolve_monitor_config) ───
    # The stride, in TRAINING STEPS, at which `coordinator/step.py::_run_gate_interval` runs
    # the live hard-abort gates and publishes the LAW-18 `monitor_gates` summary. It is a
    # `monitor.*` key and NOT a `train.*` one because it decides when the run's SAFETY
    # machinery looks, not what the trainer does.
    #
    # WHY IT EXISTS. Until R242 both of those hung off `train.log_interval`, the NARRATION
    # cadence — so at run5's minted `log_interval: 1000` the draw-rate hard abort could take
    # its first sample no earlier than training step 1000 and no `monitor_gates` event existed
    # before then: armed machinery with a blind first kilometre, and the instrument that would
    # have shown it was switched off by the same knob. R242 splits the two; `log_interval`
    # reverts to narration-only.
    #
    # `ge=1` for the reason `train.log_interval` carries the same bound (WPMINT DR-7, restated
    # here for THIS knob): at `<= 0` the whole live hard-abort family stops being evaluated
    # (checks=0, fires=0, skips=0) AND the `monitor_gates` event that would make the deadness
    # readable stops emitting, together — while gate 12 goes on auditing the draw-rate row
    # ARMED. There is no legitimate "never gate" posture, so the schema cannot express one.
    #
    # NOTE for the mint record: every committed config mints this EQUAL to its own
    # `train.log_interval`, so the shipped arming cadence and every armed value are unchanged
    # in effect by R242's landing. The operator's real gate stride, and the `consec` re-derived
    # in gate-interval units, are mint-prereg rows — not values this schema chooses.
    gate_interval: int = Field(ge=1)

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

    # ── actor-lag invariant (WP-UNFREEZE K2/K3; watchdog family) ──────────────────────
    # `ge=1`, no zero-disable sentinel — disablement is the arming flag's job, one
    # authority. Cross-checked > train.actor_sync_cadence_steps at the RunConfig level.
    actor_lag_threshold_steps: int = Field(ge=1)
    actor_lag_abort_enabled: bool

    # ── out-of-process supervisor flag defaults (monitor/supervise.py) ────────────────
    supervisor_stale_after_sec: float = Field(ge=0)
    supervisor_poll_interval_sec: float = Field(ge=0)
    supervisor_kill_grace_sec: float = Field(ge=0)
    supervisor_max_relaunches: int = Field(ge=0)

    # ── drain/terminal-eval hard caps (DESIGN_P2.md §4.3; schema-only, see DrainCapsConfig) ──
    drain: DrainCapsConfig

    # ── disk guard (WPMAIN/R122; schema-only, see DiskGuardConfig) ────────────────────
    # Read by `mantis.config.resolve.disk_guard.resolve_disk_guard` and threaded into
    # `DiskGuard(...)` by `mantis.run.compose_run` — NOT part of the 1:1 `MonitorConfig`
    # copy, exactly like `drain`, and dropped by name in `resolve_monitor_config`.
    disk_guard: DiskGuardConfig
