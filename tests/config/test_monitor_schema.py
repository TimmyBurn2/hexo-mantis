"""SC-A3 oracle — `MonitorSchemaConfig` (31 fields: 29 scalars + the `drain` and `disk_guard`
sub-blocks) + `DrainCapsConfig` (4 fields) + `DiskGuardConfig` (schema/liveness pinned in
tests/config/test_disk_guard_keys.py) +
`resolve_monitor_config` round-trip (DESIGN_P2.md §4 / PREREG_P2.md suite #6).

RED-at-import until IMPL lands `mantis.config.schema.monitor.MonitorSchemaConfig` /
`DrainCapsConfig` + `mantis.config.resolve.monitor.resolve_monitor_config`. Census pattern
as the other SC-A schema suites. The round-trip tests are the mutation self-test (LAW-07):
`resolve_monitor_config` must be a pure 1:1 field copy onto `mantis.monitor.config.
MonitorConfig` — a field-name-equality assertion between the two field sets, so renaming a
field on EITHER struct without the other breaks this suite.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.resolve import resolve_monitor_config
from mantis.config.schema import DrainCapsConfig, MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig

# Every value = the CURRENT `monitor.config.MonitorConfig` dataclass default, minted
# verbatim (DESIGN_P2.md §4.2 — zero behavior change; enumerated by direct read of
# monitor/config.py, the corrected field count — 27 at SC-A3, 29 since WP-UNFREEZE
# added actor_lag_threshold_steps / actor_lag_abort_enabled (R-30)).
VALID_MONITOR_SCALARS: dict = {
    "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
    "alert_loss_increase_window": 3, "wr_hard_abort_enabled": False,
    "wr_rolling_consecutive_evals": 2, "wr_rolling_threshold": 0.10,
    "wr_rolling_min_step": 20000, "wr_collapse_from_peak_ratio": 0.5,
    "wr_collapse_min_step": 25000, "wr_collapse_consecutive_evals": 3,
    "wr_early_death_threshold": 0.05, "wr_early_death_min_step": 15000,
    "axis_warn": 0.45, "axis_alert": 0.50,
    "heartbeat_deadline_train_step_sec": 1800.0,
    "heartbeat_deadline_inference_dispatch_sec": 1800.0,
    "heartbeat_deadline_selfplay_drain_sec": 1800.0,
    "heartbeat_deadline_eval_round_sec": 1800.0,
    "heartbeat_poll_interval_sec": 5.0, "heartbeat_file_interval_sec": 15.0,
    "heartbeat_close_out_deadline_sec": 14400.0, "heartbeat_fire_effect_timeout_sec": 30.0,
    "supervisor_stale_after_sec": 900.0, "supervisor_poll_interval_sec": 30.0,
    "supervisor_kill_grace_sec": 30.0, "supervisor_max_relaunches": 5,
    "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
}
VALID_DRAIN: dict = {
    "final_eval_drain_timeout_sec": 900.0, "eval_final_drain_safety_factor": 3.0,
    "eval_final_drain_hard_cap_sec": 14400.0, "terminal_eval_hard_cap_sec": 14400.0,
}
#: WPMAIN / R122: the minted `monitor.disk_guard` family. It is the SECOND schema-only
#: sub-block (after `drain`) — it feeds `mantis.train.lifecycle.disk_guard.DiskGuard`
#: through `resolve_disk_guard` and is NOT part of the 1:1 `MonitorConfig` copy.
VALID_DISK_GUARD: dict = {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0}
VALID_MONITOR: dict = dict(VALID_MONITOR_SCALARS, drain=dict(VALID_DRAIN),
                           disk_guard=dict(VALID_DISK_GUARD))

#: Re-derived from the population this file NAMES, never transcribed: `MONITOR_FIELDS` is
#: the payload's own key set, which is `MonitorSchemaConfig.model_fields` (29 scalars + the
#: two sub-blocks = 31). The runtime `MonitorConfig` dataclass stays at 29 — both sub-blocks
#: are popped by name in `resolve_monitor_config`, each because it has its OWN resolver.
MONITOR_FIELDS = sorted(VALID_MONITOR)
DRAIN_FIELDS = sorted(VALID_DRAIN)


def _monitor(**over: object) -> dict:
    out = dict(VALID_MONITOR)
    out.update(over)
    return out


def _drain(**over: object) -> dict:
    out = dict(VALID_DRAIN)
    out.update(over)
    return out


# ── MonitorSchemaConfig ───────────────────────────────────────────────────────────────
def test_monitor_valid_payload_constructs_clean():
    cfg = MonitorSchemaConfig.model_validate(VALID_MONITOR)
    assert cfg.alert_entropy_min == 1.0
    assert cfg.drain.final_eval_drain_timeout_sec == 900.0


@pytest.mark.parametrize("field", MONITOR_FIELDS)
def test_monitor_missing_field_rejected(field: str):
    payload = _monitor()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        MonitorSchemaConfig.model_validate(payload)


def test_monitor_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_monitor_knob"):
        MonitorSchemaConfig.model_validate(_monitor(bogus_monitor_knob=1))


def test_monitor_has_no_pydantic_level_default():
    for name, field in MonitorSchemaConfig.model_fields.items():
        assert field.is_required(), f"MonitorSchemaConfig.{name} has a code-side default"


def test_monitor_bound_examples_reject_negative_thresholds():
    # a representative (not exhaustive) sample of the >=0-domain fields.
    for field in ("alert_entropy_min", "alert_grad_norm_max", "wr_rolling_threshold",
                  "heartbeat_poll_interval_sec", "supervisor_max_relaunches"):
        with pytest.raises(ValidationError):
            MonitorSchemaConfig.model_validate(_monitor(**{field: -1}))


# ── DrainCapsConfig ───────────────────────────────────────────────────────────────────
def test_drain_caps_valid_payload_constructs_clean():
    cfg = DrainCapsConfig.model_validate(VALID_DRAIN)
    assert cfg.final_eval_drain_timeout_sec == 900.0


@pytest.mark.parametrize("field", DRAIN_FIELDS)
def test_drain_caps_missing_field_rejected(field: str):
    payload = _drain()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        DrainCapsConfig.model_validate(payload)


def test_drain_caps_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_drain_knob"):
        DrainCapsConfig.model_validate(_drain(bogus_drain_knob=1))


@pytest.mark.parametrize("field", DRAIN_FIELDS)
def test_drain_caps_zero_or_negative_rejected(field: str):
    # every DrainCapsConfig field is Field(gt=0) (a zero-or-negative drain/join bound is
    # domain-nonsense — a subprocess.join(0) is not a real bound).
    with pytest.raises(ValidationError):
        DrainCapsConfig.model_validate(_drain(**{field: 0.0}))


def test_drain_caps_has_no_pydantic_level_default():
    for name, field in DrainCapsConfig.model_fields.items():
        assert field.is_required(), f"DrainCapsConfig.{name} has a code-side default"


# ── resolve_monitor_config round-trip (LAW-07 mutation self-test) ─────────────────────
def test_monitor_schema_scalar_fields_equal_monitor_config_dataclass_fields():
    # The excluded set is ENUMERATED, one named block per member, and must stay that way:
    # widening it to "ignore anything the dataclass lacks" would let any future schema field
    # vanish from this equality silently — which is the same weaken-class move the sibling
    # census (`tests/config/test_disk_guard_keys.py`) forbids on `resolve_monitor_config`'s
    # pop. `disk_guard` joins `drain` here for the same reason `drain` was there: it is
    # schema-only and has its own resolver.
    schema_fields = set(MONITOR_FIELDS) - {"drain", "disk_guard"}
    dataclass_fields = set(MonitorConfig.__dataclass_fields__)
    assert schema_fields == dataclass_fields, (
        f"schema-only: {schema_fields - dataclass_fields}; "
        f"dataclass-only: {dataclass_fields - schema_fields}"
    )


def test_resolve_monitor_config_round_trips_every_field_unchanged():
    cfg = MonitorSchemaConfig.model_validate(VALID_MONITOR)
    resolved = resolve_monitor_config(cfg)
    assert isinstance(resolved, MonitorConfig)
    for field in VALID_MONITOR_SCALARS:
        assert getattr(resolved, field) == getattr(cfg, field), field
