"""`MonitorSchemaConfig` + `DrainCapsConfig` + the `resolve_monitor_config` round-trip.

The round-trip tests are the mutation self-test: `resolve_monitor_config` must be a pure 1:1
field copy onto `mantis.monitor.config.MonitorConfig`, asserted as field-name equality between
the two field sets, so renaming a field on EITHER struct alone breaks this suite."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.resolve import resolve_monitor_config
from mantis.config.schema import (
    DrainCapsConfig,
    MonitorSchemaConfig,
    operational_default_fields,
)
from mantis.monitor.config import MonitorConfig

# Every value is the CURRENT `MonitorConfig` dataclass default, minted verbatim for zero
# behaviour change.
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
#: The minted `monitor.disk_guard` family: the SECOND schema-only sub-block after `drain`, fed
#: to `DiskGuard` through `resolve_disk_guard` and NOT part of the 1:1 `MonitorConfig` copy.
VALID_DISK_GUARD: dict = {"interval_sec": 60.0, "warn_gb": 10.0, "fail_gb": 5.0}
#: `monitor.gate_interval`, the ARMING cadence: a SCALAR that is nonetheless NOT in
#: `VALID_MONITOR_SCALARS`, which everywhere here means "scalar of the 1:1 `MonitorConfig` copy".
#: Schema-only — its reader is `compose_run` -> `StepCoordinatorConfig.gate_interval`.
VALID_GATE_INTERVAL: int = 1000
VALID_MONITOR: dict = dict(VALID_MONITOR_SCALARS, gate_interval=VALID_GATE_INTERVAL,
                           drain=dict(VALID_DRAIN), disk_guard=dict(VALID_DISK_GUARD))

#: Re-derived from the population this file NAMES, never transcribed; the runtime dataclass is
#: smaller because the schema-only members are popped BY NAME, each with its own reader.
MONITOR_FIELDS = sorted(VALID_MONITOR)
DRAIN_FIELDS = sorted(VALID_DRAIN)


def _monitor(**over: object) -> dict:
    out = dict(VALID_MONITOR)
    out.update(over)
    return out


_MONITOR_DEFAULTED = operational_default_fields("monitor")


def _drain(**over: object) -> dict:
    out = dict(VALID_DRAIN)
    out.update(over)
    return out


def test_monitor_valid_payload_constructs_clean():
    cfg = MonitorSchemaConfig.model_validate(VALID_MONITOR)
    assert cfg.alert_entropy_min == 1.0
    assert cfg.drain.final_eval_drain_timeout_sec == 900.0


@pytest.mark.parametrize("field", sorted(set(MONITOR_FIELDS) - _MONITOR_DEFAULTED))
def test_monitor_missing_field_rejected(field: str):
    payload = _monitor()
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        MonitorSchemaConfig.model_validate(payload)


@pytest.mark.parametrize("field", sorted(_MONITOR_DEFAULTED))
def test_an_operational_field_is_OMITTABLE_and_falls_to_its_declared_default(field: str):
    """Omitting an operational constant must be LEGAL, and must land on the value the registry
    says: asserting only "no error" would be satisfied by a default of anything at all."""
    payload = _monitor()
    del payload[field]
    cfg = MonitorSchemaConfig.model_validate(payload)
    assert getattr(cfg, field) == MonitorSchemaConfig.model_fields[field].get_default(
        call_default_factory=True), (
        f"monitor.{field} is omittable but did not land on its schema default"
    )


def test_monitor_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_monitor_knob"):
        MonitorSchemaConfig.model_validate(_monitor(bogus_monitor_knob=1))


def test_monitor_has_no_pydantic_level_default_EXCEPT_the_declared_operational_ones():
    """A default is legal ONLY where the schema's registry declares the key operational, checked
    BOTH ways: an undeclared default reds, and a declared-but-required key is a stale exemption."""
    for name, field in MonitorSchemaConfig.model_fields.items():
        if name in _MONITOR_DEFAULTED:
            assert not field.is_required(), (
                f"MonitorSchemaConfig.{name} is declared operational in "
                "OPERATIONAL_DEFAULT_KEYS but is still required — the declaration is stale"
            )
            continue
        assert field.is_required(), (
            f"MonitorSchemaConfig.{name} has a code-side default and is not declared in "
            "OPERATIONAL_DEFAULT_KEYS; R1 puts a default in the schema field or nowhere, and "
            "the registry is what says which keys earned one"
        )


def test_monitor_gate_interval_is_required_and_at_least_one():
    """The ARMING cadence has NO code-side default and no off value: a non-positive stride stops
    the live hard-abort family AND the summary that would make the deadness readable, together,
    while the audit still calls the row ARMED."""
    assert MonitorSchemaConfig.model_fields["gate_interval"].is_required(), (
        "monitor.gate_interval carries a code-side default — R1/R242: the config is then not "
        "its only authority and a caller inherits an ARMING posture"
    )
    for bad in (0, -1):
        with pytest.raises(ValidationError, match="gate_interval"):
            MonitorSchemaConfig.model_validate(_monitor(gate_interval=bad))
    assert MonitorSchemaConfig.model_validate(_monitor(gate_interval=1)).gate_interval == 1


def test_monitor_bound_examples_reject_negative_thresholds():
    # a representative (not exhaustive) sample of the >=0-domain fields.
    for field in ("alert_entropy_min", "alert_grad_norm_max", "wr_rolling_threshold",
                  "heartbeat_poll_interval_sec", "supervisor_max_relaunches"):
        with pytest.raises(ValidationError):
            MonitorSchemaConfig.model_validate(_monitor(**{field: -1}))


def test_drain_caps_valid_payload_constructs_clean():
    cfg = DrainCapsConfig.model_validate(VALID_DRAIN)
    assert cfg.final_eval_drain_timeout_sec == 900.0


@pytest.mark.parametrize("field", DRAIN_FIELDS)
def test_an_omitted_drain_cap_lands_on_its_declared_default(field: str):
    """Omitting a declared-operational cap is legal, and the VALUE is asserted because "no
    error" alone would be satisfied by any default."""
    payload = _drain()
    del payload[field]
    cfg = DrainCapsConfig.model_validate(payload)
    assert getattr(cfg, field) == DrainCapsConfig.model_fields[field].get_default(
        call_default_factory=True), f"monitor.drain.{field} did not land on its schema default"


def test_drain_caps_extra_key_rejected():
    with pytest.raises(ValidationError, match="bogus_drain_knob"):
        DrainCapsConfig.model_validate(_drain(bogus_drain_knob=1))


@pytest.mark.parametrize("field", DRAIN_FIELDS)
def test_drain_caps_zero_or_negative_rejected(field: str):
    # Every field is `Field(gt=0)`: a `subprocess.join(0)` is not a real bound.
    with pytest.raises(ValidationError):
        DrainCapsConfig.model_validate(_drain(**{field: 0.0}))


def test_every_drain_cap_is_a_declared_operational_default():
    """All four are subprocess-join bounds, so all four are declared — and the EQUALITY is
    asserted, because a fifth cap without a registry row would pass a one-way check."""
    assert set(DrainCapsConfig.model_fields) == operational_default_fields("monitor.drain")
    for name, field in DrainCapsConfig.model_fields.items():
        assert not field.is_required(), (
            f"DrainCapsConfig.{name} is declared operational but is still required"
        )


def test_monitor_schema_scalar_fields_equal_monitor_config_dataclass_fields():
    # The excluded set is ENUMERATED, one named member at a time: widening it to "ignore
    # anything the dataclass lacks" would let a future schema field vanish silently. Each
    # excluded name is schema-only with its own reader elsewhere.
    schema_fields = set(MONITOR_FIELDS) - {"gate_interval", "drain", "disk_guard"}
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
