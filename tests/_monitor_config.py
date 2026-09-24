"""The ONE test factory for a `MonitorConfig`: schema defaults through the real resolver, test values where the schema requires one."""
from __future__ import annotations

import dataclasses
from typing import Any

from mantis.config.resolve import resolve_monitor_config
from mantis.config.schema import MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig

#: Scalar `monitor.*` values shared with tests/config/test_monitor_schema.py's 1:1 copy
#: assertion, so the two files cannot drift apart.
SHARED_MONITOR_SCALARS: dict[str, Any] = {
    "alert_entropy_min": 1.0, "collapse_threshold_nats": 1.5, "alert_grad_norm_max": 10.0,
    "alert_loss_increase_window": 3, "axis_warn": 0.45, "axis_alert": 0.50,
    "actor_lag_threshold_steps": 100, "actor_lag_abort_enabled": False,
    "supervisor_kill_grace_sec": 30.0,
}

#: Test values for the schema-REQUIRED `monitor.*` fields; every other field takes its schema default.
REQUIRED_TEST_VALUES: dict[str, Any] = dict(SHARED_MONITOR_SCALARS, gate_interval=1)


def monitor_config(**over: Any) -> MonitorConfig:
    """A resolved `MonitorConfig` with `over` replacing fields unvalidated, as a direct construction would."""
    base = resolve_monitor_config(MonitorSchemaConfig.model_validate(REQUIRED_TEST_VALUES))
    return dataclasses.replace(base, **over)
