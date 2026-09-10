"""Resolve `MonitorSchemaConfig` into the runtime `MonitorConfig` by a 1:1 field copy.

`gate_interval`, `drain` and `disk_guard` are not part of that copy and are dropped here,
because `MonitorConfig` holds none of the three; each is read instead by the consumer named on
its own line below. A drop is only legitimate BECAUSE that other reader exists — delete one and
the key becomes a minted, schema-validated key nothing reads.

The three drops stay ENUMERATED, one named key per line: a comprehension over the dataclass's
own field names would silently swallow every future unmatched key.
"""
from __future__ import annotations

from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig


def resolve_monitor_config(cfg: MonitorSchemaConfig) -> MonitorConfig:
    """Build the runtime `MonitorConfig` from a validated `MonitorSchemaConfig` section."""
    data = cfg.model_dump()
    data.pop("gate_interval")  # -> mantis.run.compose_run -> StepCoordinatorConfig.gate_interval
    data.pop("drain")        # -> mantis.config.resolve.drain.resolve_drain_caps
    data.pop("disk_guard")   # -> mantis.config.resolve.disk_guard.resolve_disk_guard
    return MonitorConfig(**data)
