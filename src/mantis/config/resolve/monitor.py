"""`resolve_monitor_config` — `MonitorSchemaConfig` -> `mantis.monitor.config.MonitorConfig`
(DESIGN_P2.md §4.2). A pure 1:1 field copy: the pydantic field set is a strict copy of the
dataclass field set by construction (`drain` is schema-only, dropped before construction —
it feeds `DrainCaps`/`StepCoordinatorConfig`, not `MonitorConfig`), so a field added to one
side and not the other is caught by `tests/config/test_monitor_schema.py`'s field-name-
equality mutation self-test (LAW-07), not silently.
"""
from __future__ import annotations

from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig


def resolve_monitor_config(cfg: MonitorSchemaConfig) -> MonitorConfig:
    """Build the runtime `MonitorConfig` from a validated `MonitorSchemaConfig` section."""
    data = cfg.model_dump()
    data.pop("drain")
    return MonitorConfig(**data)
