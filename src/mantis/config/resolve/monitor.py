"""`resolve_monitor_config` — `MonitorSchemaConfig` -> `mantis.monitor.config.MonitorConfig`
(DESIGN_P2.md §4.2). A pure 1:1 field copy: the pydantic field set is a strict copy of the
dataclass field set by construction, so a field added to one side and not the other is
caught by `tests/config/test_monitor_schema.py`'s field-name-equality mutation self-test
(LAW-07), not silently.

`drain` is NOT part of that copy and is dropped here — `MonitorConfig` has no such field.
It is read by its OWN resolver, `mantis.config.resolve.drain.resolve_drain_caps`, which
`compose_run` threads into `StepCoordinatorConfig` and thence into
`mantis.eval.pipeline.DrainCaps`. Until WPMINT Phase K-A (R93) that second resolver did not
exist and this `pop` was the END of the block's journey: four minted, schema-validated,
registry-claimed keys read by nothing, while `run.py` used a hardcoded `900.0` and three
dataclass defaults instead (the DR-11 finding). The drop is only legitimate BECAUSE another
reader exists — if `resolve_drain_caps` is ever deleted, this line becomes that defect again.
"""
from __future__ import annotations

from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig


def resolve_monitor_config(cfg: MonitorSchemaConfig) -> MonitorConfig:
    """Build the runtime `MonitorConfig` from a validated `MonitorSchemaConfig` section."""
    data = cfg.model_dump()
    data.pop("drain")
    return MonitorConfig(**data)
