"""`resolve_monitor_config` — `MonitorSchemaConfig` -> `mantis.monitor.config.MonitorConfig`
(DESIGN_P2.md §4.2). A pure 1:1 field copy: the pydantic field set is a strict copy of the
dataclass field set by construction, so a field added to one side and not the other is
caught by `tests/config/test_monitor_schema.py`'s field-name-equality mutation self-test
(LAW-07), not silently.

`drain` and `disk_guard` are NOT part of that copy and are dropped here — `MonitorConfig`
has neither field. Each is read by its OWN resolver:
`mantis.config.resolve.drain.resolve_drain_caps`, which `compose_run` threads into
`StepCoordinatorConfig` and thence into `mantis.eval.pipeline.DrainCaps`; and
`mantis.config.resolve.disk_guard.resolve_disk_guard`, which `compose_run` threads into
`mantis.train.lifecycle.disk_guard.DiskGuard`. Until WPMINT Phase K-A (R93) the drain
resolver did not exist and its `pop` was the END of the block's journey: four minted,
schema-validated, registry-claimed keys read by nothing, while `run.py` used a hardcoded
`900.0` and three dataclass defaults instead (the DR-11 finding). Each drop is only
legitimate BECAUSE another reader exists — if either resolver is ever deleted, its line
becomes that defect again.

Both drops are written ENUMERATED, one named block per line, and must stay that way. A
comprehension over the dataclass's own field names would fix today's break and then silently
swallow every future unmatched key — which is the DR-11 defect itself, restated as a
mechanism instead of an accident (WPMAIN F-10; pinned by
`tests/config/test_disk_guard_keys.py::test_the_monitor_resolver_drops_disk_guard_by_name_never_by_a_filter`).
"""
from __future__ import annotations

from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.monitor.config import MonitorConfig


def resolve_monitor_config(cfg: MonitorSchemaConfig) -> MonitorConfig:
    """Build the runtime `MonitorConfig` from a validated `MonitorSchemaConfig` section."""
    data = cfg.model_dump()
    data.pop("drain")        # -> mantis.config.resolve.drain.resolve_drain_caps
    data.pop("disk_guard")   # -> mantis.config.resolve.disk_guard.resolve_disk_guard
    return MonitorConfig(**data)
