"""`resolve_monitor_config` — `MonitorSchemaConfig` -> `mantis.monitor.config.MonitorConfig`
(DESIGN_P2.md §4.2). A pure 1:1 field copy: the pydantic field set is a strict copy of the
dataclass field set by construction, so a field added to one side and not the other is
caught by `tests/config/test_monitor_schema.py`'s field-name-equality mutation self-test
(LAW-07), not silently.

`gate_interval`, `drain` and `disk_guard` are NOT part of that copy and are dropped here —
`MonitorConfig` has none of the three. `gate_interval` (R242 / ADJ-D12) is read by
`mantis.run.compose_run`, which names `config.monitor.gate_interval` directly and threads it
into `StepCoordinatorConfig.gate_interval`, whence
`train/coordinator/step.py::_run_gate_interval` runs the live hard-abort gates and publishes
the LAW-18 `monitor_gates` summary on that stride. It is deliberately NOT added to
`MonitorConfig`: that dataclass carries a code-side default for each of its legacy fields, so
a 28th one here would be the second-authority shape R1 kills and would contradict R242's "no
default". The other two are read by their OWN resolver:
`mantis.config.resolve.drain.resolve_drain_caps`, which `compose_run` threads into
`StepCoordinatorConfig` and thence into `mantis.eval.pipeline.DrainCaps`; and
`mantis.config.resolve.disk_guard.resolve_disk_guard`, which `compose_run` threads into
`mantis.train.lifecycle.disk_guard.DiskGuard`. Until WPMINT Phase K-A (R93) the drain
resolver did not exist and its `pop` was the END of the block's journey: four minted,
schema-validated, registry-claimed keys read by nothing, while `run.py` used a hardcoded
`900.0` and three dataclass defaults instead (the DR-11 finding). Each drop is only
legitimate BECAUSE another reader exists — if either resolver is ever deleted, its line
becomes that defect again.

All three drops are written ENUMERATED, one named key per line, and must stay that way. A
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
    data.pop("gate_interval")  # -> mantis.run.compose_run -> StepCoordinatorConfig.gate_interval
    data.pop("drain")        # -> mantis.config.resolve.drain.resolve_drain_caps
    data.pop("disk_guard")   # -> mantis.config.resolve.disk_guard.resolve_disk_guard
    return MonitorConfig(**data)
