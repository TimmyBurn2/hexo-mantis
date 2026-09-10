"""The ONE read path for the disk guard's thresholds.

`monitor.disk_guard.*` is read HERE and nowhere else; the composition root threads the resolved
spec into `DiskGuard`, which WARNs below `warn_gb` and SIGTERMs the run below `fail_gb`. This
resolver is also what legitimates `resolve_monitor_config`'s enumerated `data.pop("disk_guard")`:
the drop is only legitimate because another reader exists.

The frozen spec dataclass sits beside the resolver because `mantis.train` must not import the
pydantic schema, so the seam type lives on this side of the DAG. NO CODE-SIDE DEFAULT ANYWHERE ON
THE PATH — a parameter default is a migrated authority, not an absent one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiskGuardSpec:
    """The resolved disk-guard cadence + thresholds, in the guard's own units.

    `warn_gb` / `fail_gb` are decimal-GB free-space thresholds, matching the `/1e9` divisor the
    guard calibrates against. `keep_all` is deliberately absent: it is a pruning knob the
    thresholds ignore, it has no config key, and the root passes `False` explicitly.
    """

    interval_sec: float
    warn_gb: float
    fail_gb: float


def resolve_disk_guard(monitor_section: Any) -> DiskGuardSpec:
    """Return the validated disk-guard spec from the `monitor.disk_guard` block."""
    block = monitor_section.disk_guard
    return DiskGuardSpec(
        interval_sec=float(block.interval_sec),
        warn_gb=float(block.warn_gb),
        fail_gb=float(block.fail_gb),
    )


__all__ = ["DiskGuardSpec", "resolve_disk_guard"]
