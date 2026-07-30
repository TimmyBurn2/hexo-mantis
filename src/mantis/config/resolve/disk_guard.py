"""`resolve_disk_guard` — the ONE read path for the disk guard's thresholds (WPMAIN, R122).

`monitor.disk_guard.*` is read HERE and nowhere else. The composition root
(`mantis.run.compose_run`) threads the resolved spec into
`mantis.train.lifecycle.disk_guard.DiskGuard`, whose `check_once` emits `disk_free`, WARNs
below `warn_gb` and SIGTERMs the run below `fail_gb` — LAW-16's third leg, which had never
run in any composed process.

WHY THIS FILE EXISTS AT ALL, and why it is not a style preference. Before WPMAIN the guard
was constructed at exactly one site, `mantis.train.subsystems.build_subsystems`, which had
ZERO callers; its thresholds arrived as `config.get("disk_guard", {}).get("interval_sec",
60.0)`-shaped code-side defaults over a key that existed in no schema and no config. R121(b)
mandates that the root construct the guard and R1 forbids the construction values being
literals or `dict.get` defaults, so R122 grants the family: one block, one resolver, three
typed leaves. This module is that resolver, and it is ALSO what legitimates
`resolve_monitor_config`'s enumerated `data.pop("disk_guard")` — the sibling file states the
law verbatim: the drop is only legitimate BECAUSE another reader exists. Without this
function that pop is the DR-11 defect (minted, schema-validated, registry-claimed keys read
by nothing) re-created on a second block.

Mirrors `mantis.config.resolve.drain.resolve_drain_caps` deliberately, down to the frozen
spec dataclass defined beside the resolver: `mantis.train` must not import the pydantic
schema, so the seam type lives on this side of the DAG.

NO CODE-SIDE DEFAULT ANYWHERE ON THE PATH (R1/LAW-08). `DiskGuardConfig` carries `gt=0` on
all three leaves plus the `fail_gb < warn_gb` model rule, and `DiskGuard.__init__` was
stripped to required-keyword parameters by the same change — a parameter default is a
MIGRATED authority (MF-2 Attack B), not an absent one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiskGuardSpec:
    """The resolved disk-guard cadence + thresholds, in the guard's own units.

    `interval_sec` is the guard thread's poll cadence; `warn_gb` / `fail_gb` are decimal-GB
    free-space thresholds (the `/1e9` divisor `disk_guard.py` calibrates against). `keep_all`
    is deliberately absent: it is a pruning knob the thresholds ignore, it has no config key
    (R122), and the root passes `False` explicitly with a disclosure comment.
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
