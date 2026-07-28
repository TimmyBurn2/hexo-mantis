"""`resolve_drain_caps` — the ONE read path for the close-out drain / terminal-eval caps
(WPMINT Phase K-A, ruling R93; the DR-11 finding).

`monitor.drain.*` is read HERE and nowhere else. The composition root
(`mantis.run.compose_run`) threads the resolved spec into `_step_coordinator_config`, whose
four same-named `StepCoordinatorConfig` fields `compose_run` then lifts into
`mantis.eval.pipeline.DrainCaps` — the object `drain_budget_sec` (the mid-run/teardown
`drain_pending` bound) and `_run_terminal_sync`'s `budget_sec` actually read.

WHY THIS FILE EXISTS AT ALL. The four keys were minted, schema-validated (`DrainCapsConfig`,
`schema/monitor.py`) and claimed by BOTH consumer registries — and read by nothing.
`config/resolve/monitor.py` did `data.pop("drain")` and dropped them on the floor, while
`run.py` built the real `DrainCaps` from a hardcoded `900.0` plus three
`StepCoordinatorConfig` terminal defaults. WPMINT Phase DR found it while verifying DR-6.
The registry citation was not merely stale: it NAMED a live function that never saw the
value, which is why R93 makes every citation Phase K touches verifiable BY MUTATION (set the
knob, observe the consumer) rather than by grep — a grep cannot tell a reader from a `pop`.

NO CODE-SIDE DEFAULT ANYWHERE ON THE PATH (R1/LAW-08). The four
`DEFAULT_FINAL_EVAL_DRAIN_*`/`DEFAULT_TERMINAL_EVAL_HARD_CAP_SEC` constants and the three
dataclass terminal defaults they fed are DELETED by the same change: with the schema block
authoritative, a dataclass default is a second authority a caller can silently inherit, and
`DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC` was already a dead twin of `run.py`'s bare literal.
`DrainCapsConfig`'s `Field(gt=0)` means no number disables a join bound — `join(0)` is not a
bound — so there is no off sentinel to inherit either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrainCapsSpec:
    """The resolved drain/terminal-eval wall-clock caps.

    A frozen dataclass defined beside the resolver rather than the pydantic block or
    `eval.pipeline.DrainCaps` itself, for the reason `DrawRateAbortSpec` states:
    `train/coordinator/config.py` is the DAG-clean seam layer, so nothing in `mantis.train`
    imports the schema class — and `mantis.config` must not import `mantis.eval`.
    """

    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float
    eval_final_drain_hard_cap_sec: float
    terminal_eval_hard_cap_sec: float


def resolve_drain_caps(monitor_section: Any) -> DrainCapsSpec:
    """Return the validated drain/terminal-eval caps from the `monitor.drain` block."""
    block = monitor_section.drain
    return DrainCapsSpec(
        final_eval_drain_timeout_sec=float(block.final_eval_drain_timeout_sec),
        eval_final_drain_safety_factor=float(block.eval_final_drain_safety_factor),
        eval_final_drain_hard_cap_sec=float(block.eval_final_drain_hard_cap_sec),
        terminal_eval_hard_cap_sec=float(block.terminal_eval_hard_cap_sec),
    )


__all__ = ["DrainCapsSpec", "resolve_drain_caps"]
