"""`resolve_drain_caps` — the ONE read path for the close-out drain / terminal-eval caps.

`monitor.drain.*` is read here and nowhere else. The composition root threads the resolved
spec into `_step_coordinator_config`, whose four same-named fields it then lifts into
`mantis.eval.pipeline.DrainCaps` — the object `drain_budget_sec` and `_run_terminal_sync`'s
`budget_sec` actually read.

There is no code-side default anywhere on the path: with the schema block authoritative, a
dataclass default is a second authority a caller can silently inherit. `DrainCapsConfig`'s
`Field(gt=0)` means no number disables a join bound, so there is no off sentinel either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrainCapsSpec:
    """The resolved drain/terminal-eval wall-clock caps.

    Defined beside the resolver rather than on the pydantic block or `eval.pipeline.DrainCaps`
    because nothing in `mantis.train` may import the schema class and `mantis.config` may not
    import `mantis.eval`.
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
