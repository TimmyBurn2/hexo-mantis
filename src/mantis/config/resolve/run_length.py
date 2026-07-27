"""`resolve_max_train_steps` — the ONE read path for the run-length knob (WPAX S-4, §4.4).

`train.max_train_steps` is read HERE and nowhere else; the composition root
(`mantis.run.compose_run`) threads the resolved value into `StepCoordinatorConfig.stop_step`,
which is the real stop condition (`coordinator/step.py` O2). `train.total_steps` is ONLY the
LR-scheduler horizon (`trainer/core.py` cosine `t_max`) and is NOT a run-length authority — no
stop condition reads it.

The returned value is an ABSOLUTE step ceiling, not a per-process budget: `StepCoordinator`
seeds `self._train_step` from `trainer.step`, so a run RESUMED past this ceiling terminates
immediately having performed zero syncs. That is correct — a run past its cap is done — but it
looks exactly like a frozen actor, so it is stated here and pinned by a named resume oracle.

No code-side default and no disable sentinel (R1/R49): the schema field is the sole authority
and a missing key never reaches here (pydantic rejects it at load, naming the key).
"""
from __future__ import annotations

from typing import Any


def resolve_max_train_steps(train_section: Any) -> int:
    """Return the validated absolute run-length ceiling in coordinator training steps."""
    return int(train_section.max_train_steps)


__all__ = ["resolve_max_train_steps"]
