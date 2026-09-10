"""`resolve_coordinator_knobs` — the ONE read path for the step-coordinator knobs.

The `train.*` keys named here are read HERE and nowhere else; the composition root threads the
resolved spec into `_step_coordinator_config`, which is then a pure transport holding ZERO
literals, so the run's shape is stated entirely by its minted config. It used to carry
"smoke-grade defaults" that decided what every run WAS from a literal no config could see.

Two measured defects died with those literals. `batch_size`: the coordinator field was DEAD, the
live authority a `.get(..., 256)` chain whose lookups both miss on the production path, so the
batch size was unconditionally 256 while the field beside it said 8 — `train.batch_size` is
minted at 256, so this file moves the authority, never the number. `log_interval`: narration
only, its `ge=1` bound standing on there being no legitimate "never narrate" posture.

No code-side default anywhere on the path: every field is required in the schema, the spec
carries no field default, and the builder takes it keyword-only with no default, because a
parameter default is where the authority migrates when a field default is deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoordinatorKnobsSpec:
    """The resolved step-coordinator knobs.

    A frozen dataclass beside the resolver rather than the pydantic section, so nothing in
    `mantis.train` imports a schema class to consume it. The field NAMES are
    `StepCoordinatorConfig`'s where the two differ (`replay_capacity` -> `capacity`,
    `replay_capacity_schedule` -> `buffer_schedule`), and those renames do not propagate into
    the runtime object.
    """

    eval_interval: int
    log_interval: int
    min_buf_size: int
    capacity: int
    buffer_schedule: tuple[dict[str, Any], ...]
    training_steps_per_game: float
    max_train_burst: int
    batch_size: int
    augment: bool
    recency_weight: float
    hard_gn_threshold: float
    hard_gn_min_steps: int
    terminal_eval_enabled: bool
    selfplay_stall_timeout_sec: float


def resolve_coordinator_knobs(train_section: Any) -> CoordinatorKnobsSpec:
    """Return the validated step-coordinator knobs from the `train` section."""
    return CoordinatorKnobsSpec(
        eval_interval=int(train_section.eval_interval),
        log_interval=int(train_section.log_interval),
        min_buf_size=int(train_section.min_buf_size),
        capacity=int(train_section.replay_capacity),
        # The consumer indexes each stage as a MAPPING (`stage["step"]`/`stage["capacity"]`),
        # so the schema blocks are flattened here rather than at the consumer.
        buffer_schedule=tuple(
            {"step": int(stage.step), "capacity": int(stage.capacity)}
            for stage in train_section.replay_capacity_schedule
        ),
        training_steps_per_game=float(train_section.training_steps_per_game),
        max_train_burst=int(train_section.max_train_burst),
        batch_size=int(train_section.batch_size),
        augment=bool(train_section.augment),
        recency_weight=float(train_section.recency_weight),
        hard_gn_threshold=float(train_section.hard_gn_threshold),
        hard_gn_min_steps=int(train_section.hard_gn_min_steps),
        terminal_eval_enabled=bool(train_section.terminal_eval_enabled),
        selfplay_stall_timeout_sec=float(train_section.selfplay_stall_timeout_sec),
    )


__all__ = ["CoordinatorKnobsSpec", "resolve_coordinator_knobs"]
