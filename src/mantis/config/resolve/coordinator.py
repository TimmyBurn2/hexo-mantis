"""`resolve_coordinator_knobs` — the ONE read path for the 18 step-coordinator knobs
(WPMINT Phase K-B, `CARD-COORD-KNOBS` / R78 as clarified by R80).

WAS NINETEEN. `buffer_save_interval` -> `checkpoint_interval` is DELETED by R178(a) under
R116/LAW-08: WP12-R Phase CS (F-CS-2) measured the replay-buffer save production-dead on
every leg, so the key had no reachable effect and does not ship into the run5 mint record.
The rename seam it justified retires with it, and `StepCoordinatorConfig.checkpoint_interval`
— which existed only to receive it — is deleted at the same time. Buffer persistence returns
only as ONE design under CARD-RESUME (R178(c), post-mint).

The eighteen `train.*` keys this file names are read HERE and nowhere else. The composition
root (`mantis.run.compose_run`) threads the resolved spec into `_step_coordinator_config`,
which is now a pure transport: with `stop_step` (S-4), `draw_rate_abort` (Phase D),
`drain_caps` (Phase K-A) and these eighteen all arriving as parameters, that builder holds
**zero literals** and the run's shape is stated entirely by its minted config.

WHAT THIS CLOSES. `_step_coordinator_config` used to open with the sentence "Smoke-grade
defaults for the ~22 knobs CARD-COORD-KNOBS still owns", and it meant it: `eval_interval`,
`log_interval`, `batch_size`, `augment`, `hard_gn_threshold`, `selfplay_stall_timeout_sec`
and thirteen more decided what every run WAS, from a literal no config could see and no mint
record published. R78's card named the deadline (pre-run5-mint) and this is it.

TWO MEASURED DEFECTS DIE WITH THE LITERALS, and both are worth naming because neither was
visible from the config:

* `batch_size` — the coordinator field was DEAD. The live authority was
  `train_cfg.get("batch_size", full_config.get("batch_size", 256))` in
  `coordinator/step.py`, and WPMINT Phase K-A measured that BOTH lookups miss on the
  production path, so the batch size was unconditionally the literal `256` while the field
  beside it said `8`. `train.batch_size` is minted at **256**: this file moves the authority,
  never the number.
* `log_interval` — NARRATION ONLY since R242; it runs no gate. WPMINT DR-7's measurement
  (that `log_interval <= 0` killed the entire hard-abort family AND the `monitor_gates` event
  that would have shown it, while gate 12 still audited the draw-rate row ARMED) was the
  original ground for `ge=1`, and that argument MOVED WITH THE GATES to
  `monitor.gate_interval`. The bound stays here on its own footing — there is no legitimate
  "never narrate" posture — and this resolver is where the value that cannot be zero arrives.

SIX SIBLINGS ARE NOT HERE, BY ADJUDICATION (call K-a). `composition_interval`,
`value_probe_interval`, `soft_ew_threshold`, `soft_ew_min_pts`, `instrumentation_enabled` and
`bot_corpus_path` were `StepCoordinatorConfig` fields with NO reader anywhere in `src/` —
re-verified at HEAD by grep and by recording every attribute read on a live coordinator config
across the whole test tier. They are DELETED, not authored: a config key with no live consumer
is the R1/LAW-08 violation this package exists to prevent, so typing them in would have
created the defect the card meant to close.

NO CODE-SIDE DEFAULT ANYWHERE ON THE PATH (R1/LAW-08/R83). Every field below is required in
the schema, `CoordinatorKnobsSpec` carries no field default, and `_step_coordinator_config`
takes the spec as a keyword-only parameter with no default — a parameter default is where the
authority MIGRATES to when a field default is deleted (MF-2 Attack B), and `drain_caps` is the
worked precedent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoordinatorKnobsSpec:
    """The resolved step-coordinator knobs.

    A frozen dataclass beside the resolver rather than the pydantic section, for
    `DrawRateAbortSpec`/`DrainCapsSpec`'s reason: `train/coordinator/config.py` is the
    DAG-clean seam layer, so nothing in `mantis.train` imports a schema class to consume this.

    The field NAMES are `StepCoordinatorConfig`'s, not the schema's, in the two places the
    two differ — `replay_capacity` -> `capacity`, `replay_capacity_schedule` ->
    `buffer_schedule`. A THIRD rename stood here, `buffer_save_interval` ->
    `checkpoint_interval`, and it is gone with the key R178(a) deletes: it existed because
    `train.checkpoint_interval` (the TRAINER's periodic save) would otherwise have collided
    with a same-named coordinator key, and with the coordinator field deleted there is no
    collision left to disambiguate. The two surviving renames do not propagate into the
    runtime object, whose field names are read by `coordinator/step.py`.
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
    mixing_initial_w: float
    mixing_min_w: float
    mixing_decay_steps: float
    hard_gn_threshold: float
    hard_gn_min_steps: int
    terminal_eval_enabled: bool
    bot_batch_share: float
    selfplay_stall_timeout_sec: float


def resolve_coordinator_knobs(train_section: Any) -> CoordinatorKnobsSpec:
    """Return the validated step-coordinator knobs from the `train` section."""
    return CoordinatorKnobsSpec(
        eval_interval=int(train_section.eval_interval),
        log_interval=int(train_section.log_interval),
        min_buf_size=int(train_section.min_buf_size),
        capacity=int(train_section.replay_capacity),
        # The consumer (`coordinator/step.py` D1) indexes each stage as a MAPPING
        # (`stage["step"]` / `stage["capacity"]`), so the schema blocks are flattened here
        # rather than at the consumer: this phase authors the value, it does not re-shape the
        # runtime contract the value arrives at.
        buffer_schedule=tuple(
            {"step": int(stage.step), "capacity": int(stage.capacity)}
            for stage in train_section.replay_capacity_schedule
        ),
        training_steps_per_game=float(train_section.training_steps_per_game),
        max_train_burst=int(train_section.max_train_burst),
        batch_size=int(train_section.batch_size),
        augment=bool(train_section.augment),
        recency_weight=float(train_section.recency_weight),
        mixing_initial_w=float(train_section.mixing_initial_w),
        mixing_min_w=float(train_section.mixing_min_w),
        mixing_decay_steps=float(train_section.mixing_decay_steps),
        hard_gn_threshold=float(train_section.hard_gn_threshold),
        hard_gn_min_steps=int(train_section.hard_gn_min_steps),
        terminal_eval_enabled=bool(train_section.terminal_eval_enabled),
        bot_batch_share=float(train_section.bot_batch_share),
        selfplay_stall_timeout_sec=float(train_section.selfplay_stall_timeout_sec),
    )


__all__ = ["CoordinatorKnobsSpec", "resolve_coordinator_knobs"]
