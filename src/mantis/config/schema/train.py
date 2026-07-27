"""`TrainConfig` — training hyperparameters as first-class schema fields (R-TRAINCONFIG-SCHEMA
closure, DESIGN_P2.md §2). Training knobs are no longer a code-side-default `TrainHParams`
R1-exception: this schema field IS the sole default authority (R1); `TrainHParams` (still a
`@dataclass(frozen=True)` runtime object in `mantis.train.trainer.core`) is built FROM a
validated `TrainConfig`, never independently defaulted.
"""
from typing import Literal

from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel


class TrainConfig(StrictModel):
    """Training hyperparameters (R-TRAINCONFIG-SCHEMA). Every field REQUIRED — no terminal
    default anywhere in this class; the minted value in each `configs/*.yaml` is the sole
    default authority (R1).

    `entropy_reg_weight` (R37.1/.3): a positive coefficient on a subtracted entropy bonus
    (`trainer/core.py` `_train_on_batch` / `losses.py`) — larger = more exploration
    pressure. The historical `-0.005` was a sign-leak from the loss formula and never
    existed in this tree (ADJ-01). Deliberately carries NO `Field(ge=0)` bound: the floor
    is enforced solely by `_entropy_sign` below so the NAMED sign-law error (not a bare
    pydantic bound message) is always what a negative value raises (a `Field(ge=0)` would
    fire first and make the named message unreachable).

    The graph-run dense-only-weights ban (`trainer/core.py`) is UNTOUCHED (R37.4/LAW-07):
    it stays the single train-step-time authority in `train_step_from_graph_batch`; this
    schema only bounds the sign, never duplicates the representation-aware ban (it has no
    `representation` to check against).
    """

    # optimizer / schedule
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    fp16: bool
    amp_dtype: Literal["fp16", "bf16"]  # grid-path only; graph is bf16-pinned regardless (R30b)
    lr_schedule: Literal["cosine", "none"]
    total_steps: int = Field(ge=1)
    scheduler_t_max: int | None = Field(default=..., ge=1)  # no terminal default; None is real
    eta_min: float = Field(ge=0)
    min_lr: float | None = Field(default=..., ge=0)
    checkpoint_interval: int = Field(ge=0)
    # WP-UNFREEZE K1: continuous actor-sync cadence in coordinator training steps.
    # `ge=1` means NO disabled value exists — the schema cannot express "don't sync"
    # (R49 enforced at the type level). Resolved ONLY by
    # `mantis.config.resolve.actor_sync.resolve_actor_sync_cadence`.
    actor_sync_cadence_steps: int = Field(ge=1)
    # WPAX S-4 (F-C re-anchor): the RUN-LENGTH authority, in coordinator training steps.
    # Consumed by mantis.config.resolve.run_length.resolve_max_train_steps ->
    # StepCoordinatorConfig.stop_step, the real stop condition (coordinator/step.py O2).
    # Distinct from total_steps, which is only the LR-scheduler horizon.
    #
    # ABSOLUTE, not per-process: StepCoordinator seeds `self._train_step` from
    # `trainer.step` and re-reads it each burst, so a run resumed past this ceiling
    # terminates immediately having performed ZERO syncs. That is correct — a run past its
    # cap is done — but it looks exactly like a frozen actor, so it is stated here, in
    # resolve_max_train_steps' docstring, and pinned by the resume oracle.
    #
    # `ge=1` is a floor, not THE floor: the reachability validator dominates it. With any
    # legal cadence the smallest expressible run is 3 (cadence 1 < threshold 2 < 3).
    max_train_steps: int = Field(ge=1)

    # loss selection + targets
    completed_q_values: bool
    value_target: Literal["pure_outcome_z"]
    policy_target: Literal["raw_visit_distribution"]
    draw_reward: float
    ply_cap_value: float
    policy_prune_frac: float = Field(ge=0, lt=1)

    # entropy (R37) — see class docstring: NO Field(ge=0), _entropy_sign is the sole gate.
    entropy_reg_weight: float

    # aux/loss weights (mirrors the graph-forbidden dense-only weight names — values only,
    # never the ban itself; see the class docstring)
    aux_opp_reply_weight: float = Field(ge=0)
    uncertainty_weight: float = Field(ge=0)
    ownership_weight: float = Field(ge=0)
    threat_weight: float = Field(ge=0)
    aux_chain_weight: float = Field(ge=0)
    ply_index_weight: float = Field(ge=0)
    threat_pos_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def _entropy_sign(self) -> "TrainConfig":
        if self.entropy_reg_weight < 0.0:
            raise ValueError(
                "train.entropy_reg_weight must be >= 0: it is a positive coefficient on a "
                "subtracted entropy bonus (trainer/core.py _train_on_batch / losses.py) — "
                "larger = more exploration pressure. A negative value would flip it into a "
                "policy-sharpening penalty, not a smaller bonus."
            )
        return self
