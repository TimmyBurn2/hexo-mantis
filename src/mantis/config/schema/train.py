"""`TrainConfig` — training hyperparameters as first-class schema fields (R-TRAINCONFIG-SCHEMA
closure, DESIGN_P2.md §2). Training knobs are no longer a code-side-default `TrainHParams`
R1-exception: this schema field IS the sole default authority (R1); `TrainHParams` (still a
`@dataclass(frozen=True)` runtime object in `mantis.train.trainer.core`) is built FROM a
validated `TrainConfig`, never independently defaulted.
"""
from typing import Literal

from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel


class DrawRateAbortConfig(StrictModel):
    """The draw-rate collapse hard abort's terms — ONE block, ONE fact (WPAX Phase D,
    R65 re-scoped by R80, shaped by R79 as amended by R83; the STATISTIC replaced by R92).

    The fact under single authority is *"is the draw-rate collapse abort armed, and on what
    terms"*. It has three INSEPARABLE components, which is why they are a nested block and
    not three flat `X | None` keys: three independent keys give three authorities over one
    fact and can disagree in ways no predicate can adjudicate (`threshold: 0.25,
    N_pool_min: null` is neither armed nor disarmed). The block makes disagreement
    unrepresentable — the three arrive together or not at all.

    ALL THREE ARE RUN-SCOPED CONSTANTS (R82/R85/R92), pre-registered at mint prereg, which
    is "the only place they may change". They are not tunables; changing one means
    re-minting with a recorded delta and a fresh prereg, never editing a config in place
    (R1).

    WPMINT Phase DS (R92) DELETED `min_samples` and added `N_pool_min`. The per-worker
    inclusion bar is gone with the statistic it guarded: the gated metric is now the POOLED
    COUNT-WEIGHTED rate `Sum(draws) / Sum(completed)` over the union of worker windows
    (`train.coordinator.config.pooled_draw_rate`). Phase DR measured the retired metric —
    an unweighted mean over the *included* set — firing at a true pool rate of 0.0319 and
    staying silent at 0.968 (DR-3), and fabricating a healthy `0.0` from an empty included
    set (DR-4). The count-weighted rate cannot exclude anyone, so neither defect has a
    surface left.

    The bounds are bounds on the METRIC's own range, not policy:

    * `threshold` — `pooled_draw_rate` is `Sum(draws)/Sum(completed)`, a fraction in [0, 1],
      and the predicate is `all(value >= threshold)`, an UPPER bound. `gt=0` alone leaves
      the high half open: a threshold > 1.0 can never be met, is accepted, and reads ARMED
      to the armed-abort manifest — "armed in the config, absent in effect"
      (`schema/core.py`'s own words for the sibling defect). Reachable by the natural
      percent slip (an operator meaning 35% writes `35`), so `le=1` closes it.
      DISCLOSED RESIDUAL: `1e-300` still loads. That is a hair-trigger, not a disarm, and
      the type does not close it — but `N_pool_min` now bounds it HONESTLY, which is the
      claim `min_samples` could not make. WPMINT DR-2 measured the old docstring's
      mitigation to be arithmetically FALSE: `1/min_samples` bounded ONE WORKER's rate while
      the compared value was a MEAN over N included workers, whose floor was
      `1/(min_samples*N)` — 0.0003125 at N=64, understated by a factor of N. Under R92 the
      compared value IS `Sum/Sum`, so its smallest non-zero value at the bar is exactly
      `1/N_pool_min` (0.02 at run5's 50) AT EVERY WORKER COUNT. The residual survives only
      BELOW that floor, and `_one_drawn_game_cannot_fire_the_abort` closes the part of it
      that matters (a single drawn game reaching the threshold).
    * `min_step` — R80's second guard, unchanged by R92. No "disabled" value exists
      (`ge=1`), and the twin cross-validator in `schema/core.py` closes the top end against
      `train.max_train_steps`.
    * `N_pool_min` — R92's evidence bar, proposed at 50 by DESIGN_DS from measured deque
      geometry. Below it the gate makes NO OBSERVATION (a `None`, skip-counted, never
      appended), so ADJ-19's healthy-`0.0`-from-nothing is answered by TYPE rather than by
      value. Its TOP end is closed by `schema/core.py`'s
      `_draw_rate_evidence_bar_is_reachable`, NOT by an `le=` here: the ceiling is
      `DRAW_RATE_WINDOW * selfplay.n_workers` (measured — `Sum(len(dq))` saturates there),
      which spans two sections and so cannot live on this field. That cross-validator is
      what re-establishes the load-bearing bound `min_samples: le=DRAW_RATE_WINDOW` carried;
      for every config on this tree (`n_workers: 1`) it evaluates to the same number.

    Read by exactly one path: `mantis.config.resolve.draw_rate.resolve_draw_rate_abort`.
    """

    threshold: float = Field(gt=0, le=1)
    min_step: int = Field(ge=1)
    N_pool_min: int = Field(ge=1)

    @model_validator(mode="after")
    def _one_drawn_game_cannot_fire_the_abort(self) -> "DrawRateAbortConfig":
        """ADJ-14's own defect, re-expressed on R92's statistic (WPMINT DR-9's class).

        `ge=1` alone admits `N_pool_min` values at which a SINGLE drawn game meets the
        threshold: the pooled rate's smallest non-zero value at the bar is `1/N_pool_min`,
        so at `N_pool_min=4` with `threshold=0.25` one drawn game in four fires the hard
        abort. That is the one-game saturation R80 ordered closed, on a new axis. The rule
        is derived entirely from values already in this block — no invented number — and
        run5 satisfies it with three orders of margin (0.02 < 0.25).
        """
        if 1.0 / self.N_pool_min >= self.threshold:
            raise ValueError(
                f"train.draw_rate_abort.N_pool_min ({self.N_pool_min}) is too small for "
                f"threshold {self.threshold}: the pooled rate's smallest non-zero value at "
                f"the bar is 1/{self.N_pool_min} = {1.0 / self.N_pool_min}, so ONE drawn "
                f"game would meet the threshold and fire the hard abort. Raise N_pool_min "
                f"above {int(1.0 / self.threshold)}, or RAISE the threshold above "
                f"{1.0 / self.N_pool_min}. Lowering the threshold NEVER resolves this — it "
                f"makes 1/N_pool_min >= threshold more true, not less"
            )
        return self


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
    # WPAX Phase D (R65/R80/R79+R83): the draw-rate collapse hard abort's ARMING SURFACE.
    # `None` is EXPLICITLY OFF — a word an operator writes deliberately, never a default
    # that happens to disable (R79(1)); a block is ARMED on exactly those terms. There is
    # no boolean enable beside it, because the value already gates its own check and a
    # boolean would be a second authority over one fact (R79/R1/LAW-08).
    # `default=...` is this class's own no-terminal-default idiom (see `scheduler_t_max` /
    # `min_lr` above): absence is an error naming the key (R1/LAW-11).
    # Consumed by mantis.config.resolve.draw_rate.resolve_draw_rate_abort ->
    # compose_run -> StepCoordinatorConfig.draw_rate_abort.
    draw_rate_abort: DrawRateAbortConfig | None = Field(default=...)

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
