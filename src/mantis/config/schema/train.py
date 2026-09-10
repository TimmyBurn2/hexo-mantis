# >300 justify (R8). One class per config block, plus the per-field GROUNDS the house style
# requires: what each bound is a bound ON (the mechanism's own range, never policy), which
# defect it makes inexpressible, and the ONE resolver that reads it. Splitting a class out
# would put a field and its grounds on opposite sides of an import, and splitting the classes
# apart would separate `train.draw_rate_abort.consec` from the terms that travel with it.
"""`TrainConfig` — training hyperparameters as first-class schema fields.

This schema field IS the sole default authority; `TrainHParams` (a frozen dataclass in
`mantis.train.trainer.core`) is built FROM a validated `TrainConfig`, never independently
defaulted.
"""
from typing import Literal

from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel


class DrawRateAbortConfig(StrictModel):
    """The draw-rate collapse hard abort's terms — ONE block, ONE fact.

    The four components (`threshold`, `min_step`, `N_pool_min`, `consec`) are INSEPARABLE, so
    they are a nested block and not four flat `X | None` keys, which would give four authorities
    over one fact and could disagree in ways no predicate can adjudicate. All four are
    RUN-SCOPED CONSTANTS pre-registered at mint prereg, the only place they may change.

    The bounds are bounds on the METRIC's own range, not policy:

    * `threshold` — the metric is `Sum(draws)/Sum(completed)`, a fraction in [0, 1], and the
      predicate is an UPPER bound, so `gt=0` alone leaves the high half open: a threshold above
      1.0 can never be met, is accepted, and reads ARMED. `le=1` closes the natural percent slip
      (35% written as `35`). DISCLOSED RESIDUAL: `1e-300` still loads, but the smallest non-zero
      pooled rate at the bar is exactly `1/N_pool_min` (0.02 at run5's 50) at every worker count.
    * `min_step` — `ge=1`, no disabled value; the top end is closed against
      `train.max_train_steps` by the cross-validator in `schema/core.py`.
    * `N_pool_min` — the evidence bar, proposed at 50 from measured deque geometry. Below it the
      gate makes NO OBSERVATION, so a healthy `0.0` fabricated from an empty set is answered by
      TYPE. Its ceiling, `DRAW_RATE_WINDOW * selfplay.n_workers`, spans two sections and so is
      enforced by a cross-validator rather than an `le=` here.
    * `consec` — authored here rather than as a flat `train.*` key, which could be set on a
      config whose block is `null`, i.e. a term of an abort nobody armed. `ge=1`, and NO upper
      bound is needed because the gate's history ring is sized BY this value at the point of
      use. DISCLOSED: an observation is ATTEMPTED once per `monitor.gate_interval` steps, and a
      boundary that observes nothing neither advances nor RESETS, so `consec` is a
      sustained-ness bar whose step span is a lower bound and it does not delay the first fire.

    Read by exactly one path: `mantis.config.resolve.draw_rate.resolve_draw_rate_abort`.
    """

    threshold: float = Field(gt=0, le=1)
    min_step: int = Field(ge=1)
    N_pool_min: int = Field(ge=1)
    consec: int = Field(ge=1)

    @model_validator(mode="after")
    def _one_drawn_game_cannot_fire_the_abort(self) -> "DrawRateAbortConfig":
        """Refuse an `N_pool_min` at which a SINGLE drawn game meets the threshold.

        `ge=1` alone admits values where the pooled rate's smallest non-zero value at the bar,
        `1/N_pool_min`, already reaches the threshold: at `N_pool_min=4` with `threshold=0.25`
        one drawn game in four fires the hard abort. Derived from values already in this block;
        run5 satisfies it by 12.5x (0.02 < 0.25).

        Ratified on measured grounds: the healthy draw rate is ~0.00025 (draws arise only from
        ply-cap truncation), so the induced floor of 0.02 at `N_pool_min=50` sits ~80x above
        healthy. COST: `threshold <= 1/N_pool_min` becomes INEXPRESSIBLE, and a sub-floor
        threshold needs a schema amendment rather than a config edit.
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


class ReplayCapacityStage(StrictModel):
    """One step of `train.replay_capacity_schedule` — at train step S, grow the buffer to C.

    A nested block rather than two parallel lists, whose lengths could disagree. `capacity` is
    `ge=1` because a buffer that holds nothing is not a buffer, and the walker only ever GROWS,
    so a stage below the current capacity is silently inert rather than a shrink.
    """

    step: int = Field(ge=0)
    capacity: int = Field(ge=1)


class MicrobatchCapsConfig(StrictModel):
    """The GRAPH training step's memory bound — ONE block, ONE fact.

    Both members are sized TOGETHER from ONE measured cost model against ONE budget
    (`peak ~ a + b*E + c*N`), so two flat keys would give two authorities over one byte budget.

    `train.batch_size` bounds the number of GRAPHS and neither quantity that drives memory:
    `_GINEConv.forward` materialises per-edge `[E, hidden]` tensors and the JK-cat `[N,
    L*hidden]`, both SUMS over the sampled graphs. Measured: `E = 18 735 930` at
    `batch_size: 256`, one 8.94 GiB request on a 15.48 GiB card, node counts spanning
    26 -> 5 234. BOTH members, because many low-degree graphs pass an edge-only bound.

    Enforced by SPLITTING at graph boundaries and ACCUMULATING gradients, so the optimizer
    result is the un-split step's — never a truncation, never a drop. A single graph over
    either member raises `GraphMicroBatchOverCap`, naming which.

    `ge=1` on both and NO off value: an uncapped graph step is the defect this block exists to
    make unconstructible, so a disable sentinel would switch the fix back off.

    GRAPH-ROUTE ONLY, scoped by the SCHEMA rather than by a call site:
    `RunConfig._arch_scoped_keys_are_present_iff_their_arch` requires the block on a graph config
    and refuses it on any other, so a grid mint never invents a number for a quantity that run
    has none of. Read late, through `resolve_microbatch_caps` and a CALLABLE the grid arm is
    never given, so a grid `full_config` with no `train` section stays loadable.
    """

    max_edges: int = Field(ge=1)
    max_nodes: int = Field(ge=1)


class EmaConfig(StrictModel):
    """The EMA lever's arming keys.

    `resolve_ema_config` used to read four `.get(...)` names off a `RunConfig` that has
    `extra="forbid"` and no `ema` leaf, so EMA was OFF on every run and no config could turn it
    on — invisible, because a disabled lever and an absent one produce the same run. REQUIRED
    members, minted `enabled: false`, with the dead reader's own values, so arming the lever
    changes only `enabled` and not the regime underneath it.
    """

    enabled: bool
    decay: float = Field(ge=0.0, lt=1.0)
    update_every: int = Field(ge=1)


class TrainConfig(StrictModel):
    """Training hyperparameters. Every field REQUIRED — no terminal default anywhere in this
    class; the minted value in each `configs/*.yaml` is the sole default authority.
    """

    # optimizer / schedule
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    # The run DEVICE is a CONFIG FACT, not a CLI flag: `--device` let a preflight point a
    # CUDA-minted run at the CPU. CLOSED vocabulary, narrower than the dead flag — device
    # indices (`cuda:1`) are unrepresentable, and widening the enum is a named design act.
    device: Literal["cpu", "cuda"]
    lr_schedule: Literal["cosine", "none"]
    total_steps: int = Field(ge=1)
    scheduler_t_max: int | None = Field(default=..., ge=1)  # no terminal default; None is real
    eta_min: float = Field(ge=0)
    checkpoint_interval: int = Field(ge=0)
    # Continuous actor-sync cadence in coordinator training steps. `ge=1` means NO disabled
    # value exists. Resolved only by `mantis.config.resolve.actor_sync`.
    actor_sync_cadence_steps: int = Field(ge=1)
    # The EMA lever's arming block, REQUIRED so every config states its posture explicitly:
    # `resolve_ema_config` used to read four names no schema had, so the lever was unreachable.
    ema: EmaConfig
    # The RUN-LENGTH authority, in coordinator training steps, consumed by
    # `resolve_max_train_steps` -> `StepCoordinatorConfig.stop_step`; distinct from
    # `total_steps`, the LR-scheduler horizon. ABSOLUTE, not per-process: a run resumed past
    # this ceiling terminates immediately, which is correct but looks like a frozen actor.
    max_train_steps: int = Field(ge=1)
    # The draw-rate collapse hard abort's ARMING SURFACE. `None` is EXPLICITLY OFF, and there
    # is no boolean enable beside it, which would be a second authority over one fact.
    # `default=...` is this class's no-terminal-default idiom: absence names the key.
    draw_rate_abort: DrawRateAbortConfig | None = Field(default=...)

    # The step-coordinator knobs: builder literals and dataclass terminal defaults that decided
    # what the run IS while the minted config said nothing. FLAT `train.*` keys and NOT a
    # `train.coordinator` block — naming a config block after a dataclass was ruled against.
    # Six sibling fields were DELETED rather than authored: they had no reader in `src/`.
    #
    # `eval_interval` — the promotion-decision cadence. `ge=1`: at `<= 0` the whole
    # eval/promotion pipeline is off while nothing says so. The off posture is the typed
    # `eval_enabled: false` key, which IS the fact rather than a number that happens to disable.
    eval_interval: int = Field(ge=1)
    # `log_interval` — NARRATION ONLY, and it runs NO gate: the hard-abort family moved to
    # `monitor.gate_interval`, because at a minted 1000 no draw-rate abort could fire before
    # training step 1000. `ge=1` because there is no legitimate "never narrate" posture.
    log_interval: int = Field(ge=1)
    # `buffer_save_interval` is DELETED: the replay-BUFFER save cadence, measured
    # production-dead. `train.checkpoint_interval` above is the TRAINER's periodic save.
    # `min_buf_size` — the warmup floor, below which `step()` returns `in_warmup`. `ge=1`
    # because a floor of 0 means "train on an empty buffer", which the sampler cannot satisfy.
    min_buf_size: int = Field(ge=1)
    # `replay_capacity` — the replay window, i.e. the distribution the learner trains on.
    # Renamed from the dataclass field `capacity`, which names nothing as `train.capacity`.
    replay_capacity: int = Field(ge=1)
    # `replay_capacity_schedule` — the step-keyed ramp; `[]` means "no ramp", not an off
    # switch. `_schedule_idx` never rewinds, so `_stages_are_strictly_increasing` below makes a
    # non-increasing schedule (which would silently skip stages) unrepresentable.
    replay_capacity_schedule: list[ReplayCapacityStage]
    # `training_steps_per_game` — the sample-reuse ratio. `gt=0` because `_steps_budget` floors
    # its result at 1, so `0` means "one step per round" while reading as an off switch.
    training_steps_per_game: float = Field(gt=0)
    # `max_train_burst` — the ceiling of that budget. `ge=1` because the `max(1, ...)` floor is
    # INSIDE the `min(...)`, so `0` clamps the budget to 0 and stops the learner silently.
    max_train_burst: int = Field(ge=1)
    # `batch_size` — the training batch, AUTHORED HERE AND NOWHERE ELSE. It used to be a dict
    # lookup whose two levels both miss on the production path, so the size was unconditionally
    # a literal 256; the minted value is 256, so only the authority moved.
    batch_size: int = Field(ge=1)
    # `microbatch_caps` — the GRAPH step's memory bound, which `batch_size` does not give.
    # ARCH-SCOPED: `None` here is the ABSENCE of the key, never a value, and presence is read
    # off `model_fields_set` so an explicit `null` is refused on a graph config too.
    microbatch_caps: MicrobatchCapsConfig | None = None
    # `augment` — 12-fold hex-symmetry augmentation of every sampled batch. It multiplies the
    # effective dataset, so two runs that differ only here are not comparable.
    augment: bool
    # `recency_weight` — the fraction of each batch drawn from the recency window. `ge=0, le=1`
    # is that fraction's own range: above 1 the sampler clamps and the difference is unreal.
    recency_weight: float = Field(ge=0, le=1)
    # `hard_gn_threshold` / `hard_gn_min_steps` — the `grad_norm_hard_abort` gate. `gt=0`
    # because a threshold of 0 fires on every finite step; `allow_inf_nan=False` because the
    # gate guards on `math.isfinite`, so an infinite threshold reads ARMED and can never be met.
    # DISCLOSED: the shipped `1e9` is finite and unreachable, so this bound still admits an
    # effectively-disarmed abort; the derivable ceiling lives in the armed-abort manifest.
    hard_gn_threshold: float = Field(gt=0, allow_inf_nan=False)
    # `ge=1`: at `0` the gate fires the FIRST time the threshold is exceeded, the opposite of
    # sustained instability. DISCLOSED: a very large value disarms the abort just as quietly.
    hard_gn_min_steps: int = Field(ge=1)
    # `terminal_eval_enabled` — whether close-out runs a terminal eval round, i.e. whether the
    # run gets its LAST promotion opportunity. A REGIME fact that once had three authorities.
    terminal_eval_enabled: bool
    # `selfplay_stall_timeout_sec` — the stall watchdog's wall-clock budget. `gt=0` and
    # `allow_inf_nan=False` because the watchdog is ALWAYS ARMED, while `watchdog.py` lets
    # `timeout_sec <= 0` disable the fire while the arm-log still emits. An OPERATIONAL
    # CONSTANT, which is why it alone carries a default.
    selfplay_stall_timeout_sec: float = Field(default=1800.0, gt=0, allow_inf_nan=False)

    # loss selection + targets. `completed_q_values` is DELETED here and on `selfplay`: which
    # loss the trainer applies follows from `policy_target`, itself pinned to `search.kind`.
    value_target: Literal["pure_outcome_z"]
    policy_target: Literal["raw_visit_distribution", "completed_improved_policy"]
    draw_reward: float
    ply_cap_value: float
    #: The POLICY weight a fast-arm (`is_full_search == 0`) row carries; value is always
    #: supervised on those rows, and this replaces a gate that discarded the fast arm's policy
    #: outright. `ge=0` and not `gt=0` BECAUSE 0.0 is the shipped value, which reproduces that
    #: gate. Read once per step via `fast_policy_weight_provider`, the same provider shape
    #: `train.microbatch_caps` carries: a grid `full_config` has no `train` section at all.
    fast_policy_weight: float = Field(ge=0)

    @model_validator(mode="after")
    def _stages_are_strictly_increasing(self) -> "TrainConfig":
        """Require `train.replay_capacity_schedule` to be strictly increasing in `step`.

        The consumer's cursor never rewinds, so out-of-order stages are applied in the same pass
        at the earlier step and the ramp the run performs is not the one written; equal steps
        are the same defect. An empty schedule satisfies this vacuously.
        """
        steps = [stage.step for stage in self.replay_capacity_schedule]
        if any(later <= earlier for earlier, later in zip(steps, steps[1:], strict=False)):
            raise ValueError(
                f"train.replay_capacity_schedule steps must be strictly increasing; got "
                f"{steps}. The consumer's cursor only moves forward, so an out-of-order or "
                "duplicated step is consumed in the same pass as the one before it and the "
                "ramp the run performs is not the ramp that was written"
            )
        return self
