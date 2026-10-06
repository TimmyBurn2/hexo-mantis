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


class PlyCapAbortConfig(StrictModel):
    """The ply-cap halt: the last `window_games` games' cap fraction STRICTLY above `rate` at or past `min_step` halts; `null` is OFF."""

    rate: float = Field(gt=0, le=1)
    window_games: int = Field(ge=1)
    min_step: int = Field(ge=1)

    @model_validator(mode="after")
    def _one_cap_game_cannot_fire_the_abort(self) -> "PlyCapAbortConfig":
        """Refuse a window at which ONE cap game (`1/window_games`) already exceeds the rate."""
        if 1.0 / self.window_games > self.rate:
            raise ValueError(
                f"train.ply_cap_abort.window_games ({self.window_games}) is too small for rate "
                f"{self.rate}: ONE cap game in the window reads 1/{self.window_games} = "
                f"{1.0 / self.window_games}, above the rate, and would fire the hard abort. "
                f"Raise window_games above {int(1.0 / self.rate)} or raise the rate"
            )
        return self


class HeldoutGapConfig(StrictModel):
    """The in-run held-out witness (v37): every `interval` steps a forward-only loss over a FROZEN slice of `ring` (`batches` samples under `seed`, re-seeded per read), reported against the train loss since the last read; `ring_sha256` pins the file; `null` is the explicit OFF."""

    ring: str = Field(min_length=1)
    ring_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batches: int = Field(ge=1)
    seed: int
    interval: int = Field(ge=1)


class LrCycleConfig(StrictModel):
    """A declared second cosine cycle: from trainer step `start_step` the LR is `lr` cosine-annealed to `eta_min` over `t_max` steps, then `eta_min`; config-owned, so a resume runs it whatever the checkpoint's scheduler state says."""

    start_step: int = Field(ge=1)
    lr: float = Field(gt=0, allow_inf_nan=False)
    eta_min: float = Field(ge=0, allow_inf_nan=False)
    t_max: int = Field(ge=1)

    @model_validator(mode="after")
    def _the_cycle_anneals_downward(self) -> "LrCycleConfig":
        """Refuse a floor above the cycle's peak, which would anneal the LR upward."""
        if self.eta_min > self.lr:
            raise ValueError(f"train.lr_cycle.eta_min ({self.eta_min}) is above its lr ({self.lr})")
        return self


class TrainConfig(StrictModel):
    """Training hyperparameters. Every field REQUIRED — no terminal default anywhere in this
    class; the minted value in each `configs/*.yaml` is the sole default authority.
    """

    # optimizer / schedule
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    # A CONFIG FACT, not a CLI flag, so a preflight cannot point a CUDA-minted run at the CPU;
    # CLOSED vocabulary: device indices (`cuda:1`) are unrepresentable.
    device: Literal["cpu", "cuda"]
    lr_schedule: Literal["cosine", "none"]
    # The LR anneal's horizon (the floored cosine's T_max), never a run length: `max_train_steps` is that.
    scheduler_t_max: int = Field(ge=1)
    eta_min: float = Field(ge=0)
    # The second cosine cycle's ARMING SURFACE, the `default=...` idiom: `null` is OFF (the floored cosine alone).
    lr_cycle: LrCycleConfig | None = Field(default=...)
    checkpoint_interval: int = Field(ge=0)
    # Continuous actor-sync cadence in coordinator training steps. `ge=1` means NO disabled
    # value exists. Resolved only by `mantis.config.resolve.actor_sync`.
    actor_sync_cadence_steps: int = Field(ge=1)
    # The EMA lever's arming block, REQUIRED so every config states its posture explicitly.
    ema: EmaConfig
    # The RUN-LENGTH authority (`resolve_max_train_steps` -> `stop_step`), not the LR horizon;
    # ABSOLUTE, so a run resumed past it stops at once, which looks like a frozen actor.
    max_train_steps: int = Field(ge=1)
    # The draw-rate abort's ARMING SURFACE: `None` is EXPLICITLY OFF, with no second boolean
    # authority; `default=...` is the no-terminal-default idiom, so absence names the key.
    draw_rate_abort: DrawRateAbortConfig | None = Field(default=...)
    # The ply-cap attractor halt's ARMING SURFACE, the same idiom.
    ply_cap_abort: PlyCapAbortConfig | None = Field(default=...)
    # The held-out gap witness's ARMING SURFACE, the same idiom: `null` is OFF.
    heldout_gap: HeldoutGapConfig | None = Field(default=...)

    # The step-coordinator knobs, FLAT `train.*` keys (no block named after a dataclass).
    # `eval_interval` — the promotion cadence. `ge=1`: `<= 0` silently kills promotion; the
    # off posture is the typed `eval_enabled: false`.
    eval_interval: int = Field(ge=1)
    # `log_interval` — NARRATION ONLY; the gates run on `monitor.gate_interval`. `ge=1`: there
    # is no legitimate "never narrate" posture.
    log_interval: int = Field(ge=1)
    # `min_buf_size` — the warmup floor, below which `step()` returns `in_warmup`. `ge=1`
    # because a floor of 0 means "train on an empty buffer", which the sampler cannot satisfy.
    min_buf_size: int = Field(ge=1)
    # `replay_capacity` — the replay window, i.e. the distribution the learner trains on.
    replay_capacity: int = Field(ge=1)
    # `training_steps_per_game` — the sample-reuse ratio. `gt=0` because off the fill ramp `_steps_budget`
    # floors its result at 1, so `0` means "one step per round" while reading as an off switch.
    training_steps_per_game: float = Field(gt=0)
    # The ratio scaled by the ring's fill (rows/capacity): a row inserted while it fills is drawn as a steady row is.
    training_steps_fill_ramp: bool
    # `max_train_burst` — the ceiling of that budget. `ge=1` because the ceiling is the `min(...)`'s
    # outer term, so `0` clamps the budget to 0 and stops the learner silently.
    max_train_burst: int = Field(ge=1)
    # `batch_size` — the training batch, AUTHORED HERE AND NOWHERE ELSE.
    batch_size: int = Field(ge=1)
    # `microbatch_caps` — the GRAPH step's memory bound. ARCH-SCOPED: `None` is the key's
    # ABSENCE, read off `model_fields_set`, so an explicit `null` is refused on a graph config.
    microbatch_caps: MicrobatchCapsConfig | None = None
    # `augment` — 12-fold hex-symmetry augmentation of every sampled batch. It multiplies the
    # effective dataset, so two runs that differ only here are not comparable.
    augment: bool
    # `terminal_eval_enabled` — whether close-out runs a terminal eval round, i.e. whether the
    # run gets its LAST promotion opportunity.
    terminal_eval_enabled: bool
    # `selfplay_stall_timeout_sec` — the ALWAYS-ARMED stall watchdog's budget (`<= 0` would
    # silently disable its fire); an OPERATIONAL CONSTANT, hence its default.
    selfplay_stall_timeout_sec: float = Field(default=1800.0, gt=0, allow_inf_nan=False)

    # loss selection + targets: which policy loss the trainer applies follows from
    # `policy_target`, itself pinned to `search.kind`; a game with no winner trains no value.
    policy_target: Literal["raw_visit_distribution", "completed_improved_policy"]
    # The value target: `w·v_search + (1−w)·z` on a row carrying the ring's search value (Σπ′·completedQ), z elsewhere.
    value_target_search_weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    # The value loss's keep probability per value-valid row, re-drawn every step; 0 is off (keeping all is not a
    # second spelling of off, hence `lt=1`).
    value_mask_redraw_p: float = Field(ge=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _a_cycle_restarts_a_cosine_inside_the_run(self) -> "TrainConfig":
        """Refuse a second cycle on a schedule that has no cosine, or one that starts at or past the run's last step."""
        if self.lr_cycle is None:
            return self
        if self.lr_schedule != "cosine":
            raise ValueError(f"train.lr_cycle restarts a cosine, but train.lr_schedule is {self.lr_schedule!r}")
        if self.lr_cycle.start_step >= self.max_train_steps:
            raise ValueError(
                f"train.lr_cycle.start_step ({self.lr_cycle.start_step}) is not below train.max_train_steps "
                f"({self.max_train_steps}), so the cycle would never run")
        return self
