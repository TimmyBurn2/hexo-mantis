# >300 justify (R8), stated at this file's MEASURED size of 455 lines (was 430 at K-B, 454 at
# DSV-2, which added R94's ratification grounds; DSV-3 corrects the term count). WPMINT Phase K-B
# authors the 19 step-coordinator knobs `CARD-COORD-KNOBS` (R78/R80) owned, and roughly four
# fifths of the added length is the per-field GROUNDS the house style requires: what the bound
# is a bound ON (the mechanism's own range, never policy), which defect it makes
# inexpressible, and the ONE resolver that reads it. `DrawRateAbortConfig`'s docstring set that
# precedent and it is the reason MF-1's open upper half was catchable at all. Splitting the
# class would put a field and its grounds on opposite sides of an import, and splitting the two
# classes apart would separate `train.draw_rate_abort.consec` from the three terms R80 says
# "travel together". The executable content is 4 validators and ~50 field lines.
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
    terms"*. It has four INSEPARABLE components (`threshold`, `min_step`, `N_pool_min`,
    `consec`), which is why they are a nested block and not four flat `X | None` keys: four
    independent keys give four authorities over one
    fact and can disagree in ways no predicate can adjudicate (`threshold: 0.25,
    N_pool_min: null` is neither armed nor disarmed). The block makes disagreement
    unrepresentable — the four arrive together or not at all.

    ALL FOUR ARE RUN-SCOPED CONSTANTS (R82/R85/R92), pre-registered at mint prereg, which
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
      `_draw_rate_evidence_bar_within_configured_capacity`, NOT by an `le=` here: the ceiling is
      `DRAW_RATE_WINDOW * selfplay.n_workers` (measured — `Sum(len(dq))` saturates there),
      which spans two sections and so cannot live on this field. That cross-validator is
      what re-establishes the load-bearing bound `min_samples: le=DRAW_RATE_WINDOW` carried;
      for every config on this tree (`n_workers: 1`) it evaluates to the same number.

    * `consec` — WPMINT Phase K-B (R78/R80, adjudication call K-b). The FOURTH term, and it
      is authored HERE rather than as a flat `train.*` key because R80's "the terms travel
      together" is the whole reason this block exists: a `consec` living outside it could be
      set on a config whose block is `null`, i.e. a term of an abort nobody armed. It was a
      `StepCoordinatorConfig` code-side default (`draw_rate_consec: int = 3`) until this
      phase; R92's prereg row NAMES `consec=3` among the values that stand, so authoring it
      preserves that value exactly and moves only WHO says it. `ge=1` because `consec` counts
      consecutive OBSERVATIONS and a rule that needs zero of them is not a rule; there is no
      "disabled" value, the same posture `min_step` takes. NO upper bound is invented: the
      only honest one would be the abort history's own depth (`_GATE_HISTORY_DEPTH = 32` in
      `train/coordinator/step.py`), which is a runtime constant of a module `mantis.config`
      must not import, and a wrong guess would either reject a legal prereg or admit an
      unreachable one. DISCLOSED: a `consec` above that depth makes the abort unfireable
      while it audits ARMED — the fifth face of "armed in the config, absent in effect", left
      OPEN and written down rather than closed with an invented number (R84's class).
      DISCLOSED (WPMINT DS-VERIFY): `consec` counts consecutive CHECKS at a stride of
      `train.log_interval` train steps, so at the shipped 1000 three samples span 2000 steps —
      it is a sustained-ness bar in steps, not in checks, and it does NOT delay the first
      fire.

    Read by exactly one path: `mantis.config.resolve.draw_rate.resolve_draw_rate_abort`.
    """

    threshold: float = Field(gt=0, le=1)
    min_step: int = Field(ge=1)
    N_pool_min: int = Field(ge=1)
    consec: int = Field(ge=1)

    @model_validator(mode="after")
    def _one_drawn_game_cannot_fire_the_abort(self) -> "DrawRateAbortConfig":
        """ADJ-14's own defect, re-expressed on R92's statistic (WPMINT DR-9's class).

        `ge=1` alone admits `N_pool_min` values at which a SINGLE drawn game meets the
        threshold: the pooled rate's smallest non-zero value at the bar is `1/N_pool_min`,
        so at `N_pool_min=4` with `threshold=0.25` one drawn game in four fires the hard
        abort. That is the one-game saturation R80 ordered closed, on a new axis. The rule
        is derived entirely from values already in this block — no invented number — and
        run5 satisfies it by a factor of **12.5** (0.02 < 0.25), i.e. ~1.1 orders.
        (WPMINT DSV-2 confirmatory pass, DSV2-1: this sentence read "three orders of margin"
        from `75bdaf0` until 2026-07-29. That was arithmetically FALSE — `0.25 / 0.02 = 12.5`
        — and it CONTRADICTED the R94 grounds three lines below, which state the separate and
        correct 80x figure for floor-vs-healthy. Two different ratios had been conflated:
        floor-vs-THRESHOLD is 12.5x, floor-vs-HEALTHY is 80x.)

        RATIFIED BY **R94** (ADJ-21), with the operator's grounds recorded HERE so the next
        reader finds them at the constraint rather than in a register they may not open.
        This validator was added unbidden by Phase DS — no ruling had asked for it — so it
        was queued rather than assumed, and R94 ratified it on measured grounds:

        * **healthy draw rate ≈ 0.00025** (draws arise only from ply-cap truncation);
        * the induced threshold floor is `1/N_pool_min` = **0.02** at run5's `N_pool_min=50`,
          i.e. **≈ 80x above healthy** — far enough above the healthy rate that no
          legitimate arming posture is lost, and near enough that it is not an arbitrary wall.

        WHAT IT COSTS, STATED PLAINLY: this makes `threshold <= 1/N_pool_min` INEXPRESSIBLE
        (measured on a one-worker pool: 0.25 and 0.03 accepted; 0.02, 0.019 and 1e-300
        rejected). R94's escape hatch is deliberately expensive — **a sub-floor threshold
        requires a `docs(design)` schema amendment, not a config edit**, which is R9's shape
        applied to a bound. Side effect the operator accepted: this closes MF-1's long-
        DISCLOSED `1e-300` hair-trigger residual at the type, which R83 could not close and
        R92 did not aim at.
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
    """One step of `train.replay_capacity_schedule` — "at train step S, grow the replay
    buffer to C" (WPMINT Phase K-B).

    A nested block rather than two parallel lists for `DrawRateAbortConfig`'s reason: a
    `steps: []` and a `capacities: []` that disagree in length is a state no predicate can
    adjudicate. Under NIT-3 a `list[SubModel]` field is ONE consumer-registry leaf
    (`eval.ladder.rungs` is the precedent), so the two inner names are covered by the
    schedule's own entry rather than by two of their own.

    `capacity` is `ge=1` because a buffer that holds nothing is not a buffer, and the walker
    (`train/coordinator/step.py` D1) only ever GROWS — `if new_cap > self.buffer.capacity` —
    so a stage below the current capacity is silently inert rather than a shrink.
    """

    step: int = Field(ge=0)
    capacity: int = Field(ge=1)


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

    # ── the step-coordinator knobs (WPMINT Phase K-B — CARD-COORD-KNOBS, R78 as clarified
    # by R80) ──────────────────────────────────────────────────────────────────────────────
    # These 19 were `_step_coordinator_config`'s builder literals and `StepCoordinatorConfig`
    # terminal defaults — unauthored code-side numbers that decided what the run IS while the
    # minted config said nothing about them. They are FLAT `train.*` keys and NOT a
    # `train.coordinator` / `train.step_coordinator` block: naming a config block after a
    # dataclass was RULED AGAINST at WPAX Phase D (§2, pinned verbatim by
    # `tests/config/test_drawrate_arming_authority.py`), and these are training
    # hyperparameters by nature — the coordinator is the object that reads them, not the fact
    # they express. Read by ONE path,
    # `mantis.config.resolve.coordinator.resolve_coordinator_knobs`.
    #
    # SIX SIBLING FIELDS ARE DELETED RATHER THAN AUTHORED (adjudication call K-a):
    # `composition_interval`, `value_probe_interval`, `soft_ew_threshold`, `soft_ew_min_pts`,
    # `instrumentation_enabled` and `bot_corpus_path` had NO reader in `src/` — re-verified at
    # HEAD by grep AND by recording every attribute read on a live `StepCoordinatorConfig`
    # across the whole test tier. Typing a dead knob into this class would CREATE the R1 /
    # LAW-08 violation the class exists to prevent.
    #
    # `eval_interval` — the promotion-decision cadence, in coordinator train steps. `ge=1`
    # for `actor_sync_cadence_steps`' reason: at `<= 0` `_maybe_kick_eval` returns
    # `(False, False)` on every step and `promotion_capable_rounds` returns `[]`, so the
    # ENTIRE eval/promotion pipeline is off while nothing says so. No disabled value exists;
    # `compose_run(eval_enabled=False)` is the one place eval is turned off, and it is a
    # parameter, not a number that happens to disable (R79(1)).
    eval_interval: int = Field(ge=1)
    # `log_interval` — the boundary at which the run emits its payload events, runs the WARN
    # rules, runs BOTH live hard-abort gates and publishes the LAW-18 `monitor_gates` summary
    # (`_run_log_interval`). `ge=1` closes WPMINT DR-7, which MEASURED that `log_interval <= 0`
    # kills the whole hard-abort family AND the `monitor_gates` event together
    # (`checks=0, fires=0, skips=0`) while gate 12 still audits the draw-rate row ARMED — a
    # fifth "armed in the config, absent in effect" axis, and the one where the instrument
    # that would make it visible is switched off by the same knob. There is no legitimate
    # "never log" posture, so the schema cannot express one.
    log_interval: int = Field(ge=1)
    # `buffer_save_interval` — the REPLAY-BUFFER save cadence (`_try_save_buffer`), NOT the
    # trainer checkpoint cadence. RENAMED from the dataclass field `checkpoint_interval` for
    # exactly that reason: `train.checkpoint_interval` above is a different, already-authored
    # fact (the trainer's periodic save), and two config keys spelled the same would be the
    # duplicated-authority class R1 exists to kill — the census that found this knob flagged
    # the collision as its headline hazard. `ge=0` because `0` is REAL and is what every
    # committed config ships: no cadence save, with close-out and the shutdown-signal path
    # still saving. That is a persistence CADENCE, not a guard, so a zero here disables
    # nothing LAW-16 requires.
    buffer_save_interval: int = Field(ge=0)
    # `min_buf_size` — the warmup floor: below it `step()` returns `in_warmup` and the learner
    # sees nothing. `ge=1` because a floor of 0 means "train on an empty buffer", which the
    # sampler cannot satisfy; `1` (the shipped value) already means "train on the first
    # sample that lands".
    min_buf_size: int = Field(ge=1)
    # `replay_capacity` — the replay window, i.e. the sample distribution the learner trains
    # on. RENAMED from the dataclass field `capacity`: a bare `train.capacity` names nothing
    # on its own, and the dataclass field is disambiguated by its class where a config key is
    # not. `ge=1` for `ReplayCapacityStage.capacity`'s reason.
    replay_capacity: int = Field(ge=1)
    # `replay_capacity_schedule` — the step-keyed ramp over `replay_capacity` (D1 in
    # `coordinator/step.py`). `[]` is the shipped posture and means "no ramp"; it is not a
    # disguised off switch, because `replay_capacity` alone is a complete answer.
    # `_schedule_idx` advances monotonically and never rewinds, so a schedule whose steps are
    # not strictly increasing would silently skip stages — `_stages_are_strictly_increasing`
    # below makes that unrepresentable rather than merely unlikely.
    replay_capacity_schedule: list[ReplayCapacityStage]
    # `training_steps_per_game` — the sample-reuse ratio, the core AlphaZero learner/actor
    # throughput axis: `_steps_budget(new_games, this, max_train_burst)`. `gt=0` because
    # `_steps_budget` floors its own result at 1, so a `0` here does NOT mean "no training" —
    # it means "one step per round" while reading as an off switch. A value whose effect
    # contradicts its spelling is the defect, not the number.
    training_steps_per_game: float = Field(gt=0)
    # `max_train_burst` — the ceiling of that same budget. `ge=1` for the same reason: the
    # `max(1, ...)` floor is INSIDE the `min(...)`, so `max_train_burst=0` really does clamp
    # the budget to 0 and stops the learner permanently while self-play keeps producing. That
    # is a silent stall, not a configuration.
    max_train_burst: int = Field(ge=1)
    # `batch_size` — the training batch. AUTHORED HERE AND NOWHERE ELSE. Until this phase
    # `coordinator/step.py::_run_training_step` read
    # `train_cfg.get("batch_size", full_config.get("batch_size", 256))`, and WPMINT Phase K-A
    # MEASURED that `compose_run` passes `train_cfg={}` and a `full_config` whose top-level
    # keys are the RunConfig SECTIONS — so both lookups missed and the production batch size
    # was unconditionally the literal `256`, while `StepCoordinatorConfig.batch_size` (the
    # builder's `8`) sat beside it unread. The minted value is therefore **256**, not 8: this
    # phase moves the authority, never the number. `ge=1` — a batch of zero samples is not a
    # batch.
    batch_size: int = Field(ge=1)
    # `augment` — 12-fold hex-symmetry augmentation of every sampled batch. A first-order
    # "what is this run" fact: it multiplies the effective dataset, so two runs that differ
    # only here are not comparable.
    augment: bool
    # `recency_weight` — the fraction of each batch drawn from the recency window
    # (`recent_frac` in the Rust sampler: `n_recent = round(batch_size * recent_frac)`,
    # clamped to `batch_size`). `ge=0, le=1` is that fraction's own range: above 1 the clamp
    # silently makes every value identical to 1, so the config could express a difference the
    # run cannot have.
    recency_weight: float = Field(ge=0, le=1)
    # `mixing_initial_w` / `mixing_min_w` / `mixing_decay_steps` — the pretrained-corpus
    # mixing schedule, `w_pre = max(min_w, initial_w * exp(-step / decay_steps))`
    # (`train/mixing.py`). The two weights are FRACTIONS OF A BATCH (`n_pre = ceil(w_pre *
    # (batch_size - n_bot))`), so `ge=0, le=1` is the quantity's own range, not policy;
    # `decay_steps` is a DIVISOR, so `gt=0` — at `0` the schedule raises ZeroDivisionError on
    # the first mixed step, which is a crash the schema can make unreachable.
    # `_mixing_floor_is_below_its_start` below closes the ordering.
    mixing_initial_w: float = Field(ge=0, le=1)
    mixing_min_w: float = Field(ge=0, le=1)
    mixing_decay_steps: float = Field(gt=0)
    # `hard_gn_threshold` / `hard_gn_min_steps` — the `grad_norm_hard_abort` gate
    # (`coordinator/step.py` D3): fire when `grad_norm > threshold` for `min_steps`
    # consecutive training steps. `gt=0` because a threshold of 0 fires on every finite step,
    # and `allow_inf_nan=False` because the gate guards on `math.isfinite(step_gn)` — an
    # infinite threshold is accepted, reads ARMED, and can never be met.
    # DISCLOSED, and deliberately NOT closed here: the shipped `1e9` is finite, positive and
    # unreachable by any real gradient norm, so this bound admits an effectively-DISARMED
    # abort. Closing it needs a CEILING, and no honest ceiling is derivable from this field
    # alone — inventing one is the class R84 refused. The ceiling that IS derivable lives one
    # layer up, in the armed-abort manifest, where WPMINT Phase K-B adds this gate as a
    # DEFERRED row whose `Mechanism.CONFIG_THRESHOLD_BELOW_CEILING` reads the ceiling off
    # `monitor.alert_grad_norm_max` — the value the operator already pre-registered for this
    # same quantity. DEFERRED prints loudly and does not gate, which is the honest posture for
    # a threshold nobody has pre-registered.
    hard_gn_threshold: float = Field(gt=0, allow_inf_nan=False)
    # `ge=1`: at `0` the gate fires the FIRST time the threshold is exceeded, which is the
    # opposite of the "sustained instability" the rule describes. DISCLOSED: a very large
    # value disarms the abort without touching the threshold — the same open upper half, and
    # it has no derivable ceiling either.
    hard_gn_min_steps: int = Field(ge=1)
    # `terminal_eval_enabled` — whether close-out runs a terminal eval round at all, i.e.
    # whether the run gets its LAST promotion opportunity (`coordinator/drain.py`
    # `run_terminal_eval`). A REGIME fact, and until WPMINT Phase K-A it had THREE authorities:
    # this key did not exist, the dataclass carried `= True`, and `drain.py` carried a
    # `getattr(cfg, "terminal_eval_enabled", True)` fallback. K-A retired the `getattr`; this
    # key retires the dataclass default, leaving one.
    terminal_eval_enabled: bool
    # `bot_batch_share` — the fraction of each training batch drawn from the bot corpus
    # (`n_bot = round(this * batch_size)`). `ge=0, le=1` is that fraction's range.
    # DISCLOSED: its sibling `bot_corpus_path` is one of the six DEAD fields deleted by this
    # phase — nothing in `src/` ever read a path to populate a bot buffer — so a non-zero
    # share here allocates batch slots from a `bot_buffer` that only an injecting caller can
    # supply. `0.0` is the shipped value and the only one the composition root can honour.
    bot_batch_share: float = Field(ge=0, le=1)
    # `selfplay_stall_timeout_sec` — the self-play stall watchdog's wall-clock budget
    # (2026-07-11 run2 eval-boundary wedge; `train/lifecycle/watchdog.py`). `gt=0` and
    # `allow_inf_nan=False` because LAW-16 says the stall watchdog is ALWAYS ARMED, while
    # `watchdog.py`'s own contract is that `timeout_sec <= 0` disables the fire AND THE
    # ARM-LOG STILL EMITS — a disarmed watchdog that logs as armed. The schema cannot express
    # that posture; `watchdog.py` keeps the arm for direct constructions, and this is the
    # `actor_sync_cadence_steps` idiom applied to the guard LAW-16 names.
    selfplay_stall_timeout_sec: float = Field(gt=0, allow_inf_nan=False)

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

    @model_validator(mode="after")
    def _mixing_floor_is_below_its_start(self) -> "TrainConfig":
        """`mixing_min_w > mixing_initial_w` is a schedule that never decays (WPMINT K-B).

        `w_pre = max(min_w, initial_w * exp(-step / decay_steps))`. With the floor above the
        start the `max` takes `min_w` at EVERY step, so `mixing_initial_w` and
        `mixing_decay_steps` both stop having any effect while still reading as the schedule's
        terms — three keys, one of which silently dominates the other two. Derived entirely
        from the arithmetic in `train/mixing.py`; no number is invented, and equality is legal
        (a flat schedule is a real posture, and it is what the shipped `0.0 / 0.0` is).
        """
        if self.mixing_min_w > self.mixing_initial_w:
            raise ValueError(
                f"train.mixing_min_w ({self.mixing_min_w}) is above train.mixing_initial_w "
                f"({self.mixing_initial_w}): w_pre = max(min_w, initial_w * exp(-step/"
                "decay_steps)), so the floor would win at every step and both "
                "train.mixing_initial_w and train.mixing_decay_steps would decide nothing "
                "while still reading as the schedule's terms"
            )
        return self

    @model_validator(mode="after")
    def _stages_are_strictly_increasing(self) -> "TrainConfig":
        """`train.replay_capacity_schedule` must be strictly increasing in `step` (K-B).

        The consumer walks it with a monotone cursor that never rewinds
        (`coordinator/step.py` D1: `while _schedule_idx < len(schedule) and _train_step >=
        schedule[_schedule_idx]["step"]`). Out-of-order stages are therefore not "applied in a
        different order" — the later ones are applied in the same pass at the earlier step, so
        the ramp the operator wrote is not the ramp the run performs. Equal steps are the same
        defect with two stages competing for one boundary. An empty schedule (every committed
        config) satisfies this vacuously.
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
