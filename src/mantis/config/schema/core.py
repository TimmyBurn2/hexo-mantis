"""Run-config schema (contract run-config-schema v1).

>300 justify (R8). This file carries the `RunConfig` root model and this schema's
cross-field `model_validator`s — the ones on `RunConfig` span SECTIONS
(train x selfplay, train x monitor, identity x selfplay), so they cannot live in any
section module.

Every model is strict: unknown key = hard error, missing key = hard error, silent scalar
coercions (str->int, float->int, bool->int) rejected, values immutable. NO code-side
defaults — a default lives in exactly one place: the schema field (repo_design §5). Identity
keys carry no terminal defaults at all; representation is the closed set {grid, graph}
(registry.toml + repo_design §3 ground truth — LAW-11).
"""
from typing import Literal

from pydantic import Field, field_validator, model_validator

from mantis.config.schema._base import StrictModel
from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.config.schema.selfplay import InferenceConfig, SelfplayConfig
from mantis.config.schema.train import TrainConfig
from mantis.encoding import EncodingRegistryError, lookup
from mantis.util.constants import DRAW_RATE_WINDOW

SCHEMA_VERSION = 1

#: RED-TEAM-2 F-RT2-1 (BLOCKER fix): an obviously-generous finite ceiling for a
#: timeout/grace-period float field whose value feeds `proc.join(timeout)` arithmetic
#: (pipeline.py isolation law 2) — `multiprocessing.Process.join` cannot accept
#: `float("inf")` (raises `OverflowError` deep inside `selectors.select()`), so a
#: floor-only bound (`gt=0`/`ge=0`) that admits `+inf` is not actually a bound for this
#: arithmetic. One day (86400.0s) is deliberately more generous than the
#: `StepCoordinatorConfig` drain-cap family (`monitor.drain.eval_final_drain_hard_cap_sec`
#: / `terminal_eval_hard_cap_sec`, minted 14400.0 = 4h; the `DEFAULT_*_HARD_CAP_SEC`
#: constants were DELETED at WPMINT K-A) since these two bound a single eval round /
#: kill-grace, never a whole drain budget — a named constant, never an inline literal (R1).
_EVAL_TIMEOUT_CEILING_SEC = 86400.0


class IdentityConfig(StrictModel):
    """Identity keys have no terminal defaults (repo_design §5): absent = error.

    ``representation`` is cross-checked against the encoding's registry representation at
    validation time (F1 runtime guard): a graph encoding declared ``representation: grid`` (or
    vice versa) is REJECTED at load, so the LAW-06 amp pin (resolve_amp_dtype reads this field)
    cannot be bypassed by a LAW-11-inconsistent config. Frozen sourced representation from the
    encoding spec, making disagreement structurally impossible; this guard restores that invariant.
    """

    encoding: str = Field(min_length=1)
    representation: Literal["grid", "graph"]

    @model_validator(mode="after")
    def _representation_matches_registry(self) -> "IdentityConfig":
        try:
            spec = lookup(self.encoding)
        except EncodingRegistryError as exc:
            raise ValueError(str(exc)) from exc
        if self.representation != spec.representation:
            raise ValueError(
                f"identity.representation={self.representation!r} disagrees with the registry "
                f"representation {spec.representation!r} for encoding {self.encoding!r} "
                "(LAW-11 identity consistency; a mismatch would bypass the LAW-06 amp-dtype pin)."
            )
        return self


class LadderRung(StrictModel):
    """One opponent-ladder rung (design §c.1). `bot` is the resolver kind (closed set,
    WP11-A); `depth` is sealbot's fixed-depth bar (LAW-15), `opponent_sims` the
    kraken/strix opponent-side sims — exactly one of the two is meaningful per `bot`, both
    travel as `None` where inapplicable rather than a sentinel int (R1: no code default).

    RED-TEAM F2 (MAJOR): a fixed-depth bar of 0 or negative, an opponent-sims count of 0 or
    negative, or a rung that can never play a single game (`games_max < 1`) are each
    domain-nonsense — bounded here as named `Field` constraints (pydantic includes the
    field path in the raised error), never a silent clamp.
    """

    name: str = Field(min_length=1)
    bot: Literal["sealbot", "kraken", "strix", "random"]
    variant: str = Field(min_length=1)
    depth: int | None = Field(ge=1)
    opponent_sims: int | None = Field(ge=1)
    opening_book: str = Field(min_length=1)
    deploy_matched: bool
    games_max: int = Field(ge=1)


class GateConfig(StrictModel):
    """The run3 deploy-strength gate, knob-for-knob (LIVE knobs only).

    `screen_confirm_hi` is DELIBERATELY NOT PORTED (MUST-FIX 1): stored-but-never-read in
    run3 (the escalation decision is the single lower-bound test `wr_screen >=
    screen_confirm_lo`) — a dead schema key would violate LAW-08/R1; `extra="forbid"`
    rejects a minted `screen_confirm_hi`.

    RED-TEAM F2 (MAJOR) bounds: `promotion_winrate` is a win-rate fraction, domain `[0,1]`
    — an out-of-range value (e.g. `2.0`) previously loaded silently and then permanently
    and silently disabled promotion (`wr_confirm >= 2.0` can never be true), the exact
    silently-disabled-lever class R1/LAW-08 exist to kill. `screen_confirm_lo` is the same
    kind of win-rate-fraction threshold. The remaining fields are game/resample COUNTS
    (`screen_games`, `confirm_games`, `deploy_sims`, `bootstrap_resamples`,
    `min_distinct_per_pair`) or a round CADENCE (`stride`) — all must be >=1: a count of 0
    cannot produce a game/resample/distinct-pair to measure, and `stride=0` would divide by
    zero at `round_idx % cfg.gate.stride` (pipeline.py `_build_round_spec`). `seed_base` is
    an RNG seed with no domain restriction (any int is a valid seed).
    """

    stride: int = Field(ge=1)
    screen_games: int = Field(ge=1)
    confirm_games: int = Field(ge=1)
    promotion_winrate: float = Field(ge=0, le=1)
    screen_confirm_lo: float = Field(ge=0, le=1)
    deploy_sims: int = Field(ge=1)
    opening_book: str = Field(min_length=1)
    bootstrap_resamples: int = Field(ge=1)
    min_distinct_per_pair: int = Field(ge=1)
    seed_base: int


class LadderConfig(StrictModel):
    """The opponent-ladder schema (STATE §5): ordered rungs + every scheduling/hysteresis
    threshold as a named field — never a code literal (rule 4).

    RED-TEAM F2 (MAJOR) bounds: `bootstrap_ci_level` is a confidence LEVEL, mathematical
    domain strictly `(0,1)` — `pair_bootstrap_wr_ci` computes `alpha = (1 - level) / 2` and
    calls `np.quantile(..., alpha)`/`np.quantile(..., 1 - alpha)`; a level outside `(0,1)`
    either raises deep inside a worker subprocess (`level > 1` -> negative `alpha` ->
    `np.quantile` raises at runtime, not at config load) or silently returns a
    statistically-inverted-but-plausible-looking CI (`level < 0`), previously undetected at
    the config boundary R1 promises. `round_games`/`bootstrap_resamples` are per-round
    COUNTS (>=1: zero games or zero resamples measures nothing); `min_games_per_active_rung`
    is a per-rung FLOOR that is legitimately allowed to be 0 (no floor); `calibration_games`
    is the saturated-rung calibration COUNT — STATE §5 requires calibration "never fully
    retired", so it must be >=1 (0 would silently retire a graduated rung's Elo-scale
    anchor forever); `bt_prior_games` is the BT fit's regularizer pseudo-count, domain
    `>=0` (a negative prior is not a pseudo-count at all).

    RED-TEAM-2 F-RT2-1 (BLOCKER) sweep: `bt_prior_games`'s floor-only bound (`ge=0`) also
    silently admitted `float("inf")` — traced downstream (bt.py `fit_bt`): an `inf` prior
    added to the win matrix produces `inf`/`inf` MM-iteration ratios (NaN), a degenerate
    but non-crashing corruption of every downstream BT rating and `p_hat`/scheduling
    value. `allow_inf_nan=False` closes it the same way as the two timeout fields above
    (no finite ceiling needed here — this field is not `proc.join()` timeout arithmetic,
    so any finite value is domain-legal). `graduation_wr_lower_ci` /
    `activation_wr_lower_ci` / `graduation_consec_rounds` / `calibration_every_k_rounds`
    already carry their own bounds via `_validate_ladder` below (unchanged here).
    `bootstrap_seed` is an RNG seed with no domain restriction.
    """

    rungs: list[LadderRung]
    round_games: int = Field(ge=1)
    min_games_per_active_rung: int = Field(ge=0)
    graduation_wr_lower_ci: float
    graduation_consec_rounds: int
    activation_wr_lower_ci: float
    calibration_every_k_rounds: int
    calibration_games: int = Field(ge=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_ci_level: float = Field(gt=0, lt=1)
    bt_prior_games: float = Field(ge=0, allow_inf_nan=False)
    bootstrap_seed: int

    @model_validator(mode="after")
    def _validate_ladder(self) -> "LadderConfig":
        if not self.rungs:
            raise ValueError("eval.ladder.rungs must be non-empty")
        names = [r.name for r in self.rungs]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"eval.ladder.rungs: rung 'name' must be unique; duplicate name(s): {dupes}"
            )
        if not (0 < self.activation_wr_lower_ci <= self.graduation_wr_lower_ci < 1):
            raise ValueError(
                "eval.ladder: thresholds must satisfy "
                "0 < activation_wr_lower_ci <= graduation_wr_lower_ci < 1 "
                f"(got activation_wr_lower_ci={self.activation_wr_lower_ci}, "
                f"graduation_wr_lower_ci={self.graduation_wr_lower_ci})"
            )
        if self.graduation_consec_rounds < 1:
            raise ValueError("eval.ladder.graduation_consec_rounds must be >= 1")
        if self.calibration_every_k_rounds < 1:
            raise ValueError("eval.ladder.calibration_every_k_rounds must be >= 1")
        return self


class PlyCapAdjudicationConfig(StrictModel):
    """How a PLY-CAPPED eval game is resolved — ONE block, ONE fact (F-R-P2B-5 companion).

    The fact under single authority is *"is ply-cap adjudication armed, and on what
    criterion"*. `null` is ARMED=NO, explicitly, and it is the posture every shipped config
    takes: with the block absent the arena's legacy arm runs and a capped game is a draw,
    byte-for-byte the pre-existing behaviour. A block is ARMED=YES on its own terms. The two
    are disjoint TYPES rather than two regions of one range, the shape `DrawRateAbortConfig`
    established under R79 — a boolean beside the criterion could contradict it, a criterion
    with no margin could not be evaluated, so the terms arrive together or not at all.

    WHY THE FACT NEEDS AN AUTHORITY AT ALL. The eval arena caps a game at
    `mantis.arena.match._DEFAULT_MAX_PLIES` because the board is unbounded, and until now
    every capped game scored `"draw"` — the same label a finished, genuinely balanced game
    gets. On the live shakedown burn that collapse consumed the whole outcome channel:
    `draw_rate` measured 1.0 with `avg_game_length` at the 128-move cap, so at early strength
    the promotion instrument's only reading was a constant. This block does not decide what
    the replacement reading should be; it makes the decision REPRESENTABLE and leaves the
    values to mint prereg.

    * `criterion` — the CLOSED set `mantis.arena.adjudicate.PLY_CAP_CRITERIA`, and the schema
      `Literal` is the same two names so a criterion the adjudicator cannot implement is a
      config-load error rather than a round-time refusal. `longest_run_margin` is a property
      of the placed stones and is seat-neutral; `immediate_win_margin` counts completing
      moves and is NOT seat-neutral (at the cap one side never moves again). That difference
      is the choice, and it is the operator's.
    * `min_margin` — `ge=1` bounds the MECHANISM, not the policy: the margin is a signed
      difference between two equally-measured sides, so a margin of 0 means "measured equal"
      and a rule that awarded a game on it would not be a margin rule at all. No upper bound
      is invented — the reachable maximum depends on the criterion and the position, and a
      ceiling this layer cannot derive is a number it may not own (R84's class). The VALUE is
      a mint-prereg row: this schema names no default and there is none anywhere in code (R1).

    Read by exactly one path: `mantis.config.resolve.eval_posture.resolve_ply_cap_adjudication`.
    """

    criterion: Literal["longest_run_margin", "immediate_win_margin"]
    min_margin: int = Field(ge=1)


class StrengthFloorConfig(StrictModel):
    """The cheap probe that gates the EXPENSIVE ladder — ONE block, ONE fact (F-R-P2B-5).

    The fact under single authority is *"is the ladder gated on a strength floor, and on what
    terms"*. `null` is ARMED=NO and is the posture every shipped config takes: no probe is
    played, no phase is reordered, and the round runs the gate block -> rungs -> random floor
    exactly as before. The R79 disjoint-types shape again, for the same reason.

    THE MEASURED PROBLEM THIS EXISTS FOR. A terminal eval round at step 33 spent its full
    `monitor.drain.terminal_eval_hard_cap_sec` budget and completed ZERO of its spec'd games
    while the worker was healthy and computing throughout (F-R-P2B-5). The round's own
    ordering is why nothing survived: the gate block runs FIRST and is the most expensive
    phase in it, so a candidate too weak to finish games burns the whole budget before the
    cheapest opponent it has is ever reached. An armed floor plays that cheapest opponent
    first, on a bounded number of games, and refuses the rest of the round with a truthful
    event rather than a four-hour silence.

    THE TERMS TRAVEL TOGETHER (R80's reason, on this fact). A probe size without a bar
    measures nothing; a bar without a probe size is a bar on an unknown n; and the two bars
    below answer DIFFERENT halves of the measurement, so neither substitutes for the other.

    * `probe_games` — `ge=1`, the probe's whole budget, denominated in GAMES rather than
      seconds on purpose. LAW-15 is explicit that a strength bar must be a reproducible
      instrument and names the incident where a wall-clock bar flipped a verdict a
      fixed-depth bar reversed; a timeout field here would re-create exactly that, so there
      is none. Termination is guaranteed structurally instead — every arena game ends at
      the ply cap if it ends no earlier.
    * `min_decisive_rate` — the fraction of probe games that must end DECISIVELY (a real win
      or loss, not a ply-cap non-result). This is the axis the burn actually measured: at
      `draw_rate` 1.0 every game was a cap non-result, so a WR bar alone would have read a
      healthy-looking 0.5 off a round with no information in it. Domain `[0,1]` because it
      is a fraction of the probe.
    * `min_winrate` — the draw-aware win rate against the same cheapest opponent, domain
      `[0,1]`. A value of `0.0` makes this conjunct vacuous, and that is a LEGAL and
      EXPLICIT posture rather than a silently-disabled lever: the operator who wants the
      decisiveness bar alone says so in the config, where it is readable, instead of the
      absence of a key saying it for them.

    Read by exactly one path: `mantis.config.resolve.eval_posture.resolve_strength_floor`.
    """

    probe_games: int = Field(ge=1)
    min_decisive_rate: float = Field(ge=0, le=1)
    min_winrate: float = Field(ge=0, le=1)


class EvalConfig(StrictModel):
    """Eval opponent simulation counts (resolve_eval_model_sims reads these — no code default).

    RED-TEAM F2 (MAJOR) bounds: a `*_model_sims` count of 0 or negative cannot run a single
    MCTS simulation, so all four are `>=1` (the RED_TEAM-reproduced silent-load of
    `random_model_sims=-5` is now a named `ValidationError` at config load, not a downstream
    surprise). `random_floor_games` is legitimately mintable at `0` (A-2: parity mints the
    floor DISABLED) so its bound is `>=0`, not `>=1`. `round_timeout_sec` bounds the
    isolation-law join/escalation arithmetic (pipeline.py's poller compares elapsed wall
    time against it) and must be strictly positive; `worker_kill_grace_sec` is a grace
    period, domain `>=0`.

    RED-TEAM-2 F-RT2-1 (BLOCKER): a floor-only bound (`gt=0`/`ge=0`) admits `float("inf")`
    (mathematically `inf > 0` and `inf >= 0`), which previously loaded SILENTLY via a
    genuine YAML `.inf` literal and reproduced F1's exact silent-poller-death failure mode
    — `_escalate_and_finalize` (pipeline.py) calls a real `multiprocessing.Process.join`
    with this value, which raises an uncaught `OverflowError` for a non-finite timeout.
    Both fields now carry `allow_inf_nan=False` (rejects `inf`/`-inf`/`nan` with a named
    pydantic `finite_number` error) PLUS a finite ceiling
    (`_EVAL_TIMEOUT_CEILING_SEC`, see above) — the isolation-law arithmetic this WP's own
    docstring names as the reason these fields exist can no longer be handed a value it
    cannot execute.
    """

    random_model_sims: int = Field(ge=1)
    sealbot_model_sims: int = Field(ge=1)
    kraken_model_sims: int = Field(ge=1)
    strix_model_sims: int = Field(ge=1)
    random_floor_games: int = Field(ge=0)
    worker_device: Literal["cuda", "cpu"]
    round_timeout_sec: float = Field(gt=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    worker_kill_grace_sec: float = Field(ge=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    #: `default=...` is this schema's own no-terminal-default idiom (the shape
    #: `train.scheduler_t_max` / `train.draw_rate_abort` already carry): the key is REQUIRED
    #: and an absent one is an error naming it, while `None` is a real, explicit posture the
    #: config must state. Both blocks below ship `null` in every committed config, which is
    #: the identity value — the run behaves exactly as it did before the blocks existed —
    #: and arming either one is a mint event, never an IMPL edit (R1).
    ply_cap_adjudication: PlyCapAdjudicationConfig | None = Field(default=...)
    strength_floor: StrengthFloorConfig | None = Field(default=...)
    gate: GateConfig
    ladder: LadderConfig


class RunConfig(StrictModel):
    """Top-level run config: explicit, complete, schema_version-pinned.

    ``SelfplayConfig``/``InferenceConfig`` live in ``schema/selfplay.py`` and
    ``MonitorSchemaConfig``/``DrainCapsConfig`` in ``schema/monitor.py`` (§10 file-size
    split) — there is deliberately NO ``legal_move_radius``/``legal_move_radius_schedule``
    field anywhere on this tree (DESIGN_P2.md §5, shape (ii), SC-A4: the encoding registry
    alone is the radius authority; ``RadiusStage`` and its resolver module
    (``mantis.config.resolve.radius``) are retired, not merely unused).
    """

    schema_version: int
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    seed: int
    # WPMAIN / R120: the run's eval posture is a CONFIG FACT, not a `compose_run` parameter.
    # It used to be `compose_run(eval_enabled: bool = True)` — a code-side default for a
    # decision the minted config is supposed to author (R1), and a forcing route the
    # preflight child could have used to boot a posture run5 never declared (R64's "may
    # never force False" was enforced by a COMMENT). The parameter is deleted, so no caller
    # anywhere can override this; `mantis.run.compose_run` is the one live consumer, reading
    # it in both branches (the `wired_sources` declaration and the eval-pipeline build).
    # TOP-LEVEL rather than under `eval`, because it is a root-composition fact spanning the
    # eval and monitor wired-source surfaces — `train.terminal_eval_enabled` stays the
    # distinct close-out knob it already is.
    eval_enabled: bool
    identity: IdentityConfig
    eval: EvalConfig
    train: TrainConfig
    selfplay: SelfplayConfig
    inference: InferenceConfig
    monitor: MonitorSchemaConfig

    @field_validator("schema_version")
    @classmethod
    def _pin_schema_version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {v}")
        return v

    @model_validator(mode="after")
    def _policy_target_completed_q_consistency(self) -> "RunConfig":
        # ADJUDICATION_QUEUE closing note / DESIGN_P2.md §2: `train.policy_target` is
        # produced in self-play (gated by `selfplay.completed_q_values`) and the SAME
        # decision also selects the train-side loss (`train.completed_q_values`). One
        # decision, two consumers across two seams — this cross-section validator keeps
        # them from becoming two independently-editable knobs kept in sync only by
        # convention. Inert at mint time (all three sides pin to the single live combo);
        # fires the day one flag flips without the others.
        raw = self.train.policy_target == "raw_visit_distribution"
        train_off = not self.train.completed_q_values
        selfplay_off = not self.selfplay.completed_q_values
        if not (raw == train_off == selfplay_off):
            raise ValueError(
                "policy_target/completed_q_values disagree across sections: "
                f"train.policy_target={self.train.policy_target!r}, "
                f"train.completed_q_values={self.train.completed_q_values!r}, "
                f"selfplay.completed_q_values={self.selfplay.completed_q_values!r} — "
                "all three must agree (one decision, not three independently-editable "
                "knobs)."
            )
        return self

    @model_validator(mode="after")
    def _actor_lag_threshold_exceeds_sync_cadence(self) -> "RunConfig":
        # WP-UNFREEZE §5: the lag threshold (monitor section) and the sync cadence
        # (train section) share one invariant, so the check lives here, the ONE model
        # that sees both. Unconditional: healthy configs satisfy it trivially, and a
        # disarmed-but-nonsensical pair would spam exceed-events.
        if self.monitor.actor_lag_threshold_steps <= self.train.actor_sync_cadence_steps:
            raise ValueError(
                "actor_lag_threshold_steps (N) must exceed train.actor_sync_cadence_steps: "
                "a threshold at or below the sync cadence fires under healthy operation"
            )
        return self

    @model_validator(mode="after")
    def _actor_sync_knobs_fit_inside_the_run(self) -> "RunConfig":
        """Both step-clock knobs must be reachable within the run (RED-TEAM F-2).

        `ge=1` alone does not make "never sync" inexpressible. A cadence at or beyond
        `max_train_steps` lets the actor take its single unconditional first sync and then
        freeze for the entire run — run3's failure, expressed in a config that validated
        clean. And because the threshold must exceed the cadence, such a config also
        pushes the lag threshold out of reach, so the exit-45 invariant that exists to
        catch a frozen actor could never fire on one. The two knobs failed open together.

        Requiring both to be strictly inside `max_train_steps` is what actually makes
        "don't sync" unrepresentable, which is what R49 asks of the config surface.

        WPAX S-4 (F-C): the bound is anchored to `train.max_train_steps`, the RUN-LENGTH
        authority, not to `train.total_steps`, which is only the LR-scheduler horizon. On
        the proxy the bound could be satisfied and still be wrong — a 2000-step run with
        `total_steps: 1000000` blessed a cadence of 999 999.
        """
        total = self.train.max_train_steps
        if self.train.actor_sync_cadence_steps >= total:
            raise ValueError(
                f"train.actor_sync_cadence_steps "
                f"({self.train.actor_sync_cadence_steps}) must be < train.max_train_steps "
                f"({total}): a cadence the run never reaches means the actor syncs once "
                f"and then never again, which is the frozen actor this WP removed"
            )
        if self.monitor.actor_lag_threshold_steps >= total:
            raise ValueError(
                f"monitor.actor_lag_threshold_steps "
                f"({self.monitor.actor_lag_threshold_steps}) must be < train.max_train_steps "
                f"({total}): a threshold the run never reaches is an invariant that can "
                f"never fire — armed in the config, absent in effect"
            )
        # WPAX Phase D (R83, R71's class-fix law): the TWIN of the rule above, on the
        # draw-rate abort's own step floor. `min_step >= max_train_steps` is a guard the
        # run never passes, so the row audits ARMED while the abort can never fire — the
        # same defect, on a third axis (the first two are `threshold > 1.0` and
        # `N_pool_min > DRAW_RATE_WINDOW * selfplay.n_workers`, the first closed at the type
        # in schema/train.py and the second by the validator below).
        # `None` is the EXPLICIT disarmed posture and is skipped: there is no floor to
        # place inside the run when the operator has declined to arm the abort.
        block = self.train.draw_rate_abort
        if block is not None and block.min_step >= total:
            raise ValueError(
                f"train.draw_rate_abort.min_step ({block.min_step}) must be < "
                f"train.max_train_steps ({total}): a step floor the run never reaches is "
                f"an invariant that can never fire — armed in the config, absent in effect"
            )
        return self

    @model_validator(mode="after")
    def _graph_sims_regime_fits_the_hexg_record_format(self) -> "RunConfig":
        """R255/ADJ-D34 — the Phase-T guard's capacity is DERIVED from the sims regime,
        and the schema validates the relation explicitly, so an unsupported regime is a
        MINT-time error, never a boot surprise.

        ADJ-D34's defect was the inversion: a ``MAX_VISITS = 128`` literal on the armed
        boot path refused the prereg'd PCR 600/75 SIMS-REGIME row while every config
        validated clean — the failure surface sat exactly one stage too late. The
        derivation authority is ONE Rust function
        (``mantis_selfplay::replay::hexg::derived_visit_capacity``), called here through
        its bridge twin and by ``SelfPlayRunner::new`` at boot, so the two surfaces
        cannot drift onto second formulas. No key is transcribed and no ceiling is
        restated here (R98 derive-at-point-of-use): this validator only forwards the
        regime keys and re-raises the engine's refusal with the field paths attached.

        Graph-scoped: dense-362 records carry no HEXG visit slot, so the relation does
        not constrain grid configs (R250's absence principle, mint-side). The
        completed-Q leg of the derivation (child-count-wide support vs
        ``MAX_CHILDREN_PER_NODE``) is unreachable from a validated ``RunConfig`` today —
        ``train.policy_target`` is the single-member Literal ``"raw_visit_distribution"``,
        so ``_policy_target_completed_q_consistency`` already forbids
        ``selfplay.completed_q_values=true`` — but it rides the same call so the day
        that Literal widens, the mint check is already standing.

        The function-scope import mirrors ``mantis.run._select_buffer``'s stated
        posture: ``mantis._engine`` is already a transitive dependency of this module
        (``mantis.encoding.lookup`` above), so this adds no import-DAG edge.
        """
        if self.identity.representation != "graph":
            return self
        from mantis._engine import derived_hexg_visit_capacity

        sp = self.selfplay
        pc = sp.playout_cap
        try:
            derived_hexg_visit_capacity(
                n_simulations=sp.mcts.n_simulations,
                standard_sims=pc.standard_sims,
                fast_prob=pc.fast_prob,
                fast_sims=pc.fast_sims,
                full_search_prob=pc.full_search_prob,
                n_sims_quick=pc.n_sims_quick,
                n_sims_full=pc.n_sims_full,
                leaf_batch_size=sp.leaf_batch_size,
                completed_q_values=sp.completed_q_values,
            )
        except ValueError as exc:
            raise ValueError(
                "the selfplay sims regime cannot be honored by the HEXG graph record "
                f"format: {exc} [derived from selfplay.mcts.n_simulations, "
                "selfplay.playout_cap.{standard_sims,fast_prob,fast_sims,"
                "full_search_prob,n_sims_quick,n_sims_full}, selfplay.leaf_batch_size, "
                "selfplay.completed_q_values — R255/ADJ-D34: refused at mint, "
                "never at boot]"
            ) from exc
        return self

    @model_validator(mode="after")
    def _draw_rate_evidence_bar_within_configured_capacity(self) -> "RunConfig":
        """WPMINT Phase DS (R92), **re-scoped by R95 (ADJ-22)**.

        WHAT THIS ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. It asserts one CONFIG-DOMAIN
        fact: `N_pool_min` does not exceed the evidence CAPACITY the config itself declares.
        The draw-rate abort's bar is compared against `Sum(completed)` over the UNION of the
        pool's per-worker windows; each window is a `deque(maxlen=DRAW_RATE_WINDOW)`, so the
        sum can never exceed `DRAW_RATE_WINDOW * selfplay.n_workers` — measured at 1/2/8/32
        workers in `tests/selfplay/test_drawrate_pooled_statistic.py`, not inferred. Both
        operands are visible at load time: one is a shipped constant, the other a config key.
        A bar above that ceiling asks for more evidence than the configured pool can
        physically hold, and THAT is a fact this validator can witness.

        It does **NOT** assert that the bar is REACHABLE. This validator used to be called
        `_draw_rate_evidence_bar_is_reachable`, and that name was an OVERCLAIM (ADJ-22):
        reachability depends on how many workers actually report, which load time cannot
        see. At the ceiling (`N_pool_min == DRAW_RATE_WINDOW * n_workers`, which is run5's
        posture — 50 == 50 x 1) EVERY configured worker must fill its entire window before
        the bar is met, so a single silent worker leaves it unmet for the whole run while
        the config validates clean. No config-time arithmetic can close that, because the
        missing input is a runtime one.

        R95 settles it by fixing the CLAIM rather than widening the check: **a validator's
        name and message may assert only what its inputs can witness.** Evidence sufficiency
        stays runtime's, where R92 already owns it — below the bar the gate makes NO
        OBSERVATION (a `None`, skip-counted, never appended), so an unmet bar is visible as
        an absence of observations rather than fabricated into a healthy `0.0`, and
        zero-completion starvation is explicitly the STALL family's jurisdiction (R92).

        This validator is what re-establishes the load-bearing bound R92 deleted.
        `min_samples` carried `le=DRAW_RATE_WINDOW` (`util/constants.py`'s LOAD-BEARING
        COUPLING note), and deleting the key would have deleted its pin. The bound could not
        follow it onto `N_pool_min` as a plain `le=`, because the ceiling now depends on
        `selfplay.n_workers` — a config-authored value in ANOTHER SECTION. `RunConfig` is
        the one model that sees both, exactly as it is for the actor-sync twin above.

        It carries its OWN name rather than joining the validator above: that one is about
        the STEP CLOCK and this one is about EVIDENCE. A rule hidden inside a validator named
        for a different axis is a false name at the moment it fires (R73).
        """
        block = self.train.draw_rate_abort
        if block is None:
            return self
        ceiling = DRAW_RATE_WINDOW * self.selfplay.n_workers
        if block.N_pool_min > ceiling:
            raise ValueError(
                f"train.draw_rate_abort.N_pool_min ({block.N_pool_min}) must be <= "
                f"{ceiling} = DRAW_RATE_WINDOW ({DRAW_RATE_WINDOW}) * selfplay.n_workers "
                f"({self.selfplay.n_workers}): the pool's per-worker draw windows cannot "
                f"hold more completed games than that between them, so a larger bar asks "
                f"for more evidence than this configuration can physically hold. "
                f"(R95: this is a CAPACITY check, not a reachability one — whether the bar "
                f"is actually met depends on how many workers report, which load time "
                f"cannot see. An unmet bar is visible at runtime as an absence of "
                f"observations, never as a healthy 0.0 — R92.)"
            )
        return self
