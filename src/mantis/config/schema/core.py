"""Run-config schema (contract run-config-schema v1).

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

SCHEMA_VERSION = 1

#: RED-TEAM-2 F-RT2-1 (BLOCKER fix): an obviously-generous finite ceiling for a
#: timeout/grace-period float field whose value feeds `proc.join(timeout)` arithmetic
#: (pipeline.py isolation law 2) — `multiprocessing.Process.join` cannot accept
#: `float("inf")` (raises `OverflowError` deep inside `selectors.select()`), so a
#: floor-only bound (`gt=0`/`ge=0`) that admits `+inf` is not actually a bound for this
#: arithmetic. One day (86400.0s) is deliberately more generous than the
#: `StepCoordinatorConfig` drain-cap family (coordinator/config.py
#: DEFAULT_*_HARD_CAP_SEC = 14400.0, 4h) since these two fields bound a single eval round
#: / kill-grace, never a whole drain budget — named module constant, never an inline
#: magic literal (R1).
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
        `total_steps` lets the actor take its single unconditional first sync and then
        freeze for the entire run — run3's failure, expressed in a config that validated
        clean. And because the threshold must exceed the cadence, such a config also
        pushes the lag threshold out of reach, so the exit-45 invariant that exists to
        catch a frozen actor could never fire on one. The two knobs failed open together.

        Requiring both to be strictly inside `total_steps` is what actually makes
        "don't sync" unrepresentable, which is what R49 asks of the config surface.
        """
        total = self.train.total_steps
        if self.train.actor_sync_cadence_steps >= total:
            raise ValueError(
                f"train.actor_sync_cadence_steps "
                f"({self.train.actor_sync_cadence_steps}) must be < train.total_steps "
                f"({total}): a cadence the run never reaches means the actor syncs once "
                f"and then never again, which is the frozen actor this WP removed"
            )
        if self.monitor.actor_lag_threshold_steps >= total:
            raise ValueError(
                f"monitor.actor_lag_threshold_steps "
                f"({self.monitor.actor_lag_threshold_steps}) must be < train.total_steps "
                f"({total}): a threshold the run never reaches is an invariant that can "
                f"never fire — armed in the config, absent in effect"
            )
        return self
