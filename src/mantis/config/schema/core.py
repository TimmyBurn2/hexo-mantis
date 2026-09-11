"""Run-config schema (contract run-config-schema v1).

>300 justify (R8). This file carries the `RunConfig` root model and the cross-field
`model_validator`s that span SECTIONS (train x selfplay, train x monitor, identity x
selfplay), so they cannot live in any section module.

Every model is strict: unknown key = hard error, missing key = hard error, silent scalar
coercions rejected, values immutable. A default lives in exactly one place — the schema
field; identity keys carry no terminal defaults, and representation is the closed set
{grid, graph}.
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from mantis.config.schema._base import StrictModel
from mantis.config.schema.monitor import MonitorSchemaConfig
from mantis.config.schema.search import SearchConfig
from mantis.config.schema.selfplay import (
    MAX_ARMED_SIMS,
    MAX_ARMED_SIMS_GUMBEL,
    InferenceConfig,
    SelfplayConfig,
)
from mantis.config.schema.train import TrainConfig
from mantis.encoding import EncodingRegistryError, lookup
from mantis.util.constants import DRAW_RATE_WINDOW

SCHEMA_VERSION = 1

#: A finite ceiling for a timeout float that feeds `proc.join(timeout)` arithmetic:
#: `multiprocessing.Process.join` raises `OverflowError` on `float("inf")`, so a floor-only
#: bound is not a bound here. One day (86400.0 s) bounds one eval round or kill-grace.
_EVAL_TIMEOUT_CEILING_SEC = 86400.0


@dataclass(frozen=True)
class ArchScopedKey:
    """One config block that belongs to exactly ONE representation: `section`/`field` locate it,
    `arch` is the one representation on which it is legal and REQUIRED, and `grounds` says why the
    key means something only there.

    Raises:
        ValueError: any field is empty, or `arch` is outside the closed representation set.
    """

    section: str
    field: str
    arch: str
    grounds: str

    def __post_init__(self) -> None:
        for name in ("section", "field", "arch", "grounds"):
            if not getattr(self, name):
                raise ValueError(f"ArchScopedKey.{name} is empty")
        if self.arch not in ("grid", "graph"):
            raise ValueError(
                f"ArchScopedKey.arch={self.arch!r} is outside the closed representation set "
                "{'grid', 'graph'} (LAW-11)"
            )


#: The ARCH-SCOPED half of the schema partition — the ONE authority for which keys are legal on
#: which representation. DECLARED, not derived: there is no producer in the tree to read it off.
ARCH_SCOPED_KEYS: tuple[ArchScopedKey, ...] = (
    ArchScopedKey(
        section="train", field="microbatch_caps", arch="graph",
        grounds="the members are counted in EDGES and NODES of a sampled graph batch; the "
                "dense batch is a fixed-shape tensor already bounded by train.batch_size, so "
                "a grid run has no quantity for this block to bound",
    ),
    ArchScopedKey(
        section="inference", field="fused_graph_caps", arch="graph",
        grounds="the members bound one FUSED GRAPH forward in edges and nodes; a grid round "
                "builds no graph server, so a grid run has no fused forward to cap",
    ),
)


#: The OPERATIONAL-CONSTANT half of the schema partition — the ONE authority for which keys carry a
#: schema default and therefore leave the YAML. A key defaults when it names how the PROCESS is
#: operated and no run has ever decided it differently; ARMING keys stay REQUIRED.
OPERATIONAL_DEFAULT_KEYS: tuple[tuple[str, str], ...] = (
    ("eval.round_timeout_sec", "a round's wall-clock bound — how long the eval process may "
                               "run, not what the round measures"),
    ("eval.worker_kill_grace_sec", "the grace a killed eval worker gets before SIGKILL"),
    ("train.selfplay_stall_timeout_sec", "the stall watchdog's wall-clock budget; LAW-16 "
                                         "keeps it always armed, so only its length is here"),
    ("selfplay.results_queue_cap", "the results queue's back-pressure bound"),
    ("selfplay.log_investigation_metrics", "a diagnostic verbosity switch: it gates what is "
                                           "WRITTEN, never what is played"),
    ("monitor.heartbeat_deadline_train_step_sec", "watchdog deadline"),
    ("monitor.heartbeat_deadline_inference_dispatch_sec", "watchdog deadline"),
    ("monitor.heartbeat_deadline_selfplay_drain_sec", "watchdog deadline"),
    ("monitor.heartbeat_deadline_eval_round_sec", "watchdog deadline"),
    ("monitor.heartbeat_poll_interval_sec", "watchdog poll cadence"),
    ("monitor.heartbeat_file_interval_sec", "heartbeat-file write cadence"),
    ("monitor.heartbeat_close_out_deadline_sec", "close-out deadline"),
    ("monitor.heartbeat_fire_effect_timeout_sec", "how long a fired watchdog waits for its "
                                                  "own effect before escalating"),
    ("monitor.supervisor_stale_after_sec", "supervisor staleness bound"),
    ("monitor.supervisor_poll_interval_sec", "supervisor poll cadence"),
    ("monitor.supervisor_max_relaunches", "supervisor relaunch ceiling"),
    ("monitor.drain", "the four drain/terminal-eval caps are subprocess-join bounds; the "
                      "BLOCK defaults so a config that names none of them omits it whole"),
    ("monitor.drain.final_eval_drain_timeout_sec", "subprocess-join bound"),
    ("monitor.drain.eval_final_drain_safety_factor", "subprocess-join bound"),
    ("monitor.drain.eval_final_drain_hard_cap_sec", "subprocess-join bound"),
    ("monitor.drain.terminal_eval_hard_cap_sec", "subprocess-join bound"),
    ("monitor.disk_guard", "the three disk thresholds; the BLOCK defaults for `drain`'s "
                           "reason"),
    ("monitor.disk_guard.interval_sec", "disk-guard poll cadence"),
    ("monitor.disk_guard.warn_gb", "disk-guard warn threshold"),
    ("monitor.disk_guard.fail_gb", "disk-guard fail threshold"),
    ("inference.edge_geometry_check", "where check 14 runs — inline on the serving loop or on "
                                      "a checker thread after the batch is served (R347(e)); "
                                      "it runs on every batch either way, so this operates "
                                      "the server and decides nothing a run measures"),
    ("inference.compile_trunk", "whether the SERVING forward's trunk is torch-compiled; the "
                                "default is the shipped eager path and arming it is a mint "
                                "act whose numerics consequence (eager-to-bf16-noise on the "
                                "serving side only) the A4 record states"),
)


def operational_default_fields(prefix: str) -> frozenset[str]:
    """The immediate field names under `prefix` that carry an operational schema default.

    Args:
        prefix: a dotted section path (`"monitor"`, `"monitor.drain"`), or `""` for the root.

    Returns:
        The field names, without the prefix.

    Raises:
        ValueError: `prefix` names no row at all, so a census filtering on it would silently
            exempt nothing.
    """
    head = f"{prefix}." if prefix else ""
    names = frozenset(
        key[len(head):] for key, _grounds in OPERATIONAL_DEFAULT_KEYS
        if key.startswith(head) and "." not in key[len(head):]
    )
    if not names:
        raise ValueError(
            f"operational_default_fields({prefix!r}) matched no row of "
            "OPERATIONAL_DEFAULT_KEYS: a census filtering on a section that has moved exempts "
            "nothing and passes forever"
        )
    return names


class WarmStartConfig(StrictModel):
    """The BC warm-start SOURCE: which checkpoint a fresh run's weights come from, and which net
    that checkpoint is. A BLOCK, so `checkpoint` without `net_hash` is UNCONSTRUCTIBLE, `null` on
    the parent is the explicit no-warm-start posture, and `net_hash` is the parameter hash of the
    net rebuilt from the checkpoint's own stamp rather than a file digest."""

    checkpoint: str = Field(min_length=1)
    net_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class IdentityConfig(StrictModel):
    """Identity keys have no terminal defaults — absent = error for ``encoding`` and
    ``representation``; ``arch_kind`` and ``warm_start`` are the two optional leaves.
    ``representation`` is cross-checked against the encoding's registry representation at load, an
    absent ``arch_kind`` resolves to the representation's incumbent, and an absent ``warm_start``
    means NO transfer."""

    encoding: str = Field(min_length=1)
    representation: Literal["grid", "graph"]
    arch_kind: str | None = Field(default=None, min_length=1)
    warm_start: WarmStartConfig | None = Field(default=None)

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
    """One opponent-ladder rung. Exactly one of `depth`/`opponent_sims` is meaningful per `bot` and
    the other travels as `None` rather than as a sentinel int; the bounds are named `Field`
    constraints, so a rung that can never play a game is a named error and not a silent clamp."""

    name: str = Field(min_length=1)
    bot: Literal["sealbot", "random"]
    variant: str = Field(min_length=1)
    depth: int | None = Field(ge=1)
    opponent_sims: int | None = Field(ge=1)
    opening_book: str = Field(min_length=1)
    deploy_matched: bool
    games_max: int = Field(ge=1)


class GateConfig(StrictModel):
    """The run3 deploy-strength gate, knob-for-knob (LIVE knobs only).

    `screen_confirm_hi` is deliberately NOT ported, so a minted one is rejected by
    `extra="forbid"`. The win-rate fractions are bounded to `[0,1]`, because an out-of-range value
    loads silently and then disables promotion permanently; the counts and `stride` are `>=1`."""

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
    """The opponent-ladder schema: ordered rungs plus every scheduling/hysteresis threshold as a
    named field, never a code literal.

    `bootstrap_ci_level` is bounded to the open `(0,1)`, since outside it `np.quantile` either
    raises inside a worker subprocess or returns a statistically inverted CI; `bt_prior_games`
    carries `allow_inf_nan=False`, because an `inf` prior makes every BT rating NaN."""

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
    """How a PLY-CAPPED eval game is resolved — ONE block carrying the whole fact "is ply-cap
    adjudication armed, and on what criterion".

    `null` is ARMED=NO and is the posture every shipped config takes. The two postures are disjoint
    TYPES rather than two regions of one range, so a boolean cannot contradict the criterion and a
    criterion cannot arrive without its margin. Read by exactly one path:
    `mantis.config.resolve.eval_posture.resolve_ply_cap_adjudication`."""

    criterion: Literal["longest_run_margin", "immediate_win_margin"]
    min_margin: int = Field(ge=1)


class StrengthFloorConfig(StrictModel):
    """The cheap probe that gates the EXPENSIVE ladder — ONE block carrying the whole fact "is the
    ladder gated on a strength floor, and on what terms".

    `null` is ARMED=NO and is the posture every shipped config takes. `probe_games` is denominated
    in GAMES rather than seconds so the bar stays a reproducible instrument, and `min_winrate` at
    `0.0` is a legal EXPLICIT posture rather than a silently disabled lever. Read by exactly one
    path: `mantis.config.resolve.eval_posture.resolve_strength_floor`."""

    probe_games: int = Field(ge=1)
    min_decisive_rate: float = Field(ge=0, le=1)
    min_winrate: float = Field(ge=0, le=1)


class EvalConfig(StrictModel):
    """Eval opponent simulation counts and process bounds — `resolve_eval_model_sims` reads these
    and there is no code default.

    `round_timeout_sec` and `worker_kill_grace_sec` carry `allow_inf_nan=False` plus a finite
    ceiling: a genuine YAML `.inf` literal reaches a real `multiprocessing.Process.join`, which
    raises `OverflowError` on a non-finite timeout."""

    random_model_sims: int = Field(ge=1)
    sealbot_model_sims: int = Field(ge=1)
    random_floor_games: int = Field(ge=0)
    worker_device: Literal["cuda", "cpu"]
    # OPERATIONAL CONSTANTS: a round's wall-clock bound and a killed worker's grace are how the
    # eval PROCESS is operated, not what the round measures; the games, sims and bars stay required.
    round_timeout_sec: float = Field(
        default=3600.0, gt=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    worker_kill_grace_sec: float = Field(
        default=10.0, ge=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    #: `default=...` is this schema's no-terminal-default idiom: the key is REQUIRED and an absent
    #: one is an error naming it, while `None` is a real, explicit posture. Both blocks below ship
    #: `null` in every committed config, and arming either is a mint event.
    ply_cap_adjudication: PlyCapAdjudicationConfig | None = Field(default=...)
    strength_floor: StrengthFloorConfig | None = Field(default=...)
    #: The gate-block concurrency row: how many gate games run IN FLIGHT, one thread each, sharing
    #: the round's two inference engines. `1` is byte-exact the serial loop that ran before the
    #: parameter existed; the floor probe, rung battery and random floor stay serial deliberately.
    concurrency: int = Field(ge=1, default=1)
    gate: GateConfig
    ladder: LadderConfig


class RunConfig(StrictModel):
    """Top-level run config: explicit, complete, schema_version-pinned. There is deliberately NO
    ``legal_move_radius``/``legal_move_radius_schedule`` field anywhere on this tree — the encoding
    registry alone is the radius authority."""

    schema_version: int
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    seed: int
    # The run's eval posture is a CONFIG FACT, not a `compose_run` parameter: the parameter is
    # deleted, so no caller can override it. TOP-LEVEL because it spans the eval and monitor
    # wired-source surfaces.
    eval_enabled: bool
    # The CUDA caching allocator's REGIME, as a closed token set or the `null` placeholder;
    # top-level because trainer, inference server and eval child are three consumers in two
    # processes. Measured 2026-08-22: 14.98 GiB card high-water under DEFAULT against 11.36 under
    # `expandable_segments:True` at matched config and duration, and DEFAULT was kept because a cap
    # fitted under the better posture would depend on an unminted environment variable. `null` is
    # refused at boot by `mantis.config.resolve.allocator_posture`.
    allocator_posture: Literal["default", "expandable_segments"] | None
    identity: IdentityConfig
    # The SEARCH REGIME, top-level because self-play searches with it and `arena.deploy_head`
    # searches with it, and the deploy-matched claim is that the two are the SAME search.
    search: SearchConfig
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

    @model_serializer(mode="wrap")
    def _drop_arch_scoped_keys_this_arch_does_not_have(self, handler) -> dict:
        """Serialize, then DELETE every arch-scoped block this config did not carry.

        pydantic emits an unset field as `None` and the arch validator reads presence off
        `model_fields_set`, which a dump has none of, so without this a grid config's dump would
        not re-validate. Driven from `ARCH_SCOPED_KEYS` so it cannot drift from the rule it serves.
        """
        data = handler(self)
        for key in ARCH_SCOPED_KEYS:
            section = data.get(key.section)
            if isinstance(section, dict) and key.field not in getattr(
                self, key.section
            ).model_fields_set:
                section.pop(key.field, None)
        return data

    @model_validator(mode="after")
    def _arch_scoped_keys_are_present_iff_their_arch(self) -> "RunConfig":
        """Every `ARCH_SCOPED_KEYS` block is REQUIRED on its own representation and REFUSED on any
        other.

        Presence is read off `model_fields_set`, NOT off the value: an explicit `null` on
        `inference.fused_graph_caps` is the "minted but uncalibrated" placeholder, a different fact
        from the block being ABSENT, and a value test would let a grid config satisfy this rule by
        minting a placeholder.

        Raises:
            ValueError: a config carries an arch-scoped block that its representation does not
                have, or omits one that its representation requires.
        """
        for key in ARCH_SCOPED_KEYS:
            section = getattr(self, key.section)
            carried = key.field in section.model_fields_set
            valued = getattr(section, key.field) is not None
            if self.identity.representation == key.arch:
                if not (carried and valued):
                    raise ValueError(
                        f"{key.section}.{key.field} is REQUIRED on "
                        f"representation={key.arch!r} and this config "
                        f"{'sets it to null' if carried else 'omits it'} "
                        f"(identity.representation={self.identity.representation!r}). "
                        f"Grounds: {key.grounds}. Absent is an ERROR, never a default (R1/"
                        "LAW-11) — a cap that silently became absent-and-unbounded would still "
                        "report as present."
                    )
            elif carried:
                raise ValueError(
                    f"{key.section}.{key.field} is ARCH-SCOPED to "
                    f"representation={key.arch!r} and this config declares "
                    f"identity.representation={self.identity.representation!r}, so the block "
                    "must be ABSENT, not minted. Grounds: "
                    f"{key.grounds}. Delete the block from the file (R322(d), "
                    "`SEAM_V1_DESIGN` §3: an arch-scoped key reachable outside its arch is a "
                    "red row)."
                )
        return self

    @model_validator(mode="after")
    def _policy_target_matches_the_search_kind(self) -> "RunConfig":
        """`train.policy_target` states what the SEARCH produced, so it follows `search.kind`.

        It is not derived away because it is the key the CHECKPOINT stamp carries, so it is the
        record of what a stored ring's rows MEAN.

        Raises:
            ValueError: the target and the search kind disagree.
        """
        expected = (
            "completed_improved_policy"
            if self.search.kind == "gumbel"
            else "raw_visit_distribution"
        )
        if self.train.policy_target != expected:
            raise ValueError(
                f"train.policy_target={self.train.policy_target!r} disagrees with "
                f"search.kind={self.search.kind!r}, which produces {expected!r}. The search "
                "builds the target; a config that trains one target's loss on the other "
                "target's rows is the defect this pairing exists to make unmintable."
            )
        return self

    @model_validator(mode="after")
    def _search_kind_fits_the_node_pool(self) -> "RunConfig":
        """The Gumbel kind's sim ceiling, refused at MINT and not at boot.

        Its ceiling is ``MAX_ARMED_SIMS_GUMBEL``, below the ``MAX_ARMED_SIMS`` the field bounds
        carry, and a `Field(le=...)` cannot express it because the applicable ceiling depends on a
        key in ANOTHER SECTION.

        Raises:
            ValueError: an armed sims knob exceeds the Gumbel kind's ceiling.
        """
        if self.search.kind != "gumbel":
            return self
        armed = {
            "selfplay.mcts.n_simulations": self.selfplay.mcts.n_simulations,
            "selfplay.playout_cap.standard_sims": self.selfplay.playout_cap.standard_sims,
            "selfplay.playout_cap.fast_sims": self.selfplay.playout_cap.fast_sims,
            "selfplay.playout_cap.n_sims_quick": self.selfplay.playout_cap.n_sims_quick,
            "selfplay.playout_cap.n_sims_full": self.selfplay.playout_cap.n_sims_full,
        }
        over = {k: v for k, v in armed.items() if v > MAX_ARMED_SIMS_GUMBEL}
        if over:
            raise ValueError(
                f"search.kind='gumbel' lowers the node-pool sim ceiling to "
                f"{MAX_ARMED_SIMS_GUMBEL} (from {MAX_ARMED_SIMS}), because that kind "
                "reaches the root's FULL legal set and spends MAX_ROOT_CHILDREN pool slots "
                "on it instead of MAX_CHILDREN_PER_NODE. Over the ceiling: "
                + ", ".join(f"{k}={v}" for k, v in sorted(over.items()))
                + " — lower the budget, or mint search.kind='puct'."
            )
        return self

    @model_validator(mode="after")
    def _actor_lag_threshold_exceeds_sync_cadence(self) -> "RunConfig":
        # The lag threshold (monitor) and the sync cadence (train) share one invariant, so the
        # check lives here, the ONE model that sees both.
        if self.monitor.actor_lag_threshold_steps <= self.train.actor_sync_cadence_steps:
            raise ValueError(
                "actor_lag_threshold_steps (N) must exceed train.actor_sync_cadence_steps: "
                "a threshold at or below the sync cadence fires under healthy operation"
            )
        return self

    @model_validator(mode="after")
    def _actor_sync_knobs_fit_inside_the_run(self) -> "RunConfig":
        """Both step-clock knobs must be reachable within the run.

        `ge=1` alone does not make "never sync" inexpressible: a cadence at or beyond
        `max_train_steps` lets the actor take its single unconditional first sync and then freeze,
        and it also pushes the lag threshold out of reach, so the two knobs fail open together. The
        bound is anchored to `train.max_train_steps`, the RUN-LENGTH authority, not to
        `train.total_steps`, which is only the LR-scheduler horizon.
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
        # The TWIN of the rule above, on the draw-rate abort's own step floor: `min_step >=
        # max_train_steps` audits ARMED while the abort can never fire. `None` is the EXPLICIT
        # disarmed posture and is skipped.
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
        """The HEXG visit capacity is DERIVED from the sims regime, so an unsupported regime is a
        MINT-time error and never a boot surprise.

        The derivation authority is ONE Rust function called both here through its bridge twin and
        by ``SelfPlayRunner::new`` at boot, so the two surfaces cannot drift onto second formulas:
        this validator forwards the regime keys and re-raises the engine's refusal with the field
        paths attached. Graph-scoped, since dense records carry no HEXG visit slot; on the Gumbel
        arm the slot count is the minted ``selfplay.gumbel_m``.
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
                gumbel_m=sp.gumbel_m,
                search_kind=self.search.kind,
            )
        except ValueError as exc:
            raise ValueError(
                "the selfplay sims regime cannot be honored by the HEXG graph record "
                f"format: {exc} [derived from selfplay.mcts.n_simulations, "
                "selfplay.playout_cap.{standard_sims,fast_prob,fast_sims,"
                "full_search_prob,n_sims_quick,n_sims_full}, selfplay.leaf_batch_size, "
                "selfplay.gumbel_m, search.kind — R255/ADJ-D34 + R347(a): refused at mint, "
                "never at boot]"
            ) from exc
        return self

    @model_validator(mode="after")
    def _ply_cap_within_the_rings_stone_ceiling(self) -> "RunConfig":
        """`selfplay.max_game_moves <= mantis._engine.max_stones()` on the GRAPH path.

        The HEXG record stores a position's stones in a FIXED-WIDTH slot, so a ply cap above that
        ceiling configures a run whose own late positions the ring cannot hold and the refusal
        would arrive per-record mid-run. Measured: encoding the human corpus at radius 8 lost
        7 866 of 547 251 ply rows (1.4374 %) to it, and the ruling is that `MAX_STONES` stays 256
        while the corpus truncates. The ceiling is read from the engine, never typed here.

        Raises:
            ValueError: the configured ply cap exceeds the ring's per-record stone ceiling.
        """
        if self.identity.representation != "graph":
            return self
        from mantis._engine import max_stones

        ceiling = max_stones()
        cap = self.selfplay.max_game_moves
        if cap > ceiling:
            raise ValueError(
                f"selfplay.max_game_moves = {cap} exceeds the HEXG ring's per-record stone "
                f"ceiling MAX_STONES = {ceiling}: a position at ply {ceiling + 1} carries "
                f"{ceiling + 1} stones and push_graph_position REFUSES it, so this run would "
                "discard its own late positions one record at a time instead of failing here. "
                "Lower the cap, or raise MAX_STONES as a mint-class change with its ring "
                "memory cost measured (R328 amendment, 2026-09-01)."
            )
        return self

    @model_validator(mode="after")
    def _draw_rate_evidence_bar_within_configured_capacity(self) -> "RunConfig":
        """`train.draw_rate_abort.N_pool_min` does not exceed the evidence CAPACITY the config
        itself declares: the pooled sum can never exceed `DRAW_RATE_WINDOW * selfplay.n_workers`.

        It does NOT assert that the bar is REACHABLE — that depends on how many workers actually
        report, which load time cannot see. The bound could not follow `min_samples` onto
        `N_pool_min` as a plain `le=`, because the ceiling depends on a key in ANOTHER SECTION.
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
