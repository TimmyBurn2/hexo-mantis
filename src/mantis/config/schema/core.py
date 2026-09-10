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


@dataclass(frozen=True)
class ArchScopedKey:
    """One config block that belongs to exactly ONE representation (R322(d)).

    `section` names the `RunConfig` field holding it and `field` the block inside that
    section; `arch` is the one `identity.representation` on which the block is legal, and on
    which it is REQUIRED. `grounds` says why the key means something only on that arch — a
    scoping with no grounds is indistinguishable from an oversight waved through, the standard
    gate 17's exemptions are held to.

    A frozen DATACLASS and deliberately NOT a `StrictModel`: this is metadata ABOUT config, not
    a config block, and the O16 census walks `StrictModel` subclasses asserting every one is
    reachable from `RunConfig` and has no code-side default. A registry record that satisfied
    neither would have to be exempted from an exactness check whose whole value is having no
    exemptions.

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


#: THE ARCH-SCOPED HALF OF THE SCHEMA PARTITION — the ONE authority (R322(d),
#: `SEAM_V1_DESIGN` §3: "The schema splits shared vs arch-scoped. An arch-scoped key reachable
#: outside its arch is a red row").
#:
#: This registry did not exist before B2, and its absence WAS the defect: the call sites were
#: arch-gated by hand and the schema was not, so a grid run was REQUIRED to carry two graph-only
#: cap blocks counted in EDGES and NODES, and the mint had to invent a number for a quantity
#: that run has none of. The gate was one `if` at one call site, and the class this exists for
#: is the one where the NEXT call site forgets it.
#:
#: DECLARED, not derived, and that is argued rather than assumed: whether a key belongs to one
#: representation is a judgment about what the key MEANS, and there is no producer in the tree
#: to read it off. What IS derived is everything it is checked against — the live schema, the
#: presence facts, and both reachability answers (the conformance suite's T9 section).
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


#: THE OPERATIONAL-CONSTANT HALF OF THE SCHEMA PARTITION (R347 / CONFIG-1) — the ONE authority
#: for which keys carry a schema default and therefore leave the YAML.
#:
#: THE CRITERION, stated once so every row can be checked against it: a key defaults when it
#: names how the PROCESS is operated — a watchdog deadline, a poll interval, a join bound, a
#: disk threshold, a queue cap, a diagnostic switch — and no run has ever decided it
#: differently. It stays REQUIRED when a run really chooses it. That is why `gate_interval`,
#: the actor-lag pair, `supervisor_kill_grace_sec` and the whole WR/axis warn family are NOT
#: here: those are ARMING, and R1's silently-disabled-opponent reason bites hardest exactly
#: there.
#:
#: THIS IS R1, NOT AN EXCEPTION TO IT. R1's own words are "NO code-side defaults — a default
#: lives only in the schema field". A default here is that sentence's second clause; what R1
#: forbids is the `dict.get(key, fallback)` at a call site, which is a SECOND authority. A run
#: that wants another value mints the row (`mint_config.py --mint-row`) — the same act, and
#: visible in the config's stamped header — so nothing becomes hand-varied.
#:
#: DECLARED, not derived, for `ARCH_SCOPED_KEYS`' reason: "operational" is a judgment about
#: what a key MEANS, and a census that read the answer off `is_required()` would be satisfied
#: by any default anyone adds, which is the exact event it exists to catch.
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
)


def operational_default_fields(prefix: str) -> frozenset[str]:
    """The immediate field names under `prefix` that carry an operational schema default.

    Args:
        prefix: a dotted section path (`"monitor"`, `"monitor.drain"`), or `""` for the root.

    Returns:
        The field names, without the prefix — the shape a per-model census compares against
        `model.model_fields`.

    Raises:
        ValueError: `prefix` names no row at all, which means a census is filtering on a
            section that has moved and would silently exempt nothing.
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
    """The BC warm-start SOURCE: which checkpoint a fresh run's representation+policy weights
    come from, and which net that checkpoint is (R332(d), AUDIT-1 F-19).

    A BLOCK rather than two sibling leaves, and that is the whole design. Both members are
    REQUIRED *inside* it, so `checkpoint` without `net_hash` is UNCONSTRUCTIBLE — and a path
    with no hash is precisely the shape that lets a run warm-start from whatever file happens
    to be sitting there. `null` on the parent is the EXPLICIT no-warm-start posture, the same
    "block or null" the `train.draw_rate_abort` row uses for the same reason.

    `net_hash` is `mantis.model.identity.net_param_hash` over the net REBUILT FROM the
    checkpoint's own stamp — not a file digest. A file digest changes with re-saves,
    compression and metadata; the parameter hash is the thing a prereg means when it names an
    artifact, and it is the same currency `worker_sweep`, `acceptance_witness` and T10 already
    report, so a prereg row and a sweep row can be compared without a conversion.
    """

    checkpoint: str = Field(min_length=1)
    net_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class IdentityConfig(StrictModel):
    """Identity keys have no terminal defaults (repo_design §5): absent = error — for
    ``encoding`` and ``representation``. ``arch_kind`` is the ONE optional identity leaf, below.

    ``representation`` is cross-checked against the encoding's registry representation at
    validation time (F1 runtime guard): a graph encoding declared ``representation: grid`` (or
    vice versa) is REJECTED at load, so the LAW-06 amp pin (resolve_amp_dtype reads this field)
    cannot be bypassed by a LAW-11-inconsistent config. Frozen sourced representation from the
    encoding spec, making disagreement structurally impossible; this guard restores that invariant.

    ``arch_kind`` is THE ARCH-SELECTOR ROW (R330(e), candidate D of R322(d)): which member of
    ``mantis.model.ARCH_KINDS`` a run builds. R323(b) rules that it enters production configs
    ONLY as a minted row at run6's mint, so it is schema-OPTIONAL and every committed config
    omits it; ``mantis.model.arch.arch_from_spec_and_config`` reads it, hands a present value to
    ``select_arch`` (an unknown kind, or one the representation does not admit, is refused there
    by name at construction — this schema cannot import the vocabulary without a config↔model
    cycle, gate 9), and resolves an ABSENT row to the representation's incumbent, a history fact
    pinned against every minted file. The ``None`` is therefore not a fallback carrying a guess:
    it states "this config predates the row", and the arch such configs have always built is
    pinned by test. Artifacts never read this row — a checkpoint's arch is its stamp's.

    ``warm_start`` is THE BC WARM-START ROW (R332(d), AUDIT-1 F-19) and it is optional for
    ``arch_kind``'s reason, not a new one: it enters production configs only as a minted row at
    run6's mint, so every committed config omits it today. It is an IDENTITY key because it
    decides what net a run starts from — the same class of fact as which arch it builds. Its
    consumer is ``mantis.train.warmstart.resolve_bc_warm_start`` -> ``apply_bc_warm_start``,
    called from ``mantis.train.orchestrator.init_trainer``'s fresh-run branch; an absent row
    means NO transfer, which is what every run before the row did.
    """

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
    """One opponent-ladder rung (design §c.1). `bot` is the resolver kind (closed set,
    WP11-A); `depth` is sealbot's fixed-depth bar (LAW-15), `opponent_sims` the
    opponent-side sims — exactly one of the two is meaningful per `bot`, and both travel as
    `None` where inapplicable rather than as a sentinel int (R1: no code default).

    RED-TEAM F2 (MAJOR): a fixed-depth bar of 0 or negative, an opponent-sims count of 0 or
    negative, or a rung that can never play a single game (`games_max < 1`) are each
    domain-nonsense — bounded here as named `Field` constraints (pydantic includes the
    field path in the raised error), never a silent clamp.
    """

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
    `mantis.arena.match.DEFAULT_MAX_PLIES` because the board is unbounded, and until now
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
    random_floor_games: int = Field(ge=0)
    worker_device: Literal["cuda", "cpu"]
    # OPERATIONAL CONSTANTS (R347/CONFIG-1): a round's wall-clock bound and the grace a
    # killed worker gets are how the eval PROCESS is operated, not what the round measures —
    # every committed config has minted the same pair. The games, sims and bars beside them
    # stay required, because those are what a run decides.
    round_timeout_sec: float = Field(
        default=3600.0, gt=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    worker_kill_grace_sec: float = Field(
        default=10.0, ge=0, le=_EVAL_TIMEOUT_CEILING_SEC, allow_inf_nan=False)
    #: `default=...` is this schema's own no-terminal-default idiom (the shape
    #: `train.scheduler_t_max` / `train.draw_rate_abort` already carry): the key is REQUIRED
    #: and an absent one is an error naming it, while `None` is a real, explicit posture the
    #: config must state. Both blocks below ship `null` in every committed config, which is
    #: the identity value — the run behaves exactly as it did before the blocks existed —
    #: and arming either one is a mint event, never an IMPL edit (R1).
    ply_cap_adjudication: PlyCapAdjudicationConfig | None = Field(default=...)
    strength_floor: StrengthFloorConfig | None = Field(default=...)
    #: THE GATE-BLOCK CONCURRENCY ROW (R339(b)): how many gate games run IN FLIGHT, one thread
    #: each, sharing the round's two inference engines. OPTIONAL with a real default for
    #: `identity.arch_kind`'s reason and not a new one — it enters production configs only as a
    #: minted row, and `1` is not a fallback carrying a guess: it is the SERIAL loop
    #: `play_paired_match` ran before the parameter existed, byte-exact, so an absent row and a
    #: minted `1` are the same round. Its ONE consumer is `mantis.eval.worker._play_gate_block`,
    #: reached through `RoundSpec.concurrency`; the floor probe, the rung battery and the random
    #: floor keep the serial arm deliberately (a LAW-07 gate input and LAW-04's Elo channel must
    #: not become nondeterministic to save ~200 s of a ~7 000 s round).
    concurrency: int = Field(ge=1, default=1)
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
    # RECAL-PREP / R308(g)(i): the CUDA caching allocator's REGIME, as a closed token set
    # (`default` | `expandable_segments`) or the R119 `null` placeholder. TOP-LEVEL for
    # `eval_enabled`'s own recorded grounds — it is a root-composition fact spanning more
    # than one section's surface: the trainer, the self-play inference server and the eval
    # child are three consumers in two processes, and any sectional home would make one of
    # the three read another section's key.
    #
    # WHY IT IS A CONFIG KEY AT ALL, measured. The 2026-08-22 re-calibration sitting measured
    # 14.98 GiB of card high-water under DEFAULT against 11.36 under `expandable_segments:True`
    # at matched config and duration, and kept DEFAULT anyway — not on the measurement, but
    # because a cap fitted under the better posture would depend on an environment variable no
    # config minted, no gate checked and no `armed_aborts` row covered. That is a minted value
    # with an unminted precondition (R1's silent-authority class), and this field is its
    # removal. The VALUE is a MEASUREMENT the re-sit takes under R282(b); `null` is refused at
    # boot by `mantis.config.resolve.allocator_posture`, so no CUDA run can proceed on a regime
    # nobody measured, and no `Literal` member may be minted by anything but that sitting.
    allocator_posture: Literal["default", "expandable_segments"] | None
    identity: IdentityConfig
    # The SEARCH REGIME, top-level for `eval_enabled`'s own recorded grounds: it is a
    # root-composition fact spanning more than one section's surface. Self-play searches
    # with it and `arena.deploy_head` searches with it, and LAW-15's deploy-matched claim
    # is precisely that the two are the SAME search — a key under `selfplay` would make
    # the eval head a reader of another surface's section, which is how the deploy head
    # came to run a regime the run never declared.
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
        """Serialize, then DELETE every arch-scoped block this config did not carry (R322(d)).

        WHY THIS IS PART OF THE SCOPING AND NOT A COSMETIC. An arch-scoped block is
        `Block | None`, and pydantic emits an unset field as `None` — so without this, a grid
        config's `model_dump()` carried `train.microbatch_caps: None` and **would not
        re-validate**, because the validator above reads presence off `model_fields_set` and a
        dump has no `model_fields_set` to read. The round trip
        `RunConfig.model_validate(config.model_dump())` is load-bearing: `mantis.run` hands the
        dump to every resolver, and the mint preflight re-validates it.

        The absent/explicit-null distinction survives INPUT and is deliberately NOT
        representable in OUTPUT: on the way in, an explicit `null` on a foreign arch is refused
        by name; on the way out, "this arch does not have this key" is expressed the only way a
        plain mapping can express it — the key is not there.

        Driven from `ARCH_SCOPED_KEYS` so it cannot drift from the rule it serves.
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
        """R322(d) / `SEAM_V1_DESIGN` §3 — every `ARCH_SCOPED_KEYS` block is REQUIRED on its
        own representation and REFUSED on any other.

        THE DEFECT THIS CLOSES. `RunConfig` is `extra="forbid"` with every key required, so
        before this validator a grid config was not merely ALLOWED to carry the two graph-only
        cap blocks — it was REQUIRED to, and `tools/mint_config.py` had to write a number for a
        quantity a grid run has none of. The call sites were gated by hand (`run.py` resolves
        `fused_graph_caps` only on the graph branch; `coordinator/step.py` hands
        `microbatch_caps` to the graph arm as a thunk), and everything BELOW the call site was
        not: the schema demanded the key and the resolver served it to a config of either arch.

        PRESENCE IS READ OFF `model_fields_set`, NOT OFF THE VALUE, and the distinction is
        load-bearing on `inference.fused_graph_caps`: an explicit `null` there is the R119
        PLACEHOLDER on the block's MEMBERS ("minted but uncalibrated", refused at boot by
        `resolve_fused_graph_caps`), which is a different fact from the block being ABSENT
        ("this arch has no such key"). A value test would collapse the two and let a grid config
        satisfy this rule by minting a placeholder.

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
        """`train.policy_target` states what the SEARCH produced, so it follows the kind.

        The exported target is built by the search itself: `search.kind: gumbel` exports
        the completed-Q improved policy and `puct` exports the temperature-annealed visit
        distribution. `train.policy_target` names which of those the LOSS is applied to,
        and a config in which the two disagree trains one target's loss on the other
        target's rows — the exact class the resume guard closed on a checkpoint boundary
        (`train.checkpoints`) and which is closed here at mint.

        `policy_target` is not deleted in favour of derivation, and that is deliberate: it
        is the key the CHECKPOINT stamp carries, so it is the record of what a stored
        ring's rows MEAN. Deriving it would leave a resume with nothing to compare against.

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

        That kind reaches its root's full legal set, so it spends ``MAX_ROOT_CHILDREN``
        pool slots on the root instead of ``MAX_CHILDREN_PER_NODE`` and its ceiling is
        ``MAX_ARMED_SIMS_GUMBEL``, below the ``MAX_ARMED_SIMS`` the field bounds carry. A
        `Field(le=...)` cannot express this — the applicable ceiling depends on a key in
        ANOTHER SECTION — so without this validator a config in the gap between the two
        bounds would validate clean and be refused by ``SelfPlayRunner::new`` at boot.

        That inversion is the one R255/ADJ-D34 closed for the HEXG visit capacity, in this
        same class, and re-opening it a section away would be the same defect wearing a
        different key's name.

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
        THE BLOCKER THIS DOCSTRING USED TO RECORD IS CLOSED (R347(a)). ``search.kind:
        gumbel`` on a graph run was refused outright, on the grounds that its exported
        target puts mass on the whole legal set and the legal set is not a constant. The
        ruling's answer is that the ROW does not have to carry that support: under
        Sequential Halving only the ``selfplay.gumbel_m`` sampled candidates are ever
        visited, every other legal action's completed-Q target is the recording prior times
        one scalar, and the row stores the m explicit entries plus that one scalar. So on the
        Gumbel arm the slot count is the MINTED ``gumbel_m``, the sims regime does not enter,
        and the refusal on that arm is an m past ``HEXG_GUMBEL_M_MAX``.

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

        WHY THIS EXISTS. The HEXG record stores a position's stones in a FIXED-WIDTH slot —
        `stones_qr` is `[capacity * MAX_STONES * 2]` — so `push_graph_position` refuses any
        position with more stones than `MAX_STONES`, by name and at write time. A ply cap
        above that ceiling therefore configures a run whose own late positions the ring cannot
        hold, and the refusal arrives per-record in the middle of a run rather than at mint.

        THE MEASURED INSTANCE THAT AUTHORED IT. Encoding the R247 human corpus at radius 8
        hit this: 88 of 8 698 games exceed 257 plies and 7 866 of 547 251 ply rows (1.4374 %)
        cannot be stored. The architect ruled `MAX_STONES` STAYS 256 and the corpus truncates
        with its loss counted — on the ground that the RUN's own games will never reach it,
        the ply-cap prereg row targeting ~256. **This validator is what makes that ground
        enforceable instead of assumed**: a later prereg cannot quietly raise the cap past the
        ring, and if the operator wants a higher cap then `MAX_STONES` rises FIRST, as a
        mint-class change with its memory cost measured.

        THE CEILING IS READ FROM THE ENGINE, NEVER TYPED. `max_stones()` was added to the
        bridge for this relation precisely so the number is not transcribed here; a `256` in
        this file would be a second authority over a fixed-width allocation, which is the one
        place a drift would be silent.

        GRAPH ONLY, and that is a scope rather than an oversight: the ceiling belongs to the
        HEXG ring, which the dense path does not use.

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
        see. At the ceiling (`N_pool_min == DRAW_RATE_WINDOW * n_workers`) EVERY configured
        worker must fill its entire window before the bar is met, so a single silent worker
        leaves it unmet for the whole run while the config validates clean. No config-time
        arithmetic can close that, because the missing input is a runtime one.

        RUN5 IS NO LONGER AT THAT CEILING, and the example is corrected rather than deleted
        (R311(h)/F-WS-4, same act as RECAL-SITTING-5's mint). This paragraph read "which is
        run5's posture — 50 == 50 x 1"; the mint takes `selfplay.n_workers` to the Phase W
        pick, so run5's ceiling is `DRAW_RATE_WINDOW * 14` and `N_pool_min: 50` now sits far
        below it. THE WARNING IS UNCHANGED AND IS NOT WEAKENED BY THAT: off the ceiling, the
        bar needs a smaller FRACTION of the configured workers to report, but it still needs
        some of them to, and load time still cannot see which do.

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
