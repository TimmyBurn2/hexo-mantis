"""`SelfplayConfig`/`MctsConfig`/`PlayoutCapConfig`/`InferenceConfig` — self-play + inference
knobs as first-class schema fields (R-SELFPLAYCONFIG-SCHEMA closure, DESIGN_P2.md §3).

`legal_move_radius`/`legal_move_radius_schedule` are DELIBERATELY ABSENT (DESIGN_P2.md §5,
shape (ii)): the encoding registry alone is the radius authority for every run today — nothing
in the current build path reads a config-level radius override, so a schema field for it would
be a consumer-less knob (R1/LAW-08). `RadiusStage`/`resolve_radius_from_schedule` are formally
retired by a later chunk (SC-A4); dropping the field from this class is forced now because a
`SelfplayConfig` reshape cannot carry it AND satisfy `extra="forbid"` simultaneously with the
old key.
"""
from typing import Literal

from pydantic import Field, model_validator

from mantis._engine import mcts_max_armed_sims, mcts_max_armed_sims_mctx
from mantis.config.schema._base import StrictModel

#: The largest sim budget the MCTS node pool can serve, READ FROM THE ENGINE (AUDIT-1 F-21).
#:
#: `finish_expansion` panics on pool overflow, and `select_leaves` expands TT-hit leaves
#: without counting them against the batch, so one move can add up to
#: `4 * sims * MAX_CHILDREN_PER_NODE` children. `n_simulations` was `Field(ge=1)` with NO
#: ceiling, so a config could arm a budget that halts the run at the first move crossing it.
#: Derived from `MAX_NODES / (4 * MAX_CHILDREN_PER_NODE)` in `mantis-search` and read across
#: the bridge rather than re-typed — a literal here would be a second authority for a bound
#: only the pool knows, and it would go stale the day either constant moves.
MAX_ARMED_SIMS: int = mcts_max_armed_sims()

#: The same bound under the CORRECTED Gumbel dialect, which spends `MAX_ROOT_CHILDREN` pool
#: slots on its root instead of `MAX_CHILDREN_PER_NODE` (GUMBEL-REPAIR-1). It is LOWER, and
#: it is a second constant rather than a smaller shared one so the ceiling every existing
#: config validates against does not move for a dialect nothing arms. Field bounds below
#: keep the LOOSE value — a `Field(le=...)` cannot see a sibling key — and the tighter one
#: is applied by `_gumbel_dialect_fits_the_node_pool`.
MAX_ARMED_SIMS_MCTX: int = mcts_max_armed_sims_mctx()


class MctsConfig(StrictModel):
    """Flat-regime MCTS knobs (`# mcts ns` in `hparams.py`)."""

    n_simulations: int = Field(ge=1, le=MAX_ARMED_SIMS)
    c_puct: float = Field(gt=0)
    fpu_reduction: float
    quiescence_enabled: bool
    quiescence_blend_2: float = Field(ge=0, le=1)
    dirichlet_alpha: float = Field(gt=0)
    # The schema field IS the config key (`mcts.dirichlet_epsilon`) — retires hparams.py's
    # `mcts.epsilon`-vs-`dirichlet_epsilon` key/field-spelling mismatch by construction.
    dirichlet_epsilon: float = Field(ge=0, le=1)
    dirichlet_enabled: bool


class PlayoutCapConfig(StrictModel):
    """Playout-cap-randomization (PCR) knobs (`# playout_cap ns` in `hparams.py`)."""

    fast_sims: int = Field(ge=1, le=MAX_ARMED_SIMS)
    fast_prob: float = Field(ge=0, le=1)
    standard_sims: int = Field(ge=0, le=MAX_ARMED_SIMS)
    full_search_prob: float = Field(ge=0, le=1)
    n_sims_quick: int = Field(ge=0, le=MAX_ARMED_SIMS)
    n_sims_full: int = Field(ge=0, le=MAX_ARMED_SIMS)
    zoi_enabled: bool
    zoi_lookback: int = Field(ge=0)
    zoi_margin: int = Field(ge=0)
    # The schema field IS the config key — retires hparams.py's
    # `_resolve_playout_cap_temperature` key/field-spelling shim by construction.
    temperature_threshold_compound_moves: int = Field(ge=0)
    temp_min: float = Field(ge=0)

    @model_validator(mode="after")
    def _mutual_exclusion(self) -> "PlayoutCapConfig":
        # Ports hparams.py's two frozen hard errors onto the schema seam, plus the Phase-2
        # "PCR quick>full" RED-TEAM-lens check (REV1 MUST-FIX #4).
        if self.full_search_prob > 0.0 and self.fast_prob > 0.0:
            raise ValueError(
                "playout_cap: fast_prob and full_search_prob are mutually exclusive"
            )
        if self.full_search_prob > 0.0 and (self.n_sims_quick <= 0 or self.n_sims_full <= 0):
            raise ValueError(
                "playout_cap: full_search_prob > 0 requires n_sims_quick>0 and n_sims_full>0"
            )
        if (
            self.full_search_prob > 0.0
            and self.n_sims_quick > 0
            and self.n_sims_full > 0
            and self.n_sims_quick > self.n_sims_full
        ):
            raise ValueError(
                "playout_cap: n_sims_quick must be <= n_sims_full (quick>full is a "
                "nonsensical playout-cap-randomization preset)"
            )
        # V-PCR (R40, WPSC Phase 3 SC-B4): the two genuinely-missing PlayoutCapConfig
        # checks. Gated on "both presets are set" (n_sims_quick>0 and n_sims_full>0), NOT
        # on full_search_prob>0 — gating there would false-fire on every minted config's
        # all-zero disabled shape (0, 0, 0.0).
        if self.n_sims_quick > 0 and self.n_sims_full > 0:
            if self.n_sims_quick == self.n_sims_full:
                raise ValueError(
                    "playout_cap: n_sims_quick == n_sims_full is a no-op randomization "
                    "(quick and full presets must differ)"
                )
            if self.full_search_prob <= 0.0 or self.full_search_prob >= 1.0:
                raise ValueError(
                    "playout_cap: full_search_prob must be in (0, 1) when both n_sims_quick "
                    "and n_sims_full are configured (0 or 1 makes one preset permanently "
                    "unreachable — degenerate randomization)"
                )
        return self


class SelfplayConfig(StrictModel):
    """Self-play worker/search knobs (`# selfplay ns` + monitoring/instrumentation in
    `hparams.py`). See the module docstring for why no radius field exists here.

    ``gumbel_variant`` SELECTS A DIALECT, not a feature (GUMBEL-REPAIR-1). ``legacy`` is
    the arm every shipped config carries and reproduces the shipped behaviour exactly;
    ``mctx`` is the corrected arm, matching `google-deepmind/mctx`'s
    ``gumbel_muzero_policy`` on the points the repair verified as deviations — root
    sampling over the FULL legal set, the mixed-value completion taken off the root's RAW
    network value, min-max rescaled Q with Mctx's ``value_scale`` transform, completed-Q
    interior selection, and a Sequential-Halving schedule that consumes the budget
    exactly. It is ONE key rather than five because the corrections are what corrected
    Gumbel IS: R345(d) compares "corrected Gumbel" against PUCT as a single arm, and five
    independent booleans would mint 32 dialects nobody has measured.

    ``c_scale`` IS Mctx's ``value_scale`` under the ``mctx`` variant — the same slot in
    ``(c_visit + max_visits) * scale * q``, where ``c_visit`` is Mctx's ``maxvisit_init``.
    A second key for the same slot would be the duplicate-authority class R1 exists to
    kill. Their defaults differ by an order of magnitude and the difference is REAL: Mctx
    ships ``value_scale=0.1`` against rescaled Q in [0, 1], the legacy arm ships
    ``c_scale=1.0`` against raw Q in [-1, 1], so a mint that moves the variant without
    moving the scale changes how peaked every target is. The mint states the value; the
    schema will not guess it.

    ``gumbel_root_counts`` decides whether the root's own evaluation is charged against
    ``mcts.n_simulations``. ``true`` is the shipped behaviour (the root eval spends one of
    N, so the halving schedule gets N-1). It is a key and not a constant because it is
    exactly the term that makes "equal NN work" stateable when corrected Gumbel is
    compared against PUCT at a fixed leaf budget.
    """

    n_workers: int = Field(ge=1)
    leaf_batch_size: int = Field(ge=1)
    max_game_moves: int = Field(ge=1)
    inference_pool_size: int | None = Field(ge=1)
    completed_q_values: bool
    c_visit: float = Field(gt=0)
    c_scale: float = Field(gt=0)
    gumbel_mcts: bool
    gumbel_m: int = Field(ge=1)
    gumbel_explore_moves: int = Field(ge=0)
    gumbel_variant: Literal["legacy", "mctx"]
    gumbel_root_counts: bool
    results_queue_cap: int = Field(ge=1)
    random_opening_plies: int = Field(ge=0)
    rotation_enabled: bool
    forced_win_policy_enabled: bool
    forced_win_policy_depth: int = Field(ge=1)
    forced_win_policy_weight: float = Field(ge=0)
    solver_enabled: bool
    solver_depth: int = Field(ge=1)
    solver_node_budget: int = Field(ge=1)
    solver_neighbor_dist: int = Field(ge=0)
    solver_visit_weight: float = Field(ge=0, le=1)
    seed_fraction: float = Field(ge=0, le=1)
    seed_corpus_path: str | None
    log_investigation_metrics: bool
    instrumentation_enabled: bool
    mcts: MctsConfig
    playout_cap: PlayoutCapConfig

    @model_validator(mode="after")
    def _gumbel_dialect_fits_the_node_pool(self) -> "SelfplayConfig":
        """The CORRECTED Gumbel dialect's sim ceiling, refused at MINT and not at boot.

        The corrected arm expands its root over the full legal set, so it spends
        ``MAX_ROOT_CHILDREN`` pool slots on the root instead of ``MAX_CHILDREN_PER_NODE``
        and its ceiling is ``MAX_ARMED_SIMS_MCTX``, below the ``MAX_ARMED_SIMS`` the field
        bounds carry. A `Field(le=...)` cannot express this — the applicable ceiling depends
        on two sibling keys — so without this validator a config in the gap between the two
        bounds would validate clean and be refused by ``SelfPlayRunner::new`` at boot.

        That inversion is the one R255/ADJ-D34 closed for the HEXG visit capacity, in this
        same class, and re-opening it a section away would be the same defect wearing a
        different key's name. INERT while ``gumbel_mcts`` is false, which is every committed
        config.

        Raises:
            ValueError: an armed sims knob exceeds the corrected dialect's ceiling.
        """
        if not (self.gumbel_mcts and self.gumbel_variant == "mctx"):
            return self
        armed = {
            "selfplay.mcts.n_simulations": self.mcts.n_simulations,
            "selfplay.playout_cap.standard_sims": self.playout_cap.standard_sims,
            "selfplay.playout_cap.fast_sims": self.playout_cap.fast_sims,
            "selfplay.playout_cap.n_sims_quick": self.playout_cap.n_sims_quick,
            "selfplay.playout_cap.n_sims_full": self.playout_cap.n_sims_full,
        }
        over = {k: v for k, v in armed.items() if v > MAX_ARMED_SIMS_MCTX}
        if over:
            raise ValueError(
                "selfplay.gumbel_variant='mctx' lowers the node-pool sim ceiling to "
                f"{MAX_ARMED_SIMS_MCTX} (from {MAX_ARMED_SIMS}), because that dialect "
                "expands the root over its FULL legal set and spends MAX_ROOT_CHILDREN pool "
                "slots on it instead of MAX_CHILDREN_PER_NODE. Over the ceiling: "
                + ", ".join(f"{k}={v}" for k, v in sorted(over.items()))
                + " — lower the budget, or mint the legacy dialect."
            )
        return self


class FusedGraphCapsConfig(StrictModel):
    """The GRAPH inference forward's memory bound — ONE block, ONE fact (F-816-10, R276(f);
    `MicrobatchCapsConfig`'s shape applied to the other consumer of the same card).

    The fact is "how big may ONE fused inference forward be", and it has TWO INSEPARABLE
    components, which is why this is a nested block and not two flat keys: the members are
    sized TOGETHER from ONE measured cost model against ONE budget (`peak ~ a + b*E + c*N`,
    so `a + b*max_fused_edges + c*max_fused_nodes <= budget`), and two independent keys would
    give two authorities over one byte budget and let an operator mint one and forget the
    other.

    `inference.inference_batch_size` bounds the number of GRAPHS in a pop; it bounds neither
    quantity that drives memory. E and N are SUMS over the fused graphs, and Design A raised
    the fuse to `n_workers x leaf_batch_size` / `inference_batch_size` graphs without
    re-fitting the partner term of the budget the training cap was sized against — the
    inference term has no bound at all today, so any change that raises it invalidates the
    train-side fit. Both caps divide ONE card and must be re-fitted together.

    BOTH MEMBERS, because N is unbounded off-distribution by the builder's own arithmetic:
    two dummy edges per real node force `E >= 2(N-1)`, so an edge-only cap `C` admits
    `N <= C/2 + 1` — and at the measured per-node byte cost that unbounded member's worst
    case EXCEEDS the bounded member's. One member bounds neither term of `peak ~ a + b*E +
    c*N`.

    `ge=1` on the int arm and NO "uncapped" sentinel: the off state is deliberately
    unrepresentable, because an unbounded fused forward is the defect this block exists to
    make unconstructible and a disable sentinel would be a switch for turning the fix off
    (R79, `MicrobatchCapsConfig`'s recorded refusal). The bound is the mechanism's own range:
    a fused forward of zero edges is not a fused forward.

    `null` IS NOT AN OFF STATE. It is the R119 placeholder: schema-VALID, so gate 7 stays
    green and the repo ships a complete config, and runtime-REFUSED, so a graph run on an
    uncalibrated production config CANNOT CONSTRUCT ITS INFERENCE SERVER.
    `UncalibratedFusedGraphCapsError` names the member, the calibration entry point that
    produces the value and the `tools/mint_config.py --set` line that mints it. The value is
    the operator's act at the box sitting, from `python -m mantis.diagnostics.
    fusion_calibrate` — never a dispatcher's number on a mint-critical card. The in-repo
    precedent for a schema-valid, production-illegal placeholder awaiting an operator mint is
    `train.checkpoint_interval: 0` (R137) and `eval.random_floor_games: 0` (R147/R272(d));
    the difference — and it is an improvement on both — is that this one RAISES instead of
    running.

    GRAPH-ROUTE ONLY, AND NOW SCOPED BY THE SCHEMA RATHER THAN BY A CALL SITE (R322(d)): the
    scoping lives in `core.ARCH_SCOPED_KEYS` and is enforced by
    `RunConfig._arch_scoped_keys_are_present_iff_their_arch`, so a non-graph config carrying
    this block is REFUSED at validation instead of being required to mint one. The dense batch
    is a fixed-shape tensor already bounded by
    `inference_batch_size`, so there is no unbounded quantity there for a cap to bound. The
    five non-production configs mint values that are NON-BINDING BY CONSTRUCTION (derived
    from each template's own `max_game_moves` and the registry's widest legal-move radius,
    never chosen), because a smoke config whose cap bound would make CI exercise a split by
    accident and the split's coverage must come from the oracles.

    Read by ONE path: `mantis.config.resolve.fused_graph_caps.resolve_fused_graph_caps` ->
    `InferenceServer.__init__` (the GRAPH branch only, EAGERLY at construction) ->
    `_run_graph_loop`'s `plan_fused_forwards` partition. Eager and not lazy because `__init__`
    already branches on the representation, so the read is naturally route-scoped, and
    failing a mis-minted run in the first second beats failing it three hours in.
    """

    max_fused_edges: int | None = Field(ge=1)
    max_fused_nodes: int | None = Field(ge=1)


class InferenceConfig(StrictModel):
    """Inference-server knobs (`InferenceHParams`'s R1-exception, closed by SC-A2).

    A SIBLING of `SelfplayConfig` on `RunConfig` (not nested): `InferenceHParams` is already
    a fully separate dataclass from `SelfPlayHParams` at the Python level
    (`inference_server.py` builds it independently of `pool.py`'s `SelfPlayHParams`), so the
    schema stays 1:1 with that split.
    """

    inference_batch_size: int = Field(ge=1)
    inference_max_wait_ms: int = Field(ge=0)
    trace_inference: bool
    compile_inference: bool
    compile_inference_mode: str = Field(min_length=1)
    compile_inference_dynamic: bool
    perf_timing: bool
    perf_sync_cuda: bool
    # ARCH-SCOPED (R322(d)): `None` is the ABSENCE of the key, never a value — see
    # `TrainConfig.microbatch_caps` for the shape and `RunConfig` for the enforcement.
    # It does NOT collide with the R119 `null` PLACEHOLDER, which lives on the two
    # MEMBERS and means "minted but uncalibrated"; absence of the BLOCK means "this arch
    # has no such key", and the two are distinguished by `model_fields_set`.
    fused_graph_caps: FusedGraphCapsConfig | None = None
