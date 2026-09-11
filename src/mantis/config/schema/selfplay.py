"""Self-play and inference knobs as first-class schema fields.

`legal_move_radius` / `legal_move_radius_schedule` are DELIBERATELY ABSENT: the encoding registry
alone is the radius authority, and nothing in the build path reads a config-level override, so a
schema field for it would be a consumer-less knob (R1/LAW-08).
"""

from typing import Literal

from pydantic import Field, model_validator

from mantis._engine import mcts_max_armed_sims, mcts_max_armed_sims_gumbel
from mantis.config.schema._base import StrictModel

#: The largest sim budget the MCTS node pool can serve, READ FROM THE ENGINE. `finish_expansion`
#: panics on pool overflow and `select_leaves` expands TT-hit leaves without counting them, so one
#: move can add up to `4 * sims * MAX_CHILDREN_PER_NODE` children. Derived across the bridge rather
#: than re-typed: a literal would be a second authority for a bound only the pool knows.
MAX_ARMED_SIMS: int = mcts_max_armed_sims()

#: The same bound under `search.kind: gumbel`, which spends `MAX_ROOT_CHILDREN` pool slots on its
#: root. It is LOWER, and it is a second constant rather than a smaller shared one so the ceiling a
#: PUCT config validates against does not move. The field bounds below keep the LOOSE value —
#: a `Field(le=...)` cannot see a key in another SECTION — and `RunConfig` applies the tighter one.
MAX_ARMED_SIMS_GUMBEL: int = mcts_max_armed_sims_gumbel()


class MctsConfig(StrictModel):
    """Flat-regime MCTS knobs (`# mcts ns` in `hparams.py`)."""

    n_simulations: int = Field(ge=1, le=MAX_ARMED_SIMS)
    c_puct: float = Field(gt=0)
    fpu_reduction: float
    quiescence_enabled: bool
    quiescence_blend_2: float = Field(ge=0, le=1)
    dirichlet_alpha: float = Field(gt=0)
    # The schema field IS the config key, which retires the old key/field spelling mismatch.
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
    # The schema field IS the config key, which retires the old resolver shim.
    temperature_threshold_compound_moves: int = Field(ge=0)
    temp_min: float = Field(ge=0)

    @model_validator(mode="after")
    def _mutual_exclusion(self) -> "PlayoutCapConfig":
        # The two frozen hard errors, plus the "PCR quick > full" check.
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
        # Gated on "both presets are set", NOT on `full_search_prob > 0`: gating there would
        # false-fire on every minted config's all-zero disabled shape.
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
    """Self-play worker/search knobs. See the module docstring for why no radius field exists.

    ``c_scale`` IS Mctx's ``value_scale`` under ``search.kind: gumbel`` — the same slot as
    ``c_visit``'s ``maxvisit_init`` — so a second key for it would be the duplicate-authority
    class. BOTH ARE REQUIRED WITH NO DEFAULT: the published board-game setting (cvisit = 50,
    cscale = 1.0) and the mctx library's Atari default (0.1) differ by an order of magnitude, and
    which one a run arms changes how peaked every exported target is.

    ``gumbel_m`` is Mctx's ``max_num_considered_actions`` and ``gumbel_explore_moves`` the span of
    opening plies that sample instead of taking the Sequential-Halving winner; both are inert
    under ``search.kind: puct``.
    """

    n_workers: int = Field(ge=1)
    leaf_batch_size: int = Field(ge=1)
    max_game_moves: int = Field(ge=1)
    c_visit: float = Field(gt=0)
    c_scale: float = Field(gt=0)
    gumbel_m: int = Field(ge=1)
    gumbel_explore_moves: int = Field(ge=0)
    # OPERATIONAL CONSTANT: a queue's back-pressure bound.
    results_queue_cap: int = Field(default=10000, ge=1)
    random_opening_plies: int = Field(ge=0)
    # OPERATIONAL CONSTANT: a diagnostic verbosity switch. It gates only what is WRITTEN, never
    # what is played, which is what makes it operational rather than a disabled-opponent knob.
    log_investigation_metrics: bool = True
    mcts: MctsConfig
    playout_cap: PlayoutCapConfig


class FusedGraphCapsConfig(StrictModel):
    """The GRAPH inference forward's memory bound — ONE block, ONE fact.

    The fact is "how big may ONE fused inference forward be", and its TWO components are
    INSEPARABLE: they are sized TOGETHER from ONE measured cost model against ONE budget
    (`peak ~ a + b*E + c*N`), so two flat keys would be two authorities over one byte budget.
    `inference_batch_size` bounds the number of GRAPHS in a pop and neither quantity that drives
    memory. BOTH MEMBERS, because N is unbounded off-distribution by the builder's own arithmetic:
    two dummy edges per real node force `E >= 2(N-1)`, so an edge-only cap `C` admits
    `N <= C/2 + 1`, whose worst case exceeds the bounded member's at the measured per-node cost.

    `ge=1` and NO "uncapped" sentinel: the off state is deliberately unrepresentable. `null` is not
    an off state either — it is the placeholder that is schema-VALID, so the repo ships a complete
    config, and runtime-REFUSED, so a graph run on an uncalibrated production config cannot
    construct its inference server; the error names the member, the calibration entry point and the
    mint line.

    GRAPH-ROUTE ONLY, scoped by the schema rather than by a call site, so a non-graph config
    carrying this block is REFUSED at validation. Read by ONE path, EAGERLY at
    `InferenceServer.__init__`'s graph branch, because failing a mis-minted run in the first second
    beats failing it three hours in.
    """

    max_fused_edges: int | None = Field(ge=1)
    max_fused_nodes: int | None = Field(ge=1)


class InferenceConfig(StrictModel):
    """Inference-server knobs. A SIBLING of `SelfplayConfig` on `RunConfig`, not nested: the two
    hparams dataclasses are already fully separate at the Python level, so the schema stays 1:1
    with that split."""

    inference_batch_size: int = Field(ge=1)
    inference_max_wait_ms: int = Field(ge=0)
    # R347(e)'s lever: check 14 (`verify_edge_geometry`) runs inline on the server's critical path
    # or on a checker thread after the batch is served; it runs on EVERY batch either way.
    edge_geometry_check: Literal["inline", "checker_thread"] = Field(default="inline")
    # A4-3: the SERVING trunk runs through `torch.compile(dynamic=True)`; the trainer and the eval
    # child stay eager. Off is the shipped numerics; on is eager-to-bf16-noise, never bit-exact.
    compile_trunk: bool = Field(default=False)
    # ARCH-SCOPED: `None` is the ABSENCE of the key, never a value. It does NOT collide with the
    # `null` PLACEHOLDER on the two MEMBERS, which means "minted but uncalibrated"; absence of the
    # BLOCK means "this arch has no such key", and the two are distinguished by `model_fields_set`.
    fused_graph_caps: FusedGraphCapsConfig | None = None
