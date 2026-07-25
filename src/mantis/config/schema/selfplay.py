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
from pydantic import Field, model_validator

from mantis.config.schema._base import StrictModel


class MctsConfig(StrictModel):
    """Flat-regime MCTS knobs (`# mcts ns` in `hparams.py`)."""

    n_simulations: int = Field(ge=1)
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

    fast_sims: int = Field(ge=1)
    fast_prob: float = Field(ge=0, le=1)
    standard_sims: int = Field(ge=0)
    full_search_prob: float = Field(ge=0, le=1)
    n_sims_quick: int = Field(ge=0)
    n_sims_full: int = Field(ge=0)
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
    `hparams.py`). See the module docstring for why no radius field exists here."""

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
