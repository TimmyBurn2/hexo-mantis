"""Self-play knob resolution: validated `SelfplayConfig`/`InferenceConfig` -> typed hparams ->
`SelfPlayRunnerConfig`. `from_config` reads a validated mapping's sections directly: no namespace
fallback, no code-side default, the schema being the sole default authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mantis._engine import SelfPlayRunnerConfig
from mantis.config.resolve.puct import resolve_puct_constants
from mantis.config.resolve.search import resolve_selfplay_search_kind
from mantis.config.resolve.tactics import resolve_selfplay_tactics
from mantis.encoding import EncodingSpec, resolve_from_config
from mantis.model import RepresentationMismatch


@dataclass(frozen=True)
class ResolvedPoolEncoding:
    """Every encoding-derived value the pool wires through the Rust runner."""

    registry_spec: Any  # EncodingSpec (full schema)
    encoding_name: str
    board_size: int
    trunk_size: int
    n_kept_planes: int


def is_graph_representation(spec: Any) -> bool:
    """Closed match on ``spec.representation`` — no dense-by-default arm. `"grid"` is
    REFUSED BY NAME rather than answered `False`, because a spec still declaring the deleted
    dense path would otherwise be handed a graph buffer.

    Args:
        spec: an encoding spec, or anything carrying a ``representation`` attribute.

    Returns:
        `True` — the one representation this project has.

    Raises:
        RepresentationMismatch: the representation is absent, `"grid"`, or unknown.
    """
    rep = getattr(spec, "representation", None)
    if rep == "graph":
        return True
    if rep == "grid":
        raise RepresentationMismatch(
            f"representation 'grid' on encoding spec {getattr(spec, 'name', spec)!r} — the "
            "dense path was DELETED (R346(f)) and `archive/grid-path` carries it. Answering "
            "this False would route the caller to a graph buffer under a dense declaration."
        )
    raise RepresentationMismatch(
        f"unknown representation {rep!r} on encoding spec "
        f"{getattr(spec, 'name', spec)!r} — self-play dispatches on a closed "
        "one-element set (axis graph) and has no default."
    )


def resolve_pool_encoding(
    config: dict[str, Any], arch: Any | None = None
) -> ResolvedPoolEncoding:
    """Resolve the pool's encoding, refusing an arch whose canvas ``board_size`` disagrees with it.
    Raises: MissingEncodingError, EncodingDeclarationConflictError; ValueError on that mismatch."""
    registry_spec: EncodingSpec = resolve_from_config(config)
    spec = registry_spec
    if arch is not None:
        arch_board_size = int(getattr(arch, "board_size", spec.board_size))
        if arch_board_size != spec.board_size:
            raise ValueError(
                f"WorkerPool: arch.board_size={arch_board_size} disagrees "
                f"with resolved encoding {spec.name!r} (board_size="
                f"{spec.board_size}). Fix the variant `encoding.version` or "
                f"the checkpoint hparam mismatch before re-launching."
            )
    return ResolvedPoolEncoding(
        registry_spec=registry_spec,
        encoding_name=spec.name,
        board_size=spec.board_size,
        trunk_size=spec.trunk_size,
        n_kept_planes=len(spec.kept_plane_indices),
    )


@dataclass(frozen=True, kw_only=True)
class SelfPlayHParams:
    """Every ctor-time self-play knob, resolved once; every field required and keyword-only."""

    # selfplay ns
    n_workers: int
    leaf_batch_size: int
    max_moves_per_game: int
    #: `selfplay.search.kind`, REQUIRED with no default: it selects the root mechanism, the interior
    #: selector AND the exported target's semantics, so a default would boot an undeclared regime.
    search_kind: str
    #: `selfplay.search.tactics` as the bridge arms it, `None` the explicit off: every worker tree's block.
    tactics: dict[str, Any] | None
    c_visit: float
    c_scale: float
    #: `selfplay.q_rescale`, REQUIRED with no default: the σ's rescale switch.
    q_rescale: bool
    #: `selfplay.search_stats_every`, REQUIRED with no default: 1-in-N games record their roots.
    search_stats_every: int
    gumbel_m: int
    #: `selfplay.gumbel_m_quick`: the quick arm's m.
    gumbel_m_quick: int
    gumbel_explore_moves: int
    results_queue_cap: int
    random_opening_plies: int
    # mcts ns
    c_puct: float
    fpu_reduction: float
    quiescence_enabled: bool
    quiescence_blend_2: float
    dirichlet_alpha: float
    dirichlet_epsilon: float  # field name == schema key (mcts.dirichlet_epsilon)
    dirichlet_enabled: bool
    # playout_cap ns
    full_search_prob: float
    n_sims_quick: int
    n_sims_full: int
    # The runner ctor kwarg spelling differs from the schema field name.
    temp_threshold_compound_moves: int
    temp_min: float  # field name == config key
    # monitoring / instrumentation ns
    log_investigation_metrics: bool

    @property
    def effective_sims_per_move(self) -> int:
        """Per-MOVE sims for the sims/sec bill: the full search's, an over-bill under an armed cap."""
        return self.n_sims_full

    @classmethod
    def from_config(
        cls, config: dict[str, Any], n_workers: int | None = None
    ) -> SelfPlayHParams:
        """Resolve every ctor-time knob off a validated mapping's `selfplay` section.
        An unvalidated mapping can carry `playout_cap.n_sims_full == 0`, so it is checked here.
        Raises: ValueError — no effective per-move sim count; `MissingTacticsError`, no `selfplay.search.tactics`."""
        sp = config["selfplay"]
        mcts_cfg = sp["mcts"]
        pc = sp["playout_cap"]
        puct = resolve_puct_constants(config)

        hp = cls(
            n_workers=int(n_workers if n_workers is not None else sp["n_workers"]),
            leaf_batch_size=int(sp["leaf_batch_size"]),
            max_moves_per_game=int(sp["max_game_moves"]),
            # THE self-play selector; the deploy head reads its own key.
            search_kind=resolve_selfplay_search_kind(config),
            tactics=resolve_selfplay_tactics(config),
            c_visit=float(sp["c_visit"]),
            c_scale=float(sp["c_scale"]),
            q_rescale=bool(sp["q_rescale"]),
            search_stats_every=int(sp["search_stats_every"]),
            gumbel_m=int(sp["gumbel_m"]),
            gumbel_m_quick=int(sp["gumbel_m_quick"]),
            gumbel_explore_moves=int(sp["gumbel_explore_moves"]),
            results_queue_cap=int(sp["results_queue_cap"]),
            random_opening_plies=int(sp["random_opening_plies"]),
            c_puct=puct.c_puct,
            fpu_reduction=puct.fpu_reduction,
            quiescence_enabled=puct.quiescence_enabled,
            quiescence_blend_2=puct.quiescence_blend_2,
            dirichlet_alpha=float(mcts_cfg["dirichlet_alpha"]),
            dirichlet_epsilon=float(mcts_cfg["dirichlet_epsilon"]),
            dirichlet_enabled=bool(mcts_cfg["dirichlet_enabled"]),
            full_search_prob=float(pc["full_search_prob"]),
            n_sims_quick=int(pc["n_sims_quick"]),
            n_sims_full=int(pc["n_sims_full"]),
            temp_threshold_compound_moves=int(pc["temperature_threshold_compound_moves"]),
            temp_min=float(pc["temp_min"]),
            log_investigation_metrics=bool(sp["log_investigation_metrics"]),
        )
        if hp.effective_sims_per_move <= 0:
            raise ValueError(
                "sims/sec: could not resolve the per-move sim count — "
                f"playout_cap.n_sims_full={hp.n_sims_full}; set it > 0."
            )
        return hp


@dataclass(frozen=True)
class InferenceHParams:
    """Every ctor-time inference-server knob, resolved once."""

    inference_batch_size: int
    inference_max_wait_ms: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceHParams:
        """Resolve every ctor-time knob off a validated mapping's `inference` section."""
        inf = config["inference"]
        return cls(
            inference_batch_size=int(inf["inference_batch_size"]),
            inference_max_wait_ms=int(inf["inference_max_wait_ms"]),
        )


def build_runner_config(
    hp: SelfPlayHParams,
    *,
    spec_dims: ResolvedPoolEncoding,
    encoding_name: str,
) -> SelfPlayRunnerConfig:
    """Assemble the Rust `SelfPlayRunnerConfig`. `feature_len`/`policy_len` are NOT passed: both
    are spec-derived Rust-side and the committed ctor rejects them, as it has no field for the two
    KILLed knobs either.

    Raises:
        RepresentationMismatch: the resolved spec is not a graph encoding.
        ValueError: the Rust config refuses a knob, including an unknown search kind or a malformed tactics block.
    """
    spec = spec_dims.registry_spec
    is_graph_representation(spec)

    cfg = SelfPlayRunnerConfig(
        n_workers=hp.n_workers,
        max_moves_per_game=hp.max_moves_per_game,
        n_simulations=hp.n_sims_full,
        leaf_batch_size=hp.leaf_batch_size,
        c_puct=hp.c_puct,
        fpu_reduction=hp.fpu_reduction,
        temp_threshold_compound_moves=hp.temp_threshold_compound_moves,
        quiescence_enabled=hp.quiescence_enabled,
        quiescence_blend_2=hp.quiescence_blend_2,
        temp_min=hp.temp_min,
        c_visit=hp.c_visit,
        c_scale=hp.c_scale,
        q_rescale=hp.q_rescale,
        search_stats_every=hp.search_stats_every,
        gumbel_m=hp.gumbel_m,
        gumbel_m_quick=hp.gumbel_m_quick,
        gumbel_explore_moves=hp.gumbel_explore_moves,
        dirichlet_alpha=hp.dirichlet_alpha,
        dirichlet_epsilon=hp.dirichlet_epsilon,
        dirichlet_enabled=hp.dirichlet_enabled,
        results_queue_cap=hp.results_queue_cap,
        full_search_prob=hp.full_search_prob,
        n_sims_quick=hp.n_sims_quick,
        random_opening_plies=hp.random_opening_plies,
        encoding_name=encoding_name,
    )
    # The Rust setter REFUSES an unknown search kind, so a typo is a boot error, not a PUCT search.
    cfg.search_kind = hp.search_kind
    cfg.configure_tactics(hp.tactics)
    return cfg


__all__ = [
    "InferenceHParams",
    "ResolvedPoolEncoding",
    "SelfPlayHParams",
    "build_runner_config",
    "is_graph_representation",
    "resolve_pool_encoding",
]
