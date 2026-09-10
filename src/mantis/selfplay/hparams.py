"""Self-play knob resolution: validated `SelfplayConfig`/`InferenceConfig` -> typed hparams ->
`SelfPlayRunnerConfig`.

>300 justify: ONE concern — everything the pool/server constructors used to read inline out of
the config. Keeping the hparam dataclasses, the encoding resolve, the seed-corpus parse and the
runner-config assembly together makes the config->runner wire greppable in one place, and that
wire is write-only from Python. `from_config` reads a validated mapping's sections directly: no
namespace fallback, no code-side default, the schema being the sole default authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mantis._engine import SelfPlayRunnerConfig
from mantis.config.resolve.search import resolve_search_kind
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
    """Closed match on ``spec.representation`` — no dense-by-default arm (LAW-11). `"grid"` is
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
    """Resolve every encoding-derived value the pool needs: ``board_size`` is canvas geometry and
    ``trunk_size`` the per-cluster NN-input window that all buffer and reshape dims use. Only the
    arch cross-check reads the canvas value, so a mis-paired arch loud-fails before any runner."""
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
    """Every ctor-time self-play knob, resolved once; `kw_only` so REQUIRED `fast_sims` fits."""

    # selfplay ns
    n_workers: int = 1
    leaf_batch_size: int = 8
    max_moves_per_game: int = 128
    #: `search.kind`, REQUIRED with no default: it selects the root mechanism, the interior
    #: selector AND the exported target's semantics, so a default would boot an undeclared regime.
    search_kind: str
    c_visit: float = 50.0
    c_scale: float = 1.0
    gumbel_m: int = 16
    gumbel_explore_moves: int = 10
    results_queue_cap: int = 10_000
    random_opening_plies: int = 0
    # mcts ns
    n_simulations: int = 50
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    quiescence_enabled: bool = True
    quiescence_blend_2: float = 0.3
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25       # field name == schema key (mcts.dirichlet_epsilon)
    dirichlet_enabled: bool = True
    # playout_cap ns — fast_sims REQUIRED (no default; missing key = ValueError)
    fast_sims: int
    fast_prob: float = 0.0
    standard_sims: int = 0
    full_search_prob: float = 0.0
    n_sims_quick: int = 0
    n_sims_full: int = 0
    # The runner ctor kwarg spelling differs from the schema field name.
    temp_threshold_compound_moves: int = 0
    temp_min: float = 0.5                 # field name == config key
    # training ns
    draw_value: float = -0.5
    ply_cap_value: float = -0.5
    # monitoring / instrumentation ns
    log_investigation_metrics: bool = True

    @property
    def effective_sims_per_move(self) -> int:
        """Effective per-MOVE sim count for the sims/sec bill: under a playout cap it bills at
        the full-search ceiling, an over-bill, never the falsified per-GAME under-bill."""
        if self.full_search_prob > 0.0:
            return self.n_sims_full
        return self.n_simulations

    @classmethod
    def from_config(
        cls, config: dict[str, Any], n_workers: int | None = None
    ) -> SelfPlayHParams:
        """Resolve every ctor-time knob off a validated mapping's `selfplay`/`train` sections.
        `effective_sims_per_move == 0` has no schema equivalent, since it spans
        `mcts.n_simulations` AND `playout_cap.*`, so it stays the one runtime hard error here."""
        sp = config["selfplay"]
        mcts_cfg = sp["mcts"]
        pc = sp["playout_cap"]
        train = config["train"]

        hp = cls(
            n_workers=int(n_workers if n_workers is not None else sp["n_workers"]),
            leaf_batch_size=int(sp["leaf_batch_size"]),
            max_moves_per_game=int(sp["max_game_moves"]),
            # THE ONE SELECTOR, shared with `build_eval_pipeline`, so the bar and the workers
            # cannot read two call sites that happen to agree.
            search_kind=resolve_search_kind(config),
            c_visit=float(sp["c_visit"]),
            c_scale=float(sp["c_scale"]),
            gumbel_m=int(sp["gumbel_m"]),
            gumbel_explore_moves=int(sp["gumbel_explore_moves"]),
            results_queue_cap=int(sp["results_queue_cap"]),
            random_opening_plies=int(sp["random_opening_plies"]),
            n_simulations=int(mcts_cfg["n_simulations"]),
            c_puct=float(mcts_cfg["c_puct"]),
            fpu_reduction=float(mcts_cfg["fpu_reduction"]),
            quiescence_enabled=bool(mcts_cfg["quiescence_enabled"]),
            quiescence_blend_2=float(mcts_cfg["quiescence_blend_2"]),
            dirichlet_alpha=float(mcts_cfg["dirichlet_alpha"]),
            dirichlet_epsilon=float(mcts_cfg["dirichlet_epsilon"]),
            dirichlet_enabled=bool(mcts_cfg["dirichlet_enabled"]),
            fast_sims=int(pc["fast_sims"]),
            fast_prob=float(pc["fast_prob"]),
            standard_sims=int(pc["standard_sims"]),
            full_search_prob=float(pc["full_search_prob"]),
            n_sims_quick=int(pc["n_sims_quick"]),
            n_sims_full=int(pc["n_sims_full"]),
            temp_threshold_compound_moves=int(pc["temperature_threshold_compound_moves"]),
            temp_min=float(pc["temp_min"]),
            # Cross-section read: draw_reward/ply_cap_value define `pure_outcome_z`.
            draw_value=float(train["draw_reward"]),
            ply_cap_value=float(train["ply_cap_value"]),
            log_investigation_metrics=bool(sp["log_investigation_metrics"]),
        )
        if hp.effective_sims_per_move <= 0:
            raise ValueError(
                "sims/sec: could not resolve effective per-move sim count — "
                f"full_search_prob={hp.full_search_prob}, "
                f"n_sims_full={hp.n_sims_full}, n_simulations={hp.n_simulations}. "
                "Set mcts.n_simulations > 0 (flat regime) or "
                "playout_cap.n_sims_full > 0 (move-level cap regime)."
            )
        return hp


@dataclass(frozen=True)
class InferenceHParams:
    """Every ctor-time inference-server knob (same R1-exception as `SelfPlayHParams`)."""

    inference_batch_size: int = 64
    inference_max_wait_ms: int = 10
    # diagnostics ns

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceHParams:
        """Resolve every ctor-time knob off a validated mapping's `inference` section."""
        inf = config["inference"]
        return cls(
            inference_batch_size=int(inf["inference_batch_size"]),
            inference_max_wait_ms=int(inf["inference_max_wait_ms"]),
        )


@dataclass(frozen=True)
class PoolDims:
    """Dense NN-input and buffer dims from the resolved encoding; both 0 on a graph spec."""

    feat_len: int
    chain_len: int
    pol_len: int


def build_runner_config(
    hp: SelfPlayHParams,
    *,
    spec_dims: ResolvedPoolEncoding,
    encoding_name: str,
) -> tuple[SelfPlayRunnerConfig, PoolDims]:
    """Assemble the Rust `SelfPlayRunnerConfig` and the dense pool dims. `feature_len`/
    `policy_len` are NOT passed: both are spec-derived Rust-side and the committed ctor rejects
    them, as it has no field for the two KILLed knobs either."""
    spec = spec_dims.registry_spec
    trunk_size = spec_dims.trunk_size
    if is_graph_representation(spec):
        dims = PoolDims(0, 0, int(spec.policy_logit_count))
    else:
        dims = PoolDims(
            spec_dims.n_kept_planes * trunk_size * trunk_size,
            6 * trunk_size * trunk_size,
            int(spec.policy_logit_count),
        )

    cfg = SelfPlayRunnerConfig(
        n_workers=hp.n_workers,
        max_moves_per_game=hp.max_moves_per_game,
        n_simulations=hp.n_simulations,
        leaf_batch_size=hp.leaf_batch_size,
        c_puct=hp.c_puct,
        fpu_reduction=hp.fpu_reduction,
        fast_prob=hp.fast_prob,
        fast_sims=hp.fast_sims,
        standard_sims=hp.standard_sims,
        temp_threshold_compound_moves=hp.temp_threshold_compound_moves,
        draw_reward=hp.draw_value,
        ply_cap_value=hp.ply_cap_value,
        quiescence_enabled=hp.quiescence_enabled,
        quiescence_blend_2=hp.quiescence_blend_2,
        temp_min=hp.temp_min,
        c_visit=hp.c_visit,
        c_scale=hp.c_scale,
        gumbel_m=hp.gumbel_m,
        gumbel_explore_moves=hp.gumbel_explore_moves,
        dirichlet_alpha=hp.dirichlet_alpha,
        dirichlet_epsilon=hp.dirichlet_epsilon,
        dirichlet_enabled=hp.dirichlet_enabled,
        results_queue_cap=hp.results_queue_cap,
        full_search_prob=hp.full_search_prob,
        n_sims_quick=hp.n_sims_quick,
        n_sims_full=hp.n_sims_full,
        random_opening_plies=hp.random_opening_plies,
        encoding_name=encoding_name,
    )
    # The Rust setter REFUSES an unknown search kind, so a typo is a boot error, not a PUCT search.
    cfg.search_kind = hp.search_kind
    return cfg, dims


__all__ = [
    "InferenceHParams",
    "PoolDims",
    "ResolvedPoolEncoding",
    "SelfPlayHParams",
    "build_runner_config",
    "is_graph_representation",
    "resolve_pool_encoding",
]
