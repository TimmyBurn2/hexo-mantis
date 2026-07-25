"""Self-play knob resolution: validated `SelfplayConfig`/`InferenceConfig` → typed hparams →
`SelfPlayRunnerConfig`.

>300 justify: ONE concern — everything the pool/server constructors used to read out of the
config used to be read inline. Keeping the two hparam dataclasses, the encoding resolve, the
seed-corpus parse and the runner-config assembly in one file means the config→runner wire
(write-only from Python: the Rust config exposes no ctor getters) is greppable in one place;
splitting it would scatter the very reads R-SELFPLAYCONFIG-SCHEMA inventoried.

**R-SELFPLAYCONFIG-SCHEMA closure (WPSC Phase 2 SC-A2).** `SelfPlayHParams.from_config` /
`InferenceHParams.from_config` now read a validated `RunConfig`-shaped mapping's `selfplay`/
`mcts`/`playout_cap`/`inference` sections (a `SelfplayConfig`/`InferenceConfig`
`.model_dump()`-shaped dict, or an equivalent explicit mapping) directly — no raw flat/legacy
dict, no top-level namespace fallback, no code-side `.get(k, default)`. The schema (`mantis.
config.schema.selfplay`) is the sole default authority (R1); the two hard-error checks that
used to live here (`fast_sims` required, `fast_prob`/`full_search_prob` mutual exclusion)
are now `PlayoutCapConfig` schema bounds/validators — not duplicated here (LAW-07: single
authority). `legal_move_radius_schedule` is gone from the schema entirely (DESIGN_P2.md §5);
nothing in this module ever read it.

`_resolve_playout_cap_temperature`'s key/field-spelling shim is RETIRED: the schema field IS
`temperature_threshold_compound_moves` (matching the config key one-to-one), so there is
nothing left to resolve — `from_config` reads `pc["temperature_threshold_compound_moves"]`/
`pc["temp_min"]` directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mantis._engine import SelfPlayRunnerConfig
from mantis.encoding import EncodingSpec, resolve_from_config
from mantis.model import RepresentationMismatch


# ---------------------------------------------------------------------------
# Seed corpus
# ---------------------------------------------------------------------------
def _load_seed_corpus(
    path: str | None, seed_fraction: float
) -> list[list[tuple[int, int]]] | None:
    """Parse the seed-corpus JSONL into move prefixes for the Rust runner.

    One JSON object per line; only ``seed_moves`` (a list of ``[q, r]`` pairs) is consumed
    here. Returns a list of ``(q, r)`` tuple-prefixes, or ``None`` when no path is
    configured.

    Loud ``ValueError`` on a malformed/empty corpus when ``seed_fraction > 0`` — a seeded
    run with no usable prefixes is a silent no-op the operator must catch. A path with
    ``seed_fraction == 0`` is parsed for validation but never fires.
    """
    if path is None:
        if seed_fraction > 0.0:
            raise ValueError(
                "selfplay.seed_fraction > 0 requires selfplay.seed_corpus_path — "
                "no corpus to seed from."
            )
        return None
    prefixes: list[list[tuple[int, int]]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                moves = obj["seed_moves"]
                prefixes.append([(int(q), int(r)) for q, r in moves])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"selfplay.seed_corpus_path {path!r} is malformed (expected JSONL with a "
            f"'seed_moves' list-of-[q,r] per line): {exc}"
        ) from exc
    if seed_fraction > 0.0 and not prefixes:
        raise ValueError(
            f"selfplay.seed_corpus_path {path!r} yielded ZERO prefixes but "
            f"seed_fraction={seed_fraction} > 0 — seeding would be a silent no-op."
        )
    return prefixes


# ---------------------------------------------------------------------------
# Encoding resolve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedPoolEncoding:
    """Every encoding-derived value the pool wires through the Rust runner.

      - ``registry_spec`` — canonical `mantis.encoding.EncodingSpec`.
      - ``encoding_name`` — registry name (e.g. "v6", "v6w25"), wired to the Rust
        `SelfPlayRunner` via ``encoding_name=``.
      - ``board_size`` / ``trunk_size`` / ``n_kept_planes`` — scalar dims reused for the
        buffer + reshape geometry.
    """

    registry_spec: Any  # EncodingSpec (full schema)
    encoding_name: str
    board_size: int
    trunk_size: int
    n_kept_planes: int


def is_graph_representation(spec: Any) -> bool:
    """Closed match on ``spec.representation`` — no dense-by-default arm (LAW-11 / AM-1).

    The frozen original defaulted an absent attribute to the DENSE kind, so an unknown or
    absent representation silently routed down the dense path. Every registry spec carries
    a required ``representation``, so on all reachable inputs this is byte-identical; the
    raising arm covers only inputs unreachable today (AM-1).
    """
    rep = getattr(spec, "representation", None)
    if rep == "graph":
        return True
    if rep == "grid":
        return False
    raise RepresentationMismatch(
        f"unknown representation {rep!r} on encoding spec "
        f"{getattr(spec, 'name', spec)!r} — self-play dispatches on a closed "
        "two-element set (dense grid / axis graph) and has no dense default."
    )


def resolve_pool_encoding(
    config: dict[str, Any], arch: Any | None = None
) -> ResolvedPoolEncoding:
    """Resolve every encoding-derived value the pool needs.

    ``board_size`` is canvas geometry (physical hex grid extent); ``trunk_size`` is the
    per-cluster NN-input window (== board_size for single-window encodings). All NN-input /
    buffer / reshape dims use ``trunk_size``; only the arch cross-check uses the canvas
    value.

    When ``arch`` is supplied, cross-checks its DECLARED ``board_size`` against the resolved
    canvas geometry — a mis-paired arch+config loud-fails with ``ValueError`` before any
    Rust runner is built. `GnnArch` declares no ``board_size`` and therefore passes
    vacuously, which is exactly the frozen behaviour for a graph net. Nothing is sniffed off
    a live `nn.Module`.
    """
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


# ---------------------------------------------------------------------------
# Hparam dataclasses (R-SELFPLAYCONFIG-SCHEMA — see the module header)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class SelfPlayHParams:
    """Every ctor-time self-play knob, resolved once. `kw_only` so the REQUIRED
    `fast_sims` (no default — a missing key is a hard error) can sit in its own namespace
    block rather than being hoisted above the defaulted fields."""

    # selfplay ns
    n_workers: int = 1
    leaf_batch_size: int = 8
    max_moves_per_game: int = 128
    inference_pool_size: int | None = None
    completed_q_values: bool = False
    c_visit: float = 50.0
    c_scale: float = 1.0
    gumbel_mcts: bool = False
    gumbel_m: int = 16
    gumbel_explore_moves: int = 10
    results_queue_cap: int = 10_000
    random_opening_plies: int = 0
    rotation_enabled: bool = True
    forced_win_policy_enabled: bool = False
    forced_win_policy_depth: int = 2
    forced_win_policy_weight: float = 1.0
    solver_enabled: bool = False
    solver_depth: int = 16
    solver_node_budget: int = 50_000
    solver_neighbor_dist: int = 2
    solver_visit_weight: float = 0.3
    seed_fraction: float = 0.0
    seed_corpus_path: str | None = None
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
    zoi_enabled: bool = False
    zoi_lookback: int = 16
    zoi_margin: int = 5
    # Runner ctor kwarg spelling differs from the schema field name
    # (`PlayoutCapConfig.temperature_threshold_compound_moves`); `from_config` reads the
    # schema field directly (no resolver shim — retired, R-SELFPLAYCONFIG-SCHEMA closure).
    temp_threshold_compound_moves: int = 0
    temp_min: float = 0.5                 # field name == config key
    # training ns
    draw_value: float = -0.5
    ply_cap_value: float = -0.5
    # monitoring / instrumentation ns
    log_investigation_metrics: bool = True
    instrumentation_enabled: bool = False

    @property
    def effective_sims_per_move(self) -> int:
        """Effective per-MOVE sim count for the sims/sec bill.

        Under a move-level playout cap, full-search moves cost ``n_sims_full`` and quick
        moves cost ``n_sims_quick``; we bill at ``n_sims_full`` per move (the full-search
        ceiling — quick moves are cheaper, so the running estimate is an over-bill bounded
        by ``n_sims_full``, never the falsified per-GAME under-bill). With no playout cap
        every move runs the flat ``n_simulations``. Both branches read existing config.
        """
        if self.full_search_prob > 0.0:
            return self.n_sims_full
        return self.n_simulations

    @classmethod
    def from_config(
        cls, config: dict[str, Any], n_workers: int | None = None
    ) -> SelfPlayHParams:
        """Resolve every ctor-time knob off a validated `RunConfig`-shaped mapping's
        `selfplay`/`train` sections (R-SELFPLAYCONFIG-SCHEMA closure) — direct nested reads,
        no top-level namespace fallback, no code-side default. `fast_sims`-required and the
        `fast_prob`/`full_search_prob` mutual-exclusion checks are now `PlayoutCapConfig`
        schema bounds/validators (not duplicated here, LAW-07); `effective_sims_per_move==0`
        has no schema equivalent (it spans `mcts.n_simulations` AND `playout_cap.*`), so it
        stays the one runtime hard error this resolver still raises.
        """
        sp = config["selfplay"]
        mcts_cfg = sp["mcts"]
        pc = sp["playout_cap"]
        train = config["train"]

        hp = cls(
            n_workers=int(n_workers if n_workers is not None else sp["n_workers"]),
            leaf_batch_size=int(sp["leaf_batch_size"]),
            max_moves_per_game=int(sp["max_game_moves"]),
            inference_pool_size=(
                int(sp["inference_pool_size"]) if sp["inference_pool_size"] is not None else None
            ),
            completed_q_values=bool(sp["completed_q_values"]),
            c_visit=float(sp["c_visit"]),
            c_scale=float(sp["c_scale"]),
            gumbel_mcts=bool(sp["gumbel_mcts"]),
            gumbel_m=int(sp["gumbel_m"]),
            gumbel_explore_moves=int(sp["gumbel_explore_moves"]),
            results_queue_cap=int(sp["results_queue_cap"]),
            random_opening_plies=int(sp["random_opening_plies"]),
            rotation_enabled=bool(sp["rotation_enabled"]),
            forced_win_policy_enabled=bool(sp["forced_win_policy_enabled"]),
            forced_win_policy_depth=int(sp["forced_win_policy_depth"]),
            forced_win_policy_weight=float(sp["forced_win_policy_weight"]),
            solver_enabled=bool(sp["solver_enabled"]),
            solver_depth=int(sp["solver_depth"]),
            solver_node_budget=int(sp["solver_node_budget"]),
            solver_neighbor_dist=int(sp["solver_neighbor_dist"]),
            solver_visit_weight=float(sp["solver_visit_weight"]),
            seed_fraction=float(sp["seed_fraction"]),
            seed_corpus_path=sp["seed_corpus_path"],
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
            zoi_enabled=bool(pc["zoi_enabled"]),
            zoi_lookback=int(pc["zoi_lookback"]),
            zoi_margin=int(pc["zoi_margin"]),
            temp_threshold_compound_moves=int(pc["temperature_threshold_compound_moves"]),
            temp_min=float(pc["temp_min"]),
            # Cross-section read (DESIGN_P2.md §2 note): draw_reward/ply_cap_value are part
            # of `pure_outcome_z`'s definition and live on TrainConfig, not SelfplayConfig.
            draw_value=float(train["draw_reward"]),
            ply_cap_value=float(train["ply_cap_value"]),
            log_investigation_metrics=bool(sp["log_investigation_metrics"]),
            instrumentation_enabled=bool(sp["instrumentation_enabled"]),
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
    trace_inference: bool = True
    compile_inference: bool = False
    compile_inference_mode: str = "default"
    compile_inference_dynamic: bool = True
    # diagnostics ns
    perf_timing: bool = False
    perf_sync_cuda: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InferenceHParams:
        """Resolve every ctor-time knob off a validated `RunConfig`-shaped mapping's
        `inference` section (R-SELFPLAYCONFIG-SCHEMA closure) — direct nested reads, no
        top-level namespace fallback."""
        inf = config["inference"]
        return cls(
            inference_batch_size=int(inf["inference_batch_size"]),
            inference_max_wait_ms=int(inf["inference_max_wait_ms"]),
            trace_inference=bool(inf["trace_inference"]),
            compile_inference=bool(inf["compile_inference"]),
            compile_inference_mode=str(inf["compile_inference_mode"]),
            compile_inference_dynamic=bool(inf["compile_inference_dynamic"]),
            perf_timing=bool(inf["perf_timing"]),
            perf_sync_cuda=bool(inf["perf_sync_cuda"]),
        )


# ---------------------------------------------------------------------------
# Runner-config assembly (the config→Rust wire; write-only from Python)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoolDims:
    """Dense NN-input / buffer dims derived from the resolved encoding.

    A graph spec has `n_planes=0` / `kept_plane_indices=[]`, so the dense feat/chain
    derivation is degenerate (not meaningful) and both are 0 — the graph drain branch never
    reads them.
    """

    feat_len: int
    chain_len: int
    pol_len: int


def build_runner_config(
    hp: SelfPlayHParams,
    *,
    spec_dims: ResolvedPoolEncoding,
    encoding_name: str,
    seed_prefixes: list[list[tuple[int, int]]] | None,
) -> tuple[SelfPlayRunnerConfig, PoolDims]:
    """Assemble the Rust `SelfPlayRunnerConfig` and the dense pool dims.

    `feature_len` / `policy_len` are NOT passed: both are spec-derived Rust-side (the runner
    derives them from ``encoding_name``) and the committed ctor rejects them. The two
    WP4/WP6-KILLed self-play knobs (the per-game radius jitter ctor kwarg and the interior
    selection-rule post-ctor attr) are likewise neither read nor set — the committed Rust
    config has neither field, and the census keeps both names dead.

    Dims are returned rather than assigned so the caller keeps a single source: grid →
    ``(n_kept_planes·trunk², 6·trunk², policy_logit_count)``; graph → ``(0, 0,
    policy_logit_count)``.
    """
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
        zoi_enabled=hp.zoi_enabled,
        zoi_lookback=hp.zoi_lookback,
        zoi_margin=hp.zoi_margin,
        completed_q_values=hp.completed_q_values,
        c_visit=hp.c_visit,
        c_scale=hp.c_scale,
        gumbel_mcts=hp.gumbel_mcts,
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
        selfplay_rotation_enabled=hp.rotation_enabled,
        encoding_name=encoding_name,
        inference_pool_size=hp.inference_pool_size,
    )
    # Forced-win → one-hot POLICY target. Set as config attributes rather than ctor kwargs
    # so the positional Rust ctor surface stays untouched. Default OFF.
    cfg.forced_win_policy_enabled = hp.forced_win_policy_enabled
    cfg.forced_win_policy_depth = hp.forced_win_policy_depth
    cfg.forced_win_policy_weight = hp.forced_win_policy_weight
    # Solver-in-loop SOFT visit-injection (visit_weight < 1.0 = SOFT, NOT one-hot).
    cfg.solver_enabled = hp.solver_enabled
    cfg.solver_depth = hp.solver_depth
    cfg.solver_node_budget = hp.solver_node_budget
    cfg.solver_neighbor_dist = hp.solver_neighbor_dist
    cfg.solver_visit_weight = hp.solver_visit_weight
    # Trap-corpus START-POSITION seeding: the JSONL corpus is parsed Python-side into move
    # prefixes; Rust dry-replays every prefix once at runner construction.
    cfg.seed_fraction = hp.seed_fraction
    if seed_prefixes is not None:
        cfg.seed_corpus = seed_prefixes
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
