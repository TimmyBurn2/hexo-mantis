"""Self-play knob resolution: config dict → typed hparams → `SelfPlayRunnerConfig`.

>300 justify: ONE concern — everything the pool/server constructors used to read out of the
raw config dict inline. Keeping the two hparam dataclasses, the encoding resolve, the
seed-corpus parse and the runner-config assembly in one file means the config→runner wire
(write-only from Python: the Rust config exposes no ctor getters) is greppable in one place;
splitting it would scatter the very reads R-SELFPLAYCONFIG-SCHEMA exists to inventory.

**Tracked R1-exception — R-SELFPLAYCONFIG-SCHEMA.** `SelfPlayHParams` / `InferenceHParams`
carry code-side field defaults that reproduce the frozen inline `.get()` defaults verbatim.
CLAUDE.md R1 ("no code-side defaults; a default lives only in the schema field") is
therefore **NOT intact for these knobs** — this is the operator-pinned Option A exception,
owed before the run5 config mint and retiring in the WP8 schema-extension commit alongside
R-TRAINCONFIG-SCHEMA. `legal_move_radius_schedule` is NOT here: it stays a real config key
read through the committed radius resolver.

`_resolve_playout_cap_temperature` is a byte-exact LOCAL copy of the frozen self-play
temperature resolver (`src/mantis/config/` is out of this work package's write scope);
relocation to `config/resolve/temperature.py` rides the same schema-extension commit. No
EVAL-temperature surface is created here — self-play schedule only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mantis._engine import SelfPlayRunnerConfig
from mantis.encoding import EncodingSpec, resolve_from_config
from mantis.model import RepresentationMismatch


# ---------------------------------------------------------------------------
# Temperature resolver — byte-exact local copy (relocation debt tracked).
# ---------------------------------------------------------------------------
def _resolve_playout_cap_temperature(pc: dict[str, Any]) -> tuple[int, float]:
    """Resolve ``(temp_threshold_compound_moves, temp_min)`` from a ``playout_cap`` dict.

    Fallback = cosine-OFF ``(0, 0.5)`` — mirrors the Rust ``SelfPlayRunnerConfig`` default
    and the documented production posture: a variant that omits these keys inherits a
    constant tau=0.5, and must NOT silently re-arm the draw-collapse cosine (the legacy
    fallback was the toxic 15 / 0.05). Schedule-ON values pass through unchanged.

    NOTE the key/field asymmetry this resolver exists to absorb: the CONFIG KEY is
    ``temperature_threshold_compound_moves``; the runner ctor kwarg (and the
    `SelfPlayHParams` field) is ``temp_threshold_compound_moves``. Reading the shorter
    spelling off the config returns None on every config → this fallback fires → an
    operator's temperature schedule is silently disabled with no error.
    """
    thr = pc.get("temperature_threshold_compound_moves")
    tmin = pc.get("temp_min")
    return (
        int(thr) if thr is not None else 0,       # absent OR explicit null -> OFF
        float(tmin) if tmin is not None else 0.5,
    )


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
    dirichlet_epsilon: float = 0.25       # CONFIG KEY: `mcts.epsilon` (not the field name)
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
    # CONFIG KEY: `playout_cap.temperature_threshold_compound_moves` — NOT this field's
    # name. Resolved through `_resolve_playout_cap_temperature`; reading the field spelling
    # off the config returns None on every config and silently disables the schedule.
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
        """Resolve every ctor-time knob, with the frozen namespace fallbacks and the four
        frozen hard errors, in the frozen order."""
        sp = config.get("selfplay", config)
        mcts_cfg = config.get("mcts", config)
        training_cfg = config.get("training", config)
        mon_cfg = config.get("monitoring", config)
        instr_cfg = config.get("instrumentation", {}) or {}

        pc = sp.get("playout_cap", config.get("playout_cap", {}))
        if "fast_sims" not in pc:
            raise ValueError(
                "playout_cap.fast_sims must be set in the self-play config — "
                "no silent defaults"
            )

        # Move-level and game-level playout caps are mutually exclusive: move-level
        # (full_search_prob) overrides the game-level (fast_prob/fast_sims) sim selection
        # inside the worker loop, so running both at once silently ignores the latter.
        fast_prob_cfg = float(pc.get("fast_prob", 0.0))
        full_search_prob_cfg = float(pc.get("full_search_prob", 0.0))
        n_sims_quick_cfg = int(pc.get("n_sims_quick", 0))
        n_sims_full_cfg = int(pc.get("n_sims_full", 0))
        if full_search_prob_cfg > 0.0 and fast_prob_cfg > 0.0:
            raise ValueError(
                "playout_cap: fast_prob and full_search_prob are mutually exclusive — "
                "move-level cap (full_search_prob) overrides game-level cap (fast_prob). "
                f"Got fast_prob={fast_prob_cfg}, full_search_prob={full_search_prob_cfg}. "
                "Set one of them to 0 in the self-play config."
            )
        if full_search_prob_cfg > 0.0 and (n_sims_quick_cfg <= 0 or n_sims_full_cfg <= 0):
            raise ValueError(
                "playout_cap: full_search_prob > 0 requires n_sims_quick > 0 AND "
                "n_sims_full > 0. "
                f"Got full_search_prob={full_search_prob_cfg}, "
                f"n_sims_quick={n_sims_quick_cfg}, n_sims_full={n_sims_full_cfg}."
            )

        # Within-game temperature defaults to cosine-OFF (0, 0.5) when a variant omits the
        # playout_cap keys — never the legacy 15 / 0.05.
        temp_threshold, temp_min = _resolve_playout_cap_temperature(pc)

        inference_pool_size = sp.get("inference_pool_size", None)
        if inference_pool_size is not None:
            inference_pool_size = int(inference_pool_size)

        hp = cls(
            n_workers=int(n_workers if n_workers is not None else sp.get("n_workers", 1)),
            leaf_batch_size=int(sp.get("leaf_batch_size", 8)),
            max_moves_per_game=int(
                sp.get("max_game_moves", sp.get("max_moves_per_game", 128))
            ),
            inference_pool_size=inference_pool_size,
            completed_q_values=bool(sp.get("completed_q_values", False)),
            c_visit=float(sp.get("c_visit", 50.0)),
            c_scale=float(sp.get("c_scale", 1.0)),
            gumbel_mcts=bool(sp.get("gumbel_mcts", False)),
            gumbel_m=int(sp.get("gumbel_m", 16)),
            gumbel_explore_moves=int(sp.get("gumbel_explore_moves", 10)),
            results_queue_cap=int(sp.get("results_queue_cap", 10_000)),
            random_opening_plies=int(sp.get("random_opening_plies", 0)),
            rotation_enabled=bool(sp.get("rotation_enabled", True)),
            forced_win_policy_enabled=bool(sp.get("forced_win_policy_enabled", False)),
            forced_win_policy_depth=int(sp.get("forced_win_policy_depth", 2)),
            forced_win_policy_weight=float(sp.get("forced_win_policy_weight", 1.0)),
            solver_enabled=bool(sp.get("solver_enabled", False)),
            solver_depth=int(sp.get("solver_depth", 16)),
            solver_node_budget=int(sp.get("solver_node_budget", 50_000)),
            solver_neighbor_dist=int(sp.get("solver_neighbor_dist", 2)),
            solver_visit_weight=float(sp.get("solver_visit_weight", 0.3)),
            seed_fraction=float(sp.get("seed_fraction", 0.0)),
            seed_corpus_path=sp.get("seed_corpus_path", None),
            n_simulations=int(
                mcts_cfg.get("n_simulations", config.get("n_simulations", 50))
            ),
            c_puct=float(mcts_cfg.get("c_puct", 1.5)),
            fpu_reduction=float(mcts_cfg.get("fpu_reduction", 0.25)),
            quiescence_enabled=bool(mcts_cfg.get("quiescence_enabled", True)),
            quiescence_blend_2=float(mcts_cfg.get("quiescence_blend_2", 0.3)),
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(mcts_cfg.get("epsilon", 0.25)),
            dirichlet_enabled=bool(mcts_cfg.get("dirichlet_enabled", True)),
            fast_sims=int(pc["fast_sims"]),
            fast_prob=fast_prob_cfg,
            standard_sims=int(pc.get("standard_sims", 0)),
            full_search_prob=full_search_prob_cfg,
            n_sims_quick=n_sims_quick_cfg,
            n_sims_full=n_sims_full_cfg,
            zoi_enabled=bool(pc.get("zoi_enabled", False)),
            zoi_lookback=int(pc.get("zoi_lookback", 16)),
            zoi_margin=int(pc.get("zoi_margin", 5)),
            temp_threshold_compound_moves=temp_threshold,
            temp_min=temp_min,
            draw_value=float(training_cfg.get("draw_value", -0.5)),
            # ply_cap_value is split from draw_reward so the value head sees distinct
            # targets for organic draws vs ply-cap truncations; falls back to draw_value.
            ply_cap_value=float(
                training_cfg.get("ply_cap_value", training_cfg.get("draw_value", -0.5))
            ),
            log_investigation_metrics=bool(mon_cfg.get(
                "log_investigation_metrics",
                config.get("log_investigation_metrics", True),
            )),
            instrumentation_enabled=bool(instr_cfg.get("enabled", False)),
        )
        if hp.effective_sims_per_move <= 0:
            raise ValueError(
                "sims/sec: could not resolve effective per-move sim count — "
                f"full_search_prob={full_search_prob_cfg}, "
                f"n_sims_full={n_sims_full_cfg}, n_simulations={hp.n_simulations}. "
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
        sp = config.get("selfplay", config)
        raw_diag = config.get("diagnostics")
        diag = raw_diag if isinstance(raw_diag, dict) else {}
        return cls(
            inference_batch_size=int(sp.get("inference_batch_size", 64)),
            inference_max_wait_ms=int(float(sp.get("inference_max_wait_ms", 10.0))),
            trace_inference=bool(sp.get("trace_inference", True)),
            compile_inference=bool(sp.get("compile_inference", False)),
            compile_inference_mode=str(sp.get("compile_inference_mode", "default")),
            compile_inference_dynamic=bool(sp.get("compile_inference_dynamic", True)),
            perf_timing=bool(diag.get("perf_timing", False)),
            perf_sync_cuda=bool(diag.get("perf_sync_cuda", False)),
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
