"""Suite H (surface) — H-01, H-12 — plus the two Suite-E rows that assert on the POOL.

>300 justify: one object, one contract. The frozen consumer surface (H-01), the import-DAG
proof (H-12), the facade-wrapping pin (E-07) and the per-buffer-kind composition rule
(E-05's pool half) all bind `WorkerPool` construction and share the same real-pool factory
plus the captured buffer fill; splitting them would duplicate both and let the two copies
drift.

IMPL-written (non-⊕) per DESIGN §b. E-07 and E-05's pool half were recorded as OWED by the
slice that built the facade — the facade side of the guard is pinned there, the pool side
is pinned here.

Why the whole file matters: the trainer duck-types the pool as `Any` and reads ~20 members
off it. Nothing type-checks that seam, so a member renamed or dropped during this split is
invisible until the first integrated run, where it appears as an `AttributeError` several
minutes into training. H-01 is the only mechanical check that the split preserved it.
"""
from __future__ import annotations

import ast
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.selfplay.buffers import BufferKind, BufferKindMismatch, ReplayFacade
from mantis.selfplay.pool import WorkerPool
from mantis.selfplay.pool_hooks import ActorSyncTarget, InferenceStats, RunnerStats
from mantis.train.coordinator.config import WorkerPoolLike

SELFPLAY_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"

# The captured `#C3e` deterministic fill (PREREG §3 CORRECTION 2 table): capacity 64,
# encoding v6, 40 rows — 10 organic draws at −0.5, 8 ply-cap truncations at −0.7, 10 wins
# at +1.0, 12 losses at −1.0 — with 30 of the rows attributed to self-play.
FILL_CAPACITY = 64
FILL_ROWS = 40
FILL_SELF_PLAY_PUSHED = 30
# The captured terminal-reason script: 5 six-in-a-row, 3 colony, 2 ply-cap, 1 other-draw.
TERMINAL_SCRIPT = [0] * 5 + [1] * 3 + [2] * 2 + [3] * 1
# `buffer_composition()` on that fill under `train.draw_reward=-0.5, ply_cap_value=-0.7`.
EXPECTED_COMPOSITION = {
    "buffer_size": 40,
    "buffer_capacity": 64,
    "corpus_fraction": 0.25,
    "draw_target_fraction": 0.45,
    "six_terminal_fraction": 5 / 11,
    "colony_terminal_fraction": 3 / 11,
    "cap_terminal_fraction": 2 / 11,
    "other_draw_fraction": 1 / 11,
    "n_games_observed": 11,
}

# Every member DESIGN §a.3 enumerates from the committed trainer, with the kind the
# trainer uses it as. A missing row here is a runtime break at first integration.
FROZEN_ATTRS = ("games_completed", "draws", "n_workers", "recent_buffer", "encoding_spec")
FROZEN_PROPERTIES = (
    "recent_move_histories", "avg_game_length", "search_kind", "x_winrate", "o_winrate",
    # F-816-2: the THIRD outcome share, and a property for the same reason its two siblings
    # are — the denominator belongs to the pool. `iteration_complete` used to divide the raw
    # `draws` attribute by the coordinator's stale game count and emitted values above 1.
    "draw_rate",
    "sims_per_sec", "batch_fill_pct",
)
FROZEN_METHODS = (
    "runner_stats", "inference_stats", "check_producer_health", "stop", "start",
    "buffer_composition", "pooled_draw_counts", "current_stride5_p90",
    "update_checkpoint_step", "sync_inference_weights", "latest_replay_path",
    "model_version_summary", "terminal_reason_counts",
)

RUNNER_STATS_FIELDS = {
    "games_completed", "positions_generated", "x_wins", "o_wins", "draws",
    "model_version", "mcts_quiescence_fires", "mcts_mean_depth",
    "mcts_mean_root_concentration", "cluster_value_std_mean",
    "cluster_policy_disagreement_mean", "cluster_variance_sample_count",
    "solver_moves_eligible", "solver_win_proven", "solver_injected",
    "solver_injected_offwindow", "solver_budget_exhausted",
    "solver_moves_eligible_seeded", "solver_injected_seeded", "seeded_games_started",
    # WP12-R Phase T target-integrity counters (LAW-18; the byte-frozen oracle
    # bank fixes these names — see tests/selfplay/test_target_law18_counters.py).
    "export_offwindow_mass_moves", "gridls_zero_policy_rows",
    "target_integrity_defects",
    # R275(b): the SEAM conjunct of the same class the three above guard — a leaf
    # inference that FAILED, counted separately from the record-dispatch refusals so the
    # two conjuncts stay distinguishable. Pin: tests/selfplay/test_inference_seam_counter.py.
    "inference_failures_total",
    # Item 10(b) / R250: the DENSE record path's K distribution — the LAW-18 fire-rate
    # log for the K-cluster lever. A THIRD family again: not a Phase-T latch and not a
    # lifecycle counter but a per-encoding instrument, `None` where no producer exists
    # and dropped entirely from the event stream on a graph run.
    "k_cluster_histogram",
    # R256/ADJ-D37: the forced-win coverage-clip counter — the same per-encoding
    # instrument family as the histogram, gated the INVERSE way (present on graph,
    # dropped on dense; `None` where no producer exists). Pins:
    # tests/train/test_uncovered_forced_win.py.
    "uncovered_forced_win",
    # Worker threads that died by panic (item 3). A DIFFERENT family from the three
    # above despite sitting beside them: those are Phase-T target-integrity latches,
    # this is a lifecycle counter. Kept out of `_TARGET_INTEGRITY_COUNTERS` for that
    # reason — see the queue's ADJ-D9.
    "worker_panics",
    "runner_encoding",
}
INFERENCE_STATS_FIELDS = {"forward_count", "total_requests", "encoding_spec"}

# The import DAG selfplay is allowed. `mantis.eval` / `mantis.train` / `mantis.bots` are
# absent BY CONSTRUCTION: promotion is a callee surface and every outward collaborator is
# injected. `mantis.util` is the granted leaf edge (coordinate maths).
ALLOWED_MANTIS_ROOTS = {
    "mantis._engine", "mantis.encoding", "mantis.env", "mantis.model", "mantis.config",
    "mantis.monitor", "mantis.util", "mantis.selfplay",
}
FORBIDDEN_MANTIS_ROOTS = {"mantis.eval", "mantis.train", "mantis.bots"}


def _cfg(encoding: str, **over: Any) -> dict[str, Any]:
    # WPSC Phase 2 SC-A2 reshape: `selfplay`/`inference`/`train` are now nested schema-shaped
    # sections (SelfPlayHParams.from_config / InferenceHParams.from_config no longer read a
    # flat dict with top-level-namespace fallback). `over` still layers onto `selfplay` (its
    # historical target — no call site in this file uses it today).
    selfplay: dict[str, Any] = {
        "n_workers": 1, "leaf_batch_size": 8, "max_game_moves": 128,
        "c_visit": 50.0,
        "c_scale": 1.0, "gumbel_m": 16, "gumbel_explore_moves": 10,
        "results_queue_cap": 10_000, "random_opening_plies": 0,
        "log_investigation_metrics": True,
        "mcts": {"n_simulations": 50, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "quiescence_enabled": True, "quiescence_blend_2": 0.3,
                 "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25, "dirichlet_enabled": True},
        "playout_cap": {"fast_sims": 40, "fast_prob": 0.0, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
    }
    selfplay.update(over)
    inference = {
        "inference_batch_size": 4,
        "inference_max_wait_ms": 10,
        # F-816-10 (R276(f)): the graph arm resolves the fused-forward memory bound at
        # construction. NON-BINDING BY CONSTRUCTION here — this fixture is about wiring, and
        # a cap that bound would make it exercise a split with nothing asserting the M.
        "fused_graph_caps": {"max_fused_edges": 57149441, "max_fused_nodes": 1785921},
    }
    train = {"draw_reward": -0.5, "ply_cap_value": -0.5}
    return {"encoding": encoding, "search": {"kind": "puct"}, "selfplay": selfplay,
            "inference": inference, "train": train}


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def _graph_pool(device: torch.device, buffer: Any = None, **cfg_over: Any) -> WorkerPool:
    spec = lookup("gnn_axis_v1")
    arch = GnnArch(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim,
                   hidden=16, num_layers=1)
    raw = buffer if buffer is not None else HexgBuffer(capacity=32, visit_capacity=128,
                                                       encoding="gnn_axis_v1")
    return WorkerPool(build_net(arch), _cfg("gnn_axis_v1", **cfg_over), device, raw, arch=arch)


# ═══ H-01 — the frozen consumer surface ══════════════════════════════════════════
def test_pool_presents_every_frozen_member(device) -> None:
    """H-01 — PASS iff every member the committed trainer reads exists on the pool with
    the right kind: plain attributes as attributes, properties as properties on the CLASS
    (so they are computed, not snapshotted at construction), and methods as callables.

    FAIL = an `AttributeError` minutes into the first real run, or — worse for a property
    demoted to an attribute — a value frozen at construction that the trainer keeps
    reading as if it were live."""
    pool = _graph_pool(device)

    for name in FROZEN_ATTRS:
        assert hasattr(pool, name), f"missing attribute {name!r}"
        assert not isinstance(getattr(type(pool), name, None), property), (
            f"{name} must be a plain attribute"
        )
    for name in FROZEN_PROPERTIES:
        assert isinstance(getattr(type(pool), name, None), property), (
            f"{name} must be a property — a plain attribute would freeze at construction"
        )
        getattr(pool, name)  # must be readable on a fresh pool, not just declared
    for name in FROZEN_METHODS:
        assert callable(getattr(pool, name, None)), f"missing method {name!r}"


def test_pool_satisfies_both_runtime_protocols(device) -> None:
    """H-01 (Protocol arm) — PASS iff the pool satisfies the trainer's committed
    `WorkerPoolLike` AND this package's `ActorSyncTarget` (WP-UNFREEZE, R49).

    The second assertion is `ActorSyncTarget`'s LAW-08 live consumer: a Protocol nothing
    checks is a dead declaration. It is also the mechanical statement of the DAG rule —
    actor sync is something the train-side engine does TO the pool, so the pool must
    satisfy the shape without importing anything from the train or eval side."""
    pool = _graph_pool(device)
    assert isinstance(pool, WorkerPoolLike)
    assert isinstance(pool, ActorSyncTarget)


def test_snapshot_dataclass_field_sets_are_frozen(device) -> None:
    """H-01 (snapshot arm) — PASS iff the two snapshot dataclasses carry exactly their
    documented field sets. A dropped field bites here rather than in a monitor that
    silently stops reporting a counter."""
    assert set(RunnerStats.__dataclass_fields__) == RUNNER_STATS_FIELDS
    assert set(InferenceStats.__dataclass_fields__) == INFERENCE_STATS_FIELDS

    pool = _graph_pool(device)
    rstats = pool.runner_stats()
    assert isinstance(rstats, RunnerStats)
    # The trainer's regime-gated cluster block reads these four by name.
    for name in ("mcts_mean_depth", "mcts_mean_root_concentration"):
        assert isinstance(getattr(rstats, name), float)
    # ADJ-D32 / R249: the two CLUSTER means are `float | None`, and on a pool that has
    # played nothing they are None — no samples, so no measurement. This assertion used
    # to demand a `float`, which is how a fabricated 0.0 travelled the whole seam
    # unchallenged; the snapshot must carry the absence, not paper over it.
    assert rstats.cluster_variance_sample_count == 0
    for name in ("cluster_value_std_mean", "cluster_policy_disagreement_mean"):
        assert getattr(rstats, name) is None, (
            f"{name} must be None at zero cluster-variance samples (R249), got "
            f"{getattr(rstats, name)!r}"
        )
    assert isinstance(rstats.cluster_variance_sample_count, int)
    # Item 10(b), INVERTED by R346(f). K — how many cluster views a recorded position
    # expands into — was knowable only at the dense record path, and that path is gone, so
    # the engine no longer exposes the getter and the field reads as the no-producer `None`
    # on every pool. Pinned as ABSENT rather than deleted: a fabricated tuple here would be
    # a phantom instrument, which is the one thing the field must never become.
    assert rstats.k_cluster_histogram is None, (
        "K is a dense-record-path quantity and the dense record path is deleted; a value "
        "here means something is fabricating one"
    )

    istats = pool.inference_stats()
    assert isinstance(istats, InferenceStats)
    assert istats.encoding_spec is pool.encoding_spec


def test_winrates_are_computed_from_the_right_counters(device) -> None:
    """H-01 (winrate arm) — PASS iff `x_winrate == x_wins / games_completed` and
    `o_winrate == o_wins / games_completed`, with both 0.0 at zero games.

    The pair is emitted as `win_rate_p0` / `win_rate_p1`. A swap is silent, permanent, and
    invisible in aggregate — the two series simply trade places for the whole run — so the
    values are asserted against DIFFERENT scripted counters, not against each other."""
    pool = _graph_pool(device)
    assert pool.x_winrate == 0.0 and pool.o_winrate == 0.0, "zero games ⇒ 0.0, not NaN"

    pool.games_completed = 10
    pool.x_wins = 7
    pool.o_wins = 2
    assert pool.x_winrate == 0.7
    assert pool.o_winrate == 0.2


def test_no_op_recorder_default_reports_no_replay(device) -> None:
    """H-01 (default-seam arm) — PASS iff a pool built without a recorder answers
    `latest_replay_path() is None` and accepts `update_checkpoint_step` silently. The
    default is inert BY DESIGN, and being explicit about it stops a future reader from
    reading `None` as a bug."""
    pool = _graph_pool(device)
    assert pool.latest_replay_path() is None
    pool.update_checkpoint_step(17)  # must not raise on the no-op recorder


# ═══ E-07 — the pool pushes THROUGH the facade ═══════════════════════════════════
def test_pool_replay_buffer_is_the_facade(device) -> None:
    """E-07 — PASS iff `pool.replay_buffer` IS a `ReplayFacade` wrapping the exact raw
    buffer handed to the constructor, with the kind resolved from the pool's own spec.

    The drain reads `pool.replay_buffer`, so if the facade is used at all it must be that
    attribute. Without this pin the facade can be constructed and bypassed — or never
    constructed — with every other oracle still green, and the mislabel guard becomes dead
    code that proves nothing."""
    graph_raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    graph_pool = _graph_pool(device, buffer=graph_raw)
    assert isinstance(graph_pool.replay_buffer, ReplayFacade)
    assert graph_pool.replay_buffer.raw is graph_raw, (
        "the facade must wrap the ctor's buffer, not a copy")
    assert graph_pool.replay_buffer.kind is BufferKind.GRAPH


def test_pool_does_not_keep_a_second_handle_on_the_raw_buffer(device) -> None:
    """E-07 (bypass arm) — PASS iff no pool attribute other than the facade holds the raw
    buffer. A stashed second reference is the simplest way to satisfy every other oracle
    while pushing around the guard."""
    raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    pool = _graph_pool(device, buffer=raw)

    holders = [name for name, value in vars(pool).items() if value is raw]
    assert holders == [], (
        f"the raw buffer is reachable off the pool at {holders} — the push path could "
        "bypass the facade and the mislabel guard would be dead code"
    )


def test_graph_pool_buffer_composition_is_nan_and_that_is_parity(device) -> None:
    """E-05 (graph arm) — PASS iff a graph pool reports `draw_target_fraction` as NaN.

    The graph buffer genuinely does not expose the outcome-band getter on EITHER side, so
    NaN is the true old-side value and porting it is parity. Fabricating a number here
    would be an undeclared behaviour change — and since the production identity is a graph
    encoding, this is the value a monitor will actually see, which is exactly why it must
    not be quietly invented."""
    pool = _graph_pool(device)
    pool.config["train"] = {"draw_reward": -0.5, "ply_cap_value": -0.7}

    composition = pool.buffer_composition()
    assert math.isnan(composition["draw_target_fraction"])
    assert composition["buffer_size"] == 0
    assert composition["corpus_fraction"] == 1.0
    assert not hasattr(pool.replay_buffer.raw, "outcome_in_range_count"), (
        "the graph buffer must not gain the getter — that would CREATE a metric that "
        "does not exist old-side"
    )


# ═══ H-12 — the import-DAG proof, by AST ════════════════════════════════════════
def _top_level_imports(path: Path) -> set[str]:
    """Module names imported at MODULE level (deferred imports inside functions are the
    author's business; the DAG is about what importing the package pulls in)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_selfplay_imports_stay_inside_the_declared_dag() -> None:
    """H-12 — PASS iff no module under `src/mantis/selfplay` imports outside the declared
    allowlist at module level, and in particular imports NOTHING from `mantis.eval`,
    `mantis.train` or `mantis.bots`.

    This is the mechanical form of the one-way rule: self-play is a producer, and the
    evaluator and trainer are its consumers. Promotion looks like it wants an edge — the
    pool must be told about a new checkpoint — but it is a CALLEE surface, so the arrow
    still points the other way. FAIL = a cycle that will eventually be discovered as an
    import-time deadlock rather than as a design decision."""
    offenders: dict[str, set[str]] = {}
    checked = 0
    for path in sorted(SELFPLAY_SRC.glob("*.py")):
        checked += 1
        bad = set()
        for name in _top_level_imports(path):
            if not name.startswith("mantis"):
                continue
            root = ".".join(name.split(".")[:2])
            if root not in ALLOWED_MANTIS_ROOTS:
                bad.add(name)
        if bad:
            offenders[path.name] = bad

    assert checked >= 10, "the glob found suspiciously few modules to check"
    assert not offenders, f"imports outside the declared selfplay DAG: {offenders}"


def test_no_selfplay_to_eval_train_or_bots_edge_anywhere() -> None:
    """H-12 (explicit-forbid arm) — PASS iff the three forbidden roots appear in NO import
    statement, at module level or otherwise, anywhere under `src/mantis/selfplay`.

    Stated separately from the allowlist because an allowlist can be widened in a
    one-character diff; this row names the three packages that must never appear and would
    have to be deleted outright to pass."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(SELFPLAY_SRC.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        bad = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bad |= {r for r in FORBIDDEN_MANTIS_ROOTS
                            if alias.name == r or alias.name.startswith(r + ".")}
            elif isinstance(node, ast.ImportFrom) and node.module:
                bad |= {r for r in FORBIDDEN_MANTIS_ROOTS
                        if node.module == r or node.module.startswith(r + ".")}
        if bad:
            offenders[path.name] = bad
    assert not offenders, f"forbidden import edge from selfplay: {offenders}"


def test_the_dag_check_would_notice_a_forbidden_import(tmp_path) -> None:
    """H-12 (mutation self-test) — PASS iff the checker's own logic flags a module that
    DOES import from the eval side. Without this the two rows above could be passing
    because they parse nothing (a wrong glob, a swallowed error), and a checker that
    cannot fail is worse than no checker (LAW-07)."""
    doctored = tmp_path / "doctored.py"
    doctored.write_text("from mantis.eval.pipeline import Something\nimport mantis.train\n")
    names = _top_level_imports(doctored)
    flagged = {n for n in names
               if ".".join(n.split(".")[:2]) not in ALLOWED_MANTIS_ROOTS}
    assert flagged == {"mantis.eval.pipeline", "mantis.train"}
