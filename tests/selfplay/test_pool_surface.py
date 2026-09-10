"""Suite H (surface) — the frozen consumer surface and the import-DAG proof — plus the two
Suite-E rows that assert on the POOL.

>300 justify: one object, one contract. The consumer surface, the import-DAG proof, the
facade-wrapping pin and the per-buffer-kind composition rule all bind `WorkerPool` construction
and share the same real-pool factory plus the captured buffer fill; splitting them would duplicate
both and let the copies drift.

The trainer duck-types the pool as `Any` and reads ~20 members off it. Nothing type-checks that
seam, so a member renamed or dropped during the split is invisible until the first integrated run,
where it appears as an `AttributeError` minutes into training.
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

# The captured deterministic fill: capacity 64, encoding v6, 40 rows — 10 organic draws at −0.5,
# 8 ply-cap truncations at −0.7, 10 wins at +1.0, 12 losses at −1.0 — 30 of them self-play.
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

# Every member the committed trainer reads, with the kind it uses it as. A missing row here is a
# runtime break at first integration.
FROZEN_ATTRS = ("games_completed", "draws", "n_workers", "recent_buffer", "encoding_spec")
FROZEN_PROPERTIES = (
    "recent_move_histories", "avg_game_length", "search_kind", "x_winrate", "o_winrate",
    # The THIRD outcome share, a property because the denominator belongs to the pool:
    # `iteration_complete` used to divide the raw `draws` attribute by a stale game count.
    "draw_rate",
    "sims_per_sec", "batch_fill_pct",
)
FROZEN_METHODS = (
    "runner_stats", "inference_stats", "check_producer_health", "stop", "start",
    "buffer_composition", "pooled_draw_counts", "current_stride5_p90",
    "update_checkpoint_step", "sync_inference_weights", "latest_replay_path",
    "model_version_summary", "terminal_reason_counts",
)

#: R346(f) took FIFTEEN of these. The engine exposes a getter for none of them, so each would have
#: snapshotted its wheel-compat default forever — a fabricated reading with no producer. The set
#: below is what a live engine still answers, asserted EXACTLY, so a field that comes back without
#: a getter reds here.
RUNNER_STATS_FIELDS = {
    "games_completed", "positions_generated", "x_wins", "o_wins", "draws",
    "model_version", "mcts_quiescence_fires", "mcts_mean_depth",
    "mcts_mean_root_concentration",
    # WP12-R Phase T target-integrity counters (LAW-18; the byte-frozen oracle
    # bank fixes these names — see tests/selfplay/test_target_law18_counters.py).
    "export_offwindow_mass_moves", "target_integrity_defects",
    # The SEAM conjunct of the same class the two above guard: a leaf inference that FAILED,
    # counted separately from the record-dispatch refusals so the two stay distinguishable.
    "inference_failures_total",
    # Worker threads that died by panic — a lifecycle counter, a DIFFERENT family from the
    # target-integrity latches beside it, which is why it is kept out of that tuple.
    "worker_panics",
    "runner_encoding",
}
INFERENCE_STATS_FIELDS = {"forward_count", "total_requests", "encoding_spec"}

# The import DAG selfplay is allowed. `mantis.eval` / `mantis.train` / `mantis.bots` are absent BY
# CONSTRUCTION: promotion is a callee surface and every outward collaborator is injected.
ALLOWED_MANTIS_ROOTS = {
    "mantis._engine", "mantis.encoding", "mantis.env", "mantis.model", "mantis.config",
    "mantis.monitor", "mantis.util", "mantis.selfplay",
}
FORBIDDEN_MANTIS_ROOTS = {"mantis.eval", "mantis.train", "mantis.bots"}


def _cfg(encoding: str, **over: Any) -> dict[str, Any]:
    # `selfplay`/`inference`/`train` are nested schema-shaped sections, so the `from_config`
    # readers no longer take a flat dict with a top-level-namespace fallback. `over` still layers
    # onto `selfplay`.
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
        # The graph arm resolves the fused-forward memory bound at construction. NON-BINDING BY
        # CONSTRUCTION here: this fixture is about wiring, and a cap that bound would exercise a
        # split with nothing asserting the M.
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


def test_pool_presents_every_frozen_member(device) -> None:
    """H-01 — PASS iff every member the committed trainer reads exists on the pool with the right kind: plain attributes as attributes, properties as properties on the CLASS (so they are computed, not snapshotted at construction), and methods as callables."""
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
    """H-01 (Protocol arm) — PASS iff the pool satisfies the trainer's committed `WorkerPoolLike` AND this package's `ActorSyncTarget` (WP-UNFREEZE, R49)."""
    pool = _graph_pool(device)
    assert isinstance(pool, WorkerPoolLike)
    assert isinstance(pool, ActorSyncTarget)


def test_snapshot_dataclass_field_sets_are_frozen(device) -> None:
    """H-01 (snapshot arm) — PASS iff the two snapshot dataclasses carry exactly their documented field sets."""
    assert set(RunnerStats.__dataclass_fields__) == RUNNER_STATS_FIELDS
    assert set(InferenceStats.__dataclass_fields__) == INFERENCE_STATS_FIELDS

    pool = _graph_pool(device)
    rstats = pool.runner_stats()
    assert isinstance(rstats, RunnerStats)
    # The trainer's regime-gated block reads these two by name.
    for name in ("mcts_mean_depth", "mcts_mean_root_concentration"):
        assert isinstance(getattr(rstats, name), float)
    # The other side of the deletion above, driven rather than declared: the engine answers none
    # of the fifteen, so a snapshot carrying one would be carrying a fabrication.
    for gone in ("cluster_value_std_mean", "cluster_variance_sample_count",
                 "k_cluster_histogram", "uncovered_forced_win", "gridls_zero_policy_rows",
                 "solver_injected", "seeded_games_started"):
        assert not hasattr(rstats, gone), (
            f"{gone} is back on the snapshot; the engine exposes no getter for it, so its "
            "value can only be a wheel-compat default reading as a measurement"
        )

    istats = pool.inference_stats()
    assert isinstance(istats, InferenceStats)
    assert istats.encoding_spec is pool.encoding_spec


def test_winrates_are_computed_from_the_right_counters(device) -> None:
    """H-01 (winrate arm) — PASS iff `x_winrate == x_wins / games_completed` and `o_winrate == o_wins / games_completed`, with both 0.0 at zero games."""
    pool = _graph_pool(device)
    assert pool.x_winrate == 0.0 and pool.o_winrate == 0.0, "zero games ⇒ 0.0, not NaN"

    pool.games_completed = 10
    pool.x_wins = 7
    pool.o_wins = 2
    assert pool.x_winrate == 0.7
    assert pool.o_winrate == 0.2


def test_no_op_recorder_default_reports_no_replay(device) -> None:
    """H-01 (default-seam arm) — PASS iff a pool built without a recorder answers `latest_replay_path() is None` and accepts `update_checkpoint_step` silently."""
    pool = _graph_pool(device)
    assert pool.latest_replay_path() is None
    pool.update_checkpoint_step(17)  # must not raise on the no-op recorder


def test_pool_replay_buffer_is_the_facade(device) -> None:
    """E-07 — PASS iff `pool.replay_buffer` IS a `ReplayFacade` wrapping the exact raw buffer handed to the constructor, with the kind resolved from the pool's own spec."""
    graph_raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    graph_pool = _graph_pool(device, buffer=graph_raw)
    assert isinstance(graph_pool.replay_buffer, ReplayFacade)
    assert graph_pool.replay_buffer.raw is graph_raw, (
        "the facade must wrap the ctor's buffer, not a copy")
    assert graph_pool.replay_buffer.kind is BufferKind.GRAPH


def test_pool_does_not_keep_a_second_handle_on_the_raw_buffer(device) -> None:
    """E-07 (bypass arm) — PASS iff no pool attribute other than the facade holds the raw buffer."""
    raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    pool = _graph_pool(device, buffer=raw)

    holders = [name for name, value in vars(pool).items() if value is raw]
    assert holders == [], (
        f"the raw buffer is reachable off the pool at {holders} — the push path could "
        "bypass the facade and the mislabel guard would be dead code"
    )


def test_graph_pool_buffer_composition_is_nan_and_that_is_parity(device) -> None:
    """E-05 (graph arm) — PASS iff a graph pool reports `draw_target_fraction` as NaN."""
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


def _top_level_imports(path: Path) -> set[str]:
    """Module names imported at MODULE level (deferred imports inside functions are the author's business; the DAG is about what importing the package pulls in)."""
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
    """H-12 — PASS iff no module under `src/mantis/selfplay` imports outside the declared allowlist at module level, and in particular imports NOTHING from `mantis.eval`, `mantis.train` or `mantis.bots`."""
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
    """H-12 (explicit-forbid arm) — PASS iff the three forbidden roots appear in NO import statement, at module level or otherwise, anywhere under `src/mantis/selfplay`."""
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
    """H-12 (mutation self-test) — PASS iff the checker's own logic flags a module that DOES import from the eval side."""
    doctored = tmp_path / "doctored.py"
    doctored.write_text("from mantis.eval.pipeline import Something\nimport mantis.train\n")
    names = _top_level_imports(doctored)
    flagged = {n for n in names
               if ".".join(n.split(".")[:2]) not in ALLOWED_MANTIS_ROOTS}
    assert flagged == {"mantis.eval.pipeline", "mantis.train"}
