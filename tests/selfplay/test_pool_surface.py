"""Suite H (surface) — the frozen consumer surface and the import-DAG proof — plus the
Suite-E rows that pin the pool's replay-buffer facade.

The trainer duck-types the pool as `Any` and reads ~20 members off it. Nothing type-checks that
seam, so a member renamed or dropped during the split is invisible until the first integrated run,
where it appears as an `AttributeError` minutes into training.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from _pool_harness import graph_pool
from mantis._engine import HexgBuffer
from mantis.selfplay.buffers import BufferKind, ReplayFacade
from mantis.selfplay.pool_hooks import ActorSyncTarget, InferenceStats, RunnerStats
from mantis.train.coordinator.config import WorkerPoolLike

SELFPLAY_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis" / "selfplay"

# Every member the committed trainer reads, with the kind it uses it as. A missing row here is a
# runtime break at first integration.
FROZEN_ATTRS = ("games_completed", "draws", "n_workers", "encoding_spec")
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
    "update_checkpoint_step", "sync_inference_weights",
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
    # The playout-cap draw's two arms and the Gumbel round-width terms (LAW-18).
    "pcr_full_moves", "pcr_quick_moves", "gumbel_round_leaves", "gumbel_rounds",
    # WP12-R Phase T target-integrity counters (LAW-18; the byte-frozen oracle
    # bank fixes these names — see tests/selfplay/test_target_law18_counters.py).
    "export_offwindow_mass_moves", "target_integrity_defects",
    # The SEAM conjunct of the same class the two above guard: a leaf inference that FAILED,
    # counted separately from the record-dispatch refusals so the two stay distinguishable.
    "inference_failures_total",
    # Graph rows the results-queue cap dropped before a drain read them.
    "positions_dropped",
    # Worker threads that died by panic — a lifecycle counter, a DIFFERENT family from the
    # target-integrity latches beside it, which is why it is kept out of that tuple.
    "worker_panics",
}
INFERENCE_STATS_FIELDS = {"forward_count", "total_requests", "encoding_spec"}

# The import DAG selfplay is allowed. `mantis.eval` / `mantis.train` / `mantis.bots` are absent BY
# CONSTRUCTION: promotion is a callee surface and every outward collaborator is injected.
ALLOWED_MANTIS_ROOTS = {
    "mantis._engine", "mantis.encoding", "mantis.model", "mantis.config",
    "mantis.monitor", "mantis.util", "mantis.selfplay",
}
FORBIDDEN_MANTIS_ROOTS = {"mantis.eval", "mantis.train", "mantis.bots"}


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def test_pool_presents_every_frozen_member(device) -> None:
    """H-01 — PASS iff every member the committed trainer reads exists on the pool with the right kind: plain attributes as attributes, properties as properties on the CLASS (so they are computed, not snapshotted at construction), and methods as callables."""
    pool = graph_pool(device=device, capacity=32, n_simulations=50, fast_sims=40)

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
    pool = graph_pool(device=device, capacity=32, n_simulations=50, fast_sims=40)
    assert isinstance(pool, WorkerPoolLike)
    assert isinstance(pool, ActorSyncTarget)


def test_snapshot_dataclass_field_sets_are_frozen(device) -> None:
    """H-01 (snapshot arm) — PASS iff the two snapshot dataclasses carry exactly their documented field sets."""
    assert set(RunnerStats.__dataclass_fields__) == RUNNER_STATS_FIELDS
    assert set(InferenceStats.__dataclass_fields__) == INFERENCE_STATS_FIELDS

    pool = graph_pool(device=device, capacity=32, n_simulations=50, fast_sims=40)
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
    pool = graph_pool(device=device, capacity=32, n_simulations=50, fast_sims=40)
    assert pool.x_winrate == 0.0 and pool.o_winrate == 0.0, "zero games ⇒ 0.0, not NaN"

    pool.games_completed = 10
    pool.x_wins = 7
    pool.o_wins = 2
    assert pool.x_winrate == 0.7
    assert pool.o_winrate == 0.2


def test_no_op_recorder_default_accepts_a_step(device) -> None:
    """H-01 (default-seam arm) — PASS iff a pool built without a recorder accepts `update_checkpoint_step` silently."""
    pool = graph_pool(device=device, capacity=32, n_simulations=50, fast_sims=40)
    pool.update_checkpoint_step(17)  # must not raise on the no-op recorder


def test_pool_replay_buffer_is_the_facade(device) -> None:
    """E-07 — PASS iff `pool.replay_buffer` IS a `ReplayFacade` wrapping the exact raw buffer handed to the constructor, with the kind resolved from the pool's own spec."""
    graph_raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    wrapped = graph_pool(device=device, buffer=graph_raw, capacity=32,
                          n_simulations=50, fast_sims=40)
    assert isinstance(wrapped.replay_buffer, ReplayFacade)
    assert wrapped.replay_buffer.raw is graph_raw, (
        "the facade must wrap the ctor's buffer, not a copy")
    assert wrapped.replay_buffer.kind is BufferKind.GRAPH


def test_pool_does_not_keep_a_second_handle_on_the_raw_buffer(device) -> None:
    """E-07 (bypass arm) — PASS iff no pool attribute other than the facade holds the raw buffer."""
    raw = HexgBuffer(capacity=32, encoding="gnn_axis_v1", visit_capacity=128)
    pool = graph_pool(device=device, buffer=raw, capacity=32, n_simulations=50, fast_sims=40)

    holders = [name for name, value in vars(pool).items() if value is raw]
    assert holders == [], (
        f"the raw buffer is reachable off the pool at {holders} — the push path could "
        "bypass the facade and the mislabel guard would be dead code"
    )


def _top_level_imports(path: Path) -> set[str]:
    """Module names imported at MODULE level (deferred imports inside functions are the author's business; the DAG is about what importing the package pulls in)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    doctored.write_text("from mantis.eval.pipeline import Something\nimport mantis.train\n", encoding="utf-8")
    names = _top_level_imports(doctored)
    flagged = {n for n in names
               if ".".join(n.split(".")[:2]) not in ALLOWED_MANTIS_ROOTS}
    assert flagged == {"mantis.eval.pipeline", "mantis.train"}
