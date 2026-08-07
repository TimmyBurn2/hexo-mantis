"""R256/ADJ-D37 + LAW-18 — the uncovered_forced_win counter reaches the ONE channel on
the arm whose mechanism produces it, and is ABSENT elsewhere.

THE MECHANISM. `apply_forced_win_one_hot_ls_counted` (crates/mantis-selfplay/src/records.rs;
both the O1 forced-win arm and the D-WS3 solver hook route through it) refuses a PROVEN
forced win when the K-cluster WINDOW criterion says the winning cell is uncovered — a pure
target loss the record never witnesses. The counter ticks exactly on that refusal while the
injecting lever is armed (`!covered && weight > 0`); its Rust producer + mutation pins live
in `records::ls_tests` (killers: deleted `fetch_add`, inverted coverage, dropped weight
conjunct, helper-vs-primitive drift).

THE MAPPING (R256, correcting R250's). The mechanism runs wherever LS targets are built —
measured TRUE on the graph arm (run5's own) and FALSE on the shipped dense grids
`v6`/`v6w25`. R250's principle stands, the mapping is re-derived from code: the instrument
attaches to the mechanism's measured live path, so the emitter publishes it on the GRAPH
arm and OMITS it on dense — the inverse gating of the K histogram one function over, keyed
on the SAME `is_graph_run` authority so the two subtractions cannot disagree about the arm.
(Disclosed, not hidden: `v6_live2_ls` is also LS — its Rust counter ticks but R256 lands
EMISSION on the graph path; the dense-LS stream gap is a recorded queue disclosure, not an
oversight.)

SHAPE. Cumulative `{"total": int, "per_position": float | None}` — the 10(b) cumulative
precedent, with the rate over the snapshot's own cumulative `positions_generated`
denominator; `per_position` is `None` when no position has been recorded (a rate over zero
positions is not a measurement, R249). No producer (an engine build predating the getter)
→ keyed `None`, never a fabricated zero block.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mantis import _engine
from mantis.train.events import (
    UNCOVERED_FORCED_WIN_KEY,
    emit_iteration_complete_event,
    uncovered_forced_win_block,
)

GRID_CONFIG: dict[str, Any] = {"identity": {"encoding": "v6_live2_ls",
                                            "representation": "grid"}}
GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _Pool:
    gumbel_mcts = False
    avg_game_length = 12.0
    x_winrate = 0.5
    o_winrate = 0.4
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 50.0
    inference_batch_timing = None
    recent_move_histories: list[list[tuple[int, int]]] = []


class _Buffer:
    size = 7
    capacity = 64


def _rstats(total: Any, positions: Any = 100, *, omit_getter: bool = False) -> Any:
    ns = SimpleNamespace(
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
        cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None,
        cluster_variance_sample_count=0,
        k_cluster_histogram=(0,) * 9,
        positions_generated=positions,
    )
    if not omit_getter:
        ns.uncovered_forced_win = total
    return ns


def _emit(config: dict[str, Any], rstats: Any) -> dict[str, Any]:
    sink = _Sink()
    emit_iteration_complete_event(
        11, 0.0, 10, 4, _Pool(), _Buffer(), config, {}, 64,
        lambda: 0.0, None, {}, rstats, sink,
    )
    assert len(sink.events) == 1
    return sink.events[0]


# ═══ the absence rule — R250/R256 on the DENSE arm (the 10(b) gate, inverted) ═══════
def test_a_dense_run_carries_no_uncovered_forced_win_key_at_all() -> None:
    """FALSIFYING MUTATION: delete the `if not graph_run: return {}` arm of
    `uncovered_forced_win_block` — the key then ships on the shipped dense grids, reading
    a truthless 0 exactly where R250 forbids one (the D37 arm-(i) trap, resurrected)."""
    payload = _emit(GRID_CONFIG, _rstats(0))
    assert UNCOVERED_FORCED_WIN_KEY not in payload, (
        f"R256: {UNCOVERED_FORCED_WIN_KEY} must be ABSENT on a dense run; got "
        f"{payload.get(UNCOVERED_FORCED_WIN_KEY)!r}"
    )


def test_dense_absence_holds_even_when_the_snapshot_reads_nonzero() -> None:
    """The gate is about the ARM (the run's validated identity), never the reading —
    `v6_live2_ls` is itself LS, so its Rust counter CAN tick; emission stays graph-scoped
    per R256's explicit landing, and the dense-LS gap is a queue disclosure."""
    payload = _emit(GRID_CONFIG, _rstats(37))
    assert UNCOVERED_FORCED_WIN_KEY not in payload


# ═══ the presence rule — the graph arm publishes the LAW-18 fire rate ═══════════════
def test_a_graph_run_publishes_total_and_per_position() -> None:
    """FALSIFYING MUTATION: gate the block on `graph_run` the same way 10(b) is gated
    (copy-paste symmetry) — the instrument then vanishes from the ONLY arm whose stream
    R256 lands it on, reading zero exactly where the drops happen (the F-27 canary)."""
    payload = _emit(GRAPH_CONFIG, _rstats(37, positions=100))
    block = payload[UNCOVERED_FORCED_WIN_KEY]
    assert block["total"] == 37
    assert abs(block["per_position"] - 0.37) < 1e-12


def test_a_truthful_zero_total_is_published_not_dropped() -> None:
    """A raw counter is truthful at 0 (the R249 distinction: the COUNT is evidence, only a
    derived MEAN over zero samples is fabrication) — a healthy graph run publishes 0."""
    payload = _emit(GRAPH_CONFIG, _rstats(0, positions=50))
    block = payload[UNCOVERED_FORCED_WIN_KEY]
    assert block["total"] == 0
    assert block["per_position"] == 0.0


def test_zero_positions_yields_a_none_rate_never_a_fabricated_one() -> None:
    payload = _emit(GRAPH_CONFIG, _rstats(0, positions=0))
    block = payload[UNCOVERED_FORCED_WIN_KEY]
    assert block["total"] == 0
    assert block["per_position"] is None, "a rate over zero recorded positions is not a measurement"


def test_an_engine_build_without_the_getter_publishes_none_not_zeros() -> None:
    """The event_manifest unproduced-field convention: keyed, `None`, never `{total: 0}`."""
    payload = _emit(GRAPH_CONFIG, _rstats(0, omit_getter=True))
    assert payload[UNCOVERED_FORCED_WIN_KEY] is None


# ═══ block-level unit (the same fn the emitter calls) ═══════════════════════════════
def test_the_block_fn_is_the_single_spelling_authority() -> None:
    assert uncovered_forced_win_block(_rstats(5, positions=10), graph_run=False) == {}
    graph = uncovered_forced_win_block(_rstats(5, positions=10), graph_run=True)
    assert graph == {UNCOVERED_FORCED_WIN_KEY: {"total": 5, "per_position": 0.5}}


# ═══ the real FFI producer exists (wheel-compat caveat as for the Phase-T family) ═══
def test_the_real_engine_getter_exists_and_reads_zero_on_a_fresh_runner() -> None:
    """LAW-07 producer leg at the seam: the shipped engine exposes the getter and a fresh
    runner reads a truthful 0. (That the counter FIRES is pinned Rust-side in
    `records::ls_tests`; `runner_stats` threads it with a wheel-compat `None` default —
    the k_cluster_histogram posture, an old wheel has measured nothing — so Python-side
    visibility alone is not a producer proof.)"""
    runner = _engine.SelfPlayRunner(_engine.SelfPlayRunnerConfig(encoding_name="gnn_axis_v1"))
    assert runner.uncovered_forced_win == 0


# ═══ the production stats surface carries the hop (the review's H1 catch) ═══════════
def test_the_production_runner_stats_dataclass_carries_the_counter_to_the_emitter() -> None:
    """THE wiring pin this instrument was missing at review: every double above supplies
    the attribute by construction, so only an rstats that IS the production
    `pool_hooks.RunnerStats` can red the hop `pool_hooks` forgot. FALSIFYING MUTATION:
    delete the `uncovered_forced_win` field — the ctor kwarg below is then a TypeError.
    (The `runner_stats()` threading line's own killer is the NEXT test; the pair jointly
    covers both halves of the hop R256 ruled the instrument onto.)"""
    from mantis.selfplay.pool_hooks import RunnerStats

    rstats = RunnerStats(
        games_completed=1, positions_generated=100, x_wins=1, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25, cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None, cluster_variance_sample_count=0,
        uncovered_forced_win=37,
    )
    payload = _emit(GRAPH_CONFIG, rstats)
    assert payload[UNCOVERED_FORCED_WIN_KEY] == {"total": 37, "per_position": 0.37}


def test_runner_stats_threads_the_real_getter_and_none_when_absent() -> None:
    """The `test_target_law18_counters` threading pattern, for this counter: the wrapper
    reads the bridge getter when present and carries the wheel-compat `None` (never a
    fabricated produced-0) when the engine build predates it."""
    from mantis.selfplay.pool_hooks import runner_stats

    runner = _engine.SelfPlayRunner(_engine.SelfPlayRunnerConfig(encoding_name="gnn_axis_v1"))
    stats = runner_stats(SimpleNamespace(_runner=runner))
    assert stats.uncovered_forced_win == 0

    class _OldWheel:
        """Every attribute `runner_stats` reads EXCEPT the None-defaulted getters
        (an engine build predating both this counter and the K histogram)."""

        def __getattr__(self, name: str):
            if name in ("uncovered_forced_win", "k_cluster_histogram"):
                raise AttributeError(name)
            return 0

    old = runner_stats(SimpleNamespace(_runner=_OldWheel()))
    assert old.uncovered_forced_win is None, "an old wheel measured nothing — None, not 0"


# ═══ routing census — the raw primitive has no caller outside its home module ═══════
def test_no_call_site_bypasses_the_counted_helper() -> None:
    """The helper's contract ("both mechanism sites route through it, so mechanism and
    instrument cannot drift apart") is enforceable only if the raw primitive
    `apply_forced_win_one_hot_ls(` keeps ZERO callers outside records.rs — it is still
    `pub(crate)`, so a future edit could re-inline one arm and silently unhook the
    instrument from that site with every other pin green. FALSIFYING MUTATION: revert
    either search_drive.rs site to the raw primitive."""
    from pathlib import Path

    crate_src = Path(__file__).resolve().parents[2] / "crates" / "mantis-selfplay" / "src"
    offenders: list[str] = []
    for path in crate_src.rglob("*.rs"):
        if path.name == "records.rs":
            continue  # the primitive's home: definition, counted wrapper, in-src oracles
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "apply_forced_win_one_hot_ls(" in line:
                offenders.append(f"{path.relative_to(crate_src)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "the raw coverage-gated primitive must have no caller outside records.rs — "
        "route through apply_forced_win_one_hot_ls_counted so the R256 instrument "
        "rides every mechanism site; found:\n" + "\n".join(offenders)
    )
