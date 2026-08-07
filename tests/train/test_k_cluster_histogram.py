# >300 justify (R8): one instrument, pinned along its whole length, and a production
# `StepCoordinator` that costs most of the file. The chain this card lands runs Rust atomic ->
# snapshot -> bridge getter -> `pool_hooks.runner_stats` -> `events.k_cluster_histogram_block`
# -> sink, and the ONE claim being made is that a K measured at the dense record path arrives
# in the run's own stream labelled correctly and is ABSENT on the arm that cannot measure it.
# Every row below is a different segment of that single chain and they share the coordinator
# rig, the config constants and the label vocabulary; splitting them would fork the rig into
# copies that drift while both stay green, and would separate the absence pin from the
# presence pin that stops "absent" being satisfied by an emitter that publishes nothing.
"""Item 10(b) / R250 + LAW-18 — the in-run K histogram reaches the ONE channel, and is
ABSENT on the arm with no producer.

WHAT WAS MISSING. K — the number of cluster views a recorded position expands into — is
computed inside `crates/mantis-selfplay/src/runner/record.rs::record_position` and was
visible nowhere else. A run could be told K_avg after the fact and still not distinguish
"K is 1 on every position, the multi-window lever is dead" from "K is spread and the lever
is doing work". That is verbatim the starved-vs-ineffective distinction LAW-18 says a
post-hoc offline probe cannot make, for the lever the whole K-cluster path exists to drive.
A mean cannot separate those two; a distribution can, which is why this is a histogram and
not another `{total, delta, per_position}` row.

WHAT R250 REQUIRES OF IT. "An instrument for a mechanism an encoding does not have is absent
from that encoding's event stream — never zero, never null-as-value." The graph arm has no
such mechanism: `record_position_graph_dispatch` does not take the histogram as a parameter
at all, so its buckets are zero for want of a producer. Publishing them would be a worse
fabrication than R249's scalar `0.0`, because a histogram has SHAPE — nine zeros read as a
measured distribution over K, not as a missing instrument. The absence is decided by the
same `is_graph_run` authority the cluster block uses (commit b349ec4), deliberately: two
subtractions on the same grounds must not be able to disagree about which arm a run is on.

WHY THE REAL-FFI ROW LIVES HERE and not under `tests/bridge/`: there is no existing bridge
file for this getter family, and the getter's only interesting property is that its length
is what the emitter derives its labels from — a fact about the seam's two ends together. It
is pinned beside the emitter that consumes it rather than in a file of its own.

NOT PINNED HERE, and deliberately: that the Rust counter FIRES. That burden is Rust-side
(`mantis_selfplay::runner::record::k_histogram_tests::record_position_counts_each_position_in_its_own_k_bucket`,
whose falsifying mutation is deleting the `fetch_add`, and
`::the_graph_record_path_never_touches_the_k_histogram` for the absence half). The
`getattr(r, "k_cluster_histogram", None)` wheel-compat default in `runner_stats` means
Python-side visibility is NOT a producer proof — the same recorded caveat
`tests/selfplay/test_target_law18_counters.py` carries for the Phase-T counters.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mantis import _engine
from mantis.config.loader import load_config
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.selfplay.pool_hooks import runner_stats
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.events import (
    K_CLUSTER_HISTOGRAM_KEY,
    emit_iteration_complete_event,
    k_cluster_histogram_block,
)
from mantis.train.lifecycle.signals import ShutdownState

REPO_ROOT = Path(__file__).resolve().parents[2]

# Identity blocks in the shape `RunConfig` dumps. `identity.representation` is required,
# defaultless and registry-cross-checked (`config/schema/core.py`), so reading it is reading
# the operator's validated declaration rather than a second authority.
GRID_CONFIG: dict[str, Any] = {"identity": {"encoding": "v6_live2_ls",
                                            "representation": "grid"}}
GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}

#: A live-looking reading: distinct per bucket so a transposed label/count zip is visible.
#: Nine values because that is what the shipped engine reports — asserted against the real
#: getter by `test_the_real_getter_and_the_label_derivation_agree_on_the_bucket_count`
#: rather than restated as a constant anywhere.
SEEDED_BUCKETS = (11, 22, 33, 44, 55, 66, 77, 88, 99)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


class _Pool:
    """The narrow `PoolTelemetryLike` surface, in run5's PUCT regime."""

    gumbel_mcts = False
    avg_game_length = 12.0
    x_winrate = 0.5
    o_winrate = 0.4
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 50.0
    inference_batch_timing = None
    recent_move_histories: list[list[tuple[int, int]]] = []


class _GumbelPool(_Pool):
    gumbel_mcts = True


class _Buffer:
    size = 7
    capacity = 64


def _rstats(buckets: Any) -> Any:
    """A `RunnerStats`-shaped double carrying only what this instrument reads, plus the
    cluster fields the shared builder touches on the way past."""
    return SimpleNamespace(
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
        cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None,
        cluster_variance_sample_count=0,
        k_cluster_histogram=buckets,
    )


def _emit(config: dict[str, Any], rstats: Any, pool: Any | None = None) -> dict[str, Any]:
    sink = _Sink()
    emit_iteration_complete_event(
        11, 0.0, 10, 4, pool or _Pool(), _Buffer(), config, {}, 64,
        lambda: 0.0, None, {}, rstats, sink,
    )
    assert len(sink.events) == 1
    return sink.events[0]


# ═══ the absence rule — R250 on the graph arm ════════════════════════════════════════
def test_a_graph_run_carries_no_k_histogram_key_at_all() -> None:
    """R250: OMITTED, not `None` and not nine zeros.

    FALSIFYING MUTATION: delete the `if graph_run: return {}` arm of
    `k_cluster_histogram_block`. The key then ships on every `iteration_complete` of a graph
    run, describing a K distribution for a path that never computes a K.
    """
    payload = _emit(GRAPH_CONFIG, _rstats((0,) * 9))

    assert K_CLUSTER_HISTOGRAM_KEY not in payload, (
        f"R250: {K_CLUSTER_HISTOGRAM_KEY} must be ABSENT on a graph representation — "
        f"`record_position_graph_dispatch` does not take the histogram as a parameter, so "
        f"there is no producer to report for. Got {payload.get(K_CLUSTER_HISTOGRAM_KEY)!r}."
    )


def test_graph_absence_holds_even_when_the_snapshot_reads_nonzero() -> None:
    """R250 is an ABSENCE rule about the ARM, not a zero-check about the reading. A graph run
    whose snapshot somehow carried K counts is reporting something that arm cannot produce —
    the anomaly is fixed at the source, never laundered into the event stream as a plausible
    distribution."""
    payload = _emit(GRAPH_CONFIG, _rstats(SEEDED_BUCKETS))

    assert K_CLUSTER_HISTOGRAM_KEY not in payload, (
        f"R250: absent on graph regardless of the reading; got "
        f"{payload.get(K_CLUSTER_HISTOGRAM_KEY)!r}"
    )


def test_the_histogram_is_not_regime_gated_the_way_the_cluster_block_is() -> None:
    """A deliberate difference from its neighbour, pinned so it is not "tidied" into
    symmetry. The cluster stats are PUCT-descent-specific and carry `None` under Gumbel
    (CONFRES S2); K is a property of the RECORD path and is measured identically under either
    descent, so a Gumbel grid run publishes the real distribution."""
    payload = _emit(GRID_CONFIG, _rstats(SEEDED_BUCKETS), pool=_GumbelPool())

    assert payload[K_CLUSTER_HISTOGRAM_KEY]["1"] == SEEDED_BUCKETS[0], (
        "the K histogram must survive the Gumbel regime gate — it is not a descent statistic"
    )


# ═══ the presence rule — a grid run publishes the real, correctly labelled distribution ══
def test_the_producer_chain_carries_each_bucket_to_its_own_label() -> None:
    """The full PYTHON half of the chain in one drive: a bridge getter's list travels through
    the REAL `pool_hooks.runner_stats` into the REAL builder and arrives labelled by K.

    DISTINCT per bucket because that is the only shape of test a transposed or reversed zip
    cannot survive — a run of equal counts passes any labelling. This is the same argument
    `test_runner_stats_reads_each_cluster_mean_from_its_own_getter` makes for the cluster
    pair, applied to nine values instead of two.

    FALSIFYING MUTATIONS: reverse `labels` or `counts` in `k_cluster_histogram_block`; or
    coerce the `None` default in `pool_hooks._k_histogram` to zeros.
    """
    pool = SimpleNamespace(_runner=SimpleNamespace(
        k_cluster_histogram=list(SEEDED_BUCKETS),
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
        cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None,
        cluster_variance_sample_count=0,
    ))

    rstats = runner_stats(pool)
    assert rstats.k_cluster_histogram == SEEDED_BUCKETS, (
        "pool_hooks.runner_stats must carry the getter's buckets through in ORDER and frozen"
    )

    published = _emit(GRID_CONFIG, rstats)[K_CLUSTER_HISTOGRAM_KEY]
    for i, count in enumerate(SEEDED_BUCKETS[:-1]):
        assert published[str(i + 1)] == count, (
            f"bucket {i} counts positions recorded at K={i + 1} and must be published under "
            f'that label; got {published[str(i + 1)]} under "{i + 1}", expected {count}'
        )
    assert published[f">{len(SEEDED_BUCKETS) - 1}"] == SEEDED_BUCKETS[-1], (
        "the LAST bucket is the guard for every K outside the real range and must be "
        "labelled as an inequality, never as a K"
    )


def test_the_labels_are_derived_from_the_vector_length_not_transcribed() -> None:
    """R192(e) / derive-or-delete: widening the Rust bucket array must relabel this payload
    with NO Python edit. A transcribed `">8"` would survive a widening and then mis-name the
    guard bucket forever, which is the stale-count class R8's second half exists to stop.

    FALSIFYING MUTATION: hard-code the labels (`["1", ..., "8", ">8"]`) in
    `k_cluster_histogram_block`. This row reds on the four-bucket case below.
    """
    four = k_cluster_histogram_block(_rstats((1, 2, 3, 4)), graph_run=False)
    assert four[K_CLUSTER_HISTOGRAM_KEY] == {"1": 1, "2": 2, "3": 3, ">3": 4}

    twelve = k_cluster_histogram_block(_rstats(tuple(range(12))), graph_run=False)
    assert list(twelve[K_CLUSTER_HISTOGRAM_KEY]) == [str(i) for i in range(1, 12)] + [">11"]


def test_an_engine_build_without_the_getter_publishes_none_not_zeros() -> None:
    """The wheel-compat arm, and the R249 lesson applied one instrument over: an engine that
    cannot report has measured NOTHING, and a fabricated all-zero distribution would be
    indistinguishable from a run where K was genuinely never anything.

    `None` here is the event_manifest unproduced-field convention — KEYED, so a consumer can
    see the field exists and has no producer. That is a different statement from R250's
    absence, which says the field has no MEANING on this arm.

    FALSIFYING MUTATION: change `pool_hooks.runner_stats`'s `getattr(r, "k_cluster_histogram",
    None)` default to `()` or to a tuple of zeros.
    """
    old_wheel = SimpleNamespace(
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
        cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None,
        cluster_variance_sample_count=0,
    )  # NO k_cluster_histogram getter at all

    rstats = runner_stats(SimpleNamespace(_runner=old_wheel))
    assert rstats.k_cluster_histogram is None, (
        "an engine build without the getter must report absence, not a fabricated zero "
        "distribution"
    )

    payload = _emit(GRID_CONFIG, rstats)
    assert K_CLUSTER_HISTOGRAM_KEY in payload
    assert payload[K_CLUSTER_HISTOGRAM_KEY] is None, (
        "no producer is a keyed None (event_manifest convention), never zeros and never — on "
        "a GRID run — the R250 absence, which would confuse 'unwired' with 'inapplicable'"
    )


# ═══ the real seam — a production coordinator, and the real FFI ══════════════════════
class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = None
    cluster_policy_disagreement_mean = None
    cluster_variance_sample_count = 0
    k_cluster_histogram = SEEDED_BUCKETS


class _CoordPool:
    def __init__(self) -> None:
        self.games_completed = 5
        self.gumbel_mcts = False
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def _loss(self) -> dict[str, float]:
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]:
        self.step += 1
        return self._loss()

    def save_checkpoint(self, loss_info) -> None:
        return None


class _CoordBuffer:
    def __init__(self) -> None:
        self.size = 1000
        self.capacity = 100_000

    def resize(self, n: int) -> None:
        self.capacity = n

    def save_to_path(self, p) -> None:
        return None

    def sample_batch_with_pos(self, n: int, augment: bool):
        return (None,) * 9


class _EvalPipeline:
    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        return {"status": "skipped"}

    def drain_pending(self):
        return None

    def poll_completed(self):
        return None


def _coordinator(full_config: dict[str, Any]):
    # Minted-config-derived knobs (the WPMINT Phase K-A/K-B precedent every coordinator test
    # in this directory follows — no hand-restated numbers).
    config = load_config(REPO_ROOT / "configs" / "dev_example.yaml")
    cfg = dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=resolve_drain_caps(config.monitor),
                                 gate_interval=config.monitor.gate_interval,
                                 knobs=resolve_coordinator_knobs(config.train)),
        eval_interval=1, log_interval=1000, gate_interval=1000, min_buf_size=10,
    )
    sink = _Sink()
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_CoordBuffer(), pretrained_buffer=None, recent_buffer=None,
        pool=_CoordPool(), eval_pipeline=_EvalPipeline(),
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None, config=cfg,
        full_config=full_config, train_cfg={}, mixing_cfg={}, sink=sink,
        monitor_cfg=MonitorConfig(),
    )
    return coord, cfg, sink


def test_a_real_coordinator_publishes_the_k_histogram_on_a_grid_run() -> None:
    """The producer test the `k_cluster_histogram` manifest row cites (R4 / LAW-07).

    Driven through a full `StepCoordinator.step()` so the production loop's own route to
    `_emit_iteration_complete` is exercised — a graph-only absence pin can be satisfied by an
    emitter that has simply stopped publishing the block, and this is what forbids that.

    FALSIFYING MUTATION (M-1): drop the `k_cluster_histogram_block` update from
    `emit_iteration_complete_event`. The instrument then exists everywhere except the run's
    own event stream, which is the half-wired state LAW-07 exists to prevent.
    """
    coord, _cfg, sink = _coordinator(GRID_CONFIG)
    coord.step()

    events = sink.named("iteration_complete")
    assert len(events) == 1, f"expected exactly one iteration_complete, got {len(events)}"
    published = events[0][K_CLUSTER_HISTOGRAM_KEY]
    assert published["1"] == SEEDED_BUCKETS[0], (
        "a production coordinator on a grid run must publish the live K distribution"
    )


def test_a_real_coordinator_omits_the_k_histogram_on_a_graph_run() -> None:
    """R250 asserted on the stream a PRODUCTION `StepCoordinator` produces.

    Driven through `_emit_iteration_complete` — the one site that hands the builder its
    `config` argument — rather than a full `step()`, because a graph declaration additionally
    routes the training step through `dispatch` and demands a graph-capable buffer
    (`RepresentationRouteError`). That is a different seam; the grid row above carries the
    full-`step()` evidence that the production loop really reaches this method.

    FALSIFYING MUTATION (M-2): pass `{}` as the builder's `config` in
    `coordinator/step.py::_emit_iteration_complete`. `is_graph_run` then reads non-graph and a
    nine-zero distribution ships on every iteration of a run that computes no K.
    """
    coord, cfg, sink = _coordinator(GRAPH_CONFIG)
    coord._emit_iteration_complete(cfg)

    events = sink.named("iteration_complete")
    assert len(events) == 1
    assert K_CLUSTER_HISTOGRAM_KEY not in events[0], (
        f"R250: {K_CLUSTER_HISTOGRAM_KEY} reached the sink as "
        f"{events[0].get(K_CLUSTER_HISTOGRAM_KEY)!r} from a real coordinator on a graph run"
    )


def test_the_real_getter_and_the_label_derivation_agree_on_the_bucket_count() -> None:
    """The FFI crossing: a real `SelfPlayRunner` reports a list of ints, one per bucket, and
    the emitter labels exactly that many.

    A fresh runner has recorded nothing, so every bucket is 0 — and unlike R249's zero-sample
    means this zero is publishable ON A GRID RUN, because the producer EXISTS there and "no
    position recorded yet" is a truthful reading of it. The seeded-value parity is carried in
    Rust (`stats_snapshot_reads_back_each_private_atomic`), since nothing outside
    `mantis-selfplay` can seed the atomics.

    This is also the only row that would notice the getter being renamed or dropped: every
    other row here feeds `runner_stats` a double, and `pool_hooks`'s `getattr(..., None)`
    default swallows a missing getter into the honest-looking `None` arm.
    """
    runner = _engine.SelfPlayRunner(_engine.SelfPlayRunnerConfig(encoding_name="v6"))
    buckets = runner.k_cluster_histogram

    assert isinstance(buckets, list) and buckets, "the getter must report a non-empty vector"
    assert all(isinstance(b, int) for b in buckets), f"bucket counts must be ints: {buckets}"
    assert set(buckets) == {0}, f"a fresh runner has recorded nothing: {buckets}"

    published = k_cluster_histogram_block(
        SimpleNamespace(k_cluster_histogram=tuple(buckets)), graph_run=False
    )[K_CLUSTER_HISTOGRAM_KEY]
    assert len(published) == len(buckets), (
        "every bucket the engine reports must get a label — a mismatch means the emitter is "
        "silently dropping or inventing a bucket"
    )
    assert list(published)[-1].startswith(">"), "the last label is the out-of-range guard"


def test_both_engine_stubs_declare_the_k_histogram_getter() -> None:
    """The two `_engine.pyi` twins are the ONLY type authority for the FFI getters — pyright
    reads the stub, never the compiled module — and nothing else in the repo checks either
    against the getter's real shape.

    FALSIFYING MUTATION: drop the getter from either twin, or declare it `-> int`.
    """
    twins = (REPO_ROOT / "src" / "mantis" / "_engine.pyi",
             REPO_ROOT / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi")

    for twin in twins:
        text = twin.read_text(encoding="utf-8")
        assert f"def {K_CLUSTER_HISTOGRAM_KEY}(self) -> list[int]: ..." in text, (
            f"{twin.relative_to(REPO_ROOT)} must declare {K_CLUSTER_HISTOGRAM_KEY} as "
            f"`list[int]`: the getter hands back one count per bucket, and a stub that says "
            f"otherwise type-checks a consumer that fails at runtime."
        )
