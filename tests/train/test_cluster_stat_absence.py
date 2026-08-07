"""ADJ-D32 / R249 + R250 — the phantom cluster metrics: zero samples publish NOTHING.

`iteration_complete` published `cluster_value_std_mean: 0.0` and
`cluster_policy_disagreement_mean: 0.0` on run5's graph arm, and both zeros were
FABRICATIONS. The chain that made them: the graph arm returns into
`infer_and_expand_graph` (`search_drive.rs:253`) BEFORE any variance code runs — the
`ClusterVarianceAtomics` are not even passed to the graph function — so
`cluster_variance_samples` is permanently 0 there; the only writers are the dense arm's
`*k >= 2` branch, and `gnn_axis_v1` has `k_max = 1`. With the count pinned at 0,
`derived_mean_f64`'s `count == 0 -> 0.0` guard turned "no measurement" into "measured
zero" forever, and a reader cannot tell a settled cluster ensemble from an absent one.

The fix has two halves and this file pins both:

  R249  zero count -> `None` at the bridge getter, and the emitter DROPS a `None` mean
        rather than publishing it. Never `null`, never 0.
  R250  on a GRAPH representation the three cluster fields are ABSENT — not `None` —
        because there is no producer on that arm at all. `mcts_root_concentration`
        STAYS: it is accumulated once per search in `play_one_move`
        (`search_drive.rs:678`), path-independent, and is not a cluster field.

FALSIFYING MUTATIONS, each named against the test that ACTUALLY catches it (a mutation
attributed to the wrong test is worse than an unattributed one — a future reader trusts it
and stops looking, the SF-7 class):
  (a) make the builder publish the `None` mean as a value (`{k: v}` unconditionally) —
      the zero-sample tests here red on the key being present;
  (b) drop the graph arm from `regime_gated_cluster_stats` — the graph tests here red;
  (c) re-coerce `None` to 0.0 in `pool_hooks._optional_mean` — the producer-chain test here
      reds, because that snapshot layer is what this file drives.
  NOT caught here: restoring `derived_mean_f64`'s zero-count `0.0`. Nothing in this file
  crosses the FFI — the producer-chain test feeds `runner_stats` a hand-built runner double
  whose getters already read `None`. That mutation's pin is the Rust unit test
  `runner.rs::tests::zero_count_derived_mean_is_none_never_zero` and its Python leg
  `tests/bridge/test_runner_derived_means.py::test_cluster_means_read_none_not_zero_at_zero_samples`,
  which builds a REAL `SelfPlayRunner` and reads the real getter.

The PUCT/Gumbel axis (CONFRES S2 "always-keyed, value under PUCT, None under Gumbel") is
NOT this card's subject and is pinned here as a regression guard only.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mantis.selfplay.pool_hooks import runner_stats
from mantis.train.events import emit_iteration_complete_event, is_graph_run

#: The three fields R250 makes ABSENT on a graph run.
CLUSTER_KEYS = (
    "cluster_value_std_mean",
    "cluster_policy_disagreement_mean",
    "cluster_variance_sample_count",
)
#: The two derived means R249 drops at zero count (the count itself is truthful and stays).
CLUSTER_MEANS = CLUSTER_KEYS[:2]

# Identity blocks in the shape `RunConfig` dumps — `identity.representation` is a required,
# defaultless, registry-cross-checked field (`config/schema/core.py`), so reading it is
# reading the operator's validated declaration, not a second authority.
GRID_CONFIG: dict[str, Any] = {"identity": {"encoding": "v6_live2_ls", "representation": "grid"}}
GRAPH_CONFIG: dict[str, Any] = {"identity": {"encoding": "gnn_axis_v1",
                                             "representation": "graph"}}


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _Pool:
    """The narrow `PoolTelemetryLike` surface, in run5's PUCT regime (`gumbel_mcts: false`
    at `configs/run5.yaml:160`) — the arm on which the fabricated zeros were observed."""

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


def _rstats(*, value_std: float | None, disagreement: float | None, count: int) -> Any:
    """A `RunnerStats`-shaped double. `value_std`/`disagreement` are `float | None` — the
    post-R249 bridge type, where `None` means "no samples", never "measured zero"."""
    return SimpleNamespace(
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
        cluster_value_std_mean=value_std,
        cluster_policy_disagreement_mean=disagreement,
        cluster_variance_sample_count=count,
    )


def _emit(config: dict[str, Any], rstats: Any, pool: Any | None = None) -> dict[str, Any]:
    sink = _Sink()
    emit_iteration_complete_event(
        11, 0.0, 10, 4, pool or _Pool(), _Buffer(), config, {}, 64,
        lambda: 0.0, None, {}, rstats, sink,
    )
    assert len(sink.events) == 1
    return sink.events[0]


# ═══ pin 1 — the MUTATION pin: a zero-count mean is never published, in any form ═══
def test_zero_sample_dense_run_publishes_neither_cluster_mean() -> None:
    """R249 — dense arm, zero samples: BOTH means are dropped from the payload. Not 0.0
    (the fabrication), not `null` (which a JSON reader coerces to 0 just as readily)."""
    payload = _emit(GRID_CONFIG, _rstats(value_std=None, disagreement=None, count=0))

    for key in CLUSTER_MEANS:
        assert key not in payload, (
            f"R249: {key} must be ABSENT from iteration_complete when "
            f"cluster_variance_sample_count is 0 — a mean over zero samples is not a "
            f"measurement. Got {payload.get(key)!r}. If 0.0, the bridge zero-guard is back; "
            f"if None, the emitter is publishing None fields instead of dropping them."
        )
    assert payload["cluster_variance_sample_count"] == 0, (
        "R249: the SAMPLE COUNT is truthful at zero and stays — it is the field that lets a "
        "reader see the means are missing because nothing was measured."
    )
    assert payload["mcts_root_concentration"] == 0.25, (
        "mcts_root_concentration is not a cluster field and is live on both arms "
        "(search_drive.rs:678, path-independent) — it must survive the drop."
    )


def test_the_producer_chain_carries_none_from_the_bridge_getter_to_the_sink() -> None:
    """R249 across the two PYTHON layers in one drive: a getter reading `None` (zero count)
    travels through the REAL `pool_hooks.runner_stats` snapshot into the REAL builder and
    arrives as an absent key.

    FALSIFYING MUTATION: re-coerce `None` to 0.0 in `pool_hooks._optional_mean` — a 0.0
    leaving the snapshot is a value, and a value gets published.

    NOT a pin on the Rust getter: the runner double below hands `runner_stats` a `None`
    directly, so reverting `derived_mean_f64`'s zero-count arm cannot reach this test. That
    mutation is caught at the FFI crossing by
    `tests/bridge/test_runner_derived_means.py::test_cluster_means_read_none_not_zero_at_zero_samples`."""
    pool = SimpleNamespace(_runner=SimpleNamespace(
        cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None,
        cluster_variance_sample_count=0,
        mcts_mean_depth=3.0,
        mcts_mean_root_concentration=0.25,
    ))

    rstats = runner_stats(pool)
    assert rstats.cluster_value_std_mean is None, (
        "R249: pool_hooks.runner_stats must carry the getter's None through, NOT coerce it "
        "to 0.0 — `float(None)` would raise, and a `getattr(..., 0.0)` default would "
        "re-fabricate exactly the zero this card removes."
    )
    assert rstats.cluster_policy_disagreement_mean is None

    payload = _emit(GRID_CONFIG, rstats)
    for key in CLUSTER_MEANS:
        assert key not in payload, f"R249: {key} reached the sink as {payload.get(key)!r}"


# ═══ pin 2 — R250 absence on a graph representation ═══
def test_graph_run_carries_no_cluster_field_at_all() -> None:
    """R250 — on a graph run the event stream carries NO cluster field: not the two means,
    and not the sample count either. The count is not "0 samples so far" on that arm, it is
    "this arm has no such instrument"; publishing a 0 invites the same misreading."""
    payload = _emit(GRAPH_CONFIG, _rstats(value_std=None, disagreement=None, count=0))

    for key in CLUSTER_KEYS:
        assert key not in payload, (
            f"R250: {key} must be ABSENT on a graph representation — the cluster-variance "
            f"atomics are structurally unreachable there (search_drive.rs:253 returns into "
            f"infer_and_expand_graph before any variance code runs, and the atomics are not "
            f"passed to it). Got {payload.get(key)!r}."
        )
    assert payload["mcts_root_concentration"] == 0.25, (
        "R250 keeps mcts_root_concentration: it is live on the graph path and is not a "
        "cluster field."
    )


def test_graph_absence_holds_even_when_the_snapshot_reads_nonzero() -> None:
    """R250 is an ABSENCE rule about the arm, not a zero-check about the reading. A graph
    run whose snapshot somehow carried cluster numbers would be reporting something the
    graph arm cannot produce — the fields stay absent, and the anomaly does not get laundered
    into the event stream as a plausible measurement."""
    payload = _emit(GRAPH_CONFIG, _rstats(value_std=0.4, disagreement=0.3, count=9))

    for key in CLUSTER_KEYS:
        assert key not in payload, (
            f"R250: {key} must be absent on graph regardless of the snapshot reading; "
            f"got {payload.get(key)!r}"
        )


# ═══ pin 3 — a dense run with samples still publishes the REAL values ═══
def test_dense_run_with_samples_publishes_the_real_values() -> None:
    """The fix must not silence a LIVE instrument: with samples on the dense arm, all three
    fields carry their real readings, unrounded and unaltered."""
    payload = _emit(GRID_CONFIG, _rstats(value_std=0.375, disagreement=0.5, count=7))

    assert payload["cluster_value_std_mean"] == 0.375
    assert payload["cluster_policy_disagreement_mean"] == 0.5
    assert payload["cluster_variance_sample_count"] == 7


def test_dense_run_drops_only_the_mean_that_has_no_samples() -> None:
    """The drop is per-FIELD, driven by the value the producer handed over — a live reading
    beside a missing one is published, so a half-wired producer cannot silence the half that
    works."""
    payload = _emit(GRID_CONFIG, _rstats(value_std=0.125, disagreement=None, count=3))

    assert payload["cluster_value_std_mean"] == 0.125
    assert "cluster_policy_disagreement_mean" not in payload
    assert payload["cluster_variance_sample_count"] == 3


# ═══ the R250 predicate itself — its documented degenerate arms, pinned ═══
def test_is_graph_run_reads_the_declared_representation() -> None:
    """The positive and negative arms the predicate exists for."""
    assert is_graph_run(GRAPH_CONFIG) is True
    assert is_graph_run(GRID_CONFIG) is False


def test_is_graph_run_falls_back_to_non_graph_on_every_degenerate_config() -> None:
    """`is_graph_run` documents a NON-GRAPH fallback for a config that declares nothing, and
    the direction is load-bearing in one direction only: non-graph means the R249 zero-count
    rules apply, so an undeclared config still publishes a live dense instrument. Flipping
    any of these to True would SILENCE a real producer — permanently and greenly, since the
    graph arm asserts absence — so each documented arm is pinned rather than left to the
    reading of one `.get` chain.

    This also guards the degradation flagged as N-2: several coordinators historically got
    `full_config={}` (`tests/test_run_strict_composition.py`'s O-S1b), and `{}` must land on
    the arm that reports rather than the arm that omits.
    """
    assert is_graph_run({}) is False, "no identity block at all"
    assert is_graph_run({"identity": {}}) is False, "identity block with no representation"
    assert is_graph_run({"identity": {"representation": None}}) is False, "explicit None"
    assert is_graph_run({"identity": "gnn_axis_v1"}) is False, (
        "a non-Mapping identity — the `isinstance(identity, Mapping)` guard the code "
        "documents; without it this raises AttributeError mid-emit"
    )
    assert is_graph_run({"identity": {"representation": "grid"}}) is False


# ═══ regression guard — the PUCT/Gumbel axis (CONFRES S2) is untouched by this card ═══
def test_gumbel_grid_run_keeps_the_schema_stable_none_keys() -> None:
    """CONFRES S2, unchanged: under Gumbel the PUCT-descent-specific stats are always-keyed
    and carry `None`, so the grid payload's shape is regime-stable. R249 changes the
    zero-count arm, not this one."""
    payload = _emit(GRID_CONFIG, _rstats(value_std=None, disagreement=None, count=0),
                    pool=_GumbelPool())

    for key in (*CLUSTER_KEYS, "mcts_root_concentration"):
        assert key in payload and payload[key] is None, (
            f"CONFRES S2: {key} must stay keyed-with-None under Gumbel on a grid run; got "
            f"{payload.get(key, '<absent>')!r}"
        )


def test_graph_absence_wins_over_the_gumbel_none_keys() -> None:
    """R250 is about the ARM and outranks the regime gate: a graph run under Gumbel still
    carries no cluster key. `mcts_root_concentration` keeps its regime-gated `None`."""
    payload = _emit(GRAPH_CONFIG, _rstats(value_std=None, disagreement=None, count=0),
                    pool=_GumbelPool())

    for key in CLUSTER_KEYS:
        assert key not in payload, f"R250 outranks the S2 regime gate; got {payload[key]!r}"
    assert "mcts_root_concentration" in payload and payload["mcts_root_concentration"] is None
