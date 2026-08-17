# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally, and a number that must be re-edited
# whenever a row is added will eventually be wrong and then be read as evidence.
# The rows here are ONE claim — "the lever logs its own fire rate IN-RUN, all the way to the
# sink" — and R164's whole content is that the producer half and the arrival half must not
# live in different files: a test-visible-only counter passes every producer row ever written.
# The `iteration_complete` fakes (pool, buffer, rstats, sink) are the arrival half's rig and
# are shared by every row that drives it.
"""⊕ F-816-10 F4 — the fusion counters (LAW-18, R164, LAW-07).

Written by ORACLE-WRITE **before** the feature exists. Every row drives the REAL
`InferenceServer._run_graph_loop` over a real wire, and the arrival rows drive the REAL
`pool_hooks.inference_batch_timing` -> `emit_iteration_complete_event` chain, so a counter
that exists only in a snapshot a test reads reds here.

WHY DISTRIBUTIONS AND NOT MEANS. `_occupancy_agg`'s own docstring is the reason, and it
transfers verbatim: *"A mean ratio alone cannot distinguish 'always 1 request per forward'
from 'sometimes 64, sometimes 0'"*. For a MEMORY bound the tail IS the question — a mean fused
E of 400 k with a max of 9 M is a run that OOMs, and the two readings agree on the mean. The
histogram key is the bucket's power-of-two LOWER bound, the same rule the occupancy histogram
already uses.

WHY PER PART AND NOT PER POP. The part is what the GPU sees and what the cap bounds. A pop's
total is recoverable as the sum over its parts; the reverse is not.

The defect each row is the ONLY witness to:

- **FG4-01** — a producer that counts POPS instead of PARTS. It passes every non-splitting
  row, so it is paired with FG4-02 rather than asserted alone.
- **FG4-02** — a deleted split (`fusion_parts == 1` at any occupancy), and a `fusion_splits`
  that counts parts rather than split POPS.
- **FG4-03** — an EDGES-ONLY implementation. It passes every other row here (the MB-19
  mutation transplanted from `tests/train/test_graph_microbatch_bound.py`).
- **FG4-04** — R164's own failure mode: a counter that is visible to a test and never reaches
  the ONE channel. `iteration_complete` is driven end to end, not asserted on the snapshot.
- **FG4-05** — a fabricated zero block on a grid run, where there is NO producer
  (`docs/contracts/event_manifest.md`'s unproduced-field convention, the F-10 class).
- **FG4-06** — an off-by-one greedy that admits one over-bound part. Read off the INSTRUMENT
  rather than off the planner, because the instrument is what an operator will trust at the
  box and it can disagree with the planner if it measures the wrong tensor.
- **FG4-07** — an idle lever indistinguishable from a missing one. `fusion_splits` and
  `fusion_bound_hits` stay VISIBLE at 0 on the producing path (the `empty_polls` /
  `target_integrity_defects` posture).
- **FG4-08** — a mean smuggled in where a distribution was promised.
- **FG4-09** — `_forward_count` redefined to count PARTS. It is the denominator of
  `batch_fill_pct`, whose meaning is *requests per pop against `inference_batch_size`* — an
  occupancy, not a GPU-forward count — and it is banked on both sides of the R274(d) bench, so
  redefining it would move a metric silently (design §4.4).
- **FG4-10** — a distribution shipped without the bound it was measured against, which is
  unreadable for the same reason `batch_size`/`max_wait_ms` already travel with the occupancy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import _fused_graph_harness as H
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.pool_hooks import batch_fill_pct, inference_batch_timing
from mantis.train.events import emit_iteration_complete_event

#: Eight graphs whose per-graph edge counts are equal, so an edges cap at `k` graphs' worth
#: produces a plan of known M without depending on the ragged layout.
_EIGHT = [3, 3, 3, 3, 3, 3, 3, 3]


def _fusion(server: InferenceServer) -> dict[str, Any]:
    snap = server.batch_timing_snapshot()
    assert "fusion" in snap, (
        "`batch_timing_snapshot` carries no `fusion` block — the lever has no in-run "
        "instrument at all (LAW-18)")
    return snap["fusion"]


# ═══ FG4-01/02/03 — the producer rows ════════════════════════════════════════════════════
def test_fg4_01_a_pop_that_fits_reports_one_part_and_no_split(monkeypatch) -> None:
    """FG4-01 — the non-splitting path is INSTRUMENTED, not silent: one part, zero splits, one
    histogram sample on each distribution. This is the path every smoke config takes."""
    payload = H.build_payload(_EIGHT)
    ec, nc = H.per_graph_counts(payload)
    server, batcher, _ = H.drive_one_pop(monkeypatch, payload)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)

    assert f["fusion_parts"] == 1, f"one pop that fits is ONE forward; got {f['fusion_parts']}"
    assert f["fusion_splits"] == 0
    assert f["fusion_bound_hits"] == {"edges": 0, "nodes": 0}
    assert f["fused_batch_edges"]["count"] == 1
    assert f["fused_batch_nodes"]["count"] == 1
    assert f["fused_batch_edges"]["total"] == int(ec.sum())
    assert f["fused_batch_nodes"]["total"] == int(nc.sum())


def test_fg4_02_a_pop_that_must_split_reports_its_parts_and_its_cuts(monkeypatch) -> None:
    """FG4-02 — M parts, ONE split pop, M histogram samples, and M-1 edge-driven cuts.

    Two mutations die here and nowhere else: deleting the split (`fusion_parts == 1`) and
    counting parts as pops (`fusion_splits == M`). `fusion_splits` is the LEVER'S OWN FIRE
    RATE — LAW-18's subject — so it counts POPS THAT SPLIT, not cuts."""
    payload = H.build_payload(_EIGHT)
    ec, nc = H.per_graph_counts(payload)
    cap_e = 2 * int(ec[0])          # exactly two graphs per forward
    server, batcher, net = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=cap_e, max_fused_nodes=10 ** 9)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)
    expected_m = 4

    assert f["fusion_parts"] == expected_m, (
        f"eight equal graphs under a two-graph cap is {expected_m} forwards; got "
        f"{f['fusion_parts']}")
    assert f["fusion_splits"] == 1, (
        f"ONE pop split, so the lever fired once; got {f['fusion_splits']} — a value of "
        f"{expected_m} means the counter is counting parts, not split pops")
    assert f["fusion_bound_hits"]["edges"] == expected_m - 1, (
        "every cut was forced by the EDGES member and must be attributed to it")
    assert f["fused_batch_edges"]["count"] == expected_m
    assert f["fused_batch_nodes"]["count"] == expected_m
    assert f["fused_batch_edges"]["total"] == int(ec.sum()), (
        "the parts' edge totals must sum to the pop's — a dropped part shows up here")
    assert f["fused_batch_nodes"]["total"] == int(nc.sum())
    assert len(net.calls) == expected_m, (
        f"the model must be forwarded once per part; it was called {len(net.calls)} times")


def test_fg4_03_a_node_driven_split_attributes_its_cuts_to_the_node_member(
    monkeypatch
) -> None:
    """FG4-03 — the MB-19 mutation: an edges-only implementation passes every other row here.

    The bank member is node-heavy and edge-light by construction, which is also the point
    D-5 makes the calibration sweep carry: V-D's death must be reconfirmed by measurement, not
    by hand count."""
    payload = H.build_payload(_EIGHT, edges_per_graph=[1] * 8)
    _ec, nc = H.per_graph_counts(payload)
    cap_n = 2 * int(nc[0])
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=10 ** 9, max_fused_nodes=cap_n)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)

    assert f["fusion_parts"] > 1, (
        "the NODES member did not bind — an edges-only implementation is unbounded in N, and "
        "design §1.4 shows the unbounded member is the LARGER of the two in the worst case")
    assert f["fusion_bound_hits"]["nodes"] >= 1, (
        f"a node-driven split attributed no cut to the node member: {f['fusion_bound_hits']}")
    assert f["fusion_bound_hits"]["edges"] == 0, (
        "no cut here was forced by edges; mis-attribution makes the instrument lie about "
        "which member to re-fit at the box")


# ═══ FG4-06/07/08/10 — the shape of what is reported ═════════════════════════════════════
_BOUND_BANK = [
    ("uniform", [3] * 8, None),
    ("ragged", [2, 5, 3, 7, 4, 6, 1, 8], None),
    ("one dominant", [1, 1, 20, 1, 1], None),
    ("edge-light node-heavy", [9] * 6, [1] * 6),
]


@pytest.mark.parametrize(("label", "legal", "edges"), _BOUND_BANK,
                         ids=[r[0] for r in _BOUND_BANK])
def test_fg4_06_no_part_ever_exceeds_either_cap_on_the_instrument(
    monkeypatch, label: str, legal: list[int], edges: list[int] | None
) -> None:
    """FG4-06 — the bound, read off the INSTRUMENT rather than off the planner.

    A planner that partitions correctly and an instrument that measures the wrong tensor
    disagree, and it is the instrument the operator reads at the box when deciding whether the
    cap held. `max` is the reading that matters for a memory bound; a mean cannot fail."""
    payload = H.build_payload(legal, edges)
    ec, nc = H.per_graph_counts(payload)
    cap_e, cap_n = int(ec.max()) + 1, int(nc.max()) + 1
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=cap_e, max_fused_nodes=cap_n)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)

    assert f["fused_batch_edges"]["max"] <= f["caps"]["max_fused_edges"], (
        f"a part carried {f['fused_batch_edges']['max']} edges against a cap of "
        f"{f['caps']['max_fused_edges']} — the bound does not bound")
    assert f["fused_batch_nodes"]["max"] <= f["caps"]["max_fused_nodes"], (
        f"a part carried {f['fused_batch_nodes']['max']} nodes against a cap of "
        f"{f['caps']['max_fused_nodes']}")


def test_fg4_07_the_lever_stays_visible_at_zero_on_the_producing_path(monkeypatch) -> None:
    """FG4-07 — an IDLE lever must be distinguishable from a MISSING one.

    `fusion_splits == 0` on a graph run that never split is a measurement; `fusion` absent or
    `None` on the same run would be "no producer". The two mean opposite things to whoever
    reads the burst, and §11's third falsifier (`fusion_splits == 0` across a burst that
    reaches ply > 120) can only fire if the zero is published."""
    payload = H.build_payload([3, 3])
    server, _batcher, _ = H.drive_one_pop(monkeypatch, payload)
    f = _fusion(server)
    assert f is not None, "a graph run has a producer; the block must not be None"
    assert f["fusion_splits"] == 0
    assert f["fusion_bound_hits"] == {"edges": 0, "nodes": 0}


def test_fg4_07_the_instrument_is_defined_before_the_first_forward(monkeypatch) -> None:
    """FG4-07 second limb — read before any pop: the caps are already known (they were
    resolved at construction, §3.3), the counters are defined zeros and the two distributions
    are `None` because no part has been measured. No division by zero, no fabricated zero."""
    import mantis.selfplay.graph_collate as collate_mod

    monkeypatch.setattr(collate_mod, "collate_graph_batch", H.collate_from_payload)
    server = InferenceServer(
        H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(4_500_000, 170_000),
        batcher=H.ScriptedGraphBatcher([]), encoding_spec=H.GRAPH_SPEC,
    )
    f = _fusion(server)
    assert f["caps"] == {"max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}
    assert (f["fusion_parts"], f["fusion_splits"]) == (0, 0)
    assert f["fused_batch_edges"] is None and f["fused_batch_nodes"] is None, (
        "no part has been measured, so the distributions have no producer yet — `None`, "
        "never a zeroed histogram")


def test_fg4_08_the_distributions_are_power_of_two_bucketed_histograms(monkeypatch) -> None:
    """FG4-08 — histograms with power-of-two LOWER-bound keys, not means.

    Asserted as a distribution (every key a power of two, every part in the bucket whose lower
    bound it clears) rather than as a summary, because a mean fused-E is exactly the reading
    that cannot tell a safe run from one about to OOM."""
    payload = H.build_payload([1, 1, 1, 30])
    ec, nc = H.per_graph_counts(payload)
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=int(ec.max()), max_fused_nodes=10 ** 9)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)

    for name in ("fused_batch_edges", "fused_batch_nodes"):
        hist = f[name]["histogram"]
        assert hist, f"{name} carries no histogram — a mean was shipped instead"
        for key, count in hist.items():
            k = int(key)
            assert k > 0 and (k & (k - 1)) == 0, (
                f"{name} histogram key {key!r} is not a power-of-two lower bound")
            assert count >= 1
        assert sum(hist.values()) == f[name]["count"], (
            f"{name}'s histogram does not account for every part")
        assert f[name]["min"] <= f[name]["mean"] <= f[name]["max"]
    biggest = f["fused_batch_edges"]["max"]
    bucket = 1 << (int(biggest).bit_length() - 1)
    assert str(bucket) in f["fused_batch_edges"]["histogram"], (
        f"the largest part ({biggest} edges) is not in bucket {bucket} — the key is not the "
        "bucket's LOWER bound")


def test_fg4_10_the_caps_travel_with_the_distributions(monkeypatch) -> None:
    """FG4-10 — an occupancy is unreadable without the bound it was measured against, the
    same reason `batch_size`/`max_wait_ms` already ride the block. A histogram whose maximum
    is 4.4 M edges says nothing until the cap beside it says 4.5 M or 9 M."""
    payload = H.build_payload([3, 3])
    server, _batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=4_500_000, max_fused_nodes=170_000)
    assert _fusion(server)["caps"] == {
        "max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}


# ═══ FG4-09 — `_forward_count` stays one per POP ═════════════════════════════════════════
def test_fg4_09_forward_count_stays_one_per_pop_under_a_split(monkeypatch) -> None:
    """FG4-09 — design §4.4's first behavioural delta IMPL must not accidentally "fix".

    `_forward_count` is `batch_fill_pct`'s DENOMINATOR (`pool_hooks.batch_fill_pct`), and that
    metric means *requests per pop against `inference_batch_size`* — an occupancy. Counting
    parts there would divide by M and silently move a number banked on both sides of the
    R274(d) bench. `fusion_parts` is where GPU forwards are counted, and this row asserts the
    two are DIFFERENT under a split, which is the only regime that can tell them apart."""
    payload = H.build_payload(_EIGHT)
    ec, _nc = H.per_graph_counts(payload)
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=2 * int(ec[0]), max_fused_nodes=10 ** 9,
        batch_size=64)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    f = _fusion(server)

    assert server.forward_count == 1, (
        f"one served pop is ONE `_forward_count`; got {server.forward_count}. It is "
        "`batch_fill_pct`'s denominator, not a GPU-forward count (design §4.4)")
    assert f["fusion_parts"] == 4, "the drive must actually have split"
    assert server.total_requests == 8
    assert batch_fill_pct(_TelemetryPool(server)) == pytest.approx(8 / 64 * 100.0), (
        "batch_fill_pct moved — the occupancy metric now reads the split instead of the pop")
    occ = server.batch_timing_snapshot()["occupancy"]
    assert occ["count"] == 1 and occ["total"] == 8, (
        "the occupancy block is measured at the POP and must be blind to the split "
        "(design §6.2: the split is downstream of the pop)")


def test_fg4_09_collate_is_recorded_once_per_part_not_once_per_pop(monkeypatch) -> None:
    """FG4-09 second limb — design §4.4's OTHER behavioural delta, recorded so it is not read
    as drift when someone diffs the two counters.

    `_record_collate` now fires once per PART, so `collate.count == sum(M)` where it used to
    equal `queue_wait.count`. The asymmetry is already the documented design of those two
    counters (a batch whose collate raises still contributes a real wait sample), and this row
    is what stops the new inequality being mistaken for a leak — and stops an implementation
    that collates ONCE and slices tensors afterwards, which would be the post-collate design
    §4.1(1) rejects: a design whose first allocation is proportional to the uncapped quantity
    cannot meet a bound."""
    payload = H.build_payload(_EIGHT)
    ec, _nc = H.per_graph_counts(payload)
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=2 * int(ec[0]), max_fused_nodes=10 ** 9)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"
    snap = server.batch_timing_snapshot()

    assert snap["collate"]["count"] == snap["fusion"]["fusion_parts"] == 4, (
        f"collate.count={snap['collate']['count']} against "
        f"fusion_parts={snap['fusion']['fusion_parts']} — the collate must run PER PART "
        "(pre-collate splitting is the mechanism; a single whole-pop collate materialises "
        "the full-E tensors the cap exists to bound)")
    assert snap["queue_wait"]["count"] == 1, (
        "the wait is measured at the POP; only the collate follows the split")


# ═══ FG4-04/05 — arrival at the sink ═════════════════════════════════════════════════════
class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _TelemetryPool:
    """The narrow telemetry surface over a REAL inference server — the batching member goes
    through the REAL `pool_hooks` function, so this drives the production producer."""

    gumbel_mcts = True
    avg_game_length = 12.0
    x_winrate = 0.5
    o_winrate = 0.4
    draw_rate = 0.1
    draws = 1
    sims_per_sec = 100.0
    recent_move_histories: list[list[tuple[int, int]]] = []

    def __init__(self, server: InferenceServer) -> None:
        self._inference_server = server

    @property
    def batch_fill_pct(self) -> float:
        return batch_fill_pct(self)

    @property
    def inference_batch_timing(self) -> dict[str, Any]:
        return inference_batch_timing(self)


class _Buffer:
    size = 7
    capacity = 64


class _RStats:
    mcts_mean_depth = 3.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


def _emit(pool: Any) -> dict[str, Any]:
    sink = _ListSink()
    emit_iteration_complete_event(
        11, 0.0, 10, 4, pool, _Buffer(), {}, {}, 64,
        lambda: 0.0, None, {}, _RStats(), sink,
    )
    assert len(sink.events) == 1
    return sink.events[0]


def test_fg4_04_the_fusion_block_reaches_the_sink_on_iteration_complete(monkeypatch) -> None:
    """FG4-04 — R164, discharged the only way it can be: the block travels server -> hook ->
    builder -> sink, whole, and is compared against the server's own snapshot.

    A test-visible-only counter satisfies FG4-01..03 completely and reaches nobody. This row
    is the difference, and it is why the counters ride an EXISTING event field
    (`iteration_complete.inference_batching`) rather than a new one nothing consumes."""
    payload = H.build_payload(_EIGHT)
    ec, _nc = H.per_graph_counts(payload)
    server, batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=2 * int(ec[0]), max_fused_nodes=10 ** 9)
    assert batcher.failures == [], f"the drive failed: {batcher.failures}"

    payload_event = _emit(_TelemetryPool(server))
    block = payload_event["inference_batching"]
    assert block is not None
    assert "fusion" in block, (
        "`fusion` did not reach `iteration_complete.inference_batching` — a counter visible "
        "only to a test FAILS LAW-18/R164 by construction")
    fusion = block["fusion"]
    assert fusion["fusion_parts"] == 4
    assert fusion["fusion_splits"] == 1
    assert fusion["caps"]["max_fused_edges"] == 2 * int(ec[0])
    assert fusion == server.batch_timing_snapshot()["fusion"], (
        "the emitted block is not the server's own snapshot — something restated it on the "
        "way, and a restatement can drift")


class _FakeDenseBatcher:
    """Drives the dense `run()` loop for `n_batches` iterations, then stops it."""

    def __init__(self, feature_len: int, n_batches: int = 2, n_requests: int = 2) -> None:
        self._left = n_batches
        self._ids = list(range(1, n_requests + 1))
        self._batch = np.ascontiguousarray(np.zeros((n_requests, feature_len), np.float32))
        self.server: InferenceServer | None = None
        self.closed = 0

    def next_inference_batch(self, batch_size: int, max_wait_ms: float):
        if self._left <= 0:
            assert self.server is not None
            self.server._stop_event.set()
            return [], self._batch
        self._left -= 1
        return list(self._ids), self._batch

    def submit_inference_results(self, ids, policies, values) -> None:
        return None

    def submit_inference_failure(self, ids, error_msg: str) -> None:
        return None

    def bump_model_version(self) -> int:
        return 1

    def close(self) -> None:
        self.closed += 1


class _DenseStubNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = int(x.shape[0])
        return (torch.zeros(n, H.GRID_SPEC.policy_logit_count), torch.zeros(n, 1),
                torch.zeros(n, 1))


def test_fg4_05_a_grid_run_reports_no_fusion_producer_rather_than_a_zero() -> None:
    """FG4-05 — the whole block is `None` on a grid run, after REAL dense forwards.

    The grid branch never reads the caps (design §3.3: the dense batch is a fixed-shape tensor
    already bounded by `inference_batch_size`), so it has no producer here. A zeroed block
    would read as "the fusion lever ran and never fired" — the F-10 class in miniature, and
    the exact misreading `docs/contracts/event_manifest.md`'s unproduced-field convention
    exists to prevent."""
    feature_len = H.GRID_SPEC.n_planes * H.GRID_SPEC.trunk_size * H.GRID_SPEC.trunk_size
    batcher = _FakeDenseBatcher(feature_len, n_batches=2)
    server = InferenceServer(
        _DenseStubNet(), torch.device("cpu"), H.grid_cfg(batch_size=4),
        batcher=batcher, encoding_spec=H.GRID_SPEC,
    )
    batcher.server = server
    server.run()

    snap = server.batch_timing_snapshot()
    assert server.forward_count == 2, "the dense loop must actually have run"
    assert snap["representation"] == "grid"
    assert "fusion" in snap, "the key must be PRESENT and carry None, not be absent"
    assert snap["fusion"] is None, (
        f"a grid run fabricated a fusion block: {snap['fusion']}")
    assert _emit(_TelemetryPool(server))["inference_batching"]["fusion"] is None
