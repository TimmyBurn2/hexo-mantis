# >300 justify (R8): ONE claim — the lever logs its own fire rate in-run, all the way to the
# sink — whose producer half and arrival half must not live in different files, because a
# test-visible-only counter passes every producer row ever written.
"""The fusion counters, driven over the real wire from producer to sink.

Distributions, not means: for a MEMORY bound the tail IS the question — a mean fused E of 400 k
with a max of 9 M is a run that OOMs, and both readings agree on the mean. Histogram keys are
the bucket's power-of-two LOWER bound. Per PART, not per pop: the part is what the cap bounds.
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

#: Equal per-graph edge counts, so an edges cap at `k` graphs' worth gives a known M.
_EIGHT = [3, 3, 3, 3, 3, 3, 3, 3]


def _fusion(server: InferenceServer) -> dict[str, Any]:
    snap = server.batch_timing_snapshot()
    assert "fusion" in snap, (
        "`batch_timing_snapshot` carries no `fusion` block — the lever has no in-run "
        "instrument at all (LAW-18)")
    return snap["fusion"]


def test_fg4_01_a_pop_that_fits_reports_one_part_and_no_split(monkeypatch) -> None:
    """The non-splitting path — every smoke config's path — is instrumented, not silent."""
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
    """A split pop reports M parts, ONE split, M histogram samples and M-1 edge-driven cuts.

    `fusion_splits` is the lever's own fire rate, so it counts POPS THAT SPLIT, not cuts.
    Killer: delete the split, or count parts as pops.
    """
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
    """A node-driven split attributes its cuts to the node member.

    Killer: an edges-only implementation, which passes every other row here.
    """
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
    """No part exceeds either cap, read off the INSTRUMENT rather than off the planner —
    a correct planner and an instrument measuring the wrong tensor disagree."""
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
    """An IDLE lever stays visible at zero: `fusion_splits == 0` is a measurement, while an
    absent or `None` block means "no producer"."""
    payload = H.build_payload([3, 3])
    server, _batcher, _ = H.drive_one_pop(monkeypatch, payload)
    f = _fusion(server)
    assert f is not None, "a graph run has a producer; the block must not be None"
    assert f["fusion_splits"] == 0
    assert f["fusion_bound_hits"] == {"edges": 0, "nodes": 0}


def test_fg4_07_the_instrument_is_defined_before_the_first_forward(monkeypatch) -> None:
    """Before the first forward the caps are known, the counters are zero and the
    distributions are `None` — no division by zero, no fabricated zero."""
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
    """The distributions are power-of-two LOWER-bound bucketed histograms, not means."""
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
    """The caps travel with the distributions — a max of 4.4 M edges says nothing until the
    cap beside it says 4.5 M or 9 M."""
    payload = H.build_payload([3, 3])
    server, _batcher, _ = H.drive_one_pop(
        monkeypatch, payload, max_fused_edges=4_500_000, max_fused_nodes=170_000)
    assert _fusion(server)["caps"] == {
        "max_fused_edges": 4_500_000, "max_fused_nodes": 170_000}


def test_fg4_09_forward_count_stays_one_per_pop_under_a_split(monkeypatch) -> None:
    """`_forward_count` stays one per POP under a split.

    It is `batch_fill_pct`'s denominator and that metric is an occupancy, so counting parts
    there would divide by M and silently move a banked number; `fusion_parts` counts forwards.
    """
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
    """Collate is recorded once per PART, so `collate.count == sum(M)` rather than
    `queue_wait.count` — by design, not a leak. Killer: collate ONCE and slice afterwards,
    whose first allocation is proportional to the uncapped quantity."""
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


class _ListSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))


class _TelemetryPool:
    """Narrow telemetry surface over a REAL server, through the REAL `pool_hooks`."""

    search_kind = "gumbel"
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
    """The block travels server -> hook -> builder -> sink whole, matching the snapshot —
    a test-visible-only counter satisfies every producer row above and reaches nobody."""
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
