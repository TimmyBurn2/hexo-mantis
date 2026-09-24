"""Suite Q3 — the inference batching instrument (collector wait / collate / occupancy).

Both batching fields get their producer test here, driving the REAL producer through the real
hook into an injected sink. The occupancy DISTRIBUTION is asserted and not just the ratio,
because a mean cannot separate "one request per forward" from "sometimes 64, sometimes 0".
"""
from __future__ import annotations

import time
from typing import Any

import pytest
import torch

import _fused_graph_harness as H
from _fused_graph_harness import (
    CountingGraphBatcher,
    FiniteGraphNet,
    TelemetryPool,
    emit_iteration,
    hand_built_batch,
    server_cfg,
    wire_for,
)
import mantis.selfplay.graph_collate as collate_mod
from mantis.selfplay.graph_collate import GraphBatch
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.pool_hooks import batch_fill_pct


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


def _run_graph_server(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    counts: list[int],
    *,
    wait_s: float = 0.0,
    collate_s: float = 0.0,
    batch_size: int = 8,
) -> InferenceServer:
    """Build a graph `InferenceServer`, drive its loop over `counts`, return the server."""
    batch = hand_built_batch()

    def _collate(*_a: Any, **_kw: Any) -> GraphBatch:
        if collate_s:
            time.sleep(collate_s)
        return batch

    monkeypatch.setattr(collate_mod, "collate_graph_batch", _collate)
    batcher = CountingGraphBatcher(wire_for(), counts, wait_s=wait_s)
    server = InferenceServer(
        FiniteGraphNet(), device, server_cfg(inference_batch_size=batch_size),
        batcher=batcher, encoding_spec=H.GRAPH_SPEC,
    )
    batcher.server = server
    server.run()
    return server


def test_the_graph_loop_measures_its_own_collector_wait_and_collate_cost(
    device, monkeypatch
) -> None:
    """The collector wait inside `next_graph_batch` and the collate cost are MEASURED per served
    batch: a wait pegged at `inference_max_wait_ms` says the batch threshold was never reached.
    """
    server = _run_graph_server(
        device, monkeypatch, [2, 2, 2], wait_s=0.005, collate_s=0.002,
    )
    snap = server.batch_timing_snapshot()

    assert snap["representation"] == "graph"
    assert snap["batch_size"] == 8
    assert snap["queue_wait"]["count"] == 3
    assert snap["queue_wait"]["min_ms"] >= 5.0
    assert snap["queue_wait"]["mean_ms"] >= 5.0
    assert snap["collate"]["count"] == 3
    assert snap["collate"]["min_ms"] >= 2.0
    # The stop-pop returned no requests: a deadline that expired empty is counted apart
    # from the served waits, never folded into their mean.
    assert snap["empty_polls"] == 1


def test_an_occupancy_histogram_separates_always_one_from_a_mixed_load(
    device, monkeypatch
) -> None:
    """min/max/histogram resolve what the mean cannot: two loads with the SAME mean occupancy
    are indistinguishable by ratio, so the histogram is asserted as a distribution."""
    server = _run_graph_server(device, monkeypatch, [1, 1, 8], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert occ["count"] == 3
    assert occ["total"] == 10
    assert occ["min"] == 1
    assert occ["max"] == 8
    assert occ["histogram"] == {"1": 2, "8": 1}

    flat = _run_graph_server(device, monkeypatch, [1, 1, 1], batch_size=8)
    flat_occ = flat.batch_timing_snapshot()["occupancy"]
    assert flat_occ["min"] == 1 and flat_occ["max"] == 1
    assert flat_occ["histogram"] == {"1": 3}
    assert flat_occ["histogram"] != occ["histogram"]


def test_the_instrument_is_defined_before_the_first_forward(device, monkeypatch) -> None:
    """Read before any batch: every derived reading is `None`, never a fabricated zero."""
    monkeypatch.setattr(collate_mod, "collate_graph_batch", lambda *a, **kw: None)
    server = InferenceServer(
        FiniteGraphNet(), device, server_cfg(inference_batch_size=64),
        batcher=CountingGraphBatcher(wire_for(), []), encoding_spec=H.GRAPH_SPEC,
    )
    snap = server.batch_timing_snapshot()
    assert (snap["queue_wait"], snap["collate"], snap["occupancy"]) == (None, None, None)
    assert snap["batch_size"] == 64
    assert snap["max_wait_ms"] == 20.0
    assert snap["empty_polls"] == 0


def test_the_batching_block_reaches_the_sink_on_iteration_complete(
    device, monkeypatch
) -> None:
    """The manifest producer test for `inference_batching`: a LIVE graph-loop measurement
    travels pool -> hook -> builder -> sink, whole and unfabricated."""
    server = _run_graph_server(device, monkeypatch, [2, 2], wait_s=0.003)
    payload = emit_iteration(TelemetryPool(server))

    block = payload["inference_batching"]
    assert block["representation"] == "graph"
    assert block["queue_wait"]["count"] == 2
    assert block["queue_wait"]["min_ms"] >= 3.0
    assert block["occupancy"]["histogram"] == {"2": 2}
    assert block == server.batch_timing_snapshot()


def test_batch_fill_pct_reaches_the_sink_from_the_live_inference_counters(
    device, monkeypatch
) -> None:
    """The manifest producer test for `batch_fill_pct`: 2 requests per forward against 8 slots
    is 25%, and the SAME server's occupancy block agrees."""
    server = _run_graph_server(device, monkeypatch, [2, 2, 2], batch_size=8)
    payload = emit_iteration(TelemetryPool(server))

    assert payload["batch_fill_pct"] == pytest.approx(25.0)
    assert payload["inference_batching"]["occupancy"]["fill_pct_mean"] == pytest.approx(
        25.0
    )


def test_the_occupancy_histogram_leaves_the_one_bucket_under_a_batched_submit(
    device, monkeypatch
) -> None:
    """No forward serves a single graph. The counts are SCRIPTED by the fake batcher, so this
    pins what the instrument reports, not the Rust dispatch (pinned queue-side and cross-FFI).
    """
    server = _run_graph_server(device, monkeypatch, [8, 8, 7, 8, 6], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert occ["histogram"].get("1", 0) == 0, "no forward may serve a single graph"
    assert occ["max"] == 8
    assert occ["mean"] > 1.0
    # Absolute counts are the primary reading; fill_pct is knob-relative (a mean
    # occupancy of ~6 reads as ~75% at batch_size 8 and ~9.4% at 64, same physics).
    assert occ["fill_pct_mean"] == pytest.approx(occ["mean"] / 8 * 100.0, rel=1e-6)


def test_batch_fill_pct_and_the_occupancy_block_agree_on_the_same_forwards(
    device, monkeypatch
) -> None:
    """`batch_fill_pct` and `inference_batching.occupancy` come from DIFFERENT accumulators over
    the same loop, so a drift between them means one counter stopped being fed."""
    server = _run_graph_server(device, monkeypatch, [8, 4, 8], batch_size=8)
    occ = server.batch_timing_snapshot()["occupancy"]

    assert batch_fill_pct(TelemetryPool(server)) == pytest.approx(
        occ["fill_pct_mean"], rel=1e-6
    )
    assert occ["total"] == server.total_requests
    assert occ["count"] == server.forward_count


def test_a_telemetry_source_without_the_producer_publishes_none_never_zero() -> None:
    """The key is always present and carries `None`, which a consumer reads as "no producer"."""

    class _NoInstrumentPool:
        """A telemetry source with NO batching producer — no such member at all."""

        search_kind = "gumbel"
        avg_game_length = 12.0
        x_winrate = 0.5
        o_winrate = 0.4
        draw_rate = 0.1  # the third outcome share.
        draws = 1
        sims_per_sec = 100.0
        batch_fill_pct = 0.0
        recent_move_histories: list[list[tuple[int, int]]] = []

    assert not hasattr(_NoInstrumentPool(), "inference_batch_timing")
    payload = emit_iteration(_NoInstrumentPool())
    assert "inference_batching" in payload
    assert payload["inference_batching"] is None
