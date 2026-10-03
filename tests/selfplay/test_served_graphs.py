"""The served forward's shape buckets: a ladder over the fused caps, each part in the smallest bucket that holds it, and on CUDA one replayed graph per bucket that serves what the padded eager forward serves."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.selfplay.graph_collate import collate_graph_batch, GraphBatch, GraphWirePayload
from mantis.selfplay.graph_wire_split import plan_fused_forwards, slice_graph_wire
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.served_graphs import Bucket, bucket_ladder, BucketedForward, served_outputs
import mantis.selfplay.served_graphs as served_graphs

_CAPS = FusedGraphCapsSpec(max_fused_edges=1_373_143, max_fused_nodes=56_645)


def test_the_ladder_tops_out_at_the_caps_and_rises_geometrically():
    ladder = bucket_ladder(_CAPS, batch_size=64)
    top = ladder[-1]
    assert (top.n_nodes, top.n_edges, top.n_graphs) == (56_646, 1_373_143, 65)
    assert all(a.n_nodes < b.n_nodes and a.n_edges < b.n_edges for a, b in zip(ladder, ladder[1:]))
    assert all(b.n_legal == b.n_nodes for b in ladder)
    assert 1.15 < ladder[-1].n_nodes / ladder[-2].n_nodes < 1.35


def test_a_part_lands_in_the_smallest_bucket_that_holds_it_with_a_padding_node_to_spare():
    ladder = bucket_ladder(_CAPS, batch_size=64)
    small = ladder[0]
    assert Bucket.of(ladder, 64, small.n_nodes - 1, small.n_edges) == small
    assert Bucket.of(ladder, 64, small.n_nodes, small.n_edges) == ladder[1], "no padding node left"
    assert Bucket.of(ladder, 64, 10, small.n_edges + 1) == ladder[1]
    assert Bucket.of(ladder, 64, 3, 2) == small, "a tiny part pads to the smallest bucket"
    assert Bucket.of(ladder, 64, 56_645, 1_373_143) == ladder[-1], "a part at the caps fits the top"
    assert Bucket.of(ladder, 65, 10, 10) is None, "more graphs than the batch never fits"
    assert Bucket.of(ladder, 64, small.n_nodes - 1, 10) == ladder[1], "one padding node for 100k self-loops stalls a row"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_a_replayed_bucket_serves_the_padded_eager_forward_bit_for_bit_and_follows_a_weight_swap(
        payload_fields, compiled: bool) -> None:
    from test_edge_vocab_table import _collate, _net, _sizes, _vocab


    net = _net("cuda")
    vocab = _vocab("cuda")
    real = _collate(payload_fields("b6"), "cuda", coded_edges=True)
    b, n, e, lg = _sizes(real)
    ladder = bucket_ladder(FusedGraphCapsSpec(max_fused_edges=2 * e, max_fused_nodes=2 * n), batch_size=8)
    bucket = Bucket.of(ladder, *_sizes(real)[:3])
    assert bucket is not None
    pad = (bucket.n_graphs, bucket.n_nodes, bucket.n_edges, bucket.n_legal)

    torch._dynamo.reset()
    trunk = torch.compile(net.representation, dynamic=True) if compiled else None

    def serve(batch):
        return served_outputs(net, batch, vocab, trunk=trunk, amp_dtype=torch.bfloat16)

    graphs = BucketedForward(serve, torch.device("cuda"), node_feat_dim=11)
    for _ in range(2):
        batch = _collate(payload_fields("b6"), "cuda", coded_edges=True, pad_to=pad, device_out=graphs.inputs(bucket))
        with torch.inference_mode():
            got = [t.clone() for t in graphs.run(bucket, batch)]
            want = serve(_collate(payload_fields("b6"), "cuda", coded_edges=True, pad_to=pad))
        # The empty padding graphs' values are NaN and never served: the part's own rows are compared.
        assert torch.equal(got[0][:lg], want[0][:lg]) and torch.equal(got[1][:b], want[1][:b])
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.5)
    batch = _collate(payload_fields("b6"), "cuda", coded_edges=True, pad_to=pad, device_out=graphs.inputs(bucket))
    with torch.inference_mode():
        swapped = [t.clone() for t in graphs.run(bucket, batch)]
        fresh = serve(_collate(payload_fields("b6"), "cuda", coded_edges=True, pad_to=pad))
    assert torch.equal(swapped[0][:lg], fresh[0][:lg]) and torch.equal(swapped[1][:b], fresh[1][:b])
    assert graphs.captured == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
def test_the_server_replays_every_part_a_bucket_holds_and_counts_it(payload_fields):
    import _fused_graph_harness as H
    from test_edge_vocab_table import _net


    pops = [GraphWirePayload(**payload_fields("b6")) for _ in range(3)]
    batcher = H.ScriptedGraphBatcher(pops)
    server = InferenceServer(_net("cuda"), torch.device("cuda"), H.graph_cfg(), batcher=batcher,
                             encoding_spec=lookup("gnn_axis_v1"), collate_check_period=1)
    batcher.server = server
    server.run()
    assert batcher.failures == [] and len(batcher.results) == 3
    block = server.batch_timing_snapshot()["served_graphs"]
    assert block["enabled"] and block["captured"] == 1 and block["replayed_parts"] == 3 and block["eager_parts"] == 0


def _padded_eager(net: torch.nn.Module, payload, bucket: Bucket) -> tuple[np.ndarray, np.ndarray]:
    """One part's own padded eager forward, sliced to its rows: what its replay must hand back."""
    from test_edge_vocab_table import GEOMETRY, _vocab


    batch = collate_graph_batch(payload, device="cuda", semantic="off", coded_edges=True,
                                pad_to=(bucket.n_graphs, bucket.n_nodes, bucket.n_edges, bucket.n_legal), **GEOMETRY)
    with torch.inference_mode():
        probs, values = served_outputs(net, batch, _vocab("cuda"), trunk=None, amp_dtype=torch.bfloat16)
    lg, b = int(np.asarray(payload.legal_offsets)[-1]), int(payload.n_graphs)
    return probs[:lg].cpu().numpy(), values[:b].cpu().numpy()


def _serve_pops(net: torch.nn.Module, pops: list, caps: FusedGraphCapsSpec):
    import _fused_graph_harness as H


    batcher = H.ScriptedGraphBatcher(pops)
    server = InferenceServer(net, torch.device("cuda"), H.graph_cfg(batch_size=8), batcher=batcher,
                             encoding_spec=lookup("gnn_axis_v1"), fused_graph_caps=caps, collate_check_period=1)
    batcher.server = server
    server.run()
    assert batcher.failures == [], batcher.failures
    return server, batcher.results


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
def test_distinct_pops_replayed_from_one_bucket_each_receive_their_own_outputs(payload_fields) -> None:
    """Three different pops through one captured graph: a stale input or an overwritten output hands one pop another's numbers."""
    from test_edge_vocab_table import _net, _small_payload


    net = _net("cuda")
    pops = [GraphWirePayload(**payload_fields("b6")), _small_payload(), GraphWirePayload(**payload_fields("b1"))]
    n = max(int(np.asarray(p.node_offsets)[-1]) for p in pops)
    e = max(int(np.asarray(p.edge_offsets)[-1]) for p in pops)
    caps = FusedGraphCapsSpec(max_fused_edges=2 * e, max_fused_nodes=2 * n)
    server, results = _serve_pops(net, pops, caps)
    assert len(server._ladder) == 1, "the caps were chosen for a one-bucket ladder"
    bucket = server._ladder[0]
    block = server.batch_timing_snapshot()["served_graphs"]
    assert block["captured"] == 1 and block["replayed_parts"] == 3 and block["eager_parts"] == 0
    for (_ids, probs, _offsets, values), payload in zip(results, pops, strict=True):
        want_probs, want_values = _padded_eager(net, payload, bucket)
        assert np.array_equal(probs, want_probs) and np.array_equal(values, want_values)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
def test_a_pop_split_into_two_parts_of_one_bucket_receives_each_parts_own_outputs(payload_fields) -> None:
    """The second part's replay overwrites the bucket's outputs while the first part's copy is queued: stream order must keep both."""
    from test_edge_vocab_table import _net


    net = _net("cuda")
    payload = GraphWirePayload(**payload_fields("b6"))
    n, e = int(np.asarray(payload.node_offsets)[-1]), int(np.asarray(payload.edge_offsets)[-1])
    caps = FusedGraphCapsSpec(max_fused_edges=e, max_fused_nodes=int(n * 0.6))
    plan = plan_fused_forwards(payload.edge_offsets, payload.node_offsets, caps)
    assert len(plan) == 2, plan
    server, results = _serve_pops(net, [payload], caps)
    assert len(server._ladder) == 1, "the caps were chosen for a one-bucket ladder"
    bucket = server._ladder[0]
    assert server.batch_timing_snapshot()["served_graphs"]["replayed_parts"] == 2
    parts = [_padded_eager(net, slice_graph_wire(payload, g0, g1), bucket) for g0, g1 in plan]
    ((_ids, probs, _offsets, values),) = results
    assert np.array_equal(probs, np.concatenate([p for p, _v in parts]))
    assert np.array_equal(values, np.concatenate([v for _p, v in parts]))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
def test_two_buckets_sharing_one_pool_replay_each_their_own_outputs_interleaved(payload_fields) -> None:
    """Bucket B's capture and replays reuse the pool bucket A's intermediates live in: A's later replays must still serve A's numbers."""
    from test_edge_vocab_table import _collate, _net, _sizes, _vocab


    net, vocab = _net("cuda"), _vocab("cuda")
    real = _collate(payload_fields("b6"), "cuda", coded_edges=True)
    _b, n, e, _lg = _sizes(real)
    small, large = Bucket(9, n + 400, e + 9000, n + 400), Bucket(9, 2 * n, 2 * e, 2 * n)
    graphs = BucketedForward(lambda b: served_outputs(net, b, vocab, trunk=None, amp_dtype=torch.bfloat16),
                             torch.device("cuda"), node_feat_dim=11)
    for name, bucket in (("b6", small), ("b1", large), ("b6", small), ("b1", small), ("b6", large)):
        payload = GraphWirePayload(**payload_fields(name))
        pad = (bucket.n_graphs, bucket.n_nodes, bucket.n_edges, bucket.n_legal)
        staged = _collate(payload_fields(name), "cuda", coded_edges=True, pad_to=pad, device_out=graphs.inputs(bucket))
        with torch.inference_mode():
            probs, values = graphs.run(bucket, staged)
            got = (probs.cpu().numpy(), values.cpu().numpy())
        want = _padded_eager(net, payload, bucket)
        lg, b = len(want[0]), len(want[1])
        assert np.array_equal(got[0][:lg], want[0]) and np.array_equal(got[1][:b], want[1]), (name, bucket)
    assert graphs.captured == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: graph capture and replay")
def test_a_capture_finds_room_that_another_streams_cache_was_holding(
        payload_fields: Callable[[str], dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture cannot free cached blocks, so a co-resident trainer's cache filling the card must not fail the capture."""
    from test_edge_vocab_table import _collate, _net, _sizes, _vocab


    # While no graph is alive the RNG allocates outside the capture, whose OOM retry frees the cache by luck: keep one.
    first = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first):
        torch.ones(1, device="cuda").add_(1)
    net, vocab = _net("cuda"), _vocab("cuda")
    _b, n, e, _lg = _sizes(_collate(payload_fields("b6"), "cuda", coded_edges=True))
    bucket = Bucket(9, n + 400, e + 9000, n + 400)
    pad = (bucket.n_graphs, bucket.n_nodes, bucket.n_edges, bucket.n_legal)

    def serve(batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return served_outputs(net, batch, vocab, trunk=None, amp_dtype=torch.bfloat16)

    graphs = BucketedForward(serve, torch.device("cuda"), node_feat_dim=11)
    staged = _collate(payload_fields("b6"), "cuda", coded_edges=True, pad_to=pad, device_out=graphs.inputs(bucket))
    with torch.inference_mode():
        serve(staged)
    # The warm-up ran above: one inside the capture could hit the cap and free the cache before the capture by luck.
    monkeypatch.setattr(served_graphs, "_WARMUP", 0)
    with torch.cuda.stream(torch.cuda.Stream()):
        hog = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    del hog
    torch.cuda.synchronize()
    total, prior = torch.cuda.get_device_properties(0).total_memory, torch.cuda.get_per_process_memory_fraction()
    torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + (1 << 20)) / total)
    try:
        with torch.inference_mode():
            probs, values = graphs.run(bucket, staged)
            got = (probs.cpu().numpy(), values.cpu().numpy())
    finally:
        torch.cuda.set_per_process_memory_fraction(prior)
    want = _padded_eager(net, GraphWirePayload(**payload_fields("b6")), bucket)
    assert np.array_equal(got[0][:len(want[0])], want[0]) and np.array_equal(got[1][:len(want[1])], want[1])
