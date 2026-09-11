"""A4-4: the one-deep pipeline dispatches every pop's result to ITS OWN waiters, in order."""
from __future__ import annotations

import time
from typing import Any

import _fused_graph_harness as H
import numpy as np
import pytest
import torch

from mantis.selfplay.inference_server import InferenceServer


class _IdTaggedBatcher(H.ScriptedGraphBatcher):
    """Distinct request ids per pop, so a dispatch to the wrong pop is visible by id."""

    def __init__(self, pops, *, empty_polls_before: dict[int, int] | None = None) -> None:
        super().__init__(pops)
        self._served = 0
        self._empty_before = dict(empty_polls_before or {})

    def next_graph_batch(self, batch_size: int, max_wait_ms: float):
        pending = self._empty_before.get(self._served, 0)
        if pending:
            self._empty_before[self._served] = pending - 1
            return [], None
        if not self._pops:
            assert self.server is not None
            self.server._stop_event.set()
            return [], None
        payload = self._pops.pop(0)
        base = 1000 * (self._served + 1)
        self._served += 1
        return list(range(base, base + int(payload.n_graphs))), payload


def _expected(net: torch.nn.Module, payload) -> tuple[np.ndarray, np.ndarray]:
    """The un-pipelined answer for one pop: the sentinel net over the un-split collate."""
    batch = H.collate_from_payload(payload)
    from mantis.selfplay.graph_collate import segment_softmax, stone_mask_from_batch

    logits, values, _ = net.forward_batch(batch.x, batch.edge_index, batch.edge_attr,
                                          batch.legal_node_gather, stone_mask_from_batch(batch),
                                          batch.node_offsets)
    return segment_softmax(logits, batch.legal_offsets).numpy(), values.reshape(-1).numpy()


def _drive(monkeypatch, pops, *, net=None, **batcher_kw) -> tuple[InferenceServer, _IdTaggedBatcher, Any]:
    import mantis.selfplay.graph_collate as collate_mod

    monkeypatch.setattr(collate_mod, "collate_graph_batch", H.collate_from_payload)
    model = net if net is not None else H.SentinelGraphNet()
    batcher = _IdTaggedBatcher(pops, **batcher_kw)
    server = InferenceServer(model, torch.device("cpu"), H.graph_cfg(), batcher=batcher,
                             encoding_spec=H.GRAPH_SPEC)
    batcher.server = server
    server.run()
    return server, batcher, model


@pytest.mark.parametrize("shapes", [
    [[2, 5, 3], [7, 1], [4, 4, 4, 4]],
    [[3], [6, 2], [1, 1, 1], [9]],
])
def test_every_pop_is_dispatched_to_its_own_ids_with_its_own_answer(monkeypatch, shapes) -> None:
    pops = [H.build_payload(legal, uid_base=100 * (i + 1)) for i, legal in enumerate(shapes)]
    reference = H.SentinelGraphNet()
    server, batcher, _net = _drive(monkeypatch, list(pops))
    assert batcher.failures == []
    assert [ids for ids, *_ in batcher.results] == [
        list(range(1000 * (i + 1), 1000 * (i + 1) + len(legal))) for i, legal in enumerate(shapes)]
    for (ids, probs, offsets, values), payload in zip(batcher.results, pops, strict=True):
        want_probs, want_values = _expected(reference, payload)
        assert np.array_equal(offsets, np.asarray(payload.legal_offsets))
        assert np.array_equal(probs, want_probs), f"probs of pop {ids[0] // 1000} moved"
        assert np.array_equal(values, want_values), f"values of pop {ids[0] // 1000} moved"
    assert server.forward_count == len(shapes)
    assert server.batch_timing_snapshot()["pipeline"]["gpu_wait"]["count"] == len(shapes)


def test_a_forward_failure_lands_on_the_failing_pop_and_its_neighbours_are_served(monkeypatch) -> None:
    pops = [H.build_payload([2, 3], uid_base=100), H.build_payload([4], uid_base=200),
            H.build_payload([1, 1, 1], uid_base=300)]
    server, batcher, _net = _drive(monkeypatch, pops, net=H.SentinelGraphNet(oom_on_call=2))
    assert [ids for ids, *_ in batcher.results] == [[1000, 1001], [3000, 3001, 3002]]
    assert [ids for ids, _msg in batcher.failures] == [[2000]]
    assert "out of memory" in batcher.failures[0][1]


class _IdleAfterOneBatcher(_IdTaggedBatcher):
    """Hands out ONE pop, then blocks as an idle queue would until that pop has been dispatched."""

    def next_graph_batch(self, batch_size: int, max_wait_ms: float):
        if self._served == 1:
            deadline = time.monotonic() + 10.0
            while not self.results and time.monotonic() < deadline:
                time.sleep(0.002)
            self.blocked_until_dispatch = bool(self.results)
            assert self.server is not None
            self.server._stop_event.set()
            return [], None
        return super().next_graph_batch(batch_size, max_wait_ms)


def test_a_pop_is_dispatched_while_the_server_thread_still_waits_for_the_next_one(monkeypatch) -> None:
    import mantis.selfplay.graph_collate as collate_mod

    monkeypatch.setattr(collate_mod, "collate_graph_batch", H.collate_from_payload)
    heartbeats: list[str] = []
    batcher = _IdleAfterOneBatcher([H.build_payload([2, 3], uid_base=100)])
    server = InferenceServer(H.SentinelGraphNet(), torch.device("cpu"), H.graph_cfg(),
                             batcher=batcher, encoding_spec=H.GRAPH_SPEC,
                             heartbeat=heartbeats.append)
    batcher.server = server
    server.run()
    assert batcher.blocked_until_dispatch, (
        "the pop's results never arrived while the server thread was blocked in the next pop: "
        "dispatch must not depend on another pop or on the pop deadline")
    assert [ids for ids, *_ in batcher.results] == [[1000, 1001]] and heartbeats == ["inference_dispatch"]
    assert server.batch_timing_snapshot()["pipeline"]["depth"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LOUD SKIP — the CUDA pipeline path (event, pinned D2H) needs a GPU")
def test_every_pop_is_dispatched_to_its_own_ids_on_the_cuda_path(monkeypatch) -> None:
    """The CPU drives record no event and no pinned D2H; this one does (red team 13)."""
    import mantis.selfplay.graph_collate as collate_mod

    shapes = [[2, 5, 3], [7, 1], [4, 4, 4, 4], [3, 3]]
    pops = [H.build_payload(legal, uid_base=100 * (i + 1)) for i, legal in enumerate(shapes)]
    reference = H.SentinelGraphNet()

    def collate_on_cuda(wire, *_a, **_kw):
        batch = H.collate_from_payload(wire)
        for name in ("x", "edge_index", "edge_attr", "legal_offsets", "legal_node_gather",
                     "node_offsets", "n_stones"):
            setattr(batch, name, getattr(batch, name).cuda())
        batch.device = "cuda"
        return batch

    monkeypatch.setattr(collate_mod, "collate_graph_batch", collate_on_cuda)
    batcher = _IdTaggedBatcher(list(pops))
    server = InferenceServer(H.SentinelGraphNet().cuda(), torch.device("cuda"), H.graph_cfg(),
                             batcher=batcher, encoding_spec=H.GRAPH_SPEC)
    batcher.server = server
    server.run()
    assert batcher.failures == []
    assert [ids for ids, *_ in batcher.results] == [
        list(range(1000 * (i + 1), 1000 * (i + 1) + len(legal))) for i, legal in enumerate(shapes)]
    for (ids, probs, offsets, values), payload in zip(batcher.results, pops, strict=True):
        want_probs, want_values = _expected(reference, payload)
        assert np.array_equal(offsets, np.asarray(payload.legal_offsets))
        assert np.allclose(probs, want_probs, atol=1e-6), f"probs of pop {ids[0] // 1000} moved"
        assert np.allclose(values, want_values, atol=1e-6), f"values of pop {ids[0] // 1000} moved"
    block = server.batch_timing_snapshot()["pipeline"]
    assert block["gpu_wait"]["count"] == len(shapes) and block["launch"]["count"] == len(shapes)
