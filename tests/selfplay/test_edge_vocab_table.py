"""The served forward reads each layer's projection of the edge vocabulary by code: the collate ships codes, the net serves what the per-edge forward serves."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _wire_geometry import geometry_kwargs

from mantis._engine import InferenceBatcher, edge_vocabulary
from mantis.config.census import production_configs
from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import arch_from_spec_and_config, build_net
from mantis.selfplay.graph_collate import (
    EdgeAttrGeometryMismatch,
    GraphWirePayload,
    collate_graph_batch,
    graph_wire_from_rust,
    stone_mask_from_batch,
)

GEOMETRY: dict[str, int] = geometry_kwargs()
_REPO = Path(__file__).resolve().parents[2]


def _collate(fields: dict[str, Any], device: str = "cpu", **kw: Any):
    return collate_graph_batch(GraphWirePayload(**fields), device=device, semantic="off", **{**GEOMETRY, **kw})


def _vocab(device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(np.asarray(edge_vocabulary(GEOMETRY["win_length"]))).reshape(-1, 5).to(device)


def _net(device: str = "cpu") -> torch.nn.Module:
    config = load_config(production_configs(_REPO)[0]).model_dump()
    torch.manual_seed(20261003)
    return build_net(arch_from_spec_and_config(lookup(config["identity"]["encoding"]), config)).to(device).eval()


def _serve(net: torch.nn.Module, batch: Any, edges: torch.Tensor, vocab: torch.Tensor | None, amp: bool):
    with torch.inference_mode(), torch.autocast(edges.device.type, dtype=torch.bfloat16, enabled=amp):
        return net.forward_batch(batch.x, batch.edge_index, edges, batch.legal_node_gather,
                                 stone_mask_from_batch(batch), batch.node_offsets, edge_vocab=vocab)


def test_the_vocabulary_is_the_win_lengths_rows() -> None:
    assert _vocab().shape == (91, 5) and GEOMETRY["win_length"] == 6


@pytest.mark.parametrize("name", ["b6", "b1"])
def test_a_coded_collate_ships_each_edges_vocabulary_code_and_nothing_else_changes(payload_fields, name) -> None:
    plain = _collate(payload_fields(name))
    coded = _collate(payload_fields(name), coded_edges=True)
    assert coded.edge_attr is None and coded.edge_code.dtype == torch.uint8
    assert torch.equal(_vocab()[coded.edge_code.long()], plain.edge_attr)
    for f in ("x", "edge_index", "legal_offsets", "legal_node_gather", "node_offsets", "n_stones"):
        assert torch.equal(getattr(coded, f), getattr(plain, f)), f


def test_a_coded_collate_refuses_a_row_outside_the_vocabulary_whatever_the_semantic_mode(payload_fields) -> None:
    fields = payload_fields("b6")
    fields["edge_attr"][3] = 2.5
    with pytest.raises(EdgeAttrGeometryMismatch):
        _collate(fields, coded_edges=True)


@pytest.mark.parametrize("name", ["b6", "b1"])
def test_a_coded_forward_serves_the_per_edge_forward(payload_fields, name) -> None:
    net = _net()
    plain = _collate(payload_fields(name))
    coded = _collate(payload_fields(name), coded_edges=True)
    want = _serve(net, plain, plain.edge_attr, None, amp=False)
    got = _serve(net, coded, coded.edge_code, _vocab(), amp=False)
    for w, g in zip(want, got, strict=True):
        assert torch.equal(g, w)


def test_a_coded_forward_refuses_to_train(payload_fields) -> None:
    net = _net().train()
    coded = _collate(payload_fields("b6"), coded_edges=True)
    with pytest.raises(ValueError, match="coded"):
        net.forward_batch(coded.x, coded.edge_index, coded.edge_code, coded.legal_node_gather,
                          stone_mask_from_batch(coded), coded.node_offsets, edge_vocab=_vocab())


def _small_payload() -> GraphWirePayload:
    """An empty board's wire, fused by the real batcher: its per-edge GEMM runs far below the table's 1024 rows."""
    spec = lookup("gnn_axis_v1")
    batcher = InferenceBatcher(encoding_spec=spec)
    submit = threading.Thread(target=batcher.submit_graphs_and_wait, args=([([], 1, 1)], 1))
    submit.start()
    ids, wire = batcher.next_graph_batch(1, 100)
    while not ids:  # an empty pop means the submit has not queued yet
        ids, wire = batcher.next_graph_batch(1, 100)
    payload = graph_wire_from_rust(wire)
    counts = np.diff(np.asarray(payload.legal_offsets))
    batcher.submit_graph_inference_results(ids, np.repeat(1.0 / counts, counts).astype(np.float32),
                                           np.asarray(payload.legal_offsets), np.zeros(len(ids), np.float32))
    submit.join()
    return payload


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA: the bf16 served forward and its kernel")
@pytest.mark.parametrize("name", ["b6", "b1", "empty_board"])
def test_a_coded_cuda_forward_is_the_per_edge_forward_bit_for_bit(payload_fields, name) -> None:
    """The table's GEMMs and the per-edge ones round alike on the gating card: the served bits, not only its decisions."""
    net = _net("cuda")
    payload = _small_payload() if name == "empty_board" else GraphWirePayload(**payload_fields(name))
    plain = collate_graph_batch(payload, device="cuda", semantic="off", **GEOMETRY)
    coded = collate_graph_batch(payload, device="cuda", semantic="off", coded_edges=True, **GEOMETRY)
    want = _serve(net, plain, plain.edge_attr, None, amp=True)
    got = _serve(net, coded, coded.edge_code, _vocab("cuda"), amp=True)
    for w, g in zip(want, got, strict=True):
        assert torch.equal(g, w), f"max |d| {float((g - w).abs().max()):.4g}"


def _sizes(batch) -> tuple[int, int, int, int]:
    return (int(batch.n_graphs), int(batch.x.shape[0]), int(batch.edge_index.shape[1]), int(batch.legal_node_gather.shape[0]))


@pytest.mark.parametrize("name", ["b6", "b1"])
def test_a_padded_coded_collate_holds_the_real_part_then_one_padding_graph(payload_fields, name):
    real = _collate(payload_fields(name), coded_edges=True)
    b, n, e, lg = _sizes(real)
    pad = (b + 2, n + 9, e + 31, lg + 5)
    padded = _collate(payload_fields(name), coded_edges=True, pad_to=pad)
    assert padded.n_graphs == b + 2 and padded.x.shape == (n + 9, 11) and padded.edge_index.shape == (2, e + 31)
    assert torch.equal(padded.x[:n], real.x) and not padded.x[n:].any()
    assert torch.equal(padded.edge_index[:, :e], real.edge_index)
    assert torch.equal(padded.edge_code[:e], real.edge_code) and not padded.edge_code[e:].any()
    assert torch.equal(padded.legal_node_gather[:lg], real.legal_node_gather)
    assert padded.node_offsets.tolist() == real.node_offsets.tolist() + [n, n + 9]
    assert padded.legal_offsets.tolist() == real.legal_offsets.tolist() + [lg, lg + 5]


@pytest.mark.parametrize("name", ["b6", "b1"])
def test_a_padded_forward_serves_the_real_rows_as_the_unpadded_forward(payload_fields, name):
    net = _net()
    real = _collate(payload_fields(name), coded_edges=True)
    b, n, e, lg = _sizes(real)
    padded = _collate(payload_fields(name), coded_edges=True, pad_to=(b + 1, n + 64, e + 1000, lg + 40))
    want = _serve(net, real, real.edge_code, _vocab(), amp=False)
    got = _serve(net, padded, padded.edge_code, _vocab(), amp=False)
    torch.testing.assert_close(got[0][:lg], want[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(got[1][:b], want[1], rtol=0, atol=1e-6)


def test_a_collate_into_given_buffers_fills_them_and_returns_them(payload_fields):
    real = _collate(payload_fields("b6"), coded_edges=True)
    b, n, e, lg = _sizes(real)
    pad = (b + 1, n + 8, e + 8, lg + 8)
    buffers = [torch.full(((n + 8) * 11,), 7.0), torch.full((2 * (e + 8),), 7, dtype=torch.int64),
               torch.full((e + 8,), 7, dtype=torch.uint8), torch.full((b + 2,), 7, dtype=torch.int64),
               torch.full((lg + 8,), 7, dtype=torch.int64), torch.full((b + 2,), 7, dtype=torch.int64),
               torch.full((b + 1,), 7, dtype=torch.int64)]
    padded = _collate(payload_fields("b6"), coded_edges=True, pad_to=pad, device_out=buffers)
    assert padded.x.data_ptr() == buffers[0].data_ptr() and padded.edge_code.data_ptr() == buffers[2].data_ptr()
    assert torch.equal(padded.x[:n], real.x)
