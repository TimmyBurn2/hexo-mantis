"""The served forward of one fused part, and its replay from one CUDA graph per padded shape bucket."""
from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.selfplay.graph_collate import GraphBatch, segment_softmax, stone_mask_from_batch

#: Each bucket holds this much more than the one below it, so a part pads by at most this factor.
_RATIO = 1.25
#: The smallest bucket's nodes: a smaller part pads up to it, which costs less than a capture of its own.
_SMALLEST_NODES = 4096
#: Eager runs before a capture, so compilation, autotuning and library handles happen outside the graph.
_WARMUP = 2
#: A padding node's self-loops are one serial walk in the fused kernel: a bucket gives each at most this many.
_PAD_EDGES_PER_SINK = 256
#: One capture in the process at a time (torch's rule): two servers meeting new buckets together would collide.
_CAPTURE_LOCK = threading.Lock()


class ServedGraphCaptureError(RuntimeError):
    """A bucket's forward could not be captured; the forward refuses every later part (run-fatal, never retried)."""


@dataclass(frozen=True)
class Bucket:
    """One padded shape: every part it holds collates to these sizes, its last graph the padding graph."""

    n_graphs: int
    n_nodes: int
    n_edges: int
    n_legal: int

    @staticmethod
    def of(ladder: tuple[Bucket, ...], n_graphs: int, n_nodes: int, n_edges: int) -> Bucket | None:
        """The smallest bucket holding the part with a padding graph and padding nodes enough for its padding edges, or None."""
        return next((b for b in ladder if n_graphs < b.n_graphs and n_nodes < b.n_nodes and n_edges <= b.n_edges
                     and b.n_edges - n_edges <= (b.n_nodes - n_nodes) * _PAD_EDGES_PER_SINK), None)


def bucket_ladder(caps: FusedGraphCapsSpec, *, batch_size: int) -> tuple[Bucket, ...]:
    """Smallest first: from the fused caps down by `_RATIO` while at least `_SMALLEST_NODES` nodes; legal slots = nodes."""
    rungs: list[Bucket] = []
    while True:
        scale = _RATIO ** len(rungs)
        nodes = math.ceil(caps.max_fused_nodes / scale)
        if rungs and nodes < _SMALLEST_NODES:
            return tuple(reversed(rungs))
        rungs.append(Bucket(batch_size + 1, nodes + 1, math.ceil(caps.max_fused_edges / scale), nodes + 1))


def served_outputs(model: Any, batch: GraphBatch, edge_vocab: Tensor, *, trunk: Any,
                   amp_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """`(probs, values)` of one coded part: bf16 autocast without its weight-cast cache, which a replay would outlive; fp32 softmax. Raises: ValueError — under grad (the coded path has no backward)."""
    stone_mask = stone_mask_from_batch(batch)
    device_type = batch.x.device.type
    kwargs = {} if trunk is None else {"trunk": trunk}
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=device_type == "cuda", cache_enabled=False):
        logits, value, _bins = model.forward_batch(batch.x, batch.edge_index, batch.edge_code, batch.legal_node_gather,
                                                   stone_mask, batch.node_offsets, edge_vocab=edge_vocab, **kwargs)
    # Segment-softmax in float32 corrects reduced-precision drift and is segment-LOCAL, so a part's is the pop's.
    return segment_softmax(logits.float(), batch.legal_offsets), value.detach().float().reshape(-1)


class BucketedForward:
    """Per bucket: the flat inputs the collate fills, one captured graph and its outputs, all in one pool; a padded batch breaks the wire's every-graph-has-a-legal-node and ascending-gather invariants and its empty graphs' values are NaN, so only the part's own rows are ever read."""

    def __init__(self, serve: Callable[[GraphBatch], tuple[Tensor, Tensor]], device: torch.device, *,
                 node_feat_dim: int) -> None:
        self._serve = serve
        self._device = device
        self._node_feat_dim = node_feat_dim
        self._inputs: dict[Bucket, list[Tensor]] = {}
        self._graphs: dict[Bucket, tuple[torch.cuda.CUDAGraph, tuple[Tensor, Tensor]]] = {}
        self._pool: Any = None
        self._capture_stream = torch.cuda.Stream(device)
        self._failed: ServedGraphCaptureError | None = None
        self.captured = 0

    def inputs(self, bucket: Bucket) -> list[Tensor]:
        """The bucket's flat buffers in the coded collate's order: x, edge_index, codes, legal_offsets, gather, node_offsets, n_stones."""
        if bucket not in self._inputs:
            b, dev = bucket, self._device
            self._inputs[bucket] = [
                torch.empty(b.n_nodes * self._node_feat_dim, dtype=torch.float32, device=dev),
                torch.empty(2 * b.n_edges, dtype=torch.int64, device=dev),
                torch.empty(b.n_edges, dtype=torch.uint8, device=dev),
                torch.empty(b.n_graphs + 1, dtype=torch.int64, device=dev),
                torch.empty(b.n_legal, dtype=torch.int64, device=dev),
                torch.empty(b.n_graphs + 1, dtype=torch.int64, device=dev),
                torch.empty(b.n_graphs, dtype=torch.int64, device=dev),
            ]
        return self._inputs[bucket]

    def holds(self, bucket: Bucket, batch: GraphBatch) -> bool:
        """Whether `batch`'s seven tensors are the bucket's own inputs, which is all a replay reads."""
        mine = (batch.x, batch.edge_index, batch.edge_code, batch.legal_offsets, batch.legal_node_gather,
                batch.node_offsets, batch.n_stones)
        return all(t.data_ptr() == b.data_ptr() for t, b in zip(mine, self.inputs(bucket), strict=True))

    def run(self, bucket: Bucket, batch: GraphBatch) -> tuple[Tensor, Tensor]:
        """The bucket's padded outputs for the part just collated into its inputs, valid until the next replay of any bucket (one pool). Raises: ServedGraphCaptureError — this or an earlier capture of this forward failed."""
        if self._failed is not None:
            raise ServedGraphCaptureError(str(self._failed)) from self._failed
        if bucket not in self._graphs:
            try:
                self._graphs[bucket] = self._capture(batch)
            except Exception as exc:
                self._failed = ServedGraphCaptureError(f"capturing {bucket} failed: {exc!r}")
                raise self._failed from exc
            self.captured += 1
        graph, outputs = self._graphs[bucket]
        graph.replay()
        return outputs

    def _capture(self, batch: GraphBatch) -> tuple[torch.cuda.CUDAGraph, tuple[Tensor, Tensor]]:
        """Warm up on the serving stream, release the cache (the capturing pop's one device wait), then capture on this forward's own stream."""
        current = torch.cuda.current_stream(self._device)
        stream = self._capture_stream
        with _CAPTURE_LOCK:
            for _ in range(_WARMUP):
                self._serve(batch)
            # A capture cannot free cached blocks, so a cache filling the card (the trainer's) would fail it: free it first.
            torch.cuda.empty_cache()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                # Thread-local: the retire thread's event waits and a trainer's launches stay legal during the capture.
                graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
                try:
                    outputs = self._serve(batch)
                except BaseException:
                    with contextlib.suppress(Exception):
                        graph.capture_end()
                    raise
                graph.capture_end()
            current.wait_stream(stream)
        self._pool = graph.pool()
        return graph, outputs


__all__ = ["Bucket", "BucketedForward", "ServedGraphCaptureError", "bucket_ladder", "served_outputs"]
