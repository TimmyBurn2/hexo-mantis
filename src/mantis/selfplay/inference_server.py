"""THE batched inference server — the one dispatch loop in the tree.

>300 justify: the ONE inference loop. Rust owns request concurrency; this module is the whole
Python side of the dispatch seam, and splitting the dense and graph loops apart would create a
second place a batch can be prepared, submitted or failed.

The representation is resolved ONCE at construction from the encoding spec, a closed match with no
dense-by-default arm; autocast is bf16 UNCONDITIONALLY on the graph loop (LAW-06).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from mantis._engine import InferenceBatcher
from mantis.config.resolve.fused_graph_caps import (
    FusedGraphCapsSpec,
    resolve_fused_graph_caps,
)
from mantis.encoding import EncodingSpec as RegistrySpec
from mantis.encoding import resolve_from_config
from mantis.model import amp_dtype_for
from mantis.selfplay.hparams import InferenceHParams, is_graph_representation
from mantis.selfplay.pool_hooks import EventSink

_LOG = logging.getLogger(__name__)

# Emitted once per dispatched batch when a sink is injected. The named alias `HeartbeatFn` lives
# in `pool_hooks`; the server needs only the structural type and must not import the pool for it.
_HEARTBEAT_SOURCE = "inference_dispatch"


def _timing_agg(
    count: int, total_s: float, min_s: float | None, max_s: float | None
) -> dict[str, Any] | None:
    """One timing accumulator as an event sub-block, or `None` when nothing was measured — the
    unproduced-field convention. The raw `count`/`total_ms` travel beside the derived `mean_ms` so
    a consumer can difference two events and recover an INTERVAL mean."""
    if count == 0:
        return None
    return {
        "count": count,
        "total_ms": round(total_s * 1e3, 6),
        "mean_ms": round((total_s / count) * 1e3, 6),
        "min_ms": None if min_s is None else round(min_s * 1e3, 6),
        "max_ms": None if max_s is None else round(max_s * 1e3, 6),
    }


def _pow2_bucket(n: int) -> int:
    """The power-of-two LOWER bound of `n`'s histogram bucket, extracted so the occupancy and
    fused-part histograms cannot drift onto two transcriptions of one bucketing rule."""
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def _size_agg(
    count: int, total: int, min_n: int | None, max_n: int | None, hist: dict[int, int]
) -> dict[str, Any] | None:
    """One per-part size distribution (fused nodes or fused edges), or `None` at zero samples. A
    DISTRIBUTION and not a mean, because for a memory bound the TAIL is the question: a mean
    fused-E of 400 k with a max of 9 M is a run that OOMs while agreeing on the mean."""
    if count == 0:
        return None
    return {
        "count": count,
        "total": total,
        "mean": round(total / count, 6),
        "min": min_n,
        "max": max_n,
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }


def _fusion_bound_hits(
    plan: tuple[tuple[int, int], ...],
    edge_counts: np.ndarray,
    node_counts: np.ndarray,
    caps: FusedGraphCapsSpec,
) -> tuple[int, int]:
    """`(edge-forced cuts, node-forced cuts)` for one plan — the ATTRIBUTION half of the bound,
    because re-fitting the wrong cap moves no peak. Edges are tested first, matching the planner's
    own order."""
    edges = nodes = 0
    for (g0, g1), (next_g0, _next_g1) in zip(plan, plan[1:], strict=False):
        acc_e = int(edge_counts[g0:g1].sum())
        acc_n = int(node_counts[g0:g1].sum())
        if acc_e + int(edge_counts[next_g0]) > caps.max_fused_edges:
            edges += 1
        elif acc_n + int(node_counts[next_g0]) > caps.max_fused_nodes:
            nodes += 1
        else:  # pragma: no cover — a cut the planner could not have made
            raise RuntimeError(
                f"InferenceServer: plan cut at graph {next_g0} breaches neither member of "
                f"inference.fused_graph_caps — the instrument and the planner disagree about "
                f"the same partition, and a counter that cannot be attributed is worse than "
                f"no counter (plan={plan})"
            )
    return edges, nodes


def _occupancy_agg(
    count: int,
    total: int,
    min_n: int | None,
    max_n: int | None,
    hist: dict[int, int],
    batch_size: int,
) -> dict[str, Any] | None:
    """The served-batch occupancy distribution, or `None` when nothing was measured: a mean ratio
    cannot distinguish "always 1 per forward" from "sometimes 64, sometimes 0"."""
    if count == 0:
        return None
    return {
        "count": count,
        "total": total,
        "mean": round(total / count, 6),
        "min": min_n,
        "max": max_n,
        "fill_pct_mean": round((total / (count * max(batch_size, 1))) * 100.0, 6),
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }


#: What a caller hands the server so a contract failure lands on disk. A CALLABLE and not a dict,
#: because the context changes DURING the round and a snapshot would record the arming, not the fire.
CollateDumpTarget = tuple[str, "Callable[[], dict[str, Any]]"]


class InferenceServer(threading.Thread):
    """Thin Python inference loop backed by a Rust-owned batching queue."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: dict[str, Any],
        batcher: InferenceBatcher | None = None,
        encoding_spec: RegistrySpec | None = None,
        *,
        heartbeat: Callable[[str], None] | None = None,
        sink: EventSink | None = None,
        fused_graph_caps: FusedGraphCapsSpec | None = None,
        collate_check_period: int | None = None,
        collate_dump: CollateDumpTarget | None = None,
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.model.eval()
        self.device = device
        # The selfplay-local structural `EventSink`, NOT `mantis.train.emit.EventSink`: this
        # module must not import the train-side Protocol.
        self._sink = sink
        self._first_enqueued_emitted = False
        self._first_served_emitted = False
        # Behaviour-neutral by default: with no sink injected the emission points do nothing.
        self._heartbeat = heartbeat

        # `None` is the PRE-EXISTING rate — the canary period derived from the pop width, which
        # run6's `inference_batch_size: 64` makes 1-in-64. Not a config key: it is a path property.
        self._collate_check_period = collate_check_period
        self._collate_dump = collate_dump
        hp = InferenceHParams.from_config(config)
        self._batch_size = hp.inference_batch_size
        self._max_wait_ms = hp.inference_max_wait_ms

        # Encoding spec comes from the registry; standalone callers fall back to the config.
        if encoding_spec is None:
            self.encoding_spec: RegistrySpec = resolve_from_config(config)
        elif isinstance(encoding_spec, RegistrySpec):
            self.encoding_spec = encoding_spec
        else:
            raise TypeError(
                f"InferenceServer: unrecognised encoding_spec type "
                f"{type(encoding_spec).__name__!r}; expected mantis.encoding.EncodingSpec"
            )
        # Representation discriminant, and a closed match: the CNN staging / trace / (C,H,W) setup
        # below is grid-only, and an unknown representation raises rather than defaulting dense.
        self._is_graph = is_graph_representation(self.encoding_spec)
        self._policy_len = self.encoding_spec.policy_logit_count

        # Graph-loop batching instrumentation (LAW-18), written ONLY by `_run_graph_loop`, so a
        # grid run reports `None` per derived reading rather than a fabricated 0.
        self._batch_wait_count = 0
        self._batch_wait_total_s = 0.0
        self._batch_wait_min_s: float | None = None
        self._batch_wait_max_s: float | None = None
        self._collate_count = 0
        self._collate_total_s = 0.0
        self._collate_min_s: float | None = None
        self._collate_max_s: float | None = None
        self._occupancy_total = 0
        self._occupancy_min: int | None = None
        self._occupancy_max: int | None = None
        self._occupancy_hist: dict[int, int] = {}
        self._empty_polls = 0

        # The lever's OWN fire rate, in-run, measured PER PART: a pop's total is the sum over its
        # parts while the reverse is not. The counters stay VISIBLE at 0 on the producing path, or
        # the falsifier "`fusion_splits == 0` across a burst past ply 120" can never fire.
        self._fusion_parts = 0
        self._fusion_splits = 0
        self._fusion_bound_hits = {"edges": 0, "nodes": 0}
        self._fused_edges_count = 0
        self._fused_edges_total = 0
        self._fused_edges_min: int | None = None
        self._fused_edges_max: int | None = None
        self._fused_edges_hist: dict[int, int] = {}
        self._fused_nodes_count = 0
        self._fused_nodes_total = 0
        self._fused_nodes_min: int | None = None
        self._fused_nodes_max: int | None = None
        self._fused_nodes_hist: dict[int, int] = {}
        self._fused_caps: FusedGraphCapsSpec | None = None

        if self._is_graph:
        # The fused-forward memory bound, resolved ONCE and EAGERLY here: failing a mis-minted run
        # in the first second beats failing it three hours in. An explicit spec WINS over config.
            if fused_graph_caps is None:
                self._fused_caps = resolve_fused_graph_caps(config)
            else:
                self._fused_caps = fused_graph_caps
        # Graph mode: block-diagonal graph tensors, not a CNN — no H2D staging, no trace, no shape.
            self._feature_len = 0
            self._shape: tuple[int, int, int] | None = None
            self._board_size = self.encoding_spec.trunk_size
            self._batcher = batcher or InferenceBatcher(encoding_spec=self.encoding_spec)
            self._stop_event = threading.Event()
            self._weights_lock = threading.Lock()
            self._forward_count = 0
            self._total_requests = 0
            self._traced_model: Any = None
            self._h2d_staging: torch.Tensor | None = None
        else:
            # H2D staging sizes to the TRUNK window, the spatial dim the model accepts. For
            # single-window encodings trunk_size == board_size; multi-window encodings diverge.
            board_size = self.encoding_spec.trunk_size
            # Rust workers emit exactly `spec.kept_plane_indices` planes, so the wire width is the
            # ACTIVE encoding's plane count and never a hard-coded channel count.
            wire_channels = self.encoding_spec.n_planes
            self._feature_len = wire_channels * board_size * board_size
            self._shape = (wire_channels, board_size, board_size)

            self._batcher = batcher or InferenceBatcher(
                feature_len=self._feature_len,
                policy_len=self._policy_len,
            )
            self._stop_event = threading.Event()
            self._weights_lock = threading.Lock()
            self._forward_count = 0
            self._total_requests = 0

            # Pinned host staging buffer: a DMA-engine copy on CUDA, no-op on CPU.
            if self.device.type == "cuda":
                self._h2d_staging = torch.empty(
                    (self._batch_size, wire_channels, board_size, board_size),
                    dtype=torch.float32,
                    pin_memory=True,
                )
            else:
                self._h2d_staging = None

        # Autocast dtype, representation-aware: bf16 UNCONDITIONALLY on the graph loop (LAW-06),
        # since fp16 GINE sum-aggregation overflows; the dense path must match the trainer's knob.
        _representation = "graph" if self._is_graph else "grid"
        self._amp_dtype = amp_dtype_for(_representation)

    @property
    def batcher(self) -> InferenceBatcher:
        return self._batcher

    def stop(self) -> None:
        self._stop_event.set()
        self._batcher.close()

    def load_state_dict_safe(self, state_dict: dict) -> None:
        """Thread-safe weight swap — blocks until any in-flight forward completes. A compiled
        ``OptimizedModule`` is unwrapped once so the load targets the underlying parameters IN
        PLACE, and the batcher's monotonic ``model_version`` is bumped after the swap."""
        with self._weights_lock:
            target = getattr(self.model, "_orig_mod", self.model)
            target.load_state_dict(state_dict)
            target.eval()
            self.model.eval()
        # Bump after release: workers reading the atomic gate on post-swap visibility, not the lock.
        new_version = self._batcher.bump_model_version()
        _LOG.info(
            "inference_model_version_bump context=%s model_version=%s",
            "inference_server", new_version,
        )

    def submit_and_wait(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Synchronous single-state inference for test / diagnostic use, bypassing the Rust queue.

        Raises:
            ValueError: prefixed with ``"Model inference failed: "`` if the wrapped model forward
                raises. Translating it keeps callers waiting on a `threading.Event` from
                deadlocking on a thread-bound exception.
        """
        # Match the dispatcher's batch-prep contract (explicit C-contiguous f32).
        arr = np.ascontiguousarray(state, dtype=np.float32).reshape(self._shape)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        # The traced graph shares parameter storage with ``self.model``, so weight swaps propagate.
        fwd_model = self._traced_model if self._traced_model is not None else self.model
        try:
            with self._weights_lock:
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self._amp_dtype,
                        enabled=self.device.type == "cuda",
                    ):
                        log_policy, value, _v_logit = fwd_model(tensor)
        except Exception as exc:  # noqa: BLE001 — translated + re-raised, never swallowed
            raise ValueError(f"Model inference failed: {exc}") from exc

        probs = log_policy.float().exp()
        probs = probs / probs.sum(dim=-1, keepdim=True)
        policy_np = probs.squeeze(0).cpu().numpy().astype(np.float32)
        value_f = float(value.squeeze().cpu().item())

        self._total_requests += 1
        self._forward_count += 1
        return policy_np, value_f

    def infer(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        return self.submit_and_wait(state)

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def total_requests(self) -> int:
        return self._total_requests

    def _record_batch_wait(self, wait_s: float, n_requests: int) -> None:
        """Accumulate ONE served pop: the collector wait that produced it plus its occupancy.
        AGGREGATE, never emit — this runs once per NN forward. `wait_s` pegged at
        `inference_max_wait_ms` means every batch ran to the collector's deadline."""
        self._batch_wait_count += 1
        self._batch_wait_total_s += wait_s
        if self._batch_wait_min_s is None or wait_s < self._batch_wait_min_s:
            self._batch_wait_min_s = wait_s
        if self._batch_wait_max_s is None or wait_s > self._batch_wait_max_s:
            self._batch_wait_max_s = wait_s
        self._occupancy_total += n_requests
        if self._occupancy_min is None or n_requests < self._occupancy_min:
            self._occupancy_min = n_requests
        if self._occupancy_max is None or n_requests > self._occupancy_max:
            self._occupancy_max = n_requests
        bucket = _pow2_bucket(n_requests)
        self._occupancy_hist[bucket] = self._occupancy_hist.get(bucket, 0) + 1

    def _dump_collate_failure(
        self, wire: Any, error: BaseException, span: tuple[int, int]
    ) -> None:
        """Write the offending batch before the caller re-raises. Never raises. A server with no
        dump target does nothing: the instrument is armed per PATH, and the write only happens on a
        contract failure, which is run-fatal."""
        if self._collate_dump is None:
            return
        from mantis.selfplay.collate_dump import write_collate_dump

        dump_dir, context_fn = self._collate_dump
        context: dict[str, Any]
        try:
            context = dict(context_fn())
        except Exception:  # noqa: BLE001 — a context that raises must not eat the dump
            context = {"context_error": "context callable raised"}
        context["fused_span"] = [int(span[0]), int(span[1])]
        path = write_collate_dump(wire, dump_dir=dump_dir, context=context, error=error)
        if path is None:
            _LOG.error("F-816-37 dump-on-fire FAILED to write under %s", dump_dir)
        else:
            _LOG.error("F-816-37 dump-on-fire wrote %s", path)

    def _record_collate(self, collate_s: float) -> None:
        """Accumulate ONE successful `collate_graph_batch`, counted SEPARATELY from the wait: a
        batch whose collate raises still contributes a real wait sample."""
        self._collate_count += 1
        self._collate_total_s += collate_s
        if self._collate_min_s is None or collate_s < self._collate_min_s:
            self._collate_min_s = collate_s
        if self._collate_max_s is None or collate_s > self._collate_max_s:
            self._collate_max_s = collate_s

    def _record_fusion_plan(self, n_parts: int, edge_hits: int, node_hits: int) -> None:
        """Accumulate ONE plan: whether the lever fired, and which member forced each cut.
        `fusion_splits` counts POPS THAT SPLIT, not cuts; attribution is `fusion_bound_hits`, which
        is what tells an operator which member to re-fit."""
        if n_parts > 1:
            self._fusion_splits += 1
        self._fusion_bound_hits["edges"] += edge_hits
        self._fusion_bound_hits["nodes"] += node_hits

    def _record_fusion_part(self, n_nodes: int, n_edges: int) -> None:
        """Accumulate ONE bounded forward's `(N, E)`. Per PART, never per pop, because the part is
        where the cap applies; `_forward_count` stays a per-POP occupancy denominator."""
        self._fusion_parts += 1
        self._fused_edges_count += 1
        self._fused_edges_total += n_edges
        if self._fused_edges_min is None or n_edges < self._fused_edges_min:
            self._fused_edges_min = n_edges
        if self._fused_edges_max is None or n_edges > self._fused_edges_max:
            self._fused_edges_max = n_edges
        bucket_e = _pow2_bucket(n_edges)
        self._fused_edges_hist[bucket_e] = self._fused_edges_hist.get(bucket_e, 0) + 1
        self._fused_nodes_count += 1
        self._fused_nodes_total += n_nodes
        if self._fused_nodes_min is None or n_nodes < self._fused_nodes_min:
            self._fused_nodes_min = n_nodes
        if self._fused_nodes_max is None or n_nodes > self._fused_nodes_max:
            self._fused_nodes_max = n_nodes
        bucket_n = _pow2_bucket(n_nodes)
        self._fused_nodes_hist[bucket_n] = self._fused_nodes_hist.get(bucket_n, 0) + 1

    def _fusion_snapshot(self) -> dict[str, Any] | None:
        """The `fusion` sub-block, or `None` on a GRID run — a zeroed block would read as "the
        lever ran and never fired". `caps` travels WITH the distributions, because a maximum of
        4.4 M edges says nothing without the cap beside it."""
        caps = self._fused_caps
        if not self._is_graph or caps is None:
            return None
        return {
            "caps": {
                "max_fused_edges": caps.max_fused_edges,
                "max_fused_nodes": caps.max_fused_nodes,
            },
            "fusion_parts": self._fusion_parts,
            "fusion_splits": self._fusion_splits,
            "fusion_bound_hits": dict(self._fusion_bound_hits),
            "fused_batch_nodes": _size_agg(
                self._fused_nodes_count, self._fused_nodes_total,
                self._fused_nodes_min, self._fused_nodes_max, self._fused_nodes_hist,
            ),
            "fused_batch_edges": _size_agg(
                self._fused_edges_count, self._fused_edges_total,
                self._fused_edges_min, self._fused_edges_max, self._fused_edges_hist,
            ),
        }

    def batch_timing_snapshot(self) -> dict[str, Any]:
        """Cumulative-since-start snapshot of the graph loop's batching instrument. `batch_size`
        and `max_wait_ms` travel with it, since a wait or an occupancy is unreadable without the
        deadline and the denominator behind it; a reading with no sample is `None`, never 0."""
        return {
            "representation": "graph" if self._is_graph else "grid",
            "batch_size": self._batch_size,
            "max_wait_ms": self._max_wait_ms,
            "queue_wait": _timing_agg(
                self._batch_wait_count, self._batch_wait_total_s,
                self._batch_wait_min_s, self._batch_wait_max_s,
            ),
            "collate": _timing_agg(
                self._collate_count, self._collate_total_s,
                self._collate_min_s, self._collate_max_s,
            ),
            "occupancy": _occupancy_agg(
                self._batch_wait_count, self._occupancy_total, self._occupancy_min,
                self._occupancy_max, self._occupancy_hist, self._batch_size,
            ),
            # An idle counter stays VISIBLE at 0 on the producing path; `None` where none exists.
            "empty_polls": self._empty_polls if self._is_graph else None,
            # The memory bound's own in-run instrument: PRESENT with a `None` value on a grid run,
            # never absent, since an absent key and a null one differ only if the key is always there.
            "fusion": self._fusion_snapshot(),
        }

    def _run_graph_loop(self) -> None:
        """Ragged axis-graph inference loop, MEMORY-BOUNDED.

        Pull a block-diagonal graph wire from Rust, convert it to a payload ONCE, partition that
        payload at GRAPH boundaries under `inference.fused_graph_caps`, and run one
        `collate_graph_batch` + `GnnNet.forward_batch` (bf16 autocast — LAW-06) + segment-softmax
        per PART, freeing each part before the next so only one part's tensors are ever resident.

        The split is PRE-COLLATE, because a design whose first allocation is proportional to the
        uncapped quantity cannot meet a bound; `PyGraphWire`'s getters COPY, so the payload is read
        ONCE and the parts are numpy views of it. ONE SUBMIT, after every part has run, against the
        UNSLICED `legal_offsets`, so a mid-plan failure has submitted NOTHING. Every failure — a
        real `OutOfMemoryError` included — dies loud through the SAME `except`, and there is
        deliberately no OOM handler, because the only reason to catch one is to retry.
        """
        from mantis.selfplay.graph_collate import (
            GraphContractError,
            collate_graph_batch,
            graph_wire_from_rust,
            reset_semantic_canary,
            segment_softmax,
            stone_mask_from_batch,
        )
        from mantis.selfplay.graph_wire_split import (
            plan_fused_forwards,
            slice_graph_wire,
        )

        caps = self._fused_caps
        if caps is None:
            # Unreachable by construction: the caps are resolved before this thread starts. A None
            # here is a wiring break, and running unbounded must not be an available outcome.
            raise RuntimeError(
                "InferenceServer graph loop: no fused-graph caps resolved — the graph branch "
                "of __init__ must produce them before the loop runs (inference.fused_graph_"
                "caps)."
            )
        spec = self.encoding_spec
        # A graph spec carries all three graph fields; None means a grid spec routed here.
        win_length = spec.win_length
        node_feat_dim = spec.node_feat_dim
        edge_feat_dim = spec.edge_feat_dim
        if win_length is None or node_feat_dim is None or edge_feat_dim is None:
            raise RuntimeError(
                f"InferenceServer graph loop: encoding spec {spec.name!r} is missing graph "
                f"fields (win_length={win_length}, node_feat_dim={node_feat_dim}, "
                f"edge_feat_dim={edge_feat_dim}) — a non-graph spec routed to the graph loop."
            )
        # First batch after (re)start runs the FULL semantic/geometric layer.
        reset_semantic_canary()
        canary_period = (
            int(self._batch_size) if self._collate_check_period is None
            else int(self._collate_check_period)
        )

        try:
            while not self._stop_event.is_set():
                try:
                    _t_wait_start = time.perf_counter()
                    request_ids, wire = self._batcher.next_graph_batch(
                        self._batch_size, self._max_wait_ms,
                    )
                    _wait_s = time.perf_counter() - _t_wait_start
                    if not request_ids:
                        # An empty pop is a deadline that expired with nothing queued: not a
                        # served-batch wait, and it must not enter the wait mean.
                        self._empty_polls += 1
                        continue
                    self._record_batch_wait(_wait_s, len(request_ids))
                    if not self._first_enqueued_emitted:
                        self._first_enqueued_emitted = True
                        if self._sink is not None:
                            self._sink.emit({
                                "event": "first_inference_enqueued",
                                "batch_size": len(request_ids),
                                "representation": "graph",
                            })
                    self._total_requests += len(request_ids)
                    try:
                        # ONE read of each Rust getter, then pure-numpy views per part.
                        payload = graph_wire_from_rust(wire)
                        edge_counts = np.diff(
                            np.asarray(payload.edge_offsets, dtype=np.int64)
                        )
                        node_counts = np.diff(
                            np.asarray(payload.node_offsets, dtype=np.int64)
                        )
                        plan = plan_fused_forwards(
                            payload.edge_offsets, payload.node_offsets, caps,
                        )
                        self._record_fusion_plan(
                            len(plan),
                            *_fusion_bound_hits(plan, edge_counts, node_counts, caps),
                        )
                        probs_parts: list[np.ndarray] = []
                        values_parts: list[np.ndarray] = []
                        for g0, g1 in plan:
                            sub = slice_graph_wire(payload, g0, g1)
                            _t_collate_start = time.perf_counter()
                            try:
                                batch = collate_graph_batch(
                                    sub,
                                    expected_version=1,
                                    trunk_size=spec.trunk_size,
                                    win_length=win_length,
                                    node_feat_dim=node_feat_dim,
                                    edge_feat_dim=edge_feat_dim,
                                    device=str(self.device),
                                    semantic="canary",
                                    canary_period=canary_period,
                                )
                            except GraphContractError as exc:
                                # DUMP-ON-FIRE: the SLICE is what the check read, so the slice is
                                # what is saved. The dump can only ADD an artifact, never replace it.
                                self._dump_collate_failure(sub, exc, (g0, g1))
                                raise
                            # Per PART, not per pop: `collate.count == sum(M)`, and the asymmetry
                            # is recorded so it is not read as a leak.
                            self._record_collate(time.perf_counter() - _t_collate_start)
                            stone_mask = stone_mask_from_batch(batch)
                            if self._forward_count == 0:
                                assert not self.model.training, (
                                    "InferenceServer(graph) model entered hot loop in "
                                    "train() mode; eval() should be set at __init__"
                                )
                            with self._weights_lock, torch.inference_mode():
                                with torch.autocast(
                                    device_type=self.device.type,
                                    dtype=self._amp_dtype,
                                    enabled=self.device.type == "cuda",
                                ):
                                    # `forward_batch` is GnnNet's real method; nn.Module's
                                    # __getattr__ types dynamic attrs as Tensor | Module.
                                    policy_logits, value, _bins = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                                        batch.x,
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.legal_node_gather,
                                        stone_mask,
                                        batch.node_offsets,
                                    )
                            # Segment-softmax in float32 corrects reduced-precision drift and is
                            # segment-LOCAL, so a part's softmax is the un-split forward's softmax.
                            probs = segment_softmax(
                                policy_logits.float(), batch.legal_offsets
                            )
                            # Always-on finiteness gate: a NaN/Inf output otherwise reaches backup()
                            # and poisons the tree silently, and the numeric asserts are release-out.
                            if not bool(torch.isfinite(probs).all()) or not bool(
                                torch.isfinite(value).all()
                            ):
                                raise RuntimeError(
                                    "NonFiniteModelOutput: graph forward produced NaN/Inf "
                                    f"(probs finite={bool(torch.isfinite(probs).all())}, "
                                    f"values finite={bool(torch.isfinite(value).all())})"
                                )
                            probs_parts.append(np.ascontiguousarray(
                                probs.detach().cpu().numpy(), dtype=np.float32
                            ))
                            values_parts.append(np.ascontiguousarray(
                                value.detach().float().cpu().numpy().reshape(-1),
                                dtype=np.float32,
                            ))
                            self._record_fusion_part(
                                int(node_counts[g0:g1].sum()),
                                int(edge_counts[g0:g1].sum()),
                            )
                            # One part resident at a time: the bound is on the PEAK, so the previous
                            # part's device tensors must be gone before the next's exist.
                            del sub, batch, stone_mask, policy_logits, value, probs
                        # ONE submit per pop, against the payload's own UNSLICED offsets: the
                        # parts' offsets are re-based and would segment the concatenation wrongly.
                        self._batcher.submit_graph_inference_results(
                            request_ids,
                            np.ascontiguousarray(
                                np.concatenate(probs_parts), dtype=np.float32
                            ),
                            np.ascontiguousarray(
                                np.asarray(payload.legal_offsets), dtype=np.int64
                            ),
                            np.ascontiguousarray(
                                np.concatenate(values_parts), dtype=np.float32
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 — reported to Rust waiters
                        error_msg = f"Graph inference failed: {exc}"
                        _LOG.error(
                            "graph_inference_forward_failed context=%s error_type=%s "
                            "error=%s tb=%s",
                            "inference_server", type(exc).__name__,
                            str(exc)[:300] or repr(exc)[:300],
                            traceback.format_exc()[:1500],
                        )
                        self._batcher.submit_graph_inference_failure(request_ids, error_msg)
                        continue
                    self._forward_count += 1
                    if not self._first_served_emitted:
                        self._first_served_emitted = True
                        if self._sink is not None:
                            self._sink.emit({
                                "event": "first_inference_served",
                                "batch_size": len(request_ids),
                                "representation": "graph",
                            })
                    if self._heartbeat is not None:
                        self._heartbeat(_HEARTBEAT_SOURCE)
                except Exception as exc:  # noqa: BLE001 — loop keeps serving next batch
                    _LOG.exception("inference_server_graph_loop_error error=%s", exc)
                    if self._stop_event.is_set():
                        break
        finally:
            self._batcher.close()

    def run(self) -> None:
        """Serve inference until `stop()`. One loop: the ragged axis-graph one."""
        self._run_graph_loop()


__all__ = ["InferenceServer"]
