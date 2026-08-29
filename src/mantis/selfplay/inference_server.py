"""THE batched inference server — the one dispatch loop in the tree.

>300 justify: the ONE inference loop. Rust owns request concurrency; this module is the
whole Python side of the dispatch seam — construction (representation dispatch, H2D
staging, trace/compile setup), the dense loop, the graph loop, the synchronous
`submit_and_wait` face and the thread-safe weight swap. Splitting the two loops apart
would create a second place a batch can be prepared, submitted or failed, which is
exactly the duplication this WP consolidates (three old inference paths → one server).

Rust owns request concurrency via `InferenceBatcher`. Python runs a thin loop: fetch the
fused batch from Rust, run the model forward, submit policy/value outputs back, and wake
the blocked game threads.

The representation is resolved ONCE at construction from the encoding spec (closed match,
LAW-11 — there is no dense-by-default arm) and selects the loop:

  * grid  → `run()`'s dense loop: pinned-staging H2D, optional TorchScript trace or
    `torch.compile`, autocast at the configured `amp_dtype`, one merged D2H.
  * graph → `_run_graph_loop()`: `collate_graph_batch` (the ONE wire reader) →
    `GnnNet.forward_batch` → per-graph segment softmax → ragged submit. Autocast is
    bf16 UNCONDITIONALLY on this path (LAW-06) — `amp_dtype_for` owns that pin.

`LocalInferenceEngine`'s graph leg constructs and rides THIS server rather than
re-implementing a loop (`inference_local.py`), so self-play and offline eval share one
implementation of the graph seam.
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

# Emitted once per dispatched batch when a sink is injected. The named alias
# `HeartbeatFn` lives with the other injection Protocols in `pool_hooks`; the server
# only needs the structural type, and must not import the pool to get it.
_HEARTBEAT_SOURCE = "inference_dispatch"


def _timing_agg(
    count: int, total_s: float, min_s: float | None, max_s: float | None
) -> dict[str, Any] | None:
    """One timing accumulator as an event sub-block, or `None` when nothing was measured.

    `None` at zero samples is deliberate and is the whole point: a field with no producer
    on this path must NOT read as a real `0.0` measurement
    (docs/contracts/event_manifest.md, the unproduced-field convention — the F-10 class in
    miniature). The RAW `count`/`total_ms` travel beside the derived `mean_ms` so a
    consumer can difference two consecutive events and recover an INTERVAL mean; the
    min/max are run-cumulative extremes and do NOT difference.
    """
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
    """The power-of-two LOWER bound of `n`'s histogram bucket (`1`, `2`, `4`, …).

    Extracted so the occupancy histogram and the fused-part histograms cannot drift apart:
    two transcriptions of one bucketing rule would put two different keys on the same reading
    and neither event would say which one it used.
    """
    return 1 << (n.bit_length() - 1) if n > 0 else 0


def _size_agg(
    count: int, total: int, min_n: int | None, max_n: int | None, hist: dict[int, int]
) -> dict[str, Any] | None:
    """One per-part size distribution (fused nodes or fused edges), or `None` at zero samples.

    DISTRIBUTION AND NOT A MEAN, for `_occupancy_agg`'s recorded reason applied to memory: a
    mean fused-E of 400 k with a max of 9 M is a run that OOMs, and the two readings agree on
    the mean. For a memory bound the TAIL is the question, so `max` and the histogram travel
    with it — `max` is what an operator reads at the box to decide whether the cap held.

    `None` at zero samples is the unproduced-field convention: before the first part runs there
    is no producer, and a zeroed histogram would read as "parts ran and were empty".
    """
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
    """`(edge-forced cuts, node-forced cuts)` for one plan — the ATTRIBUTION half of the bound.

    A cut sits between part `m` and part `m+1`, and it happened because adding part `m+1`'s
    FIRST graph to part `m` would have breached a member. Which member is the whole question an
    operator asks at the box: an instrument that only counts cuts says the cap bound and cannot
    say WHICH cap to re-fit, and re-fitting the wrong one moves no peak.

    Edges are tested first, matching the planner's own evaluation order, so a cut that breaches
    both members is attributed to the same member the planner would name.
    """
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
    """The served-batch occupancy distribution, or `None` when nothing was measured.

    A mean ratio alone cannot distinguish "always 1 request per forward" from "sometimes
    64, sometimes 0" — the two agree on the ratio and disagree completely on what the
    queue is doing. So min/max and a power-of-two histogram travel with it; the histogram
    key is the bucket's LOWER bound (`1`, `2`, `4`, … requests per forward).
    """
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
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.model.eval()
        self.device = device
        # WP12R Step 3 narration (R216/R218): the selfplay-local `EventSink` Protocol
        # (`pool_hooks.py:32-40`), structural (single `emit(Mapping) -> None`). NOT
        # `mantis.train.emit.EventSink` — this module must not import the train-side
        # Protocol (R216: no `selfplay → train` DAG edge). Same structural-typing pattern
        # as `HeartbeatFn` (`:47-49`: "the server only needs the structural type").
        self._sink = sink
        self._first_enqueued_emitted = False
        self._first_served_emitted = False
        # Behaviour-neutral by default: with no sink injected the emission points below
        # do nothing at all. The consuming watchdog is not built here.
        self._heartbeat = heartbeat

        hp = InferenceHParams.from_config(config)
        self._batch_size = hp.inference_batch_size
        self._max_wait_ms = hp.inference_max_wait_ms

        # Encoding spec comes from the registry. Standalone callers (no kwarg) fall back
        # to resolving from config.
        if encoding_spec is None:
            self.encoding_spec: RegistrySpec = resolve_from_config(config)
        elif isinstance(encoding_spec, RegistrySpec):
            self.encoding_spec = encoding_spec
        else:
            raise TypeError(
                f"InferenceServer: unrecognised encoding_spec type "
                f"{type(encoding_spec).__name__!r}; expected mantis.encoding.EncodingSpec"
            )
        # Representation discriminant: a `graph` encoding routes to the ragged axis-graph
        # seam (`_run_graph_loop`); the CNN H2D-staging / TorchScript-trace / (C,H,W)-shape
        # setup below is grid-only and would be meaningless (n_planes=0, no state stride)
        # for a graph spec. Closed match — an unknown representation raises rather than
        # defaulting dense.
        self._is_graph = is_graph_representation(self.encoding_spec)
        self._policy_len = self.encoding_spec.policy_logit_count

        # ── graph-loop batching instrumentation (LAW-18) ─────────────────────────────
        # Written ONLY by `_run_graph_loop`. The dense loop leaves every accumulator at
        # its zero, so `batch_timing_snapshot` reports `None` for each derived reading on
        # a grid run — "no producer on this path", never a fabricated 0. Assigned before
        # the representation branch so the accessor never raises on either arm.
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

        # PERF-BASELINE (2026-08-29) DIAGNOSTIC stage accumulators for the graph loop.
        # Written only under `diagnostics.perf_timing`; read by `perf_stage_snapshot`.
        self._stage_count: dict[str, int] = {}
        self._stage_total_s: dict[str, float] = {}
        self._stage_max_s: dict[str, float] = {}

        # ── fused-forward instrumentation (LAW-18 / R164) ────────────────────────────
        # The lever's OWN fire rate, in-run. Written ONLY by `_run_graph_loop`; measured PER
        # PART, because the part is what the GPU sees and what the cap bounds — a pop's total
        # is recoverable as the sum over its parts and the reverse is not. `fusion_splits` and
        # `fusion_bound_hits` stay VISIBLE at 0 on the producing path (the `empty_polls`
        # posture): an idle lever must be distinguishable from a missing one, because §11's
        # falsifier ("`fusion_splits == 0` across a burst that reaches ply > 120") can only
        # fire if the zero is published.
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
            # The fused-forward memory bound, resolved ONCE and EAGERLY on the route that has
            # one (F-816-10). Eager, not lazy: `__init__` already branches on the
            # representation, so the read is naturally route-scoped and failing a mis-minted
            # run in the first second beats failing it three hours in. An explicit spec WINS
            # over the config and skips the read entirely — that is the D-1 threading arm, for
            # the standalone callers that have no `RunConfig` to mint a value against and must
            # NOT grow a second authority by hardcoding one.
            if fused_graph_caps is None:
                self._fused_caps = resolve_fused_graph_caps(config)
            else:
                self._fused_caps = fused_graph_caps
            # Graph mode: the model is a `GnnNet` consuming block-diagonal graph tensors,
            # not a CNN. No H2D staging, no trace, no (C,H,W) shape.
            self._feature_len = 0
            self._shape: tuple[int, int, int] | None = None
            self._board_size = self.encoding_spec.trunk_size
            self._batcher = batcher or InferenceBatcher(encoding_spec=self.encoding_spec)
            self._stop_event = threading.Event()
            self._weights_lock = threading.Lock()
            self._forward_count = 0
            self._total_requests = 0
            # Inert grid-path attributes so shared accessors don't raise.
            self._trace_inference = False
            self._traced_model: Any = None
            self._compile_inference = False
            self._compile_mode: str | None = None
            self._compile_dynamic = False
            self._h2d_staging: torch.Tensor | None = None
        else:
            # H2D staging tensors size to the TRUNK window (the spatial dim the model
            # actually accepts). For the single-window encodings trunk_size == board_size,
            # so this is a no-op semantic shift today; multi-window encodings diverge.
            board_size = self.encoding_spec.trunk_size
            # Rust workers emit exactly `spec.kept_plane_indices` planes. The wire width is
            # the ACTIVE encoding's plane count, never a hard-coded channel count;
            # sub-selection of input channels happens inside `model.forward()`.
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

            self._setup_inference_path(hp, board_size)

            # Pinned host staging buffer for async H2D. Enables a DMA-engine copy on CUDA
            # (`non_blocking=True`); no-op on CPU.
            if self.device.type == "cuda":
                self._h2d_staging = torch.empty(
                    (self._batch_size, wire_channels, board_size, board_size),
                    dtype=torch.float32,
                    pin_memory=True,
                )
            else:
                self._h2d_staging = None

        # Perf-investigation probes.
        self._perf_timing = hp.perf_timing
        self._perf_sync_cuda = hp.perf_sync_cuda
        if self._perf_sync_cuda and torch.cuda.is_available():
            _LOG.warning(
                "perf_sync_cuda_enabled_serialising_stream context=%s impact=%s remedy=%s",
                "inference_server",
                "expect_30_50_pct_throughput_drop",
                "unset_diagnostics.perf_sync_cuda_in_production_config",
            )

        # Autocast dtype — representation-aware. The graph loop is pinned to bf16
        # UNCONDITIONALLY (LAW-06): fp16 GINE sum-aggregation overflows on
        # production-scale graphs. The dense path reads the `train.amp_dtype` knob and must
        # match the trainer's choice for weight-sync consistency. R30b: hard key access, no
        # fallback — config["train"]["amp_dtype"] is a required schema field.
        _representation = "graph" if self._is_graph else "grid"
        self._amp_dtype = amp_dtype_for(_representation, config["train"]["amp_dtype"])

    def _setup_inference_path(self, hp: InferenceHParams, board_size: int) -> None:
        """Configure the trace OR compile path for the inference model.

        Mutually exclusive: `trace_inference` and `compile_inference` cannot both be
        enabled. Sets `_trace_inference`, `_traced_model`, `_compile_inference`,
        `_compile_mode`, `_compile_dynamic`; may replace `self.model` with a
        `torch.compile` wrapper. Called once at `__init__` — the run loop reads the
        resolved attributes, so there is no per-batch overhead from this helper.
        """
        if self._shape is None:
            # Grid-only helper: __init__ calls it exclusively from the dense arm, after
            # `_shape` is assigned; the graph arm never routes here.
            raise RuntimeError(
                "InferenceServer._setup_inference_path: no (C, H, W) shape — the dense "
                "setup was entered for a graph encoding."
            )
        # TorchScript trace of the eval forward: collapses ~100 `nn.Module` `_call_impl`
        # invocations per forward into one ScriptModule whose parameters SHARE storage
        # with `model`, so `load_state_dict_safe`'s in-place mutation keeps flowing into
        # the traced graph without re-tracing.
        self._trace_inference = hp.trace_inference
        self._traced_model: Any = None
        if self._trace_inference:
            try:
                self.model.requires_grad_(False)
                with torch.inference_mode():
                    _example = torch.zeros(
                        self._batch_size, *self._shape, device=self.device,
                    )
                    self._traced_model = torch.jit.trace(
                        self.model, _example, strict=False,
                    )
                _LOG.info(
                    "inference_trace_compiled context=%s batch_size=%s board_size=%s",
                    "inference_server", self._batch_size, board_size,
                )
            except Exception as exc:  # noqa: BLE001 — degrade to the untraced module, logged
                _LOG.warning(
                    "inference_trace_failed_falling_back context=%s error=%s",
                    "inference_server", str(exc)[:200],
                )
                self._traced_model = None

        # `torch.compile` knob. Mutually exclusive with trace — both attack the same
        # bottleneck (Python dispatch / kernel-launch overhead) and stacking them does not
        # compose. Mode `default` is thread-safe from any caller; `reduce-overhead`
        # requires the dispatcher thread's TLS to own the cudagraph_trees context.
        self._compile_inference = hp.compile_inference
        self._compile_mode = hp.compile_inference_mode
        self._compile_dynamic = hp.compile_inference_dynamic
        if self._compile_inference and self._trace_inference:
            raise ValueError(
                "compile_inference and trace_inference are mutually exclusive; "
                "set one to false in the selfplay config."
            )
        if self._compile_inference:
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self._compile_mode,
                    dynamic=self._compile_dynamic,
                )
                _LOG.info(
                    "inference_compile_enabled context=%s mode=%s dynamic=%s",
                    "inference_server", self._compile_mode, self._compile_dynamic,
                )
            except Exception as exc:  # noqa: BLE001 — degrade to eager, logged
                _LOG.warning(
                    "inference_compile_failed_falling_back context=%s error=%s",
                    "inference_server", str(exc)[:200],
                )
                self._compile_inference = False

    @property
    def batcher(self) -> InferenceBatcher:
        return self._batcher

    def stop(self) -> None:
        self._stop_event.set()
        self._batcher.close()

    def load_state_dict_safe(self, state_dict: dict) -> None:
        """Thread-safe weight swap — blocks until any in-flight forward completes.

        Callers pass bare (non-``_orig_mod.*``-prefixed) state_dict keys. When
        ``self.model`` is a compiled `OptimizedModule`, ``load_state_dict`` would
        otherwise demand the prefixed keys; unwrap once here so the load targets the
        underlying parameters IN PLACE (the wrapper keeps dispatching through them, and
        the trace path relies on the same propagation).

        Bumps the batcher's monotonic ``model_version`` after the swap so workers can
        attribute each move to a specific weight epoch.
        """
        with self._weights_lock:
            target = getattr(self.model, "_orig_mod", self.model)
            target.load_state_dict(state_dict)
            target.eval()
            self.model.eval()
        # Bump after release — workers reading the atomic don't gate on the lock, only on
        # the post-swap visibility of new params.
        new_version = self._batcher.bump_model_version()
        _LOG.info(
            "inference_model_version_bump context=%s model_version=%s",
            "inference_server", new_version,
        )

    def submit_and_wait(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Synchronous single-state inference for test / diagnostic use.

        Runs the model forward in-process under ``_weights_lock``, mirroring the
        dispatcher loop's hot path (trace/compile model selection, prep, autocast).
        Bypasses the Rust queue — production self-play goes through the dispatcher thread
        and Rust workers and does not call this method.

        Raises:
            ValueError: prefixed with ``"Model inference failed: "`` if the wrapped model
                forward raises. Translating the underlying error keeps callers waiting on
                a `threading.Event` from deadlocking on a thread-bound exception (the
                dispatcher path carries the same contract through
                ``submit_inference_failure``).
        """
        # Match the dispatcher's batch-prep contract (explicit C-contiguous f32).
        arr = np.ascontiguousarray(state, dtype=np.float32).reshape(self._shape)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        # Choose the traced graph when available — it shares parameter storage with
        # ``self.model``, so weight swaps propagate without re-tracing.
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

    # ── batching instrumentation (LAW-18) ───────────────────────────────────────
    def _record_batch_wait(self, wait_s: float, n_requests: int) -> None:
        """Accumulate ONE served pop: the collector wait that produced it + its occupancy.

        AGGREGATE, never emit (LAW-09): this runs once per NN forward — potentially
        thousands of times a second — so an event per call would make the sink the
        bottleneck and would itself be the hot-path change this instrument exists to
        measure around. What reaches the ONE channel is a SNAPSHOT on an existing event.

        `wait_s` is the wall time spent inside `next_graph_batch`, i.e. exactly the Rust
        collector's `batch_size / 2`-or-deadline wait (`queues/graph.rs`): if it sits at
        `inference_max_wait_ms` on every forward, the collector never reached its
        threshold and every batch ran to the deadline.
        """
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

    def _record_stage(self, name: str, dt: float) -> None:
        """Fold one DIAGNOSTIC stage sample into the cumulative accumulators."""
        self._stage_count[name] = self._stage_count.get(name, 0) + 1
        self._stage_total_s[name] = self._stage_total_s.get(name, 0.0) + dt
        prev = self._stage_max_s.get(name)
        if prev is None or dt > prev:
            self._stage_max_s[name] = dt

    def perf_stage_snapshot(self) -> dict[str, Any]:
        """Cumulative per-stage timings for the graph loop (PERF-BASELINE diagnostic).

        Empty when `diagnostics.perf_timing` is off — an absent stage is an un-armed
        timer, never a measured zero.
        """
        return {
            "perf_timing": self._perf_timing,
            "perf_sync_cuda": self._perf_sync_cuda,
            "forward_count": self._forward_count,
            "total_requests": self._total_requests,
            "stages": {
                name: {
                    "count": count,
                    "total_ms": self._stage_total_s[name] * 1e3,
                    "mean_ms": (self._stage_total_s[name] / count) * 1e3,
                    "max_ms": self._stage_max_s[name] * 1e3,
                }
                for name, count in sorted(self._stage_count.items())
            },
        }

    def _record_collate(self, collate_s: float) -> None:
        """Accumulate ONE successful `collate_graph_batch`. Counted SEPARATELY from the
        wait: a batch whose collate raises still contributes a real wait sample, and
        lock-stepping the two counters would hide that asymmetry."""
        self._collate_count += 1
        self._collate_total_s += collate_s
        if self._collate_min_s is None or collate_s < self._collate_min_s:
            self._collate_min_s = collate_s
        if self._collate_max_s is None or collate_s > self._collate_max_s:
            self._collate_max_s = collate_s

    def _record_fusion_plan(self, n_parts: int, edge_hits: int, node_hits: int) -> None:
        """Accumulate ONE plan: whether the lever fired, and which member forced each cut.

        `fusion_splits` counts POPS THAT SPLIT, not cuts — it is the LEVER'S OWN FIRE RATE,
        which is what LAW-18 asks a lever under test to log. The cut count is
        `fusion_parts - fusion_splits`-shaped information and is carried instead by
        `fusion_bound_hits`, whose whole job is ATTRIBUTION: an instrument that cannot say
        which member forced the cut cannot tell the operator which member to re-fit at the box.
        """
        if n_parts > 1:
            self._fusion_splits += 1
        self._fusion_bound_hits["edges"] += edge_hits
        self._fusion_bound_hits["nodes"] += node_hits

    def _record_fusion_part(self, n_nodes: int, n_edges: int) -> None:
        """Accumulate ONE bounded forward's `(N, E)`. Per PART, never per pop.

        This is the reading the cap is denominated in, so it is measured where the cap applies.
        `_forward_count` deliberately does NOT move here: it is `batch_fill_pct`'s denominator
        and means *requests per POP against `inference_batch_size`* — an occupancy, not a
        GPU-forward count — and it is banked on both sides of the R274(d) bench.
        """
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
        """The `fusion` sub-block, or `None` on a GRID run.

        `None` and not a zeroed block: the grid branch never reads the caps and never plans a
        split, so it has NO PRODUCER here, and a zeroed block would read as "the fusion lever
        ran and never fired" — the opposite statement, and the F-10 class in miniature.

        `caps` travels WITH the distributions for the same reason `batch_size`/`max_wait_ms`
        already ride the block: a histogram whose maximum is 4.4 M edges says nothing until the
        cap beside it says 4.5 M or 9 M.
        """
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
        """Cumulative-since-start snapshot of the graph loop's batching instrument.

        The block that reaches `iteration_complete` (LAW-18: a lever under test logs its
        own fire rate IN-RUN — a post-hoc offline probe cannot distinguish a starved queue
        from an ineffective one). `batch_size` and `max_wait_ms` travel with it because a
        wait or an occupancy is unreadable without the deadline and the denominator that
        produced it.

        Every derived reading is `None` when its accumulator took no sample — including
        the whole of a GRID run, whose dense loop is not instrumented and therefore has no
        producer here.
        """
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
            # An idle counter stays VISIBLE at 0 on the producing path (the
            # `target_integrity_defects` posture); `None` on the path with no producer.
            "empty_polls": self._empty_polls if self._is_graph else None,
            # The memory bound's own in-run instrument (F-816-10, LAW-18/R164). PRESENT with a
            # `None` value on a grid run, never absent: an absent key and a null one are the
            # same statement here only if the key is always there to carry it.
            "fusion": self._fusion_snapshot(),
        }

    # ── Thread body ─────────────────────────────────────────────────────────────
    def _padding_active(self) -> bool:
        """The compile + `reduce-overhead` path replays a captured CUDA graph, which
        requires a fixed input shape: each batch is padded up to ``self._batch_size`` and
        outputs are sliced back to the actual request count."""
        return (
            self._compile_inference
            and self._compile_mode == "reduce-overhead"
            and self._h2d_staging is not None
        )

    def _warmup_compile_path(self) -> None:
        """CUDA-graph TLS warmup for compile + `reduce-overhead`.

        The cudagraph_trees state lives in C++ dynamic TLS — the first forward must run on
        THIS dispatcher thread so the captured graph binds here, not to the thread that
        built the wrapper. The warmup tensor is padded to the production batch size so the
        graph is captured for the steady-state shape. Failures degrade to
        fall-back-on-first-batch behaviour. No-op for non-CUDA or non-`reduce-overhead`.
        """
        if (
            self._compile_inference
            and self._compile_mode == "reduce-overhead"
            and self.device.type == "cuda"
        ):
            if self._shape is None:
                # `_compile_inference` is pinned False on the graph arm of __init__, and
                # the dense arm always sets `_shape` — a None here is a wiring break.
                raise RuntimeError(
                    "InferenceServer._warmup_compile_path: compile warmup requires the "
                    "dense (C, H, W) shape; graph mode never enables compile_inference."
                )
            try:
                with self._weights_lock:
                    with torch.inference_mode():
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=self._amp_dtype,
                        ):
                            if self._h2d_staging is not None:
                                self._h2d_staging.zero_()
                                warmup_tensor = self._h2d_staging.to(
                                    self.device, non_blocking=True,
                                )
                            else:
                                warmup_tensor = torch.zeros(
                                    self._batch_size, *self._shape, device=self.device,
                                )
                            _ = self.model(warmup_tensor)
                torch.cuda.synchronize()
                _LOG.info(
                    "inference_compile_warmup_dispatcher context=%s batch_size=%s mode=%s",
                    "inference_server", self._batch_size, self._compile_mode,
                )
            except Exception as exc:  # noqa: BLE001 — warmup is best-effort, logged
                _LOG.warning(
                    "inference_compile_warmup_failed context=%s error=%s",
                    "inference_server", str(exc)[:200],
                )

    def _run_graph_loop(self) -> None:
        """Ragged axis-graph inference loop, MEMORY-BOUNDED (F-816-10, verdict V-A).

        Pull a block-diagonal graph wire from Rust, convert it to a payload ONCE, partition
        that payload at GRAPH boundaries under `inference.fused_graph_caps`, and run one
        `collate_graph_batch` + `GnnNet.forward_batch` (bf16 autocast on CUDA — LAW-06) +
        segment-softmax per PART, freeing each part before the next so only one part's tensors
        are ever resident. The parts' probs and values are concatenated in plan order and
        submitted in ONE call against the UNSLICED `legal_offsets`; the Rust side assembles
        each leaf's legal-set policy, never a dense scatter.

        THE SPLIT IS PRE-COLLATE, and that seam is the whole design. A post-collate split
        materialises the full-E tensors first, and a design whose first allocation is
        proportional to the uncapped quantity cannot meet a bound.

        THE COPY-OUT TRAP IS ALREADY PAID. `PyGraphWire`'s getters COPY into fresh numpy
        arrays, so reading them per part would copy every array M times.
        `graph_wire_from_rust` reads each getter exactly ONCE — the same count HEAD's single
        `collate_graph_batch` did — and the parts are numpy views of that payload. Do not
        "optimise" this back into per-part getter reads.

        ONE SUBMIT, AFTER EVERY PART HAS RUN. The FFI checks `(ids, probs, legal_offsets,
        values)` for self-consistency on entry, and the one submit satisfies it with the
        payload's own unsliced offsets. It also means a mid-plan failure has submitted NOTHING,
        so every id fails uniformly and there is no partial-success bookkeeping to get wrong.

        Any planner refusal, resolver error or forward exception — including a real
        `OutOfMemoryError` — dies loud through the SAME `except` via
        `submit_graph_inference_failure`. The graph queue has no dense interpretation, so there
        is no silent fallback, and there is deliberately no OOM handler: the only reason to
        catch a memory failure specifically is to retry, and a retry is the silent
        catch-and-retry R276(f) forbids by name.
        """
        from mantis.selfplay.graph_collate import (
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
            # Unreachable by construction: the graph branch of `__init__` resolves or is
            # handed the caps before this thread can start. A None here is a wiring break, and
            # running unbounded is the one outcome that must not be available.
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
        canary_period = int(self._batch_size)  # cheap; a knob if it ever matters
        _perf = self._perf_timing
        _sync = self._perf_sync_cuda and self.device.type == "cuda"

        try:
            while not self._stop_event.is_set():
                try:
                    _t_wait_start = time.perf_counter()
                    request_ids, wire = self._batcher.next_graph_batch(
                        self._batch_size, self._max_wait_ms,
                    )
                    _wait_s = time.perf_counter() - _t_wait_start
                    if not request_ids:
                        # An empty pop is a deadline that expired with nothing queued. It
                        # is NOT a served-batch wait and must not enter the wait mean —
                        # an idle server would otherwise peg it at max_wait_ms and hide
                        # what the served batches actually cost.
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
                        _t0 = time.perf_counter() if _perf else 0.0
                        payload = graph_wire_from_rust(wire)
                        if _perf:
                            self._record_stage(
                                "wire_copyout", time.perf_counter() - _t0)
                            _t0 = time.perf_counter()
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
                        if _perf:
                            self._record_stage("plan", time.perf_counter() - _t0)
                        probs_parts: list[np.ndarray] = []
                        values_parts: list[np.ndarray] = []
                        for g0, g1 in plan:
                            _t0 = time.perf_counter() if _perf else 0.0
                            sub = slice_graph_wire(payload, g0, g1)
                            if _perf:
                                self._record_stage(
                                    "slice", time.perf_counter() - _t0)
                            _t_collate_start = time.perf_counter()
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
                            # Per PART, not per pop: `collate.count == sum(M)` where it used to
                            # equal `queue_wait.count`. The asymmetry is intended and recorded
                            # so it is not read as a leak.
                            _collate_s = time.perf_counter() - _t_collate_start
                            self._record_collate(_collate_s)
                            if _perf:
                                self._record_stage("collate_h2d", _collate_s)
                                _t0 = time.perf_counter()
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
                                    # nn.Module.__getattr__ types dynamic attrs as
                                    # Tensor | Module; `forward_batch` is GnnNet's real method.
                                    policy_logits, value, _bins = self.model.forward_batch(  # pyright: ignore[reportCallIssue]
                                        batch.x,
                                        batch.edge_index,
                                        batch.edge_attr,
                                        batch.legal_node_gather,
                                        stone_mask,
                                        batch.node_offsets,
                                    )
                            if _perf:
                                if _sync:
                                    torch.cuda.synchronize()
                                self._record_stage(
                                    "forward", time.perf_counter() - _t0)
                                _t0 = time.perf_counter()
                            # Segment-softmax in float32 (corrects reduced-precision drift,
                            # exactly like the dense path re-normalizes exp()). Segment-LOCAL
                            # by construction, so a part's softmax is the un-split forward's
                            # softmax for those graphs.
                            probs = segment_softmax(
                                policy_logits.float(), batch.legal_offsets
                            )
                            # Always-on finiteness gate: a NaN/Inf model output otherwise
                            # reaches backup() and poisons the tree SILENTLY, and the
                            # downstream numeric debug asserts are compiled out of release
                            # builds.
                            if not bool(torch.isfinite(probs).all()) or not bool(
                                torch.isfinite(value).all()
                            ):
                                raise RuntimeError(
                                    "NonFiniteModelOutput: graph forward produced NaN/Inf "
                                    f"(probs finite={bool(torch.isfinite(probs).all())}, "
                                    f"values finite={bool(torch.isfinite(value).all())})"
                                )
                            if _perf:
                                if _sync:
                                    torch.cuda.synchronize()
                                self._record_stage(
                                    "postproc", time.perf_counter() - _t0)
                                _t0 = time.perf_counter()
                            probs_parts.append(np.ascontiguousarray(
                                probs.detach().cpu().numpy(), dtype=np.float32
                            ))
                            values_parts.append(np.ascontiguousarray(
                                value.detach().float().cpu().numpy().reshape(-1),
                                dtype=np.float32,
                            ))
                            if _perf:
                                self._record_stage("d2h", time.perf_counter() - _t0)
                            self._record_fusion_part(
                                int(node_counts[g0:g1].sum()),
                                int(edge_counts[g0:g1].sum()),
                            )
                            # One part resident at a time — the bound is on the PEAK, so the
                            # previous part's device tensors must be gone before the next
                            # part's are built.
                            del sub, batch, stone_mask, policy_logits, value, probs
                        # ONE submit per pop, against the payload's own UNSLICED offsets: the
                        # parts' offsets are re-based and would segment the concatenation
                        # wrongly from the first part onward.
                        _t0 = time.perf_counter() if _perf else 0.0
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
                        if _perf:
                            self._record_stage("submit", time.perf_counter() - _t0)
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
        if self._is_graph:
            self._run_graph_loop()
            return
        # Dense loop from here down; the dense arm of __init__ always sets `_shape`.
        shape = self._shape
        if shape is None:
            raise RuntimeError(
                "InferenceServer.run: dense loop entered with no (C, H, W) shape — "
                "grid/graph construction invariant broken."
            )
        _perf = self._perf_timing
        _sync = self._perf_sync_cuda and self.device.type == "cuda"

        # Log which CUDA stream this thread is on, once at thread start: if it matches the
        # trainer stream (both default), there is no overlap.
        if self.device.type == "cuda":
            try:
                current_stream = torch.cuda.current_stream(self.device)
                default_stream = torch.cuda.default_stream(self.device)
                _LOG.info(
                    "cuda_stream_audit context=%s current_stream_ptr=%s "
                    "default_stream_ptr=%s on_default_stream=%s",
                    "inference_server",
                    int(current_stream.cuda_stream),
                    int(default_stream.cuda_stream),
                    current_stream.cuda_stream == default_stream.cuda_stream,
                )
            except Exception as exc:  # noqa: BLE001 — audit only, logged
                _LOG.warning(
                    "cuda_stream_audit_failed context=%s error=%s",
                    "inference_server", exc,
                )

        self._warmup_compile_path()

        try:
            while not self._stop_event.is_set():
                try:
                    _t_fetch_start = time.perf_counter() if _perf else 0.0
                    request_ids, batch = self._batcher.next_inference_batch(
                        self._batch_size,
                        self._max_wait_ms,
                    )
                    if not request_ids:
                        continue
                    if not self._first_enqueued_emitted:
                        self._first_enqueued_emitted = True
                        if self._sink is not None:
                            self._sink.emit({
                                "event": "first_inference_enqueued",
                                "batch_size": len(request_ids),
                                "representation": "dense",
                            })
                    _t_fetched = time.perf_counter() if _perf else 0.0
                    # Bound here so the `_perf` log block below is never reading an
                    # unbound name; the real values are assigned only under `_perf`.
                    _t_h2d_done = _t_forward_done = _t_d2h_done = 0.0

                    self._total_requests += len(request_ids)

                    try:
                        # The Rust contract on the bound supplier guarantees `batch` is
                        # already a float32 C-contiguous numpy array, so no defensive
                        # `ascontiguousarray` copy runs here (it cost an unconditional
                        # per-batch memcpy). A debug-only assert holds the contract and
                        # disappears under `python -O`.
                        if __debug__:
                            assert batch.dtype == np.float32, (
                                f"InferenceBatcher.next_inference_batch returned "
                                f"dtype={batch.dtype}; the contract guarantees float32"
                            )
                            assert batch.flags["C_CONTIGUOUS"], (
                                f"InferenceBatcher.next_inference_batch returned "
                                f"flags={batch.flags}; the contract guarantees C-contiguous"
                            )
                        batch_np = batch
                        n = len(request_ids)
                        _pad = self._padding_active()
                        if self._h2d_staging is not None:
                            assert n <= self._batch_size, (
                                f"inference batch size {n} exceeds staging capacity "
                                f"{self._batch_size} — config divergence between "
                                f"InferenceBatcher and InferenceServer"
                            )
                            # Staged async H2D: CPU→pinned copy, then DMA to GPU. The
                            # previous batch's H2D is already complete by this point
                            # (prior forward + .cpu() synced the default stream), so
                            # reusing the staging buffer is safe.
                            self._h2d_staging[:n].copy_(
                                torch.from_numpy(batch_np).view(n, *shape)
                            )
                            if _pad:
                                # Zero padding for the CUDA graph's fixed shape. Padded
                                # rows are discarded post-forward via host[:n] slicing.
                                if n < self._batch_size:
                                    self._h2d_staging[n:].zero_()
                                tensor = self._h2d_staging.to(
                                    self.device, non_blocking=True,
                                )
                            else:
                                tensor = self._h2d_staging[:n].to(
                                    self.device, non_blocking=True,
                                )
                        else:
                            tensor = (
                                torch.from_numpy(batch_np)
                                .to(self.device)
                                .reshape(n, *shape)
                            )
                        if _perf:
                            if _sync:
                                torch.cuda.synchronize()
                            _t_h2d_done = time.perf_counter()
                        if self._forward_count == 0:
                            assert not self.model.training, (
                                "InferenceServer model entered hot loop in train() mode; "
                                "eval() should be set at __init__ and re-applied in "
                                "load_state_dict_safe"
                            )
                        # Use the traced graph when available (it shares parameter storage
                        # with self.model, so weight swaps propagate without re-tracing).
                        fwd_model = (
                            self._traced_model
                            if self._traced_model is not None
                            else self.model
                        )
                        with self._weights_lock:
                            with torch.inference_mode():
                                # autocast on CUDA only; CPU autocast accepts bfloat16
                                # only, so it is disabled entirely on CPU.
                                with torch.autocast(
                                    device_type=self.device.type,
                                    dtype=self._amp_dtype,
                                    enabled=self.device.type == "cuda",
                                ):
                                    log_policy, value, _v_logit = fwd_model(tensor)
                        if _perf:
                            if _sync:
                                torch.cuda.synchronize()
                            _t_forward_done = time.perf_counter()

                        # .float() forces float32 regardless of the autocast dtype.
                        # Re-normalize after exp() to correct rounding drift.
                        probs = log_policy.float().exp()
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        # Merged D2H: one async copy instead of two. Layout is
                        # [fwd_n, policy_len + 1] — the last column carries the squeezed
                        # scalar value; splitting on the host is L2-cache cheap.
                        v = value.squeeze(-1).float().unsqueeze(-1)
                        host = torch.cat([probs, v], dim=-1).cpu().numpy()
                        # host is (n, …) under the variable-shape path and (batch_size, …)
                        # under the padded path; slice to the request count either way so
                        # padded-zero rows never reach Rust.
                        policies = np.ascontiguousarray(host[:n, :self._policy_len])
                        values = np.ascontiguousarray(host[:n, self._policy_len])
                        if _perf:
                            _t_d2h_done = time.perf_counter()

                        self._batcher.submit_inference_results(
                            request_ids,
                            policies,
                            values,
                        )
                        if _perf:
                            # submit_us closes the 2nd (return) FFI crossing so the 5
                            # buckets sum to the full fetch→submit cycle; the fetch
                            # crossing already lives inside fetch_wait_us.
                            _t_submit_done = time.perf_counter()
                            _LOG.info(
                                "inference_batch_timing batch_n=%s fetch_wait_us=%s "
                                "h2d_us=%s forward_us=%s d2h_scatter_us=%s submit_us=%s "
                                "sync_cuda=%s forward_count=%s",
                                len(request_ids),
                                (_t_fetched - _t_fetch_start) * 1e6,
                                (_t_h2d_done - _t_fetched) * 1e6,
                                (_t_forward_done - _t_h2d_done) * 1e6,
                                (_t_d2h_done - _t_forward_done) * 1e6,
                                (_t_submit_done - _t_d2h_done) * 1e6,
                                _sync,
                                self._forward_count + 1,
                            )
                    except Exception as exc:  # noqa: BLE001 — reported to Rust waiters
                        # Explicitly signal failure to Rust waiters rather than returning
                        # dummy data or failing silently. Message format is stable for
                        # downstream tests / log parsers.
                        error_msg = f"Model inference failed: {exc}"
                        # Surface the type + traceback even when str(exc) is empty.
                        _LOG.error(
                            "inference_forward_failed context=%s error_type=%s error=%s "
                            "tb=%s",
                            "inference_server", type(exc).__name__,
                            str(exc)[:300] or repr(exc)[:300],
                            traceback.format_exc()[:1500],
                        )
                        self._batcher.submit_inference_failure(request_ids, error_msg)
                        # Do not raise — the server can recover for the next batch.
                        continue

                    self._forward_count += 1
                    if not self._first_served_emitted:
                        self._first_served_emitted = True
                        if self._sink is not None:
                            self._sink.emit({
                                "event": "first_inference_served",
                                "batch_size": len(request_ids),
                                "representation": "dense",
                            })
                    if self._heartbeat is not None:
                        self._heartbeat(_HEARTBEAT_SOURCE)
                except Exception as exc:  # noqa: BLE001 — loop keeps serving next batch
                    _LOG.exception("inference_server_loop_error error=%s", exc)
                    if self._stop_event.is_set():
                        break
        finally:
            # Release blocked Rust waiters even if this thread exits unexpectedly.
            self._batcher.close()


__all__ = ["InferenceServer"]
