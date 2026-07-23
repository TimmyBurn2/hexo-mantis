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
from mantis.encoding import EncodingSpec as RegistrySpec
from mantis.encoding import resolve_from_config
from mantis.model import amp_dtype_for
from mantis.selfplay.hparams import InferenceHParams, is_graph_representation

_LOG = logging.getLogger(__name__)

# Emitted once per dispatched batch when a sink is injected. The named alias
# `HeartbeatFn` lives with the other injection Protocols in `pool_hooks`; the server
# only needs the structural type, and must not import the pool to get it.
_HEARTBEAT_SOURCE = "inference_dispatch"


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
    ) -> None:
        super().__init__(daemon=True, name="inference-server")
        self.model = model
        self.model.eval()
        self.device = device
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

        if self._is_graph:
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
        # production-scale graphs. The dense path reads the `amp_dtype` knob and must
        # match the trainer's choice for weight-sync consistency.
        _representation = "graph" if self._is_graph else "grid"
        self._amp_dtype = amp_dtype_for(_representation, config)

    def _setup_inference_path(self, hp: InferenceHParams, board_size: int) -> None:
        """Configure the trace OR compile path for the inference model.

        Mutually exclusive: `trace_inference` and `compile_inference` cannot both be
        enabled. Sets `_trace_inference`, `_traced_model`, `_compile_inference`,
        `_compile_mode`, `_compile_dynamic`; may replace `self.model` with a
        `torch.compile` wrapper. Called once at `__init__` — the run loop reads the
        resolved attributes, so there is no per-batch overhead from this helper.
        """
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
        """Ragged axis-graph inference loop.

        Pull a block-diagonal graph wire from Rust, run the single `collate_graph_batch`
        resolver, forward the `GnnNet` (bf16 autocast on CUDA — LAW-06), segment-softmax
        per graph's legal nodes, and submit the flat ragged probs back; the Rust side
        assembles each leaf's legal-set policy, never a dense scatter. Any resolver or
        forward exception dies loud via `submit_graph_inference_failure` — the graph queue
        has no dense interpretation, so there is no silent fallback.
        """
        from mantis.selfplay.graph_collate import (
            collate_graph_batch,
            reset_semantic_canary,
            segment_softmax,
            stone_mask_from_batch,
        )

        spec = self.encoding_spec
        # First batch after (re)start runs the FULL semantic/geometric layer.
        reset_semantic_canary()
        canary_period = int(self._batch_size)  # cheap; a knob if it ever matters

        try:
            while not self._stop_event.is_set():
                try:
                    request_ids, wire = self._batcher.next_graph_batch(
                        self._batch_size, self._max_wait_ms,
                    )
                    if not request_ids:
                        continue
                    self._total_requests += len(request_ids)
                    try:
                        batch = collate_graph_batch(
                            wire,
                            expected_version=1,
                            trunk_size=spec.trunk_size,
                            win_length=spec.win_length,
                            node_feat_dim=spec.node_feat_dim,
                            edge_feat_dim=spec.edge_feat_dim,
                            device=str(self.device),
                            semantic="canary",
                            canary_period=canary_period,
                        )
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
                                policy_logits, value, _bins = self.model.forward_batch(
                                    batch.x,
                                    batch.edge_index,
                                    batch.edge_attr,
                                    batch.legal_mask,
                                    stone_mask,
                                    batch.node_offsets,
                                )
                        # Segment-softmax in float32 (corrects reduced-precision drift,
                        # exactly like the dense path re-normalizes exp()).
                        probs = segment_softmax(policy_logits.float(), batch.legal_offsets)
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
                        probs_np = np.ascontiguousarray(
                            probs.detach().cpu().numpy(), dtype=np.float32
                        )
                        legal_offsets_np = np.ascontiguousarray(
                            batch.legal_offsets.detach().cpu().numpy(), dtype=np.int64
                        )
                        values_np = np.ascontiguousarray(
                            value.detach().float().cpu().numpy().reshape(-1),
                            dtype=np.float32,
                        )
                        self._batcher.submit_graph_inference_results(
                            request_ids, probs_np, legal_offsets_np, values_np,
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
                    _t_fetched = time.perf_counter() if _perf else 0.0

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
                                torch.from_numpy(batch_np).view(n, *self._shape)
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
                                .reshape(n, *self._shape)
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
