"""Self-play: the worker pool over the Rust runner, the ONE inference server, the graph
wire reader, and the replay-buffer facade.

Layout — four concerns, and the pool is split by implementation rather than by object
(the trainer duck-types it as a single collaborator):

  * `pool` / `pool_drain` / `pool_push` / `pool_hooks` — lifecycle, result drain, buffer
    push, promotion hooks + injection Protocols.
  * `inference_server` — THE batched dispatch loop; `inference_local` is the offline face
    whose graph leg rides that same server rather than re-implementing one.
  * `graph_collate` — the fused-batch contract reader and the named contract errors.
  * `buffers` / `hparams` / `instrumentation` / `utils` / `worker` — the facade, the knob
    resolution, the per-game telemetry, the temperature schedule, the bot-side glue.

Nothing here imports `mantis.eval`, `mantis.train` or `mantis.bots`: outward collaborators
(event sink, recorder, heartbeat) are injected with no-op defaults, and promotion is a
CALLEE surface — an evaluator reaches the pool, never the reverse.
"""

from mantis.selfplay.buffers import BufferKind, BufferKindMismatch, ReplayFacade
from mantis.selfplay.graph_collate import (
    WIN_AXES,
    AugRoundTripMismatch,
    BatchCountMismatch,
    DtypeMismatch,
    EdgeAttrDimMismatch,
    EdgeAttrGeometryMismatch,
    EdgeCrossesGraphBoundary,
    EdgeIndexOutOfBounds,
    EmptyLegalSet,
    GatherNotLegalNode,
    GraphBatch,
    GraphContractError,
    GraphContractVersionMismatch,
    GraphWirePayload,
    NodeCountChecksum,
    NodeFeatDimMismatch,
    NonNativeSampleBuilder,
    OffsetsNonMonotonic,
    ScatterGatherCrossesGraph,
    ScatterSlotAliasing,
    ScatterSlotCanonicalMismatch,
    ScatterSlotOutOfBounds,
    collate_graph_batch,
    graph_wire_from_rust,
    reset_semantic_canary,
    segment_softmax,
    stone_mask_from_batch,
)
from mantis.selfplay.hparams import (
    InferenceHParams,
    PoolDims,
    ResolvedPoolEncoding,
    SelfPlayHParams,
    build_runner_config,
    is_graph_representation,
    resolve_pool_encoding,
)
from mantis.selfplay.inference_local import LocalInferenceEngine
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.instrumentation import PoolInstrumentation
from mantis.selfplay.pool import WorkerPool
from mantis.selfplay.pool_hooks import (
    ActorSyncTarget,
    EventSink,
    HeartbeatFn,
    InferenceStats,
    NullRecorder,
    RecorderLike,
    RunnerStats,
)
from mantis.selfplay.utils import get_temperature, quarter_cosine_temperature
from mantis.selfplay.worker import SelfPlayWorker

__all__ = [
    "WIN_AXES",
    "AugRoundTripMismatch",
    "BatchCountMismatch",
    "BufferKind",
    "BufferKindMismatch",
    "DtypeMismatch",
    "EdgeAttrDimMismatch",
    "EdgeAttrGeometryMismatch",
    "EdgeCrossesGraphBoundary",
    "EdgeIndexOutOfBounds",
    "EmptyLegalSet",
    "EventSink",
    "GatherNotLegalNode",
    "GraphBatch",
    "GraphContractError",
    "GraphContractVersionMismatch",
    "GraphWirePayload",
    "HeartbeatFn",
    "InferenceHParams",
    "InferenceServer",
    "InferenceStats",
    "LocalInferenceEngine",
    "NodeCountChecksum",
    "NodeFeatDimMismatch",
    "NonNativeSampleBuilder",
    "NullRecorder",
    "OffsetsNonMonotonic",
    "PoolDims",
    "PoolInstrumentation",
    "ActorSyncTarget",
    "RecorderLike",
    "ReplayFacade",
    "ResolvedPoolEncoding",
    "RunnerStats",
    "ScatterGatherCrossesGraph",
    "ScatterSlotAliasing",
    "ScatterSlotCanonicalMismatch",
    "ScatterSlotOutOfBounds",
    "SelfPlayHParams",
    "SelfPlayWorker",
    "WorkerPool",
    "build_runner_config",
    "collate_graph_batch",
    "get_temperature",
    "graph_wire_from_rust",
    "is_graph_representation",
    "quarter_cosine_temperature",
    "reset_semantic_canary",
    "resolve_pool_encoding",
    "segment_softmax",
    "stone_mask_from_batch",
]
